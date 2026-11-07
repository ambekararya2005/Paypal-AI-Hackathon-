"""Agent-mandate features: was the purchase inside the envelope the human signed?

This is the group the project exists for. Every other feature in the firewall
would be computable on a 2015 authorisation stream; these read the AP2 mandate
block, and they are what make MANTIS a detector for *agentic* commerce rather
than a card-fraud model with a new column.

Four questions, in the order an issuer would ask them
------------------------------------------------------
1. **Is this agent anyone?** ``kya_registered``, and whether a token is present
   at all. Cheap, deterministic, and mostly L0's job — carried here so L1 can
   learn how it interacts with everything else.
2. **Was this inside the mandate?** Amount against the ceiling, category against
   the signed list, merchant against the allow-list. The *shape* of the breach
   distinguishes F1-01 (cart tampering) from F1-02 (scope inflation) from F1-04
   (category drift), which is why the three are separate features rather than
   one "in scope" boolean.
3. **Was the mandate still alive?** Age against TTL. A mandate at 0.98 of its
   window is unusual and legitimate; one past 1.0 is F1-10.
4. **Did the session look like what it claims to be?** This is the subtle one and
   it is where the CLEAN attacks live.

The human-presence mismatch, and why it is a ratio rather than a rule
----------------------------------------------------------------------
``human_present`` drives liability: an authorisation with a human watching shifts
the loss. So it is worth forging, which is F1-09. The naive detector is
``human_present AND low cursor entropy`` — and it does not work, because Day 3
deliberately widened the population to contain **passive** humans: people who
watch the agent work on a second screen and never touch the device. 11% of
genuinely supervised sessions look machine-like.

That widening is the reason this feature is a graded mismatch score rather than a
boolean. Measured before the widening, the boolean was a *perfect* detector for
F1-09; after it, the same rule sits at 16.7% recall for 3.7% false positives.
The graded version lets L1 combine weak presence evidence with everything else
instead of betting the layer on a rule that a generator artefact was propping up.

Deliberation against stake
---------------------------
The population draws deliberation latency as a function of the amount: an agent
spends longer on a flight than on a recharge. An injected attack collapses that
— the decision was made by the injected text, not by deliberation — so the
residual of latency against what the ticket deserves carries more than the raw
latency does. This is the CLEAN attacks' main hope of being caught by L1 at all.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

__all__ = ["MandateBaselines", "mandate_features"]

#: Below this, a session's cursor telemetry reads as machine-driven. Taken from
#: the geometric mean of the two log-normals the population draws from, which is
#: where the passive and hands-on components are equally dense.
_CURSOR_MACHINE_LIKE: Final[float] = 1.0


class MandateBaselines:
    """Population baselines for the behavioural columns, fitted on train only.

    Only three numbers, but they must come from the training split for the same
    reason the entity profiles must: a z-score against a mean that includes the
    test period is a z-score against the future.
    """

    __slots__ = (
        "deliberation_intercept",
        "deliberation_sigma",
        "deliberation_slope",
        "provenance_mean",
        "provenance_std",
        "tool_call_mean",
        "tool_call_std",
    )

    def __init__(self) -> None:
        self.deliberation_intercept = float("nan")
        self.deliberation_slope = float("nan")
        self.deliberation_sigma = float("nan")
        self.tool_call_mean = float("nan")
        self.tool_call_std = float("nan")
        self.provenance_mean = float("nan")
        self.provenance_std = float("nan")

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> MandateBaselines:
        """Regress log-deliberation on log-amount over legitimate agentic rows."""
        self = cls()
        agentic = frame[frame["ag_agent_id"].notna()]
        if agentic.empty:
            return self

        latency = agentic["ag_deliberation_latency_ms"].to_numpy(dtype=float)
        amount = agentic["amount"].to_numpy(dtype=float)
        ok = np.isfinite(latency) & (latency > 0) & np.isfinite(amount)
        if ok.sum() >= 100:
            y = np.log(latency[ok])
            x = np.log1p(amount[ok])
            slope, intercept = np.polyfit(x, y, 1)
            self.deliberation_slope = float(slope)
            self.deliberation_intercept = float(intercept)
            self.deliberation_sigma = float(np.std(y - (slope * x + intercept)))

        calls = agentic["ag_tool_call_count"].to_numpy(dtype=float)
        self.tool_call_mean = float(np.nanmean(calls))
        self.tool_call_std = float(np.nanstd(calls)) or 1.0

        lengths = agentic["ag_provenance_chain"].map(
            lambda v: len(v) if isinstance(v, (list, np.ndarray)) else np.nan
        ).to_numpy(dtype=float)
        self.provenance_mean = float(np.nanmean(lengths))
        self.provenance_std = float(np.nanstd(lengths)) or 1.0
        return self


def _list_len(series: pd.Series) -> np.ndarray:
    """Length of a list-valued column, NaN off the agentic rail."""
    return series.map(
        lambda v: float(len(v)) if isinstance(v, (list, np.ndarray)) else np.nan
    ).to_numpy(dtype=float)


def _contains(haystack: pd.Series, needle: pd.Series) -> np.ndarray:
    """1.0 when the row's value is in its own list column, 0.0 when not, NaN when absent.

    An **empty** allow-list means unconstrained, not "nothing allowed" — the
    schema says so — so it returns 1.0. Getting that backwards would make every
    legitimate open-scope mandate look like a merchant violation and would put
    L0's false-positive rate through the roof.
    """
    out = np.full(len(haystack), np.nan, dtype=float)
    for i, (allowed, value) in enumerate(zip(haystack.to_numpy(), needle.to_numpy(), strict=True)):
        if not isinstance(allowed, (list, np.ndarray)):
            continue
        if len(allowed) == 0:
            out[i] = 1.0
        else:
            out[i] = 1.0 if str(value) in {str(a) for a in allowed} else 0.0
    return out


def mandate_features(frame: pd.DataFrame, baselines: MandateBaselines) -> pd.DataFrame:
    """Mandate, delegation, provenance and behavioural features.

    Every column is NaN off the agentic rail, which is correct: a classic
    authorisation has no mandate to be inside or outside of, and a zero would be
    a claim that it was perfectly compliant.
    """
    out = pd.DataFrame(index=frame.index)
    has_agent = frame["ag_agent_id"].notna().to_numpy()
    amount = frame["amount"].to_numpy(dtype=float)

    def _agentic_only(values: np.ndarray) -> np.ndarray:
        return np.where(has_agent, values, np.nan)

    # -- 1. identity ---------------------------------------------------------- #
    out["mnd_kya_registered"] = _agentic_only(
        frame["ag_kya_registered"].map({True: 1.0, False: 0.0}).to_numpy(dtype=float)
    )
    out["mnd_kya_token_missing"] = _agentic_only(
        frame["ag_kya_token"].isna().to_numpy().astype(float)
    )
    out["mnd_consent_valid"] = _agentic_only(
        frame["ag_consent_sig_valid"].map({True: 1.0, False: 0.0}).to_numpy(dtype=float)
    )

    # -- 2. scope divergence, one feature per SHAPE of breach ------------------ #
    ceiling = frame["ag_scope_max_amount"].to_numpy(dtype=float)
    out["mnd_amount_over_ceiling"] = np.where(ceiling > 0, amount / ceiling, np.nan)
    out["mnd_ceiling_breached"] = np.where(ceiling > 0, (amount > ceiling).astype(float), np.nan)
    out["mnd_mcc_in_scope"] = _contains(frame["ag_scope_categories"], frame["mcc"])
    out["mnd_merchant_in_scope"] = _contains(
        frame["ag_scope_allowed_merchants"], frame["merchant_id"]
    )
    out["mnd_scope_categories_n"] = _list_len(frame["ag_scope_categories"])
    out["mnd_scope_merchants_n"] = _list_len(frame["ag_scope_allowed_merchants"])
    out["mnd_scope_max_items"] = _agentic_only(
        frame["ag_scope_max_items"].to_numpy(dtype=float)
    )

    # -- 3. is the mandate still alive ----------------------------------------- #
    issued = frame["ag_mandate_issued_ts"]
    age = (frame["ts"] - issued).dt.total_seconds().to_numpy(dtype=float)
    ttl = frame["ag_mandate_ttl_seconds"].to_numpy(dtype=float)
    out["mnd_age_seconds"] = age
    out["mnd_age_over_ttl"] = np.where(ttl > 0, age / ttl, np.nan)
    out["mnd_expired"] = np.where(ttl > 0, (age > ttl).astype(float), np.nan)
    out["mnd_ttl_seconds"] = _agentic_only(ttl)

    # -- delegation and provenance --------------------------------------------- #
    out["mnd_delegation_depth"] = _agentic_only(
        frame["ag_delegation_depth"].to_numpy(dtype=float)
    )
    provenance = _list_len(frame["ag_provenance_chain"])
    out["mnd_provenance_len"] = provenance
    out["mnd_ingested_len"] = _list_len(frame["ag_ingested_content_ids"])
    if baselines.provenance_std == baselines.provenance_std:
        out["mnd_provenance_z"] = (provenance - baselines.provenance_mean) / max(
            baselines.provenance_std, 1e-9
        )

    # -- 4. does the session look like what it claims -------------------------- #
    cursor = frame["ag_cursor_entropy"].to_numpy(dtype=float)
    dwell = frame["ag_dwell_time_ms"].to_numpy(dtype=float)
    human = frame["ag_human_present"].map({True: 1.0, False: 0.0}).to_numpy(dtype=float)
    out["mnd_human_present"] = _agentic_only(human)
    out["mnd_cursor_entropy"] = cursor
    out["mnd_dwell_ms"] = dwell

    # The graded mismatch, not the boolean rule. See the module docstring: the
    # boolean was a perfect detector only because the population had no passive
    # humans in it, and Day 3 fixed the population rather than the metric.
    machine_like = np.clip(_CURSOR_MACHINE_LIKE - cursor, 0.0, None) / _CURSOR_MACHINE_LIKE
    out["mnd_presence_mismatch"] = np.where(human > 0, machine_like, np.nan)
    out["mnd_dwell_per_tool_call"] = np.where(
        frame["ag_tool_call_count"].to_numpy(dtype=float) > 0,
        dwell / frame["ag_tool_call_count"].to_numpy(dtype=float),
        np.nan,
    )

    calls = frame["ag_tool_call_count"].to_numpy(dtype=float)
    out["mnd_tool_calls"] = calls
    if baselines.tool_call_std == baselines.tool_call_std:
        out["mnd_tool_calls_z"] = (calls - baselines.tool_call_mean) / max(
            baselines.tool_call_std, 1e-9
        )

    latency = frame["ag_deliberation_latency_ms"].to_numpy(dtype=float)
    out["mnd_deliberation_ms"] = latency
    if baselines.deliberation_sigma == baselines.deliberation_sigma:
        # Residual against what a ticket this size deserves. An injected decision
        # was not deliberated; it was read off a page.
        expected = baselines.deliberation_intercept + baselines.deliberation_slope * np.log1p(
            amount
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            observed = np.where(latency > 0, np.log(latency), np.nan)
        out["mnd_deliberation_residual_z"] = (observed - expected) / max(
            baselines.deliberation_sigma, 1e-9
        )
    return out
