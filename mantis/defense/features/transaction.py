"""Per-row features: everything readable off a single authorisation message.

Nothing here needs history, which is what makes this group the cheap half of the
latency budget — it is pure column arithmetic and it vectorises. The expensive
half is :mod:`mantis.defense.features.velocity`.

What is deliberately absent
----------------------------
``auth_response``, ``settled`` and ``settlement_lag_hours`` of the row being
scored. They are in the parquet and they are not features; see
:data:`mantis.defense.features.spec.FUTURE_COLUMNS` for the argument.

The orphan-credit indicator, and why its *absence* is the signal
-----------------------------------------------------------------
``original_event_id`` is the original-transaction reference an acquirer echoes
on a refund or a credit. Money going **out** with nothing referenced behind it is
an orphan credit, and it is the shape F1-03 exists to produce: a refund-logic
hijack pays out against a purchase that never happened. Day 3 got this right by
making the schema *permit* the orphan rather than forbidding it — a schema that
required the reference would have made the attack unrepresentable, which is
exactly the Day 0 specification error the 1.1.0 amendment was written to undo.

So the feature is the null pattern, not the value: ``txn_orphan_outbound`` fires
when an outbound flow carries no reference. The value itself is an identifier and
is excluded like every other identifier.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

__all__ = ["OUTBOUND_TYPES", "transaction_features"]

#: Processing codes that push value back out to the cardholder. The direction is
#: carried by the type, never by the sign of the amount — see CLAUDE.md §4.
OUTBOUND_TYPES: Final[tuple[str, ...]] = ("refund", "reversal", "credit")

#: Rails where the authorisation physically happens at the merchant, so a missing
#: location means something different than it does on a remote rail.
_PHYSICAL_CHANNELS: Final[tuple[str, ...]] = ("card_present", "upi_p2m", "upi_p2p")

#: Categoricals passed through as ``category`` dtype for LightGBM to split on
#: natively. Every one of them is low-cardinality and none is an identifier.
CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = (
    "channel",
    "entry_mode",
    "threeds_result",
    "txn_type",
    "mcc",
    "merchant_country",
    "ag_agent_platform",
    "ag_mandate_type",
)


def _round_number_score(amount: np.ndarray) -> np.ndarray:
    """How "round" an amount is: 0 ordinary, 3 a round thousand.

    Structuring and coerced-transfer attacks both produce suspiciously tidy
    numbers — F3-19's victim sends exactly 4,000 because that is what the caller
    told them to send. Legitimate traffic produces round numbers too (the
    population snaps deliberately, which is why the amount KS is non-zero by
    design), so this is evidence rather than proof, which is the right shape for
    an L1 feature.
    """
    cents = np.round(amount * 100).astype(np.int64)
    score = np.zeros(len(amount), dtype=float)
    score += (cents % 100 == 0).astype(float)
    score += (cents % 10_000 == 0).astype(float)
    score += (cents % 100_000 == 0).astype(float)
    return score


def transaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-row features. Pure function of ``frame``; no fitted state, no history."""
    out = pd.DataFrame(index=frame.index)
    amount = frame["amount"].to_numpy(dtype=float)
    ts = frame["ts"]

    # -- the ticket ---------------------------------------------------------- #
    out["txn_amount"] = amount
    out["txn_log_amount"] = np.log1p(amount)
    out["txn_round_score"] = _round_number_score(amount)

    # -- when ---------------------------------------------------------------- #
    hour = ts.dt.hour.to_numpy()
    out["txn_hour"] = hour.astype(float)
    out["txn_dow"] = ts.dt.dayofweek.to_numpy().astype(float)
    out["txn_is_weekend"] = (ts.dt.dayofweek >= 5).to_numpy().astype(float)
    # 01:00-05:00. Not "unusual" on its own — the agentic rail runs all night by
    # construction — but it interacts with rail and customer history.
    out["txn_is_night"] = ((hour >= 1) & (hour <= 5)).astype(float)

    # -- direction of the money ---------------------------------------------- #
    txn_type = frame["txn_type"].to_numpy()
    outbound = np.isin(txn_type, OUTBOUND_TYPES)
    out["txn_is_outbound"] = outbound.astype(float)
    # See the module docstring: the ABSENCE of the reference is the signal.
    out["txn_orphan_outbound"] = (outbound & frame["original_event_id"].isna().to_numpy()).astype(
        float
    )
    out["txn_is_refund"] = (txn_type == "refund").astype(float)
    out["txn_is_credit"] = (txn_type == "credit").astype(float)
    out["txn_is_preauth"] = (txn_type == "preauth").astype(float)

    # -- what the message is missing ------------------------------------------ #
    # Null patterns are real signal on a real auth stream: a card-not-present
    # message legitimately arrives without geo about a fifth of the time, and
    # which fields are absent says something about how it was presented.
    channel = frame["channel"].to_numpy()
    physical = np.isin(channel, _PHYSICAL_CHANNELS)
    out["txn_geo_missing"] = frame["lat"].isna().to_numpy().astype(float)
    out["txn_geo_missing_on_physical"] = (physical & frame["lat"].isna().to_numpy()).astype(float)
    out["txn_terminal_missing"] = frame["terminal_id"].isna().to_numpy().astype(float)
    out["txn_device_missing"] = frame["device_id"].isna().to_numpy().astype(float)
    out["txn_ip_missing"] = frame["ip"].isna().to_numpy().astype(float)

    # -- where ---------------------------------------------------------------- #
    out["txn_cross_border"] = (frame["merchant_country"].to_numpy() != "IN").astype(float)

    # -- rail identity -------------------------------------------------------- #
    # Unhideable and definitional (CLAUDE.md, Day 1). Stating it as a feature is
    # honest; the thing that would NOT be honest is quoting a headline recall
    # computed across both rails, which is why RESULTS.md reports per-rail.
    out["txn_is_agentic_block"] = frame["ag_agent_id"].notna().to_numpy().astype(float)

    for column in CATEGORICAL_COLUMNS:
        if column in frame.columns:
            out[column] = frame[column].astype("category")
    return out
