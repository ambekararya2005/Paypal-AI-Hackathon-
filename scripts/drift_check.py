"""Day 1's distribution comparison, re-run against everything added since.

    python scripts/drift_check.py [--n 200000] [--seed 1337]

Why this exists on Day 4 rather than Day 7
-------------------------------------------
The Day 1 gate compared three numbers against the reference: amount KS,
hour-of-day total variation, MCC mix max delta. Since it passed, the population
has grown **nine** new behaviours — a per-channel decline rate, refunds bound to
real earlier purchases, reversals, pre-authorisation holds, orphan-free credits,
bimodal settlement lag, a passive-human tail, an instant-refund tail, and a
deeper delegation tail. Three of those (the last three) were added *specifically
to make attacks harder to detect*, which is exactly the kind of change that can
quietly bend the legitimate marginals while nobody is looking.

The fidelity scorecard is a scored deliverable. Finding out on Day 7 that a
Day 3 change moved a marginal is the worst possible time to find out. So this
script measures every distribution the reference specifies, not just the three
Day 1 happened to print.

What "drift" means here, precisely
-----------------------------------
``ReferenceStats`` **is** the specification the simulator draws from. So this
compares the realised population against its own spec, and a deviation means one
of two things: the simulator does not do what the reference says, or a later
change introduced a path that bypasses it. It does **not** measure agreement
with real-world card data — no such file is committed, and claiming otherwise
would be the exact overclaim this project keeps refusing to make. That
comparison needs an external reference and belongs to the Day 7 scorecard's TSTR
number.

Why the flags are bootstrapped rather than thresholded
-------------------------------------------------------
"JSD under 0.02 is fine" is a number somebody made up. Every distance here is
computed against a **null band** instead: draw ``n`` samples from the target
distribution itself, ``BOOTSTRAP`` times, and record the 99th percentile of the
distance that pure sampling noise produces at that ``n`` and that support size.
A realised distance inside the band is indistinguishable from noise. One outside
it is a real deviation whose size can then be argued about on its merits.

This matters because the supports differ wildly — ``txn_type`` has 5 levels and
``mcc`` has 24, and 200,000 draws give you far tighter agreement on the first
than the second. A single flat threshold would flag one and excuse the other for
reasons that have nothing to do with fidelity.

Distances
---------
* **Categorical** — Jensen-Shannon divergence, base 2, so 0 is identical and 1
  is disjoint support. Symmetric and finite even when a level is missing on one
  side, which KL is not.
* **Continuous** — the Kolmogorov-Smirnov statistic against the reference's own
  analytic CDF (a per-MCC log-normal mixture for amounts, a per-channel
  log-normal for settlement lag). The 99% critical value is ``1.63/sqrt(n)``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mantis.core.paths import DOCS_DIR, ensure_dir  # noqa: E402
from mantis.foundry.base.reference import ReferenceStats, load_reference_stats  # noqa: E402
from mantis.foundry.base.simulator import SimulationConfig, simulate_frame  # noqa: E402

# Day 7 lifted these out of this script so the fidelity scorecard could use the
# same implementation rather than a second copy that drifts from it. The formulae
# are unchanged, so every number this script printed on Day 4 it still prints;
# only their address moved.
from mantis.foundry.fidelity.metrics import (  # noqa: E402
    KS_99,
    entity_jsd_null_band,
    jsd,
    jsd_null_band,
)
from mantis.foundry.fidelity.metrics import ratio as _ratio  # noqa: E402

# --------------------------------------------------------------------------- #
# Distances
# --------------------------------------------------------------------------- #


def _mixture_amount_cdf(stats: ReferenceStats, x: np.ndarray) -> np.ndarray:
    """Reference amount CDF: the weight-mixture of per-MCC log-normals."""
    from scipy.stats import norm

    x = np.maximum(np.asarray(x, dtype=float), 1e-9)
    out = np.zeros_like(x)
    total = sum(p.weight for p in stats.mcc_profiles)
    for profile in stats.mcc_profiles:
        z = (np.log(x) - profile.log_amount_mu) / profile.log_amount_sigma
        out += (profile.weight / total) * norm.cdf(z)
    return out


# --------------------------------------------------------------------------- #
# Reference-implied targets
# --------------------------------------------------------------------------- #


def _channel_target(frame: pd.DataFrame, stats: ReferenceStats) -> dict[str, float]:
    """Reference channel mix, conditioned on the realised category mix.

    The reference does not name a channel distribution directly: it names one
    **per MCC**, and the agentic rail is carved out first by a Gumbel-top-k draw
    that hits ``agentic_share`` exactly. So the honest target is the mixture the
    reference implies given the categories that were actually drawn — which is
    what tests the channel-given-category draw rather than re-testing the
    category draw underneath it.
    """
    classic = frame[frame["channel"] != "agentic"]
    mcc_mix = classic["mcc"].value_counts(normalize=True)
    by_mcc = {p.mcc: p.channel_weights for p in stats.mcc_profiles}

    target: dict[str, float] = {}
    for mcc, share in mcc_mix.items():
        for channel, weight in by_mcc.get(str(mcc), {}).items():
            target[channel] = target.get(channel, 0.0) + float(share) * float(weight)

    agentic = stats.agentic_share
    target = {k: v * (1.0 - agentic) for k, v in target.items()}
    target["agentic"] = agentic
    return target


def _txn_type_target(stats: ReferenceStats) -> dict[str, float]:
    """Processing-code mix. Purchase is the remainder, as in the simulator."""
    named = {
        "refund": stats.refund_share,
        "reversal": stats.reversal_share,
        "credit": stats.credit_share,
        "preauth": stats.preauth_share,
    }
    return {"purchase": 1.0 - sum(named.values()), **named}


def _auth_target(frame: pd.DataFrame, stats: ReferenceStats) -> dict[str, float]:
    """Approve/decline mix, conditioned on the realised rail mix.

    The reference names a decline rate **per channel**, and the simulator tilts
    it by the ticket size (``decline_amount_tilt``), which is deliberate and
    which this target does not model. So a small positive deviation here is
    expected and is annotated rather than treated as a defect: the tilt moves
    mass between rails without changing any rail's nominal rate.
    """
    mix = frame["channel"].value_counts(normalize=True)
    declined = sum(float(share) * stats.decline_rate[str(ch)] for ch, share in mix.items())
    target = {"approved": 1.0 - declined}
    for reason, weight in stats.decline_reason_weights.items():
        target[reason] = declined * weight
    return target


def _conditional_targets(
    stats: ReferenceStats, prior: dict[str, dict[str, float]], name: str
) -> list[tuple[str, str, dict[str, float]]]:
    """Expand a channel-conditional prior into one row per channel."""
    return [(f"{name} | {channel}", "channel", weights) for channel, weights in prior.items()]


# --------------------------------------------------------------------------- #
# The comparison
# --------------------------------------------------------------------------- #


def categorical_rows(
    frame: pd.DataFrame, stats: ReferenceStats, rng: np.random.Generator
) -> list[dict[str, Any]]:
    """Every categorical the reference specifies, measured against it."""
    agentic = frame[frame["channel"] == "agentic"]
    purchases = frame[frame["txn_type"] == "purchase"]

    specs: list[tuple[str, pd.Series, dict[str, float], str]] = [
        (
            "mcc",
            frame["mcc"],
            {p.mcc: p.weight for p in stats.mcc_profiles},
            "drawn directly from mcc_profiles weights",
        ),
        (
            "channel",
            frame["channel"],
            _channel_target(frame, stats),
            "per-MCC channel weights, conditioned on the realised category mix",
        ),
        (
            "hour_of_day",
            frame["ts"].dt.hour.astype(str),
            _hour_target(frame, stats),
            "human curve blended toward uniform on the agentic share",
        ),
        (
            "day_of_week",
            frame["ts"].dt.dayofweek.astype(str),
            _dow_target(frame, stats),
            "dow_weights x the calendar: a 90-day window is not a whole number of weeks",
        ),
        (
            "txn_type",
            frame["txn_type"],
            _txn_type_target(stats),
            "AMENDMENT 1.1.0 -- refund/reversal/credit/preauth shares",
        ),
        (
            "decline_reason | declined",
            frame.loc[frame["auth_response"] != "approved", "auth_response"],
            stats.decline_reason_weights,
            "AMENDMENT 1.1.0 -- reason mix among declines only; remapping shifts it, see below",
        ),
        (
            "mandate_type",
            agentic["ag_mandate_type"],
            stats.mandate_type_weights,
            "mandate_type_weights",
        ),
        (
            "delegation_depth",
            agentic["ag_delegation_depth"].astype("Int64").astype(str),
            stats.delegation_depth_weights,
            "WIDENED DAY 3 to depth 5 so F1-05 was not a free catch",
        ),
    ]

    conditionals = (
        ("entry_mode", stats.entry_mode_weights),
        ("threeds", stats.threeds_weights),
    )
    for name, prior in conditionals:
        column = "entry_mode" if name == "entry_mode" else "threeds_result"
        for channel, weights in prior.items():
            rows = frame[frame["channel"] == channel]
            if rows.empty:
                continue
            specs.append(
                (f"{name} | {channel}", rows[column], weights, f"{name}_weights[{channel}]")
            )

    # Purchases only: refunds and reversals inherit their source's rail, so
    # including them would test the binding, not the draw.
    del purchases

    out: list[dict[str, Any]] = []
    for name, series, target, note in specs:
        series = series.dropna()
        n = len(series)
        if n == 0 or not target:
            continue
        levels = sorted(set(map(str, target)) | set(map(str, series.unique())))
        observed = series.astype(str).value_counts(normalize=True)
        p = np.array([observed.get(level, 0.0) for level in levels], dtype=float)
        q = np.array([float(target.get(level, 0.0)) for level in levels], dtype=float)
        q = q / q.sum()
        distance = jsd(p, q)
        band = jsd_null_band(q, n, rng)
        out.append(
            {
                "feature": name,
                "kind": "JSD",
                "n": n,
                "levels": len(levels),
                "distance": distance,
                "band": band,
                "ratio": _ratio(distance, band),
                "note": note,
            }
        )
    return out


def _dow_target(frame: pd.DataFrame, stats: ReferenceStats) -> dict[str, float]:
    """Day-of-week weights, times how many of each weekday the window contains.

    A 90-day window holds twelve of some weekdays and thirteen of others, so the
    reference weights are not the marginal you should expect to see -- they are a
    per-day rate. Comparing against the raw weights flags a calendar fact as a
    simulator defect, which is how a real deviation gets lost in the noise of a
    fake one.
    """
    days = frame["ts"].dt.normalize()
    per_weekday = days.drop_duplicates().dt.dayofweek.value_counts()
    weights = {str(i): stats.dow_weights[i] * float(per_weekday.get(i, 0)) for i in range(7)}
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def entity_rows(
    frame: pd.DataFrame, stats: ReferenceStats, rng: np.random.Generator
) -> list[dict[str, Any]]:
    """Columns drawn once per entity and inherited by that entity's events.

    Separated out because their null band has to be computed at the entity level.
    See :func:`entity_jsd_null_band`.
    """
    agentic = frame[frame["channel"] == "agentic"]
    specs: list[tuple[str, pd.DataFrame, str, str, dict[str, float], str]] = [
        (
            "merchant_country",
            frame,
            "merchant_id",
            "merchant_country",
            stats.merchant_country_weights,
            "one draw per merchant; Zipf popularity makes the effective n the merchant count",
        ),
        (
            "card_bin",
            frame,
            "card_entity",
            "card_bin",
            {b.card_bin: b.weight for b in stats.card_bins},
            "one draw per card; effective n is the card count, not the event count",
        ),
        (
            "agent_platform",
            agentic,
            "ag_agent_id",
            "ag_agent_platform",
            stats.agent_platform_weights,
            "one draw per agent; effective n is the agent count",
        ),
    ]

    out: list[dict[str, Any]] = []
    for name, source, entity_col, value_col, target, note in specs:
        source = source.copy()
        if entity_col == "card_entity":
            # A card is a (customer, bin) pair: one customer holds several.
            source[entity_col] = (
                source["customer_id"].astype(str) + "|" + source["card_bin"].astype(str)
            )
        source = source[source[value_col].notna() & source[entity_col].notna()]
        if source.empty:
            continue
        levels = sorted(set(map(str, target)) | set(map(str, source[value_col].unique())))
        observed = source[value_col].astype(str).value_counts(normalize=True)
        p_obs = np.array([observed.get(level, 0.0) for level in levels], dtype=float)
        q = np.array([float(target.get(level, 0.0)) for level in levels], dtype=float)
        q = q / q.sum()

        per_entity = source.groupby(entity_col, observed=True).size()
        band = entity_jsd_null_band(per_entity.to_numpy(), q, rng)
        distance = jsd(p_obs, q)
        out.append(
            {
                "feature": name,
                "kind": "JSD*",
                "n": int(per_entity.size),
                "levels": len(levels),
                "distance": distance,
                "band": band,
                "ratio": _ratio(distance, band),
                "note": note,
            }
        )
    return out


def _hour_target(frame: pd.DataFrame, stats: ReferenceStats) -> dict[str, float]:
    """Hour curve: the volume-weighted mix of human and agentic shapes.

    Lifted from ``calibration.calibration_report`` so the two cannot disagree.
    Comparing against the human curve alone would flag a deviation the simulator
    was explicitly told to produce.
    """
    human = np.asarray(stats.hour_weights, dtype=float)
    blend = stats.agentic_hour_uniform_blend
    agent = (1.0 - blend) * human + blend / 24.0
    agent /= agent.sum()
    share = float((frame["channel"] == "agentic").mean())
    mixed = (1.0 - share) * human + share * agent
    return {str(i): float(w) for i, w in enumerate(mixed)}


def continuous_rows(frame: pd.DataFrame, stats: ReferenceStats) -> list[dict[str, Any]]:
    """Every continuous column the reference gives an analytic CDF for."""
    from scipy.stats import kstest, norm

    out: list[dict[str, Any]] = []

    amount = frame["amount"].to_numpy(dtype=float)
    result = kstest(amount, lambda x: _mixture_amount_cdf(stats, np.asarray(x)))
    out.append(
        {
            "feature": "amount",
            "kind": "KS",
            "n": len(amount),
            "levels": 0,
            "distance": float(result.statistic),
            "band": KS_99 / math.sqrt(len(amount)),
            "ratio": float(result.statistic) / (KS_99 / math.sqrt(len(amount))),
            "note": "non-zero BY DESIGN: round-number snapping is deliberate, see Day 1",
        }
    )

    # Settlement lag, per rail, on approved settled purchases only. Refunds carry
    # their own instant-share mixture and pre-auths their own hold behaviour;
    # folding them in would test three things through one number.
    settled = frame[
        (frame["txn_type"] == "purchase")
        & (frame["auth_response"] == "approved")
        & frame["settled"]
        & frame["settlement_lag_hours"].notna()
    ]
    sigma = stats.settlement_lag_sigma
    for channel, median in stats.settlement_lag_median_hours.items():
        lag = settled.loc[settled["channel"] == channel, "settlement_lag_hours"].to_numpy(float)
        lag = lag[lag > 0]
        if lag.size < 200:
            continue
        def _cdf(x: np.ndarray, m: float = median) -> np.ndarray:
            return norm.cdf((np.log(np.maximum(x, 1e-9)) - math.log(m)) / sigma)

        result = kstest(lag, _cdf)
        band = KS_99 / math.sqrt(lag.size)
        out.append(
            {
                "feature": f"settlement_lag | {channel}",
                "kind": "KS",
                "n": int(lag.size),
                "levels": 0,
                "distance": float(result.statistic),
                "band": band,
                "ratio": float(result.statistic) / band,
                "note": "AMENDMENT 1.1.0 -- lognormal(log median, sigma); BIMODAL across rails",
            }
        )
    return out


#: Rails where the authorisation physically happens at the merchant, so geo is
#: always present. Mirrors ``simulator._PHYSICAL_CHANNELS``.
_PHYSICAL: Final[tuple[str, ...]] = ("card_present", "upi_p2m", "upi_p2p")

#: The two log-normals ``simulator._draw_agentic_flags`` draws cursor entropy
#: from: hands-on sessions and passive ones. Mirrored here so the passive share
#: can be recovered rather than guessed at with an arbitrary threshold.
_CURSOR_HANDS_ON: Final[tuple[float, float]] = (2.4, 0.40)
_CURSOR_PASSIVE: Final[tuple[float, float]] = (0.42, 0.45)


def _passive_share(human: pd.DataFrame) -> float:
    """Recover the passive share from the cursor-entropy mixture.

    The simulator emits no "passive" flag -- passivity shows up only as a session
    whose telemetry was drawn from the low log-normal instead of the high one.
    Thresholding the column and calling the answer a share would be wrong by the
    overlap between the two components, which at these parameters is several
    points.

    So invert the mixture instead. With a cut at ``c``, the observed share below
    it is ``pi * P(passive < c) + (1 - pi) * P(hands_on < c)``, and both those
    probabilities are known exactly from the log-normal parameters above. One
    line of algebra returns ``pi``. The cut sits at the geometric mean of the two
    medians, where the components are equally dense and the estimator is least
    sensitive to getting them slightly wrong.
    """
    from scipy.stats import norm

    if human.empty:
        return float("nan")
    cut = math.sqrt(_CURSOR_HANDS_ON[0] * _CURSOR_PASSIVE[0])
    p_passive = float(norm.cdf((math.log(cut) - math.log(_CURSOR_PASSIVE[0])) / _CURSOR_PASSIVE[1]))
    p_hands = float(
        norm.cdf((math.log(cut) - math.log(_CURSOR_HANDS_ON[0])) / _CURSOR_HANDS_ON[1])
    )
    observed = float((human["ag_cursor_entropy"].astype(float) < cut).mean())
    return (observed - p_hands) / max(p_passive - p_hands, 1e-9)


def scalar_rows(frame: pd.DataFrame, stats: ReferenceStats) -> list[dict[str, Any]]:
    """Single-number shares the reference pins, including the three widened tails.

    These are the ones a reader will actually go looking for, because three of
    them were moved on Day 3 to stop an attack being a free catch. If any of
    them had failed to take effect, an attack would be far easier than the
    scorecard claims.
    """
    agentic = frame[frame["channel"] == "agentic"]
    refunds = frame[frame["txn_type"] == "refund"]
    human = agentic[agentic["ag_human_present"] == True]  # noqa: E712

    # The rails where an "instant" refund means anything. UPI already clears in
    # seconds, so on UPI a refund is instant by rail rather than by the merchant
    # offering, and counting those would roughly double the measured share.
    slow_rails = [c for c, m in stats.settlement_lag_median_hours.items() if m > 1.0]
    card_refunds = refunds[refunds["channel"].isin(slow_rails)]

    checks: list[tuple[str, float, float, str]] = [
        ("agentic_share", float((frame["channel"] == "agentic").mean()), stats.agentic_share, ""),
        (
            "geo_missing | remote rails",
            float(frame.loc[~frame["channel"].isin(_PHYSICAL), "lat"].isna().mean()),
            stats.remote_geo_missing_p,
            "denominator is the non-physical rails; UPI at a QR carries merchant geo",
        ),
        (
            "unsettled | approved",
            float((~frame.loc[frame["auth_response"] == "approved", "settled"]).mean()),
            stats.unsettled_share,
            "reversals never settle BY DEFINITION, so the realised share sits above the prior",
        ),
        (
            "dispute_rate | settled purch",
            float(
                frame.loc[(frame["txn_type"] == "purchase") & frame["settled"], "dispute_outcome"]
                .notna()
                .mean()
            ),
            stats.dispute_rate,
            "POST-HOC: carried for evaluation, never a scoring feature",
        ),
        (
            "refund_instant | card rails",
            float((card_refunds["settlement_lag_hours"].fillna(1e9) < 1.0).mean())
            if len(card_refunds)
            else float("nan"),
            stats.refund_instant_share,
            "WIDENED DAY 3: without it F1-03 sat at 0.996 on settlement lag alone",
        ),
        (
            "human_passive | present",
            _passive_share(human),
            stats.human_present_passive_share,
            "deconvolved from the cursor-entropy mixture; see _passive_share",
        ),
        (
            "delegation_depth >= 4",
            float((agentic["ag_delegation_depth"].fillna(0) >= 4).mean()),
            stats.delegation_depth_weights.get("4", 0.0)
            + stats.delegation_depth_weights.get("5", 0.0),
            "WIDENED DAY 3: without it depth>=4 was a perfect F1-05 detector",
        ),
    ]
    return [
        {
            "feature": name,
            "realised": realised,
            "target": target,
            "delta": realised - target,
            "note": note,
        }
        for name, realised, target, note in checks
    ]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def decline_rate_rows(frame: pd.DataFrame, stats: ReferenceStats) -> list[dict[str, Any]]:
    """Realised decline rate per rail against the rate the reference names.

    Split out from the ``auth_response`` marginal because the aggregate hides
    which rail moved, and on Day 4 this is the block that matters: the whole
    point of ``decline_ratio`` features is that an attack's decline rate is read
    **against the background's**. If the background rate is not the rate the
    reference claims, every lift quoted against it is quoted against the wrong
    denominator.

    Only ``purchase`` and ``preauth`` are declinable -- the issuer does not
    refuse to give money back -- so those are the denominator.
    """
    declinable = frame[frame["txn_type"].isin(["purchase", "preauth"])]
    out: list[dict[str, Any]] = []
    for channel, target in stats.decline_rate.items():
        rows = declinable[declinable["channel"] == channel]
        if rows.empty:
            continue
        realised = float((rows["auth_response"] != "approved").mean())
        out.append(
            {
                "feature": f"decline_rate | {channel}",
                "realised": realised,
                "target": float(target),
                "delta": realised - float(target),
                "note": f"x{realised / max(target, 1e-9):.2f} the named rate  (n={len(rows):,})",
            }
        )
    return out


def format_distance_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        f"  {'feature':<26} {'kind':<5} {'n':>8} {'lv':>3} {'distance':>10} "
        f"{'null band':>10} {'x band':>7}  verdict",
        f"  {'-' * 26} {'-' * 5} {'-' * 8} {'-' * 3} {'-' * 10} {'-' * 10} {'-' * 7}  {'-' * 7}",
    ]
    for r in sorted(rows, key=lambda x: -float(x["ratio"])):
        ratio = float(r["ratio"])
        verdict = "ok" if ratio <= 1.0 else ("watch" if ratio <= 3.0 else "DRIFT")
        levels = str(r["levels"]) if r["levels"] else "-"
        lines.append(
            f"  {r['feature']:<26} {r['kind']:<5} {int(r['n']):>8,} {levels:>3} "
            f"{float(r['distance']):>10.5f} {float(r['band']):>10.5f} {ratio:>7.2f}  {verdict}"
        )
    return "\n".join(lines)


def format_scalar_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        f"  {'share':<30} {'realised':>10} {'target':>10} {'delta':>10}  note",
        f"  {'-' * 30} {'-' * 10} {'-' * 10} {'-' * 10}  {'-' * 40}",
    ]
    for r in rows:
        lines.append(
            f"  {r['feature']:<30} {float(r['realised']):>10.4f} {float(r['target']):>10.4f} "
            f"{float(r['delta']):>+10.4f}  {r['note']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/drift_check.py")
    parser.add_argument("--n", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--json", type=Path, default=DOCS_DIR / "drift_check.json")
    args = parser.parse_args(argv)

    stats = load_reference_stats()
    frame = simulate_frame(
        SimulationConfig(n_events=args.n, seed=args.seed, n_customers=5_000, n_merchants=12_000),
        stats,
    )
    rng = np.random.default_rng(args.seed)

    print("=" * 96)
    print("DISTRIBUTION DRIFT CHECK - realised population vs the reference it was drawn from")
    print("=" * 96)
    print(f"  {len(frame):,} events, seed {args.seed}, reference source '{stats.source}'")
    print()
    print("  This measures the simulator against its own specification. It is NOT a")
    print("  claim of agreement with real card data -- no such file is committed, and")
    print("  that comparison is the Day 7 scorecard's TSTR number.")
    print()

    rows = (
        categorical_rows(frame, stats, rng)
        + entity_rows(frame, stats, rng)
        + continuous_rows(frame, stats)
    )
    print("marginal distances vs reference (99% sampling-noise null band)")
    print()
    print(format_distance_table(rows))
    print()
    print("  'null band' is the distance that n honest draws from the target produce")
    print("  at the 99th percentile -- bootstrapped for JSD, 1.63/sqrt(n) for KS. A")
    print("  ratio at or under 1.00 is indistinguishable from sampling noise.")
    print()

    rates = decline_rate_rows(frame, stats)
    print("AMENDMENT 1.1.0 -- realised decline rate per rail vs the rate the reference names")
    print()
    print(format_scalar_table(rates))
    print()

    scalars = scalar_rows(frame, stats)
    print("pinned shares, including the three tails widened on Day 3 to weaken attacks")
    print()
    print(format_scalar_table(scalars))
    print()

    drifted = [r for r in rows if float(r["ratio"]) > 3.0]
    watch = [r for r in rows if 1.0 < float(r["ratio"]) <= 3.0]
    print("-" * 96)
    if drifted:
        print(f"DRIFT on {len(drifted)}: {', '.join(str(r['feature']) for r in drifted)}")
    else:
        print("No marginal drifted beyond 3x its sampling-noise band.")
    if watch:
        print(f"Watch ({len(watch)}, inside 3x): {', '.join(str(r['feature']) for r in watch)}")
    print("-" * 96)

    ensure_dir(args.json.parent)
    args.json.write_text(
        json.dumps(
            {
                "n_events": len(frame),
                "seed": args.seed,
                "reference_source": stats.source,
                "marginals": rows,
                "pinned_shares": scalars,
                "decline_rates": rates,
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
