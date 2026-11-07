"""The separability gate, run over the **built feature matrix** instead of raw columns.

The blind spot this closes
----------------------------
``mantis.foundry.injectors.probe`` explodes one authorisation message into the
columns a depth-1 stump may split on, and gates every attack at 0.95 AUC. It has
a structural limitation that was written down on Day 4 and not fixed:

    **it probes raw columns.**

``mnd_deliberation_residual_z`` — the residual of an agent's deliberation latency
against what a ticket that size deserves — separates F1-01 at **0.99 AUC on its
own**, above the gate, and the gate never saw it, because it is not a column in
the parquet. It is two columns and a regression. A trivially-derived feature
walked straight past a gate whose entire purpose is to catch trivially-detectable
attacks.

Day 5 adds twenty-eight more derived features in the ``gph_`` block, several of
which are exactly the kind of thing that could be accidentally perfect — a
component size, a fan-in ratio, a first-time-at-this-merchant counter. Shipping
those without extending the gate would be repeating the Day 4 mistake at four
times the scale.

So this script runs the same probe over the same slices, against the **232
features L1 actually trains on**.

What a high number here does and does not mean
------------------------------------------------
It is not automatically a defect. The gate on raw columns says *"no single fact
on the authorisation message may give the attack away"*, which is a statement
about the **generator's realism**. A derived feature scoring highly says
something different: *"one of our features is very good at this attack"*, which
is sometimes exactly what a feature is for. ``vel_mandate_hash_lifetime_count``
**is** the replay detector for F1-10; it is supposed to separate it.

The distinction is causal, and it has to be made by reading the feature:

* **Legitimate** — the feature measures the attack's *mechanism*. A replay count
  catching a replay is detection.
* **An artefact** — the feature measures something the *generator* did that the
  attack does not require. ``mnd_deliberation_residual_z`` at 0.99 is this:
  ``collapse_deliberation`` resamples latency from the low band unconditionally,
  so a ₹50,000 purchase gets a ₹200 purchase's deliberation time. The signal is
  real; the magnitude is the generator's.

This script therefore **flags and ranks rather than passing or failing**. Every
feature above the gate is printed with its slice, its n, and the attack it
separates, and each one has to be adjudicated by a human against that
distinction. Known adjudications live in :data:`ADJUDICATED` so that the list
shrinks to genuinely new findings on each run.

    python scripts/probe_derived.py
    python scripts/probe_derived.py --n 200000     # the gate-sized background
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # so the script runs without an editable install
    sys.path.insert(0, str(REPO_ROOT))

from mantis.defense.features import FeatureBuilder  # noqa: E402
from mantis.foundry.base.reference import load_reference_stats  # noqa: E402
from mantis.foundry.base.simulator import SimulationConfig, simulate_frame  # noqa: E402
from mantis.foundry.injectors import REGISTRY, get_injector  # noqa: E402
from mantis.foundry.injectors.base import PopulationView, run_injector  # noqa: E402
from mantis.foundry.injectors.probe import (  # noqa: E402
    GATE_AUC,
    THIN_SLICE_ROWS,
    _stump_auc,
    build_slices,
)
from mantis.foundry.llm.corpus import load_content_store  # noqa: E402

#: Sentinel for a missing derived value. Far below any real feature so that a
#: stump can split "absent" off cleanly — the probe *wants* nullity to be
#: measurable, unlike L2, which had to stop it dominating.
_MISSING: Final[float] = -1.0e12

#: Categoricals above this many levels are identifiers and are skipped, matching
#: the raw probe's rule.
_MAX_LEVELS: Final[int] = 32

#: Findings already adjudicated by a human, with the verdict. A finding on this
#: list is still measured and still printed — it is moved out of the "NEW" block,
#: not out of the report, because a number that stops being printed is a number
#: that stops being checked.
ADJUDICATED: Final[dict[tuple[str, str], str]] = {
    ("F1-01", "mnd_deliberation_residual_z"): (
        "ARTEFACT. collapse_deliberation resamples latency from the low band "
        "unconditionally, so a large ticket gets a small ticket's deliberation time. "
        "Signal real, magnitude the generator's. Ablated in RESULTS.md: removing it "
        "costs F1 only 0.569 -> 0.549. The foundry fix -- collapse latency relative to "
        "what the amount deserves -- is an outstanding item because it re-rolls every "
        "Day 3 number."
    ),
    ("F1-01", "mnd_amount_over_ceiling"): (
        "LEGITIMATE. amount / scope_max is F1-01's own second declared observable signal "
        "('settled amount exceeds the cart ceiling the human saw, usually by a modest "
        "margin chosen to stay under review thresholds', card F1-01, layer L1). Note it "
        "is the RATIO, not the breach flag: mnd_ceiling_breached stays at zero, which is "
        "why F1-01 remains a CLEAN attack that trips no L0 clause. A graded feature "
        "reading a graded mechanism is detection."
    ),
    ("F1-04", "mnd_mcc_in_scope"): (
        "DEFINITIONAL. F1-04 *is* intent-mandate category drift -- transacting outside "
        "the categories the human's mandate named. 'The MCC is not in scope' is not "
        "evidence about the attack, it is the attack, in the same way txn_type=refund is "
        "for F1-03. This is why F1-04 is a HARD card and why its recall is 0.939: that "
        "number is a rule firing, not a model learning, and the writeup must not present "
        "it as the latter."
    ),
    ("F1-05", "mnd_delegation_depth"): (
        "KNOWN AND ALREADY PRICED. CLAUDE.md records this: the population's delegation "
        "tail was widened to depth 5 precisely because depth >= 4 had been a perfect "
        "detector, and the raw-column probe now reads 0.94 -- 'the closest number in the "
        "atlas to the gate, and reported as such'. 0.952 here is the same quantity on a "
        "6,002-row slice. F1-05's own docstring concedes depth alone will not carry the "
        "card and that the real answer is L4."
    ),
    ("F1-10", "vel_mandate_hash_lifetime_count"): (
        "LEGITIMATE. Counting prior presentations of the same signed mandate digest "
        "IS the replay detector. A feature catching the mechanism it was written for "
        "is detection, not leakage."
    ),
    ("F6-39", "ent_mcc_amount_z"): (
        "LEGITIMATE MECHANISM, NARROW GENERATOR. The injector's own docstring says 'the "
        "tell is amount *given* mcc, never amount alone', and ent_mcc_amount_z is exactly "
        "that quantity -- miscoding a high-ticket sale under a low-ticket category is what "
        "transaction laundering is. The caveat is the runner-up: mcc=7832 alone scores "
        "0.904, because the injector draws its declared category from only six MCCs. That "
        "part is generator narrowness, not attack signal, and widening _DECLARED_MCCS is "
        "an outstanding foundry item."
    ),
    ("F6-40", "txn_round_score"): (
        "ARTEFACT, AND EXACTLY THE BLIND SPOT THIS SCRIPT EXISTS FOR. The raw-column probe "
        "measures a binary 'is a round number' and reads it as unremarkable -- the "
        "injector's docstring says so, and cites the background's own 16% round-snapping "
        "as the reason. txn_round_score is *graded* (how round, over several step sizes), "
        "it is not a column in the parquet, and graded it separates F6-40 at 0.966. The "
        "attack does not require every top-up to be that round; a real cash-out ring has "
        "untidy amounts too. Widening the injector's snapping is an outstanding foundry "
        "item, and until it is done F6-40's recall should be read with this attached."
    ),
}


def numeric_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Explode the feature matrix into columns a depth-1 stump can split on."""
    out: dict[str, np.ndarray] = {}
    for column in features.columns:
        series = features[column]
        if str(series.dtype) == "category":
            levels = series.dropna().unique()
            if len(levels) > _MAX_LEVELS:
                continue
            as_string = series.astype("string")
            for level in sorted(map(str, levels)):
                out[f"{column}={level}"] = (
                    as_string.eq(level).fillna(False).to_numpy(dtype=float)
                )
            continue
        if pd.api.types.is_bool_dtype(series):
            out[column] = series.to_numpy(dtype=float)
            continue
        if pd.api.types.is_numeric_dtype(series):
            values = series.to_numpy(dtype=float)
            out[column] = np.nan_to_num(values, nan=_MISSING, posinf=1e12, neginf=-1e12)
            # Nullity separately, because "absent" and "very small" are different
            # facts and a single sentinel column conflates them for a stump.
            if np.isnan(values).any():
                out[f"{column}_isnull"] = np.isnan(values).astype(float)
    return pd.DataFrame(out, index=features.index)


def probe_derived(
    matrix: pd.DataFrame,
    labels: np.ndarray,
    mask: np.ndarray,
    *,
    top: int = 5,
) -> pd.DataFrame:
    """Best single derived feature separating ``labels`` inside ``mask``."""
    rows = []
    values = matrix.to_numpy(dtype=float)
    y = labels[mask]
    if y.sum() == 0 or y.sum() == len(y):
        return pd.DataFrame(columns=["feature", "auc", "direction"])
    sub = values[mask]
    for index, name in enumerate(matrix.columns):
        auc, direction = _stump_auc(sub[:, index], y)
        rows.append((name, auc, direction))
    return (
        pd.DataFrame(rows, columns=["feature", "auc", "direction"])
        .sort_values("auc", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/probe_derived.py")
    parser.add_argument("--n", type=int, default=60_000, help="background events")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--top", type=int, default=4, help="runners-up printed per attack")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    load_content_store()
    print("MANTIS derived-feature separability probe")
    print("=" * 96)
    print("  The raw-column gate cannot see a feature that is two columns and a regression.")
    print("  This runs the same probe over the matrix L1 actually trains on.")
    print()

    stats = load_reference_stats()
    background = simulate_frame(
        SimulationConfig(n_events=args.n, seed=args.seed, window_days=90), stats
    )
    view = PopulationView.build(background)
    print(f"  background {len(background):,} events")

    attacks = {
        card: run_injector(get_injector(card), view, seed=args.seed)
        for card in sorted(REGISTRY)
    }
    combined = pd.concat([background, *attacks.values()], ignore_index=True)
    combined = combined.sort_values("ts", kind="stable").reset_index(drop=True)
    print(f"  {len(attacks)} injectors, {len(combined):,} rows total")

    print("  building the feature matrix...")
    builder = FeatureBuilder()
    X = builder.fit_transform_stream(combined, pd.Series(True, index=combined.index))
    matrix = numeric_matrix(X)
    print(f"  {X.shape[1]} features -> {matrix.shape[1]} probe columns")
    print()

    attack_id = combined["attack_id"].fillna("").to_numpy()
    is_background = attack_id == ""
    slices = build_slices(background, REGISTRY)

    findings: list[dict[str, object]] = []
    print(f"  {'card':<8} {'slice n':>9} {'best derived feature':<40} {'auc':>7} {'dir':>4}")
    print(f"  {'-' * 8} {'-' * 9} {'-' * 40} {'-' * 7} {'-' * 4}")

    for card in sorted(attacks):
        is_card = attack_id == card
        card_slice = slices.get(card)
        if card_slice is not None:
            # The slice was declared over the background; extend it to the
            # combined frame, where every attack row is inside its own slice by
            # construction (the slice audit proves the mask is a function of the
            # declared columns alone, and the attack rows satisfy them).
            background_positions = np.flatnonzero(is_background)
            in_slice = np.zeros(len(combined), dtype=bool)
            in_slice[background_positions] = card_slice
            comparison = in_slice | is_card
        else:
            comparison = is_background | is_card

        n_slice = int((comparison & is_background).sum())
        ranked = probe_derived(matrix, is_card, comparison, top=args.top)
        if ranked.empty:
            continue
        best = ranked.iloc[0]
        flag = "" if best["auc"] <= GATE_AUC else "  <== ABOVE GATE"
        thin = "!" if n_slice < THIN_SLICE_ROWS else " "
        print(f"  {card:<8} {n_slice:>8,}{thin} {best['feature']:<40} "
              f"{best['auc']:>7.3f} {best['direction']:>4}{flag}")
        for runner in ranked.iloc[1:].itertuples():
            print(f"  {'':<8} {'':>9} {runner.feature:<40} {runner.auc:>7.3f} "
                  f"{runner.direction:>4}")
        if best["auc"] > GATE_AUC:
            findings.append(
                {
                    "attack_id": card,
                    "feature": str(best["feature"]),
                    "auc": float(best["auc"]),
                    "slice_n": n_slice,
                }
            )
        print()

    print("=" * 96)
    if not findings:
        print(f"  No derived feature separates any attack above {GATE_AUC:.2f} inside its slice.")
    else:
        known = [f for f in findings if (f["attack_id"], f["feature"]) in ADJUDICATED]
        new = [f for f in findings if (f["attack_id"], f["feature"]) not in ADJUDICATED]
        print(f"  {len(findings)} feature(s) above the {GATE_AUC:.2f} gate: "
              f"{len(known)} already adjudicated, {len(new)} NEW.")
        print()
        for finding in known:
            key = (finding["attack_id"], finding["feature"])
            print(f"  [known] {finding['attack_id']} {finding['feature']} "
                  f"{finding['auc']:.3f}")
            print(f"          {ADJUDICATED[key]}")
            print()
        for finding in new:
            print(f"  [NEW]   {finding['attack_id']} {finding['feature']} "
                  f"{finding['auc']:.3f}  (slice n={finding['slice_n']:,})")
            print("          Adjudicate: does this feature measure the attack's MECHANISM "
                  "(legitimate) or")
            print("          something the GENERATOR did that the attack does not require "
                  "(artefact)?")
            print("          Record the verdict in ADJUDICATED at the top of this script.")
            print()
        if new:
            print("  Exiting non-zero because there is an unadjudicated finding. That is not a")
            print("  failure of the data — it is a finding that has not been read by a human "
                  "yet.")

    print(f"  {time.perf_counter() - started:.0f}s")
    return 1 if any((f["attack_id"], f["feature"]) not in ADJUDICATED for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
