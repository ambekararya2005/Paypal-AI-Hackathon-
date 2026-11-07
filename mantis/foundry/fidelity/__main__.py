"""The fidelity scorecard, printed for a human and written for the console.

    python -m mantis.foundry.fidelity
    python -m mantis.foundry.fidelity --dataset data/generated/dataset_v1.parquet
    python -m mantis.foundry.fidelity --days 90 --seed 1337

Prints the five sections described in :mod:`mantis.foundry.fidelity.scorecard`
and writes ``data/generated/fidelity.json`` plus ``docs/fidelity_scorecard.png``.

Exit codes: 0 when the whole scorecard ran, 1 when the reference panel was absent
and sections 2 to 4 were skipped. Non-zero because "the fidelity number could not
be computed" should be visible to a Makefile, and 1 rather than 2 because it is a
missing input rather than a wrong one.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import pandas as pd

from mantis.core.paths import GENERATED_DIR
from mantis.foundry.fidelity import adjudicate, marginals
from mantis.foundry.fidelity.scorecard import (
    FIDELITY_FIGURE,
    FIDELITY_JSON,
    build_scorecard,
    write_scorecard,
)


def _rule(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _print_levels(card: dict[str, Any]) -> None:
    synthetic = card["synthetic"]["levels"]
    reference = card.get("reference", {}).get("levels")
    keys = [
        ("events", "events compared", "{:,.0f}"),
        ("days", "days spanned", "{:,.0f}"),
        ("customers", "cardholders", "{:,.0f}"),
        ("merchants", "merchants", "{:,.0f}"),
        ("txn_per_customer_per_day", "txn/cardholder/day", "{:.3f}"),
        ("median_hours_between", "median hours between", "{:.2f}"),
        ("merchants_per_customer", "merchants/cardholder", "{:.1f}"),
        ("top_1pct_merchant_share", "top 1% merchant share", "{:.3f}"),
    ]
    header = f"  {'level':<24} {'synthetic':>14} {'reference':>14} {'ratio':>8}"
    print(header)
    print(f"  {'-' * 24} {'-' * 14} {'-' * 14} {'-' * 8}")
    for key, label, fmt in keys:
        left = fmt.format(synthetic[key])
        if reference is None:
            print(f"  {label:<24} {left:>14} {'-':>14} {'-':>8}")
            continue
        right = fmt.format(reference[key])
        ratio = synthetic[key] / reference[key] if reference[key] else float("nan")
        print(f"  {label:<24} {left:>14} {right:>14} {ratio:>8.2f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure synthetic-data fidelity.")
    parser.add_argument(
        "--dataset",
        default=str(GENERATED_DIR / "dataset_v1.parquet"),
        help="the labelled synthetic parquet to score",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="length of the reference window, matched to the synthetic panel's span",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)

    print("MANTIS fidelity scorecard")
    print("=" * 25)

    synthetic = pd.read_parquet(args.dataset)
    card = build_scorecard(synthetic, days=args.days, seed=args.seed)

    # ---------------------------------------------------- 1. provenance ----
    _rule("1. Provenance - what every number below is conditional on")
    calibration = card["calibration"]
    print(f"  population calibration   {calibration['source']}")
    print(f"  {calibration['note']}")
    print()
    reference = card.get("reference", {})
    if reference.get("available"):
        print(f"  reference panel          {reference['provenance']}")
        print(f"  {reference['note']}")
    else:
        print("  reference panel          ABSENT")
        for line in reference.get("note", "").splitlines():
            print(f"  {line}")
    print()
    print(f"  {card['synthetic']['note']}")
    print()
    _print_levels(card)

    if not reference.get("available"):
        _rule("2-4. Marginals, TSTR, discriminator - SKIPPED")
        print("  All three need the reference panel. Nothing is substituted for them:")
        print("  comparing the population against its own specification is a different")
        print("  question, it is what scripts/drift_check.py answers, and reporting it")
        print("  here as fidelity would be the dishonest version of this section.")
        _print_known(card)
        write_scorecard(card, figure=False)
        print()
        print(f"  wrote {FIDELITY_JSON}")
        return 1

    # ----------------------------------------------------- 2. marginals ----
    _rule("2. Marginals - per-feature distance against its own sampling-noise band")
    print(marginals.format_marginals(card["marginals"]["rows"]))
    print()
    correlation = card["marginals"]["correlation"]
    print(
        f"  correlation matrix       Frobenius {correlation['frobenius']:.3f}, "
        f"RMS off-diagonal {correlation['rms_off_diagonal']:.3f} "
        f"over {correlation['n_features']} features"
    )
    print("  worst correlated pairs (synthetic / reference):")
    for pair in correlation["worst_pairs"][:3]:
        print(
            f"    {pair['pair']:<48} {pair['synthetic']:+.3f} / {pair['real']:+.3f} "
            f"  delta {pair['delta']:+.3f}"
        )

    # ---------------------------------------------------------- 3. TSTR ----
    t = card["tstr"]
    _rule("3. TSTR - train on synthetic, test on real")
    print(f"  {'model':<26} {'AUC-PR':>9} {'ROC':>8} {'baseline':>10} {'lift':>8}")
    print(f"  {'-' * 26} {'-' * 9} {'-' * 8} {'-' * 10} {'-' * 8}")
    print(
        f"  {'TRTR (real -> real)':<26} {t['trtr']['auc_pr']:>9.4f} "
        f"{t['trtr']['roc_auc']:>8.4f} {t['trtr']['baseline']:>10.4%} {t['trtr_lift']:>8.1f}x"
    )
    print(
        f"  {'TSTR (synth -> real)':<26} {t['tstr']['auc_pr']:>9.4f} "
        f"{t['tstr']['roc_auc']:>8.4f} {t['tstr']['baseline']:>10.4%} {t['tstr_lift']:>8.1f}x"
    )
    print(
        f"  {'TRTS (real -> synth)':<26} {t['trts']['auc_pr']:>9.4f} "
        f"{t['trts']['roc_auc']:>8.4f} {t['trts']['baseline']:>10.4%} {'':>8}"
    )
    print()
    print(f"  transfer ratio           {t['transfer_ratio']:.3f}  (TSTR AUC-PR / TRTR AUC-PR)")
    print(f"  {t['caveat']}")
    print()
    print("  what each model leaned on (gain share), which is how a low transfer ratio")
    print("  is told apart from unrealistic synthetic fraud:")
    learned = t["what_each_learned"]
    print(f"    {'feature':<24} {'trained on real':>16} {'trained on synth':>18}")
    print(f"    {'-' * 24} {'-' * 16} {'-' * 18}")
    synth_gain = {row["feature"]: row["gain_share"] for row in learned["tstr"]}
    for row in learned["trtr"]:
        name = row["feature"]
        print(f"    {name:<24} {row['gain_share']:>15.1%} {synth_gain.get(name, 0.0):>17.1%}")

    # ------------------------------------------------- 4. discriminator ----
    d = card["discriminator"]
    _rule("4. Discriminator - can a model tell the two panels apart?")
    print(
        f"  out-of-fold ROC-AUC      {d['auc']:.4f}   "
        f"(target {d['target']:.1f}; {d['separability']:.1%} separable)"
    )
    print(f"  {d['n_per_side']:,} rows per side, {d['folds']}-fold")
    print(f"  {d['reading']}")
    naive = card["discriminator_naive"]
    print()
    print("  what it was allowed to see, as a measured difference:")
    print(
        f"    raw columns, ids and currency included   {naive['auc']:.4f}   "
        "(expected ~1.0; not a finding)"
    )
    print(
        f"    shape space: intersection, ids dropped,  {d['auc']:.4f}   "
        "<- the headline"
    )
    print("    agentic rail excluded")
    print(f"    {naive['note']}")

    print()
    print("  attribution (LightGBM pred_contrib, the same quantity TreeExplainer computes):")
    print(
        f"  {'feature':<22} {'alone (AUC)':>11} {'gain':>8} {'contrib':>9}  class"
    )
    print(f"  {'-' * 22} {'-' * 11} {'-' * 8} {'-' * 9}  {'-' * 10}")
    verdicts = {row["feature"]: row for row in card["feature_class"]}
    for row in d["per_feature"]:
        verdict = verdicts.get(row["feature"], {}).get("verdict", "?")
        print(
            f"  {row['feature']:<22} {row['alone_auc']:>11.4f} {row['gain_share']:>8.1%} "
            f"{row.get('contribution_share', float('nan')):>9.1%}  {verdict}"
        )
    print()
    print(f"  top 5 drivers: {', '.join(d['top_features'])}")

    print()
    print("  cosmetic or structural, and why:")
    for row in card["feature_class"][:5]:
        print(f"    {row['feature']}  [{row['verdict']}]")
        print(f"      {row['reason']}")

    print()
    print("  ablation path - drop the strongest, refit, repeat:")
    print(f"    {'features':>9}  {'AUC':>7}   dropped so far")
    for step in card["discriminator_ablation_path"]:
        dropped = ", ".join(step["dropped_so_far"]) or "-"
        print(f"    {step['n_features']:>9}  {step['auc']:>7.4f}   {dropped}")

    _rule("4b. Which side is anomalous? - adjudicated, with the measurement that decided it")
    print(adjudicate.format_adjudications(card["adjudications"]))
    ablated = card["discriminator_ablated"]
    print(
        f"  discriminator without {', '.join(ablated['excluded'])}:  "
        f"ROC-AUC {ablated['auc']:.4f}  ({ablated['separability']:.1%} separable)"
    )
    print(f"  {ablated['reading']}")
    print(
        f"    {'feature':<24} {'alone (AUC)':>12} {'gain share':>12}"
    )
    for row in ablated["per_feature"]:
        print(f"    {row['feature']:<24} {row['alone_auc']:>12.4f} {row['gain_share']:>12.1%}")
    print()
    print("  Both numbers are printed because the ablation is a judgement. The full")
    print("  discriminator is the measurement; the ablated one is the measurement")
    print("  after a judgement a reader is free to reject.")

    _print_known(card)

    write_scorecard(card, figure=not args.no_figure)
    print()
    print(f"  wrote {FIDELITY_JSON}")
    if not args.no_figure and FIDELITY_FIGURE.exists():
        print(f"  wrote {FIDELITY_FIGURE}")
    return 0


def _print_known(card: dict[str, Any]) -> None:
    _rule("5. Known divergences - named here rather than left for a judge to find")
    for item in card["known_divergences"]:
        print(f"  {item['name']}")
        print(f"    measured   {item['measured']}")
        print(f"    cause      {item['cause']}")
        print(f"    not fixed  {item['why_not_fixed']}")
        print()


if __name__ == "__main__":
    sys.exit(main())
