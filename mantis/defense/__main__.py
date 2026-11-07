"""Run the firewall experiment and write RESULTS.md.

    python -m mantis.defense                 # uses the cached 5-seed pool
    python -m mantis.defense --rebuild-pool  # regenerates it (about a minute)

Produces, in order: the pooled evaluation dataset; the five layers at a recall
**curve** over 0.1/0.5/1.0% FPR, event-level and campaign-level side by side; the
fitted fusion weights; per-rail and per-family breakdowns; the
leave-one-family-out table; L3's two generalisation tests; and the ablation of
the one feature that turned out to be too good.

If ``python -m mantis.loop`` has been run, its evasion curve and the zero-day
comparison are folded into RESULTS.md as well — those two together are the
submission's argument, and they belong in the same document as everything they
are being compared against.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

from mantis.core.paths import GENERATED_DIR, REPO_ROOT, ensure_dir
from mantis.defense.experiment import LAYER_ORDER, ExperimentResult, run_experiment
from mantis.defense.metrics import FPR_GRID, OPERATING_FPR
from mantis.defense.pool import POOL_SEEDS, build_pool
from mantis.defense.report import write_results

POOL_PARQUET: Path = GENERATED_DIR / "pool_5seed.parquet"

#: The fitted experiment, cached so RESULTS.md can be re-rendered without
#: refitting eight LightGBM models.
#:
#: This exists because the document's *prose* gets edited far more often than its
#: numbers do, and a fifteen-minute refit to fix a sentence is a fifteen-minute
#: refit nobody does — which is how a generated document quietly starts being
#: hand-edited, and how it stops being true. ``--render-only`` re-runs the
#: renderer against the cached result and nothing else.
RESULT_CACHE: Path = GENERATED_DIR / "experiment_result.pkl"


def load_pool(rebuild: bool) -> pd.DataFrame:
    """Load the pooled dataset, generating it when absent or when asked."""
    if rebuild or not POOL_PARQUET.exists():
        print(f"building the pooled dataset over {len(POOL_SEEDS)} seeds...")
        pool = build_pool()
        ensure_dir(POOL_PARQUET.parent)
        pool.to_parquet(POOL_PARQUET, index=False)
        print(f"  wrote {POOL_PARQUET}")
        return pool
    print(f"loading cached pool from {POOL_PARQUET}")
    return pd.read_parquet(POOL_PARQUET)


def print_summary(result: ExperimentResult) -> None:
    """The block a judge reads off the terminal."""
    print()
    print("=" * 96)
    print("MANDATE FIREWALL - Day 5 results")
    print("=" * 96)
    print(f"  train {result.n_train:,}  test {result.n_test:,}  "
          f"features {result.n_features} (graph {result.graph_features})  "
          f"test prevalence {result.prevalence:.4%}")
    print(f"  OPERATING POINT: recall is quoted at "
          f"{', '.join(f'{f:.1%}' for f in FPR_GRID)} FPR on legitimate traffic. "
          f"Headline is {OPERATING_FPR:.1%}.")
    print()

    grid = "  ".join(f"@{f:.1%}" for f in FPR_GRID)
    print(f"  {'layer':<8} {'AUC-PR':>8} {'ROC':>7}   {grid}   {'campaign':>9} {'first alert':>12}")
    print(f"  {'-' * 8} {'-' * 8} {'-' * 7}   {'-' * len(grid)}   {'-' * 9} {'-' * 12}")
    for name in LAYER_ORDER:
        layer = result.layers[name]
        cells = "  ".join(f"{layer.curve[f][0]:>6.3f}" for f in FPR_GRID)
        camp = layer.campaigns
        first = "n/a" if camp.median_index != camp.median_index else f"{camp.median_index:.0f}"
        print(f"  {name:<8} {layer.report.auc_pr:>8.4f} {layer.report.auc_roc:>7.4f}   {cells}   "
              f"{camp.recall:>9.3f} {first:>12}")
    print()
    print("  event-level recall on the left, campaign-level (was the ring flagged at all) on")
    print("  the right. Both are labelled because each flatters a different kind of detector.")
    print()

    if len(result.fusion_weights):
        print("  fusion weights, fitted on an inner split of the training window")
        for row in result.fusion_weights.itertuples():
            present = (
                "     n/a" if row.weight_present != row.weight_present
                else f"{row.weight_present:+8.3f}"
            )
            print(f"    {row.layer:<6} percentile {row.weight_percentile:+7.3f}   "
                  f"raw {row.weight_score:+7.3f}   present {present}")
        print()

    if result.l1_rail:
        print("  per rail (L1) - because fraud is 5.7x concentrated on the agentic rail")
        for rail, report in result.l1_rail.items():
            print(report.line(f"    {rail}"))
        print()

    print("-" * 96)
    print("LEAVE ONE FAMILY OUT - the headline experiment")
    print("-" * 96)
    print()
    frame = result.per_family
    print(f"  {'family':<8} {'n_pos':>7} {'L1 with':>9} {'L1 HELD':>9} {'L2':>7} {'L3':>7} "
          f"{'fused':>7} {'fused HELD':>11}")
    print(f"  {'-' * 8} {'-' * 7} {'-' * 9} {'-' * 9} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 11}")
    for row in frame.itertuples():
        print(f"  {row.family:<8} {row.n_pos:>7,} {row.l1_with:>9.3f} {row.l1_heldout:>9.3f} "
              f"{row.l2:>7.3f} {row.l3:>7.3f} {row.fused_with:>7.3f} {row.fused_heldout:>11.3f}")
    if len(frame):
        drop = (frame["l1_with"] - frame["l1_heldout"]).mean()
        print()
        print(f"  mean recall lost when the family is held out of training: {drop:+.3f}")
    print()

    if len(result.per_family_campaign):
        print("  the same experiment at CAMPAIGN level (fused score)")
        print(f"  {'family':<8} {'rings':>6} {'median size':>12} {'caught':>8} {'HELD':>8} "
              f"{'1st alert':>10} {'elapsed':>8}")
        for row in result.per_family_campaign.itertuples():
            index = "n/a" if row.median_index != row.median_index else f"{row.median_index:.0f}"
            elapsed = (
                "n/a" if row.share_before_alert != row.share_before_alert
                else f"{row.share_before_alert:.0%}"
            )
            print(f"  {row.family:<8} {row.n_campaigns:>6} {row.median_size:>12.0f} "
                  f"{row.fused_with:>8.3f} {row.fused_heldout:>8.3f} {index:>10} {elapsed:>8}")
        print()

    if len(result.l3_cards):
        print("-" * 96)
        print("L3 - the text layer, and its two generalisation tests")
        print("-" * 96)
        print()
        print(f"  {'card':<8} {'n_pos':>7} {'recall':>8} {'unseen phrasing':>17} "
              f"{'n':>6} {'unseen KIND':>13}")
        for row in result.l3_cards.itertuples():
            print(f"  {row.attack_id:<8} {row.n_pos:>7,} {row.recall:>8.3f} "
                  f"{row.recall_unseen_phrasing:>17.3f} {row.n_unseen_phrasing:>6,} "
                  f"{row.recall_unseen_kind:>13.3f}")
        print()
        print("  L3 is fitted on TEXT, not on transaction labels. It has no `y` parameter.")
        print()

    if len(result.per_attack):
        print("-" * 96)
        print("per attack card, at the same operating point")
        print("-" * 96)
        print()
        header = " ".join(f"{name:>7}" for name in LAYER_ORDER)
        print(f"  {'card':<8} {'n_pos':>7} {header} {'rings':>7}")
        for row in result.per_attack.itertuples():
            cells = " ".join(f"{getattr(row, name):>7.3f}" for name in LAYER_ORDER)
            print(f"  {row.attack_id:<8} {row.n_pos:>7,} {cells} {row.campaign:>7.3f}")
        print()

    if result.decisions:
        total = max(sum(result.decisions.values()), 1)
        print("  decision layer over the test window:")
        for name, count in result.decisions.items():
            print(f"    {name:<10} {count:>9,}  {count / total:.3%}")
        print()

    print("-" * 96)
    print("top L1 features by gain")
    print("-" * 96)
    print()
    for row in result.importance.head(12).itertuples():
        print(f"  {row.feature:<38} {row.share:>7.2%}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mantis.defense")
    parser.add_argument("--rebuild-pool", action="store_true", help="regenerate the 5-seed pool")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="re-render RESULTS.md from the cached experiment result, without refitting",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--results", type=Path, default=REPO_ROOT / "RESULTS.md", help="where to write RESULTS.md"
    )
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    pool = load_pool(args.rebuild_pool)
    print(f"  {len(pool):,} events, {int(pool['is_fraud'].sum()):,} fraud "
          f"({pool['is_fraud'].mean():.4%})")
    print()

    if args.render_only:
        if not RESULT_CACHE.exists():
            print(f"no cached result at {RESULT_CACHE}; run without --render-only first")
            return 1
        print(f"re-rendering from {RESULT_CACHE} (no refit)")
        result = pickle.loads(RESULT_CACHE.read_bytes())
    else:
        result = run_experiment(pool, seed=args.seed)
        ensure_dir(RESULT_CACHE.parent)
        RESULT_CACHE.write_bytes(pickle.dumps(result))
        print_summary(result)

    write_results(result, pool, args.results)
    print(f"wrote {args.results}")
    print(f"total {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
