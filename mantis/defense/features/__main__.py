"""Build the feature matrix and print what it contains.

    python -m mantis.defense.features [--dataset data/generated/dataset_v1.parquet]

Prints the group breakdown, the per-row cost of each group (which is the number
Day 5's latency budget is spent against), the state store's memory profile, and
a discriminative-power table for a handful of features chosen because they are
the ones the amendment made buildable. Then it deliberately tries to smuggle a
label into the matrix, so the reader can see the assertion fire.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mantis.core.paths import GENERATED_DIR
from mantis.defense.features.builder import FeatureBuilder, LeakageError
from mantis.defense.features.spec import FORBIDDEN_COLUMNS, VELOCITY_KEYS
from mantis.defense.features.state import WINDOWS


def _split(frame: pd.DataFrame, train_share: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based split. Never random: a random split trains on the future."""
    frame = frame.sort_values("ts", kind="stable").reset_index(drop=True)
    cut = frame["ts"].quantile(train_share)
    return frame[frame["ts"] <= cut], frame[frame["ts"] > cut]


def _power_table(matrix: pd.DataFrame, labels: pd.Series, attack_ids: pd.Series) -> str:
    """Rank AUC of a few named features, overall and against the attack they target."""
    from sklearn.metrics import roc_auc_score

    interesting: list[tuple[str, str]] = [
        ("vel_bin_decline_ratio_24h", "F4-27"),
        ("vel_merchant_decline_ratio_24h", "F4-28"),
        ("vel_mandate_hash_lifetime_count", "F1-10"),
        ("mnd_age_over_ttl", "F1-10"),
        ("mnd_mcc_in_scope", "F1-02"),
        ("mnd_deliberation_residual_z", "F1-01"),
        ("ent_amount_vs_customer_p99", "F3-19"),
        ("ent_merchant_refund_ratio", "F1-03"),
        ("txn_orphan_outbound", "F1-03"),
        ("vel_customer_count_1h", "F6-38"),
    ]
    lines = [
        f"  {'feature':<34} {'target':<7} {'AUC vs all':>11} {'AUC vs target':>14}",
        f"  {'-' * 34} {'-' * 7} {'-' * 11} {'-' * 14}",
    ]
    y = labels.to_numpy()
    for name, target in interesting:
        if name not in matrix.columns:
            continue
        column = matrix[name].to_numpy(dtype=float)
        fallback = float(np.nanmedian(column)) if np.isfinite(column).any() else 0.0
        filled = np.nan_to_num(column, nan=fallback)
        overall = roc_auc_score(y, filled) if y.any() and not y.all() else float("nan")
        focus = (attack_ids == target).to_numpy() | ~y
        y_focus = y[focus]
        targeted = (
            roc_auc_score(y_focus, filled[focus])
            if y_focus.any() and not y_focus.all()
            else float("nan")
        )
        lines.append(
            f"  {name:<34} {target:<7} {max(overall, 1 - overall):>11.3f} "
            f"{max(targeted, 1 - targeted):>14.3f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mantis.defense.features")
    parser.add_argument("--dataset", type=Path, default=GENERATED_DIR / "dataset_v1.parquet")
    parser.add_argument("--train-share", type=float, default=0.7)
    args = parser.parse_args(argv)

    if not args.dataset.exists():
        print(f"no dataset at {args.dataset}; run `make dataset` first", file=sys.stderr)
        return 1

    frame = pd.read_parquet(args.dataset)
    train, test = _split(frame, args.train_share)

    print("=" * 78)
    print("MANDATE FIREWALL - feature layer")
    print("=" * 78)
    print(f"  dataset  {args.dataset.name}: {len(frame):,} events, "
          f"{int(frame['is_fraud'].sum()):,} fraud")
    print(f"  split    TIME-BASED at the {args.train_share:.0%} quantile of ts")
    print(f"           train {len(train):,} ({train['is_fraud'].sum():,} fraud)  "
          f"test {len(test):,} ({test['is_fraud'].sum():,} fraud)")
    print()

    t0 = time.perf_counter()
    builder = FeatureBuilder().fit(train)
    t1 = time.perf_counter()
    matrix = builder.transform(test)
    t2 = time.perf_counter()

    counts = builder.group_counts()
    print("feature groups")
    print()
    for group, count in counts.items():
        print(f"  {group:<14} {count:>4}")
    print(f"  {'TOTAL':<14} {sum(counts.values()):>4}")
    print()
    print(f"  velocity keys   : {', '.join(s.name for s in VELOCITY_KEYS)}")
    print(f"  velocity windows: {', '.join(WINDOWS)}")
    print()
    print(f"  fit on {len(train):,} rows      {t1 - t0:6.2f}s")
    print(f"  transform {len(test):,} rows  {t2 - t1:6.2f}s "
          f"({(t2 - t1) / max(len(test), 1) * 1000:.3f} ms/row)")
    print()
    print("  The per-row number is the one Day 5's p99 budget is spent against, and it")
    print("  is a single forward pass over a keyed state store rather than a groupby.")
    print("  See features/state.py: a rescan would re-read the card's whole history to")
    print("  score one authorisation, which no 50 ms budget survives.")
    print()

    print("discriminative power of the features amendment 1.1.0 made buildable")
    print()
    print(_power_table(matrix, test["is_fraud"], test["attack_id"].fillna("")))
    print()
    print("  'AUC vs target' scores that feature against ONLY the attack it was built")
    print("  for, plus legitimate traffic. A feature can be near-useless overall and")
    print("  decisive on one family; that is what a five-layer firewall is for.")
    print()

    print("the leakage assertion, fired deliberately")
    print()
    for column in ("is_fraud", "attack_id", "auth_response", "dispute_outcome"):
        probe = matrix.copy()
        probe[column] = test[column].to_numpy()
        try:
            builder._assert_no_leakage(probe)
        except LeakageError:
            print(f"  {column:<16} REJECTED")
        else:
            print(f"  {column:<16} *** ACCEPTED - THE ASSERTION IS BROKEN ***")
            return 1
    print()
    print(f"  three tiers, {len(FORBIDDEN_COLUMNS)} columns: ground truth, post-hoc dispute")
    print("  state, and the current event's own outcome. The third is the one that is")
    print("  easy to leak by accident - see features/spec.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
