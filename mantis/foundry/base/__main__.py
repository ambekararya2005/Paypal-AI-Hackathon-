"""CLI for the population foundry.

    python -m mantis.foundry.base --n 200000 --seed 7

Writes ``data/generated/population.parquet``, a sidecar manifest recording every
input that shaped the run, and the calibration figure in ``docs/``. Nothing here
touches the network, reads a credential, or downloads anything: a clean clone
with no Kaggle token produces the full population from the committed priors.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from mantis.core.paths import POPULATION_PARQUET, ensure_dir
from mantis.foundry.base.calibration import (
    CALIBRATION_PNG,
    calibration_report,
    format_report,
    plot_calibration,
)
from mantis.foundry.base.entities import build_population
from mantis.foundry.base.reference import load_reference_stats
from mantis.foundry.base.simulator import (
    DEFAULT_SEED,
    DEFAULT_WINDOW_DAYS,
    SimulationConfig,
    simulate_frame,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m mantis.foundry.base",
        description="Generate the calibrated legitimate payment population.",
    )
    parser.add_argument("--n", type=int, default=200_000, help="number of events (default 200000)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="master seed")
    parser.add_argument("--customers", type=int, default=5_000, help="cardholder count")
    parser.add_argument("--merchants", type=int, default=12_000, help="approx merchant count")
    parser.add_argument(
        "--window-days", type=int, default=DEFAULT_WINDOW_DAYS, help="observation window"
    )
    parser.add_argument("--out", type=Path, default=POPULATION_PARQUET, help="output parquet path")
    parser.add_argument(
        "--figure", type=Path, default=CALIBRATION_PNG, help="calibration figure path"
    )
    parser.add_argument("--no-figure", action="store_true", help="skip the calibration figure")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Generate, validate, write, measure, draw. Run as ``python -m mantis.foundry.base``."""
    args = _parse_args(argv)

    cfg = SimulationConfig(
        n_events=args.n,
        seed=args.seed,
        n_customers=args.customers,
        n_merchants=args.merchants,
        window_days=args.window_days,
    )
    stats = load_reference_stats()

    print(stats.describe())
    print()

    t0 = time.perf_counter()
    pop = build_population(
        stats, seed=cfg.seed, n_customers=cfg.n_customers, n_merchants=cfg.n_merchants
    )
    print(pop.describe())
    print()

    t1 = time.perf_counter()
    frame = simulate_frame(cfg, stats, pop)
    t2 = time.perf_counter()

    ensure_dir(args.out.parent)
    frame.to_parquet(args.out, index=False)
    t3 = time.perf_counter()

    report = calibration_report(frame, stats)
    print(format_report(report, stats))
    print()

    figure = None if args.no_figure else plot_calibration(frame, stats, report, args.figure)

    manifest = {
        "config": {
            "n_events": cfg.n_events,
            "seed": cfg.seed,
            "n_customers": cfg.n_customers,
            "n_merchants": cfg.n_merchants,
            "window_days": cfg.window_days,
            "start": cfg.start.isoformat(),
        },
        "reference": {
            "source": stats.source,
            "currency": stats.currency,
            "provenance": stats.provenance,
        },
        "calibration": report,
        "timings_seconds": {
            "build_population": round(t1 - t0, 3),
            "simulate": round(t2 - t1, 3),
            "write_parquet": round(t3 - t2, 3),
        },
        "outputs": {
            "parquet": str(args.out),
            "figure": str(figure) if figure else None,
        },
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    size_mb = args.out.stat().st_size / 1024 / 1024
    print(f"wrote {len(frame):,} rows x {frame.shape[1]} cols -> {args.out}  ({size_mb:.1f} MB)")
    print(f"      manifest -> {manifest_path}")
    if figure:
        print(f"      figure   -> {figure}")
    else:
        print("      figure   -> skipped (matplotlib unavailable or --no-figure)")
    print(
        f"      timings  -> entities {t1 - t0:.1f}s, simulate {t2 - t1:.1f}s, write {t3 - t2:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
