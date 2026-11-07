"""Run the closed loop and write the Day 5 gate artefacts.

    python -m mantis.loop                       # arena + zero-day, on two seeds
    python -m mantis.loop --generations 8       # a longer curve
    python -m mantis.loop --no-zero-day         # arena only, faster
    python -m mantis.loop --family F6           # a different held-out family

Produces ``data/generated/arena.json`` — the evasion curve, the surviving
genomes and the zero-day comparison — and writes any variant that survived
:data:`~mantis.loop.arena.SURVIVAL_ROUNDS` rounds into
``mantis/atlas/discovered/`` as a validated attack card.

Why two seeds and six cards rather than five seeds and the whole atlas
------------------------------------------------------------------------
Cost. Every generation expresses every genome — which runs that card's injector
against the whole background — then rebuilds the feature matrix over background
plus variants and refits L1. So the arena is linear in the background, linear in
the generations, and linear in **cards x population**. Measured: all fifteen
cards at population 8 over two pooled seeds is about two hours, which is not a
gate anybody re-runs.

The defaults are therefore two seeds and :data:`ARENA_CARDS` — six cards chosen
to cover all five implemented families, including both a CLEAN agentic attack and
an entity-level ring. What the evasion curve needs is a **slope**, and a slope
does not get truer by adding cards that behave like the ones already in it. Pass
``--cards`` to evolve any subset, or the whole atlas with a couple of hours to
spare.

The zero-day comparison recomputes its own baseline on the same two seeds, so
every number in its table is measured against the same data — it is not compared
against RESULTS.md's five-seed 0.007, and the CLI says so where it prints it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from mantis.core.paths import GENERATED_DIR
from mantis.defense.pool import POOL_SEEDS, build_pool
from mantis.loop.arena import (
    ARENA_JSON,
    SURVIVAL_ROUNDS,
    run_arena,
    run_zero_day,
    write_arena,
)
from mantis.loop.writeback import is_novel, write_discovered_cards

#: Cached background for the arena. Separate from the firewall's five-seed pool
#: because it is a different size, and silently reusing a file whose seed count
#: does not match the flags would be the sort of thing nobody notices until the
#: numbers disagree with the writeup.
ARENA_POOL: Path = GENERATED_DIR / "pool_2seed.parquet"

#: Cards the arena evolves by default: one per implemented family, plus a second
#: F1 so that both halves of the HARD/CLEAN split are represented.
#:
#: F1-01 is a CLEAN attack whose signal is in the ingested text, so it is the one
#: whose ``provenance_clean`` gene has something to defeat. F1-05 is the
#: delegation-laundering card whose own docstring says depth alone will not carry
#: it. F2-16, F3-19, F4-27 and F6-38 are the bust-out, the coerced transfer, the
#: card-testing campaign and the mule ring — four different shapes of attack, so
#: the curve is an average over genuinely different search problems rather than
#: over six versions of one.
ARENA_CARDS: tuple[str, ...] = ("F1-01", "F1-05", "F2-16", "F3-19", "F4-27", "F6-38")


def load_background(seeds: int, rebuild: bool) -> pd.DataFrame:
    """The arena's background: ``seeds`` pooled worlds, cached on disk."""
    if not rebuild and ARENA_POOL.exists():
        print(f"loading cached arena background from {ARENA_POOL}")
        return pd.read_parquet(ARENA_POOL)
    print(f"building the arena background over {seeds} seeds...")
    pool = build_pool(seeds=POOL_SEEDS[:seeds])
    ARENA_POOL.parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(ARENA_POOL, index=False)
    print(f"  wrote {ARENA_POOL}")
    return pool


def _zero_day_only(background, args, started: float) -> int:
    """Re-run only the zero-day comparison and splice it into an existing arena.json."""
    print("=" * 84)
    print(f"THE ZERO-DAY DEMONSTRATION - family {args.family}  (arena skipped)")
    print("=" * 84)
    zero_day = run_zero_day(
        background,
        family=args.family,
        generations=max(3, args.generations // 2),
        population=max(4, args.population // 2),
        seed=args.seed,
    )
    _print_zero_day(zero_day, args.family)

    if not args.out.exists():
        raise SystemExit(
            f"{args.out} does not exist; run the arena at least once before --zero-day-only"
        )
    payload = json.loads(args.out.read_text(encoding="utf-8"))
    payload["zero_day"] = zero_day.to_json()
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"updated {args.out}")
    print(f"total {time.perf_counter() - started:.0f}s")
    return 0


def _print_zero_day(zero_day, family: str) -> None:
    print()
    print(f"  recall@0.1%FPR on the {zero_day.n_test_positive:,} real {family} test events:")
    print(f"    detector trained WITH the family      {zero_day.recall_trained:.3f}")
    print(f"    family held out of training           {zero_day.recall_heldout:.3f}")
    print(f"    held out, plus loop-manufactured      {zero_day.recall_loop:.3f}   "
          f"<- {zero_day.n_variant_events:,} variant events")
    print(f"    share of the held-out gap closed      {zero_day.gap_closed:.1%}")
    print()
    print("  Measured on this run's own background, not against the five-seed numbers in")
    print("  RESULTS.md. All three rows share one dataset and one operating point.")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mantis.loop")
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--elite", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=2, help="worlds pooled as the background")
    parser.add_argument("--rebuild", action="store_true", help="regenerate the background")
    parser.add_argument(
        "--cards",
        nargs="*",
        default=None,
        help=f"atlas cards to evolve (default: {' '.join(ARENA_CARDS)}); pass 'all' for "
             "every implemented card, which takes hours",
    )
    parser.add_argument("--family", default="F1", help="family held out for the zero-day run")
    parser.add_argument("--no-zero-day", action="store_true")
    parser.add_argument(
        "--zero-day-only",
        action="store_true",
        help="skip the arena and re-run only the zero-day comparison, updating arena.json "
             "in place. The two experiments are independent -- run_zero_day evolves its own "
             "variants against its own held-out detector -- so re-running one does not "
             "invalidate the other.",
    )
    parser.add_argument("--no-writeback", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", type=Path, default=ARENA_JSON)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    background = load_background(args.seeds, args.rebuild)
    print(f"  {len(background):,} events, {int(background['is_fraud'].sum()):,} fraud")
    print()

    if args.zero_day_only:
        return _zero_day_only(background, args, started)

    print("=" * 84)
    print("THE ARENA - evolutionary adversary vs. a retraining detector")
    print("=" * 84)
    if args.cards == ["all"]:
        cards = None
    elif args.cards:
        cards = tuple(args.cards)
    else:
        cards = ARENA_CARDS
    arena = run_arena(
        background,
        cards=cards,
        generations=args.generations,
        population=args.population,
        elite=args.elite,
        seed=args.seed,
    )

    print()
    print(f"  {'gen':>4} {'variants':>9} {'events':>8} {'mean evasion':>13} {'max':>7} "
          f"{'mean fitness':>13}")
    print(f"  {'-' * 4} {'-' * 9} {'-' * 8} {'-' * 13} {'-' * 7} {'-' * 13}")
    for record in arena.generations:
        print(f"  {record['generation']:>4} {record['n_variants']:>9} {record['n_events']:>8,} "
              f"{record['mean_evasion']:>13.3f} {record['max_evasion']:>7.3f} "
              f"{record['mean_fitness']:>13.4f}")
    curve = arena.evasion_curve()
    if len(curve) > 1:
        direction = "FALLING" if curve[-1] < curve[0] else "NOT falling"
        print()
        print(f"  evasion {curve[0]:.3f} -> {curve[-1]:.3f} over {len(curve)} generations: "
              f"{direction}")
    print()
    novel = [i for i in arena.survivors if is_novel(i)]
    unmutated = [i for i in arena.survivors if not is_novel(i)]
    print(f"  survivors (>= {SURVIVAL_ROUNDS} rounds): {len(arena.survivors)} "
          f"({len(novel)} mutated, {len(unmutated)} unmutated)")
    for individual in novel[:6]:
        moved = ", ".join(f"{k}={v:.2f}" for k, v in individual.genome.genes.items())
        print(f"    {individual.genome.label():<16} evasion {individual.evasion:.3f} "
              f"payoff {individual.payoff:.3f}")
        print(f"      {moved}")
    if unmutated:
        print()
        print("  NOTE: the following survived with every gene at its default -- these are the")
        print("  UNMUTATED parent attacks, which the detector is simply bad at. They are a")
        print("  result about those cards, not a discovery, and are NOT written to the atlas:")
        for individual in unmutated:
            print(f"    {individual.genome.card_id:<8} evasion {individual.evasion:.3f} "
                  f"payoff {individual.payoff:.3f}")
    print()

    zero_day = None
    if not args.no_zero_day:
        print("=" * 84)
        print(f"THE ZERO-DAY DEMONSTRATION - family {args.family}")
        print("=" * 84)
        zero_day = run_zero_day(
            background,
            family=args.family,
            generations=max(3, args.generations // 2),
            population=max(4, args.population // 2),
            seed=args.seed,
        )
        _print_zero_day(zero_day, args.family)

    path = write_arena(arena, zero_day, args.out)
    print(f"wrote {path}")

    if not args.no_writeback and novel:
        written = write_discovered_cards(arena.survivors)
        print(f"wrote {len(written)} discovered card(s) to mantis/atlas/discovered/")
        for card_path in written:
            print(f"  {card_path.name}")
    elif not novel:
        print("no MUTATED variant survived the survival bar; nothing written back to the atlas")

    print(f"total {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
