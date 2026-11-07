"""The closed loop: an evolutionary adversary and the retraining harness.

    python -m mantis.loop

After the Day 5 reframing this package is not a flourish, it is **half of the
architecture's answer to an unseen attack** (CLAUDE.md, "The zero-day answer,
reframed"). L2 does not generalise to attacks it never saw and no longer claims
to; what generalises is manufacturing the attack first.

    from mantis.loop import run_arena, run_zero_day
    arena = run_arena(pool, generations=6)
    zero_day = run_zero_day(pool, family="F1")

:mod:`mantis.loop.genome` is what evolves, :mod:`mantis.loop.mutate` is how a
genome is applied to an injector's output without breaking the schema or the
foundry's realism guarantees, and :mod:`mantis.loop.arena` is the two
experiments.
"""

from __future__ import annotations

from mantis.loop.arena import (
    ARENA_JSON,
    SURVIVAL_ROUNDS,
    ArenaResult,
    Individual,
    ZeroDayResult,
    run_arena,
    run_zero_day,
    write_arena,
)
from mantis.loop.genome import GENE_BOUNDS, AttackGenome, crossover, mutate, random_genome
from mantis.loop.mutate import mutate_rows

__all__ = [
    "ARENA_JSON",
    "GENE_BOUNDS",
    "SURVIVAL_ROUNDS",
    "ArenaResult",
    "AttackGenome",
    "Individual",
    "ZeroDayResult",
    "crossover",
    "mutate",
    "mutate_rows",
    "random_genome",
    "run_arena",
    "run_zero_day",
    "write_arena",
]
