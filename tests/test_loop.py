"""The loop's contract: a mutated variant is still a valid instance of its card.

The loop is load-bearing after the Day 5 reframing — it is half of the
architecture's answer to an unseen attack — so the thing worth testing is not the
evasion curve (that is a measurement, and it belongs in ``arena.json``) but the
invariant underneath it:

    **every genome in the box produces rows the injector framework accepts.**

If that fails, the retrain harness is being fed data no attack could have
generated, and the zero-day comparison is measuring the mutator.

Determinism gets its own test for a specific reason: the Day 1 audit found that
``hash()`` on a string is randomised per process, and both the genome label and
the provenance rebinding are hash-derived. A regression there would silently
change every arena run and every written-back card id.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mantis.foundry.base.reference import load_reference_stats
from mantis.foundry.base.simulator import SimulationConfig, simulate_frame
from mantis.foundry.injectors import REGISTRY, get_injector
from mantis.foundry.injectors.base import PopulationView, run_injector, validate_attack_frame
from mantis.foundry.llm.corpus import load_content_store
from mantis.loop.arena import express
from mantis.loop.genome import (
    GENE_BOUNDS,
    AttackGenome,
    crossover,
    identity_genome,
    mutate,
    random_genome,
)
from mantis.loop.mutate import mutate_rows

SMALL = SimulationConfig(n_events=12_000, seed=7, n_customers=500, n_merchants=1_200)


@pytest.fixture(scope="module")
def view() -> PopulationView:
    load_content_store()
    return PopulationView.build(simulate_frame(SMALL, load_reference_stats()))


@pytest.mark.parametrize("card", ["F1-01", "F1-03", "F4-27", "F6-38"])
def test_every_random_genome_produces_a_valid_attack(view: PopulationView, card: str) -> None:
    """The invariant the retrain harness depends on, sampled across the box."""
    rng = np.random.default_rng(11)
    rows = run_injector(get_injector(card), view, seed=7)
    for _ in range(6):
        genome = random_genome(card, rng)
        mutated = mutate_rows(rows, genome, view, np.random.default_rng(3))
        validate_attack_frame(mutated, card, view.frame)
        assert len(mutated) == len(rows)
        assert (mutated["amount"] >= 0).all()
        assert bool(mutated["is_fraud"].all())


def test_amounts_stay_inside_the_populations_own_band(view: PopulationView) -> None:
    """``amount_scale`` must not win by being absurd."""
    rows = run_injector(get_injector("F6-38"), view, seed=7)
    extreme = AttackGenome(card_id="F6-38", amount_scale=GENE_BOUNDS["amount_scale"][1])
    mutated = mutate_rows(rows, extreme, view, np.random.default_rng(5))
    background_max = float(view.frame["amount"].max())
    assert float(mutated["amount"].max()) <= background_max


def test_provenance_cleaning_is_length_preserving(view: PopulationView) -> None:
    """The Day 3 fix, held.

    Planting was originally length-extending and ``ag_provenance_chain_len``
    became a 0.96 detector. A mutation that shortened the chain would let the
    loop 'evade' by breaking a property of the generator rather than of the
    attack.
    """
    rows = run_injector(get_injector("F1-01"), view, seed=7)
    genome = AttackGenome(card_id="F1-01", provenance_clean=0.9)
    mutated = mutate_rows(rows, genome, view, np.random.default_rng(5))
    before = rows["ag_ingested_content_ids"].map(lambda v: len(v) if v is not None else 0)
    after = mutated["ag_ingested_content_ids"].map(lambda v: len(v) if v is not None else 0)
    assert sorted(before.tolist()) == sorted(after.tolist())


def test_timestamps_stay_inside_the_observation_window(view: PopulationView) -> None:
    rows = run_injector(get_injector("F6-38"), view, seed=7)
    genome = AttackGenome(card_id="F6-38", time_spread=GENE_BOUNDS["time_spread"][1],
                          hour_shift=GENE_BOUNDS["hour_shift"][1])
    mutated = mutate_rows(rows, genome, view, np.random.default_rng(5))
    epoch = mutated["ts"].astype("int64").to_numpy() // 1_000_000_000
    assert epoch.min() >= view.start_epoch
    assert epoch.max() <= view.end_epoch


def test_genome_labels_are_stable_across_processes() -> None:
    """``stable_seed``, not ``hash()``. See the module docstring.

    The literal is pinned rather than compared to a second call in the same
    process, because ``hash()`` is stable *within* a process — a same-process
    comparison would pass on exactly the broken implementation this guards.
    """
    genome = AttackGenome(card_id="F1-01", amount_scale=1.25, time_spread=2.0)
    assert genome.label() == "F1-01~59067"


def test_mutation_stays_in_the_box() -> None:
    rng = np.random.default_rng(2)
    genome = random_genome("F1-01", rng)
    for _ in range(50):
        genome = mutate(genome, rng)
        for gene, (low, high) in GENE_BOUNDS.items():
            assert low <= getattr(genome, gene) <= high


def test_crossover_refuses_to_mix_cards() -> None:
    """A hybrid of two cards implements neither, which is the overclaim the atlas
    registry assertion exists to prevent."""
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError, match="evolves within"):
        crossover(identity_genome("F1-01"), identity_genome("F6-38"), rng)


def test_express_is_deterministic(view: PopulationView) -> None:
    genome = random_genome("F1-01", np.random.default_rng(4))
    first = express(genome, view, seed=7)
    second = express(genome, view, seed=7)
    pd.testing.assert_frame_equal(
        first.reset_index(drop=True), second.reset_index(drop=True)
    )


def test_the_identity_genome_changes_nothing_material(view: PopulationView) -> None:
    """The arena's own reference row has to be the unmutated attack."""
    card = next(iter(sorted(REGISTRY)))
    rows = run_injector(get_injector(card), view, seed=7).sort_values("ts").reset_index(drop=True)
    same = mutate_rows(rows, identity_genome(card), view, np.random.default_rng(1))
    assert np.allclose(rows["amount"].to_numpy(), same["amount"].to_numpy())
    assert (rows["ts"].to_numpy() == same["ts"].to_numpy()).all()
    assert (rows["merchant_id"].to_numpy() == same["merchant_id"].to_numpy()).all()
