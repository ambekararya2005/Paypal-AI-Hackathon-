"""An attack as a parameter vector, and how to mutate one.

What is being evolved
-----------------------
Not the injector. The injectors are the atlas made executable and they encode
what each attack *is*; evolving their source would mean the thing the loop
produces is no longer the card it claims to implement, which is the overclaim the
registry assertion exists to prevent.

What is evolved is a **genome of post-hoc transformations** applied to the rows
an injector emits — the operational choices a real operator tunes when their
attempts start getting declined. Spread the ring over more days. Split one
campaign into four. Rotate devices. Send the agent through one injected page
instead of three. Move less money per event. None of these change what the attack
*is*; all of them change what it *looks like*, which is exactly the space an
adaptive adversary searches.

Every gene is bounded, and the bounds are the point
-----------------------------------------------------
An unbounded search finds the degenerate optimum immediately: move ₹0 and get
caught never. So every gene has a range chosen to keep the variant a recognisable
instance of its card, and the fitness function prices volume explicitly (see
:mod:`mantis.loop.arena`). A variant that evades by not attacking scores zero.

The genes, and what each one is trying to defeat
--------------------------------------------------
``amount_scale``
    Ticket size. Trades payoff against ``ent_amount_vs_customer_p99`` and every
    amount z-score.
``time_spread``
    Stretches the intra-campaign gaps. Directly attacks the velocity windows:
    a burst spread over four times as long falls out of ``*_count_1h``.
``campaign_fanout``
    Splits one ring into several. Attacks per-campaign structure and the graph's
    component features — the classic response to a network being rolled up.
``merchant_spread``
    Re-draws a share of the merchants from the background's popular ones.
    Attacks ``gph_merchant_fanin_per_component`` and merchant concentration.
``device_rotate``
    Re-draws a share of the devices. Attacks the identity graph directly: it
    breaks the shared-device edges that make a component big.
``hour_shift``
    Moves the campaign around the clock.
``provenance_clean``
    Share of injected pages swapped back to benign content. **This is the gene
    aimed at L3**, and it is the interesting one: it is the only gene whose
    optimum is obviously "1.0, evade completely" — except that a cart-tampering
    attack with no injected page is not cart tampering, so the fitness function
    prices it as a loss of payoff rather than letting it be free.
``delegation_delta``
    Shifts the delegation depth. Attacks L0's depth clause and F1-05's signal.
``deliberation_scale``
    Stretches the agent's deliberation latency. Aimed squarely at
    ``mnd_deliberation_residual_z``, the feature the Day 4 review found separating
    F1-01 at 0.99 — the loop should discover that one on its own, and whether it
    does is a check on whether the loop works at all.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Final

import numpy as np

from mantis.foundry.injectors.base import stable_seed

__all__ = ["GENE_BOUNDS", "AttackGenome", "crossover", "mutate", "random_genome"]

#: ``gene -> (low, high)``. Sampled uniformly at generation zero and clipped
#: after every mutation, so no genome ever leaves the box.
GENE_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "amount_scale": (0.35, 1.80),
    "time_spread": (0.25, 6.00),
    "campaign_fanout": (1.00, 4.00),
    "merchant_spread": (0.00, 0.90),
    "device_rotate": (0.00, 0.90),
    "hour_shift": (-8.00, 8.00),
    "provenance_clean": (0.00, 0.90),
    "delegation_delta": (-1.00, 3.00),
    "deliberation_scale": (0.40, 4.00),
}

#: Per-gene standard deviation of a mutation, as a share of the gene's range.
_MUTATION_SCALE: Final[float] = 0.18

#: Probability an individual gene is touched by a mutation. Below one so that a
#: child stays recognisably its parent — a mutation operator that rewrites every
#: gene is a random restart wearing an evolutionary hat.
_MUTATION_RATE: Final[float] = 0.5


@dataclass(frozen=True, slots=True)
class AttackGenome:
    """The evolvable parameters of one attack variant.

    ``card_id`` is not a gene. A variant is always a variant *of a card*, and the
    loop evolves within a card rather than across the atlas — mutating F1-01 into
    F6-38 would produce something the atlas has no description of.
    """

    card_id: str
    amount_scale: float = 1.0
    time_spread: float = 1.0
    campaign_fanout: float = 1.0
    merchant_spread: float = 0.0
    device_rotate: float = 0.0
    hour_shift: float = 0.0
    provenance_clean: float = 0.0
    delegation_delta: float = 0.0
    deliberation_scale: float = 1.0

    @property
    def genes(self) -> dict[str, float]:
        """Just the evolvable numbers, without ``card_id``."""
        return {k: v for k, v in asdict(self).items() if k in GENE_BOUNDS}

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def label(self) -> str:
        """Short, stable name for the arena log and the atlas write-back.

        Hashed with :func:`~mantis.foundry.injectors.base.stable_seed` and not
        with ``hash()``. CPython randomises string hashing per process, which is
        the exact defect the Day 1 audit found in the base simulator: a label
        derived from ``hash()`` would change between runs, and every seed,
        arena.json entry and written-back card id derived from it would change
        with it.
        """
        digest = "".join(f"{round(v * 100):+05d}" for v in self.genes.values())
        return f"{self.card_id}~{stable_seed(digest) % 100_000:05d}"

    def distance(self, other: AttackGenome) -> float:
        """Normalised L2 distance in gene space, for reporting drift over generations."""
        total = 0.0
        for gene, (low, high) in GENE_BOUNDS.items():
            span = high - low
            total += ((getattr(self, gene) - getattr(other, gene)) / span) ** 2
        return float(np.sqrt(total / len(GENE_BOUNDS)))


def _clip(gene: str, value: float) -> float:
    low, high = GENE_BOUNDS[gene]
    return float(min(max(value, low), high))


def random_genome(card_id: str, rng: np.random.Generator) -> AttackGenome:
    """A uniformly-sampled point in the box. Generation zero's population."""
    return AttackGenome(
        card_id=card_id,
        **{gene: float(rng.uniform(low, high)) for gene, (low, high) in GENE_BOUNDS.items()},
    )


def identity_genome(card_id: str) -> AttackGenome:
    """The unmutated attack, for the baseline row of every arena report."""
    return AttackGenome(card_id=card_id)


def mutate(genome: AttackGenome, rng: np.random.Generator) -> AttackGenome:
    """Gaussian jitter on a random subset of genes, clipped back into the box."""
    updates: dict[str, float] = {}
    for gene, (low, high) in GENE_BOUNDS.items():
        if rng.random() > _MUTATION_RATE:
            continue
        step = rng.normal(0.0, _MUTATION_SCALE * (high - low))
        updates[gene] = _clip(gene, getattr(genome, gene) + step)
    return replace(genome, **updates)


def crossover(
    left: AttackGenome, right: AttackGenome, rng: np.random.Generator
) -> AttackGenome:
    """Uniform crossover, gene by gene.

    Uniform rather than single-point because the genes are independent knobs, not
    a sequence: there is no locality for a crossover point to respect, and
    single-point would create a spurious correlation between adjacent fields of a
    dataclass, which is an artefact of declaration order and nothing else.
    """
    if left.card_id != right.card_id:
        raise ValueError(
            f"cannot cross {left.card_id} with {right.card_id}: the loop evolves within "
            "a card, because a hybrid of two cards implements neither"
        )
    return replace(
        left,
        **{
            gene: float(getattr(left if rng.random() < 0.5 else right, gene))
            for gene in GENE_BOUNDS
        },
    )
