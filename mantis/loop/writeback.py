"""Writing a surviving variant back into the atlas.

Why the discovered cards live in their own directory
------------------------------------------------------
``mantis/atlas/cards/`` is the frozen 42. ``tests/test_atlas.py`` pins the family
counts and the implemented/mapped split precisely so that the number in the
writeup cannot drift away from the repo, and CLAUDE.md calls the implemented
count a **ratchet** that moves only when code lands. Dropping loop-generated
cards into that directory would move both numbers without any code landing, which
is exactly the overclaim the ratchet exists to prevent.

So they go to ``mantis/atlas/discovered/``, validated by the **same** pydantic
model, loadable by the same loader, and reported as their own line. The atlas
summary then reads: *42 human-authored cards, N discovered by the adversarial
loop*, which is a stronger claim than "43 cards" because a reader can see which
is which.

Why they are ``status: mapped``
---------------------------------
An ``implemented`` card must name a ``generator`` that resolves to a callable in
its own injector's module, and the registry assertion checks that at import. A
discovered variant has no module of its own — it is its parent's injector plus a
genome — so claiming ``implemented`` would either break that assertion or require
weakening it. ``mapped`` is the honest status.

The genome is written to a **sidecar** ``<id>.genome.json`` rather than into the
card, because :class:`~mantis.atlas.schema.AttackCard` forbids extra keys and
that is a rule worth keeping: a card that can carry arbitrary fields is a card
whose schema stops meaning anything. The sidecar is what makes the variant
reproducible rather than merely described —
``express(AttackGenome(**json.load(open(sidecar))["genome"]), view)`` regenerates
it exactly — and the loader globs ``*.yaml``, so it never sees the sidecar.

What the card actually says
-----------------------------
The variant inherits its parent's family, rails, actor, mitigations and
observable signals, because those are properties of the attack *class* and the
loop did not change the class — it changed how the attack is operated. What is
rewritten is the **description**: which genes moved, in which direction, and what
that means for an operator.

One thing the card deliberately does **not** claim is which of the parent's
signals the variant defeated. That would be the most useful field on it, and the
arena does not measure it: fitness is scored against the fused detector, not
per-signal, so any per-signal claim here would be inferred rather than measured.
Inferring it and writing it down as though it were measured is precisely the kind
of quiet overclaim the atlas's status ratchet exists to prevent. Per-signal
attribution is a Day 7 item.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import yaml

from mantis.atlas.loader import ATLAS
from mantis.atlas.schema import AttackCard, DiscoveredBy, Status
from mantis.loop.arena import Individual
from mantis.loop.genome import GENE_BOUNDS

__all__ = ["DISCOVERED_DIR", "describe_genome", "is_novel", "write_discovered_cards"]

#: Where discovered cards are written. Beside the atlas, not inside it.
DISCOVERED_DIR: Final[Path] = Path(__file__).resolve().parents[1] / "atlas" / "discovered"

#: Discovered ids start here, so they can never collide with a human-authored
#: card even if the atlas grows to fifty per family.
_ID_BASE: Final[int] = 50

#: How far a gene has to move from its default before the description mentions
#: it. Below this the mutation is noise and naming it would pad the card.
_NOTABLE: Final[float] = 0.20


def is_novel(individual: Individual) -> bool:
    """Whether this variant is actually a variant, rather than its own parent.

    The arena seeds every card's population with an **identity genome** — the
    unmutated attack — so that the evasion curve carries its own "what does this
    attack do without evolution" reference row. That individual competes like any
    other, and on cards the detector is already bad at it wins: the first Day 5
    run put ``F2-16`` and ``F1-05`` at the top of the survivor list with every
    gene at its default, because the unmutated bust-out evades 78% of decisions
    on its own.

    That is a real and reportable result — it says the parent card is the problem,
    not that the loop found something. Writing it into the atlas as an "evasive
    variant" would be claiming a discovery for an attack that was already in
    ``cards/``, which is exactly the overclaim the status ratchet exists to
    prevent. So the write-back skips it, and the CLI reports the count separately.
    """
    return bool(describe_genome(individual))


def describe_genome(individual: Individual) -> list[str]:
    """Plain-language phrases for the genes that actually moved."""
    genome = individual.genome
    phrases: list[str] = []
    for gene, (low, high) in GENE_BOUNDS.items():
        value = float(getattr(genome, gene))
        default = float(getattr(type(genome)(card_id=genome.card_id), gene))
        if abs(value - default) / (high - low) < _NOTABLE:
            continue
        phrases.append(_PHRASES[gene](value))
    return phrases


#: One sentence per gene, written from the operator's point of view rather than
#: the parameter's — a card that says "time_spread=4.2" is a log line, not an
#: attack description.
_PHRASES: Final[dict[str, object]] = {
    "amount_scale": lambda v: (
        f"ticket size {'reduced' if v < 1 else 'raised'} to {v:.2f}x the parent attack's, "
        "trading payoff per event against the per-customer amount baselines"
    ),
    "time_spread": lambda v: (
        f"campaign paced {v:.1f}x {'slower' if v > 1 else 'faster'}, moving the ring's activity "
        "out of the one-hour and one-day velocity windows"
    ),
    "campaign_fanout": lambda v: (
        f"the ring is split into roughly {round(v)} sub-rings, so no single campaign carries "
        "enough volume to be worth an alert"
    ),
    "merchant_spread": lambda v: (
        f"about {v:.0%} of the merchant legs are re-drawn from ordinary popular merchants, "
        "diluting beneficiary concentration"
    ),
    "device_rotate": lambda v: (
        f"about {v:.0%} of events are moved onto rotated devices, breaking the shared-device "
        "edges that make an identity component visible"
    ),
    "hour_shift": lambda v: (
        f"activity shifted {abs(v):.1f} hours {'later' if v > 0 else 'earlier'} in the day"
    ),
    "provenance_clean": lambda v: (
        f"about {v:.0%} of the injected pages are replaced with innocuous content, so the "
        "provenance trail carries less instruction-shaped text for a reader to find"
    ),
    "delegation_delta": lambda v: (
        f"delegation chain {'lengthened' if v > 0 else 'shortened'} by {abs(v):.0f}"
    ),
    "deliberation_scale": lambda v: (
        f"agent deliberation latency {'stretched' if v > 1 else 'compressed'} to {v:.1f}x, "
        "which is a direct answer to the deliberation-residual feature"
    ),
}


def _next_ids(family: str, count: int, existing: set[str]) -> list[str]:
    out: list[str] = []
    number = _ID_BASE
    while len(out) < count:
        candidate = f"{family}-{number:02d}"
        if candidate not in existing:
            out.append(candidate)
            existing.add(candidate)
        number += 1
        if number > 99:
            raise ValueError(f"no free card ids left in family {family}")
    return out


def _card_for(individual: Individual, card_id: str) -> AttackCard:
    parent = ATLAS[individual.genome.card_id]
    phrases = describe_genome(individual)
    changes = "; ".join(phrases) if phrases else "no gene moved materially from the parent"

    return AttackCard(
        id=card_id,
        name=f"{parent.name} — evasive variant {individual.genome.label().split('~')[-1]}",
        family=parent.family,
        status=Status.MAPPED,
        rails=list(parent.rails),
        actor=parent.actor,
        genai_enabler=(
            "The variant was not written by a person. It was found by MANTIS's own "
            "evolutionary adversary, which mutates the operational parameters of a known "
            "attack and selects on evasion x payoff against the live detector — the same "
            "loop an operator runs when their attempts start getting declined, except that "
            "here the defender runs it first. "
        )
        + parent.genai_enabler,
        description=(
            f"A variant of {parent.id} discovered by the adversarial loop. It evaded "
            f"{individual.evasion:.0%} of the detector's decisions at the 0.1% "
            f"false-positive operating point while retaining {individual.payoff:.2f}x the "
            f"parent attack's realised payoff, and survived {individual.survived} "
            f"consecutive rounds of retraining. What changed: {changes}. The underlying "
            f"attack is unchanged — see {parent.id} for what it is and why it works."
        ),
        preconditions=list(parent.preconditions),
        observable_signals=list(parent.observable_signals),
        mitigations=list(parent.mitigations),
        generator=None,
        detected_by=list(parent.detected_by),
        references=[f"parent card: {parent.id}", "discovered by mantis.loop.arena"],
        discovered_by=DiscoveredBy.ADVERSARIAL_LOOP,
    )


def write_discovered_cards(
    survivors: list[Individual], directory: Path = DISCOVERED_DIR
) -> list[Path]:
    """Write one YAML card per surviving variant. Returns the card paths written.

    Survivors whose genes never moved are skipped — see :func:`is_novel`. Each
    card gets a ``<id>.genome.json`` sidecar carrying the genome and the metrics
    that earned it a card; see the module docstring for why those do not live
    inside the card itself.
    """
    directory.mkdir(parents=True, exist_ok=True)
    existing = set(ATLAS) | {p.stem for p in directory.glob("*.yaml")}
    written: list[Path] = []

    by_family: dict[str, list[Individual]] = {}
    for individual in survivors:
        if not is_novel(individual):
            continue
        family = ATLAS[individual.genome.card_id].family.value
        by_family.setdefault(family, []).append(individual)

    for family, members in sorted(by_family.items()):
        for card_id, individual in zip(
            _next_ids(family, len(members), existing), members, strict=True
        ):
            card = _card_for(individual, card_id)
            path = directory / f"{card_id}.yaml"
            path.write_text(
                yaml.safe_dump(
                    card.model_dump(mode="json"),
                    sort_keys=False,
                    allow_unicode=True,
                    width=100,
                ),
                encoding="utf-8",
            )
            (directory / f"{card_id}.genome.json").write_text(
                json.dumps(
                    {
                        "card_id": card_id,
                        "parent_card_id": individual.genome.card_id,
                        "label": individual.genome.label(),
                        "genome": individual.genome.to_json(),
                        "metrics": {
                            "evasion": round(individual.evasion, 4),
                            "payoff": round(individual.payoff, 4),
                            "fitness": round(individual.fitness, 4),
                            "survived_rounds": individual.survived,
                            "n_events": individual.n_events,
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            written.append(path)
    return written
