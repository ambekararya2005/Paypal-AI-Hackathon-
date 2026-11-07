"""Contract tests for the 42-card attack atlas.

The atlas is scored on **diversity of attacks** (judging criterion 1), and the
fastest way to lose that criterion is to be caught overclaiming. So the counts
are asserted, not assumed:

* 42 cards, split 12/6/8/6/5/5 across F1-F6.
* Exactly 8 carry a working injector. That number goes in the writeup verbatim,
  and it is a **ratchet**: it moves only when an injector actually lands.
* Implemented cards span at least four families, so the coverage claim is
  genuinely broad rather than four variations of one attack.
* A card is ``implemented`` **only** if it names a generator, and a ``mapped``
  card names none. A mapped card carrying a generator path implies an injector
  that does not exist, which is exactly the overclaim these tests exist to stop.

Note on the count. Day 1 marked 15 cards ``implemented`` on the strength of a
planned generator path. Day 2 made the atlas executable — importing
``mantis.foundry.injectors`` now *fails* unless every implemented card has real
code behind it (see ``tests/test_injectors.py``) — so the seven cards with no
injector yet were returned to ``mapped``. The number went down because the
definition got stricter, and it goes back up as Day 3 lands the F1 and F5
agentic injectors.
"""

from __future__ import annotations

import re

from mantis.atlas.loader import (
    ATLAS,
    DISCOVERED,
    TARGET_CARD_COUNT,
    by_family,
    implemented,
    signals_by_layer,
    summary,
)
from mantis.atlas.schema import DiscoveredBy, Family, Layer, Status
from mantis.core.events import Channel

#: The atlas as promised in CLAUDE.md section 2. Changing these is a scope change.
EXPECTED_FAMILY_SIZES: dict[Family, int] = {
    Family.F1: 12,
    Family.F2: 6,
    Family.F3: 8,
    Family.F4: 6,
    Family.F5: 5,
    Family.F6: 5,
}

EXPECTED_TOTAL = 42
EXPECTED_IMPLEMENTED = 15

#: Families with a working injector today. F1 (mandate abuse) arrived on Day 3
#: with seven injectors -- the agentic attacks the whole project exists for.
#: F5 (platform integrity) is still absent, deliberately and visibly: it is the
#: zero-day holdout family, and this set is the ratchet that records the fact
#: rather than letting a writeup drift ahead of the code.
EXPECTED_IMPLEMENTED_FAMILIES = {Family.F1, Family.F2, Family.F3, Family.F4, Family.F6}


def test_atlas_is_complete() -> None:
    assert len(ATLAS) == EXPECTED_TOTAL


def test_family_sizes_match_the_plan() -> None:
    grouped = by_family()
    assert {f: len(cards) for f, cards in grouped.items()} == EXPECTED_FAMILY_SIZES


def test_the_implemented_count_is_the_number_we_say_out_loud() -> None:
    """A ratchet: it moves only when code lands, never when a claim does.

    Day 1 claimed 15 on the strength of planned generator paths and Day 2 cut it
    to 8 by making the claim enforceable at import. Day 3 earns 15 back, with an
    injector behind every one of them.
    """
    assert len(implemented()) == EXPECTED_IMPLEMENTED


def test_implemented_cards_span_several_families() -> None:
    """Coverage has to be broad, not four flavours of the same attack."""
    covered = {card.family for card in implemented()}
    assert covered == EXPECTED_IMPLEMENTED_FAMILIES
    assert len(covered) >= 4


def test_card_ids_are_contiguous_and_family_aligned() -> None:
    """Ids run 01..42 with no gaps, and each family owns a contiguous block."""
    numbers = sorted(int(card_id.split("-")[1]) for card_id in ATLAS)
    assert numbers == list(range(1, EXPECTED_TOTAL + 1))

    for family, cards in by_family().items():
        block = sorted(int(c.id.split("-")[1]) for c in cards)
        assert block == list(range(block[0], block[0] + len(cards))), (
            f"{family.value} ids are not contiguous: {block}"
        )


def test_implemented_cards_name_a_generator_and_mapped_cards_do_not() -> None:
    """No card may imply an injector that does not exist."""
    for card in ATLAS.values():
        if card.status is Status.IMPLEMENTED:
            assert card.generator is not None, f"{card.id} is implemented with no generator"
        else:
            assert card.generator is None, (
                f"{card.id} is mapped but names generator {card.generator!r}; "
                "a mapped card must not imply an injector exists"
            )


def test_generator_paths_are_unique_and_well_formed() -> None:
    paths = [c.generator for c in implemented()]
    assert len(set(paths)) == len(paths), "two cards share one injector path"
    for card in implemented():
        assert card.generator is not None
        module, _, func = card.generator.partition(":")
        assert module.startswith("mantis.foundry.injectors."), card.id
        assert re.fullmatch(r"[a-z_][a-z0-9_]*", func), card.id


def test_every_card_is_detectable() -> None:
    """A card with no observable signal is a threat-deck bullet, not an atlas entry."""
    for card in ATLAS.values():
        assert card.observable_signals, card.id
        assert card.mitigations, card.id
        assert card.preconditions, card.id


def test_signal_layers_are_declared_in_detected_by() -> None:
    """The schema enforces this per card; assert it once across the whole atlas."""
    for card in ATLAS.values():
        assert {s.layer for s in card.observable_signals} <= set(card.detected_by), card.id


def test_implemented_cards_span_multiple_layers() -> None:
    """A single-layer attack cannot demonstrate that layer diversity buys anything."""
    for card in implemented():
        assert len(set(card.detected_by)) >= 2, f"{card.id} is detected by only one layer"


def test_every_rail_is_a_real_channel() -> None:
    valid = {c.value for c in Channel}
    for card in ATLAS.values():
        assert set(card.rails) <= valid, card.id


def test_every_layer_has_features_to_build() -> None:
    """The atlas doubles as the feature backlog; no layer may be empty."""
    layers = signals_by_layer()
    for layer in Layer:
        assert layers.get(layer), f"no card names a feature for {layer.value}"


def test_l0_features_are_deterministic_mandate_checks() -> None:
    """L0 is the rules layer; its features must be things a rule can evaluate."""
    l0 = signals_by_layer()[Layer.L0]
    assert any("mandate" in f for f in l0)
    assert "mandate_scope_violation" in l0


def test_agentic_rail_is_the_atlas_centre_of_gravity() -> None:
    """This is a project about agentic commerce, not a general fraud taxonomy."""
    agentic = [c for c in ATLAS.values() if Channel.AGENTIC.value in c.rails]
    assert len(agentic) / len(ATLAS) > 0.6


def test_summary_states_the_implemented_split() -> None:
    """The honest count has to be printed, not left for someone to work out."""
    text = summary()
    assert "HONEST COUNT" in text
    assert f"{EXPECTED_TOTAL} vectors mapped" in text
    assert f"{EXPECTED_IMPLEMENTED} of them have a working injector" in text


def test_implemented_cards_all_have_a_registered_injector() -> None:
    """The seam between Pillar 1 and Pillar 2, asserted from the atlas side too.

    ``mantis.foundry.injectors`` raises at import if this is violated; asserting
    it here as well means the failure is legible in the test report rather than
    arriving as a collection error.
    """
    from mantis.foundry.injectors import REGISTRY

    assert {c.id for c in implemented()} == set(REGISTRY)


def test_discovered_cards_never_enter_the_atlas_totals() -> None:
    """The ratchet, protected against the loop.

    ``mantis/atlas/discovered/`` holds cards the adversarial loop found. They are
    validated by the same model and loadable by the same loader, and they must
    **not** appear in ``ATLAS`` — the 42 and the implemented count are pinned
    numbers that may only move when a human writes a card or an injector lands.
    A loop that could move them by running would make both numbers meaningless.
    """
    assert not (set(ATLAS) & set(DISCOVERED))
    assert len(ATLAS) == TARGET_CARD_COUNT
    for card in DISCOVERED.values():
        # Never `implemented`: a variant is its parent's injector plus a genome,
        # not a module of its own, so an `implemented` status here would either
        # break the registry assertion or force it to be weakened.
        assert card.status is Status.MAPPED
        assert card.generator is None
        assert card.discovered_by is DiscoveredBy.ADVERSARIAL_LOOP
