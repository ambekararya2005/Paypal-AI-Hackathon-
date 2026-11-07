"""The feature contract: what may reach a model, and what may never.

Three tiers of forbidden, not one
----------------------------------
CLAUDE.md HARD RULE 1 says label columns must never reach a feature matrix.
That is necessary and it is not sufficient, because there are two other ways a
column can be a lie:

``LABEL_COLUMNS``
    ``is_fraud``, ``attack_id``, ``attack_campaign``. Ground truth. A model that
    sees these posts 1.000 and means nothing.

``POST_HOC_COLUMNS``
    ``dispute_outcome``, ``dispute_raised_ts``. Resolve weeks to months after the
    authorisation. Using them at scoring time is reading the future.

``FUTURE_COLUMNS`` — **the tier this module adds**
    ``auth_response``, ``settled``, ``settlement_lag_hours``, of the event being
    scored. These are not labels and not post-hoc; they are the issuer's own
    decision on *this* authorisation and its clearing outcome, and at the moment
    the firewall runs, none of them exists yet. The authorisation is being scored
    in order to decide the first of them.

    This tier is easy to get wrong because the columns are right there in the
    parquet, they are not named anything suspicious, and including them makes
    every metric better. F4-27 is card testing: its signature is a decline. Feed
    the current row's ``auth_response`` to L1 and F4-27's recall jumps, and the
    number is worthless — an issuer cannot decline a transaction because it was
    declined.

    The same three columns on **earlier** events are entirely legitimate and are
    the single most valuable thing amendment 1.1.0 added. That is what the
    velocity layer is for: ``vel_bin_decline_ratio_1h`` reads prior outcomes on
    the same BIN, which the issuer genuinely holds. The asymmetry is enforced in
    :mod:`mantis.defense.features.state` — outcomes are folded into state *after*
    the current event's features are read off it.

An escape hatch, deliberately explicit
---------------------------------------
:data:`FUTURE_COLUMNS` is enforced by default and can be turned off with
``FeatureConfig(include_current_outcome=True)``. That is not a loophole, it is
an acknowledgement that a *post-authorisation* model — one scoring for
chargeback risk after the fact, or a batch review queue — genuinely does hold
the response code. If MANTIS ever grows one, the flag makes the choice a visible
line of configuration rather than an accident. Every number in RESULTS.md is
produced with it off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from mantis.core.events import LABEL_COLUMNS, POST_HOC_COLUMNS
from mantis.defense.features.state import WindowSpec

__all__ = [
    "FORBIDDEN_COLUMNS",
    "FUTURE_COLUMNS",
    "VELOCITY_KEYS",
    "FeatureConfig",
]

#: The issuer's own decision on *this* message, and its clearing outcome. Not
#: known at scoring time. See the module docstring — this is the tier that is
#: easy to leak by accident.
FUTURE_COLUMNS: Final[tuple[str, ...]] = (
    "auth_response",
    "settled",
    "settlement_lag_hours",
)

#: Everything that must never appear as a feature name, in one tuple, so the
#: builder's assertion can be a single containment check.
FORBIDDEN_COLUMNS: Final[tuple[str, ...]] = (
    *LABEL_COLUMNS,
    *POST_HOC_COLUMNS,
    *FUTURE_COLUMNS,
)

#: The entities velocity is measured over.
#:
#: ``card`` is ``(customer, bin)`` rather than a card number, because the schema
#: carries a BIN and not a PAN — which is also what an issuer's own fraud system
#: keys on when it is looking across a compromised range.
#:
#: ``bin`` on its own is the card-testing key: F4-27 spreads a finite set of
#: stolen credentials thinly across a BIN precisely so that no single *card*
#: shows velocity. Per-card velocity is useless against it and per-BIN velocity
#: is what works, which is why both are here.
#:
#: ``mandate_hash`` is the replay key. F1-10 re-presents a mandate that was
#: legitimately signed once; the only thing that distinguishes the second
#: presentation from the first is that the network has seen the digest before.
#: A count over this key *is* the replay detector.
VELOCITY_KEYS: Final[tuple[WindowSpec, ...]] = (
    WindowSpec("customer", ("customer_id",)),
    WindowSpec("card", ("customer_id", "card_bin")),
    WindowSpec("bin", ("card_bin",)),
    WindowSpec("device", ("device_id",)),
    WindowSpec("merchant", ("merchant_id",)),
    WindowSpec("agent", ("ag_agent_id",)),
    WindowSpec("mandate_hash", ("ag_mandate_hash",)),
    WindowSpec("ip", ("ip",)),
)


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Knobs the feature builder exposes, all of them defaulted to the safe side."""

    #: Let the current event's own outcome columns become features. Default off;
    #: see the module docstring for the only case that should turn it on.
    include_current_outcome: bool = False

    #: Cap on distinct levels a categorical is allowed to carry. Above this it is
    #: an identifier, and a tree splitting on it memorises entities rather than
    #: learning behaviour. Identifiers reach the model through their *velocity*,
    #: never through their value.
    max_categorical_levels: int = 64

    #: Domains treated as attacker infrastructure by the provenance features.
    #: Empty means the feature is computed against the fitted allow-list instead.
    trusted_domain_quantile: float = 0.98

    #: Include the L4 entity-graph block (``gph_*``). On by default from Day 5:
    #: F2 and F4 are entity-level attacks and a per-event matrix cannot see them.
    #: The flag exists so an ablation can turn the whole layer off by name rather
    #: than by deleting columns.
    include_graph: bool = True
