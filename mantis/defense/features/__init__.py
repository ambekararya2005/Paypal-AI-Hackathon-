"""Feature construction for the Mandate Firewall.

    from mantis.defense.features import FeatureBuilder
    builder = FeatureBuilder().fit(train)
    X = builder.transform(test)

Four groups, in increasing order of what they need to know:

``transaction``
    Readable off one authorisation message. No history, no fitted state.
``velocity``
    Counts, sums and **decline ratios** over 1h/24h/7d per customer, card, BIN,
    device, merchant, agent, IP and mandate hash. Computed by a single forward
    pass over a keyed state store, not by a groupby — see
    :mod:`mantis.defense.features.state` for why that distinction decides
    whether Day 5's latency budget is reachable.
``entity``
    How this event compares to what its customer, merchant, category and BIN
    normally do. Fitted on train only.
``mandate``
    The AP2 block: scope divergence, mandate age against TTL, delegation depth,
    provenance length, and the human-presence mismatch.

The one thing to know before using this package: three tiers of column are
forbidden from the matrix, not one. Ground truth, post-hoc dispute state, **and
the current event's own outcome**. See :mod:`mantis.defense.features.spec`.
"""

from __future__ import annotations

from mantis.defense.features.builder import FeatureBuilder, LeakageError
from mantis.defense.features.entity import EntityProfiles
from mantis.defense.features.mandate import MandateBaselines
from mantis.defense.features.spec import (
    FORBIDDEN_COLUMNS,
    FUTURE_COLUMNS,
    VELOCITY_KEYS,
    FeatureConfig,
)
from mantis.defense.features.state import WINDOWS, RollingStore, WindowSpec

__all__ = [
    "FORBIDDEN_COLUMNS",
    "FUTURE_COLUMNS",
    "VELOCITY_KEYS",
    "WINDOWS",
    "EntityProfiles",
    "FeatureBuilder",
    "FeatureConfig",
    "LeakageError",
    "MandateBaselines",
    "RollingStore",
    "WindowSpec",
]
