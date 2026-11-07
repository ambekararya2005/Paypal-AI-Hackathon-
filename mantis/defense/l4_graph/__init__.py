"""L4 — the entity graph, plus the entity-level novelty experiment.

    from mantis.defense.l4_graph import graph_features
    X = graph_features(events_sorted_by_ts)      # streamed, backward-looking

:mod:`mantis.defense.l4_graph.graph` is in the scoring path: its ``gph_*``
columns are part of the feature matrix L1 trains on.
:mod:`mantis.defense.l4_graph.entity_novelty` is not — it is the time-boxed test
of whether L2 does better at entity level than at event level, and it is
reported whichever way it comes out.
"""

from __future__ import annotations

from mantis.defense.l4_graph.entity_novelty import ENTITY_KEYS, EntityNovelty, entity_vectors
from mantis.defense.l4_graph.graph import (
    GRAPH_FEATURE_NAMES,
    EntityGraph,
    build_networkx,
    graph_features,
)

__all__ = [
    "ENTITY_KEYS",
    "GRAPH_FEATURE_NAMES",
    "EntityGraph",
    "EntityNovelty",
    "build_networkx",
    "entity_vectors",
    "graph_features",
]
