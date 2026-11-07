"""The graph layer's contract, tested where it can actually break.

The graph is the one Day 5 layer that sits **in the scoring path**, so the checks
that matter are the same two that matter for velocity, plus one that is specific
to graphs:

1. **The pass is backward-looking.** Every count an event reads must exclude that
   event. This is the read-then-fold contract, and getting it backwards is the
   quiet kind of leakage: nothing is misnamed, the numbers just contain the
   present.
2. **The components are not one giant blob.** Every component feature is
   worthless if a single union through a hub fuses the population. The test pins
   the property the design rests on rather than the implementation that achieves
   it.
3. **The layer is deterministic.** Two passes over the same frame must produce
   the same matrix — the same discipline the Day 1 audit had to impose on the
   simulator after ``hash()`` randomisation broke reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mantis.defense.l4_graph import (
    GRAPH_FEATURE_NAMES,
    EntityGraph,
    EntityNovelty,
    entity_vectors,
    graph_features,
)
from mantis.foundry.base.reference import load_reference_stats
from mantis.foundry.base.simulator import SimulationConfig, simulate_frame
from mantis.foundry.injectors import REGISTRY
from mantis.foundry.injectors.base import PopulationView, run_injector

SMALL = SimulationConfig(n_events=20_000, seed=7, n_customers=800, n_merchants=2_000)


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    background = simulate_frame(SMALL, load_reference_stats())
    view = PopulationView.build(background)
    attacks = [run_injector(REGISTRY[c], view, seed=7) for c in ("F6-38", "F2-13")]
    frame = pd.concat([background, *attacks], ignore_index=True)
    return frame.sort_values("ts", kind="stable").reset_index(drop=True)


def test_every_declared_feature_is_emitted(dataset: pd.DataFrame) -> None:
    """The name list is data, so a dropped feature is a failure and not a surprise."""
    matrix = graph_features(dataset)
    assert list(matrix.columns) == list(GRAPH_FEATURE_NAMES)
    assert len(matrix) == len(dataset)


def test_first_event_of_an_entity_sees_no_history(dataset: pd.DataFrame) -> None:
    """The read-then-fold contract, checked on the only rows where it is unambiguous.

    An entity's very first event must read zero for every count over that
    entity. If ``observe`` folded before it read, this would be one.
    """
    matrix = graph_features(dataset)
    first = dataset.groupby("customer_id", observed=True).head(1).index
    assert (matrix.loc[first, "gph_customer_events"] == 0).all()
    assert (matrix.loc[first, "gph_customer_merchants"] == 0).all()
    assert (matrix.loc[first, "gph_component_events"] == 0).all()

    first_merchant = dataset.groupby("merchant_id", observed=True).head(1).index
    assert (matrix.loc[first_merchant, "gph_merchant_customers"] == 0).all()
    assert (matrix.loc[first_merchant, "gph_merchant_fanin_7d"] == 0).all()


def test_counts_are_monotone_within_an_entity(dataset: pd.DataFrame) -> None:
    """Lifetime counters only ever grow. A drop means state was rebuilt or evicted."""
    matrix = graph_features(dataset)
    joined = matrix.assign(customer_id=dataset["customer_id"].to_numpy())
    for _, block in joined.groupby("customer_id", observed=True):
        values = block["gph_customer_events"].to_numpy()
        assert np.all(np.diff(values) >= 0)


def test_unsorted_input_is_rejected(dataset: pd.DataFrame) -> None:
    """Out-of-order events would let a component's aggregates contain the future."""
    shuffled = dataset.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="timestamp-ordered"):
        graph_features(shuffled)


def test_identity_components_are_households_not_one_blob(dataset: pd.DataFrame) -> None:
    """The assumption every component feature rests on, pinned as a property.

    Merchants and BINs are excluded from the component graph precisely because
    they are hubs. If someone adds a merchant edge, this test is what tells them
    that ``gph_component_nodes`` just became a constant.
    """
    graph = EntityGraph()
    graph_features(dataset, graph=graph)
    sizes = graph.component_sizes()
    assert len(sizes) > 100, "the identity graph collapsed into a handful of components"
    largest_share = float(sizes.iloc[0]) / float(sizes.sum())
    assert largest_share < 0.10, (
        f"the largest identity component holds {largest_share:.1%} of all nodes; a hub edge "
        "has been added and every component feature is now measuring the population"
    )


def test_the_pass_is_deterministic(dataset: pd.DataFrame) -> None:
    """Two runs, one answer. See CLAUDE.md §5."""
    first = graph_features(dataset)
    second = graph_features(dataset)
    pd.testing.assert_frame_equal(first, second)


def test_networkx_view_holds_only_identity_nodes(dataset: pd.DataFrame) -> None:
    graph = EntityGraph()
    graph_features(dataset, graph=graph)
    view = graph.to_networkx()
    kinds = {data["kind"] for _, data in view.nodes(data=True)}
    assert kinds <= {"c", "d", "a"}
    assert view.number_of_edges() > 0


def test_entity_vectors_measure_spread_not_averages(dataset: pd.DataFrame) -> None:
    """An entity is characterised by its spread; averaging a per-event matrix loses it."""
    vectors = entity_vectors(dataset, "customer_id")
    assert {"n_counterparties", "counterparty_entropy", "events_per_day"} <= set(vectors.columns)
    assert (vectors["n_events"] > 0).all()
    assert vectors["n_counterparties"].max() > 1


def test_entity_novelty_reads_no_labels(dataset: pd.DataFrame) -> None:
    """L2e must be fittable on a frame with the label column removed entirely.

    The strongest available statement that it is unsupervised: if it needed
    ``is_fraud`` it could not run at all.
    """
    stripped = dataset.drop(columns=["is_fraud", "attack_id", "attack_campaign"])
    model = EntityNovelty(seed=7).fit(stripped)
    scores = model.score(stripped)
    assert len(scores) == len(stripped)
    assert np.isfinite(scores).all()
