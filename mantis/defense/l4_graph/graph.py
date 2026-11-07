"""L4 — the entity graph, streamed.

Why a graph layer at all
-------------------------
Day 4 measured the two families that a per-event model has no way to see:
**F2 at 0.060 and F4 at 0.056** recall@0.1%FPR. Both are entity-level attacks —
a synthetic identity cohort, a bust-out, a credential set sprayed across a BIN,
a mule ring fanning into one collection point. Every individual authorisation in
them is bland by construction (that is what the foundry's 0.95 separability gate
enforces), and a model scoring one row at a time is being asked to see a ring
through a keyhole.

The signal is in the **relations**: who shares a device with whom, how many
unrelated identities pay one merchant inside a week, how many merchants one BIN
touched in an hour. That is what this module computes.

Streaming, not a snapshot — and why that is the leakage-safe choice
---------------------------------------------------------------------
The obvious implementation builds one NetworkX graph over the whole file and
reads node metrics off it. That graph contains the future: a test-period event
would be scored against a component whose size counts events that had not
happened yet. The alternative discipline used elsewhere in this package — fit on
train, map test onto the fitted values — is leak-free but *blind*: a ring whose
entities are all new in the test window gets NaN for every graph feature, which
is precisely the case the layer exists for.

So the graph is **streamed**, exactly like :mod:`mantis.defense.features.state`:
one forward pass in timestamp order, and every event's features are read off the
graph as it stood **strictly before** that event, and only then are the event's
edges folded in. Backward-looking by construction, no fit/transform split
needed, and identical code offline and online. :meth:`EntityGraph.observe` is
the same read-then-fold contract as ``RollingStore.observe`` and the same rule
applies: do not reorder the two halves.

Which edges make a component, and which deliberately do not
-------------------------------------------------------------
The components are built over the **identity graph** — customer, device, agent —
and nothing else. Merchants and BINs are excluded on purpose, and the reason is
the classic way naive graph features fail: they are **hubs**. This population has
16 BINs and a Zipf merchant popularity curve, so one union through a popular
merchant fuses the entire file into a single giant component and
``component_size`` becomes a constant. Measured on the identity graph the median
component is a household — one customer, one or two devices, one agent — which
is what makes a component of forty a statement.

Merchant-side structure is not thrown away, it is measured the right way: as
**windowed distinct counts** (``gph_merchant_fanin_7d``) and as the number of
distinct identity **components** paying one merchant inside a window
(``gph_merchant_components_7d``). The ratio between those two is the ring
detector: many payers spanning many components is a busy shop; many payers
spanning *few* components is a mule network, because the payers are related to
each other.

The "beneficiary" node
-----------------------
The schema has no beneficiary field — it is an authorisation message, and the
counterparty of an authorisation is the merchant. So merchant *is* the
beneficiary node here, and on outbound types (refund, reversal, credit) the
direction of the money is reversed but the counterparty is the same entity. That
substitution is recorded rather than hidden, in the same spirit as F6-40 standing
in for chargeback abuse on Day 2.

Bounded memory
---------------
Lifetime distinct sets are capped (:data:`DEGREE_CAP`) and windowed ones evict,
so this pass has the same memory profile online and offline. A feature that says
"more than 256 distinct merchants" is not less useful than one that says 4,113.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

from mantis.core.events import DECLINE_RESPONSES

__all__ = [
    "DEGREE_CAP",
    "GRAPH_FEATURE_NAMES",
    "EntityGraph",
    "build_networkx",
    "graph_features",
]

#: Cap on a lifetime distinct-neighbour set. Bounds memory to
#: ``DEGREE_CAP * n_keys`` rather than to the length of the file, and costs
#: nothing: past a couple of hundred neighbours the exact count is not a feature,
#: it is a hub.
DEGREE_CAP: Final[int] = 256

#: Windows the graph's distinct counters are measured over, in seconds. Shorter
#: than the velocity windows deliberately — a ring's fan-in is a burst, and a
#: 30-day window would dilute it into the merchant's ordinary trade.
_FANIN_WINDOWS: Final[dict[str, int]] = {"24h": 86_400, "7d": 604_800}

#: Window the BIN-burst counters use. Card testing sprays a range across many
#: merchants inside minutes to hours; a day is already generous.
_BURST_WINDOWS: Final[dict[str, int]] = {"1h": 3_600, "24h": 86_400}


class _WindowedDistinct:
    """Distinct values seen on one key inside a sliding window.

    A deque of ``(ts, value)`` plus a multiplicity counter. ``len(counts)`` is the
    distinct count; eviction decrements and drops keys at zero. Amortised O(1)
    per event, and memory bounded by the key's traffic inside the window rather
    than by the file.
    """

    __slots__ = ("counts", "events", "horizon")

    def __init__(self, horizon: int) -> None:
        self.horizon = horizon
        self.events: deque[tuple[float, int]] = deque()
        self.counts: dict[int, int] = {}

    def evict(self, now: float) -> None:
        cutoff = now - self.horizon
        events = self.events
        counts = self.counts
        while events and events[0][0] < cutoff:
            _, value = events.popleft()
            remaining = counts[value] - 1
            if remaining:
                counts[value] = remaining
            else:
                del counts[value]

    def distinct(self, now: float) -> int:
        """Distinct values inside this counter's window, after eviction.

        One counter per window rather than one counter queried at several
        widths. The alternative — keep the widest deque and walk back from the
        tail for the narrower query — is O(window traffic) *per event*, which on
        a 16-BIN population is 70 million inner iterations and turns a 30-second
        pass into a five-minute one. Two counters cost two evictions.
        """
        self.evict(now)
        return len(self.counts)

    def add(self, ts: float, value: int) -> None:
        self.events.append((ts, value))
        self.counts[value] = self.counts.get(value, 0) + 1


@dataclass(slots=True)
class _Component:
    """Aggregates carried on a union-find root."""

    nodes: int = 1
    customers: int = 0
    devices: int = 0
    agents: int = 0
    events: int = 0
    declines: int = 0
    agentic: int = 0
    amount: float = 0.0


#: Every column this layer emits, in order. Declared as data rather than derived
#: from a run so that a missing feature is a test failure and not a silent
#: schema change under L1.
GRAPH_FEATURE_NAMES: Final[tuple[str, ...]] = (
    # identity component
    "gph_component_nodes",
    "gph_component_customers",
    "gph_component_devices",
    "gph_component_events",
    "gph_component_amount_mean",
    "gph_component_decline_ratio",
    "gph_component_agentic_share",
    "gph_component_events_per_customer",
    # device sharing
    "gph_device_customers",
    "gph_device_merchants",
    # customer degree
    "gph_customer_devices",
    "gph_customer_merchants",
    "gph_customer_bins",
    "gph_customer_events",
    # merchant / beneficiary side
    "gph_merchant_customers",
    "gph_merchant_devices",
    "gph_merchant_fanin_24h",
    "gph_merchant_fanin_7d",
    "gph_merchant_components_7d",
    "gph_merchant_fanin_per_component",
    "gph_merchant_fanin_burst",
    # BIN burst
    "gph_bin_merchants_1h",
    "gph_bin_merchants_24h",
    "gph_bin_customers_1h",
    # agent
    "gph_agent_merchants",
    "gph_agent_customers",
    # novelty of this event's own edges
    "gph_pair_customer_merchant_prior",
    "gph_new_edge_share",
)


@dataclass(slots=True)
class EntityGraph:
    """Streaming entity graph. One instance per pass; ``observe`` per event."""

    # -- node interning -------------------------------------------------------- #
    _ids: dict[str, int] = field(default_factory=dict, repr=False)

    # -- union-find over the identity graph ------------------------------------ #
    _parent: list[int] = field(default_factory=list, repr=False)
    _rank: list[int] = field(default_factory=list, repr=False)
    _component: dict[int, _Component] = field(default_factory=dict, repr=False)

    # -- lifetime distinct neighbour sets (capped) ------------------------------ #
    _device_customers: dict[int, set[int]] = field(default_factory=dict, repr=False)
    _device_merchants: dict[int, set[int]] = field(default_factory=dict, repr=False)
    _customer_devices: dict[int, set[int]] = field(default_factory=dict, repr=False)
    _customer_merchants: dict[int, set[int]] = field(default_factory=dict, repr=False)
    _customer_bins: dict[int, set[int]] = field(default_factory=dict, repr=False)
    _merchant_customers: dict[int, set[int]] = field(default_factory=dict, repr=False)
    _merchant_devices: dict[int, set[int]] = field(default_factory=dict, repr=False)
    _agent_merchants: dict[int, set[int]] = field(default_factory=dict, repr=False)
    _agent_customers: dict[int, set[int]] = field(default_factory=dict, repr=False)

    # -- windowed distinct counters, one table per window ------------------------ #
    _merchant_fanin_24h: dict[int, _WindowedDistinct] = field(default_factory=dict, repr=False)
    _merchant_fanin_7d: dict[int, _WindowedDistinct] = field(default_factory=dict, repr=False)
    _merchant_comps_7d: dict[int, _WindowedDistinct] = field(default_factory=dict, repr=False)
    _bin_merchants_1h: dict[int, _WindowedDistinct] = field(default_factory=dict, repr=False)
    _bin_merchants_24h: dict[int, _WindowedDistinct] = field(default_factory=dict, repr=False)
    _bin_customers_1h: dict[int, _WindowedDistinct] = field(default_factory=dict, repr=False)

    # -- pair counters ------------------------------------------------------------ #
    #: ``(customer << 32) | merchant`` rather than a tuple key: a million tuple
    #: keys is about 160 MB of dict, a million ints about half that, and this is
    #: the only table that grows with the length of the file.
    _pair_counts: dict[int, int] = field(default_factory=dict, repr=False)
    _customer_events: dict[int, int] = field(default_factory=dict, repr=False)

    # -- edges retained for the NetworkX view -------------------------------------- #
    #: Identity edges only, and only distinct ones, so the analytical view is a
    #: few tens of thousands of edges rather than one per authorisation.
    _identity_edges: set[tuple[int, int]] = field(default_factory=set, repr=False)
    _labels: list[str] = field(default_factory=list, repr=False)

    # -- interning ----------------------------------------------------------------- #

    def _node(self, kind: str, value: object) -> int | None:
        """Intern ``kind:value`` to an int, or ``None`` when the value is absent."""
        if value is None or value != value:  # NaN is never equal to itself
            return None
        key = f"{kind}:{value}"
        node = self._ids.get(key)
        if node is None:
            node = len(self._parent)
            self._ids[key] = node
            self._labels.append(key)
            self._parent.append(node)
            self._rank.append(0)
            comp = _Component()
            if kind == "c":
                comp.customers = 1
            elif kind == "d":
                comp.devices = 1
            elif kind == "a":
                comp.agents = 1
            self._component[node] = comp
        return node

    # -- union-find ---------------------------------------------------------------- #

    def find(self, node: int) -> int:
        """Root of ``node``, with path compression."""
        parent = self._parent
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def _union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self._rank[a] < self._rank[b]:
            a, b = b, a
        self._parent[b] = a
        if self._rank[a] == self._rank[b]:
            self._rank[a] += 1
        ca, cb = self._component[a], self._component.pop(b)
        ca.nodes += cb.nodes
        ca.customers += cb.customers
        ca.devices += cb.devices
        ca.agents += cb.agents
        ca.events += cb.events
        ca.declines += cb.declines
        ca.agentic += cb.agentic
        ca.amount += cb.amount

    # -- helpers -------------------------------------------------------------------- #

    @staticmethod
    def _add_capped(table: dict[int, set[int]], key: int, value: int) -> None:
        bucket = table.get(key)
        if bucket is None:
            table[key] = {value}
        elif len(bucket) < DEGREE_CAP:
            bucket.add(value)

    @staticmethod
    def _size(table: dict[int, set[int]], key: int | None) -> float:
        if key is None:
            return float("nan")
        bucket = table.get(key)
        return float(len(bucket)) if bucket is not None else 0.0

    def _windowed(
        self, table: dict[int, _WindowedDistinct], key: int, horizon: int
    ) -> _WindowedDistinct:
        counter = table.get(key)
        if counter is None:
            counter = _WindowedDistinct(horizon)
            table[key] = counter
        return counter

    @staticmethod
    def _distinct(table: dict[int, _WindowedDistinct], key: int | None, now: float) -> float:
        if key is None:
            return float("nan")
        counter = table.get(key)
        return float(counter.distinct(now)) if counter is not None else 0.0

    # -- the pass ---------------------------------------------------------------------- #

    def observe(
        self,
        *,
        ts: float,
        customer: object,
        device: object,
        merchant: object,
        agent: object,
        card_bin: object,
        amount: float,
        declined: bool,
    ) -> list[float]:
        """Features for this event, **then** fold its edges into the graph.

        The order of the two halves is the correctness property this module
        exists to hold. Do not reorder them: every count returned here excludes
        the event that produced it, which is what an online scorer would see.
        """
        c = self._node("c", customer)
        d = self._node("d", device)
        m = self._node("m", merchant)
        a = self._node("a", agent)
        b = self._node("b", card_bin)

        # ---------------- read ---------------- #
        root = self.find(c) if c is not None else None
        comp = self._component.get(root) if root is not None else None
        if comp is None:
            comp_row = [float("nan")] * 8
        else:
            events = comp.events
            comp_row = [
                float(comp.nodes),
                float(comp.customers),
                float(comp.devices),
                float(events),
                comp.amount / events if events else float("nan"),
                comp.declines / events if events else float("nan"),
                comp.agentic / events if events else float("nan"),
                events / comp.customers if comp.customers else float("nan"),
            ]

        merchant_customers = self._size(self._merchant_customers, m)
        fanin_24h = self._distinct(self._merchant_fanin_24h, m, ts)
        fanin_7d = self._distinct(self._merchant_fanin_7d, m, ts)
        components_7d = self._distinct(self._merchant_comps_7d, m, ts)

        # The ring detector. Many payers spanning many components is a busy shop;
        # many payers spanning few components is a network, because the payers
        # are related to one another.
        fanin_per_component = (
            fanin_7d / components_7d if components_7d and components_7d == components_7d
            else float("nan")
        )
        # And how much of the merchant's lifetime customer base arrived this week.
        fanin_burst = (
            fanin_7d / merchant_customers
            if merchant_customers and merchant_customers == merchant_customers
            else float("nan")
        )

        pair_prior = (
            float(self._pair_counts.get((c << 32) | m, 0))
            if c is not None and m is not None
            else float("nan")
        )

        # How many of this event's own identity edges are being seen for the
        # first time. A ring assembling itself scores 1.0 on every event; an
        # established household scores 0.0.
        edges = [(c, d), (c, a), (d, a)]
        present = [(u, v) for u, v in edges if u is not None and v is not None]
        if present:
            new = sum(
                1
                for u, v in present
                if (min(u, v), max(u, v)) not in self._identity_edges
            )
            new_edge_share = new / len(present)
        else:
            new_edge_share = float("nan")

        row = [
            *comp_row,
            self._size(self._device_customers, d),
            self._size(self._device_merchants, d),
            self._size(self._customer_devices, c),
            self._size(self._customer_merchants, c),
            self._size(self._customer_bins, c),
            float(self._customer_events.get(c, 0)) if c is not None else float("nan"),
            merchant_customers,
            self._size(self._merchant_devices, m),
            fanin_24h,
            fanin_7d,
            components_7d,
            fanin_per_component,
            fanin_burst,
            self._distinct(self._bin_merchants_1h, b, ts),
            self._distinct(self._bin_merchants_24h, b, ts),
            self._distinct(self._bin_customers_1h, b, ts),
            self._size(self._agent_merchants, a),
            self._size(self._agent_customers, a),
            pair_prior,
            new_edge_share,
        ]

        # ---------------- fold ---------------- #
        for u, v in present:
            key = (min(u, v), max(u, v))
            if key not in self._identity_edges:
                self._identity_edges.add(key)
            self._union(u, v)

        if c is not None:
            self._customer_events[c] = self._customer_events.get(c, 0) + 1
            if d is not None:
                self._add_capped(self._customer_devices, c, d)
                self._add_capped(self._device_customers, d, c)
            if m is not None:
                self._add_capped(self._customer_merchants, c, m)
                self._add_capped(self._merchant_customers, m, c)
                pair = (c << 32) | m
                self._pair_counts[pair] = self._pair_counts.get(pair, 0) + 1
            if b is not None:
                self._add_capped(self._customer_bins, c, b)
        if d is not None and m is not None:
            self._add_capped(self._device_merchants, d, m)
            self._add_capped(self._merchant_devices, m, d)
        if a is not None:
            if m is not None:
                self._add_capped(self._agent_merchants, a, m)
            if c is not None:
                self._add_capped(self._agent_customers, a, c)

        if m is not None and c is not None:
            self._windowed(self._merchant_fanin_24h, m, _FANIN_WINDOWS["24h"]).add(ts, c)
            self._windowed(self._merchant_fanin_7d, m, _FANIN_WINDOWS["7d"]).add(ts, c)
            # The component id is read *after* the union above, so a ring that
            # just merged counts as one component rather than as several.
            self._windowed(self._merchant_comps_7d, m, _FANIN_WINDOWS["7d"]).add(
                ts, self.find(c)
            )
        if b is not None:
            if m is not None:
                self._windowed(self._bin_merchants_1h, b, _BURST_WINDOWS["1h"]).add(ts, m)
                self._windowed(self._bin_merchants_24h, b, _BURST_WINDOWS["24h"]).add(ts, m)
            if c is not None:
                self._windowed(self._bin_customers_1h, b, _BURST_WINDOWS["1h"]).add(ts, c)

        if root is not None:
            comp = self._component[self.find(c)]
            comp.events += 1
            comp.amount += amount
            comp.declines += int(declined)
            comp.agentic += int(a is not None)

        return row

    # -- the analytical view --------------------------------------------------------- #

    def component_sizes(self) -> pd.Series:
        """Node counts of every identity component, largest first.

        Used by the CLI to show that the identity graph really is made of
        households rather than one giant blob — which is the assumption every
        component feature rests on.
        """
        sizes = [comp.nodes for comp in self._component.values()]
        return pd.Series(sizes, name="nodes").sort_values(ascending=False)

    def to_networkx(self):
        """The identity graph as a NetworkX ``Graph``.

        Only distinct identity edges are retained during the pass, so this is a
        few tens of thousands of edges rather than one per authorisation. It is
        the reporting and analysis view — component listing, degree distribution,
        the entity-level novelty aggregates — and nothing in the scoring path
        depends on it.
        """
        import networkx as nx

        graph = nx.Graph()
        labels = self._labels
        graph.add_nodes_from(
            (labels[node], {"kind": labels[node][0]}) for node in range(len(labels))
            if labels[node][0] in "cda"
        )
        graph.add_edges_from((labels[u], labels[v]) for u, v in self._identity_edges)
        return graph


def build_networkx(frame: pd.DataFrame):
    """Convenience: stream ``frame`` and return the identity graph it produces."""
    graph = EntityGraph()
    graph_features(frame, graph=graph)
    return graph.to_networkx()


def graph_features(frame: pd.DataFrame, *, graph: EntityGraph | None = None) -> pd.DataFrame:
    """Graph features for every row, in one forward pass.

    Args:
        frame: Events **already sorted by ``ts``**. The pass asserts it, for the
            same reason the velocity pass does: out-of-order events would let a
            component contain the future.
        graph: An existing graph to continue streaming into. Fresh when omitted.

    Returns:
        A frame indexed like ``frame`` with :data:`GRAPH_FEATURE_NAMES`.
    """
    # Imported here rather than at module scope: ``features.builder`` imports this
    # module, so a top-level import of anything under ``features`` would close the
    # cycle. By call time the package is initialised and this is a dict lookup.
    from mantis.defense.features.state import as_epoch

    epoch = as_epoch(frame["ts"].dt.tz_localize(None) if frame["ts"].dt.tz else frame["ts"])
    if len(epoch) > 1 and not np.all(np.diff(epoch) >= 0):
        raise ValueError(
            "graph_features requires timestamp-ordered input; unsorted events would let a "
            "component's aggregates contain the future"
        )

    graph = graph if graph is not None else EntityGraph()
    customer = frame["customer_id"].to_numpy()
    device = frame["device_id"].to_numpy()
    merchant = frame["merchant_id"].to_numpy()
    agent = frame["ag_agent_id"].to_numpy()
    card_bin = frame["card_bin"].to_numpy()
    amount = frame["amount"].to_numpy(dtype=float)
    declined = np.isin(frame["auth_response"].to_numpy(), DECLINE_RESPONSES)

    rows = np.empty((len(frame), len(GRAPH_FEATURE_NAMES)), dtype=float)
    for i in range(len(frame)):
        rows[i] = graph.observe(
            ts=float(epoch[i]),
            customer=customer[i],
            device=device[i],
            merchant=merchant[i],
            agent=agent[i],
            card_bin=card_bin[i],
            amount=float(amount[i]),
            declined=bool(declined[i]),
        )
    return pd.DataFrame(rows, columns=list(GRAPH_FEATURE_NAMES), index=frame.index)
