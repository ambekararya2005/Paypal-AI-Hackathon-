"""Keyed rolling state — velocity features computed once, not re-scanned.

Why this is not a groupby
--------------------------
The obvious way to build ``count of authorisations on this card in the last
hour`` is a ``groupby(...).rolling('1h')`` over the whole file. It is three lines
and it is the wrong answer, for a reason that only shows up on Day 5: a groupby
re-reads history. Scoring **one** live authorisation would mean re-scanning every
prior event for that card. The p99 budget is 50 ms for the whole firewall, and a
rescan blows it on the first layer.

So the state here is shaped the way a deployed system's would be: a **keyed
store** that each event *updates* in O(log k), and that answers every window
query from the state it already holds. The offline builder walks the file once
in timestamp order and pushes each event into the store; the online scorer will
push each event as it arrives. **Both call the same code**, which is the point —
a velocity feature that is computed one way in training and another way in
production is a silent train/serve skew, and it is one of the standard ways a
fraud model that looked good offline turns out not to work.

The mechanics
-------------
Each key holds three parallel arrays — timestamps, amounts, decline flags —
plus the prefix sums of the last two. A window query is:

1. ``bisect_left`` on the timestamps for the window start,  O(log k)
2. two prefix-sum subtractions for the amount and decline totals,  O(1)

Appending is amortised O(1). Old entries are evicted from the head once they
fall out of the **longest** window anyone asks for, so memory per key is bounded
by that key's traffic in seven days rather than by the length of the file.
Eviction is what makes the offline pass and the online scorer have the same
memory profile; without it the offline pass would quietly be an unbounded
accumulator that no server could run.

The one rule that matters for correctness
------------------------------------------
**Every query is answered from state that excludes the event being scored.**
:meth:`RollingStore.observe` returns the features *first* and only then folds
the event in. Getting this backwards is self-fulfilling: ``count_1h`` would
always be at least one, ``decline_ratio`` on a declined authorisation would
include its own decline, and the model would learn a fact about the present it
will not have at scoring time. That is leakage of exactly the kind HARD RULE 1
exists to prevent, and it is much harder to spot than a stray label column
because nothing is misnamed.

What may be folded in, and when
--------------------------------
An event's **outcome** (approved/declined, settled, settlement lag) is not known
when the authorisation is being scored — it is the answer, not the question. But
it *is* known for every earlier event on the same card. So
:meth:`RollingStore.observe` takes the outcome and folds it into state for the
benefit of *later* events, while never letting it reach the features of the
event that carried it. That asymmetry is the whole reason ``decline_ratio_1h``
is a legitimate feature and ``auth_response`` is not.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Final

import numpy as np

__all__ = ["WINDOWS", "KeyState", "RollingStore", "WindowSpec"]

#: The three windows every velocity key is measured over, in seconds.
#:
#: An hour catches a burst inside a single session, a day catches a campaign
#: pacing itself under an hourly rule, and a week catches cultivation — the
#: F2-16 bust-out spends weeks looking ordinary before it does anything. Three
#: is a deliberate ceiling: each one multiplies the feature count by the number
#: of keys, and a fourth would add columns faster than it adds information.
WINDOWS: Final[dict[str, int]] = {
    "1h": 3_600,
    "24h": 86_400,
    "7d": 604_800,
}

#: Longest window, and therefore the eviction horizon.
_HORIZON: Final[int] = max(WINDOWS.values())


@dataclass(slots=True)
class WindowSpec:
    """One velocity key: what it is grouped by and what it is called."""

    name: str
    columns: tuple[str, ...]

    def key_of(self, values: tuple[object, ...]) -> str | None:
        """Join one row's key columns, or ``None`` when any part is missing.

        A missing part means the key does not apply to this event — a classic
        authorisation has no agent, a remote one has no terminal — and a velocity
        feature over a key that does not exist must be null rather than lumped
        into a shared ``"None"`` bucket. Lumping is not a rounding error: it
        would make "every event with no device" one enormous entity whose
        velocity is the file's, and a model would learn that bucket as a feature.
        """
        parts: list[str] = []
        for value in values:
            if value is None or value != value:  # NaN is never equal to itself
                return None
            parts.append(str(value))
        return "\x1f".join(parts)


@dataclass(slots=True)
class KeyState:
    """Rolling history for one key value. See the module docstring."""

    ts: list[float] = field(default_factory=list)
    #: Prefix sums, offset by one: ``cum_amount[i]`` is the total of the first
    #: ``i`` retained events. Length is always ``len(ts) + 1``.
    cum_amount: list[float] = field(default_factory=lambda: [0.0])
    cum_declines: list[float] = field(default_factory=lambda: [0.0])
    cum_outcomes: list[float] = field(default_factory=lambda: [0.0])
    cum_refunds: list[float] = field(default_factory=lambda: [0.0])
    cum_lag: list[float] = field(default_factory=lambda: [0.0])
    cum_lag_n: list[float] = field(default_factory=lambda: [0.0])
    #: How many events have been dropped off the head, so absolute positions
    #: stay meaningful after eviction.
    dropped: int = 0
    #: Lifetime counters, which eviction must not touch.
    total: int = 0
    total_amount: float = 0.0
    first_ts: float | None = None
    last_ts: float | None = None

    def evict(self, now: float) -> None:
        """Drop entries that have fallen out of the longest window.

        Bounded memory is not an optimisation here, it is the thing that makes
        the offline pass and the online scorer the same program.
        """
        cut = now - _HORIZON
        drop = bisect_left(self.ts, cut)
        if drop <= 0:
            return
        del self.ts[:drop]
        for series in (
            self.cum_amount,
            self.cum_declines,
            self.cum_outcomes,
            self.cum_refunds,
            self.cum_lag,
            self.cum_lag_n,
        ):
            del series[:drop]
        self.dropped += drop

    def append(
        self,
        ts: float,
        amount: float,
        *,
        declined: bool,
        outcome_known: bool,
        refund: bool,
        settlement_lag: float | None,
    ) -> None:
        """Fold one event into this key's state."""
        self.ts.append(ts)
        self.cum_amount.append(self.cum_amount[-1] + amount)
        self.cum_declines.append(self.cum_declines[-1] + (1.0 if declined else 0.0))
        self.cum_outcomes.append(self.cum_outcomes[-1] + (1.0 if outcome_known else 0.0))
        self.cum_refunds.append(self.cum_refunds[-1] + (1.0 if refund else 0.0))
        has_lag = settlement_lag is not None and settlement_lag == settlement_lag
        self.cum_lag.append(self.cum_lag[-1] + (settlement_lag if has_lag else 0.0))
        self.cum_lag_n.append(self.cum_lag_n[-1] + (1.0 if has_lag else 0.0))

        self.total += 1
        self.total_amount += amount
        if self.first_ts is None:
            self.first_ts = ts
        self.last_ts = ts

    def window(self, now: float, seconds: int) -> tuple[int, float, float, float, float]:
        """``(count, amount_sum, declines, outcomes, refunds)`` over ``[now - seconds, now)``."""
        lo = bisect_left(self.ts, now - seconds)
        hi = len(self.ts)
        return (
            hi - lo,
            self.cum_amount[hi] - self.cum_amount[lo],
            self.cum_declines[hi] - self.cum_declines[lo],
            self.cum_outcomes[hi] - self.cum_outcomes[lo],
            self.cum_refunds[hi] - self.cum_refunds[lo],
        )

    def mean_settlement_lag(self) -> float:
        """Mean settlement lag over everything retained, or NaN when none settled."""
        n = self.cum_lag_n[-1] - self.cum_lag_n[0]
        if n <= 0:
            return float("nan")
        return (self.cum_lag[-1] - self.cum_lag[0]) / n


class RollingStore:
    """Every velocity key for one population, updated in timestamp order.

    Usage is deliberately narrow: :meth:`observe` per event, in order. There is
    no batch entry point, because a batch entry point is how the offline and
    online paths drift apart.
    """

    def __init__(self, specs: tuple[WindowSpec, ...]) -> None:
        self.specs = specs
        self._state: dict[str, dict[str, KeyState]] = {s.name: {} for s in specs}

    def feature_names(self) -> list[str]:
        """Every column :meth:`observe` returns, in the order it returns them."""
        names: list[str] = []
        for spec in self.specs:
            for window in WINDOWS:
                names += [
                    f"vel_{spec.name}_count_{window}",
                    f"vel_{spec.name}_amount_{window}",
                    f"vel_{spec.name}_decline_ratio_{window}",
                    f"vel_{spec.name}_refund_ratio_{window}",
                ]
            names += [
                f"vel_{spec.name}_lifetime_count",
                f"vel_{spec.name}_age_seconds",
                f"vel_{spec.name}_seconds_since_prior",
                f"vel_{spec.name}_amount_vs_mean",
                f"vel_{spec.name}_mean_settlement_lag",
            ]
        return names

    def observe(
        self,
        keys: dict[str, str | None],
        *,
        ts: float,
        amount: float,
        declined: bool,
        outcome_known: bool,
        refund: bool,
        settlement_lag: float | None,
    ) -> list[float]:
        """Features for this event, **then** fold the event into state.

        The order of the two halves of this method is the correctness property
        the module docstring describes. Do not reorder them.
        """
        out: list[float] = []
        for spec in self.specs:
            key = keys.get(spec.name)
            state = self._state[spec.name].get(key) if key is not None else None

            if state is None:
                # An unseen key, or a key that does not apply to this rail. NaN
                # rather than zero: "no history" and "history of zero" are
                # different facts, and LightGBM can split on the difference.
                out += [float("nan")] * (4 * len(WINDOWS) + 5)
            else:
                state.evict(ts)
                for seconds in WINDOWS.values():
                    count, total, declines, outcomes, refunds = state.window(ts, seconds)
                    out.append(float(count))
                    out.append(float(total))
                    out.append(declines / outcomes if outcomes > 0 else float("nan"))
                    out.append(refunds / count if count > 0 else float("nan"))
                mean_amount = state.total_amount / state.total if state.total else float("nan")
                out += [
                    float(state.total),
                    ts - state.first_ts if state.first_ts is not None else float("nan"),
                    ts - state.last_ts if state.last_ts is not None else float("nan"),
                    amount / mean_amount if mean_amount == mean_amount and mean_amount
                    else float("nan"),
                    state.mean_settlement_lag(),
                ]

            if key is not None:
                if state is None:
                    state = KeyState()
                    self._state[spec.name][key] = state
                state.append(
                    ts,
                    amount,
                    declined=declined,
                    outcome_known=outcome_known,
                    refund=refund,
                    settlement_lag=settlement_lag,
                )
        return out

    def size(self) -> dict[str, int]:
        """Distinct keys held per family. Printed by the CLI as a memory sanity check."""
        return {name: len(state) for name, state in self._state.items()}

    def retained_events(self) -> int:
        """Events currently inside the eviction horizon, summed over every key."""
        return sum(
            len(key_state.ts)
            for family in self._state.values()
            for key_state in family.values()
        )


def as_epoch(values: object) -> np.ndarray:
    """Timestamps as float seconds. Tz-aware input is fine; the offset cancels."""
    series = np.asarray(values, dtype="datetime64[ns]")
    return series.astype("int64") / 1e9
