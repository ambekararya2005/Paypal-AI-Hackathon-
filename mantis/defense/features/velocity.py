"""Velocity features: the single chronological pass over the file.

This module is thin on purpose. All of the mechanism lives in
:mod:`mantis.defense.features.state`; what is here is the loop that walks events
in timestamp order, hands each one to the store, and collects what comes back.

The decline-ratio windows
--------------------------
These are the features the 1.1.0 amendment was written for, and they were
literally unbuildable before it: with no ``auth_response`` on the wire there was
nothing to take a ratio of. They are the canonical card-testing detector, and
they work on exactly the keys F4-27 is designed to defeat and the one it is not:

* **per card** — useless against F4-27 by construction. The operator spreads a
  finite credential set thinly so that no single card ever shows velocity. It is
  computed anyway, because an attack that *did* hammer one card would light it up
  and because its uselessness here is itself informative to the model.
* **per BIN** — the one that works. The campaign is concentrated in two or three
  BIN ranges, and a burst of declines inside a range is a shape no single
  cardholder produces.
* **per merchant** — catches the other side of the same campaign: an operator
  spraying attempts across a merchant's checkout.
* **per device** — catches a farm running many credentials from one runtime.

The ratio's denominator is the count of prior events **whose outcome is known**,
not the count of prior events. Those are the same number in this file — every
authorisation resolves immediately — but they will not be in a live stream where
a pre-authorisation can sit open, and writing it the correct way now means the
Day 5 scorer does not need a different formula.

Ordering
--------
The pass requires timestamp order and asserts it. Out-of-order events would let a
key's state contain the future, which is the leak this whole layer is built to
avoid. Sorting is the caller's job (``FeatureBuilder`` does it) because the
sorted order has to be applied to the labels too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mantis.core.events import DECLINE_RESPONSES
from mantis.defense.features.spec import VELOCITY_KEYS
from mantis.defense.features.state import RollingStore, WindowSpec, as_epoch

__all__ = ["velocity_features"]


def velocity_features(
    frame: pd.DataFrame, specs: tuple[WindowSpec, ...] = VELOCITY_KEYS
) -> pd.DataFrame:
    """Velocity features for every row, computed in one forward pass.

    Args:
        frame: Events **already sorted by ``ts``**.
        specs: Which keys to measure over.

    Returns:
        A frame indexed like ``frame``, one column per key/window/statistic.
    """
    epoch = as_epoch(frame["ts"].dt.tz_localize(None) if frame["ts"].dt.tz else frame["ts"])
    if len(epoch) > 1 and not np.all(np.diff(epoch) >= 0):
        raise ValueError(
            "velocity_features requires timestamp-ordered input; unsorted events would "
            "let a key's rolling state contain the future"
        )

    amount = frame["amount"].to_numpy(dtype=float)
    declined = np.isin(frame["auth_response"].to_numpy(), DECLINE_RESPONSES)
    refund = frame["txn_type"].to_numpy() == "refund"
    lag = frame["settlement_lag_hours"].to_numpy(dtype=float)

    # Columns each key needs, pulled out once rather than per row.
    key_columns: dict[str, list[np.ndarray]] = {
        spec.name: [frame[c].to_numpy() for c in spec.columns] for spec in specs
    }

    store = RollingStore(specs)
    names = store.feature_names()
    rows = np.empty((len(frame), len(names)), dtype=float)

    for i in range(len(frame)):
        keys = {
            spec.name: spec.key_of(tuple(column[i] for column in key_columns[spec.name]))
            for spec in specs
        }
        rows[i] = store.observe(
            keys,
            ts=float(epoch[i]),
            amount=float(amount[i]),
            declined=bool(declined[i]),
            # Every authorisation in this file resolves at once. Kept as its own
            # argument because a live stream's will not.
            outcome_known=True,
            refund=bool(refund[i]),
            settlement_lag=None if np.isnan(lag[i]) else float(lag[i]),
        )

    return pd.DataFrame(rows, columns=names, index=frame.index)
