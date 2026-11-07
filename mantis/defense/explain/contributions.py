"""Why this authorisation scored what it scored.

LightGBM native contributions, not SHAP
-----------------------------------------
``shap`` is in the dependency list and it is not used here. For a tree ensemble,
``booster.predict(..., pred_contrib=True)`` returns exactly the same quantity the
``TreeExplainer`` computes — per-feature contributions in log-odds that sum, with
the base value, to the raw margin — because ``TreeExplainer`` *calls into
LightGBM* for it. What SHAP adds on top is plotting and a general interface for
model families this project does not use.

What it removes is worth more than that: a dependency on the scoring path, a
model wrapper, and time. The console has to explain an event while a judge
watches, and one array from the booster is milliseconds. Day 7 owns the latency
budget; this module is written not to be the reason it is missed.

The exactness is the point. This is not an approximation of SHAP chosen for
speed — it is the same computation, read from the source that SHAP would have
read it from.

The one thing to be careful about
-----------------------------------
Contributions are in the **raw margin** space, not the calibrated one. L1's
reported score is isotonic-calibrated, so the contributions explain the ranking
rather than the probability. That is the right object anyway — the threshold is a
quantile of the score distribution, so an alert is caused by the event's *rank*,
which is exactly what the margin carries — but a UI that prints "this feature
added 0.4 to a probability of 0.83" would be wrong and this docstring is where
that is written down.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

__all__ = ["Attribution", "explain_events", "top_contributions"]

#: How many features an alert names by default. An analyst reads three or four;
#: twenty is a dump, not an explanation.
TOP_K: Final[int] = 6


@dataclass(frozen=True, slots=True)
class Attribution:
    """One feature's contribution to one event's score, in log-odds.

    ``value`` is deliberately ``object`` rather than ``float``. Seven of the
    matrix's columns are categorical — ``channel``, ``mcc``, ``entry_mode`` — and
    coercing them to a float renders the most readable line in the whole alert as
    ``channel = nan``. An analyst reading "the rail was nan" learns nothing and
    distrusts the rest of the block.
    """

    feature: str
    value: object
    contribution: float

    def line(self) -> str:
        sign = "+" if self.contribution >= 0 else "-"
        return f"{sign}{abs(self.contribution):>6.3f}  {self.feature:<38} = {_render(self.value)}"


def _render(value: object) -> str:
    """Numbers as numbers, categories as their level, missing as ``absent``."""
    if value is None:
        return "absent"
    if isinstance(value, float) and value != value:
        # NaN in this matrix means "this key does not apply to this rail" far
        # more often than it means "unknown", and that distinction is the whole
        # reason the feature layer never fills NaN with a sentinel.
        return "absent"
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return f"{value:,.4g}"
    return str(value)


def top_contributions(
    model: object, X: pd.DataFrame, *, top: int = TOP_K
) -> list[list[Attribution]]:
    """The ``top`` features driving each row's score, largest absolute first.

    Args:
        model: A fitted :class:`~mantis.defense.l1_gbdt.L1Model`.
        X: Feature matrix, the same columns L1 was fitted on.
        top: Features named per event.

    Returns:
        One list of :class:`Attribution` per row of ``X``.
    """
    booster = getattr(model, "booster", None)
    if booster is None:
        raise RuntimeError("top_contributions needs a fitted L1Model")
    names = list(model.feature_names)  # type: ignore[attr-defined]

    # The last column is the base value (the ensemble's expected margin), which
    # is not a feature and must not be ranked as one.
    contributions = np.asarray(booster.predict(X[names], pred_contrib=True), dtype=float)
    contributions = contributions[:, :-1]

    values = X[names].to_numpy(dtype=object)
    order = np.argsort(-np.abs(contributions), axis=1)[:, :top]

    out: list[list[Attribution]] = []
    for row in range(len(X)):
        out.append(
            [
                Attribution(
                    feature=names[column],
                    value=values[row, column],
                    contribution=float(contributions[row, column]),
                )
                for column in order[row]
            ]
        )
    return out



def explain_events(
    model: object,
    X: pd.DataFrame,
    frame: pd.DataFrame,
    scores: np.ndarray,
    *,
    top: int = TOP_K,
    limit: int = 5,
) -> str:
    """A human-readable block for the ``limit`` highest-scoring events.

    This is what ``python -m mantis.defense`` prints and what the live console
    will render per alert. Kept as text rather than a structure because the thing
    being demonstrated is that the firewall's output is *readable* — an alert an
    analyst cannot act on is an alert that will be ignored.
    """
    order = np.argsort(-np.asarray(scores, dtype=float))[:limit]
    attributions = top_contributions(model, X.iloc[order], top=top)

    lines: list[str] = []
    for rank, (position, reasons) in enumerate(zip(order, attributions, strict=True), start=1):
        event = frame.iloc[position]
        label = event.get("attack_id") or ("FRAUD" if event.get("is_fraud") else "legitimate")
        lines.append(
            f"  #{rank}  score {scores[position]:.4f}  {event['channel']}  "
            f"{event['amount']:,.0f} {event['currency']}  [{label}]"
        )
        lines += [f"        {reason.line()}" for reason in reasons]
        lines.append("")
    return "\n".join(lines)
