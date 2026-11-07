"""The decision layer: a score is not an action.

Everything upstream of this module produces a number. An issuer cannot act on a
number — it has to send one of a small set of responses back down the wire, and
each one has a different cost. This module is the map from the fused score to
that response, and it is deliberately the shortest module in the package: the
work was done by the layers, and the only thing left is to say where the lines
are and why.

The four actions, and what each one costs
-------------------------------------------
``APPROVE``
    The default. Costs nothing, and is wrong at a rate the model cannot see.
``CHALLENGE``
    Step up: 3-D Secure on a card rail, a re-consent prompt on an agentic one.
    Cheap and recoverable — the cardholder is inconvenienced for a few seconds —
    but it does not exist on every rail, and on the agentic rail there may be no
    human awake to answer it. That last point is why ``ag_human_present`` is
    consulted here and not only by the model.
``REVIEW``
    Hold and queue for an analyst. The expensive one: it costs staff time and,
    at the volumes an issuer runs, the queue length is the binding constraint on
    the whole firewall. The FPR budget is a queue-length budget wearing a
    statistician's clothes.
``DECLINE``
    Refuse the authorisation. Cheapest to operate and by far the most expensive
    to get wrong: a declined legitimate customer is a support call, and often a
    lost customer.

Thresholds are FPR budgets, not score values
----------------------------------------------
Each boundary is placed at a quantile of the **legitimate** score distribution,
exactly as :func:`mantis.defense.metrics.threshold_at_fpr` places the reporting
operating point. That is what makes the policy portable: retrain the model, refit
the boundaries from the new score distribution, and the number of customers
inconvenienced per million authorisations is unchanged. Hard-coding score values
would mean every retrain silently re-prices the queue.

The default budgets are the same grid the recall curve is reported over —
:data:`mantis.defense.metrics.FPR_GRID` — so the table in RESULTS.md and the
policy in this module are the same three numbers and cannot drift apart.

L0 outranks the score, always
-------------------------------
A deterministic clause firing is not evidence, it is a **protocol violation**: an
expired mandate, a scope breach, a replayed signature. Those decline regardless
of what the ensemble thinks, and the reason is not that the rule is more accurate
— it is that "the mandate had expired" is a defensible thing to tell a
cardholder and "the gradient boosting model scored 0.83" is not. This is also the
part of the firewall that is deployable today, with no training data at all,
which is half of the answer this project gives to the zero-day question.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np

__all__ = ["Decision", "PolicyThresholds", "decide", "escalate"]


class Decision(StrEnum):
    """What the firewall tells the authorisation stream to do.

    A string enum so that it serialises into a manifest, a parquet column and an
    SSE frame without a converter, and so that comparisons against the string
    form in the console keep working.
    """

    APPROVE = "approve"
    CHALLENGE = "challenge"
    REVIEW = "review"
    DECLINE = "decline"


#: Order of severity, so a caller can take a maximum over several signals.
_SEVERITY: Final[dict[Decision, int]] = {
    Decision.APPROVE: 0,
    Decision.CHALLENGE: 1,
    Decision.REVIEW: 2,
    Decision.DECLINE: 3,
}


@dataclass(frozen=True, slots=True)
class PolicyThresholds:
    """Score boundaries, fitted from the legitimate score distribution."""

    challenge: float
    review: float
    decline: float

    @classmethod
    def fit(
        cls,
        scores: np.ndarray,
        labels: np.ndarray,
        *,
        challenge_fpr: float = 0.010,
        review_fpr: float = 0.005,
        decline_fpr: float = 0.001,
    ) -> PolicyThresholds:
        """Place each boundary at its false-positive budget on legitimate traffic.

        Fitted on the negatives alone, like every other threshold in this
        package. A boundary fitted to maximise something on a labelled set is an
        operating point fitted to the answer.
        """
        from mantis.defense.metrics import threshold_at_fpr

        return cls(
            challenge=threshold_at_fpr(scores, labels, challenge_fpr),
            review=threshold_at_fpr(scores, labels, review_fpr),
            decline=threshold_at_fpr(scores, labels, decline_fpr),
        )


def decide(
    scores: np.ndarray,
    thresholds: PolicyThresholds,
    *,
    l0_violation: np.ndarray | None = None,
    human_present: np.ndarray | None = None,
) -> np.ndarray:
    """Map fused scores to :class:`Decision` values.

    Args:
        scores: Fused score per event.
        thresholds: Boundaries from :meth:`PolicyThresholds.fit`.
        l0_violation: True where a deterministic L0 clause fired. Those decline
            outright, whatever the score says.
        human_present: True where a human is at the device. Where it is False a
            ``CHALLENGE`` has nobody to answer it, so it is escalated to
            ``REVIEW`` rather than being sent into the void — an unanswerable
            step-up is a decline with extra latency.

    Returns:
        An object array of :class:`Decision`, one per event.
    """
    scores = np.asarray(scores, dtype=float)
    out = np.full(len(scores), Decision.APPROVE, dtype=object)

    finite = np.isfinite(scores)
    out[finite & (scores >= thresholds.challenge)] = Decision.CHALLENGE
    out[finite & (scores >= thresholds.review)] = Decision.REVIEW
    out[finite & (scores >= thresholds.decline)] = Decision.DECLINE

    if human_present is not None:
        unanswerable = (out == Decision.CHALLENGE) & ~np.asarray(human_present, dtype=bool)
        out[unanswerable] = Decision.REVIEW

    if l0_violation is not None:
        out[np.asarray(l0_violation, dtype=bool)] = Decision.DECLINE
    return out


def escalate(left: Decision, right: Decision) -> Decision:
    """The more severe of two decisions."""
    return left if _SEVERITY[left] >= _SEVERITY[right] else right
