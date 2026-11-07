"""Fusion, policy, and the two new reporting axes.

Day 4's fusion made the ensemble worse than L1 alone, and the cause was a design
choice nobody had measured. These tests pin the properties that failure taught:

1. **The stacker must not be beaten by its own best input.** A fusion that ranks
   worse than L1 on a problem where L1 is the only informative layer is broken,
   whatever its coefficients look like.
2. **The tail must survive the transform.** The first Day 5 fusion also lost to
   L1, for a different reason: a percentile computed against a finite reference
   saturates, and at a 0.1% FP budget the events being ranked are exactly the
   ones in the saturated region. The regression test is a layer whose *only*
   signal lives above every legitimate score.
3. **Thresholds are FPR budgets, not score values**, so a monotone rescaling of
   the score must not change a single decision.
"""

from __future__ import annotations

import numpy as np
import pytest

from mantis.defense.fusion import FusionModel, legit_percentile
from mantis.defense.metrics import (
    FPR_GRID,
    campaign_report,
    recall_at_fpr,
    recall_curve,
    threshold_at_fpr,
)
from mantis.defense.policy import Decision, PolicyThresholds, decide, escalate


def _problem(n: int = 20_000, seed: int = 7) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """L1 informative, L2 pure noise, L3 informative but present on half the rows."""
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.02
    l1 = rng.normal(0.0, 1.0, n) + 2.5 * y
    l2 = rng.normal(0.0, 1.0, n)
    l3 = np.where(rng.random(n) < 0.5, rng.normal(0.0, 1.0, n) + 1.5 * y, np.nan)
    return {"L1": l1, "L2": l2, "L3": l3}, y


def test_fusion_is_not_worse_than_its_best_input() -> None:
    """The Day 4 failure, as a test. A near-random layer must not cost recall."""
    scores, y = _problem()
    fusion = FusionModel(seed=7).fit(scores, y)
    fused = fusion.score(scores)

    l1_recall, _ = recall_at_fpr(scores["L1"], y, 0.01)
    fused_recall, _ = recall_at_fpr(fused, y, 0.01)
    assert fused_recall >= l1_recall - 0.02, (
        f"fusion ({fused_recall:.3f}) is materially worse than L1 alone ({l1_recall:.3f}); "
        "this is the Day 4 defect returning"
    )


def test_fusion_discounts_a_noise_layer() -> None:
    """The stacker should reach the right conclusion about L2 on its own."""
    scores, y = _problem()
    weights = FusionModel(seed=7).fit(scores, y).weights().set_index("layer")
    assert abs(weights.loc["L2", "weight_percentile"]) < abs(
        weights.loc["L1", "weight_percentile"]
    )


def test_the_extreme_tail_survives_the_percentile_transform() -> None:
    """The regression test for the saturation bug.

    A layer whose entire signal sits above every legitimate score maps to
    percentile 1.0 for every positive *and* for nothing else. If the design used
    percentiles alone, the fused ranking inside that tie would be decided by the
    noise layer and recall at a tight budget would collapse.
    """
    rng = np.random.default_rng(3)
    n = 20_000
    y = rng.random(n) < 0.01
    # Positives sit far above every negative, and are ordered among themselves.
    l1 = np.where(y, 100.0 + rng.random(n), rng.normal(0.0, 1.0, n))
    scores = {"L1": l1, "L2": rng.normal(0.0, 1.0, n)}

    saturated = legit_percentile(l1, l1[~y])
    assert (saturated[y] == 1.0).all(), "the fixture no longer exercises saturation"

    fused = FusionModel(seed=3).fit(scores, y).score(scores)
    recall, _ = recall_at_fpr(fused, y, 0.001)
    assert recall > 0.9, f"the tail was flattened: recall {recall:.3f}"


def test_absent_layers_score_as_no_opinion_not_as_clean() -> None:
    """NaN maps to the median legitimate percentile, never to zero."""
    reference = np.arange(100.0)
    mapped = legit_percentile(np.array([np.nan, -1e9, 50.0]), reference)
    assert mapped[0] == pytest.approx(0.5)
    assert mapped[1] == pytest.approx(0.0)
    assert mapped[2] == pytest.approx(0.5)


def test_recall_curve_is_monotone() -> None:
    """More false-positive budget can never buy less recall."""
    scores, y = _problem()
    curve = recall_curve(scores["L1"], y)
    recalls = [curve[f][0] for f in FPR_GRID]
    assert recalls == sorted(recalls)


def test_campaign_recall_is_at_least_event_recall() -> None:
    """A campaign is caught if any of its events is, so it cannot be rarer."""
    rng = np.random.default_rng(11)
    n = 5_000
    campaigns = np.array([f"cmp-{i // 20:03d}" for i in range(n)], dtype=object)
    y = np.zeros(n, dtype=bool)
    y[: n // 10] = True
    scores = rng.random(n) + 0.4 * y
    order = np.arange(n, dtype=float)
    cut = threshold_at_fpr(scores, y, 0.05)

    event_recall = float((scores[y] >= cut).mean())
    report = campaign_report(scores, y, np.where(y, campaigns, None), order, cut)
    assert report.recall >= event_recall
    assert 1 <= report.median_index <= report.median_size


def test_campaign_report_ignores_unattributed_rows() -> None:
    """Legitimate rows carry no campaign and must not become one."""
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    y = np.array([True, True, False, False])
    campaigns = np.array(["cmp-1", "cmp-1", None, None], dtype=object)
    report = campaign_report(scores, y, campaigns, np.arange(4.0), 0.5)
    assert report.n_campaigns == 1
    assert report.n_caught == 1
    assert report.median_index == 1


def test_policy_boundaries_are_invariant_to_a_monotone_rescale() -> None:
    """Thresholds are FPR budgets, so a retrain that shifts the scale re-prices nothing."""
    scores, y = _problem()
    base = scores["L1"]
    rescaled = np.exp(base / 3.0)

    first = decide(base, PolicyThresholds.fit(base, y))
    second = decide(rescaled, PolicyThresholds.fit(rescaled, y))
    assert (first == second).all()


def test_an_unanswerable_challenge_escalates_to_review() -> None:
    """A step-up with no human at the device is a decline with extra latency."""
    scores, y = _problem()
    thresholds = PolicyThresholds.fit(scores["L1"], y)
    with_human = decide(scores["L1"], thresholds, human_present=np.ones(len(y), dtype=bool))
    without = decide(scores["L1"], thresholds, human_present=np.zeros(len(y), dtype=bool))
    assert (with_human == Decision.CHALLENGE).sum() > 0
    assert (without == Decision.CHALLENGE).sum() == 0


def test_an_l0_violation_outranks_the_score() -> None:
    """"The mandate had expired" is defensible; "the ensemble scored 0.83" is not."""
    scores = np.array([0.0, 0.0, 0.0])
    thresholds = PolicyThresholds(challenge=1.0, review=2.0, decline=3.0)
    out = decide(scores, thresholds, l0_violation=np.array([False, True, False]))
    assert list(out) == [Decision.APPROVE, Decision.DECLINE, Decision.APPROVE]


def test_escalate_takes_the_more_severe() -> None:
    assert escalate(Decision.APPROVE, Decision.REVIEW) is Decision.REVIEW
    assert escalate(Decision.DECLINE, Decision.CHALLENGE) is Decision.DECLINE
    assert escalate(Decision.REVIEW, Decision.REVIEW) is Decision.REVIEW
