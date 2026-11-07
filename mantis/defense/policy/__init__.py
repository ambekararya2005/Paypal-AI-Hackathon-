"""The decision layer: fused score -> what the issuer actually does.

    from mantis.defense.policy import Decision, PolicyThresholds, decide
    thresholds = PolicyThresholds.fit(fused_legit_scores, y)
    actions = decide(fused, thresholds, l0_violation=violations)
"""

from __future__ import annotations

from mantis.defense.policy.decide import Decision, PolicyThresholds, decide, escalate

__all__ = ["Decision", "PolicyThresholds", "decide", "escalate"]
