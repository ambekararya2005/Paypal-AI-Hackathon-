"""Weighted fusion of the layer scores.

    from mantis.defense.fusion import FusionModel
    fusion = FusionModel().fit({"l1": s1, "l2": s2, "l3": s3}, y_fusion_window)
    fused = fusion.score({"l1": s1_test, "l2": s2_test, "l3": s3_test})

Day 4's unweighted noisy-OR made the ensemble **worse** than L1 alone. See
:mod:`mantis.defense.fusion.model` for why, and for where the weights are fitted
so that the stacker does not simply learn L1's overfit.
"""

from __future__ import annotations

from mantis.defense.fusion.model import INNER_TRAIN_SHARE, FusionModel, legit_percentile

__all__ = ["INNER_TRAIN_SHARE", "FusionModel", "legit_percentile"]
