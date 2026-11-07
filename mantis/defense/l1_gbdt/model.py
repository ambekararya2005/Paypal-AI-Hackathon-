"""L1 — gradient-boosted supervised detector.

LightGBM, a **time-based** split, isotonic calibration, and a threshold fitted at
0.1% FPR on legitimate traffic alone.

Why the split is time-based and not random
--------------------------------------------
A random split puts events from the same campaign on both sides of the line. The
model then sees half a ring in training and is asked to recognise the other half,
which it does easily — by recognising the ring, not the behaviour. Every velocity
and entity feature makes this worse, because they are computed from neighbours
that are now in the training set. The result is a large, meaningless number.

A time-based split asks the question a deployment actually faces: fit on what you
had, score what came next. It is strictly harder and it is the only split whose
result transfers.

Why calibration
----------------
The fusion layer combines L1 with L2 and (later) L3, and combining raw scores
from different families is meaningless — an isolation-forest score and a boosting
margin do not live on the same scale. Isotonic regression maps L1's output to
something that behaves like a probability, fitted on a held-out slice of the
training window so the calibration is not fitted on its own predictions.

The class weighting decision, and why it is *not* resampling
--------------------------------------------------------------
Prevalence is about 1%. The reflex is to oversample the minority class or to set
``is_unbalance``, and both distort the score distribution in a way that then has
to be undone before the threshold means anything. Instead the model trains on the
natural distribution with ``scale_pos_weight`` left alone: the threshold is
chosen from the score *quantile* on negatives, which is invariant to any
monotone distortion. Rebalancing would buy nothing and cost calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import pandas as pd

__all__ = ["LGB_PARAMS", "L1Model"]

#: LightGBM parameters. Deliberately small and heavily regularised.
#:
#: There are ~2,000 positives against 200,000 negatives, and a deep forest on
#: that ratio memorises campaigns — a leaf holding four events from one ring is
#: not a pattern, it is a lookup. The depth cap, the leaf minimum and the feature
#: subsampling are all there to force the model to combine weak evidence rather
#: than isolate rings, which is the property the whole HARD/CLEAN design is
#: testing for.
LGB_PARAMS: Final[dict[str, Any]] = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 40,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "verbose": -1,
    "num_threads": 0,
}

#: Share of the training window held out to fit the isotonic calibrator. Taken
#: from the **end** of the window, not at random, for the same reason the main
#: split is time-based.
CALIBRATION_SHARE: Final[float] = 0.2


@dataclass(slots=True)
class L1Model:
    """A fitted LightGBM detector plus its calibrator and operating threshold."""

    n_estimators: int = 400
    seed: int = 1337
    booster: Any = None
    calibrator: Any = None
    feature_names: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    threshold: float = float("nan")

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        *,
        timestamps: pd.Series | None = None,
    ) -> L1Model:
        """Fit the booster, then the calibrator on a held-out tail of the window.

        Args:
            X: Feature matrix from :class:`~mantis.defense.features.FeatureBuilder`.
            y: Ground truth for the training rows.
            timestamps: Used to carve the calibration slice off the *end* of the
                training window. Falls back to row order when absent.
        """
        import lightgbm as lgb
        from sklearn.isotonic import IsotonicRegression

        y = np.asarray(y, dtype=int)
        self.feature_names = list(X.columns)
        self.categorical_features = [
            c for c in X.columns if str(X[c].dtype) == "category"
        ]

        order = (
            np.argsort(timestamps.to_numpy(), kind="stable")
            if timestamps is not None
            else np.arange(len(X))
        )
        cut = int(len(order) * (1.0 - CALIBRATION_SHARE))
        fit_idx, cal_idx = order[:cut], order[cut:]
        # A calibration slice with no positives cannot calibrate anything; fall
        # back to fitting on everything and skipping calibration rather than
        # producing an isotonic map fitted on one class.
        if y[cal_idx].sum() < 10:
            fit_idx, cal_idx = order, np.empty(0, dtype=int)

        dataset = lgb.Dataset(
            X.iloc[fit_idx],
            label=y[fit_idx],
            categorical_feature=self.categorical_features or "auto",
            free_raw_data=False,
        )
        self.booster = lgb.train(
            {**LGB_PARAMS, "seed": self.seed},
            dataset,
            num_boost_round=self.n_estimators,
        )

        if cal_idx.size:
            raw = self.booster.predict(X.iloc[cal_idx])
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator.fit(raw, y[cal_idx])
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated fraud score in [0, 1]."""
        if self.booster is None:
            raise RuntimeError("L1Model.score called before fit")
        raw = np.asarray(self.booster.predict(X[self.feature_names]), dtype=float)
        if self.calibrator is None:
            return raw
        # Isotonic is a step function and produces heavy ties, which makes an
        # exact-quantile threshold unplaceable. Blending a whisker of the raw
        # margin back in breaks ties in the model's own order without moving any
        # score across a calibrated bin.
        return np.asarray(self.calibrator.predict(raw), dtype=float) + 1e-6 * raw

    def fit_threshold(self, scores: np.ndarray, y: np.ndarray, fpr: float) -> float:
        """Set and return the operating threshold from legitimate traffic."""
        from mantis.defense.metrics import threshold_at_fpr

        self.threshold = threshold_at_fpr(scores, y, fpr)
        return self.threshold

    def importance(self, top: int = 20) -> pd.DataFrame:
        """Gain-ranked feature importance. Used by the CLI and, later, by L4/explain."""
        if self.booster is None:
            raise RuntimeError("L1Model.importance called before fit")
        gains = self.booster.feature_importance(importance_type="gain")
        frame = pd.DataFrame({"feature": self.booster.feature_name(), "gain": gains})
        frame["share"] = frame["gain"] / max(frame["gain"].sum(), 1e-9)
        return frame.sort_values("gain", ascending=False).head(top).reset_index(drop=True)
