"""Weighted fusion — because the unweighted version was worse than L1 alone.

The Day 4 result this exists to fix
-------------------------------------
Day 4 combined L1 and L2 with an **unweighted** noisy-OR over their
legitimate-traffic percentiles and measured the fused recall at **0.286 against
L1's 0.361**. Fusing made the detector worse, and RESULTS.md said so rather than
quoting the better row. The cause was not subtle: L2 is close to random, and at a
fixed 0.1% false-positive budget every legitimate event a near-random layer ranks
highly consumes budget L1 would have spent on a real one.

The fix is to let the data decide how much each layer is worth. A logistic
regression over the layer percentiles does exactly that: it will drive L2's
coefficient toward zero on its own, which is both the right answer and a more
convincing demonstration than deleting the layer would be.

Fitted where the layers cannot see
------------------------------------
A stacker fitted on scores the base layers produced for their **own** training
rows learns from L1's overfit, and it learns the wrong thing: L1 separates its
training positives almost perfectly, so the stacker concludes L1 is sufficient
and gives every other layer zero weight regardless of merit.

So the training window is split again in time. Layers are fitted on the first
:data:`INNER_TRAIN_SHARE` of it and produce scores for the remainder, which none
of them has seen; the fusion weights are fitted there. The headline layers are
then refitted on the **whole** training window and the weights carried over.
Refitting is the honest ordering — throwing away 20% of the training data
permanently to pay for the stacker would cost more recall than the stacker
returns.

Percentiles, **and** the raw score — the tail is the whole game
-----------------------------------------------------------------
Each layer's score is mapped onto its percentile within the **legitimate** score
distribution. A boosting margin, an isolation forest path length and a logistic
probability share no scale; "more extreme than 99.4% of legitimate traffic" means
the same thing for all three. The percentile map is fitted on the fusion window's
legitimate rows and reused unchanged on test, so nothing about the test
distribution reaches the transform.

On its own that transform is **actively harmful here**, and the first version of
this module measured it: fused recall of 0.104 against L1's 0.553. The cause is
that a percentile computed against a finite reference **saturates**. Every score
above the largest legitimate score maps to exactly 1.0 — and at a 0.1%
false-positive budget the events being ranked are precisely the ones in that
saturated tail. L1's ordering inside its own top 0.1% is the entire signal, and
the percentile threw it away, leaving the fused ranking to be settled by whatever
L2 happened to think.

So each layer contributes **two** columns: its percentile, which is robust and
commensurable, and its raw score standardised against the fusion window's
legitimate median and spread, which preserves the ordering inside the tail. The
stacker is a logistic regression and fits a coefficient per column, so it does
not need them on a shared scale — the percentile was never load-bearing for
comparability, only for interpretability.

Missing layers
----------------
L3 is NaN on any row with no provenance chain — most of the file. A NaN is
mapped to the **median legitimate percentile** (0.5) rather than to 0, because 0
means "cleaner than every legitimate event", which is an opinion, and the whole
point of L3's NaN is that it has none. Whether a layer had an opinion at all is
passed separately as an indicator, so the stacker can learn that an agentic event
with a low L3 score is genuinely reassuring while a classic event's absent L3 is
not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import pandas as pd

__all__ = ["INNER_TRAIN_SHARE", "FusionModel", "legit_percentile"]

#: Share of the training window used to fit the base layers when producing the
#: out-of-sample scores the fusion weights are fitted on.
INNER_TRAIN_SHARE: Final[float] = 0.80

#: What a NaN layer score becomes. The median of the legitimate distribution:
#: "no opinion", not "clean".
_ABSENT_PERCENTILE: Final[float] = 0.5


def legit_percentile(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map scores onto their percentile within a legitimate score distribution."""
    reference = np.sort(np.asarray(reference, dtype=float))
    reference = reference[np.isfinite(reference)]
    scores = np.asarray(scores, dtype=float)
    if reference.size == 0:
        return np.where(np.isfinite(scores), 0.0, _ABSENT_PERCENTILE)
    out = np.searchsorted(reference, scores, side="left") / reference.size
    return np.where(np.isfinite(scores), out, _ABSENT_PERCENTILE)


@dataclass(slots=True)
class FusionModel:
    """Logistic stacker over layer percentiles. Weights fitted, never hand-set."""

    seed: int = 1337
    layers: list[str] = field(default_factory=list)
    #: Legitimate score distribution per layer, fitted on the fusion window.
    references: dict[str, np.ndarray] = field(default_factory=dict)
    #: ``(centre, spread)`` per layer, from the fusion window's legitimate rows.
    standardisation: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: Whether a layer's "had an opinion" indicator actually varies. A constant
    #: column is collinear with the intercept, and including it makes the fitted
    #: weights unreadable for no gain — every layer would print the same number
    #: there, which is exactly what the first version of this table did.
    varying: dict[str, bool] = field(default_factory=dict)
    classifier: Any = None

    def _design(self, scores: dict[str, np.ndarray]) -> np.ndarray:
        """Per layer: percentile, standardised raw score, and presence."""
        columns: list[np.ndarray] = []
        for name in self.layers:
            raw = np.asarray(scores[name], dtype=float)
            present = np.isfinite(raw)
            columns.append(legit_percentile(raw, self.references[name]))
            centre, spread = self.standardisation[name]
            columns.append(np.where(present, (np.nan_to_num(raw) - centre) / spread, 0.0))
            if self.varying[name]:
                columns.append(present.astype(float))
        return np.column_stack(columns)

    def fit(self, scores: dict[str, np.ndarray], y: np.ndarray) -> FusionModel:
        """Fit the weights on out-of-sample layer scores. See the module docstring.

        Args:
            scores: ``{layer name: score vector}`` on the **fusion window** — rows
                none of the base layers was fitted on.
            y: Ground truth for those rows.
        """
        from sklearn.linear_model import LogisticRegression

        y = np.asarray(y, dtype=int)
        self.layers = sorted(scores)
        for name in self.layers:
            values = np.asarray(scores[name], dtype=float)
            legit = values[y == 0]
            self.references[name] = legit
            finite = legit[np.isfinite(legit)]
            centre = float(np.median(finite)) if finite.size else 0.0
            spread = float(np.std(finite)) if finite.size else 1.0
            self.standardisation[name] = (centre, spread if spread > 1e-12 else 1.0)
            self.varying[name] = bool(np.isfinite(values).any() and (~np.isfinite(values)).any())
        design = self._design(scores)
        self.classifier = LogisticRegression(
            max_iter=1_000,
            C=1.0,
            class_weight="balanced",
            random_state=self.seed,
        )
        self.classifier.fit(design, y)
        return self

    def score(self, scores: dict[str, np.ndarray]) -> np.ndarray:
        """Fused score in [0, 1]."""
        if self.classifier is None:
            raise RuntimeError("FusionModel.score called before fit")
        missing = [name for name in self.layers if name not in scores]
        if missing:
            raise ValueError(f"fusion is missing layer scores: {missing}")
        return np.asarray(
            self.classifier.predict_proba(self._design(scores))[:, 1], dtype=float
        )

    def weights(self) -> pd.DataFrame:
        """The fitted coefficients, one row per layer.

        Printed by the CLI because the interesting result is not the fused number
        but *what the fusion learned*: a coefficient near zero on a layer is the
        model saying that layer carries no independent information, which is a
        cleaner statement than any table of recalls.
        """
        if self.classifier is None:
            raise RuntimeError("FusionModel.weights called before fit")
        coefficients = np.asarray(self.classifier.coef_).ravel()
        rows = []
        cursor = 0
        for name in self.layers:
            percentile = float(coefficients[cursor])
            standardised = float(coefficients[cursor + 1])
            cursor += 2
            present = float("nan")
            if self.varying[name]:
                present = float(coefficients[cursor])
                cursor += 1
            rows.append(
                {
                    "layer": name,
                    "weight_percentile": percentile,
                    "weight_score": standardised,
                    "weight_present": present,
                }
            )
        return pd.DataFrame(rows).sort_values(
            "weight_percentile", ascending=False, ignore_index=True
        )
