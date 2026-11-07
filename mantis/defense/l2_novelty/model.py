"""L2 — unsupervised novelty, trained on LEGITIMATE TRAFFIC ONLY.

**Demoted on Day 5: residual monitor and drift canary, not the zero-day layer.**

This module was written as the answer to "what about the attacks you didn't think
of". It is not, and the measurement is unambiguous: 0.4% mean per-family recall
at the 0.1% FPR operating point, 0.62 ROC. Day 5 stopped chasing that number and
reframed the layer. Its job now:

* **residual monitor** — the residue no other layer claims;
* **drift canary** — "has the shape of legitimate traffic moved", which is a
  question it genuinely answers and which nothing else in the firewall does.

It never sees a fraud label, not once, not even to choose its threshold, so its
recall is completely **unaffected** by whether an attack was in training. That
property is real and no supervised layer has it. It is also, on its own, not a
detector, and no table in this repo may present it as one.

Why it failed, which is a finding rather than a weakness
---------------------------------------------------------
**Attacks built to be distributionally faithful are, by construction, invisible
to distributional anomaly detection.** Every foundry decision pushed the attacks
*toward* the legitimate manifold — clone real rows, resample amounts inside the
target MCC's own empirical band, redraw the hour from the population's diurnal
curve, keep provenance planting length-preserving — and the separability gate is
literally a rule forbidding any raw column from separating an attack above 0.95
AUC. An isolation forest measures distance from that manifold. We spent two days
minimising it.

That is also the property real agentic fraud has: an agent paying with a
validly-signed mandate, on a real cardholder's real device, for a plausible
amount at a real merchant, is legitimate in every marginal. The fraud lives in
the **intent** (L3, the text the agent ingested) and the **relations** (L4, the
entity graph), not in the marginals. See CLAUDE.md, "The zero-day answer,
reframed".

What answers an unseen attack instead: **L0** protocol invariants, which need no
training data because a violated mandate is a broken contract rather than an
outlier; and **the closed loop**, which manufactures the variant before an
attacker does and hands L1 a label for it.

The discipline that makes the claim true
------------------------------------------
It is easy to write "unsupervised" and then quietly use labels — to pick the
contamination parameter, to select features, to choose the threshold. Each of
those makes the layer supervised in a way the write-up will not mention. So:

* ``fit`` takes only rows where ``is_fraud`` is False, and asserts it.
* ``contamination`` is fixed, not tuned. Tuning it against measured recall would
  be label leakage through a hyperparameter.
* The threshold comes from the same legitimate-only quantile rule L1's does.

Isolation Forest rather than an autoencoder
--------------------------------------------
Both were on the table. Isolation Forest wins here for three reasons that are
about this problem rather than about the algorithms in general: it handles the
mixed numeric/categorical/NaN matrix without an imputation policy that would
itself need justifying; it trains in seconds on 200k rows with no GPU, which
HARD RULE 4 requires; and its anomaly score is a path length, which is
explainable to a judge in one sentence. An autoencoder's reconstruction error on
a matrix that is 40% NaN by design would mostly measure the NaN pattern.

The missingness problem, and the Day 4 measurement that settled it
-------------------------------------------------------------------
``mnd_*`` is NaN on every classic authorisation, because a classic authorisation
has no mandate. 109 of the 204 features are more than 30% missing on legitimate
traffic and 77 of them are more than 70% missing.

The first design filled those with a sentinel far below the observed range, on
the reasoning that "missing" is a fact and the forest should be able to split on
it. That reasoning was wrong, and specifically wrong for **this** algorithm.
Isolation Forest splits on a uniformly random threshold between a feature's min
and max. With a min of ``-1e9`` and a max near ``10``, essentially every random
threshold lands in the empty gap between the sentinel and the real data, so one
split perfectly separates missing from present — and the tree's whole depth
budget is consumed rediscovering which rail the authorisation was on. The forest
was isolating on the schema, not on behaviour.

So the columns are filtered by missingness and median-imputed instead. **The rule
is chosen on unsupervised grounds and is never tuned against recall**, which
matters: picking the feature set by "which one detects more fraud" would make
this layer supervised through the back door, and the leave-one-family-out table
would be quoting a lie.

What that bought, measured: recall@0.1%FPR went from **0.0006 to 0.0042**, a 7x
improvement on a number that is still approximately zero. That is the honest
summary of this layer, and it is reported in RESULTS.md rather than smoothed
over. Unsupervised novelty at a 0.1% false-positive budget does not work on this
feature space. What survives is the narrower and still-useful claim: L2's recall
is completely **unaffected** by whether an attack was in training, which is the
one property no supervised layer has.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import pandas as pd

__all__ = ["MIN_PRESENT_SHARE", "L2Model"]

#: Retained only where a feature is present on at least this share of legitimate
#: training rows. Above the cut a column is mostly a rail indicator wearing a
#: feature's name, and an Isolation Forest cannot help but split on it first.
#: Chosen from the missingness histogram alone -- never from measured recall.
MIN_PRESENT_SHARE: Final[float] = 0.70

#: Fixed, never tuned. Tuning it against measured recall would make the layer
#: supervised through the back door — see the module docstring.
CONTAMINATION: Final[float] = 0.01


@dataclass(slots=True)
class L2Model:
    """Isolation Forest over the legitimate population."""

    n_estimators: int = 300
    max_samples: int = 50_000
    seed: int = 1337
    forest: Any = None
    feature_names: list[str] = field(default_factory=list)
    #: Per-column medians from the legitimate training rows, used to impute.
    medians: Any = None
    threshold: float = float("nan")

    def _numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        """Dense numeric view: retained columns, categoricals as codes, median-filled."""
        columns = self.feature_names or list(X.columns)
        out = X[columns].copy()
        for column in out.columns:
            if str(out[column].dtype) == "category":
                out[column] = out[column].cat.codes.astype(float)
        out = out.astype(float).replace([np.inf, -np.inf], np.nan)
        return out.fillna(self.medians if self.medians is not None else out.median())

    def fit(self, X: pd.DataFrame, is_fraud: np.ndarray) -> L2Model:
        """Fit on legitimate rows only. Raises if a labelled fraud row is passed.

        The assertion is not defensive programming, it is the layer's entire
        claim. An L2 that had seen one fraud row would still be called
        unsupervised in the table and the table would be wrong.
        """
        from sklearn.ensemble import IsolationForest

        is_fraud = np.asarray(is_fraud, dtype=bool)
        if is_fraud.any():
            raise ValueError(
                f"L2Model.fit received {int(is_fraud.sum())} fraud rows. L2 is trained on "
                "legitimate traffic ONLY -- that is the property the leave-one-family-out "
                "table rests on. Filter before calling."
            )

        present = 1.0 - X.isna().mean()
        self.feature_names = [c for c in X.columns if present.get(c, 0.0) >= MIN_PRESENT_SHARE]
        if not self.feature_names:
            raise ValueError("no feature is present often enough to fit L2 on")
        self.medians = None
        numeric = self._numeric(X)
        self.medians = numeric.median()
        numeric = numeric.fillna(self.medians)
        self.forest = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=min(self.max_samples, len(numeric)),
            contamination=CONTAMINATION,
            random_state=self.seed,
            n_jobs=-1,
        )
        self.forest.fit(numeric)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Novelty score; higher is more anomalous.

        ``score_samples`` returns higher-is-more-normal, so it is negated. Doing
        that here rather than at every call site is what stops a sign error from
        turning the layer into an inverted detector that reports 1 - recall and
        looks merely disappointing.
        """
        if self.forest is None:
            raise RuntimeError("L2Model.score called before fit")
        return -np.asarray(self.forest.score_samples(self._numeric(X)), dtype=float)

    def fit_threshold(self, scores: np.ndarray, y: np.ndarray, fpr: float) -> float:
        """Operating threshold from the legitimate score quantile."""
        from mantis.defense.metrics import threshold_at_fpr

        self.threshold = threshold_at_fpr(scores, y, fpr)
        return self.threshold
