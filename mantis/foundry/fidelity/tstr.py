"""TSTR: train on synthetic, test on real, and report the gap.

Marginal distances say the synthetic data *looks* like the real data. TSTR asks
the only question that matters downstream: **does a detector trained on it learn
anything that is true of the real thing?** That is the number an issuer would ask
for, because it is the number that decides whether this foundry is a way to
bootstrap a model or an expensive way to draw plausible histograms.

The three models
----------------
=========  ==============================  ==========================
model      trained on                      tested on
=========  ==============================  ==========================
**TRTR**   real, first 70% by time         real, last 30% by time
**TSTR**   synthetic, all of it            real, last 30% by time
**TRTS**   real, first 70% by time         synthetic (reported for
                                           symmetry; see below)
=========  ==============================  ==========================

TRTR is the ceiling. TSTR divided by TRTR is the transfer ratio, and it is the
headline. TRTS is reported because a large asymmetry between the two directions
is itself informative: TSTR far below TRTS means the synthetic fraud is *easier*
than the real thing, which is the failure mode a generator falls into when its
attacks are cartoons.

What is being asked, stated precisely so it is not over-read
-------------------------------------------------------------
The reference panel's fraud is **classic card fraud** — a stolen card used at
merchants the cardholder never visits. This project's fraud is mostly *agentic*,
which the reference panel does not contain and no panel does. So the synthetic
training set here is restricted to the classic rails (see
:mod:`mantis.foundry.fidelity.common`), and the claim TSTR supports is the narrow
one:

    A detector trained only on MANTIS's synthetic **classic-rail** fraud, on
    features that carry no currency and no country, transfers to fraud in a panel
    it has never seen, at this fraction of the ceiling.

It is **not** evidence that the agentic attacks are realistic. Nothing can be, in
the absence of an agentic panel — and manufacturing that absence into a claim is
exactly what the zero-day section of ``RESULTS.md`` refuses to do. The scorecard
prints this paragraph's first sentence next to the number.

Metrics
-------
AUC-PR and ROC-AUC. No accuracy, ever — HARD RULE 2 — and here it would be
especially meaningless, since the two panels have different prevalence (1.0%
against 0.5%) and accuracy would mostly report which one was being tested on.
AUC-PR is quoted against each test set's own baseline, for the same reason.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import pandas as pd

from mantis.foundry.fidelity.common import SHAPE_FEATURES

__all__ = ["TSTR_CAVEAT", "tstr"]

#: Printed by the scorecard beside every TSTR figure. One sentence, because a
#: caveat nobody reads is a caveat that does not exist.
TSTR_CAVEAT: Final[str] = (
    "TSTR is measured on classic-rail fraud only. The reference panel contains no "
    "agentic transactions, so no number here is evidence about the agentic attacks."
)

#: Small and heavily regularised on purpose. The question is whether the signal
#: transfers, not how much capacity can be spent memorising one panel; a deep
#: forest would answer the second question and be reported as the first.
_PARAMS: Final[dict[str, Any]] = {
    "objective": "binary",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "verbose": -1,
}


def _fit(X: pd.DataFrame, y: np.ndarray, seed: int):
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(random_state=seed, **_PARAMS)
    model.fit(X[list(SHAPE_FEATURES)], y)
    return model


def _score(model, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    if len(np.unique(y)) < 2:
        return {"auc_pr": float("nan"), "roc_auc": float("nan"), "baseline": float("nan")}
    p = model.predict_proba(X[list(SHAPE_FEATURES)])[:, 1]
    return {
        "auc_pr": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "baseline": float(np.mean(y)),
        "n": float(len(y)),
        "n_pos": float(int(np.sum(y))),
    }


def tstr(
    synthetic_shape: pd.DataFrame,
    synthetic_labels: np.ndarray,
    real_shape: pd.DataFrame,
    real_labels: np.ndarray,
    *,
    seed: int = 1337,
) -> dict[str, Any]:
    """Fit the three models and return their scores plus the transfer ratio.

    The real panel is split **by time**, at its own 70% quantile, and the split
    index is computed on the row order the shape matrix was built in — which
    :func:`mantis.foundry.fidelity.common.to_common` guarantees is chronological.
    A random split would let a model see a cardholder's later transactions while
    scoring their earlier ones, and every velocity feature would leak.
    """
    real_labels = np.asarray(real_labels, dtype=bool)
    synthetic_labels = np.asarray(synthetic_labels, dtype=bool)

    cut = int(0.7 * len(real_shape))
    real_train, real_test = real_shape.iloc[:cut], real_shape.iloc[cut:]
    y_real_train, y_real_test = real_labels[:cut], real_labels[cut:]

    trtr_model = _fit(real_train, y_real_train, seed)
    tstr_model = _fit(synthetic_shape, synthetic_labels, seed)

    trtr = _score(trtr_model, real_test, y_real_test)
    tstr_scores = _score(tstr_model, real_test, y_real_test)
    trts = _score(trtr_model, synthetic_shape, synthetic_labels)

    ceiling = trtr["auc_pr"]
    ratio = float(tstr_scores["auc_pr"] / ceiling) if ceiling and ceiling > 0 else float("nan")

    # Lift over the baseline is the honest way to read a transfer ratio when the
    # two AUC-PRs sit on different prevalences: a model that has learned nothing
    # scores the baseline, not zero, so a raw ratio of 0.4 could still be a model
    # that learned nothing at all.
    def lift(scores: dict[str, float]) -> float:
        base = scores.get("baseline", float("nan"))
        return float(scores["auc_pr"] / base) if base and base > 0 else float("nan")

    return {
        "trtr": trtr,
        "tstr": tstr_scores,
        "trts": trts,
        "transfer_ratio": ratio,
        "trtr_lift": lift(trtr),
        "tstr_lift": lift(tstr_scores),
        "features": list(SHAPE_FEATURES),
        "caveat": TSTR_CAVEAT,
        "what_each_learned": {
            "trtr": _gains(trtr_model),
            "tstr": _gains(tstr_model),
        },
    }


def _gains(model) -> list[dict[str, float]]:
    """What a fitted model actually leaned on, as gain shares.

    Printed beside the TSTR figure because a transfer ratio near zero is
    uninterpretable on its own: it can mean the synthetic fraud is unrealistic,
    or it can mean the two panels' fraud are **different phenomena** that live in
    different features. These two rows distinguish those cases, and they are the
    difference between reporting a bad number and reporting a finding.
    """
    gains = np.asarray(model.booster_.feature_importance("gain"), dtype=float)
    total = gains.sum() or 1.0
    rows = [
        {"feature": name, "gain_share": float(g / total)}
        for name, g in zip(SHAPE_FEATURES, gains, strict=True)
    ]
    rows.sort(key=lambda row: -row["gain_share"])
    return rows
