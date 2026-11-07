"""The adversarial test: can a model tell our rows from real ones?

Every other metric in this package compares one statistic at a time. A generator
can pass all of them and still be obviously synthetic, because "obviously" lives
in the interactions — the combination of a 3 a.m. timestamp, a small ticket, and a
merchant nobody uses is what gives a file away, and no marginal contains it.

So: label the synthetic rows 1, the real rows 0, fit a gradient-boosted tree, and
score it out of fold. **The target is 0.5** — an AUC of 0.5 means the classifier
cannot do better than a coin, which is the only outcome that says the two panels
are indistinguishable on these features. 1.0 means the file announces itself.

Read it as a distance, not a grade
-----------------------------------
``2 * |AUC - 0.5|`` is the total-variation-like separation, and it is reported
beside the raw AUC because "0.62" is hard to place and "24% separable" is not.

The per-feature gain table is the part with the most information in it. A
discriminator that lands above 0.5 is *telling you which column gave you away*,
ranked, which is a to-do list rather than a verdict. That list is the reason this
module exists at the end of a scorecard rather than the beginning: it names the
next thing to fix, and this project's rule is to publish that list instead of
tuning against it three days before submission.

Two honesty constraints on the setup
-------------------------------------
1. **Balanced by subsampling, not by class weight.** With unequal panel sizes a
   classifier can reach a high AUC by learning the prior, and the AUC would then
   partly report how many rows each side happened to contribute.
2. **Out-of-fold scores only.** An in-sample AUC on a boosted tree is near 1.0
   whatever the data looks like, and reporting it would turn this section into a
   measure of model capacity.

What the discriminator is NOT allowed to see, and why saying so matters
------------------------------------------------------------------------
A discriminator handed the raw columns of two payment files separates them at
1.0000 and tells you nothing. Identifiers are namespaced differently on each
side; the reference panel predates agentic commerce, so **every row carrying an
``ag_`` block is definitionally separable**; and an amount in rupees is not an
amount in dollars. Counting any of that as infidelity would be the same error
``BaseAttack.probe_slice`` fixed on Day 3 — measuring a property that *defines*
the two populations rather than one that distinguishes their behaviour.

So the input is the shape space of :mod:`mantis.foundry.fidelity.common`: the
intersection of what both panels carry, identity-shaped columns excluded by
construction, the agentic rail dropped from the synthetic side, and every
survivor made dimensionless. :func:`discriminate_naive` measures what skipping
that discipline would have produced, so the difference is documented rather than
asserted.

Interpreting the number, which is the part that was missing
------------------------------------------------------------
An AUC alone is a verdict without a reason. Three things are reported beside it:

* **Attribution** — native LightGBM ``pred_contrib``, the same quantity
  ``TreeExplainer`` computes, ranked by mean absolute contribution. Same choice
  and same reasoning as ``mantis/defense/explain/contributions.py``.
* **A greedy ablation path** — drop the strongest feature, refit, re-score, and
  repeat. A separation concentrated in one column collapses immediately; one
  distributed across a joint distribution does not. Which of those is true
  changes what the number means and what would fix it.
* **A cosmetic/structural verdict per feature** — see :data:`FEATURE_CLASS`.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import pandas as pd

from mantis.foundry.fidelity.common import SHAPE_FEATURES

__all__ = [
    "FEATURE_CLASS",
    "TARGET_AUC",
    "ablation_path",
    "discriminate",
    "discriminate_naive",
]

#: What an indistinguishable pair of panels scores. Plotted as a line on the
#: scorecard's figure, because a bar chart of AUCs with no reference line invites
#: a reader to think higher is better, and here higher is worse.
TARGET_AUC: Final[float] = 0.5

_PARAMS: Final[dict[str, Any]] = {
    "objective": "binary",
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "verbose": -1,
}


#: Cosmetic or structural, per shape feature, with the reason.
#:
#: The distinction is the one that decides whether a divergence is worth fixing
#: and how expensive fixing it would be:
#:
#: **cosmetic** — a surface property of how values were *rendered* rather than of
#: how the population behaves. Identifier shape, timestamp granularity, amount
#: rounding. Changing it moves the discriminator without changing a single thing
#: an attacker or a detector would experience.
#:
#: **structural** — a property of the joint distribution. Which merchants get the
#: volume, how a cardholder's transactions cluster in time, how amount relates to
#: category. Fixing one of these means changing the generative model, which is
#: why none of them was touched three days before submission.
#:
#: Nothing here is a fix and nothing here is an excuse. A feature can be
#: structural *and* adjudicated to the reference panel — ``merchant_rank_pct`` is
#: both, and the two statements are independent: the divergence is real and
#: joint, and the reference is nonetheless the side that departs from what an
#: acceptance estate looks like.
FEATURE_CLASS: Final[dict[str, tuple[str, str]]] = {
    "merchant_rank_pct": (
        "structural",
        "Which merchants carry the volume. Ours is Zipf (top 10% take 66%), the "
        "reference is near-uniform (14.6%). A property of the generative model, "
        "not of formatting -- and separately adjudicated to the reference, "
        "because real acceptance estates are heavy-tailed.",
    ),
    "hour": (
        "cosmetic",
        "Timestamp granularity at the coarsest scale: which hour a transaction "
        "is stamped with. The reference has no diurnal curve at all, ours does. "
        "Nothing about behaviour changes if the curve is reshaped -- it is when "
        "the generator chose to place events, and it is adjudicated to the "
        "reference for having no curve to compare against.",
    ),
    "gap_ratio_log": (
        "structural",
        "How a cardholder's transactions cluster in time relative to their own "
        "rhythm. Burstiness is a behavioural property and the arrival process "
        "that produces it is part of the generative model.",
    ),
    "burst_1h": (
        "structural",
        "The same arrival process read as a probability. 12.4% of our events "
        "follow another within the hour against 16.6% of the reference's.",
    ),
    "amount_vs_customer": (
        "structural",
        "How a ticket sits against the cardholder's own history -- a joint "
        "property of the amount draw and the customer assignment, and the single "
        "most transferable fraud signal there is.",
    ),
    "log_amount_z": (
        "cosmetic",
        "Within-category amount dispersion, and the residual separation here is "
        "**rounding**: the population snaps amounts to round numbers, and F6-40's "
        "injector snaps harder than a real ring would. The Day 5 derived probe "
        "already flagged txn_round_score at 0.966 as an artefact for exactly this "
        "reason. Softening the snap is a recorded outstanding item.",
    ),
    "dow": (
        "cosmetic",
        "Day-of-week placement. Same class as `hour`: calendar rendering, not "
        "behaviour.",
    ),
    "category_shift": (
        "structural",
        "Whether consecutive transactions change category -- basket volatility, "
        "a joint property of the category draw and the per-customer sequence. "
        "The distance is 0.0005 JSD, so it is structural and negligible.",
    ),
}


def classify(feature: str) -> tuple[str, str]:
    """``(verdict, reason)`` for a shape feature, or ``("unclassified", "")``."""
    return FEATURE_CLASS.get(feature, ("unclassified", ""))


def discriminate(
    synthetic_shape: pd.DataFrame,
    real_shape: pd.DataFrame,
    *,
    seed: int = 1337,
    folds: int = 5,
    exclude: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Fit the real-vs-synthetic classifier and return its out-of-fold AUC.

    ``exclude`` drops named features before fitting. The scorecard runs this
    twice: once on everything, and once without the axes
    :mod:`mantis.foundry.fidelity.adjudicate` attributes to the reference panel.
    Reporting only the second would be the dishonest version - the ablation is a
    judgement, so both numbers are printed and the judgement is printed with the
    measurement that decided it.

    Also returns the same measurement **per feature** — one univariate model per
    column — because the multivariate gain table tells you which feature the tree
    *used* and the univariate table tells you how separable each column is on its
    own. A feature can score high on the second and near zero on the first, which
    means another column already carries the same tell.
    """
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.default_rng(seed)
    n = min(len(synthetic_shape), len(real_shape))
    a = synthetic_shape.iloc[rng.choice(len(synthetic_shape), n, replace=False)]
    b = real_shape.iloc[rng.choice(len(real_shape), n, replace=False)]

    columns = [name for name in SHAPE_FEATURES if name not in exclude]
    X = pd.concat([a[columns], b[columns]], ignore_index=True)
    y = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])

    oof = np.zeros(len(y), dtype=float)
    gains = np.zeros(len(columns), dtype=float)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_index, test_index in splitter.split(X, y):
        model = LGBMClassifier(random_state=seed, **_PARAMS)
        model.fit(X.iloc[train_index], y[train_index])
        oof[test_index] = model.predict_proba(X.iloc[test_index])[:, 1]
        gains += np.asarray(model.booster_.feature_importance("gain"), dtype=float)

    auc = float(roc_auc_score(y, oof))
    gains = gains / gains.sum() if gains.sum() > 0 else gains

    per_feature = []
    for index, name in enumerate(columns):
        # Univariate separability, by rank rather than by fitting 10 more models:
        # a single feature's ROC-AUC against the panel label *is* the Mann-Whitney
        # statistic, so there is nothing a stump would add.
        column_auc = float(roc_auc_score(y, X[name].to_numpy(dtype=float)))
        per_feature.append(
            {
                "feature": name,
                "alone_auc": max(column_auc, 1.0 - column_auc),
                "gain_share": float(gains[index]),
            }
        )
    per_feature.sort(key=lambda row: -row["alone_auc"])

    # Native contributions from the last fold's model, on its own held-out rows.
    # Same computation TreeExplainer performs, read from the source it would have
    # called -- see mantis/defense/explain/contributions.py for the argument.
    contributions = np.asarray(
        model.booster_.predict(X.iloc[test_index], pred_contrib=True), dtype=float
    )[:, :-1]
    mean_abs = np.abs(contributions).mean(axis=0)
    total_abs = mean_abs.sum() or 1.0
    for index, row in enumerate(
        sorted(per_feature, key=lambda item: columns.index(item["feature"]))
    ):
        row["mean_abs_contribution"] = float(mean_abs[index])
        row["contribution_share"] = float(mean_abs[index] / total_abs)
    per_feature.sort(key=lambda row: -row["contribution_share"])

    return {
        "auc": auc,
        "target": TARGET_AUC,
        "separability": float(2.0 * abs(auc - TARGET_AUC)),
        "n_per_side": int(n),
        "folds": folds,
        "features": columns,
        "excluded": list(exclude),
        "per_feature": per_feature,
        "top_features": [row["feature"] for row in per_feature[:5]],
        "reading": _reading(auc),
    }


def ablation_path(
    synthetic_shape: pd.DataFrame,
    real_shape: pd.DataFrame,
    *,
    seed: int = 1337,
    folds: int = 3,
) -> list[dict[str, Any]]:
    """Drop the strongest feature, refit, re-score. Repeat until two remain.

    This is the measurement that turns an AUC into a diagnosis. If the separation
    lives in one column, the first drop collapses it and the fix is that column.
    If it survives to the bottom of the list, the separation is in the **joint**
    distribution and no single feature is the problem -- which means the
    correlation-matrix distance is the number to work against, and the fix is
    generative rather than cosmetic.

    Greedy on contribution share rather than exhaustive: the point is a shape, and
    an exhaustive subset search is 255 fits to say the same thing.

    ``folds=3`` rather than 5 because this fits once per step and the ranking is
    stable at three; the headline AUC is still the 5-fold one.
    """
    path: list[dict[str, Any]] = []
    dropped: list[str] = []
    remaining = list(SHAPE_FEATURES)

    while len(remaining) > 2:
        result = discriminate(
            synthetic_shape, real_shape, seed=seed, folds=folds, exclude=tuple(dropped)
        )
        strongest = result["per_feature"][0]["feature"]
        path.append(
            {
                "dropped_so_far": list(dropped),
                "n_features": len(remaining),
                "auc": result["auc"],
                "separability": result["separability"],
                "next_to_drop": strongest,
            }
        )
        dropped.append(strongest)
        remaining.remove(strongest)

    final = discriminate(
        synthetic_shape, real_shape, seed=seed, folds=folds, exclude=tuple(dropped)
    )
    path.append(
        {
            "dropped_so_far": list(dropped),
            "n_features": len(remaining),
            "auc": final["auc"],
            "separability": final["separability"],
            "next_to_drop": None,
        }
    )
    return path


#: Raw columns both panels carry, used only to show what NOT projecting costs.
#: ``customer_id`` and ``merchant_id`` are identity-shaped and are in this list
#: **on purpose** -- their whole role is to demonstrate the artefact.
_NAIVE_COLUMNS: Final[tuple[str, ...]] = (
    "amount",
    "hour",
    "category_code",
    "customer_code",
    "merchant_code",
)


def discriminate_naive(
    synthetic_common: pd.DataFrame,
    real_common: pd.DataFrame,
    *,
    seed: int = 1337,
    folds: int = 3,
) -> dict[str, Any]:
    """The discriminator with **none** of the discipline, as a documented contrast.

    Raw amount in two different currencies, raw category codes from two different
    taxonomies, and factorised identifiers from two different namespaces. This is
    what a fidelity number looks like when nobody has thought about what the two
    files have in common, and it exists so that the headline number's constraints
    are visible as a measured difference rather than as a claim in a docstring.

    An AUC near 1.0 here is the **expected** result and is not a finding about the
    population. The finding, if there is one, is how little the disciplined number
    improves on it -- which is exactly the comparison a reader should be able to
    make.
    """
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    def _frame(common: pd.DataFrame) -> pd.DataFrame:
        ts = pd.to_datetime(common["ts"])
        return pd.DataFrame(
            {
                "amount": common["amount"].to_numpy(dtype=float),
                "hour": ts.dt.hour.to_numpy(),
                "category_code": pd.factorize(common["category"])[0],
                "customer_code": pd.factorize(common["customer_id"])[0],
                "merchant_code": pd.factorize(common["merchant_id"])[0],
            }
        )

    rng = np.random.default_rng(seed)
    n = min(len(synthetic_common), len(real_common))
    a = _frame(synthetic_common.iloc[rng.choice(len(synthetic_common), n, replace=False)])
    b = _frame(real_common.iloc[rng.choice(len(real_common), n, replace=False)])

    X = pd.concat([a, b], ignore_index=True)[list(_NAIVE_COLUMNS)]
    y = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])

    oof = np.zeros(len(y), dtype=float)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_index, test_index in splitter.split(X, y):
        model = LGBMClassifier(random_state=seed, **_PARAMS)
        model.fit(X.iloc[train_index], y[train_index])
        oof[test_index] = model.predict_proba(X.iloc[test_index])[:, 1]

    auc = float(roc_auc_score(y, oof))
    alone = {
        name: max(
            float(roc_auc_score(y, X[name].to_numpy(dtype=float))),
            1.0 - float(roc_auc_score(y, X[name].to_numpy(dtype=float))),
        )
        for name in _NAIVE_COLUMNS
    }
    return {
        "auc": auc,
        "columns": list(_NAIVE_COLUMNS),
        "alone_auc": alone,
        "note": (
            "Raw columns, two currencies, two taxonomies, two identifier namespaces. "
            "An AUC near 1.0 is expected and is not a statement about the population."
        ),
    }


def _reading(auc: float) -> str:
    """Prose picked from the number, so a future run that gets worse says so.

    Same discipline as ``mantis/loop/report.py``'s curve reading: the text is
    derived from the measurement rather than written once against the measurement
    that happened to come out first.
    """
    separability = 2.0 * abs(auc - TARGET_AUC)
    if auc < 0.55:
        return (
            "Indistinguishable on the shape features: a classifier given both panels "
            "cannot beat a coin flip."
        )
    if auc < 0.70:
        return (
            f"Partly separable ({separability:.0%}). The per-feature table below names "
            "which columns carry the tell; every one of them is a foundry to-do."
        )
    if auc < 0.90:
        return (
            f"Clearly separable ({separability:.0%}). The two panels differ in ways a "
            "model finds easily, and the top feature below is where to look first."
        )
    return (
        f"Separable at {separability:.0%}. On these features the synthetic population "
        "is not passing for the reference panel, and the scorecard says so."
    )
