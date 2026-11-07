"""The firewall experiment: five layers, weighted fusion, and two kinds of recall.

The headline result, unchanged from Day 4
-------------------------------------------
For each implemented family, retrain L1 with that family **entirely excluded from
training** and measure its recall anyway. Put that next to the recall L1 gets when
it *has* seen the family. One question:

    *What happens to your detector when the attacker does something you did not
    put in the training set?*

Most benchmarks avoid it by splitting at random. The answer here is a
measurement, reported whichever way it comes out — and Day 5 adds the other half
of the answer, which is :mod:`mantis.loop`.

The operating point, stated once
----------------------------------
Every recall is measured at a threshold placed so that a fixed share of
**legitimate test traffic** is flagged. Each model variant gets its own threshold,
because each has its own score distribution; what is held constant is the
false-positive rate, which is the thing an issuer budgets. Holding the *threshold*
constant instead would compare models at different FP rates and the columns would
not mean anything next to each other.

Day 5 reports it as a **curve** rather than a point — 0.1%, 0.5% and 1.0%
(:data:`~mantis.defense.metrics.FPR_GRID`). One number at one budget is a point a
reader has to trust you did not pick; three is a shape. 0.1% remains the headline
because it is the tightest, and the tightest is the one an issuer can actually
staff.

Two kinds of recall, both labelled
------------------------------------
Day 4 reported event-level recall only, and event-level recall is the wrong
question for half of this atlas. A mule ring that runs 40 authorisations and is
flagged on 3 of them scores 7.5% event-level and is **caught** — one alert opens
a case and the case takes the ring. So every layer is also reported at
**campaign level**: was the campaign flagged at all, and on which of its events.

Neither number is the "real" one. Event-level flatters a layer that fires on
every event of an obvious attack; campaign-level flatters a layer that fires once
on something subtle. They are printed side by side, always, with their names on
them.

Fusion, fixed
---------------
Day 4's unweighted noisy-OR made the ensemble **worse than L1 alone** (0.286 vs
0.361) by giving a near-random L2 equal say inside a fixed FP budget. Day 5
replaces it with a logistic stacker over layer percentiles, fitted on a slice of
the training window that none of the base layers was fitted on. See
:mod:`mantis.defense.fusion`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

from mantis.defense.features import FeatureBuilder
from mantis.defense.fusion import FusionModel
from mantis.defense.l1_gbdt import L1Model
from mantis.defense.l2_novelty import L2Model
from mantis.defense.l3_text import HELD_OUT_KIND, L3Model
from mantis.defense.l4_graph import EntityNovelty
from mantis.defense.metrics import (
    OPERATING_FPR,
    CampaignReport,
    ScoreReport,
    campaign_report,
    recall_at_fixed_threshold,
    recall_curve,
    score_report,
    threshold_at_fpr,
)
from mantis.defense.policy import Decision, PolicyThresholds, decide

__all__ = ["FAMILIES", "LAYER_ORDER", "ExperimentResult", "run_experiment"]

#: Share of the pooled chronology used for training.
TRAIN_SHARE: Final[float] = 0.70

#: Share of the *training* window the base layers are fitted on when producing
#: the out-of-sample scores the fusion weights are fitted on.
FUSION_INNER_SHARE: Final[float] = 0.80

#: Families with at least one implemented injector. F5 is deliberately absent:
#: it is the zero-day holdout family (CLAUDE.md §8).
FAMILIES: Final[tuple[str, ...]] = ("F1", "F2", "F3", "F4", "F6")

#: Layers reported, in the order they appear in every table.
LAYER_ORDER: Final[tuple[str, ...]] = ("L1", "L2", "L2e", "L3", "fused")


@dataclass(slots=True)
class LayerResult:
    """One layer's scores and every reading taken off them."""

    name: str
    scores: np.ndarray
    report: ScoreReport
    curve: dict[float, tuple[float, float]]
    campaigns: CampaignReport
    threshold: float


@dataclass(slots=True)
class ExperimentResult:
    """Everything the CLI and RESULTS.md need, computed once."""

    n_train: int
    n_test: int
    n_features: int
    prevalence: float
    layers: dict[str, LayerResult] = field(default_factory=dict)
    l1_rail: dict[str, ScoreReport] = field(default_factory=dict)
    per_family: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_family_campaign: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_attack: pd.DataFrame = field(default_factory=pd.DataFrame)
    importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    ablation: pd.DataFrame = field(default_factory=pd.DataFrame)
    fusion_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    l3_holdout: pd.DataFrame = field(default_factory=pd.DataFrame)
    l3_cards: pd.DataFrame = field(default_factory=pd.DataFrame)
    graph_features: int = 0
    decisions: dict[str, int] = field(default_factory=dict)
    explanation: str = ""

    @property
    def l1_full(self) -> ScoreReport:
        return self.layers["L1"].report

    @property
    def fused(self) -> ScoreReport:
        return self.layers["fused"].report


def _layer_result(
    name: str,
    scores: np.ndarray,
    y: np.ndarray,
    campaigns: np.ndarray,
    order: np.ndarray,
    fpr: float,
) -> LayerResult:
    """Package one score vector into every reading the report needs."""
    floor = float(np.nanmin(scores)) if np.isfinite(scores).any() else 0.0
    filled = np.nan_to_num(scores, nan=floor)
    report = score_report(filled, y, fpr)
    return LayerResult(
        name=name,
        scores=filled,
        report=report,
        curve=recall_curve(filled, y),
        campaigns=campaign_report(filled, y, campaigns, order, report.threshold),
        threshold=report.threshold,
    )


def run_experiment(
    pool: pd.DataFrame,
    *,
    seed: int = 1337,
    fpr: float = OPERATING_FPR,
    verbose: bool = True,
) -> ExperimentResult:
    """Fit every layer, fuse them, then run leave-one-family-out."""

    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    pool = pool.sort_values("ts", kind="stable").reset_index(drop=True)
    cut = pool["ts"].quantile(TRAIN_SHARE)
    train_mask = pool["ts"] <= cut

    log(f"  time split at {cut:%Y-%m-%d}: train {int(train_mask.sum()):,} / "
        f"test {int((~train_mask).sum()):,}")

    log("  building features (one continuous pass over the pooled chronology)...")
    builder = FeatureBuilder()
    X = builder.fit_transform_stream(pool, train_mask)
    y = pool["is_fraud"].to_numpy(dtype=bool)
    family = pool["attack_id"].fillna("").str.slice(0, 2).to_numpy()

    tr = train_mask.to_numpy()
    te = ~tr
    X_tr, X_te = X[tr], X[te]
    # The full matrix is ~2 GB at a million rows and is not needed once it has
    # been split; the slices above are copies. Holding all three at once is what
    # turns a 15-minute run into a swap-thrashing one on a 16 GB laptop, and
    # HARD RULE 4 says this has to run on a judge's machine.
    n_features = X.shape[1]
    n_graph_features = sum(1 for c in X.columns if c.startswith("gph_"))
    del X
    y_tr, y_te = y[tr], y[te]
    fam_te = family[te]
    ts_tr = pool.loc[tr, "ts"]
    campaigns_te = pool.loc[te, "attack_campaign"].to_numpy()
    order_te = pool.loc[te, "ts"].astype("int64").to_numpy() / 1e9
    log(f"  {n_features} features; train fraud {int(y_tr.sum()):,}, "
        f"test fraud {int(y_te.sum()):,}")

    # -- L1, trained on everything --------------------------------------------- #
    log("  fitting L1 (all families)...")
    l1 = L1Model(seed=seed).fit(X_tr, y_tr, timestamps=ts_tr)
    s1 = l1.score(X_te)

    # -- L2, trained on legitimate traffic only --------------------------------- #
    log("  fitting L2 (LEGITIMATE ROWS ONLY; residual monitor, not a detector)...")
    legit_tr = X_tr[~y_tr]
    l2 = L2Model(seed=seed).fit(legit_tr, np.zeros(len(legit_tr), dtype=bool))
    del legit_tr
    s2 = l2.score(X_te)

    # -- L2e, the entity-level version of the same hypothesis -------------------- #
    log("  fitting L2e (entity-level novelty; TIME-BOXED EXPERIMENT)...")
    l2e = EntityNovelty(seed=seed).fit(pool[tr])
    s2e = l2e.score(pool[te])

    # -- L3, fitted on TEXT, never on transaction labels ------------------------- #
    log("  fitting L3 (page classifier; no transaction labels, held-out phrasings)...")
    l3 = L3Model(seed=seed).fit()
    s3 = l3.score(pool[te])

    # -- fusion: weights fitted where the base layers cannot see ----------------- #
    log("  fitting the fusion weights on an inner split of the training window...")
    inner_cut = ts_tr.quantile(FUSION_INNER_SHARE)
    inner = (ts_tr <= inner_cut).to_numpy()
    fusion_rows = ~inner
    X_inner = X_tr[inner]
    l1_inner = L1Model(seed=seed).fit(X_inner, y_tr[inner], timestamps=ts_tr[inner])
    legit_inner = X_inner[~y_tr[inner]]
    l2_inner = L2Model(seed=seed).fit(legit_inner, np.zeros(len(legit_inner), dtype=bool))
    del X_inner, legit_inner
    pool_tr = pool[tr]
    l2e_inner = EntityNovelty(seed=seed).fit(pool_tr[inner])

    X_fusion = X_tr[fusion_rows]
    fusion = FusionModel(seed=seed).fit(
        {
            "L1": l1_inner.score(X_fusion),
            "L2": l2_inner.score(X_fusion),
            "L2e": l2e_inner.score(pool_tr[fusion_rows]),
            "L3": l3.score(pool_tr[fusion_rows]),
        },
        y_tr[fusion_rows],
    )
    del X_fusion, l1_inner, l2_inner, l2e_inner
    fused = fusion.score({"L1": s1, "L2": s2, "L2e": s2e, "L3": s3})

    layers = {
        name: _layer_result(name, scores, y_te, campaigns_te, order_te, fpr)
        for name, scores in (
            ("L1", s1), ("L2", s2), ("L2e", s2e), ("L3", s3), ("fused", fused)
        )
    }

    # -- per-rail L1, because the headline is partly measuring "is this agentic" -- #
    rail_reports: dict[str, ScoreReport] = {}
    is_agentic = pool["ag_agent_id"].notna().to_numpy()[te]
    for name, mask in (("agentic", is_agentic), ("classic", ~is_agentic)):
        if mask.sum() and y_te[mask].any():
            rail_reports[name] = score_report(layers["L1"].scores[mask], y_te[mask], fpr)

    # -- THE HEADLINE: leave one family out -------------------------------------- #
    rows: list[dict[str, object]] = []
    campaign_rows: list[dict[str, object]] = []
    for fam in FAMILIES:
        in_family = fam_te == fam
        n_pos = int(in_family.sum())
        if n_pos == 0:
            continue
        log(f"  leave-one-family-out: retraining L1 without {fam} "
            f"({int((family[tr] == fam).sum()):,} training positives removed)...")

        keep = ~((family[tr] == fam) & y_tr)
        l1_held = L1Model(seed=seed).fit(X_tr[keep], y_tr[keep], timestamps=ts_tr[keep])
        s_held = l1_held.score(X_te)
        held_threshold = threshold_at_fpr(s_held, y_te, fpr)

        fused_held = fusion.score({"L1": s_held, "L2": s2, "L2e": s2e, "L3": s3})
        fused_held_threshold = threshold_at_fpr(fused_held, y_te, fpr)

        rows.append(
            {
                "family": fam,
                "n_pos": n_pos,
                "l1_with": recall_at_fixed_threshold(layers["L1"].scores, in_family,
                                                     layers["L1"].threshold),
                "l1_heldout": recall_at_fixed_threshold(s_held, in_family, held_threshold),
                "l2": recall_at_fixed_threshold(layers["L2"].scores, in_family,
                                                layers["L2"].threshold),
                "l3": recall_at_fixed_threshold(layers["L3"].scores, in_family,
                                                layers["L3"].threshold),
                "fused_with": recall_at_fixed_threshold(layers["fused"].scores, in_family,
                                                        layers["fused"].threshold),
                "fused_heldout": recall_at_fixed_threshold(fused_held, in_family,
                                                           fused_held_threshold),
            }
        )
        fam_campaigns = np.where(in_family, campaigns_te, None)
        with_report = campaign_report(
            layers["fused"].scores, in_family, fam_campaigns, order_te,
            layers["fused"].threshold,
        )
        held_report = campaign_report(
            fused_held, in_family, fam_campaigns, order_te, fused_held_threshold
        )
        campaign_rows.append(
            {
                "family": fam,
                "n_campaigns": with_report.n_campaigns,
                "median_size": with_report.median_size,
                "fused_with": with_report.recall,
                "fused_heldout": held_report.recall,
                "median_index": with_report.median_index,
                "share_before_alert": with_report.median_share_before_alert,
            }
        )
    per_family = pd.DataFrame(rows)
    per_family_campaign = pd.DataFrame(campaign_rows)

    # -- per attack card, at the full model's operating point --------------------- #
    attack_te = pool["attack_id"].fillna("").to_numpy()[te]
    attack_rows = []
    for card in sorted({a for a in attack_te if a}):
        mask = attack_te == card
        row = {"attack_id": card, "n_pos": int(mask.sum())}
        for name in LAYER_ORDER:
            row[name] = recall_at_fixed_threshold(
                layers[name].scores, mask, layers[name].threshold
            )
        card_campaigns = np.where(mask, campaigns_te, None)
        row["campaign"] = campaign_report(
            layers["fused"].scores, mask, card_campaigns, order_te, layers["fused"].threshold
        ).recall
        attack_rows.append(row)
    per_attack = pd.DataFrame(attack_rows)

    # -- L3's generalisation, measured directly ---------------------------------- #
    log("  measuring L3 against an entirely held-out adversarial kind...")
    l3_strict = L3Model(seed=seed).fit(hold_out_kind=HELD_OUT_KIND)
    s3_strict = np.nan_to_num(l3_strict.score(pool[te]), nan=0.0)
    strict_threshold = threshold_at_fpr(s3_strict, y_te, fpr)
    holdout_mask = l3.holdout_mask(pool[te])
    l3_cards = pd.DataFrame(
        [
            {
                "attack_id": card,
                "n_pos": int((attack_te == card).sum()),
                "recall": recall_at_fixed_threshold(
                    layers["L3"].scores, attack_te == card, layers["L3"].threshold
                ),
                "recall_unseen_phrasing": recall_at_fixed_threshold(
                    layers["L3"].scores, (attack_te == card) & holdout_mask,
                    layers["L3"].threshold,
                ),
                "n_unseen_phrasing": int(((attack_te == card) & holdout_mask).sum()),
                "recall_unseen_kind": recall_at_fixed_threshold(
                    s3_strict, attack_te == card, strict_threshold
                ),
            }
            for card in sorted({a for a in attack_te if a})
        ]
    )
    l3_cards = l3_cards[l3_cards["recall"] > 0.01].reset_index(drop=True)

    # -- the decision layer ------------------------------------------------------- #
    thresholds = PolicyThresholds.fit(layers["fused"].scores, y_te)
    actions = decide(
        layers["fused"].scores,
        thresholds,
        # Object dtype off the agentic rail; ``fillna`` on it warns about a future
        # downcast, so the cast is explicit. A classic authorisation has a human
        # at the terminal by definition, which is why the fill value is True.
        human_present=pool["ag_human_present"].to_numpy()[te] != False,  # noqa: E712
    )
    decisions = {d.value: int((actions == d).sum()) for d in Decision}

    # -- explanation, from LightGBM's own contributions ---------------------------- #
    from mantis.defense.explain import explain_events

    explanation = explain_events(l1, X_te, pool[te], layers["L1"].scores, limit=3)

    # -- the ablation the Day 4 review demanded ------------------------------------ #
    log("  ablating mnd_deliberation_residual_z...")
    ablation = _ablate(X_tr, y_tr, X_te, y_te, fam_te, ts_tr, seed, fpr)

    return ExperimentResult(
        n_train=int(tr.sum()),
        n_test=int(te.sum()),
        n_features=n_features,
        prevalence=float(y_te.mean()),
        layers=layers,
        l1_rail=rail_reports,
        per_family=per_family,
        per_family_campaign=per_family_campaign,
        per_attack=per_attack,
        importance=l1.importance(20),
        ablation=ablation,
        fusion_weights=fusion.weights(),
        l3_holdout=l3.holdout_generalisation(),
        l3_cards=l3_cards,
        graph_features=n_graph_features,
        decisions=decisions,
        explanation=explanation,
    )


#: The feature the Day 4 review found separating F1-01 at 0.99 AUC on its own.
#: See ``RESULTS.md`` and ``features/mandate.py``.
ABLATED_FEATURE: Final[str] = "mnd_deliberation_residual_z"


def _ablate(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_te: pd.DataFrame,
    y_te: np.ndarray,
    fam_te: np.ndarray,
    ts_tr: pd.Series,
    seed: int,
    fpr: float,
) -> pd.DataFrame:
    """Refit L1 without the one feature that is too good, and report the delta.

    ``mnd_deliberation_residual_z`` separates F1-01 at 0.99 AUC by itself, which
    is above the foundry's own 0.95 separability gate. The gate never saw it
    because the gate probes **raw columns** and this is a derived residual, so a
    trivially-derived feature walked straight past it. Reporting F1's recall both
    with and without the feature is the honest way to show how much of that
    recall rests on a generator artefact rather than on the attack.
    """
    if ABLATED_FEATURE not in X_tr.columns:
        return pd.DataFrame()
    columns = [c for c in X_tr.columns if c != ABLATED_FEATURE]
    model = L1Model(seed=seed).fit(X_tr[columns], y_tr, timestamps=ts_tr)
    scores = model.score(X_te[columns])
    threshold = threshold_at_fpr(scores, y_te, fpr)

    rows = []
    for fam in FAMILIES:
        mask = fam_te == fam
        if mask.sum():
            rows.append(
                {
                    "family": fam,
                    "n_pos": int(mask.sum()),
                    "recall_without": recall_at_fixed_threshold(scores, mask, threshold),
                }
            )
    return pd.DataFrame(rows)

