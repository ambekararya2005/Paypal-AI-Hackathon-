"""Entity-level novelty — L2's one remaining chance, taken and measured.

The hypothesis
---------------
L2 scores one authorisation at a time and Day 4 measured it at 0.4% mean
per-family recall. The stated reason it should have worked better is a ring:
every event in a mule network is bland — that is exactly what the foundry's 0.95
separability gate enforces — but the *entity* is not. A customer who touches
forty merchants in a week through one device, or a merchant who collects from
sixty payers who are all in the same identity component, is an outlier at the
level of the entity even when none of their events is an outlier at the level of
the row.

So: aggregate to the entity, fit an isolation forest on **entity vectors** rather
than event vectors, and see whether the number moves.

The discipline, unchanged
---------------------------
No labels, anywhere. The forest is fitted on every entity in the training
window — an issuer does not know which of its customers are mules, so filtering
to "legitimate entities" would be a supervision the deployment does not have.
``contamination`` is fixed at L2's value and never tuned against recall.

The caveat that must travel with the number
---------------------------------------------
This is a **batch entity-review score, not an authorisation score.** An entity's
vector is aggregated over the whole scoring window, so the score attached to that
entity's first event was computed with knowledge of their last. A real-time
authorisation scorer cannot do that; a nightly entity-risk queue can, and that is
a real deployment mode — it is what an issuer's AML and merchant-monitoring teams
actually run. The number is therefore **generous** relative to L1's per-event
operating point, and the comparison is only honest with that said out loud. It is
said out loud here, in the CLI output, and in RESULTS.md.

If it does not move the needle even with that advantage, the negative result is
worth more than the layer would have been: it would mean the ring is not
distributionally unusual at *any* aggregation level, which is a much stronger
statement about what distributional anomaly detection can do against attacks
built to be distributionally faithful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import pandas as pd

from mantis.core.events import DECLINE_RESPONSES
from mantis.defense.l2_novelty.model import CONTAMINATION

__all__ = ["ENTITY_KEYS", "EntityNovelty", "entity_vectors"]

#: The two entity types scored. Customer is the payer side (F2-13 synthetic
#: identity, F2-16 bust-out, F1-05 delegation laundering); merchant is the
#: beneficiary side (F6-38 fan-in, F6-39 shell merchant, F6-40 cash-out).
ENTITY_KEYS: Final[tuple[str, ...]] = ("customer_id", "merchant_id")

#: Entities with fewer events than this are not scored. A one-event customer has
#: no aggregate worth taking an anomaly score of, and including them would fill
#: the tail of the score distribution with noise that then eats the FP budget.
MIN_EVENTS: Final[int] = 3


def _entropy(counts: np.ndarray) -> float:
    """Shannon entropy of a count vector, in nats. Zero for a single value."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def entity_vectors(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    """One row per entity: how that entity behaved over ``frame``.

    Deliberately *not* the mean of the event feature matrix. An entity is
    characterised by its **spread** — how many distinct merchants, how
    concentrated its hours, how bursty its gaps — and averaging a per-event
    matrix destroys exactly that.
    """
    work = pd.DataFrame(
        {
            "key": frame[key].astype(str),
            "amount": frame["amount"].to_numpy(dtype=float),
            "ts": frame["ts"].dt.tz_localize(None).astype("int64") / 1e9,
            "hour": frame["ts"].dt.hour.to_numpy(),
            "declined": np.isin(frame["auth_response"].to_numpy(), DECLINE_RESPONSES),
            "agentic": frame["ag_agent_id"].notna().to_numpy(),
            "outbound": np.isin(
                frame["txn_type"].to_numpy(), ("refund", "reversal", "credit")
            ),
            "merchant": frame["merchant_id"].astype(str),
            "customer": frame["customer_id"].astype(str),
            "device": frame["device_id"].astype(str),
            "mcc": frame["mcc"].astype(str),
            "card_bin": frame["card_bin"].astype(str),
        }
    )
    grouped = work.groupby("key", observed=True, sort=True)

    out = pd.DataFrame(index=grouped.size().index)
    out["n_events"] = grouped.size()
    out["amount_mean"] = grouped["amount"].mean()
    out["amount_std"] = grouped["amount"].std()
    out["amount_max"] = grouped["amount"].max()
    out["amount_sum"] = grouped["amount"].sum()
    out["amount_cv"] = out["amount_std"] / out["amount_mean"].replace(0.0, np.nan)
    out["decline_ratio"] = grouped["declined"].mean()
    out["agentic_share"] = grouped["agentic"].mean()
    out["outbound_share"] = grouped["outbound"].mean()

    counter = "merchant" if key == "customer_id" else "customer"
    out["n_counterparties"] = grouped[counter].nunique()
    out["n_devices"] = grouped["device"].nunique()
    out["n_mccs"] = grouped["mcc"].nunique()
    out["n_bins"] = grouped["card_bin"].nunique()
    out["counterparty_ratio"] = out["n_counterparties"] / out["n_events"]
    out["device_ratio"] = out["n_devices"] / out["n_events"]

    # Timing shape: an entity's burstiness is the thing a per-event model has the
    # hardest time seeing, because every individual gap is ordinary.
    span = grouped["ts"].max() - grouped["ts"].min()
    out["span_days"] = span / 86_400.0
    out["events_per_day"] = out["n_events"] / np.maximum(out["span_days"], 1.0 / 24.0)
    out["mean_gap"] = span / np.maximum(out["n_events"] - 1, 1)
    out["hour_entropy"] = grouped["hour"].apply(
        lambda s: _entropy(np.bincount(s.to_numpy(), minlength=24).astype(float))
    )
    out["counterparty_entropy"] = grouped[counter].apply(
        lambda s: _entropy(s.value_counts().to_numpy().astype(float))
    )
    return out


@dataclass(slots=True)
class EntityNovelty:
    """Isolation forests over entity vectors, one per entity type."""

    seed: int = 1337
    n_estimators: int = 200
    forests: dict[str, Any] = field(default_factory=dict)
    columns: dict[str, list[str]] = field(default_factory=dict)
    medians: dict[str, pd.Series] = field(default_factory=dict)

    def fit(self, train: pd.DataFrame) -> EntityNovelty:
        """Fit on every entity in the training window. **No labels are read.**"""
        from sklearn.ensemble import IsolationForest

        for key in ENTITY_KEYS:
            vectors = entity_vectors(train, key)
            vectors = vectors[vectors["n_events"] >= MIN_EVENTS]
            if vectors.empty:
                continue
            numeric = vectors.replace([np.inf, -np.inf], np.nan)
            median = numeric.median()
            numeric = numeric.fillna(median)
            forest = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=CONTAMINATION,
                random_state=self.seed,
                n_jobs=-1,
            )
            forest.fit(numeric)
            self.forests[key] = forest
            self.columns[key] = list(numeric.columns)
            self.medians[key] = median
        return self

    def score_entities(self, frame: pd.DataFrame, key: str) -> pd.Series:
        """Novelty per entity over ``frame``; higher is more anomalous."""
        forest = self.forests.get(key)
        if forest is None:
            return pd.Series(dtype=float)
        vectors = entity_vectors(frame, key)
        numeric = (
            vectors.reindex(columns=self.columns[key])
            .replace([np.inf, -np.inf], np.nan)
            .fillna(self.medians[key])
        )
        scores = -np.asarray(forest.score_samples(numeric), dtype=float)
        scores[vectors["n_events"].to_numpy() < MIN_EVENTS] = np.nan
        return pd.Series(scores, index=vectors.index)

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """Per-event score: the worst of the event's two entities.

        Combined as a **rank** over each entity type's own score distribution
        rather than as raw path lengths, for the same reason the fusion layer
        uses percentiles: two forests fitted on two different entity populations
        do not share a scale.
        """
        combined = np.zeros(len(frame), dtype=float)
        for key in ENTITY_KEYS:
            per_entity = self.score_entities(frame, key)
            if per_entity.empty:
                continue
            ranked = per_entity.rank(pct=True, na_option="bottom")
            mapped = frame[key].astype(str).map(ranked).to_numpy(dtype=float)
            combined = np.maximum(combined, np.nan_to_num(mapped, nan=0.0))
        return combined
