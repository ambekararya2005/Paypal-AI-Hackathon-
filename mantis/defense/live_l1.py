"""L1 + fusion for the live scorer: a persisted artefact, scored one event at a time.

Built by ``scripts/train_live_model.py`` into ``data/models/live_l1.pkl``
(gitignored). :class:`~mantis.defense.live.LiveScorer` loads it when present and
runs L0 + L3 without it; either way ``layers_ran`` says which.

This is a separate artefact from the one behind RESULTS.md
------------------------------------------------------------
Same recipe — a pooled population file (two worlds by default; see the
script for why not five), a 70/30 time split, the 232-feature
builder, LightGBM plus isotonic calibration, fusion weights fitted on an inner
slice none of the base layers saw — with one deliberate difference: the fusion
here combines **L1 and L3 only**. L2 is a residual monitor (CLAUDE.md, "The
zero-day answer, reframed") and its fusion weight was already near zero. The
artefact's own test-window metrics are written next to it in
``live_l1.json``, labelled as such. Nothing here writes to RESULTS.md or
``data/generated/``.

What L1 can and cannot say about a PayPal order — stated before anyone asks
-----------------------------------------------------------------------------
L1 was trained on the synthetic population: rupee amounts, Indian merchants,
customers with months of history. A PayPal sandbox order is a dollar amount from
a customer and merchant L1 has never seen. Its entity features are therefore
absent, and its velocity and graph features see only the history this process
has gathered through the gate. That is the **cold-start** case, and an honest L1
score on it is weak evidence. The scorer reports the score with its percentile
against legitimate training traffic, so a reader can see how unusual it is, and
lets the fused score decide only through thresholds fitted at FPR budgets.

The single-event path
-----------------------
``FeatureBuilder.transform`` on a frame holding this process's recent gate events
plus the new one; the last row is the vector. The history is what lets a replayed
mandate hash or a burst on one customer register in the velocity block. It is
bounded, so the cost stays flat. The Day 7 finding still applies: the builder's
entity block is the expensive one on a one-row frame (~47 ms), and the plain-dict
rewrite is on the outstanding list, not done here.
"""

from __future__ import annotations

import pickle
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from mantis.core.events import TxEvent, flatten
from mantis.defense.fusion.model import legit_percentile
from mantis.defense.policy.decide import PolicyThresholds

if TYPE_CHECKING:
    from mantis.defense.live import Reason

__all__ = ["ARTEFACT_NAME", "OnlineL1", "load_online_l1"]

ARTEFACT_NAME: Final[str] = "live_l1.pkl"

#: Gate events kept as velocity/graph history for the next one.
HISTORY: Final[int] = 512


@dataclass(slots=True)
class OnlineL1:
    """What the training script persists, plus the per-process event history."""

    builder: Any  # FeatureBuilder, fitted
    l1: Any  # L1Model, fitted
    fusion: Any  # FusionModel over {"L1", "L3"}
    thresholds: PolicyThresholds
    #: Legitimate L1 scores on the test window, for the percentile in reasons.
    l1_legit_reference: np.ndarray
    manifest: dict[str, Any]
    name: str = "L1+fusion"
    history: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=HISTORY))

    def __getstate__(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__ if k != "history"}

    def __setstate__(self, state: dict[str, Any]) -> None:
        for key, value in state.items():
            setattr(self, key, value)
        self.history = deque(maxlen=HISTORY)

    def _vector(self, event: TxEvent) -> pd.DataFrame:
        rows = [*self.history, flatten(event)]
        frame = pd.DataFrame(rows)
        for column in ("ts", "ag_mandate_issued_ts", "dispute_raised_ts"):
            frame[column] = pd.to_datetime(frame[column], utc=True)
        matrix = self.builder.transform(frame)
        return matrix.iloc[[-1]]

    def _top_contributions(self, vector: pd.DataFrame, top: int = 2) -> list[tuple[str, float]]:
        contrib = np.asarray(
            self.l1.booster.predict(vector[self.l1.feature_names], pred_contrib=True)
        ).ravel()[:-1]  # last column is the bias
        order = np.argsort(contrib)[::-1][:top]
        return [(self.l1.feature_names[i], float(contrib[i])) for i in order if contrib[i] > 0]

    def score_one(self, event: TxEvent, *, l3_score: float) -> tuple[float, list[Reason]]:
        """Fused score for one event, and the L1 reasons behind it."""
        from mantis.defense.live import Reason

        vector = self._vector(event)
        self.history.append(flatten(event))
        s1 = float(self.l1.score(vector)[0])
        fused = float(self.fusion.score({"L1": np.array([s1]), "L3": np.array([l3_score])})[0])
        pct = float(legit_percentile(np.array([s1]), self.l1_legit_reference)[0])
        drivers = self._top_contributions(vector)
        said = (f"; drivers: {', '.join(f'{n} (+{v:.2f})' for n, v in drivers)}"
                if drivers else "")
        reasons = [Reason(
            "L1", "gbdt_score",
            f"L1 P(fraud)={s1:.3f}, above {pct:.1%} of legitimate training traffic; "
            f"fused {fused:.3f} vs review line {self.thresholds.review:.3f} "
            f"(cold start: no history for this customer){said}",
        )]
        return fused, reasons


def load_online_l1(model_dir: Path) -> OnlineL1 | None:
    """The artefact if ``scripts/train_live_model.py`` has run, else ``None``."""
    path = model_dir / ARTEFACT_NAME
    if not path.exists():
        return None
    with path.open("rb") as handle:
        # Our own artefact, written by our own script into a gitignored directory.
        obj = pickle.load(handle)
    return obj if isinstance(obj, OnlineL1) else None
