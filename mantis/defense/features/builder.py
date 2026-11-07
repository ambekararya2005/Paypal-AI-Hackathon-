"""The feature builder: fit on train, transform anything, assert on every call.

    builder = FeatureBuilder().fit(train)
    X_train = builder.transform(train)
    X_test  = builder.transform(test)

The assertion is the point
---------------------------
:meth:`FeatureBuilder.transform` ends by checking that no forbidden column
survived into the matrix — by **name**, against
:data:`mantis.defense.features.spec.FORBIDDEN_COLUMNS`, which covers all three
tiers: ground truth, post-hoc dispute state, and the current event's own outcome.
It raises rather than warns.

CLAUDE.md HARD RULE 1 asks for exactly this, and the reason it is a hard rule
rather than a style note is that leakage does not announce itself. It shows up as
a *good* number. A judge who sees 0.999 AUC-PR assumes the model cheated, and
they are almost always right. So the check runs on every transform, including the
ones inside the leave-one-family-out loop, and the cost is a set intersection.

Why a class rather than a function
-----------------------------------
Because half of these features are fitted quantities — per-customer baselines,
per-merchant ratios, the deliberation regression — and a function would have to
recompute them from whatever frame it was handed. That is the bug: computing a
customer's baseline from the frame you are scoring means the baseline contains
the event you are scoring against it. ``fit`` and ``transform`` being separate
methods makes that mistake require deliberate effort.

Ordering
---------
``transform`` sorts by timestamp, builds, and then restores the caller's row
order. The sort is required by the velocity pass (its state must never contain
the future); restoring the order afterwards means callers can line the matrix up
with their own labels without thinking about it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import pandas as pd

from mantis.defense.features.entity import EntityProfiles, entity_features
from mantis.defense.features.mandate import MandateBaselines, mandate_features
from mantis.defense.features.spec import FORBIDDEN_COLUMNS, VELOCITY_KEYS, FeatureConfig
from mantis.defense.features.transaction import transaction_features
from mantis.defense.features.velocity import velocity_features
from mantis.defense.l4_graph.graph import graph_features

__all__ = ["FeatureBuilder", "LeakageError"]


class LeakageError(RuntimeError):
    """A forbidden column reached the feature matrix. See CLAUDE.md HARD RULE 1."""


#: Prefixes identifying each feature group, for the CLI's summary table.
GROUP_PREFIXES: Final[dict[str, str]] = {
    "txn_": "transaction",
    "vel_": "velocity",
    "ent_": "entity",
    "mnd_": "mandate",
    "gph_": "graph",
}


@dataclass(slots=True)
class FeatureBuilder:
    """Builds the firewall's feature matrix. Fit once on train, transform anywhere."""

    config: FeatureConfig = field(default_factory=FeatureConfig)
    profiles: EntityProfiles | None = None
    baselines: MandateBaselines | None = None
    feature_names: list[str] = field(default_factory=list)

    def fit(self, train: pd.DataFrame) -> FeatureBuilder:
        """Estimate every fitted quantity from the training split.

        Fitted on the **whole** training split, fraud included. That is
        deliberate and it is not leakage: an issuer's baselines are computed from
        the traffic they actually saw, which contains whatever fraud was in it.
        Excluding known fraud would build a cleaner baseline than any deployed
        system has, and would flatter every subsequent number.
        """
        self.profiles = EntityProfiles.fit(train)
        self.baselines = MandateBaselines.fit(train)
        self.feature_names = list(self.transform(train, _skip_name_check=True).columns)
        return self

    def fit_transform_stream(
        self, frame: pd.DataFrame, train_mask: pd.Series
    ) -> pd.DataFrame:
        """Fit on the train rows, then build the matrix in **one continuous pass**.

        This is the entry point every experiment should use, and the reason is a
        bug that :meth:`transform` cannot avoid on its own.

        Velocity state is history. In production it is continuous: the card that
        was seen yesterday is still in the store today, and the split between
        "training data" and "live traffic" is a fact about when the model was
        fitted, not a break in the event stream. Calling ``transform(train)`` and
        then ``transform(test)`` builds **two** stores and throws the first away,
        so every entity's history restarts at the split boundary.

        That is not a rounding error. ``vel_mandate_hash_lifetime_count`` is the
        replay detector for F1-10: it counts prior presentations of the same
        signed mandate. If the original presentation lands in train and the
        replay in test, a per-split store scores the replay as a first sighting
        and the feature measures **0.500 AUC** — a detector that is exactly
        useless, silently, while looking like it works. Measured: 0.500 with two
        stores, 0.90+ with one.

        Continuity is not leakage, and it is worth being precise about why: the
        velocity store only ever looks **backwards**. A test-period event reads
        state accumulated from events strictly before it, some of which happen to
        be in the training window. That is what an issuer holds. What *would* be
        leakage is fitting the entity baselines on the test period, and that is
        why ``profiles`` and ``baselines`` are still fitted on ``train_mask``
        alone.

        Args:
            frame: The whole labelled file, any order.
            train_mask: Boolean over ``frame``, True for training rows.

        Returns:
            The feature matrix for **every** row of ``frame``, in ``frame``'s
            own order. Slice it with the same mask to recover train and test.
        """
        train = frame[train_mask.reindex(frame.index).fillna(False).to_numpy()]
        if train.empty:
            raise ValueError("train_mask selected no rows")
        self.profiles = EntityProfiles.fit(train)
        self.baselines = MandateBaselines.fit(train)
        matrix = self.transform(frame, _skip_name_check=True)
        self.feature_names = list(matrix.columns)
        return matrix

    def transform(self, frame: pd.DataFrame, *, _skip_name_check: bool = False) -> pd.DataFrame:
        """Build the feature matrix for ``frame``. Requires :meth:`fit` first."""
        if self.profiles is None or self.baselines is None:
            raise RuntimeError("FeatureBuilder.transform called before fit")

        order = frame.index
        ordered = frame.sort_values("ts", kind="stable")

        blocks = [
            transaction_features(ordered),
            velocity_features(ordered, VELOCITY_KEYS),
            entity_features(ordered, self.profiles),
            mandate_features(ordered, self.baselines),
        ]
        if self.config.include_graph:
            # Streamed, not fitted: the graph reads state from strictly earlier
            # events and folds the current one in afterwards, so it needs the
            # same timestamp ordering the velocity pass does and no fit step.
            # See mantis.defense.l4_graph.graph.
            blocks.append(graph_features(ordered))
        matrix = pd.concat(blocks, axis=1).reindex(order)

        if self.config.include_current_outcome:
            # The documented escape hatch. Never on for anything in RESULTS.md.
            for column in ("auth_response", "settled", "settlement_lag_hours"):
                matrix[f"posthoc_{column}"] = frame[column]

        self._assert_no_leakage(matrix)
        if not _skip_name_check and self.feature_names:
            self._assert_stable_columns(matrix)
        return matrix

    def _assert_no_leakage(self, matrix: pd.DataFrame) -> None:
        """The HARD RULE 1 check, plus the two tiers HARD RULE 1 does not name."""
        columns = set(matrix.columns)
        forbidden = FORBIDDEN_COLUMNS
        if self.config.include_current_outcome:
            forbidden = tuple(c for c in forbidden if c not in ("auth_response", "settled",
                                                               "settlement_lag_hours"))
        found = sorted(columns & set(forbidden))
        if found:
            raise LeakageError(
                f"forbidden columns reached the feature matrix: {found}. "
                "See CLAUDE.md HARD RULE 1 and features/spec.py for why each tier exists."
            )
        # A derived name is the other way this leaks: a column called
        # 'is_fraud_ratio' would pass the check above and be just as fatal.
        suspicious = sorted(
            c for c in columns if any(bad in c for bad in ("is_fraud", "attack_", "dispute_"))
        )
        if suspicious:
            raise LeakageError(
                f"feature names derived from ground truth or post-hoc state: {suspicious}"
            )

    def _assert_stable_columns(self, matrix: pd.DataFrame) -> None:
        """Train and score must see the same columns in the same order."""
        if list(matrix.columns) != self.feature_names:
            missing = sorted(set(self.feature_names) - set(matrix.columns))
            extra = sorted(set(matrix.columns) - set(self.feature_names))
            raise RuntimeError(
                f"feature matrix does not match the fitted schema; missing={missing}, extra={extra}"
            )

    def group_counts(self) -> dict[str, int]:
        """Feature count per group, for the CLI and the writeup."""
        counts = dict.fromkeys(GROUP_PREFIXES.values(), 0)
        counts["categorical"] = 0
        for name in self.feature_names:
            for prefix, group in GROUP_PREFIXES.items():
                if name.startswith(prefix):
                    counts[group] += 1
                    break
            else:
                counts["categorical"] += 1
        return counts
