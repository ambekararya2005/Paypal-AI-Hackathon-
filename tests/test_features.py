"""The feature layer's contract, tested where it can actually break.

Three things are worth a test here and the rest is arithmetic:

1. **The leakage assertion fires**, on all three tiers. This is HARD RULE 1 plus
   the two tiers HARD RULE 1 does not name, and it is the single check standing
   between this project and a 0.999 AUC nobody should believe.
2. **The velocity store never sees the present.** A feature computed from state
   that includes the event being scored is self-fulfilling, and unlike a stray
   label column nothing about it *looks* wrong.
3. **Fitted state comes from train only.** A per-customer baseline that saw the
   test period is a baseline containing the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mantis.core.events import LABEL_COLUMNS, POST_HOC_COLUMNS
from mantis.defense.features import FeatureBuilder, LeakageError
from mantis.defense.features.spec import FORBIDDEN_COLUMNS, FUTURE_COLUMNS
from mantis.defense.features.state import WINDOWS, RollingStore, WindowSpec
from mantis.foundry.base.reference import load_reference_stats
from mantis.foundry.base.simulator import SimulationConfig, simulate_frame
from mantis.foundry.injectors import REGISTRY
from mantis.foundry.injectors.base import PopulationView, run_injector

SMALL = SimulationConfig(n_events=25_000, seed=7, n_customers=1_000, n_merchants=2_500)


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    """A small labelled file: background plus a few attacks."""
    background = simulate_frame(SMALL, load_reference_stats())
    view = PopulationView.build(background)
    attacks = [
        run_injector(REGISTRY[c], view, seed=7) for c in ("F1-10", "F4-27", "F1-01")
    ]
    frame = pd.concat([background, *attacks], ignore_index=True)
    return frame.sort_values("ts", kind="stable").reset_index(drop=True)


@pytest.fixture(scope="module")
def split(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    cut = dataset["ts"].quantile(0.7)
    return dataset, dataset["ts"] <= cut


@pytest.fixture(scope="module")
def built(split: tuple[pd.DataFrame, pd.Series]) -> tuple[FeatureBuilder, pd.DataFrame]:
    frame, mask = split
    builder = FeatureBuilder()
    return builder, builder.fit_transform_stream(frame, mask)


# --------------------------------------------------------------------------- #
# 1. The leakage contract
# --------------------------------------------------------------------------- #


def test_the_three_tiers_are_all_covered() -> None:
    """FORBIDDEN_COLUMNS must be the union, not just the labels."""
    for column in (*LABEL_COLUMNS, *POST_HOC_COLUMNS, *FUTURE_COLUMNS):
        assert column in FORBIDDEN_COLUMNS


def test_no_forbidden_column_reaches_the_matrix(
    built: tuple[FeatureBuilder, pd.DataFrame],
) -> None:
    _, matrix = built
    overlap = set(matrix.columns) & set(FORBIDDEN_COLUMNS)
    assert not overlap, f"forbidden columns in the feature matrix: {sorted(overlap)}"


@pytest.mark.parametrize("column", FORBIDDEN_COLUMNS)
def test_the_assertion_actually_fires(
    column: str, built: tuple[FeatureBuilder, pd.DataFrame], dataset: pd.DataFrame
) -> None:
    """Smuggle each forbidden column in and require a raise.

    Parametrised over every tier because a check that only covers the labels
    would pass while ``auth_response`` — which makes F4-27 trivial and means
    nothing — walked straight through.
    """
    builder, matrix = built
    probe = matrix.copy()
    probe[column] = dataset[column].to_numpy()
    with pytest.raises(LeakageError):
        builder._assert_no_leakage(probe)


def test_names_derived_from_ground_truth_are_rejected(
    built: tuple[FeatureBuilder, pd.DataFrame],
) -> None:
    """A column called ``is_fraud_ratio_7d`` is not on the list and is just as fatal."""
    builder, matrix = built
    probe = matrix.copy()
    probe["vel_customer_is_fraud_ratio_7d"] = 0.0
    with pytest.raises(LeakageError):
        builder._assert_no_leakage(probe)


# --------------------------------------------------------------------------- #
# 2. The velocity store never sees the present
# --------------------------------------------------------------------------- #


def test_the_first_event_for_a_key_has_no_history() -> None:
    """An unseen key returns NaN, not zero. 'No history' and 'a history of zero'
    are different facts and a tree can split on the difference."""
    store = RollingStore((WindowSpec("customer", ("customer_id",)),))
    out = store.observe(
        {"customer": "cus-1"},
        ts=1_000.0,
        amount=100.0,
        declined=False,
        outcome_known=True,
        refund=False,
        settlement_lag=None,
    )
    assert all(np.isnan(v) for v in out)


def test_an_events_own_outcome_never_reaches_its_own_features() -> None:
    """The asymmetry the whole layer rests on.

    A declined authorisation must not raise its *own* decline ratio. It must
    raise the ratio for the next event on that key.
    """
    spec = WindowSpec("bin", ("card_bin",))
    store = RollingStore((spec,))
    names = store.feature_names()
    ratio = names.index("vel_bin_decline_ratio_1h")

    for i in range(4):
        store.observe(
            {"bin": "411111"},
            ts=1_000.0 + i,
            amount=50.0,
            declined=False,
            outcome_known=True,
            refund=False,
            settlement_lag=1.0,
        )
    # This one declines. Its own features must still show a clean history.
    out = store.observe(
        {"bin": "411111"},
        ts=1_010.0,
        amount=50.0,
        declined=True,
        outcome_known=True,
        refund=False,
        settlement_lag=None,
    )
    assert out[ratio] == 0.0, "an event's own decline leaked into its own ratio"

    # The next event sees it.
    after = store.observe(
        {"bin": "411111"},
        ts=1_020.0,
        amount=50.0,
        declined=False,
        outcome_known=True,
        refund=False,
        settlement_lag=1.0,
    )
    assert after[ratio] == pytest.approx(1 / 5)


def test_windows_evict_and_the_counts_respect_them() -> None:
    """An event older than the window must not be counted inside it."""
    spec = WindowSpec("customer", ("customer_id",))
    store = RollingStore((spec,))
    names = store.feature_names()
    count_1h = names.index("vel_customer_count_1h")
    count_7d = names.index("vel_customer_count_7d")

    base = 1_000_000.0
    for i in range(3):
        store.observe(
            {"customer": "c"}, ts=base + i, amount=10.0, declined=False,
            outcome_known=True, refund=False, settlement_lag=None,
        )
    # Two hours later: outside 1h, inside 7d.
    out = store.observe(
        {"customer": "c"}, ts=base + 2 * WINDOWS["1h"], amount=10.0, declined=False,
        outcome_known=True, refund=False, settlement_lag=None,
    )
    assert out[count_1h] == 0.0
    assert out[count_7d] == 3.0


def test_velocity_requires_sorted_input(dataset: pd.DataFrame) -> None:
    """Unsorted input would let a key's state contain the future."""
    from mantis.defense.features.velocity import velocity_features

    shuffled = dataset.sample(frac=1.0, random_state=0)
    with pytest.raises(ValueError, match="timestamp-ordered"):
        velocity_features(shuffled)


# --------------------------------------------------------------------------- #
# 3. Fitted state comes from train only
# --------------------------------------------------------------------------- #


def test_profiles_are_fitted_on_train_only(split: tuple[pd.DataFrame, pd.Series]) -> None:
    """A customer who appears only after the split must be unknown to the profiles."""
    frame, mask = split
    builder = FeatureBuilder()
    builder.fit_transform_stream(frame, mask)

    train_customers = set(frame.loc[mask.to_numpy(), "customer_id"])
    fitted = set(builder.profiles.customer_n)
    assert fitted <= train_customers, "an entity from the test period reached the profiles"


def test_test_only_entities_are_flagged_unseen(
    split: tuple[pd.DataFrame, pd.Series], built: tuple[FeatureBuilder, pd.DataFrame]
) -> None:
    """``ent_customer_unseen`` is F2-13's shape and must actually fire somewhere."""
    _frame, mask = split
    _, matrix = built
    test_rows = ~mask.to_numpy()
    unseen = matrix.loc[test_rows, "ent_customer_unseen"].to_numpy()
    assert np.nansum(unseen) > 0, "no test-period customer was new; the flag is dead"


def test_the_replay_key_sees_across_the_split(
    split: tuple[pd.DataFrame, pd.Series], built: tuple[FeatureBuilder, pd.DataFrame]
) -> None:
    """The bug ``fit_transform_stream`` exists to prevent.

    ``vel_mandate_hash_lifetime_count`` is the F1-10 replay detector. Building
    train and test through separate stores restarts every hash at zero on the
    far side of the split, which silently reduces the feature to noise.
    """
    frame, _ = split
    _, matrix = built
    replay = frame["attack_id"].to_numpy() == "F1-10"
    counts = matrix.loc[replay, "vel_mandate_hash_lifetime_count"].to_numpy()
    assert np.nanmax(counts) >= 1, "no replayed mandate had any prior sighting"


def test_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        FeatureBuilder().transform(pd.DataFrame())
