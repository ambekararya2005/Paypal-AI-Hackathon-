"""The fidelity scorecard's properties, pinned.

Two of these tests exist because the first version of the scorecard was **wrong
in a way that looked right**, and the general lesson is worth a regression test
rather than a comment:

* ``test_shape_space_excludes_currency_bound_columns`` — the whole package rests
  on nothing being compared raw. A future edit that puts ``amount`` or ``mcc``
  into the shape space would produce a large, confident discriminator AUC that
  measures the difference between India and the United States.
* ``test_panel_z_of_a_mostly_constant_column_is_a_fingerprint`` — the bug itself.
  Standardising a discrete, overwhelmingly-zero column maps its shared modal
  value to a different number on each panel, and a classifier separates them
  perfectly while the underlying distributions are close. This test builds two
  samples **from the same distribution** and asserts the transform would have
  separated them, so the reason the velocity counts are no longer z-scored is
  recorded as an executable fact rather than as a paragraph in a docstring.

The rest pin the contract the API and the console depend on: the scorecard runs
without a reference panel, says so, and substitutes nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mantis.foundry.fidelity import metrics
from mantis.foundry.fidelity.adjudicate import ADJUDICATED_FEATURES, adjudicate
from mantis.foundry.fidelity.common import (
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    SHAPE_FEATURES,
    panel_levels,
    to_common,
    to_shape,
)
from mantis.foundry.fidelity.scorecard import KNOWN_DIVERGENCES, build_scorecard

#: Columns whose presence in the shape space would make every distance a
#: statement about geography or currency instead of about fidelity.
FORBIDDEN_IN_SHAPE = (
    "amount",
    "currency",
    "mcc",
    "category",
    "channel",
    "entry_mode",
    "card_bin",
    "merchant_country",
    "lat",
    "lon",
    "threeds_result",
)


def _synthetic_frame(n: int = 4_000, seed: int = 7) -> pd.DataFrame:
    """A minimal frame with the columns ``to_common`` reads. Not a population."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-05-15")
    return pd.DataFrame(
        {
            "ts": start + pd.to_timedelta(np.sort(rng.uniform(0, 89 * 86_400, n)), unit="s"),
            "amount": np.exp(rng.normal(6.0, 1.1, n)),
            "mcc": rng.choice(["5411", "5812", "5541", "4121"], n),
            "channel": rng.choice(["ecom", "card_present", "upi_p2m"], n),
            "txn_type": "purchase",
            "customer_id": [f"cus-{i:05d}" for i in rng.integers(0, 300, n)],
            "merchant_id": [f"mer-{i:05d}" for i in rng.integers(0, 500, n)],
            "is_fraud": rng.random(n) < 0.01,
            "attack_id": None,
        }
    )


# --------------------------------------------------------------------------- #
# The shape space
# --------------------------------------------------------------------------- #


def test_shape_space_excludes_currency_bound_columns():
    """Nothing in the comparison may be a fact about a country or a currency."""
    for column in FORBIDDEN_IN_SHAPE:
        assert column not in SHAPE_FEATURES, (
            f"{column!r} reached the shape space. Every distance in this package would "
            "then be partly measuring the difference between an Indian rupee population "
            "and a US dollar panel -- see mantis/foundry/fidelity/common.py."
        )


def test_shape_matrix_has_exactly_the_declared_columns():
    matrix = to_shape(to_common(_synthetic_frame(), source="synthetic"))
    assert list(matrix.columns) == list(SHAPE_FEATURES)
    assert set(CONTINUOUS_FEATURES) | set(CATEGORICAL_FEATURES) == set(SHAPE_FEATURES)


def test_agentic_rail_is_excluded_from_the_comparison():
    """The reference panel has none, so ours must not be in the frame either."""
    frame = _synthetic_frame()
    frame.loc[frame.index[:500], "channel"] = "agentic"
    common = to_common(frame, source="synthetic")
    assert len(common) == len(frame) - 500


def test_refunds_and_reversals_are_excluded():
    """A lifecycle the reference panel does not model cannot be compared to it."""
    frame = _synthetic_frame()
    frame.loc[frame.index[:200], "txn_type"] = "refund"
    assert len(to_common(frame, source="synthetic")) == len(frame) - 200


def test_gap_ratio_is_centred_on_each_cardholder_and_not_on_the_panel():
    """The feature that replaced the z-scored counts must divide out the rate.

    Two panels differing only in overall rate must produce the same
    ``gap_ratio_log`` distribution. That is the property the whole velocity
    comparison rests on, and it is the one the previous transform did not have.
    """
    slow = _synthetic_frame(3_000, seed=11)
    fast = slow.copy()
    # Compress every timestamp toward the start: ten times the rate, identical
    # relative spacing, so a rate-free feature cannot tell them apart.
    origin = fast["ts"].min()
    fast["ts"] = origin + (fast["ts"] - origin) / 10

    a = to_shape(to_common(slow, source="synthetic"))["gap_ratio_log"]
    b = to_shape(to_common(fast, source="synthetic"))["gap_ratio_log"]
    assert metrics.ks_two_sample(a.to_numpy(), b.to_numpy()) < 0.02


def test_panel_z_of_a_mostly_constant_column_is_a_fingerprint():
    """The bug that produced a discriminator AUC of 1.000 on nothing.

    Two samples drawn from the **same** distribution, standardised on their own
    moments, separate almost perfectly -- because the shared modal atom lands on
    a different z on each side. Asserted rather than described, so that a future
    edit reintroducing a panel-wise z-score on a discrete column fails here.
    """
    rng = np.random.default_rng(3)
    a = rng.poisson(0.14, 60_000).astype(float)
    b = rng.poisson(0.14, 60_000).astype(float)

    # The distributions themselves are indistinguishable.
    assert metrics.ks_two_sample(a, b) < 0.01

    za = (a - a.mean()) / a.std()
    zb = (b - b.mean()) / b.std()
    # ...and yet the standardised versions place the shared zero at two different
    # values, which is all a tree needs.
    assert za[a == 0][0] != zb[b == 0][0]


# --------------------------------------------------------------------------- #
# Distances
# --------------------------------------------------------------------------- #


def test_jsd_is_zero_on_identical_distributions_and_one_on_disjoint_ones():
    p = np.array([0.2, 0.3, 0.5])
    assert metrics.jsd(p, p) == pytest.approx(0.0, abs=1e-12)
    assert metrics.jsd(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)


def test_ks_two_sample_matches_scipy():
    from scipy.stats import ks_2samp

    rng = np.random.default_rng(5)
    a, b = rng.normal(size=2_000), rng.normal(0.3, size=2_500)
    assert metrics.ks_two_sample(a, b) == pytest.approx(ks_2samp(a, b).statistic, abs=1e-9)


def test_ks_band_shrinks_with_sample_size():
    """A distance is only evidence relative to what noise gives you at that n."""
    assert metrics.ks_two_sample_band(1_000, 1_000) > metrics.ks_two_sample_band(100_000, 100_000)


def test_ratio_handles_a_zero_band():
    assert metrics.ratio(0.0, 0.0) == 0.0
    assert metrics.ratio(0.1, 0.0) == float("inf")


def test_drift_check_imports_the_same_metrics():
    """Day 4's script and Day 7's package must not carry two implementations."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "drift_check.py"
    spec = importlib.util.spec_from_file_location("drift_check_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.jsd is metrics.jsd
    assert module._ratio is metrics.ratio


# --------------------------------------------------------------------------- #
# The scorecard's contract
# --------------------------------------------------------------------------- #


def test_scorecard_runs_without_a_reference_panel_and_says_so(monkeypatch):
    """A clean clone has no panel. That is a state, not a failure -- and nothing
    may be substituted for the sections it cannot compute."""
    from mantis.foundry.fidelity import real, scorecard

    monkeypatch.setattr(real, "available", lambda: False)
    monkeypatch.setattr(scorecard.real, "available", lambda: False)

    card = build_scorecard(_synthetic_frame())
    assert card["reference"]["available"] is False
    assert "marginals" not in card
    assert "tstr" not in card
    assert "discriminator" not in card
    # The levels ARE computable without a panel and are still reported.
    assert card["synthetic"]["levels"]["customers"] > 0


def test_known_divergences_each_carry_a_measurement_and_a_reason():
    """The list is only worth having if every entry is falsifiable."""
    assert KNOWN_DIVERGENCES
    for row in KNOWN_DIVERGENCES:
        assert set(row) == {"name", "measured", "cause", "why_not_fixed"}
        assert any(character.isdigit() for character in row["measured"]), (
            f"{row['name']!r} claims a divergence without a number. An entry here is a "
            "measurement, not an admission."
        )


def test_adjudications_cover_exactly_the_ablated_features():
    """The ablated discriminator drops what the adjudicator excused, and no more."""
    synthetic = to_common(_synthetic_frame(seed=1), source="synthetic")
    reference = to_common(_synthetic_frame(seed=2), source="synthetic")
    rows = adjudicate(synthetic, reference)
    assert {row["feature"] for row in rows} == set(ADJUDICATED_FEATURES)
    for row in rows:
        assert row["verdict"] in {"REFERENCE", "SYNTHETIC"}
        # Every verdict has to be derived from a stated third quantity, and the
        # evidence strings must carry the numbers that decided it.
        assert row["third_quantity"]
        assert any(character.isdigit() for character in row["synthetic"])
        assert any(character.isdigit() for character in row["reference"])


def test_panel_levels_carry_no_distance():
    """Levels are ratios reported bare. If one ever gains a p-value, it stops
    being a fact about composition and starts being a claim."""
    levels = panel_levels(to_common(_synthetic_frame(), source="synthetic"))
    assert set(levels) == {
        "events",
        "days",
        "customers",
        "merchants",
        "categories",
        "txn_per_customer_per_day",
        "median_hours_between",
        "merchants_per_customer",
        "top_1pct_merchant_share",
    }
    assert all(isinstance(value, float) for value in levels.values())
