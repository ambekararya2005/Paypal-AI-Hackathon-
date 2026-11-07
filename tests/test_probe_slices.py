"""The probe slice is the one place an injector can grade its own homework.

Why this file exists
--------------------
:meth:`BaseAttack.probe_slice` narrows the background an attack is measured
against, and **the separability gate applies to that conditional number**. The
conditioning is legitimate — a deployed detector genuinely does branch on rail
and processing code, so grading an agentic attack against card-present traffic
measures the rail rather than the attack — but the slice is chosen by the class
being graded. Left unchecked, an injector could slice itself to
``agentic and provenance_chain longer than three`` and report a beautiful AUC
that meant nothing at all.

So the rule is stated and then enforced: **a slice may condition only on facts a
detector already knows before it scores, and which are not consequences of the
attack.** Three tests below, in increasing order of teeth:

1. Every declared column is on ``SLICE_ALLOWED_COLUMNS``. Cheap; catches a typo
   or a careless addition.
2. The returned mask is genuinely a **function of those columns alone**. This is
   the one that matters: it is verified by grouping the background on the
   declared columns and asserting the mask is constant inside every group. A
   slice that secretly read ``amount`` would vary within a group and fail here,
   whatever it declared.
3. The slice contains the attack. A slice that excluded some of the attack's own
   rows would be cherry-picking the easy ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mantis.core.events import LABEL_COLUMNS, POST_HOC_COLUMNS
from mantis.foundry.base.reference import ReferenceStats, load_reference_stats
from mantis.foundry.base.simulator import SimulationConfig, simulate_frame
from mantis.foundry.injectors import REGISTRY
from mantis.foundry.injectors.base import BaseAttack, PopulationView, run_injector
from mantis.foundry.injectors.probe import SLICE_ALLOWED_COLUMNS, THIN_SLICE_ROWS

SMALL = SimulationConfig(n_events=30_000, seed=7, n_customers=1_200, n_merchants=3_000)

#: Injectors that declare a slice, and are therefore graded conditionally.
SLICED = sorted(card_id for card_id, cls in REGISTRY.items() if cls.slice_columns)


@pytest.fixture(scope="module")
def stats() -> ReferenceStats:
    return load_reference_stats()


@pytest.fixture(scope="module")
def background(stats: ReferenceStats) -> pd.DataFrame:
    return simulate_frame(SMALL, stats)


# --------------------------------------------------------------------------- #
# 1. The allow-list itself
# --------------------------------------------------------------------------- #


def test_the_allow_list_excludes_everything_the_attack_controls() -> None:
    """The allow-list is the whole argument; assert what it must never contain.

    Written as a deny-list check rather than by re-stating the allow-list,
    because the failure mode is somebody *adding* a column, not removing one.
    """
    forbidden = {
        # An attack chooses these outright.
        "amount",
        "merchant_id",
        # The issuer's decision on this very message: not known at scoring time.
        "auth_response",
        "settled",
        "settlement_lag_hours",
        # Attack footprint, every one of them.
        "ag_scope_max_amount",
        "ag_scope_categories",
        "ag_provenance_chain",
        "ag_ingested_content_ids",
        "ag_deliberation_latency_ms",
        "ag_cursor_entropy",
        "ag_dwell_time_ms",
        "ag_tool_call_count",
        "ag_human_present",
        "ag_consent_sig_valid",
        "ag_kya_registered",
        "ag_delegation_depth",
        "ag_mandate_hash",
        "ag_mandate_ttl_seconds",
        *LABEL_COLUMNS,
        *POST_HOC_COLUMNS,
    }
    overlap = SLICE_ALLOWED_COLUMNS & forbidden
    assert not overlap, f"a slice must never condition on {sorted(overlap)}"


def test_default_injectors_declare_no_slice() -> None:
    """No slice means measured against the whole population. That is the default.

    Only an attack that genuinely lives inside a definitional slice of traffic
    should declare one, and the eight Day 2 injectors do not.
    """
    for card_id, cls in REGISTRY.items():
        if not cls.slice_columns:
            assert cls.probe_slice(pd.DataFrame()) is None, card_id


def test_a_slice_and_its_columns_are_declared_together() -> None:
    """Neither half is meaningful alone."""
    for card_id, cls in REGISTRY.items():
        has_columns = bool(cls.slice_columns)
        # ``probe_slice`` is only callable with a real frame, so use the
        # registry's own declaration as the source of truth for the pairing.
        overridden = cls.probe_slice.__func__ is not BaseAttack.probe_slice.__func__
        assert has_columns == overridden, (
            f"{card_id} declares slice_columns={cls.slice_columns} but "
            f"{'does not override' if not overridden else 'overrides'} probe_slice"
        )


# --------------------------------------------------------------------------- #
# 2. Per-injector: declared, legal, and honest
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("card_id", SLICED)
def test_declared_columns_are_on_the_allow_list(card_id: str) -> None:
    columns = set(REGISTRY[card_id].slice_columns)
    illegal = columns - SLICE_ALLOWED_COLUMNS
    assert not illegal, (
        f"{card_id} conditions its probe slice on {sorted(illegal)}, which a detector "
        "either does not know before scoring or which the attack itself produced"
    )


@pytest.mark.parametrize("card_id", SLICED)
def test_the_mask_depends_only_on_the_declared_columns(
    card_id: str, background: pd.DataFrame
) -> None:
    """The test with teeth: verify, do not trust the declaration.

    Group the background by the declared columns and assert the mask is constant
    within every group. If the slice read any other column, two rows agreeing on
    everything declared would still be able to disagree on membership -- and they
    would, somewhere in 30,000 rows.
    """
    cls = REGISTRY[card_id]
    mask = np.asarray(cls.probe_slice(background), dtype=bool)
    assert mask.shape == (len(background),)

    key = background[list(cls.slice_columns)].astype(str).agg("\x1f".join, axis=1)
    varies = pd.Series(mask).groupby(key.to_numpy()).nunique()
    offenders = varies[varies > 1]
    assert offenders.empty, (
        f"{card_id}: the slice mask varies inside {len(offenders)} group(s) of rows that "
        f"agree on {list(cls.slice_columns)} -- it is reading a column it did not declare"
    )


@pytest.mark.parametrize("card_id", SLICED)
def test_the_slice_contains_the_whole_attack(card_id: str, background: pd.DataFrame) -> None:
    """A slice that excluded attack rows would be measuring the easy ones.

    The same predicate applied to the attack frame must keep all of it: the
    slice names the population the attack lives in, so the attack must live in
    it.
    """
    view = PopulationView.build(background)
    attack = run_injector(REGISTRY[card_id], view, seed=7)
    inside = np.asarray(REGISTRY[card_id].probe_slice(attack), dtype=bool)
    assert inside.all(), (
        f"{card_id}: {int((~inside).sum())}/{len(attack)} attack rows fall outside the "
        "slice they are graded in"
    )


@pytest.mark.parametrize("card_id", SLICED)
def test_the_slice_is_not_a_rounding_error(card_id: str, background: pd.DataFrame) -> None:
    """A conditional AUC needs a denominator worth the name.

    This asserts only that the slice is non-degenerate. The *reporting* rule is
    separate and lives in the probe table, which prints ``slice n`` for every
    attack and marks anything under ``THIN_SLICE_ROWS`` with a ``!`` -- because a
    number measured against a few hundred rows must not be quoted as if it were
    measured against sixty thousand. F1-03's slice (agent-mediated refunds) is
    genuinely thin, and that is a fact about refunds rather than a defect, so it
    is surfaced rather than engineered away.
    """
    mask = np.asarray(REGISTRY[card_id].probe_slice(background), dtype=bool)
    n = int(mask.sum())
    assert n >= 50, f"{card_id}: slice holds {n} rows; too few to measure anything"
    assert n < len(background), f"{card_id}: the slice is the whole population; drop it"


def test_thin_slices_are_flagged_in_the_table(background: pd.DataFrame) -> None:
    """The '!' marker has to actually fire, or it is decoration.

    F1-03 is the case it exists for: legitimate agent-mediated refunds are a
    small population, so its conditional AUC rests on a few hundred rows.
    """
    from mantis.foundry.injectors.probe import build_slices, format_probe_table, probe_report

    view = PopulationView.build(background)
    attacks = {c: run_injector(REGISTRY[c], view, seed=7) for c in ("F1-01", "F1-03")}
    slices = build_slices(background, {c: REGISTRY[c] for c in attacks})
    report = probe_report(background, attacks, slices=slices, max_negatives=20_000)

    f1_03 = report[report["attack_id"] == "F1-03"].iloc[0]
    f1_01 = report[report["attack_id"] == "F1-01"].iloc[0]
    assert f1_03["slice_n"] < f1_01["slice_n"], "refunds should be the thinner slice"
    assert f1_03["slice_n"] < THIN_SLICE_ROWS

    table = format_probe_table(report)
    assert "slice n" in table
    assert "!" in table, "a thin slice must be marked in the printed table"
    # And the direction tag, which is what stops a feature being read backwards.
    assert "[hi]" in table or "[lo]" in table
