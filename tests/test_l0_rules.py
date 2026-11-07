"""L0's clauses, and the Day 3 bucket contract measured against them.

The bucket-contract test here is deliberately **descriptive, not prescriptive**.
Day 4 found that F1-05 declares itself HARD but fires no deployable clause on
more than 3% of its events, and the reconciliation — recorded in
``mantis/defense/l0_rules/__main__.py`` — is that the Day 3 *contract* is what is
wrong, because its test accepted any signal reaching 25% recall without ever
pricing that signal in false positives.

Neither side was adjusted to make the other pass. So the test below pins what is
actually true today, including the exception, and will fail loudly if either the
clause set or the injector changes underneath it. Writing it as an unconditional
"every HARD card fires a clause" would have required either weakening L0 or
relabelling the card, and both would be papering over the finding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mantis.defense.l0_rules import (
    CLAUSES,
    MAX_DELEGATION_DEPTH,
    evaluate,
    make_untrusted_domain_clause,
    trusted_domains,
)
from mantis.foundry.base.reference import load_reference_stats
from mantis.foundry.base.simulator import SimulationConfig, simulate_frame
from mantis.foundry.injectors import REGISTRY
from mantis.foundry.injectors.base import PopulationView, run_injector

SMALL = SimulationConfig(n_events=30_000, seed=7, n_customers=1_200, n_merchants=3_000)

CLEAN_CARDS = ("F1-01", "F1-03")
HARD_CARDS = ("F1-02", "F1-04", "F1-09", "F1-10")

#: The card Day 4 found the Day 3 contract could not support. See the module
#: docstring; this is recorded, not excused.
HARD_BUT_NOT_DEPLOYABLY_CATCHABLE = "F1-05"


@pytest.fixture(scope="module")
def background() -> pd.DataFrame:
    return simulate_frame(SMALL, load_reference_stats())


@pytest.fixture(scope="module")
def attacks(background: pd.DataFrame) -> dict[str, pd.DataFrame]:
    view = PopulationView.build(background)
    cards = (*CLEAN_CARDS, *HARD_CARDS, HARD_BUT_NOT_DEPLOYABLY_CATCHABLE)
    return {c: run_injector(REGISTRY[c], view, seed=7) for c in cards}


# --------------------------------------------------------------------------- #
# The clauses are deployable
# --------------------------------------------------------------------------- #


def test_no_clause_fires_on_a_classic_authorisation(background: pd.DataFrame) -> None:
    """Every clause reads a mandate, and a classic authorisation has none.

    A clause firing off-rail would be reading a null as a violation, which is the
    fastest way to make a rules layer unshippable.
    """
    classic = background[background["ag_agent_id"].isna()]
    result = evaluate(classic)
    for name, mask in result.masks.items():
        assert not mask.any(), f"{name} fired on {int(mask.sum())} classic authorisations"


def test_the_false_positive_rate_is_deployable(background: pd.DataFrame) -> None:
    """L0's whole claim is near-zero FP. Put a number on it and hold it there.

    ``kya_unregistered`` is the loose one by design — the population carries a
    deliberate ~2.8% unregistered tail so that KYA is not a free win — so the
    bound is set where that tail sits rather than at zero.
    """
    result = evaluate(background)
    agentic = background["ag_agent_id"].notna().to_numpy()
    n = max(int(agentic.sum()), 1)
    assert float(result.fired.sum()) / n < 0.05, "L0 fires on too much legitimate traffic"

    for clause in CLAUSES:
        if clause.name == "kya_unregistered":
            continue
        rate = float(result.masks[clause.name].sum()) / n
        assert rate < 0.01, f"{clause.name} fires on {rate:.2%} of legitimate agentic traffic"


def test_an_empty_allow_list_means_unconstrained(background: pd.DataFrame) -> None:
    """The schema says empty means open. Reading it as closed would fire on everything."""
    from mantis.defense.l0_rules.rules import merchant_outside_allow_list

    frame = background[background["ag_agent_id"].notna()].head(200).copy()
    frame["ag_scope_allowed_merchants"] = [[] for _ in range(len(frame))]
    assert not merchant_outside_allow_list(frame).any()


def test_delegation_clause_sits_above_the_legitimate_tail(background: pd.DataFrame) -> None:
    """The reference's tail runs to 5, so a clause at 5 must cost nothing.

    This is the number the Day 3 contract failed to price. If the reference's
    tail is ever widened further, this fails and the clause has to move with it.
    """
    depth = background["ag_delegation_depth"].dropna().to_numpy(dtype=float)
    assert depth.max() <= MAX_DELEGATION_DEPTH
    from mantis.defense.l0_rules.rules import delegation_too_deep

    assert not delegation_too_deep(background).any()


# --------------------------------------------------------------------------- #
# The Day 3 bucket contract, against the real L0
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("card_id", CLEAN_CARDS)
def test_clean_attacks_trip_no_operative_clause(
    card_id: str, attacks: dict[str, pd.DataFrame]
) -> None:
    """Zero tolerance. The CLEAN half of the Day 3 contract holds against real L0."""
    result = evaluate(attacks[card_id])
    fired = {n: float(m.mean()) for n, m in result.masks.items() if m.any()}
    assert not result.fired.any(), f"{card_id} is declared CLEAN but tripped {fired}"


@pytest.mark.parametrize("card_id", HARD_CARDS)
def test_hard_attacks_fire_a_deployable_clause(
    card_id: str, attacks: dict[str, pd.DataFrame]
) -> None:
    """A HARD card must trip a clause that costs ~nothing on legitimate traffic."""
    result = evaluate(attacks[card_id])
    best = max(float(m.mean()) for m in result.masks.values())
    assert best >= 0.25, f"{card_id} is declared HARD but its best clause fires on {best:.1%}"


def test_f1_05_is_the_documented_exception(attacks: dict[str, pd.DataFrame]) -> None:
    """F1-05 declares HARD and cannot be caught deployably. Pinned, not papered over.

    If this ever starts passing the 25% bar — because L4 landed, or because the
    injector changed — this test fails and the reconciliation in
    ``l0_rules/__main__.py`` needs rewriting. That is the intent: the exception
    should be noisy, not silent.
    """
    result = evaluate(attacks[HARD_BUT_NOT_DEPLOYABLY_CATCHABLE])
    best = max(float(m.mean()) for m in result.masks.values())
    assert best < 0.25, (
        f"{HARD_BUT_NOT_DEPLOYABLY_CATCHABLE} now fires a clause on {best:.1%} of its "
        "events. The Day 4 reconciliation in l0_rules/__main__.py is out of date."
    )


# --------------------------------------------------------------------------- #
# The clause that is declared and switched off
# --------------------------------------------------------------------------- #


def test_the_domain_clause_is_excluded_from_the_verdict(
    background: pd.DataFrame, attacks: dict[str, pd.DataFrame]
) -> None:
    """It fires on the CLEAN attacks, and must not count. That is the whole point.

    If it were operative it would catch both CLEAN cards at ~100% for free, the
    CLEAN bucket would be meaningless, and L3 would inherit a recall it never
    earned by reading a word of text.
    """
    frame = pd.concat([background, attacks["F1-01"]], ignore_index=True)
    frame = frame.sort_values("ts", kind="stable").reset_index(drop=True)
    allow = trusted_domains(frame)
    clause = make_untrusted_domain_clause(allow)
    assert clause.declared_only

    result = evaluate(frame, (*CLAUSES, clause))
    attacked = frame["attack_id"].fillna("").to_numpy() == "F1-01"
    assert result.masks["provenance_untrusted_domain"][attacked].mean() > 0.5, (
        "the declared clause should catch the planted domains; if it does not, the "
        "argument for excluding it no longer applies"
    )
    # ...and it still must not reach the verdict.
    assert not result.fired[attacked].any()


def test_evaluate_names_the_first_clause_that_fired(background: pd.DataFrame) -> None:
    """A reason string is the whole product of a rules layer."""
    result = evaluate(background)
    fired = result.fired
    if fired.any():
        assert all(r != "" for r in result.reason[fired])
    assert all(r == "" for r in result.reason[~fired])
    assert set(np.unique(result.reason[fired])) <= {c.name for c in CLAUSES}
