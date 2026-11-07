"""Contract tests for the F1 (mandate & delegation) injectors.

What is being defended, in descending order of how badly a failure would hurt:

1. **The HARD/CLEAN split.** This is the whole Day 3 design constraint. If every
   agentic attack were a clean protocol violation, a Day 4 L0 rule would catch
   all of them at near-zero false positive rate, L1 and L2 would have nothing to
   do on the rail the project is about, and the ML story would collapse to "we
   wrote some if-statements". The ``bucket`` class attribute declares which half
   an injector is in, and the tests below check the **behaviour** against the
   declaration rather than taking the declaration's word for it.
2. **CLEAN really means clean.** A CLEAN injector that quietly emitted a few
   scope violations would draw recall from L0 while claiming to be behavioural.
   That is the single easiest way to fool ourselves on this project, so it is
   asserted per clause, at zero tolerance.
3. **HARD really fires.** A HARD injector whose violation rate drifted to zero
   would silently move into the CLEAN bucket and L0's recall claim would become
   untestable.
4. **The content join.** ``ingested_content_ids`` must resolve to text, on
   attack rows *and* on legitimate ones. If only attacked ids resolved, "does
   this id resolve" would be a perfect label and L3 would score a fake 1.0
   without reading a word.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mantis.atlas.loader import ATLAS
from mantis.core.events import MandateType, TxnType
from mantis.foundry.base.reference import ReferenceStats, load_reference_stats
from mantis.foundry.base.simulator import SimulationConfig, simulate_frame
from mantis.foundry.injectors import REGISTRY
from mantis.foundry.injectors.agentic import (
    ATTACKER_DOMAINS,
    AgenticAttack,
    Bucket,
)
from mantis.foundry.injectors.base import PopulationView, run_injector
from mantis.foundry.llm.corpus import content_id_for_url, load_content_store

#: Big enough that a 100-event campaign has entities to work with, small enough
#: that the module runs in seconds.
SMALL = SimulationConfig(n_events=30_000, seed=7, n_customers=1_200, n_merchants=3_000)

#: The seven F1 cards Day 3 delivers, and the bucket each must be in. Written out
#: rather than read off the classes: this is the contract, and a test that read
#: the declaration from the thing it is testing would assert nothing.
EXPECTED_BUCKETS: dict[str, str] = {
    "F1-01": Bucket.CLEAN,
    "F1-02": Bucket.HARD,
    "F1-03": Bucket.CLEAN,
    "F1-04": Bucket.HARD,
    "F1-05": Bucket.HARD,
    "F1-09": Bucket.HARD,
    "F1-10": Bucket.HARD,
}

CLEAN_CARDS = sorted(c for c, b in EXPECTED_BUCKETS.items() if b == Bucket.CLEAN)
HARD_CARDS = sorted(c for c, b in EXPECTED_BUCKETS.items() if b == Bucket.HARD)


@pytest.fixture(scope="module")
def stats() -> ReferenceStats:
    return load_reference_stats()


@pytest.fixture(scope="module")
def background(stats: ReferenceStats) -> pd.DataFrame:
    return simulate_frame(SMALL, stats)


@pytest.fixture(scope="module")
def view(background: pd.DataFrame) -> PopulationView:
    return PopulationView.build(background)


@pytest.fixture(scope="module")
def agentic_attacks(view: PopulationView) -> dict[str, pd.DataFrame]:
    load_content_store()
    return {card_id: run_injector(REGISTRY[card_id], view, seed=7) for card_id in EXPECTED_BUCKETS}


# --------------------------------------------------------------------------- #
# L0 clause helpers — one function per rule a Day 4 L0 layer will implement
# --------------------------------------------------------------------------- #


def mcc_outside_scope(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        [str(r.mcc) not in list(r.ag_scope_categories) for r in frame.itertuples()], dtype=bool
    )


def amount_over_ceiling(frame: pd.DataFrame) -> np.ndarray:
    return (frame["amount"] > frame["ag_scope_max_amount"]).to_numpy()


def mandate_expired(frame: pd.DataFrame) -> np.ndarray:
    age = (frame["ts"] - frame["ag_mandate_issued_ts"]).dt.total_seconds()
    return (age >= frame["ag_mandate_ttl_seconds"]).to_numpy()


def consent_invalid(frame: pd.DataFrame) -> np.ndarray:
    return ~frame["ag_consent_sig_valid"].astype(bool).to_numpy()


def kya_unregistered(frame: pd.DataFrame) -> np.ndarray:
    return ~frame["ag_kya_registered"].astype(bool).to_numpy()


def merchant_off_allowlist(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        [
            len(r.ag_scope_allowed_merchants) > 0
            and r.merchant_id not in list(r.ag_scope_allowed_merchants)
            for r in frame.itertuples()
        ],
        dtype=bool,
    )


#: Every L0 clause, by the name the atlas cards use for it.
L0_CLAUSES = {
    "mcc_outside_scope": mcc_outside_scope,
    "amount_over_ceiling": amount_over_ceiling,
    "mandate_expired": mandate_expired,
    "consent_sig_invalid": consent_invalid,
    "merchant_not_in_allowlist": merchant_off_allowlist,
}


# --------------------------------------------------------------------------- #
# 1. The split itself
# --------------------------------------------------------------------------- #


def test_every_f1_injector_declares_a_bucket() -> None:
    """A missing declaration is a silent opt-out of the Day 4 assertion."""
    for card_id, expected in EXPECTED_BUCKETS.items():
        cls = REGISTRY[card_id]
        assert issubclass(cls, AgenticAttack), f"{card_id} is not an AgenticAttack"
        assert getattr(cls, "bucket", None) == expected, card_id


def test_the_split_is_actually_split() -> None:
    """Both buckets must be non-empty, or the design constraint was not met.

    This is the test that fails if someone "simplifies" Day 3 by making every
    agentic attack a protocol violation. It is deliberately blunt.
    """
    assert CLEAN_CARDS, "no CLEAN-bucket agentic attack: L1/L2/L3 have nothing to do"
    assert HARD_CARDS, "no HARD-bucket agentic attack: L0 has nothing to prove"
    assert len(CLEAN_CARDS) >= 2


@pytest.mark.parametrize("card_id", CLEAN_CARDS)
def test_clean_bucket_violates_no_l0_clause(
    card_id: str, agentic_attacks: dict[str, pd.DataFrame]
) -> None:
    """Zero tolerance, per clause.

    A CLEAN attack that tripped even a few percent of a rule would let us report
    behavioural detection while quietly drawing recall from L0. There is no
    threshold here on purpose: the claim is that these attacks are
    *cryptographically and procedurally perfect*, and "mostly perfect" is a
    different and much weaker claim.
    """
    frame = agentic_attacks[card_id]
    for name, clause in L0_CLAUSES.items():
        fired = clause(frame)
        assert not fired.any(), (
            f"{card_id} is declared CLEAN but trips {name} on "
            f"{int(fired.sum())}/{len(frame)} events"
        )


@pytest.mark.parametrize("card_id", HARD_CARDS)
def test_hard_bucket_fires_a_deterministic_clause(
    card_id: str, agentic_attacks: dict[str, pd.DataFrame]
) -> None:
    """A HARD attack must trip something an issuer can check on one message.

    F1-05's clause is delegation depth against issuer policy rather than one of
    the mandate clauses, and F1-10's is mandate freshness, so the check is "at
    least one deterministic signal fires on a substantial share" rather than a
    named clause per card.
    """
    frame = agentic_attacks[card_id]
    rates = {name: float(clause(frame).mean()) for name, clause in L0_CLAUSES.items()}
    rates["delegation_depth_over_2"] = float((frame["ag_delegation_depth"] > 2).mean())
    rates["kya_unregistered"] = float(kya_unregistered(frame).mean())
    rates["mandate_hash_reused"] = float(frame["ag_mandate_hash"].duplicated().mean())

    strongest = max(rates.values())
    assert strongest >= 0.25, (
        f"{card_id} is declared HARD but no deterministic clause fires on more than "
        f"{strongest:.1%} of events: {rates}"
    )


# --------------------------------------------------------------------------- #
# 2. Each card's own claimed signal
# --------------------------------------------------------------------------- #


def test_f1_02_inflates_scope_both_ways(agentic_attacks: dict[str, pd.DataFrame]) -> None:
    """Category drift and amount inflation, mixed rather than merged."""
    frame = agentic_attacks["F1-02"]
    assert (frame["ag_mandate_type"] == MandateType.INTENT.value).all()
    drift = mcc_outside_scope(frame).mean()
    over = amount_over_ceiling(frame).mean()
    assert 0.25 < drift < 0.85, f"category drift rate {drift:.2f}"
    assert 0.15 < over < 0.75, f"ceiling breach rate {over:.2f}"
    # Merged would mean one rule takes the whole card and neither clause is
    # separately measurable.
    both = (mcc_outside_scope(frame) & amount_over_ceiling(frame)).mean()
    assert both < max(drift, over), "every event trips every clause; the shapes were merged"


def test_f1_04_drifts_category_without_touching_the_ceiling(
    agentic_attacks: dict[str, pd.DataFrame],
) -> None:
    """The quiet sibling of F1-02: category only, amount uninformative."""
    frame = agentic_attacks["F1-04"]
    assert mcc_outside_scope(frame).all(), "F1-04 is a category-drift attack"
    assert not amount_over_ceiling(frame).any(), "F1-04 must not breach the ceiling"
    ratio = frame["amount"] / frame["ag_scope_max_amount"]
    assert ratio.median() < 0.8, "F1-04 sits too close to the ceiling to be the quiet one"


def test_f1_04_settles_in_a_category_the_customer_has_never_used(
    agentic_attacks: dict[str, pd.DataFrame], background: pd.DataFrame
) -> None:
    """``mcc_novelty_for_customer`` must be true, not merely likely."""
    frame = agentic_attacks["F1-04"]
    history = background.groupby("customer_id")["mcc"].agg(set)
    novel = [row.mcc not in history.get(row.customer_id, set()) for row in frame.itertuples()]
    assert all(novel), "F1-04 drifted into a category the customer already shops in"


def test_f1_05_launders_through_a_few_shared_subagents(
    agentic_attacks: dict[str, pd.DataFrame],
) -> None:
    """Depth, fan-out and a shared artefact — the three signals on the card."""
    frame = agentic_attacks["F1-05"]
    assert frame["ag_delegation_depth"].max() >= 4

    # Fan-out: far more principals than executing identities.
    per_campaign = frame.groupby("attack_campaign").agg(
        agents=("ag_agent_id", "nunique"), customers=("customer_id", "nunique")
    )
    assert (per_campaign["customers"] > per_campaign["agents"]).all(), (
        "no agent fan-out: each sub-agent serves at most one principal"
    )

    # A mandate artefact presented by more than one identity.
    shared = frame.groupby("ag_mandate_hash")["ag_agent_id"].nunique()
    assert (shared > 1).any(), "no mandate hash spans sub-agent identities"


def test_f1_05_executes_on_a_platform_the_customer_has_not_used(
    agentic_attacks: dict[str, pd.DataFrame], background: pd.DataFrame
) -> None:
    frame = agentic_attacks["F1-05"]
    history = background.groupby("customer_id")["ag_agent_platform"].agg(lambda s: set(s.dropna()))
    novel = [
        row.ag_agent_platform not in history.get(row.customer_id, set())
        for row in frame.itertuples()
    ]
    assert np.mean(novel) > 0.85


def test_f1_09_claims_presence_on_every_event(agentic_attacks: dict[str, pd.DataFrame]) -> None:
    frame = agentic_attacks["F1-09"]
    assert frame["ag_human_present"].astype(bool).all()
    # Both sub-shapes present: some events forged the telemetry, some did not.
    assert consent_invalid(frame).mean() > 0.2


def test_f1_09_is_not_caught_by_the_obvious_two_column_rule(
    agentic_attacks: dict[str, pd.DataFrame], background: pd.DataFrame
) -> None:
    """The presence/telemetry mismatch must not be a perfect detector.

    If it were, we would be reporting a property of this generator rather than
    of spoofing. The population's passive-human tail (``human_present_passive_share``)
    is what stops it, and this is the test that keeps that tail load-bearing.
    """
    frame = agentic_attacks["F1-09"]
    agentic_bg = background[background["ag_agent_id"].notna()]
    unwatched = agentic_bg[~agentic_bg["ag_human_present"].astype(bool)]
    floor = unwatched["ag_cursor_entropy"].quantile(0.75)

    def rule(f: pd.DataFrame) -> np.ndarray:
        return (f["ag_human_present"].astype(bool) & (f["ag_cursor_entropy"] <= floor)).to_numpy()

    recall = rule(frame).mean()
    false_positive = rule(agentic_bg).mean()
    assert recall < 0.9, f"the mismatch rule alone catches {recall:.1%} of F1-09"
    assert false_positive > 0.0, "no legitimate session trips the rule; the tail has gone"


def test_f1_10_replays_one_artefact_many_times(
    agentic_attacks: dict[str, pd.DataFrame],
) -> None:
    frame = agentic_attacks["F1-10"]
    reuse = frame["ag_mandate_hash"].value_counts()
    assert (reuse > 1).any(), "no mandate hash is reused; this is not a replay attack"
    assert mandate_expired(frame).mean() > 0.3, "too few replays arrive past the TTL"
    # The original presentation of each artefact was legitimate and unexpired.
    fresh = frame[~mandate_expired(frame)]
    assert len(fresh) > 0, "every presentation is expired; replay means re-use"


def test_f1_03_credits_are_bound_and_never_exceed_their_purchase(
    agentic_attacks: dict[str, pd.DataFrame], background: pd.DataFrame
) -> None:
    """The CLEAN shape: a matching authorisation, and no per-transaction breach."""
    frame = agentic_attacks["F1-03"]
    assert (frame["txn_type"] == TxnType.REFUND.value).all()
    assert frame["original_event_id"].notna().all(), (
        "an orphan credit is an L0 catch and belongs in a HARD-bucket card"
    )

    by_id = background.set_index("event_id")
    assert frame["original_event_id"].isin(by_id.index).all()
    source = by_id.loc[frame["original_event_id"]]
    assert (frame["amount"].to_numpy() <= source["amount"].to_numpy() + 0.01).all()
    assert (source["customer_id"].to_numpy() == frame["customer_id"].to_numpy()).all()
    assert (source["merchant_id"].to_numpy() == frame["merchant_id"].to_numpy()).all()


def test_f1_03_over_refunds_in_aggregate(agentic_attacks: dict[str, pd.DataFrame]) -> None:
    """Individually defensible, collectively not. That is the whole attack."""
    frame = agentic_attacks["F1-03"]
    per_original = frame.groupby("original_event_id").size()
    assert (per_original > 1).any(), "each purchase is credited once; nothing to detect"


def test_f1_01_stays_under_the_ceiling_but_close_to_it(
    agentic_attacks: dict[str, pd.DataFrame],
) -> None:
    frame = agentic_attacks["F1-01"]
    assert (frame["ag_mandate_type"] == MandateType.CART.value).all()
    ratio = frame["amount"] / frame["ag_scope_max_amount"]
    assert ratio.max() <= 1.0
    assert ratio.median() > 0.8, "the tampered total is not being pushed toward the ceiling"


def test_f1_01_lands_on_a_merchant_the_customer_has_never_used(
    agentic_attacks: dict[str, pd.DataFrame], background: pd.DataFrame
) -> None:
    frame = agentic_attacks["F1-01"]
    history = background.groupby("customer_id")["merchant_id"].agg(set)
    novel = [
        row.merchant_id not in history.get(row.customer_id, set()) for row in frame.itertuples()
    ]
    assert np.mean(novel) > 0.9


# --------------------------------------------------------------------------- #
# 3. Provenance and the content join
# --------------------------------------------------------------------------- #

#: The cards that plant injected content. F1-02, F1-04, F1-05 and F1-10 do not:
#: their attacks need no injected text, and adding some would blur the bucket
#: split by giving a HARD attack a CLEAN attack's evidence.
CONTENT_CARDS = ("F1-01", "F1-03")


@pytest.mark.parametrize("card_id", CONTENT_CARDS)
def test_provenance_runs_through_attacker_content(
    card_id: str, agentic_attacks: dict[str, pd.DataFrame]
) -> None:
    """The trail must be causal: attacker pages, then the merchant that was paid."""
    frame = agentic_attacks[card_id]
    for row in frame.itertuples():
        chain = list(row.ag_provenance_chain)
        assert len(chain) >= 2
        hosts = [url.split("/")[2] for url in chain]
        assert any(host in ATTACKER_DOMAINS for host in hosts[:-1]), (
            f"{card_id}: no attacker domain on the chain"
        )
        assert chain[-1].startswith("https://shop."), (
            f"{card_id}: the chain must still end at the merchant that was paid"
        )


@pytest.mark.parametrize("card_id", CONTENT_CARDS)
def test_content_ids_match_the_chain_and_resolve_to_text(
    card_id: str, agentic_attacks: dict[str, pd.DataFrame]
) -> None:
    """``ingested_content_ids`` is the join key, and it must actually join."""
    store = load_content_store()
    frame = agentic_attacks[card_id]
    saw_adversarial = False
    for row in frame.itertuples():
        chain = list(row.ag_provenance_chain)
        ids = list(row.ag_ingested_content_ids)
        assert ids == [content_id_for_url(url) for url in chain], (
            f"{card_id}: content ids do not match the chain they claim to digest"
        )
        for content_id in ids:
            artifact = store.resolve(content_id)
            assert artifact is not None, f"{card_id}: {content_id} resolves to nothing"
            assert artifact.text.strip(), f"{card_id}: {content_id} resolves to empty text"
            saw_adversarial |= artifact.is_injected
    assert saw_adversarial, f"{card_id}: nothing on any chain is adversarial"


def test_legitimate_content_ids_resolve_too(background: pd.DataFrame) -> None:
    """Otherwise 'does this id resolve' is a perfect label and L3 scores a fake 1.0.

    This is the leakage control that matters most for the text layer, and it is
    why the content store assigns unbound ids into the benign pool
    deterministically instead of returning a miss.
    """
    store = load_content_store()
    agentic = background[background["ag_ingested_content_ids"].notna()].head(200)
    assert len(agentic) > 0
    for ids in agentic["ag_ingested_content_ids"]:
        for content_id in ids:
            artifact = store.resolve(content_id)
            assert artifact is not None, f"legitimate {content_id} resolves to nothing"
            assert not artifact.is_injected, "a legitimate chain resolved to adversarial content"


# --------------------------------------------------------------------------- #
# 4. Atlas agreement
# --------------------------------------------------------------------------- #


def test_f1_cards_are_implemented_and_name_their_injector() -> None:
    for card_id in EXPECTED_BUCKETS:
        card = ATLAS[card_id]
        assert card.status.value == "implemented", card_id
        assert card.generator is not None
        assert card.generator.split(":")[0] == REGISTRY[card_id].__module__


def test_f1_injectors_reuse_existing_agent_identities(
    agentic_attacks: dict[str, pd.DataFrame], background: pd.DataFrame
) -> None:
    """A never-before-seen agent id would be caught by novelty alone.

    The interesting case is a real, registered capability behaving badly -- and
    an attack that only ever used fresh identities would teach a detector
    nothing that transfers.
    """
    known = set(background["ag_agent_id"].dropna())
    for card_id, frame in agentic_attacks.items():
        used = set(frame["ag_agent_id"].dropna())
        assert used <= known, f"{card_id} invented {len(used - known)} agent identities"
