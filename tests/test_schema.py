"""Round-trip and contract tests for the frozen event schema.

If a test here fails, either the schema changed without CLAUDE.md §4 being
followed, or something downstream is about to silently produce wrong metrics.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from mantis.core.events import (
    AGENTIC_COLUMNS,
    ALL_COLUMNS,
    CLASSIC_COLUMNS,
    DECLINE_RESPONSES,
    LABEL_COLUMNS,
    POST_HOC_COLUMNS,
    SCHEMA_VERSION,
    AgenticContext,
    AuthResponse,
    Channel,
    DisputeOutcome,
    EntryMode,
    MandateScope,
    MandateType,
    ThreeDSResult,
    TxEvent,
    TxnType,
    flatten,
)

ISSUED = datetime(2026, 8, 21, 9, 15, tzinfo=UTC)


def make_agentic_event(**overrides: object) -> TxEvent:
    """A well-formed agentic authorisation under a cart mandate."""
    defaults: dict[str, object] = {
        "event_id": "evt-agentic-0001",
        "ts": ISSUED + timedelta(seconds=42),
        "amount": 7499.0,
        "currency": "INR",
        "mcc": "5812",
        "channel": Channel.AGENTIC,
        "entry_mode": EntryMode.AGENT_TOKEN,
        "customer_id": "cust-000123",
        "card_bin": "512345",
        "merchant_id": "mer-88120",
        "merchant_country": "IN",
        "device_id": "dev-aa17",
        "ip": "203.0.113.44",
        "lat": 19.076,
        "lon": 72.8777,
        "threeds_result": ThreeDSResult.FRICTIONLESS,
        "agentic": AgenticContext(
            agent_id="agt-77aa",
            agent_platform="demo-agent-runtime",
            kya_token="kya-4f2c",
            kya_registered=True,
            mandate_type=MandateType.CART,
            mandate_id="mnd-0001",
            mandate_hash="9f2b7c1d",
            mandate_issued_ts=ISSUED,
            mandate_ttl_seconds=900,
            mandate_scope=MandateScope(
                categories=["5812", "5814"],
                max_amount=8000.0,
                max_items=4,
                allowed_merchants=["mer-88120"],
                ttl_seconds=900,
            ),
            human_present=True,
            consent_sig_valid=True,
            delegation_depth=1,
            provenance_chain=[
                "https://example-shop.test/menu",
                "https://example-reviews.test/listing/88120",
            ],
            ingested_content_ids=["sha256:aaa", "sha256:bbb"],
            tool_call_count=6,
            deliberation_latency_ms=3200,
            cursor_entropy=4.21,
            dwell_time_ms=18400,
        ),
    }
    defaults.update(overrides)
    return TxEvent(**defaults)  # type: ignore[arg-type]


def make_classic_event(**overrides: object) -> TxEvent:
    """A well-formed card-present authorisation with no agentic block."""
    defaults: dict[str, object] = {
        "event_id": "evt-classic-0001",
        "ts": ISSUED,
        "amount": 1299.5,
        "currency": "INR",
        "mcc": "5411",
        "channel": Channel.CARD_PRESENT,
        "entry_mode": EntryMode.CHIP,
        "customer_id": "cust-000456",
        "card_bin": "45678901",
        "merchant_id": "mer-11002",
        "merchant_country": "IN",
        "terminal_id": "trm-3391",
        "device_id": None,
        "ip": None,
        "lat": 12.9716,
        "lon": 77.5946,
        "threeds_result": ThreeDSResult.NOT_APPLICABLE,
    }
    defaults.update(overrides)
    return TxEvent(**defaults)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Round trips
# --------------------------------------------------------------------------- #


def test_agentic_event_round_trips() -> None:
    ev = make_agentic_event()
    restored = TxEvent.model_validate(ev.model_dump())

    assert restored == ev
    assert restored.agentic is not None
    assert restored.agentic.mandate_type is MandateType.CART
    assert restored.agentic.mandate_scope is not None
    assert restored.agentic.mandate_scope.max_amount == 8000.0
    # provenance_chain is load-bearing for L3: it must survive intact and ordered.
    assert restored.agentic.provenance_chain == ev.agentic.provenance_chain  # type: ignore[union-attr]


def test_agentic_event_round_trips_through_json() -> None:
    ev = make_agentic_event()
    restored = TxEvent.model_validate_json(ev.model_dump_json())
    assert restored == ev


def test_classic_event_round_trips() -> None:
    ev = make_classic_event()
    restored = TxEvent.model_validate(ev.model_dump())

    assert restored == ev
    assert restored.agentic is None
    assert restored.terminal_id == "trm-3391"


def test_classic_event_round_trips_through_json() -> None:
    ev = make_classic_event()
    assert TxEvent.model_validate_json(ev.model_dump_json()) == ev


# --------------------------------------------------------------------------- #
# flatten() column contract
# --------------------------------------------------------------------------- #


def test_flatten_key_set_is_identical_for_both_rails() -> None:
    """Classic and agentic rows must share an exact key set, or frames go ragged."""
    agentic_row = flatten(make_agentic_event())
    classic_row = flatten(make_classic_event())

    assert tuple(agentic_row) == ALL_COLUMNS
    assert tuple(classic_row) == ALL_COLUMNS


def test_flatten_classic_leaves_every_agentic_column_none() -> None:
    row = flatten(make_classic_event())
    assert all(row[col] is None for col in AGENTIC_COLUMNS)


def test_flatten_prefixes_and_explodes_agentic_fields() -> None:
    row = flatten(make_agentic_event())

    assert row["ag_agent_id"] == "agt-77aa"
    assert row["ag_mandate_type"] == "cart"
    assert row["ag_scope_max_amount"] == 8000.0
    assert row["ag_scope_allowed_merchants"] == ["mer-88120"]
    assert row["ag_provenance_chain"][0] == "https://example-shop.test/menu"
    # Enums flatten to their string values, not Enum members.
    assert row["channel"] == "agentic"
    assert isinstance(row["entry_mode"], str)


def test_label_columns_are_present_and_last() -> None:
    """Labels must be emitted (evaluation needs them) and easy to drop by name."""
    row = flatten(make_classic_event())
    assert set(LABEL_COLUMNS) <= set(row)
    assert ALL_COLUMNS[-len(LABEL_COLUMNS) :] == LABEL_COLUMNS


def test_flatten_does_not_alias_mutable_fields() -> None:
    """A flattened row must not hand out references into the event."""
    ev = make_agentic_event()
    row = flatten(ev)
    row["ag_provenance_chain"].append("https://attacker.test/injected")
    assert ev.agentic is not None
    assert "https://attacker.test/injected" not in ev.agentic.provenance_chain


# --------------------------------------------------------------------------- #
# Validation guardrails
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_mcc", ["581", "58122", "58a2", ""])
def test_mcc_must_be_four_digits(bad_mcc: str) -> None:
    with pytest.raises(ValidationError):
        make_classic_event(mcc=bad_mcc)


@pytest.mark.parametrize("bad_currency", ["inr", "INRX", "I1R"])
def test_currency_must_be_iso4217(bad_currency: str) -> None:
    with pytest.raises(ValidationError):
        make_classic_event(currency=bad_currency)


def test_agentic_channel_requires_agentic_context() -> None:
    with pytest.raises(ValidationError, match="requires an AgenticContext"):
        make_classic_event(channel=Channel.AGENTIC, entry_mode=EntryMode.AGENT_TOKEN)


def test_agentic_context_on_classic_rail_requires_agent_token() -> None:
    ctx = make_agentic_event().agentic
    with pytest.raises(ValidationError, match="agent_token"):
        make_classic_event(agentic=ctx)


def test_agentic_context_on_ecom_rail_is_allowed_with_agent_token() -> None:
    """Agent-originated traffic disguised as plain ecom must stay representable."""
    ev = make_classic_event(
        channel=Channel.ECOM,
        entry_mode=EntryMode.AGENT_TOKEN,
        terminal_id=None,
        agentic=make_agentic_event().agentic,
    )
    assert ev.agentic is not None


def test_label_integrity_is_enforced_both_ways() -> None:
    with pytest.raises(ValidationError, match="non-fraud event"):
        make_classic_event(attack_id="F1-01")
    with pytest.raises(ValidationError, match="requires an attack_id"):
        make_classic_event(is_fraud=True)


def test_labelled_fraud_event_is_valid() -> None:
    ev = make_agentic_event(is_fraud=True, attack_id="F1-01", attack_campaign="camp-001")
    assert flatten(ev)["attack_id"] == "F1-01"


def test_assignment_is_revalidated() -> None:
    """validate_assignment keeps injectors from writing an invalid event."""
    ev = make_classic_event()
    with pytest.raises(ValidationError):
        ev.amount = -1.0


def test_unknown_field_is_rejected() -> None:
    """extra='forbid' is what keeps the frozen schema actually frozen."""
    with pytest.raises(ValidationError):
        make_classic_event(risk_score=0.9)


# --------------------------------------------------------------------------- #
# Amendment 1.1.0: the transaction lifecycle
# --------------------------------------------------------------------------- #

#: The seventeen columns the Day 0 freeze emitted, in order. Pinned literally.
#: The amendment is additive, and "additive" means a 1.0.0 reader slicing the
#: front of the frame still gets exactly what it got before. If this list ever
#: needs editing, the change was not additive and CLAUDE.md section 4 applies.
V1_CLASSIC_COLUMNS = (
    "event_id",
    "ts",
    "amount",
    "currency",
    "mcc",
    "channel",
    "entry_mode",
    "customer_id",
    "card_bin",
    "merchant_id",
    "merchant_country",
    "terminal_id",
    "device_id",
    "ip",
    "lat",
    "lon",
    "threeds_result",
)


def test_schema_version_is_recorded() -> None:
    assert SCHEMA_VERSION == "1.1.0"


def test_amendment_is_purely_additive() -> None:
    """Nothing was renamed, retyped, reordered or removed."""
    assert CLASSIC_COLUMNS[: len(V1_CLASSIC_COLUMNS)] == V1_CLASSIC_COLUMNS
    assert ALL_COLUMNS[-len(LABEL_COLUMNS) :] == LABEL_COLUMNS
    assert set(POST_HOC_COLUMNS) <= set(ALL_COLUMNS)
    assert set(POST_HOC_COLUMNS).isdisjoint(LABEL_COLUMNS)


def test_lifecycle_defaults_reproduce_a_settled_approved_purchase() -> None:
    """Every 1.0.0 call site keeps working, unchanged, with no new arguments."""
    ev = make_classic_event()
    assert ev.txn_type is TxnType.PURCHASE
    assert ev.auth_response is AuthResponse.APPROVED
    assert ev.original_event_id is None
    assert ev.dispute_outcome is None
    assert ev.settled is True
    assert ev.settlement_lag_hours is None

    row = flatten(ev)
    assert row["txn_type"] == "purchase"
    assert row["auth_response"] == "approved"
    assert row["settled"] is True


def test_decline_responses_names_every_non_approval() -> None:
    assert set(DECLINE_RESPONSES) == {r.value for r in AuthResponse} - {"approved"}
    assert "approved" not in DECLINE_RESPONSES


def test_a_declined_authorisation_cannot_settle() -> None:
    """An impossible combination a model would happily memorise."""
    with pytest.raises(ValidationError, match="cannot settle"):
        make_classic_event(auth_response=AuthResponse.DECLINED_RISK)
    with pytest.raises(ValidationError, match="declined authorisation"):
        make_classic_event(
            auth_response=AuthResponse.DECLINED_RISK, settled=False, settlement_lag_hours=4.0
        )
    ok = make_classic_event(auth_response=AuthResponse.DECLINED_INVALID_CVV, settled=False)
    assert ok.settlement_lag_hours is None


def test_unsettled_authorisation_carries_no_lag() -> None:
    with pytest.raises(ValidationError, match="unsettled"):
        make_classic_event(settled=False, settlement_lag_hours=12.0)


def test_original_event_id_only_on_outbound_flows() -> None:
    with pytest.raises(ValidationError, match="only refunds, reversals and credits"):
        make_classic_event(original_event_id="evt-classic-9999")
    with pytest.raises(ValidationError, match="itself"):
        make_classic_event(txn_type=TxnType.REFUND, original_event_id="evt-classic-0001")

    ev = make_classic_event(txn_type=TxnType.REFUND, original_event_id="evt-classic-9999")
    assert ev.original_event_id == "evt-classic-9999"


def test_an_orphan_credit_stays_representable() -> None:
    """The central F1-03 shape: money going out with no purchase behind it.

    A schema that required ``original_event_id`` on a credit would make the
    attack unrepresentable, exactly the way 1.0.0 did. It must stay legal, and
    its detection must be the firewall's job rather than the validator's.
    """
    ev = make_classic_event(txn_type=TxnType.CREDIT, settlement_lag_hours=0.1)
    assert ev.original_event_id is None
    assert flatten(ev)["txn_type"] == "credit"


def test_dispute_fields_must_agree() -> None:
    raised = ISSUED + timedelta(days=9)
    with pytest.raises(ValidationError, match="requires dispute_raised_ts"):
        make_classic_event(dispute_outcome=DisputeOutcome.WON_CARDHOLDER)
    with pytest.raises(ValidationError, match="no dispute recorded"):
        make_classic_event(dispute_raised_ts=raised)
    with pytest.raises(ValidationError, match="before the authorisation"):
        make_classic_event(
            dispute_outcome=DisputeOutcome.RAISED, dispute_raised_ts=ISSUED - timedelta(days=1)
        )

    ev = make_classic_event(dispute_outcome=DisputeOutcome.RAISED, dispute_raised_ts=raised)
    assert flatten(ev)["dispute_outcome"] == "raised"
    # dispute_outcome='none' is the explicit "checked, nothing happened" value
    # and must not require a timestamp.
    assert make_classic_event(dispute_outcome=DisputeOutcome.NONE).dispute_raised_ts is None


def test_amount_stays_non_negative_on_outbound_flows() -> None:
    """Direction is carried by txn_type, not by the sign of the amount.

    A signed amount would silently break every quantile, KS distance and
    log-amount feature calibrated against the Day 1 population.
    """
    with pytest.raises(ValidationError):
        make_classic_event(txn_type=TxnType.REFUND, amount=-100.0)
    assert make_classic_event(txn_type=TxnType.REFUND, amount=100.0).amount == 100.0


def test_lifecycle_round_trips_through_json() -> None:
    ev = make_agentic_event(
        txn_type=TxnType.REFUND,
        original_event_id="evt-agentic-0000",
        settlement_lag_hours=0.25,
        dispute_outcome=DisputeOutcome.WON_MERCHANT,
        dispute_raised_ts=ISSUED + timedelta(days=30),
    )
    assert TxEvent.model_validate_json(ev.model_dump_json()) == ev


def test_mandate_expiry_helper() -> None:
    ctx = make_agentic_event().agentic
    assert ctx is not None
    assert ctx.expires_at() == ISSUED + timedelta(seconds=900)

    ctx_no_ttl = AgenticContext(agent_id="a", agent_platform="p")
    assert ctx_no_ttl.expires_at() is None
