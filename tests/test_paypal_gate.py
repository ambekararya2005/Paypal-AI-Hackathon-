"""The PayPal authorization gate, in mock mode: no keys, no network.

What these tests pin, in order of how much they matter:

1. **The split of authority is structural.** The agent's handle has no method
   that moves money, and nothing outside ``mantis/paypal/gate.py`` calls capture,
   void or refund — checked by reading the source tree, so a shortcut added later
   fails here rather than in a review.
2. **The three scenarios land where they are declared to.**
3. **The receipt commitment is real**: its hash is the order's ``custom_id``, and
   an order that does not commit to it is voided.
4. The mock is a state machine, not a canned success: PayPal's own 422s happen.
5. The deployed console image stays slim.
"""

from __future__ import annotations

import re
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from mantis.core.events import ALL_COLUMNS, flatten
from mantis.core.paths import REPO_ROOT
from mantis.defense.live import LiveScorer
from mantis.paypal.adapter import to_tx_event
from mantis.paypal.client import AgentPayPal, MockClient, PayPalError
from mantis.paypal.config import ConfigError, PayPalConfig, load_config
from mantis.paypal.gate import PaymentGate
from mantis.paypal.merchants import MERCHANTS
from mantis.paypal.receipt import load_receipt, receipt_hash, verify_custom_id
from mantis.paypal.scenarios import SCENARIOS, demo_mandate, scenario_inputs

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def scorer() -> LiveScorer:
    """Fitted once: L3 takes a couple of seconds and is deterministic.

    ``use_l1=False``: a clean clone has no ``data/models/``, so the tests pin the
    L0 + L3 behaviour rather than whatever a developer has trained locally.
    """
    return LiveScorer(seed=1337, use_l1=False)


def _config(tmp_path: Path) -> PayPalConfig:
    return PayPalConfig(mode="mock", client_id="", client_secret="", receipt_dir=tmp_path)


@pytest.fixture()
def gate(tmp_path: Path, scorer: LiveScorer) -> PaymentGate:
    return PaymentGate(_config(tmp_path), scorer=scorer)


def _run(gate: PaymentGate, key: str):  # -> GateRecord
    mandate = gate.create_mandate(demo_mandate(vault_id=gate.vault_id()))
    intent, context = scenario_inputs(key, mandate.mandate_id)
    return gate.submit(intent, context)


# --------------------------------------------------------------------------- #
# 1. Authority
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name", ["capture_authorization", "void_authorization", "refund_capture", "capture", "void",
             "refund", "get_order"]
)
def test_agent_handle_has_no_money_moving_method(name: str) -> None:
    agent = AgentPayPal(MockClient())
    assert not hasattr(agent, name), f"AgentPayPal exposes {name}"
    assert not hasattr(AgentPayPal, "__getattr__"), "AgentPayPal must not forward attributes"
    with pytest.raises(AttributeError):
        setattr(agent, name, lambda *a, **k: None)  # __slots__: cannot be bolted on


def test_agent_cannot_create_a_capture_intent_order() -> None:
    with pytest.raises(ValueError, match="AUTHORIZE"):
        AgentPayPal(MockClient()).create_order({"intent": "CAPTURE"}, request_id="x")


def test_only_the_gate_calls_capture_void_or_refund() -> None:
    """Read the tree: a money-moving call anywhere but gate.py is a build failure."""
    pattern = re.compile(r"\.(capture_authorization|void_authorization|refund_capture)\(")
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "mantis").rglob("*.py")
        if pattern.search(path.read_text(encoding="utf-8"))
        and path.relative_to(REPO_ROOT).as_posix() != "mantis/paypal/gate.py"
    ]
    assert offenders == [], f"capture/void/refund called outside the gate: {offenders}"


# --------------------------------------------------------------------------- #
# 2. The three scenarios
# --------------------------------------------------------------------------- #


def test_scenario_a_approves_and_captures(gate: PaymentGate) -> None:
    record = _run(gate, "A")
    assert record.decision == "approve"
    assert record.paypal_action == "capture"
    assert record.status == "captured" and record.capture_id
    assert record.amount == "32.00"
    assert record.layers_ran[:2] == ["L0", "L3"]
    assert len(record.reasons) == 3


def test_scenario_b_l0_declines_and_voids(gate: PaymentGate) -> None:
    record = _run(gate, "B")
    assert record.decision == "decline"
    assert record.status == "voided" and record.capture_id is None
    codes = [r["code"] for r in record.reasons]
    assert codes == ["scope_mcc", "scope_merchant", "amount_over_cap"]
    assert all(r["layer"] == "L0" for r in record.reasons)


def test_scenario_c_l3_holds_then_human_denies(gate: PaymentGate) -> None:
    record = _run(gate, "C")
    assert record.decision == "review", "L3 holds for consent; it never voids on its own"
    assert record.status == "held" and record.capture_id is None
    assert record.reasons[0]["layer"] == "L3"
    assert record.reasons[0]["code"] == "injected_content"
    resolved = gate.consent(record.order_id, approve=False, signer="test-human")
    assert resolved.status == "voided"
    assert resolved.consent == {"approve": False, "signer": "test-human",
                                "at": resolved.consent["at"]}


def test_held_order_captures_when_the_human_approves(gate: PaymentGate) -> None:
    record = _run(gate, "C")
    assert gate.consent(record.order_id, approve=True).status == "captured"
    from mantis.paypal.gate import GateError

    with pytest.raises(GateError) as exc:
        gate.consent(record.order_id, approve=False)
    assert exc.value.status == 409


def test_every_scenario_matches_its_declaration(gate: PaymentGate) -> None:
    for scenario in SCENARIOS:
        assert _run(gate, scenario.key).decision == scenario.expect_decision, scenario.key


class _ConfidentL1:
    """A supervised layer that is sure everything is clean, whatever it is shown."""

    name = "L1+fusion"

    def __init__(self) -> None:
        from mantis.defense.policy.decide import PolicyThresholds

        self.thresholds = PolicyThresholds(challenge=0.5, review=0.8, decline=0.9)

    def score_one(self, event: Any, *, l3_score: float) -> tuple[float, list[Any]]:
        return 0.0, []


def test_supervised_layer_cannot_veto_l3(tmp_path: Path, scorer: LiveScorer) -> None:
    """Measured regression: fused weights from the synthetic pool scored scenario C
    at 0.061 against a 0.862 review line, which would have captured an order whose
    agent read an injected page. L1 may escalate; it may not lower."""
    scorer_with_l1 = LiveScorer.__new__(LiveScorer)
    for slot in ("seed", "l3", "l3_hold_at", "_terms", "_coef"):
        setattr(scorer_with_l1, slot, getattr(scorer, slot))
    scorer_with_l1.l1 = _ConfidentL1()
    gate = PaymentGate(_config(tmp_path), scorer=scorer_with_l1)
    record = _run(gate, "C")
    assert "L1+fusion" in record.layers_ran
    assert record.decision == "review" and record.status == "held"


# --------------------------------------------------------------------------- #
# 3. The receipt commitment
# --------------------------------------------------------------------------- #


def test_receipt_hash_round_trips_into_custom_id(gate: PaymentGate, tmp_path: Path) -> None:
    record = _run(gate, "A")
    receipt = load_receipt(record.receipt_hash, tmp_path)
    assert receipt_hash(receipt) == record.receipt_hash
    held = gate._client.get_order(record.order_id)
    assert held["purchase_units"][0]["custom_id"] == record.receipt_hash
    assert verify_custom_id(held, receipt)
    receipt["order"]["total"] = "3200.00"
    assert not verify_custom_id(held, receipt), "a tampered receipt must not verify"


class _InflatingClient(MockClient):
    """A compromised path between agent and PayPal: the order is not the receipt."""

    def create_order(self, body: dict[str, Any], *, request_id: str) -> dict[str, Any]:
        unit = body["purchase_units"][0]
        unit["amount"]["value"] = "99.00"
        unit["custom_id"] = "0" * 64
        return super().create_order(body, request_id=request_id)


def test_order_that_does_not_commit_to_its_receipt_is_voided(
    tmp_path: Path, scorer: LiveScorer
) -> None:
    gate = PaymentGate(_config(tmp_path), client=_InflatingClient(), scorer=scorer)
    record = _run(gate, "A")
    assert record.decision == "decline" and record.status == "voided"
    codes = {r["code"] for r in record.reasons}
    assert {"receipt_mismatch", "amount_mismatch"} <= codes


def test_mandate_edited_after_signing_fails_consent(gate: PaymentGate) -> None:
    mandate = gate.create_mandate(demo_mandate(vault_id=gate.vault_id()))
    widened = mandate.scope.model_copy(update={"max_amount": 10_000.0})
    gate.mandates[mandate.mandate_id] = mandate.model_copy(update={"scope": widened})
    intent, context = scenario_inputs("A", mandate.mandate_id)
    record = gate.submit(intent, context)
    assert record.decision == "decline"
    assert "consent_invalid" in {r["code"] for r in record.reasons}


# --------------------------------------------------------------------------- #
# 4. The mock is PayPal's state machine
# --------------------------------------------------------------------------- #


def test_mock_refuses_capture_after_void_and_is_idempotent(gate: PaymentGate) -> None:
    record = _run(gate, "B")  # voided
    with pytest.raises(PayPalError) as exc:
        gate._client.capture_authorization(record.authorization_id, request_id="late")
    assert exc.value.status == 422 and exc.value.issue == "AUTHORIZATION_VOIDED"

    approved = _run(gate, "A")  # captured with request id <rid>-capture
    rid = f"mantis-{approved.receipt_hash[:32]}-capture"
    again = gate._client.capture_authorization(approved.authorization_id, request_id=rid)
    assert again["id"] == approved.capture_id, "same PayPal-Request-Id must not capture twice"


# --------------------------------------------------------------------------- #
# 5. Schema, config, deploy
# --------------------------------------------------------------------------- #


def test_adapter_emits_the_frozen_schema(gate: PaymentGate) -> None:
    mandate = gate.create_mandate(demo_mandate())
    intent, context = scenario_inputs("A", mandate.mandate_id)
    order = {"id": "ORDER1", "purchase_units": [{"amount": {"currency_code": "USD",
                                                            "value": f"{intent.total:.2f}"}}]}
    event = to_tx_event(order, mandate, context, MERCHANTS["m-freshmart"],
                        mandate_key=gate.config.mandate_key, now=mandate.issued_ts)
    row = flatten(event)
    assert tuple(row) == ALL_COLUMNS
    assert row["channel"] == "agentic" and row["txn_type"] == "preauth"
    assert row["amount"] == float(Decimal("32.00"))
    assert row["ag_consent_sig_valid"] is True and row["ag_kya_registered"] is True
    assert row["ag_human_present"] is False
    assert row["is_fraud"] is False


def test_config_defaults_to_mock_and_live_needs_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in ("MANTIS_PAYPAL_MODE", "PAYPAL_CLIENT_ID", "PAYPAL_CLIENT_SECRET"):
        monkeypatch.delenv(key, raising=False)
    missing = tmp_path / "absent.env"
    assert load_config(env_file=missing).mode == "mock"
    with pytest.raises(ConfigError):
        load_config("live", env_file=missing)
    env = tmp_path / ".env"
    env.write_text("PAYPAL_CLIENT_ID=abc\r\nPAYPAL_CLIENT_SECRET='xyz'\nMANTIS_PAYPAL_MODE=sandbox\n",
                   encoding="utf-8")
    config = load_config(env_file=env)
    assert config.mode == "live" and config.client_secret == "xyz"
    assert "xyz" not in repr(config)
    assert config.base_url == "https://api-m.sandbox.paypal.com"


def test_gate_routes_keep_the_console_image_slim() -> None:
    """``import mantis.api.site`` must not load the scoring stack (pyproject: serve)."""
    code = (
        "import sys, mantis.api.site; "
        "heavy = [m for m in ('pandas','numpy','sklearn','scipy','lightgbm','shap',"
        "'matplotlib','networkx','pyarrow') if m in sys.modules]; print(','.join(heavy))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=REPO_ROOT, check=True)
    assert out.stdout.strip() == "", f"mantis.api.site now imports: {out.stdout.strip()}"


def test_api_gate_round_trip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                             scorer: LiveScorer) -> None:
    from fastapi.testclient import TestClient

    import mantis.api.gate as api_gate
    from mantis.api.app import app

    monkeypatch.setattr(api_gate, "_GATE", PaymentGate(_config(tmp_path), scorer=scorer))
    client = TestClient(app)
    mandate = client.post("/mandates", json=demo_mandate().model_dump(mode="json")).json()
    intent, context = scenario_inputs("C", mandate["mandate_id"])
    body = client.post("/gate", json={"intent": intent.model_dump(mode="json"),
                                      "agent": context.model_dump(mode="json")}).json()
    for key in ("decision", "paypal_action", "order_id", "authorization_id", "capture_id",
                "reasons", "layers_ran", "latency_ms", "receipt_hash"):
        assert key in body
    assert body["status"] == "held" and len(body["reasons"]) == 3
    assert client.get(f"/gate/{body['order_id']}").json()["status"] == "held"
    resolved = client.post(f"/gate/{body['order_id']}/consent", json={"approve": False})
    assert resolved.json()["status"] == "voided"
    assert client.get("/gate/NOT-AN-ORDER").status_code == 404
