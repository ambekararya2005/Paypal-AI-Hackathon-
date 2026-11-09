"""The authorization gate: the agent reserves, MANTIS decides, MANTIS moves money.

    python -m mantis.paypal demo           # the three scenarios, end to end

The whole design in four lines
--------------------------------
1. A human signs a :class:`~mantis.paypal.models.Mandate` once.
2. The agent, holding only :class:`~mantis.paypal.client.AgentPayPal`, creates an
   ``intent=AUTHORIZE`` order and authorizes it. Funds are **reserved; nothing
   has moved**.
3. The gate reads the order back from PayPal, checks it commits to the receipt,
   maps it onto a ``TxEvent`` and scores it.
4. ``APPROVE`` -> **capture**. ``DECLINE`` -> **void**. ``CHALLENGE`` / ``REVIEW``
   -> **hold** the authorization until a human approves (capture) or denies
   (void) it through :meth:`PaymentGate.consent`.

This module is the **only** caller of ``capture_authorization`` and
``void_authorization`` in the package; ``tests/test_paypal_gate.py`` asserts it by
reading the source tree, so a future shortcut fails a test rather than a review.

In the demo the gate drives the agent's two calls itself, through the
``AgentPayPal`` handle, so one ``POST /gate`` shows the whole path. The separation
of authority is carried by the types — the agent's handle has no capture method
to call — not by a process boundary, which a deployment would add.

Idempotency
------------
Every PayPal POST carries ``PayPal-Request-Id = mantis-<receipt hash>-<step>``.
A retried gate call after a timeout re-sends the same ids, and PayPal answers from
its idempotency cache instead of creating a second order or a second capture.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Final

from mantis.defense.l3_judge import InjectionJudge
from mantis.defense.live import LiveScorer, Reason
from mantis.defense.policy.decide import Decision
from mantis.paypal.adapter import WALLET_BIN, order_amount, order_body, to_tx_event
from mantis.paypal.client import (
    AgentPayPal,
    PayPalClient,
    PayPalError,
    first_authorization,
    make_client,
)
from mantis.paypal.config import PayPalConfig, load_config
from mantis.paypal.merchants import MERCHANTS
from mantis.paypal.models import AgentContext, Mandate, MandateRequest, PurchaseIntent
from mantis.paypal.receipt import build_receipt, receipt_hash, save_receipt, verify_custom_id

__all__ = ["GateError", "GateRecord", "PaymentGate", "PaymentSource"]

#: What the gate does with each decision.
ACTION: Final[dict[Decision, str]] = {
    Decision.APPROVE: "capture",
    Decision.DECLINE: "void",
    Decision.CHALLENGE: "hold",
    Decision.REVIEW: "hold",
}

PaymentSource = str  # "card" | "vault" | "auto"


class GateError(RuntimeError):
    """A request the gate refuses, with the HTTP status the API should return."""

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        super().__init__(message)


@dataclass(slots=True)
class GateRecord:
    """Everything the gate decided about one order. Also the API response body."""

    order_id: str
    authorization_id: str
    decision: str
    paypal_action: str
    status: str
    reasons: list[dict[str, str]]
    layers_ran: list[str]
    latency_ms: float
    receipt_hash: str
    mandate_id: str
    merchant_id: str
    amount: str
    currency: str
    payment_source: str
    capture_id: str | None = None
    paypal_ms: float = 0.0
    #: Wall-clock of the LLM judge, reported apart from ``latency_ms``.
    judge_ms: float = 0.0
    consent: dict[str, Any] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    #: The basket, for the consent page.
    items: list[dict[str, Any]] = field(default_factory=list)
    #: Every reason, not only the top three.
    all_reasons: list[dict[str, str]] = field(default_factory=list)
    #: Per page read: coverage, L3 score or abstention, judge verdict and quote.
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class PaymentGate:
    """Holds the full PayPal client. Nothing else in the package gets one."""

    def __init__(self, config: PayPalConfig | None = None, *,
                 client: PayPalClient | None = None, scorer: LiveScorer | None = None,
                 payment_source: PaymentSource = "auto", record: bool = False) -> None:
        self.config = config or load_config()
        self._client: PayPalClient = client or make_client(self.config, record=record)
        self.scorer = scorer or LiveScorer(judge=_default_judge(self.config.mode))
        self.payment_source = payment_source
        self.mandates: dict[str, Mandate] = {}
        self.records: dict[str, GateRecord] = {}
        self._receipts: dict[str, dict[str, Any]] = {}
        #: order id -> (mandate id, amount): money captured or held against a
        #: mandate's budget. Keyed by order id, so a resubmitted order — same
        #: receipt, same PayPal-Request-Id, same order id — is counted once.
        self._committed: dict[str, tuple[str, Decimal]] = {}
        self._budget_lock = threading.Lock()

    @property
    def mode(self) -> str:
        return self._client.mode

    def agent(self) -> AgentPayPal:
        """The agent's handle: create and authorize, nothing that moves money."""
        return AgentPayPal(self._client)

    def vault_id(self) -> str | None:
        """The payer's vaulted wallet: minted in mock, read from the vault file live.

        ``None`` in live mode until a human has run ``vault-setup`` and approved
        it, in which case the gate falls back to the card payment source.
        """
        from mantis.paypal.vault import load_vault, mint_mock_vault

        if self.mode == "mock":
            return mint_mock_vault(self._client)
        state = load_vault()
        return state.get("vault_id") if state.get("status") == "active" else None

    # -- mandates -------------------------------------------------------------- #

    def create_mandate(self, request: MandateRequest, *, now: datetime | None = None) -> Mandate:
        mandate_id = f"mdt-{len(self.mandates) + 1:04d}"
        mandate = Mandate.issue(request, mandate_id=mandate_id, key=self.config.mandate_key,
                                now=now)
        self.mandates[mandate_id] = mandate
        return mandate

    # -- payment source ---------------------------------------------------------- #

    def _source(self, mandate: Mandate) -> tuple[str, dict[str, Any], str]:
        """``(kind, payment_source, card_bin)``. Vault when the mandate carries one."""
        wanted = self.payment_source
        if wanted == "vault" or (wanted == "auto" and mandate.vault_id):
            if not mandate.vault_id:
                raise GateError(422, "payment_source=vault but the mandate has no vault_id")
            return "vault", {"paypal": {"vault_id": mandate.vault_id}}, WALLET_BIN
        card = self.config.test_card
        return "card", {
            "card": {
                "number": card,
                "expiry": "2030-12",
                "security_code": "123",
                "name": "MANTIS Sandbox Buyer",
            }
        }, card[:6]

    # -- the gate ------------------------------------------------------------------ #

    def submit(self, intent: PurchaseIntent, context: AgentContext, *,
               page_texts: dict[str, str] | None = None) -> GateRecord:
        """The agent reserves, the gate scores and acts. See the module docstring."""
        mandate = self.mandates.get(intent.mandate_id)
        if mandate is None:
            raise GateError(404, f"unknown mandate {intent.mandate_id!r}")
        merchant = MERCHANTS.get(intent.merchant_id)
        if merchant is None:
            raise GateError(422, f"unknown merchant {intent.merchant_id!r}")

        now = datetime.now(UTC)
        receipt = build_receipt(mandate, intent, context, merchant, now=now)
        digest = receipt_hash(receipt)
        save_receipt(receipt, self.config.receipt_dir)
        rid = f"mantis-{digest[:32]}"
        kind, source, card_bin = self._source(mandate)

        # 1. the agent: create + authorize. Funds reserved, nothing moved.
        paypal_started = time.perf_counter()
        agent = self.agent()
        try:
            order = agent.create_order(
                order_body(intent, merchant, custom_id=digest, payment_source=source),
                request_id=f"{rid}-create",
            )
            if first_authorization(order) is None:
                order = agent.authorize_order(order["id"], request_id=f"{rid}-authorize")
        except PayPalError as exc:
            raise GateError(502, f"PayPal refused the authorization: {exc}") from exc
        auth = first_authorization(order)
        if auth is None:
            raise GateError(502, f"order {order.get('id')} has no authorization "
                                 f"(status {order.get('status')})")

        # 2. read PayPal's copy back and check it commits to the receipt.
        held = self._client.get_order(order["id"])
        violations: list[Reason] = []
        if not verify_custom_id(held, receipt):
            violations.append(Reason("gate", "receipt_mismatch",
                                     "order custom_id does not commit to the declared receipt"))
        amount, currency = order_amount(held)
        if Decimal(f"{amount:.2f}") != intent.total or currency != intent.currency:
            violations.append(Reason("gate", "amount_mismatch",
                                     f"PayPal holds {amount:.2f} {currency}, the agent declared "
                                     f"{intent.total:.2f} {intent.currency}"))
        budget = self._reserve_budget(mandate, order["id"], Decimal(f"{amount:.2f}"))
        if budget is not None:
            violations.append(budget)

        # 3. score. PayPal time is measured around the PayPal calls, not by
        # subtraction, so a replayed judge latency can never make it negative.
        paypal_ms = (time.perf_counter() - paypal_started) * 1000.0
        event = to_tx_event(held, mandate, context, merchant,
                            mandate_key=self.config.mandate_key, now=now, card_bin=card_bin)
        texts = dict(page_texts or {})
        texts.update({p.url: p.text for p in context.pages if p.text})
        result = self.scorer.score(event, page_texts=texts or None, gate_violations=violations)

        record = GateRecord(
            order_id=order["id"],
            authorization_id=auth["id"],
            decision=result.decision.value,
            paypal_action=ACTION[result.decision],
            status="authorized",
            reasons=[r.as_dict() for r in result.reasons],
            layers_ran=result.layers_ran,
            latency_ms=round(result.latency_ms, 2),
            receipt_hash=digest,
            mandate_id=mandate.mandate_id,
            merchant_id=merchant.merchant_id,
            amount=f"{amount:.2f}",
            currency=currency,
            payment_source=kind,
            judge_ms=round(result.judge_ms, 1),
            items=[{"name": i.name, "sku": i.sku, "quantity": i.quantity,
                    "unit_amount": f"{i.unit_amount:.2f}"} for i in intent.items],
            all_reasons=[r.as_dict() for r in result.all_reasons],
            evidence=result.evidence,
        )
        self.records[record.order_id] = record
        self._receipts[record.order_id] = receipt

        # 4. act.
        started = time.perf_counter()
        if record.paypal_action == "capture":
            self._capture(record, rid)
        elif record.paypal_action == "void":
            self._void(record, rid)
        else:
            record.status = "held"
            record.history.append({"at": _iso(), "action": "hold",
                                   "note": "awaiting human consent"})
        record.paypal_ms = round(paypal_ms + (time.perf_counter() - started) * 1000.0, 1)
        self._persist(record)
        return record

    def consent(self, order_id: str, *, approve: bool, signer: str = "human") -> GateRecord:
        """A human resolves a held order: approve -> capture, deny -> void."""
        record = self.get(order_id)
        if record.status != "held":
            raise GateError(409, f"order {order_id} is {record.status}, not held for consent")
        record.consent = {"approve": approve, "signer": signer, "at": _iso()}
        rid = f"mantis-{record.receipt_hash[:32]}"
        if approve:
            self._capture(record, rid)
        else:
            self._void(record, rid)
        self._persist(record)
        return record

    def get(self, order_id: str) -> GateRecord:
        record = self.records.get(order_id)
        if record is None:
            raise GateError(404, f"no gate record for order {order_id!r}")
        return record

    # -- the two calls only the gate may make -------------------------------------- #

    def _capture(self, record: GateRecord, rid: str) -> None:
        try:
            out = self._client.capture_authorization(record.authorization_id,
                                                     request_id=f"{rid}-capture")
        except PayPalError as exc:
            raise GateError(502, f"capture failed: {exc}") from exc
        record.capture_id = out.get("id")
        record.status = "captured"
        record.history.append({"at": _iso(), "action": "capture", "capture_id": record.capture_id,
                               "paypal_status": out.get("status")})

    def _void(self, record: GateRecord, rid: str) -> None:
        try:
            out = self._client.void_authorization(record.authorization_id,
                                                  request_id=f"{rid}-void")
        except PayPalError as exc:
            raise GateError(502, f"void failed: {exc}") from exc
        record.status = "voided"
        record.history.append({"at": _iso(), "action": "void",
                               "paypal_status": out.get("status")})
        with self._budget_lock:
            self._committed.pop(record.order_id, None)  # a voided hold frees its budget

    def _reserve_budget(self, mandate: Mandate, order_id: str, amount: Decimal) -> Reason | None:
        """The gate-level live clause ``budget_exceeded``.

        Counts every order **captured or currently held** under the mandate, not
        only captured ones, and reserves this order's amount in the same critical
        section — so two orders scored in parallel cannot both read the same
        headroom and both slip under the budget. Idempotent on the order id.
        A breach reserves nothing (the order will be voided).
        """
        if mandate.budget_total is None:
            return None
        with self._budget_lock:
            committed = sum(
                (a for oid, (mid, a) in self._committed.items()
                 if mid == mandate.mandate_id and oid != order_id),
                Decimal("0.00"),
            )
            if committed + amount > mandate.budget_total:
                return Reason(
                    "gate", "budget_exceeded",
                    f"mandate budget ${mandate.budget_total:.2f}: ${committed:.2f} already "
                    f"captured or held + ${amount:.2f} this order "
                    f"(gate-level live clause, not one of the 9 L0 clauses)",
                )
            self._committed[order_id] = (mandate.mandate_id, amount)
        return None

    def _persist(self, record: GateRecord) -> None:
        """The decision next to its receipt, for the audit trail. Gitignored."""
        directory = self.config.receipt_dir
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{record.receipt_hash}.gate.json").write_text(
            json.dumps(record.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def _default_judge(mode: str) -> InjectionJudge | None:
    """The LLM judge in the gate's own mode: cached verdicts in mock, live in live."""
    return InjectionJudge("live" if mode == "live" else "mock")


def _iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
