"""The PayPal authorization gate over HTTP. Additive: mounted beside the console API.

========================================  ==========================================
``POST /mandates``                        a human issues and signs a mandate
``POST /gate``                            agent authorizes; MANTIS scores and acts
``GET  /gate/{order_id}``                 the gate's record for one order
``POST /gate/{order_id}/consent``         a human resolves a held order
========================================  ==========================================

The one exception to "no model runs inside a request"
-------------------------------------------------------
Every other route in ``mantis/api`` reads a committed artefact. ``/gate`` cannot:
its whole job is to score an authorization that did not exist a second ago. So
the scorer is loaded **once**, on the first gate call, and a request does lookups
and one TF-IDF transform — single-digit milliseconds, measured and returned in
``latency_ms``.

The import discipline that keeps the deployed image small is preserved: this
module imports only pydantic models at module scope. pandas, scikit-learn and the
PayPal client are imported inside :func:`_gate`, so ``import mantis.api.site``
still loads none of them, and the static console deploy is unaffected.
``tests/test_paypal_gate.py`` checks that in a subprocess.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from mantis.api.consent import render_consent_page
from mantis.paypal.models import AgentContext, MandateRequest, PurchaseIntent

if TYPE_CHECKING:
    from mantis.paypal.gate import PaymentGate

__all__ = ["router"]

router = APIRouter(tags=["paypal gate"])

_LOCK = threading.Lock()
_GATE: PaymentGate | None = None


def _gate() -> PaymentGate:
    """The process-wide gate, built on first use. Mode comes from MANTIS_PAYPAL_MODE."""
    global _GATE
    with _LOCK:
        if _GATE is None:
            from mantis.paypal.config import ConfigError
            from mantis.paypal.gate import PaymentGate

            try:
                _GATE = PaymentGate()
            except ConfigError as exc:
                raise HTTPException(503, str(exc)) from exc
        return _GATE


def _raise(exc: Exception) -> None:
    from mantis.paypal.gate import GateError

    if isinstance(exc, GateError):
        raise HTTPException(exc.status, str(exc)) from exc
    raise exc


class ReasonOut(BaseModel):
    layer: str
    code: str
    detail: str


class GateResponse(BaseModel):
    """What the gate decided and what it did on PayPal."""

    decision: str = Field(description="approve | challenge | review | decline")
    paypal_action: str = Field(description="capture | void | hold")
    status: str = Field(description="captured | voided | held")
    order_id: str
    authorization_id: str
    capture_id: str | None
    reasons: list[ReasonOut] = Field(description="Top three, most severe first.")
    layers_ran: list[str]
    latency_ms: float = Field(description="MANTIS scoring only.")
    paypal_ms: float = Field(description="PayPal round trips for this call.")
    receipt_hash: str = Field(description="SHA-256 of the receipt; the order's custom_id.")
    mandate_id: str
    merchant_id: str
    amount: str
    currency: str
    payment_source: str
    consent: dict[str, Any] | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)
    judge_ms: float = Field(default=0.0, description="LLM judge wall-clock, apart from scoring.")
    items: list[dict[str, Any]] = Field(default_factory=list)
    all_reasons: list[ReasonOut] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per page read: coverage, L3 score or abstention, judge verdict, text.")


class MandateResponse(BaseModel):
    mandate_id: str
    mandate_hash: str
    signature: str
    issued_ts: str
    mode: str
    mandate: dict[str, Any]


class GateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: PurchaseIntent
    agent: AgentContext


class ConsentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approve: bool
    signer: str = "human"


@router.post("/mandates", response_model=MandateResponse)
def create_mandate(request: MandateRequest) -> MandateResponse:
    """Issue and sign a mandate. In mock mode a vault id is minted when none is given."""
    gate = _gate()
    if request.vault_id is None:
        vault_id = gate.vault_id()
        if vault_id is not None:
            request = request.model_copy(update={"vault_id": vault_id})
    mandate = gate.create_mandate(request)
    return MandateResponse(
        mandate_id=mandate.mandate_id,
        mandate_hash=mandate.digest(),
        signature=mandate.signature or "",
        issued_ts=mandate.issued_ts.isoformat(),
        mode=gate.mode,
        mandate=mandate.model_dump(mode="json", exclude={"signature", "vault_id"}),
    )


@router.post("/gate", response_model=GateResponse)
def submit(request: GateRequest) -> GateResponse:
    """The agent authorizes an order; MANTIS scores it and captures, voids or holds."""
    try:
        record = _gate().submit(request.intent, request.agent)
    except Exception as exc:
        _raise(exc)
        raise
    return GateResponse(**record.as_dict())


@router.get("/gate/{order_id}/consent", response_class=HTMLResponse)
def consent_page(order_id: str) -> HTMLResponse:
    """The human-consent page: mandate, order, reasons, highlighted injected span,
    Approve (capture) / Deny (void). See :mod:`mantis.api.consent`."""
    gate = _gate()
    try:
        record = gate.get(order_id)
    except Exception as exc:
        _raise(exc)
        raise
    return HTMLResponse(render_consent_page(record.as_dict(),
                                            gate.mandates.get(record.mandate_id)))


@router.get("/gate/{order_id}", response_model=GateResponse)
def get_record(order_id: str) -> GateResponse:
    try:
        record = _gate().get(order_id)
    except Exception as exc:
        _raise(exc)
        raise
    return GateResponse(**record.as_dict())


@router.post("/gate/{order_id}/consent", response_model=GateResponse)
def consent(order_id: str, request: ConsentRequest) -> GateResponse:
    """A human resolves a held order: approve captures, deny voids."""
    try:
        record = _gate().consent(order_id, approve=request.approve, signer=request.signer)
    except Exception as exc:
        _raise(exc)
        raise
    return GateResponse(**record.as_dict())
