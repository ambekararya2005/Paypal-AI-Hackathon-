"""``/api/gate/log`` routes and the listener that feeds them.

Registered **before** the gate router so ``/gate/log`` is not swallowed by
``/gate/{order_id}``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from mantis.api.gate_log import LOG

__all__ = ["attach", "mandate_summary", "router"]

router = APIRouter(tags=["gate log"])


def mandate_summary(gate: Any, mandate_id: str) -> dict[str, Any] | None:  # Any: PaymentGate
    """Plain-language mandate for a log row: description, CAN, CANNOT, hash."""
    mandate = getattr(gate, "mandates", {}).get(mandate_id)
    if mandate is None:
        return None
    from mantis.agent.mandate_compiler import describe

    can, cannot = describe(mandate, mandate.issued_ts)
    return {"mandate_id": mandate.mandate_id, "description": mandate.description,
            "can": can, "cannot": cannot, "hash": mandate.digest()}


def attach(gate: Any, source: str) -> Any:  # Any: PaymentGate
    """Tag ``gate``'s records with ``source`` and log every decision and consent."""
    gate.source = source
    gate.listeners.append(lambda record: LOG.record(record,
                                                    mandate_summary(gate, record.mandate_id)))
    return gate


@router.get("/gate/log")
def gate_log(limit: int = Query(200, ge=1, le=500),
             offset: int = Query(0, ge=0)) -> dict[str, Any]:
    """Newest first. Poll every ~2 s; re-render only when ``version`` changes."""
    return LOG.page(limit=limit, offset=offset)


@router.get("/gate/log/{order_id}")
def gate_log_row(order_id: str) -> dict[str, Any]:
    row = LOG.get(order_id)
    if row is None:
        raise HTTPException(404, f"no log row for order {order_id!r}")
    return row


@router.get("/gate/log/{order_id}/visitor-text")
def gate_log_visitor_text(order_id: str) -> dict[str, Any]:
    """The full visitor-written page text, links stripped. Rendered escaped."""
    if LOG.get(order_id) is None:
        raise HTTPException(404, f"no log row for order {order_id!r}")
    return {"order_id": order_id, "pages": LOG.visitor_text(order_id)}
