"""PayPal order <-> MANTIS ``TxEvent``. The schema does not move; PayPal maps into it.

    python -m mantis.paypal.adapter        # print one PayPal order as a TxEvent row

Two directions:

* :func:`order_body` turns a purchase intent into a PayPal Orders v2 request with
  ``intent=AUTHORIZE``, line items, a breakdown PayPal will reconcile to the cent,
  and the receipt hash in ``custom_id``.
* :func:`to_tx_event` turns the order **PayPal returned** — not the intent the
  agent declared — plus the mandate and the agent context into a frozen
  :class:`~mantis.core.events.TxEvent`. Scoring PayPal's copy is the point: the
  amount L0 checks against the mandate ceiling is the amount PayPal authorized.

Field map (everything lands on an existing field; nothing is added)
---------------------------------------------------------------------
==============================  ==============================================
``TxEvent``                     source
==============================  ==============================================
``event_id``                    ``paypal:<order id>``
``amount`` / ``currency``       ``purchase_units[0].amount`` as authorized
``mcc`` / ``merchant_country``  the merchant profile (orders carry no MCC)
``merchant_id``                 ``purchase_units[0].reference_id``
``customer_id``                 the mandate's customer
``card_bin``                    the card's BIN, or ``000000`` for a wallet
``channel`` / ``entry_mode``    ``agentic`` / ``agent_token``
``txn_type``                    ``preauth`` — the order intent is AUTHORIZE
``settled``                     False — nothing is captured at scoring time
``agentic.mandate_*``           the signed mandate, its hash, its scope
``agentic.consent_sig_valid``   :meth:`Mandate.verify` against the gate's key
``agentic.kya_registered``      the KYA registry, by agent id **and** token
``agentic.provenance_chain``    the URLs the agent read, in order
``agentic.ingested_content_ids``  ``content_id_for_url`` of each, as the foundry
==============================  ==============================================

``auth_response`` is left at its default (approved) because PayPal did authorize
the hold. It is a FUTURE column — the issuer's own decision on this message — and
the feature builder blocks it by name, so it cannot leak into a score.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Final

from mantis.core.events import (
    AgenticContext,
    Channel,
    EntryMode,
    TxEvent,
    TxnType,
)
from mantis.foundry.llm.corpus import content_id_for_url
from mantis.paypal.merchants import Merchant, kya_registered
from mantis.paypal.models import AgentContext, Mandate, PurchaseIntent

__all__ = ["WALLET_BIN", "order_amount", "order_body", "to_tx_event"]

#: A PayPal wallet payment has no card BIN on the wire. ``TxEvent.card_bin`` is
#: required and must be digits, so a wallet gets an all-zero BIN that no issuer
#: allocates — distinguishable, and never confused with a real range.
WALLET_BIN: Final[str] = "000000"

#: PayPal's description limit on a purchase unit.
_DESCRIPTION_MAX: Final[int] = 127


def _money(currency: str, value: Decimal) -> dict[str, str]:
    return {"currency_code": currency, "value": f"{value:.2f}"}


def order_body(
    intent: PurchaseIntent,
    merchant: Merchant,
    *,
    custom_id: str,
    payment_source: dict[str, Any] | None,
) -> dict[str, Any]:
    """An Orders v2 create request with ``intent=AUTHORIZE``."""
    description = (intent.description or f"{merchant.name} via MANTIS agent")[:_DESCRIPTION_MAX]
    body: dict[str, Any] = {
        "intent": "AUTHORIZE",
        "purchase_units": [
            {
                "reference_id": merchant.merchant_id,
                "description": description,
                "custom_id": custom_id,
                "amount": {
                    **_money(intent.currency, intent.total),
                    "breakdown": {"item_total": _money(intent.currency, intent.total)},
                },
                "items": [
                    {
                        "name": item.name,
                        "sku": item.sku or None,
                        "quantity": str(item.quantity),
                        "unit_amount": _money(intent.currency, item.unit_amount),
                        "category": item.category,
                    }
                    for item in intent.items
                ],
            }
        ],
    }
    for item in body["purchase_units"][0]["items"]:
        if item["sku"] is None:
            del item["sku"]
    if payment_source:
        body["payment_source"] = payment_source
    return body


def order_amount(order: dict[str, Any]) -> tuple[float, str]:
    """The amount and currency PayPal holds for this order."""
    unit = (order.get("purchase_units") or [{}])[0]
    amount = unit.get("amount")
    if amount is None:
        auths = (unit.get("payments") or {}).get("authorizations") or [{}]
        amount = auths[0].get("amount") or {}
    return float(amount.get("value", 0.0)), str(amount.get("currency_code", "USD"))


def to_tx_event(
    order: dict[str, Any],
    mandate: Mandate,
    context: AgentContext,
    merchant: Merchant,
    *,
    mandate_key: str,
    now: datetime,
    card_bin: str = WALLET_BIN,
) -> TxEvent:
    """Map a PayPal order onto the frozen schema. See the module docstring."""
    amount, currency = order_amount(order)
    chain = context.provenance_chain
    return TxEvent(
        event_id=f"paypal:{order['id']}",
        ts=now,
        amount=amount,
        currency=currency,
        mcc=merchant.mcc,
        channel=Channel.AGENTIC,
        entry_mode=EntryMode.AGENT_TOKEN,
        customer_id=mandate.customer_id,
        card_bin=card_bin,
        merchant_id=merchant.merchant_id,
        merchant_country=merchant.country,
        txn_type=TxnType.PREAUTH,
        settled=False,
        agentic=AgenticContext(
            agent_id=context.agent_id,
            agent_platform=context.agent_platform,
            kya_token=context.kya_token,
            kya_registered=kya_registered(context.agent_id, context.kya_token),
            mandate_type=mandate.mandate_type,
            mandate_id=mandate.mandate_id,
            mandate_hash=mandate.digest(),
            mandate_issued_ts=mandate.issued_ts,
            mandate_ttl_seconds=mandate.ttl_seconds,
            mandate_scope=mandate.scope,
            human_present=context.human_present,
            consent_sig_valid=mandate.verify(mandate_key),
            delegation_depth=context.delegation_depth,
            provenance_chain=chain,
            ingested_content_ids=[content_id_for_url(u) for u in chain],
            tool_call_count=context.tool_call_count,
            deliberation_latency_ms=context.deliberation_latency_ms,
        ),
    )


def main() -> None:
    from datetime import UTC

    from mantis.core.events import MandateScope, flatten
    from mantis.paypal.merchants import MERCHANTS
    from mantis.paypal.models import MandateRequest, OrderItem, Page

    now = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)
    mandate = Mandate.issue(
        MandateRequest(customer_id="cust-demo", agent_id="agent-shopper-01",
                       scope=MandateScope(categories=["5411"], max_amount=100.0,
                                          ttl_seconds=86_400)),
        mandate_id="mdt-demo", key="demo", now=now,
    )
    merchant = MERCHANTS["m-freshmart"]
    intent = PurchaseIntent(mandate_id="mdt-demo", merchant_id=merchant.merchant_id,
                            items=[OrderItem(name="Groceries", unit_amount=Decimal("32"))])
    body = order_body(intent, merchant, custom_id="0" * 64, payment_source=None)
    order = {"id": "DEMO0RDER0000001", **body}
    context = AgentContext(agent_id="agent-shopper-01", agent_platform="mantis-demo-agent",
                           kya_token="kya-demo-shopper-01",
                           pages=[Page(url=f"{merchant.shop_url}/basket")])
    event = to_tx_event(order, mandate, context, merchant, mandate_key="demo", now=now)
    print("PayPal order -> TxEvent (schema unchanged)")
    for key, value in flatten(event).items():
        if value is not None:
            print(f"  {key:<28} {value}")


if __name__ == "__main__":
    main()
