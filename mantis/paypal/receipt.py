"""The purchase receipt: what was agreed, hashed into the PayPal order itself.

    python -m mantis.paypal.receipt        # build one, hash it, verify it round-trips

Before the agent creates an order, the gate writes down — canonically — which
mandate it is acting under, what it is buying from whom, and what it read on the
way. The SHA-256 of that document goes into the order's ``custom_id``, so PayPal
carries a commitment to the receipt for the life of the payment.

At decision time the gate reads the order **back from PayPal** and recomputes the
hash. If the agent created the order with a different basket, amount or payee
than the one it declared, the commitment does not match and the gate voids it.
That turns "the agent said it was buying groceries" from a claim into something
checkable against the payment processor's own record — and a disputed payment
months later can be tied to the exact provenance trail behind it.

Receipts are written to ``data/receipts/`` (gitignored).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from mantis.foundry.llm.corpus import content_id_for_url
from mantis.paypal.config import RECEIPT_DIR
from mantis.paypal.merchants import Merchant
from mantis.paypal.models import AgentContext, Mandate, PurchaseIntent, canonical_json

__all__ = [
    "RECEIPT_SCHEMA",
    "build_receipt",
    "load_receipt",
    "receipt_hash",
    "save_receipt",
    "verify_custom_id",
]

RECEIPT_SCHEMA: Final[str] = "mantis.receipt/1"


def build_receipt(
    mandate: Mandate,
    intent: PurchaseIntent,
    context: AgentContext,
    merchant: Merchant,
    *,
    now: datetime,
) -> dict[str, Any]:
    """The canonical receipt as a plain dict. Amounts are strings, as on the wire."""
    return {
        "schema": RECEIPT_SCHEMA,
        "created_ts": now.isoformat(),
        "mandate": {"mandate_id": mandate.mandate_id, "mandate_hash": mandate.digest()},
        "agent": {"agent_id": context.agent_id, "agent_platform": context.agent_platform},
        "order": {
            "merchant_id": merchant.merchant_id,
            "mcc": merchant.mcc,
            "currency": intent.currency,
            "total": f"{intent.total:.2f}",
            "items": [
                {
                    "name": item.name,
                    "sku": item.sku,
                    "quantity": item.quantity,
                    "unit_amount": f"{item.unit_amount:.2f}",
                }
                for item in intent.items
            ],
        },
        "provenance": {
            "chain": context.provenance_chain,
            "content_ids": [content_id_for_url(u) for u in context.provenance_chain],
        },
    }


def receipt_hash(receipt: dict[str, Any]) -> str:
    """SHA-256 hex (64 chars) of the canonical bytes. Fits ``custom_id`` (max 127)."""
    return hashlib.sha256(canonical_json(receipt)).hexdigest()


def save_receipt(receipt: dict[str, Any], directory: Path = RECEIPT_DIR) -> Path:
    """Write ``<hash>.json``. The file name *is* the commitment."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{receipt_hash(receipt)}.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_receipt(digest: str, directory: Path = RECEIPT_DIR) -> dict[str, Any]:
    return json.loads((directory / f"{digest}.json").read_text(encoding="utf-8"))


def verify_custom_id(order: dict[str, Any], receipt: dict[str, Any]) -> bool:
    """True when the order PayPal holds commits to exactly this receipt."""
    units = order.get("purchase_units") or [{}]
    custom = units[0].get("custom_id")
    if custom is None:
        auths = (units[0].get("payments") or {}).get("authorizations") or [{}]
        custom = auths[0].get("custom_id")
    return custom == receipt_hash(receipt)


def main() -> None:
    from datetime import UTC
    from decimal import Decimal

    from mantis.core.events import MandateScope
    from mantis.paypal.merchants import MERCHANTS
    from mantis.paypal.models import MandateRequest, OrderItem, Page

    now = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)
    mandate = Mandate.issue(
        MandateRequest(customer_id="cust-demo", agent_id="agent-shopper-01",
                       scope=MandateScope(categories=["5411"], max_amount=100.0)),
        mandate_id="mdt-demo", key="demo", now=now,
    )
    merchant = MERCHANTS["m-freshmart"]
    intent = PurchaseIntent(mandate_id="mdt-demo", merchant_id=merchant.merchant_id,
                            items=[OrderItem(name="Groceries", unit_amount=Decimal("32"))])
    context = AgentContext(agent_id="agent-shopper-01", agent_platform="mantis-demo-agent",
                           pages=[Page(url=f"{merchant.shop_url}/basket")])
    receipt = build_receipt(mandate, intent, context, merchant, now=now)
    digest = receipt_hash(receipt)
    order = {"purchase_units": [{"custom_id": digest}]}
    print("MANTIS purchase receipt")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    print(f"\n  sha256 -> custom_id : {digest}")
    print(f"  round-trips         : {verify_custom_id(order, receipt)}")
    receipt["order"]["total"] = "320.00"
    print(f"  after tampering     : {verify_custom_id(order, receipt)}  (must be False)")


if __name__ == "__main__":
    main()
