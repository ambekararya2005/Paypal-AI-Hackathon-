"""``GET /api/receipts/{order_id}/verify`` — does PayPal's order commit to our receipt?

Every order the gate lets an agent create carries ``custom_id = sha256(receipt)``,
where the receipt is the canonical JSON of the mandate hash, the basket and the
provenance trail (:mod:`mantis.paypal.receipt`). Verify recomputes that hash from
the receipt on disk and compares it with the ``custom_id`` **PayPal** holds:

* ``match: true`` — the order PayPal processed is exactly the purchase the gate
  scored, and the receipt has not been edited since;
* ``match: false`` — one of them changed: the receipt file was tampered with, or the
  order is not the one the receipt describes.

Where the order comes from, in order of preference, and always reported as
``source``: the **live sandbox** (``paypal-sandbox``) when live calls are allowed;
the API gate's own mock state (``mock``); else the scrubbed copy recorded from the
sandbox (``recorded-paypal``, ``data/fixtures/paypal_orders/``), which is what a
clean clone and the kill-switched deployment use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from fastapi import APIRouter, HTTPException

from mantis.core.paths import DATA_DIR
from mantis.core.quota import QuotaExceededError, live_disabled

__all__ = ["RECORDED_ORDERS", "SEED_RECEIPTS", "router", "verify_order"]

router = APIRouter(tags=["receipts"])

RECORDED_ORDERS: Final[Path] = DATA_DIR / "fixtures" / "paypal_orders"
SEED_RECEIPTS: Final[Path] = DATA_DIR / "fixtures" / "receipts"


def _receipt_dirs() -> list[Path]:
    from mantis.paypal.config import load_config

    return [load_config().receipt_dir, SEED_RECEIPTS]


def _load_receipt(digest: str, dirs: list[Path]) -> dict[str, Any] | None:
    for directory in dirs:
        path = directory / f"{digest}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def _fetch_order(order_id: str) -> tuple[dict[str, Any] | None, str]:
    """``(order, source)``; see the module docstring for the order of preference."""
    from mantis.paypal.config import load_config

    config = load_config()
    if config.mode == "live" and not live_disabled():
        from mantis.paypal.client import LiveClient, PayPalError

        try:
            return LiveClient(config).get_order(order_id), "paypal-sandbox"
        except (PayPalError, QuotaExceededError, OSError):
            pass  # fall through to the offline copies
    import mantis.api.gate as api_gate

    gate = api_gate._GATE
    if gate is not None and gate.mode == "mock":
        try:
            return gate._client.get_order(order_id), "mock"
        except Exception:
            pass
    recorded = RECORDED_ORDERS / f"{order_id}.json"
    if recorded.exists():
        return json.loads(recorded.read_text(encoding="utf-8")), "recorded-paypal"
    return None, "none"


def verify_order(order_id: str, *, receipt_dirs: list[Path] | None = None) -> dict[str, Any]:
    from mantis.api.gate_log import LOG
    from mantis.paypal.receipt import receipt_hash

    row = LOG.get(order_id)
    if row is None or not row.get("receipt_hash"):
        raise HTTPException(404, f"no gate record for order {order_id!r}")
    expected = str(row["receipt_hash"])
    receipt = _load_receipt(expected, receipt_dirs or _receipt_dirs())
    if receipt is None:
        raise HTTPException(404, f"receipt {expected[:16]}… is not on this server")

    order, source = _fetch_order(order_id)
    if order is None:
        raise HTTPException(404, f"order {order_id} could not be read from PayPal or a recording")
    units = order.get("purchase_units") or [{}]
    custom_id = units[0].get("custom_id")
    recomputed = receipt_hash(receipt)
    match = custom_id == recomputed
    if match:
        detail = "PayPal's order commits to this exact receipt: basket, mandate and provenance."
    elif recomputed != expected:
        detail = "The receipt on disk no longer hashes to what the gate recorded: it was edited."
    else:
        detail = ("PayPal's custom_id does not commit to this receipt: "
                  "the order is not the one scored.")
    return {"order_id": order_id, "match": match, "receipt_hash": recomputed,
            "recorded_hash": expected, "custom_id": custom_id, "source": source,
            "detail": detail}


@router.get("/receipts/{order_id}/verify")
def verify(order_id: str) -> dict[str, Any]:
    return verify_order(order_id)
