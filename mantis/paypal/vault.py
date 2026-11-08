"""The one-time human approval: a PayPal Vault payment token, kept locally.

    python -m mantis.paypal vault-setup     # live: prints a link; a human approves it
    python -m mantis.paypal vault-finish    # live: exchanges it for a vault_id

Why Vault is the mandate's signature
---------------------------------------
Without Vault, every PayPal-wallet order needs the buyer to click "approve" in a
browser — which is a human-present checkout, not an agent paying. With Vault the
human approves **once**, when they grant the mandate, and PayPal issues a
``vault_id`` the agent's orders reference in ``payment_source.paypal.vault_id``.
That approval is the PayPal-side twin of the mandate HMAC: one says *this human
agreed to these terms*, the other says *this human agreed to let this merchant
charge this wallet*. Both are checked before money moves.

Measured on the sandbox app this was built against (2026-11-08): Vault setup
tokens are **enabled**; card processing is **not**
(``PAYEE_NOT_ENABLED_FOR_CARD_PROCESSING``). So Vault is the path, and the test
card stays as the fallback for apps that have Advanced Card Payments enabled.

The token lives in ``data/receipts/paypal_vault.json`` (gitignored). Mock mode
never reads it: the mock mints its own vault id, auto-approved, so a clean clone
runs the identical code path with no human and no network.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from mantis.paypal.client import PayPalClient
from mantis.paypal.config import RECEIPT_DIR

__all__ = ["VAULT_FILE", "finish_vault", "load_vault", "mint_mock_vault", "start_vault"]

VAULT_FILE: Final[Path] = RECEIPT_DIR / "paypal_vault.json"


def load_vault(path: Path = VAULT_FILE) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save(state: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def start_vault(client: PayPalClient, *, description: str,
                path: Path = VAULT_FILE) -> dict[str, Any]:
    """Create a setup token and record it. Returns the state, with the approve URL."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    out = client.create_vault_setup_token(
        description=description, request_id=f"mantis-vault-setup-{stamp}")
    approve = next((link["href"] for link in out.get("links", []) if link.get("rel") == "approve"),
                   None)
    state = {"setup_token_id": out["id"], "approve_url": approve, "status": "pending",
             "created": stamp}
    _save(state, path)
    return state


def finish_vault(client: PayPalClient, *, path: Path = VAULT_FILE) -> dict[str, Any]:
    """Exchange the approved setup token for a ``vault_id`` and record it."""
    state = load_vault(path)
    if not state.get("setup_token_id"):
        raise RuntimeError("no pending vault setup; run 'python -m mantis.paypal vault-setup'")
    out = client.create_vault_payment_token(
        state["setup_token_id"], request_id=f"mantis-vault-token-{state['setup_token_id']}")
    state.update({"vault_id": out["id"], "customer_id": (out.get("customer") or {}).get("id"),
                  "status": "active"})
    _save(state, path)
    return state


def mint_mock_vault(client: PayPalClient) -> str:
    """Mock mode: the same two calls, with the human's click simulated."""
    setup = client.create_vault_setup_token(
        description="MANTIS demo mandate", request_id="mantis-mock-vault-setup")
    token = client.create_vault_payment_token(
        setup["id"], request_id="mantis-mock-vault-token")
    return str(token["id"])
