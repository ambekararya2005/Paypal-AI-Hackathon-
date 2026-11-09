"""PayPal Orders v2 / Payments v2 over plain REST, in two interchangeable modes.

    python -m mantis.paypal.client          # mock: one order through the state machine

Why plain REST and not the PayPal Agent Toolkit
-------------------------------------------------
The Toolkit is built for an agent that pays. It exposes ``create_order`` and
``capture`` and has no *authorize* intent and no *void* — so an agent holding it
can move money end to end, and nothing sits between "the agent decided" and "the
money moved". This package splits that path in two on purpose:

* the agent may **create** an order with ``intent=AUTHORIZE`` and **authorize**
  it, which reserves funds and moves nothing;
* only the gate may **capture** (move the money) or **void** (release the hold).

:class:`AgentPayPal` is the agent's handle and has no capture, void or refund
method at all — not a disabled one, an absent one. ``tests/test_paypal_gate.py``
asserts both that and that nothing outside ``mantis/paypal/gate.py`` calls them.

The two clients
-----------------
:class:`LiveClient` talks to the PayPal sandbox. :class:`MockClient` replays the
**recorded** sandbox responses in ``data/fixtures/paypal/`` through a small state
machine, so a capture after a void fails exactly as the sandbox fails it (422
``AUTHORIZATION_VOIDED``) rather than happily returning a canned success. The
fixtures are response *templates*: server-generated ids and timestamps are
re-minted per call, and everything the request determines (amount, items,
``custom_id``) is echoed from the request — as PayPal does.

``LiveClient(record=True)`` rewrites the templates from real sandbox traffic,
scrubbing payer identity first.

Idempotency
-------------
Every POST carries a ``PayPal-Request-Id``. A retried capture with the same id is
answered from PayPal's idempotency cache instead of capturing twice; the mock
honours the same contract. The gate derives these ids from the receipt hash, so a
retry after a timeout is always a retry and never a second payment.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, Protocol

from mantis.paypal.config import FIXTURE_DIR, PayPalConfig

__all__ = [
    "AgentPayPal",
    "LiveClient",
    "MockClient",
    "PayPalClient",
    "PayPalError",
    "make_client",
]

#: Seconds to wait on a PayPal call. Sandbox p99 is well under this; a hang longer
#: than this is an outage, and the caller should see an error rather than a spinner.
HTTP_TIMEOUT: Final[float] = 30.0

#: Shave this off ``expires_in`` so a token is never presented in its last minute.
_TOKEN_MARGIN_S: Final[float] = 60.0

#: Fields scrubbed from recorded fixtures. Sandbox payer data is fake, but a
#: committed fixture is published, and the habit is worth more than the exception.
_SCRUB_KEYS: Final[frozenset[str]] = frozenset(
    {
        "email_address",
        "payer_id",
        "given_name",
        "surname",
        "full_name",
        "national_number",
        "address_line_1",
        "address_line_2",
        "admin_area_1",
        "admin_area_2",
        "postal_code",
        "account_id",
        "merchant_id",
    }
)

_TIME_KEYS: Final[frozenset[str]] = frozenset({"create_time", "update_time"})

#: PayPal holds an authorization for 29 days; its honor period is the first 3.
_AUTH_VALID_DAYS: Final[int] = 29

#: Where PayPal sends the human after approving the vault. The token is exchanged
#: by ``vault-finish`` whatever page they land on, so these need not be served.
_VAULT_RETURN_URL: Final[str] = "https://example.com/mantis/vault/approved"
_VAULT_CANCEL_URL: Final[str] = "https://example.com/mantis/vault/cancelled"
_VAULT_DESCRIPTION_MAX: Final[int] = 127


class PayPalError(RuntimeError):
    """A non-2xx PayPal response, with the fields an operator needs."""

    def __init__(self, status: int, name: str, message: str, *,
                 issue: str | None = None, debug_id: str | None = None) -> None:
        self.status = status
        self.name = name
        self.issue = issue
        self.debug_id = debug_id
        super().__init__(f"PayPal {status} {name}{f' / {issue}' if issue else ''}: {message}")

    @classmethod
    def from_body(cls, status: int, raw: bytes) -> PayPalError:
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except (ValueError, UnicodeDecodeError):
            body = {}
        details = body.get("details") or [{}]
        return cls(
            status,
            str(body.get("name") or body.get("error") or "HTTP_ERROR"),
            str(body.get("message") or body.get("error_description") or raw[:200]),
            issue=details[0].get("issue"),
            debug_id=body.get("debug_id"),
        )


class PayPalClient(Protocol):
    """The six calls the gate makes. Both clients implement exactly these."""

    mode: str

    def create_order(self, body: dict[str, Any], *, request_id: str) -> dict[str, Any]: ...

    def authorize_order(self, order_id: str, *, request_id: str) -> dict[str, Any]: ...

    def capture_authorization(
        self, authorization_id: str, *, request_id: str
    ) -> dict[str, Any]: ...

    def void_authorization(self, authorization_id: str, *, request_id: str) -> dict[str, Any]: ...

    def refund_capture(self, capture_id: str, *, request_id: str) -> dict[str, Any]: ...

    def get_order(self, order_id: str) -> dict[str, Any]: ...

    def create_vault_setup_token(self, *, description: str,
                                 request_id: str) -> dict[str, Any]: ...

    def create_vault_payment_token(self, setup_token_id: str, *,
                                   request_id: str) -> dict[str, Any]: ...


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _scrub(node: Any) -> Any:  # Any: arbitrary JSON
    if isinstance(node, dict):
        return {
            k: ("<scrubbed>" if k in _SCRUB_KEYS and isinstance(v, str) else _scrub(v))
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [_scrub(v) for v in node]
    return node


def first_authorization(order: dict[str, Any]) -> dict[str, Any] | None:
    """The authorization inside an order response, if the order has one yet."""
    for unit in order.get("purchase_units", []):
        auths = unit.get("payments", {}).get("authorizations") or []
        if auths:
            return auths[0]
    return None


# --------------------------------------------------------------------------- #
# Live
# --------------------------------------------------------------------------- #


class LiveClient:
    """PayPal sandbox over ``urllib``. One cached OAuth token per instance."""

    mode = "live"

    def __init__(self, config: PayPalConfig, *, record: bool = False,
                 fixture_dir: Path | None = None) -> None:
        if not config.has_credentials:
            raise ValueError("LiveClient needs PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET")
        self._config = config
        self._token: str | None = None
        self._token_expiry = 0.0
        self._record = record
        self._fixture_dir = fixture_dir or config.fixture_dir

    # -- transport ------------------------------------------------------------ #

    def access_token(self) -> str:
        """A cached OAuth token, e.g. for PayPal's hosted MCP server."""
        return self._access_token()

    def _access_token(self) -> str:
        if self._token and time.monotonic() < self._token_expiry:
            return self._token
        basic = base64.b64encode(
            f"{self._config.client_id}:{self._config.client_secret}".encode()
        ).decode("ascii")
        request = urllib.request.Request(
            f"{self._config.base_url}/v1/oauth2/token",
            data=urllib.parse.urlencode({"grant_type": "client_credentials"}).encode(),
            headers={
                "Authorization": f"Basic {basic}",
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise PayPalError.from_body(exc.code, exc.read()) from None
        self._token = str(body["access_token"])
        self._token_expiry = time.monotonic() + float(body.get("expires_in", 0)) - _TOKEN_MARGIN_S
        return self._token

    def _call(self, method: str, path: str, *, body: dict[str, Any] | None = None,
              request_id: str | None = None) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._access_token()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Prefer": "return=representation",
        }
        if request_id is not None:
            headers["PayPal-Request-Id"] = request_id
        data = json.dumps(body).encode("utf-8") if body is not None else (
            b"{}" if method == "POST" else None
        )
        request = urllib.request.Request(
            f"{self._config.base_url}{path}", data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise PayPalError.from_body(exc.code, exc.read()) from None
        return json.loads(raw.decode("utf-8")) if raw.strip() else {}

    # -- recording ------------------------------------------------------------ #

    def _save(self, op: str, response: dict[str, Any], ids: dict[str, str | None]) -> None:
        """Write a response as a template: ids become ``{{placeholders}}``."""
        if not self._record:
            return
        text = json.dumps(_scrub(response), indent=2, sort_keys=True)
        for name, value in ids.items():
            if value:
                text = text.replace(value, "{{" + name + "}}")
        template = json.loads(text)
        template["_recorded"] = {"op": op, "source": "paypal sandbox", "at": _now_iso()}
        self._fixture_dir.mkdir(parents=True, exist_ok=True)
        (self._fixture_dir / f"{op}.json").write_text(
            json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    # -- the six calls ---------------------------------------------------------- #

    def create_order(self, body: dict[str, Any], *, request_id: str) -> dict[str, Any]:
        out = self._call("POST", "/v2/checkout/orders", body=body, request_id=request_id)
        auth = first_authorization(out)
        self._save("create_order", out,
                   {"order_id": out.get("id"), "authorization_id": auth and auth.get("id")})
        return out

    def authorize_order(self, order_id: str, *, request_id: str) -> dict[str, Any]:
        out = self._call(
            "POST", f"/v2/checkout/orders/{order_id}/authorize", request_id=request_id
        )
        auth = first_authorization(out)
        self._save("authorize_order", out,
                   {"order_id": order_id, "authorization_id": auth and auth.get("id")})
        return out

    def capture_authorization(self, authorization_id: str, *, request_id: str) -> dict[str, Any]:
        out = self._call(
            "POST",
            f"/v2/payments/authorizations/{authorization_id}/capture",
            body={"final_capture": True},
            request_id=request_id,
        )
        self._save("capture_authorization", out,
                   {"capture_id": out.get("id"), "authorization_id": authorization_id})
        return out

    def void_authorization(self, authorization_id: str, *, request_id: str) -> dict[str, Any]:
        out = self._call(
            "POST", f"/v2/payments/authorizations/{authorization_id}/void", request_id=request_id
        )
        if not out:  # 204 No Content: read the authorization back for its status
            out = self._call("GET", f"/v2/payments/authorizations/{authorization_id}")
        self._save("void_authorization", out, {"authorization_id": authorization_id})
        return out

    def refund_capture(self, capture_id: str, *, request_id: str) -> dict[str, Any]:
        out = self._call(
            "POST", f"/v2/payments/captures/{capture_id}/refund", request_id=request_id
        )
        self._save("refund_capture", out, {"refund_id": out.get("id"), "capture_id": capture_id})
        return out

    def get_order(self, order_id: str) -> dict[str, Any]:
        out = self._call("GET", f"/v2/checkout/orders/{order_id}")
        auth = first_authorization(out)
        self._save("get_order", out,
                   {"order_id": order_id, "authorization_id": auth and auth.get("id")})
        return out

    # -- Vault: the human approves once, at mandate creation ---------------------- #

    def create_vault_setup_token(self, *, description: str, request_id: str) -> dict[str, Any]:
        """Start a PayPal-wallet vault. The response's ``approve`` link is for the human."""
        out = self._call("POST", "/v3/vault/setup-tokens", request_id=request_id, body={
            "payment_source": {"paypal": {
                "description": description[:_VAULT_DESCRIPTION_MAX],
                "usage_type": "MERCHANT",
                "experience_context": {"return_url": _VAULT_RETURN_URL,
                                       "cancel_url": _VAULT_CANCEL_URL},
            }}})
        self._save("vault_setup_token", out, {"setup_token_id": out.get("id")})
        return out

    def create_vault_payment_token(self, setup_token_id: str, *,
                                   request_id: str) -> dict[str, Any]:
        """Exchange an approved setup token for the long-lived ``vault_id``."""
        out = self._call("POST", "/v3/vault/payment-tokens", request_id=request_id, body={
            "payment_source": {"token": {"id": setup_token_id, "type": "SETUP_TOKEN"}}})
        self._save("vault_payment_token", out,
                   {"vault_id": out.get("id"), "setup_token_id": setup_token_id})
        return out


# --------------------------------------------------------------------------- #
# Mock
# --------------------------------------------------------------------------- #


class MockClient:
    """Recorded sandbox responses, replayed through PayPal's state machine.

    Deterministic: ids are derived from ``seed`` and a per-instance counter, so two
    runs of the demo print the same order ids.
    """

    mode = "mock"

    def __init__(self, fixture_dir: Path = FIXTURE_DIR, *, seed: int = 1337) -> None:
        self._fixture_dir = fixture_dir
        self._seed = seed
        self._counter = 0
        self._templates: dict[str, dict[str, Any]] = {}
        self._orders: dict[str, dict[str, Any]] = {}
        #: authorization id -> {"order_id", "status", "amount"}
        self._auths: dict[str, dict[str, Any]] = {}
        self._captures: dict[str, dict[str, Any]] = {}
        #: (op, PayPal-Request-Id) -> response. The idempotency contract.
        self._replay: dict[tuple[str, str], dict[str, Any]] = {}

    # -- plumbing ------------------------------------------------------------- #

    def _mint(self, kind: str) -> str:
        self._counter += 1
        digest = hashlib.sha256(f"{self._seed}|{kind}|{self._counter}".encode()).hexdigest()
        return digest[:17].upper()

    def _template(self, op: str) -> dict[str, Any]:
        if op not in self._templates:
            path = self._fixture_dir / f"{op}.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"no PayPal fixture {path}; run 'python -m mantis.paypal record' in live mode"
                )
            self._templates[op] = json.loads(path.read_text(encoding="utf-8"))
        return self._templates[op]

    def _render(self, op: str, ids: dict[str, str]) -> dict[str, Any]:
        text = json.dumps(self._template(op))
        for name, value in ids.items():
            text = text.replace("{{" + name + "}}", value)
        out = json.loads(text)
        out.pop("_recorded", None)
        self._stamp(out)
        return out

    def _stamp(self, node: Any) -> None:  # Any: arbitrary JSON
        now = _now_iso()
        if isinstance(node, dict):
            for key in list(node):
                if key in _TIME_KEYS:
                    node[key] = now
                elif key == "expiration_time":
                    node[key] = (
                        (datetime.now(UTC) + timedelta(days=_AUTH_VALID_DAYS))
                        .replace(microsecond=0).isoformat().replace("+00:00", "Z")
                    )
                else:
                    self._stamp(node[key])
        elif isinstance(node, list):
            for item in node:
                self._stamp(item)

    def _idempotent(self, op: str, request_id: str, fn: Any) -> dict[str, Any]:  # Any: callable
        key = (op, request_id)
        if key not in self._replay:
            self._replay[key] = fn()
        return copy.deepcopy(self._replay[key])

    @staticmethod
    def _echo_unit(out: dict[str, Any], unit: dict[str, Any]) -> None:
        """Copy the request-determined fields of the purchase unit into the response."""
        target = out["purchase_units"][0]
        for key in ("reference_id", "description", "custom_id", "amount", "items"):
            if key in unit:
                target[key] = copy.deepcopy(unit[key])
        for auth in target.get("payments", {}).get("authorizations", []):
            auth["amount"] = {k: unit["amount"][k] for k in ("currency_code", "value")}
            auth["custom_id"] = unit.get("custom_id")

    def _new_auth(self, order_id: str, unit: dict[str, Any]) -> str:
        auth_id = self._mint("auth")
        self._auths[auth_id] = {
            "order_id": order_id,
            "status": "CREATED",
            "amount": {k: unit["amount"][k] for k in ("currency_code", "value")},
            "custom_id": unit.get("custom_id"),
        }
        return auth_id

    @staticmethod
    def _error(status: int, name: str, issue: str, message: str) -> PayPalError:
        return PayPalError(status, name, message, issue=issue, debug_id="mock")

    # -- the six calls ---------------------------------------------------------- #

    def create_order(self, body: dict[str, Any], *, request_id: str) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            if body.get("intent") != "AUTHORIZE":
                raise self._error(422, "UNPROCESSABLE_ENTITY", "INVALID_INTENT",
                                  "the gate only creates AUTHORIZE orders")
            unit = body["purchase_units"][0]
            order_id = self._mint("order")
            source = body.get("payment_source") or {}
            # A card or a vaulted wallet is processed in the create call itself:
            # the order comes back COMPLETED with the authorization inside it.
            # Without one the buyer has to approve first, as on the real API.
            processed = "card" in source or bool(source.get("paypal", {}).get("vault_id"))
            if processed:
                auth_id = self._new_auth(order_id, unit)
                out = self._render("create_order",
                                   {"order_id": order_id, "authorization_id": auth_id})
            else:
                out = {"id": order_id, "status": "PAYER_ACTION_REQUIRED", "intent": "AUTHORIZE",
                       "purchase_units": [{}], "links": []}
            self._echo_unit(out, unit)
            out["status"] = "COMPLETED" if processed else out["status"]
            self._orders[order_id] = {"body": copy.deepcopy(body), "response": out}
            return out

        return self._idempotent("create_order", request_id, run)

    def authorize_order(self, order_id: str, *, request_id: str) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            order = self._orders.get(order_id)
            if order is None:
                raise self._error(404, "RESOURCE_NOT_FOUND", "INVALID_RESOURCE_ID",
                                  f"order {order_id} does not exist")
            if first_authorization(order["response"]) is not None:
                raise self._error(422, "UNPROCESSABLE_ENTITY", "ORDER_ALREADY_AUTHORIZED",
                                  "order already authorized")
            if order["response"]["status"] != "APPROVED":
                raise self._error(422, "UNPROCESSABLE_ENTITY", "ORDER_NOT_APPROVED",
                                  "the payer has not approved this order")
            unit = order["body"]["purchase_units"][0]
            auth_id = self._new_auth(order_id, unit)
            out = self._render("authorize_order",
                               {"order_id": order_id, "authorization_id": auth_id})
            self._echo_unit(out, unit)
            order["response"] = out
            return out

        return self._idempotent("authorize_order", request_id, run)

    def _auth_or_404(self, authorization_id: str) -> dict[str, Any]:
        auth = self._auths.get(authorization_id)
        if auth is None:
            raise self._error(404, "RESOURCE_NOT_FOUND", "INVALID_RESOURCE_ID",
                              f"authorization {authorization_id} does not exist")
        return auth

    def capture_authorization(self, authorization_id: str, *, request_id: str) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            auth = self._auth_or_404(authorization_id)
            if auth["status"] == "VOIDED":
                raise self._error(422, "UNPROCESSABLE_ENTITY", "AUTHORIZATION_VOIDED",
                                  "a voided authorization cannot be captured")
            if auth["status"] == "CAPTURED":
                raise self._error(422, "UNPROCESSABLE_ENTITY", "AUTHORIZATION_ALREADY_CAPTURED",
                                  "authorization already captured")
            capture_id = self._mint("capture")
            out = self._render("capture_authorization",
                               {"capture_id": capture_id, "authorization_id": authorization_id})
            out["amount"] = dict(auth["amount"])
            out["custom_id"] = auth["custom_id"]
            out["status"] = "COMPLETED"
            auth["status"] = "CAPTURED"
            self._captures[capture_id] = {"authorization_id": authorization_id,
                                          "amount": dict(auth["amount"]), "status": "COMPLETED"}
            self._sync_order(auth)
            return out

        return self._idempotent("capture_authorization", request_id, run)

    def void_authorization(self, authorization_id: str, *, request_id: str) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            auth = self._auth_or_404(authorization_id)
            if auth["status"] == "CAPTURED":
                raise self._error(422, "UNPROCESSABLE_ENTITY", "AUTHORIZATION_ALREADY_CAPTURED",
                                  "a captured authorization cannot be voided")
            if auth["status"] == "VOIDED":
                raise self._error(422, "UNPROCESSABLE_ENTITY", "AUTHORIZATION_VOIDED",
                                  "authorization already voided")
            out = self._render("void_authorization", {"authorization_id": authorization_id})
            out["amount"] = dict(auth["amount"])
            out["custom_id"] = auth["custom_id"]
            out["status"] = "VOIDED"
            auth["status"] = "VOIDED"
            self._sync_order(auth)
            return out

        return self._idempotent("void_authorization", request_id, run)

    def refund_capture(self, capture_id: str, *, request_id: str) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            capture = self._captures.get(capture_id)
            if capture is None:
                raise self._error(404, "RESOURCE_NOT_FOUND", "INVALID_RESOURCE_ID",
                                  f"capture {capture_id} does not exist")
            if capture["status"] == "REFUNDED":
                raise self._error(422, "UNPROCESSABLE_ENTITY", "CAPTURE_FULLY_REFUNDED",
                                  "capture already fully refunded")
            refund_id = self._mint("refund")
            out = self._render("refund_capture", {"refund_id": refund_id,
                                                  "capture_id": capture_id})
            out["amount"] = dict(capture["amount"])
            out["status"] = "COMPLETED"
            capture["status"] = "REFUNDED"
            return out

        return self._idempotent("refund_capture", request_id, run)

    def get_order(self, order_id: str) -> dict[str, Any]:
        order = self._orders.get(order_id)
        if order is None:
            raise self._error(404, "RESOURCE_NOT_FOUND", "INVALID_RESOURCE_ID",
                              f"order {order_id} does not exist")
        return copy.deepcopy(order["response"])

    def create_vault_setup_token(self, *, description: str, request_id: str) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            token = self._mint("setup")
            return {"id": token, "status": "PAYER_ACTION_REQUIRED",
                    "payment_source": {"paypal": {"description": description}},
                    "links": [{"rel": "approve", "method": "GET",
                               "href": f"https://www.sandbox.paypal.com/agreements/approve"
                                       f"?approval_session_id={token}"}]}

        return self._idempotent("vault_setup_token", request_id, run)

    def create_vault_payment_token(self, setup_token_id: str, *,
                                   request_id: str) -> dict[str, Any]:
        # The mock's human always approves; the live sandbox needs the click.
        return self._idempotent("vault_payment_token", request_id, lambda: {
            "id": self._mint("vault"), "customer": {"id": self._mint("customer")},
            "payment_source": {"paypal": {"usage_type": "MERCHANT"}}})

    def _sync_order(self, auth: dict[str, Any]) -> None:
        """Reflect an authorization's new status inside its order, as GET /orders does."""
        order = self._orders.get(auth["order_id"])
        if order is None:
            return
        found = first_authorization(order["response"])
        if found is not None:
            found["status"] = auth["status"]


# --------------------------------------------------------------------------- #
# The agent's handle
# --------------------------------------------------------------------------- #


class AgentPayPal:
    """What an agent is given. It can reserve funds and nothing else.

    There is no ``capture``, ``void`` or ``refund`` here, and no ``__getattr__``
    forwarding to the wrapped client: the client is held privately and only the
    two non-moving calls are re-exported. An agent that has been prompt-injected
    can create and authorize orders; it cannot turn one into money.
    """

    __slots__ = ("_client",)

    def __init__(self, client: PayPalClient) -> None:
        self._client = client

    def create_order(self, body: dict[str, Any], *, request_id: str) -> dict[str, Any]:
        if body.get("intent") != "AUTHORIZE":
            raise ValueError("an agent may only create AUTHORIZE orders; CAPTURE is the gate's")
        return self._client.create_order(body, request_id=request_id)

    def authorize_order(self, order_id: str, *, request_id: str) -> dict[str, Any]:
        return self._client.authorize_order(order_id, request_id=request_id)


def make_client(config: PayPalConfig, *, record: bool = False) -> PayPalClient:
    """The client for ``config.mode``."""
    if config.mode == "live":
        return LiveClient(config, record=record)
    return MockClient(config.fixture_dir)
