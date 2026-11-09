"""MANTIS as an MCP server: PayPal's own tools, behind the gate.

    python -m mantis.paypal.mcp_proxy              # stdio MCP server (what Claude Desktop runs)
    python -m mantis.paypal.mcp_proxy --list       # print the exposed tools and exit
    python -m mantis.paypal.mcp_proxy --record     # live: re-record the read-only fixtures

What it is
----------
PayPal publishes a hosted MCP server (``mcp.sandbox.paypal.com``, the Agent
Toolkit's tool surface). It hands an agent ``create_order``, ``pay_order`` and
``create_refund``: everything needed to move money end to end, with nothing in
between. This proxy is the same idea with the gate in the middle:

=========================  ==================================================
exposed tool               what it does
=========================  ==================================================
``create_mandate``         a human's terms, signed (HMAC) by the gate
``compile_mandate``        the same, from one plain-English sentence
``search_products``        the demo store's catalogue
``read_page``              a demo store page — **the proxy records it** as
                           provenance; nothing the client says can add to it
``create_order``           **MANTIS's own**: the gate authorizes, scores and
                           captures, voids or holds. Requires a mandate id;
                           provenance is the proxy's record of pages served
``paypal_get_order``       read-only passthrough to PayPal's hosted MCP
``paypal_list_transactions`` read-only passthrough
``paypal_list_products``   read-only passthrough
=========================  ==================================================

**Not exposed, by construction:** ``pay_order``, ``capture``, ``void``,
``create_refund``, or any upstream tool not on :data:`READ_ONLY_PASSTHROUGH`.
The passthrough is an allow-list, never a deny-list, so a tool PayPal adds next
month is not exposed by default. ``tests/test_mcp_proxy.py`` asserts the tool list.

Provenance
----------
``create_order`` accepts an optional ``provenance_urls`` list, so a call can name
what it relied on. Every URL must be one this proxy actually served through
``read_page``; anything else is refused. The chain the gate scores is always the
proxy's own record, with the text it served.

Modes
-----
``MANTIS_PAYPAL_MODE=mock`` (default): the gate uses the mock PayPal state machine
and the judge replays its cache; passthroughs replay scrubbed recorded upstream
responses from ``data/fixtures/paypal_mcp/``. ``live``: the PayPal sandbox and its
hosted MCP server, token minted from the app credentials in ``.env``.

Held orders
-----------
A held order needs a human. On the first hold, the proxy starts the consent page
on ``http://127.0.0.1:8765`` in this process (``MANTIS_CONSENT_PORT`` to change),
so the link it returns works in mock mode as well as live. Logging goes to
stderr: stdout belongs to the MCP protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

from mantis.core.paths import DATA_DIR

__all__ = ["FORBIDDEN", "READ_ONLY_PASSTHROUGH", "Proxy", "build_server"]

#: Upstream tools passed through, local name -> upstream name. An allow-list.
READ_ONLY_PASSTHROUGH: Final[dict[str, str]] = {
    "paypal_get_order": "get_order",
    "paypal_list_transactions": "list_transactions",
    "paypal_list_products": "list_products",
}

#: Words no exposed tool name may contain. Checked at build time and by the tests.
FORBIDDEN: Final[tuple[str, ...]] = ("capture", "pay_order", "refund", "void", "authorize")

UPSTREAM_SSE: Final[str] = "https://mcp.sandbox.paypal.com/sse"
FIXTURE_DIR: Final[Path] = DATA_DIR / "fixtures" / "paypal_mcp"
#: An order the recorded get_order fixture was taken from (live run D, 2026-11-09).
RECORDED_ORDER_ID: Final[str] = "9DB39582P4348282B"


def _log(message: str) -> None:
    print(f"[mantis-mcp] {message}", file=sys.stderr, flush=True)


class Proxy:
    """The state behind the tools: one gate, one session's provenance."""

    def __init__(self, mode: str | None = None) -> None:
        from mantis.paypal.config import load_config

        self.config = load_config(mode)
        self.mode = self.config.mode
        self._gate: Any = None  # Any: PaymentGate, built lazily (L3 fit takes seconds)
        self._store: Any = None  # Any: mantis.agent.store.Store
        #: (url, text) served by read_page, in order. THE provenance.
        self.served: list[tuple[str, str]] = []
        self._consent_started = False

    # -- lazies ------------------------------------------------------------ #

    @property
    def store(self) -> Any:  # Any: Store
        if self._store is None:
            from mantis.agent.store import Store

            self._store = Store(os.environ.get("MANTIS_STORE_VARIANT", "poisoned"))  # type: ignore[arg-type]
        return self._store

    @property
    def gate(self) -> Any:  # Any: PaymentGate
        if self._gate is None:
            from mantis.defense.l3_judge import InjectionJudge
            from mantis.defense.live import LiveScorer
            from mantis.paypal.gate import PaymentGate

            judge = InjectionJudge("live" if self.mode == "live" else "mock")
            self._gate = PaymentGate(self.config,
                                     scorer=LiveScorer(use_l1=False, judge=judge))
        return self._gate

    # -- local tools --------------------------------------------------------- #

    def create_mandate(self, description: str, categories: list[str], max_amount: float,
                       allowed_merchants: list[str], budget_total: float | None = None,
                       ttl_days: int = 7) -> dict[str, Any]:
        from mantis.core.events import MandateScope, MandateType
        from mantis.paypal.models import MandateRequest

        request = MandateRequest(
            customer_id="cust-mcp", agent_id="agent-shopper-01", mandate_type=MandateType.INTENT,
            scope=MandateScope(categories=categories, max_amount=max_amount,
                               allowed_merchants=allowed_merchants, ttl_seconds=ttl_days * 86_400),
            description=description,
            budget_total=Decimal(str(budget_total)) if budget_total else None,
            vault_id=self.gate.vault_id(),
        )
        mandate = self.gate.create_mandate(request)
        return {"mandate_id": mandate.mandate_id, "mandate_hash": mandate.digest(),
                "signed": bool(mandate.signature), "scope": mandate.scope.model_dump(),
                "budget_total": str(mandate.budget_total) if mandate.budget_total else None}

    def compile_mandate(self, text: str) -> dict[str, Any]:
        """One sentence -> a signed mandate, or questions. Never a guessed limit."""
        from mantis.agent.mandate_compiler import MandateCompiler

        result = MandateCompiler("live" if self.mode == "live" else "mock").compile(text)
        out = result.as_dict()
        if result.status == "ok" and result.request is not None:
            request = result.request.model_copy(update={"customer_id": "cust-mcp",
                                                        "vault_id": self.gate.vault_id()})
            mandate = self.gate.create_mandate(request)
            out.update({"mandate_id": mandate.mandate_id, "mandate_hash": mandate.digest(),
                        "signed": bool(mandate.signature)})
        out["summary"] = result.summary()
        return out

    def search_products(self, query: str) -> str:
        hits = self.store.search(query)
        return "\n".join(self.store.listing(p) for p in hits[:12]) or "No products found."

    def read_page(self, url: str) -> str:
        text = self.store.page(url)
        if text is None:
            return f"404: no page at {url}"
        self.served.append((url.strip().rstrip("/"), text))
        return text

    def create_order(self, mandate_id: str, merchant: str, items: list[dict[str, Any]],
                     provenance_urls: list[str] | None = None) -> dict[str, Any]:
        from mantis.agent.store import MERCHANT_NAMES
        from mantis.paypal.gate import GateError
        from mantis.paypal.models import AgentContext, OrderItem, Page, PurchaseIntent
        from mantis.paypal.scenarios import AGENT_ID, AGENT_PLATFORM, KYA_TOKEN

        served_urls = {u for u, _ in self.served}
        foreign = [u for u in provenance_urls or [] if u.strip().rstrip("/") not in served_urls]
        if foreign:
            return {"error": "provenance cites pages this proxy never served", "urls": foreign}
        if not self.served:
            return {"error": "no provenance: read at least one page through read_page first"}
        ids = {n.lower(): m for m, n in MERCHANT_NAMES.items()} | {m: m for m in MERCHANT_NAMES}
        merchant_id = ids.get(merchant.strip().lower())
        if merchant_id is None:
            return {"error": f"unknown merchant {merchant!r}"}
        lines = []
        for item in items:
            product = self.store.product(str(item.get("sku", "")))
            if product is None or product.merchant_id != merchant_id:
                return {"error": f"sku {item.get('sku')!r} is not sold by {merchant}"}
            lines.append(OrderItem(name=product.name, sku=product.sku,
                                   quantity=int(item.get("quantity", 1)),
                                   unit_amount=product.price,
                                   category="DIGITAL_GOODS" if product.category == "gift_card"
                                   else "PHYSICAL_GOODS"))
        intent = PurchaseIntent(mandate_id=mandate_id, merchant_id=merchant_id, items=lines,
                                description=f"{MERCHANT_NAMES[merchant_id]} order via MCP")
        context = AgentContext(agent_id=AGENT_ID, agent_platform=AGENT_PLATFORM,
                               kya_token=KYA_TOKEN,
                               pages=[Page(url=u, text=t) for u, t in self.served],
                               tool_call_count=len(self.served) + 1)
        try:
            record = self.gate.submit(intent, context)
        except GateError as exc:
            return {"error": str(exc)}
        out = {
            "decision": record.decision, "paypal_action": record.paypal_action,
            "status": record.status, "order_id": record.order_id,
            "authorization_id": record.authorization_id, "capture_id": record.capture_id,
            "amount": record.amount, "currency": record.currency,
            "layers_ran": record.layers_ran, "reasons": record.reasons,
            "receipt_hash": record.receipt_hash, "mode": self.mode,
        }
        if record.status == "held":
            out["consent_url"] = self._consent_url(record.order_id)
        return out

    # -- passthrough ----------------------------------------------------------- #

    def passthrough(self, local: str, arguments: dict[str, Any]) -> str:
        upstream = READ_ONLY_PASSTHROUGH[local]
        if self.mode != "live":
            return self._replay(upstream, arguments)
        import anyio

        return anyio.run(self._call_upstream, upstream, arguments)

    async def _call_upstream(self, name: str, arguments: dict[str, Any]) -> str:
        from mcp.client.session import ClientSession
        from mcp.client.sse import sse_client

        from mantis.paypal.client import LiveClient

        token = LiveClient(self.config).access_token()
        async with sse_client(UPSTREAM_SSE, headers={"Authorization": f"Bearer {token}"},
                              timeout=20) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)
            return "".join(getattr(c, "text", "") for c in result.content)

    def _replay(self, upstream: str, arguments: dict[str, Any]) -> str:
        if upstream == "get_order" and self._gate is not None:
            try:  # an order this mock session created: answer from the mock's state
                return json.dumps(self._gate._client.get_order(str(arguments.get("id"))))
            except Exception:
                pass
        path = FIXTURE_DIR / f"{upstream}.json"
        if not path.exists():
            return json.dumps({"error": f"no recorded fixture for {upstream}; run --record"})
        fixture = json.loads(path.read_text(encoding="utf-8"))
        return json.dumps({"mock": "recorded PayPal sandbox MCP response",
                           "recorded": fixture.get("recorded"), "response": fixture["response"]})

    # -- consent ------------------------------------------------------------------ #

    def _consent_url(self, order_id: str) -> str:
        port = int(os.environ.get("MANTIS_CONSENT_PORT", "8765"))
        if not self._consent_started:
            self._consent_started = True
            threading.Thread(target=self._serve_consent, args=(port,), daemon=True).start()
        return f"http://127.0.0.1:{port}/gate/{order_id}/consent"

    def _serve_consent(self, port: int) -> None:
        import uvicorn

        import mantis.api.gate as api_gate
        from mantis.api.app import app

        api_gate._GATE = self.gate
        _log(f"consent page on http://127.0.0.1:{port}")
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def build_server(proxy: Proxy | None = None) -> Any:  # Any: mcp.server.mcpserver.MCPServer
    """The MCP server with exactly the tools in the module docstring."""
    from mcp.server.mcpserver import MCPServer

    p = proxy or Proxy()
    server = MCPServer(
        name="mantis-paypal-gate",
        instructions=(
            "PayPal payments behind the MANTIS gate. Create a mandate first, read store pages "
            "with read_page, then call create_order. The gate decides: approve captures, "
            "decline voids, review holds for the human (a consent_url is returned). There is "
            "no tool to capture, void, pay or refund: only the gate and the human move money."
        ),
    )

    @server.tool(description="Sign a mandate: what the agent may buy, where, and how much.")
    def create_mandate(description: str, categories: list[str], max_amount: float,
                       allowed_merchants: list[str], budget_total: float | None = None,
                       ttl_days: int = 7) -> str:
        return json.dumps(p.create_mandate(description, categories, max_amount,
                                           allowed_merchants, budget_total, ttl_days))

    @server.tool(description="Sign a mandate from one plain-English sentence, e.g. 'groceries "
                             "this week, max $50/order, $150 total, only FreshMart'. Returns "
                             "what the agent CAN and CANNOT do, or questions if it is ambiguous.")
    def compile_mandate(text: str) -> str:
        return json.dumps(p.compile_mandate(text))

    @server.tool(description="Search the demo store (FreshMart, GreenLeaf Market, GiftHub).")
    def search_products(query: str) -> str:
        return p.search_products(query)

    @server.tool(description="Read a demo store page, e.g. https://shop.freshmart.test/reviews. "
                             "The proxy records every page it serves as the order's provenance.")
    def read_page(url: str) -> str:
        return p.read_page(url)

    @server.tool(description="Create a PayPal order through the MANTIS gate. items: "
                             "[{\"sku\": ..., \"quantity\": ...}]. The gate authorizes, scores, "
                             "then captures, voids, or holds it for human consent.")
    def create_order(mandate_id: str, merchant: str, items: list[dict[str, Any]],
                     provenance_urls: list[str] | None = None) -> str:
        return json.dumps(p.create_order(mandate_id, merchant, items, provenance_urls))

    @server.tool(description="Read-only: get a PayPal order (PayPal's hosted MCP).")
    def paypal_get_order(id: str) -> str:
        return p.passthrough("paypal_get_order", {"id": id})

    @server.tool(description="Read-only: list PayPal transactions (PayPal's hosted MCP).")
    def paypal_list_transactions(page_size: int = 10) -> str:
        return p.passthrough("paypal_list_transactions", {"page_size": page_size})

    @server.tool(description="Read-only: list PayPal catalog products (PayPal's hosted MCP).")
    def paypal_list_products(page_size: int = 10) -> str:
        return p.passthrough("paypal_list_products", {"page_size": page_size})

    exposed = ("create_mandate", "compile_mandate", "search_products", "read_page",
               "create_order",
               *READ_ONLY_PASSTHROUGH)
    leaked = [n for n in exposed if any(word in n for word in FORBIDDEN)]
    if leaked:  # build-time guard; tests/test_mcp_proxy.py checks the live tool list too
        raise RuntimeError(f"money-moving tool exposed: {leaked}")
    return server


def record_fixtures() -> int:
    """Live: call each passthrough upstream once and save scrubbed responses."""
    import anyio

    from mantis.paypal.client import _scrub

    proxy = Proxy("live")
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for upstream, args in (("get_order", {"id": RECORDED_ORDER_ID}),
                           ("list_transactions", {"page_size": 10}),
                           ("list_products", {"page_size": 10})):
        text = anyio.run(proxy._call_upstream, upstream, args)
        try:
            body: Any = _scrub(json.loads(text))  # Any: arbitrary JSON
        except ValueError:
            body = text
        from datetime import UTC, datetime

        (FIXTURE_DIR / f"{upstream}.json").write_text(json.dumps(
            {"tool": upstream, "arguments": args, "source": UPSTREAM_SSE,
             "recorded": datetime.now(UTC).isoformat(timespec="seconds"), "response": body},
            indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _log(f"recorded {upstream}: {str(body)[:90]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mantis.paypal.mcp_proxy")
    parser.add_argument("--list", action="store_true", help="print the exposed tools and exit")
    parser.add_argument("--record", action="store_true", help="live: re-record fixtures")
    args = parser.parse_args(argv)
    if args.record:
        return record_fixtures()
    server = build_server()
    if args.list:
        import anyio

        tools = anyio.run(_tool_names, server)
        print("\n".join(tools))
        return 0
    _log(f"starting stdio server, mode={Proxy().mode}")
    server.run("stdio")
    return 0


async def _tool_names(server: Any) -> list[str]:  # Any: MCPServer
    import inspect

    listed = server.list_tools()
    if inspect.isawaitable(listed):
        listed = await listed
    return sorted(t.name for t in listed)


if __name__ == "__main__":
    sys.exit(main())
