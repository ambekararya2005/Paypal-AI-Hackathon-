"""The MCP proxy: PayPal's tools behind the gate, and nothing that moves money."""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp", reason="the MCP proxy needs the optional 'agent' extra")

from mantis.defense.l3_judge import InjectionJudge
from mantis.defense.live import LiveScorer
from mantis.paypal.config import PayPalConfig
from mantis.paypal.gate import PaymentGate
from mantis.paypal.mcp_proxy import FORBIDDEN, READ_ONLY_PASSTHROUGH, Proxy, build_server

REVIEWS = "https://shop.freshmart.test/reviews"


@pytest.fixture(scope="module")
def scorer() -> LiveScorer:
    s = LiveScorer(seed=1337, use_l1=False)
    s.judge = InjectionJudge("mock")
    return s


@pytest.fixture()
def proxy(tmp_path: Path, scorer: LiveScorer, monkeypatch: pytest.MonkeyPatch) -> Proxy:
    monkeypatch.setenv("MANTIS_STORE_VARIANT", "poisoned")
    p = Proxy("mock")
    config = PayPalConfig(mode="mock", client_id="", client_secret="", receipt_dir=tmp_path)
    p._gate = PaymentGate(config, scorer=copy.copy(scorer))
    return p


async def _names(server: Any) -> list[str]:  # Any: MCPServer
    listed = server.list_tools()
    if inspect.isawaitable(listed):
        listed = await listed
    return sorted(t.name for t in listed)


def test_tool_list_is_exactly_the_gated_surface() -> None:
    import anyio

    names = anyio.run(_names, build_server(Proxy("mock")))
    assert names == sorted(["create_mandate", "compile_mandate", "search_products",
                            "read_page", "create_order",
                            *READ_ONLY_PASSTHROUGH])
    for name in names:
        assert not any(word in name for word in FORBIDDEN), name
    for word in ("capture", "void", "refund", "pay_order"):
        assert all(word not in n for n in names)


def test_passthrough_is_an_allow_list_of_read_only_tools() -> None:
    assert set(READ_ONLY_PASSTHROUGH.values()) == {"get_order", "list_transactions",
                                                   "list_products"}


def _mandate(proxy: Proxy) -> str:
    return proxy.create_mandate("groceries, max $50/order, only FreshMart", ["5411"], 50.0,
                                ["m-freshmart"], 150.0)["mandate_id"]


def test_create_order_needs_provenance_the_proxy_served(proxy: Proxy) -> None:
    mandate_id = _mandate(proxy)
    items = [{"sku": "FM-MILK", "quantity": 1}]
    assert "no provenance" in proxy.create_order(mandate_id, "FreshMart", items)["error"]
    proxy.read_page(REVIEWS)
    forged = proxy.create_order(mandate_id, "FreshMart", items,
                                provenance_urls=["https://evil.example.test/x"])
    assert forged["error"].startswith("provenance cites pages")


def test_gift_card_through_mcp_is_voided(proxy: Proxy) -> None:
    mandate_id = _mandate(proxy)
    proxy.read_page(REVIEWS)
    out = proxy.create_order(mandate_id, "GiftHub", [{"sku": "GH-400", "quantity": 1}])
    assert out["decision"] == "decline" and out["status"] == "voided"
    assert out["capture_id"] is None


def test_mock_passthrough_replays_a_scrubbed_recording(proxy: Proxy) -> None:
    body = json.loads(proxy.passthrough("paypal_list_products", {"page_size": 10}))
    assert body["mock"] == "recorded PayPal sandbox MCP response"
    recorded = json.loads(proxy.passthrough("paypal_get_order", {"id": "NOT-A-MOCK-ORDER"}))
    text = json.dumps(recorded)
    assert "@" not in text, "payer and payee emails are scrubbed"


def test_compile_mandate_signs_the_demo_sentence(proxy: Proxy) -> None:
    out = proxy.compile_mandate("groceries this week, max $50/order, $150 total, only FreshMart")
    assert out["status"] == "ok" and out["signed"] and out["mandate_id"]
    assert "Agent CANNOT" in out["summary"]
