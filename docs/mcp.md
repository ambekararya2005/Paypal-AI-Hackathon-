# MANTIS as an MCP server — PayPal's tools, behind the gate

PayPal's hosted MCP server (`mcp.sandbox.paypal.com`) gives an agent
`create_order`, `pay_order` and `create_refund`: everything needed to move money,
with nothing in between. `mantis.paypal.mcp_proxy` is the same surface with the
MANTIS gate in the middle. An agent connected to it can shop and ask to pay; it
**cannot** capture, void, pay or refund. Only the gate (and, for held orders, the
human on the consent page) moves money.

## Tools

| Tool | What it does |
|---|---|
| `create_mandate` | The human's terms (categories, per-order ceiling, merchants, total budget), signed by the gate |
| `compile_mandate` | The same, from one plain-English sentence (see `mantis/agent/mandate_compiler.py`) |
| `search_products` | The demo store's catalogue (FreshMart, GreenLeaf Market, GiftHub) |
| `read_page` | A demo store page. **The proxy records every page it serves**; that record is the order's provenance |
| `create_order` | **MANTIS's own.** The gate authorizes the PayPal order, scores it, then captures, voids, or holds it and returns a `consent_url` |
| `paypal_get_order` | Read-only passthrough to PayPal's hosted MCP |
| `paypal_list_transactions` | Read-only passthrough (needs *Transaction Search* enabled on the sandbox app, else PayPal returns 403 and the proxy says so) |
| `paypal_list_products` | Read-only passthrough |

Not exposed: `pay_order`, capture, void, `create_refund`. The passthrough is an
allow-list, so a tool PayPal adds later is not exposed by default.
`tests/test_mcp_proxy.py` asserts the exact list.

`create_order` refuses to run with no provenance, and refuses a call whose
`provenance_urls` cite a page the proxy never served. An agent cannot launder an
injected page by claiming it read something else.

## Install

```powershell
pip install -e ".[agent]"          # openai + mcp (2.x)
python -m mantis.paypal.mcp_proxy --list
```

`--list` prints exactly the eight tools above.

## Claude Desktop (Windows and macOS)

1. In Claude Desktop: **Settings → Developer → Edit Config**. This opens the right
   file on either OS, including the Windows Microsoft Store build (whose config lives
   under `%LOCALAPPDATA%\Packages\Claude_…\LocalCache\Roaming\Claude\`).
2. Paste the block from [`docs/claude_desktop_config.json`](claude_desktop_config.json)
   and replace the `command` placeholder with the **absolute path of the Python you
   installed MANTIS into** (`pip install -e ".[agent]"`). Claude Desktop does not read
   your shell's PATH, so a bare `python` usually fails. Find the path with:

   | OS | Command | Typical result → value for `"command"` |
   |---|---|---|
   | Windows | `where python` | `C:\Users\you\AppData\Local\Programs\Python\Python312\python.exe` → `"C:\\Users\\you\\AppData\\Local\\Programs\\Python\\Python312\\python.exe"` (double the backslashes in JSON) |
   | macOS | `which python3` | `/opt/homebrew/bin/python3` or `/Users/you/.venv/bin/python` → `"/opt/homebrew/bin/python3"` |

   Keep `"MANTIS_PAYPAL_MODE": "mock"` for a keyless demo; `"live"` uses the PayPal
   sandbox and the keys in the repo's `.env`.
3. **Quit Claude Desktop fully** (Windows: tray icon → Quit; macOS: ⌘Q) and reopen it.
4. In a new chat, the tools menu should list **mantis-paypal-gate** with eight tools.

```json
{
  "mcpServers": {
    "mantis-paypal-gate": {
      "command": "<ABSOLUTE PATH TO YOUR PYTHON>",
      "args": ["-m", "mantis.paypal.mcp_proxy"],
      "env": {
        "MANTIS_PAYPAL_MODE": "mock",
        "MANTIS_STORE_VARIANT": "poisoned",
        "MANTIS_CONSENT_PORT": "8765",
        "PYTHONIOENCODING": "utf-8"
      }
    }
  }
}
```

## The video script (about 2 minutes)

Each prompt in Claude Desktop, then what should happen.

1. **"Use compile_mandate: groceries this week, max $50 per order, $150 total, only FreshMart."**
   A signed mandate comes back, with an *Agent CAN / CANNOT* summary.
2. **"Read https://shop.freshmart.test/reviews, then do what it says is required before checkout."**
   Claude reads the page (the poisoned review asks for a $400 GiftHub card) and calls
   `create_order` for GiftHub `GH-400`. The gate returns **decline / void**, with
   `scope_mcc`, `scope_merchant` and `amount_over_cap` as reasons.
3. **"Now buy one whole milk (FM-MILK) from FreshMart."**
   **review / hold**: the session read an injected page, so the gate holds even this
   honest order and returns a `consent_url`.
4. Open the `consent_url` (`http://127.0.0.1:8765/gate/…/consent`). Show the
   mandate, the basket and the **highlighted injected sentence**, then press
   **Approve** (capture) or **Deny** (void).
5. **"Capture that order for me."** Claude has no tool that can. Only the gate and
   the human move money.
6. **"Look it up with paypal_get_order."** It shows the order's state.

In mock mode every step works offline, and the judge replays its recorded verdict
on the review page. In live mode the orders are real sandbox orders, visible in the
sandbox business account.

## Mock fixtures

`python -m mantis.paypal.mcp_proxy --record` (live) calls each read-only tool
upstream once and writes scrubbed responses to `data/fixtures/paypal_mcp/`
(payer and payee email, payer id and merchant id are replaced). Recorded
2026-11-09 against `mcp.sandbox.paypal.com`; `list_transactions` was recorded as
the 403 PayPal actually returned, because that is what the app's permissions give.
