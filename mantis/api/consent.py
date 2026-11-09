"""The human-consent page for a held order: ``GET /gate/{order_id}/consent``.

One self-contained HTML page — no build step, no script from anywhere — showing
what a person needs to approve or deny an agent's purchase:

* the **mandate** they signed, in plain terms (what the agent may buy, where, how
  much per order and in total, until when);
* the **order** the agent wants to pay for (merchant, basket, amount);
* the gate's **decision and reasons**;
* the **evidence**: every page the agent read that a text layer flagged, with the
  judge's quoted span **highlighted in the page itself**;
* **Approve** (capture) and **Deny** (void), posting to the existing
  ``POST /gate/{order_id}/consent``.

The page renders attacker-written text
---------------------------------------
The highlighted page *is* the injection. Every string from a record is passed
through :func:`html.escape` before it is placed in the document, and the quoted
span is located and wrapped in ``<mark>`` **after** escaping, so nothing in a
page can become markup. ``tests/test_agent.py`` plants a ``<script>`` in a page
and asserts it arrives as text.
"""

from __future__ import annotations

import html
import json
import re
from datetime import timedelta
from typing import Any

__all__ = ["render_consent_page"]


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


def highlight(text: str, span: str) -> str:
    """Escaped ``text`` with the first occurrence of ``span`` wrapped in <mark>.

    Matching tolerates whitespace and case differences, as the judge's own
    verification does. If the span cannot be found the text is returned escaped
    and unmarked — the page never invents a highlight.
    """
    if not span.strip():
        return _e(text)
    words = [re.escape(w) for w in span.split()]
    match = re.search(r"\s+".join(words), text, flags=re.IGNORECASE)
    if match is None:
        return _e(text)
    a, b = match.span()
    return f"{_e(text[:a])}<mark>{_e(text[a:b])}</mark>{_e(text[b:])}"


_STYLE = """
:root { --bg:#f6f7f9; --card:#fff; --ink:#14171f; --muted:#5b6475; --line:#e2e5eb;
  --ok:#15803d; --bad:#b91c1c; --warn:#a16207; --mark:#fde68a; --accent:#003087; }
@media (prefers-color-scheme: dark) { :root { --bg:#0f1115; --card:#181b22; --ink:#e7e9ee;
  --muted:#9aa3b2; --line:#2a2f3a; --ok:#4ade80; --bad:#f87171; --warn:#facc15;
  --mark:#854d0e; --accent:#7aa7ff; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:860px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:15px; margin:0 0 10px;
  text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:16px 18px; margin:14px 0; }
.pill { display:inline-block; padding:2px 10px; border-radius:999px; font-weight:600;
  font-size:13px; border:1px solid currentColor; }
.decline { color:var(--bad); } .review,.challenge,.held { color:var(--warn); }
.approve,.captured { color:var(--ok); } .voided { color:var(--bad); }
table { width:100%; border-collapse:collapse; } td,th { text-align:left; padding:6px 4px;
  border-bottom:1px solid var(--line); vertical-align:top; } th { color:var(--muted);
  font-weight:500; font-size:13px; } .num { text-align:right; font-variant-numeric:tabular-nums; }
ul.reasons { margin:0; padding-left:18px; } ul.reasons li { margin:4px 0; }
code { font-size:13px; } .layer { font-weight:600; }
pre.page { white-space:pre-wrap; word-break:break-word; background:var(--bg);
  border:1px solid var(--line); border-radius:8px; padding:12px; margin:8px 0 0;
  font:13px/1.5 ui-monospace,Consolas,monospace; max-height:320px; overflow:auto; }
mark { background:var(--mark); color:inherit; padding:0 2px; border-radius:3px; }
.meta { color:var(--muted); font-size:13px; }
.actions { display:flex; gap:12px; flex-wrap:wrap; margin-top:6px; }
button { font:inherit; font-weight:600; padding:10px 22px; border-radius:8px; cursor:pointer;
  border:1px solid var(--line); background:var(--card); color:var(--ink); }
button.approve { border-color:var(--ok); } button.deny { border-color:var(--bad); }
button:disabled { opacity:.45; cursor:default; }
#result { margin-top:10px; font-weight:600; }
"""


def _mandate_block(mandate: Any) -> str:  # Any: mantis.paypal.models.Mandate
    if mandate is None:
        return '<p class="meta">Mandate not available in this process.</p>'
    scope = mandate.scope
    expires = "no expiry"
    if scope.ttl_seconds:
        expires = (mandate.issued_ts + timedelta(seconds=scope.ttl_seconds)).strftime(
            "%Y-%m-%d %H:%M UTC")
    rows = [
        ("You asked for", mandate.description or "-"),
        ("Categories (MCC)", ", ".join(scope.categories) or "any"),
        ("Merchants", ", ".join(scope.allowed_merchants) or "any"),
        ("Per order", f"${scope.max_amount:.2f}" if scope.max_amount else "no limit"),
        ("In total", f"${mandate.budget_total:.2f}" if mandate.budget_total else "no limit"),
        ("Valid until", expires),
        ("Mandate", f"{mandate.mandate_id} &middot; hash <code>{_e(mandate.digest()[:16])}</code>"),
    ]
    body = "".join(f"<tr><th>{_e(k)}</th><td>{v if k == 'Mandate' else _e(v)}</td></tr>"
                   for k, v in rows)
    return f"<table>{body}</table>"


def _evidence_block(record: dict[str, Any]) -> str:
    flagged = [p for p in record.get("evidence", [])
               if (p.get("judge") or {}).get("injected") or p.get("l3_fired")]
    if not flagged:
        pages = record.get("evidence", [])
        return (f'<p class="meta">{len(pages)} page(s) read; none flagged by a text layer.</p>')
    out = []
    for page in flagged:
        judge = page.get("judge") or {}
        span = judge.get("quoted_span", "") if judge.get("injected") else ""
        l3 = ("abstained (vocabulary coverage "
              f"{page.get('coverage', 0):.2f})" if page.get("l3_abstained")
              else f"P(injected) {page.get('l3', 0):.2f}")
        out.append(
            f'<div class="card"><div><b>{_e(page["url"])}</b></div>'
            f'<div class="meta">L3: {_e(l3)} &middot; judge: '
            f'{"injection" if judge.get("injected") else "no opinion"}'
            f'{" — wants the agent to " + _e(judge.get("target_action")) if span else ""}</div>'
            + (f'<div class="meta">{_e(judge.get("rationale", ""))}</div>' if judge else "")
            + f'<pre class="page">{highlight(page.get("text", ""), span)}</pre></div>'
        )
    return "".join(out)


def render_consent_page(record: dict[str, Any], mandate: Any) -> str:  # Any: Mandate | None
    """The full HTML page for one gate record."""
    order_id = record["order_id"]
    status = record["status"]
    held = status == "held"
    items = "".join(
        f"<tr><td>{_e(i['name'])}</td><td><code>{_e(i['sku'])}</code></td>"
        f"<td class='num'>{_e(i['quantity'])}</td><td class='num'>${_e(i['unit_amount'])}</td></tr>"
        for i in record.get("items", [])
    )
    reasons = "".join(
        f"<li><span class='layer'>[{_e(r['layer'])}]</span> <code>{_e(r['code'])}</code> "
        f"&mdash; {_e(r['detail'])}</li>"
        for r in record.get("all_reasons") or record.get("reasons", [])
    )
    decision = _e(record["decision"])
    payload = json.dumps(order_id)
    disabled = "" if held else " disabled"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Approve agent payment</title><style>{_STYLE}</style></head>
<body><main>
<h1>Your shopping agent wants to pay</h1>
<p class="meta">MANTIS held this PayPal authorization for your decision. Funds are reserved,
nothing has moved. Order <code>{_e(order_id)}</code> &middot;
authorization <code>{_e(record['authorization_id'])}</code></p>

<div class="card"><h2>Decision</h2>
<span class="pill {decision}">{decision.upper()}</span>
<span class="pill {_e(status)}" id="status">{_e(status.upper())}</span>
<span class="meta">&nbsp;layers: {_e(' + '.join(record.get('layers_ran', [])))}
&middot; scoring {_e(record.get('latency_ms'))} ms
&middot; judge {_e(record.get('judge_ms'))} ms</span>
<ul class="reasons" style="margin-top:10px">{reasons}</ul></div>

<div class="card"><h2>The order</h2>
<p><b>{_e(record['merchant_id'])}</b> &middot; <b>${_e(record['amount'])}
{_e(record['currency'])}</b> &middot; paid by {_e(record.get('payment_source'))}</p>
<table><tr><th>Item</th><th>SKU</th><th class="num">Qty</th><th class="num">Each</th></tr>
{items}</table></div>

<div class="card"><h2>What you signed</h2>{_mandate_block(mandate)}</div>

<div class="card"><h2>What the agent read</h2>{_evidence_block(record)}</div>

<div class="card"><h2>Your call</h2>
<p class="meta">Approve captures the payment. Deny voids the authorization and releases
the funds.</p>
<div class="actions">
<button class="approve" id="approve"{disabled}>Approve &mdash; capture</button>
<button class="deny" id="deny"{disabled}>Deny &mdash; void</button></div>
<div id="result"></div></div>
</main>
<script>
const orderId = {payload};
async function decide(approve) {{
  for (const b of document.querySelectorAll("button")) b.disabled = true;
  const out = document.getElementById("result");
  out.textContent = approve ? "Capturing..." : "Voiding...";
  const base = location.pathname.replace(/\\/consent\\/?$/, "");
  try {{
    const r = await fetch(base + "/consent", {{method: "POST",
      headers: {{"Content-Type": "application/json"}},
      body: JSON.stringify({{approve, signer: "human (consent page)"}})}});
    const body = await r.json();
    if (!r.ok) throw new Error(body.detail || r.status);
    document.getElementById("status").textContent = body.status.toUpperCase();
    document.getElementById("status").className = "pill " + body.status;
    out.textContent = body.status === "captured"
      ? "Captured. PayPal capture " + body.capture_id
      : "Voided. The authorization is released.";
  }} catch (e) {{ out.textContent = "Failed: " + e.message; }}
}}
document.getElementById("approve").onclick = () => decide(true);
document.getElementById("deny").onclick = () => decide(false);
</script></body></html>"""
