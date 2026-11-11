"""The human-consent page for a held order: ``GET /gate/{order_id}/consent``.

One self-contained HTML page — no build step, no script from anywhere — laid out
like a payment authorization prompt, showing what a person needs to approve or
deny an agent's purchase:

* the **amount and merchant**, and the PayPal order it belongs to;
* the **mandate** they signed, in plain terms (what the agent may buy, where, how
  much per order and in total, until when);
* the **evidence**: every page the agent read that a text layer flagged, with the
  judge's quoted span **highlighted in the page itself**, and the judge's reason;
* the gate's **decision and reasons**;
* **Deny and void** and **Approve and capture**, posting to the existing
  ``POST /gate/{order_id}/consent``.

The page renders attacker-written text
---------------------------------------
The highlighted page *is* the injection. Every string from a record is passed
through :func:`html.escape` before it is placed in the document, and the quoted
span is located and wrapped in ``<mark>`` **after** escaping, so nothing in a
page can become markup. ``tests/test_agent.py`` plants a ``<script>`` in a page
and asserts it arrives as text, including inside the highlight.
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


_STATUS_CODES = frozenset({"protocol_ok", "not_loaded", "provenance_clean"})

_MERCHANTS = {"m-freshmart": "FreshMart", "m-greenleaf": "GreenLeaf Market", "m-gifthub": "GiftHub"}

#: The app's tokens (web/src/tokens.css), inlined: this page has no build step.
#: IBM Plex loads from the same origin's /fonts/ when the site bundle is served
#: and falls back to system-ui when it is not (the bare API).
_STYLE = """
@font-face { font-family:"IBM Plex Sans"; font-weight:400; font-display:swap;
  src:url("/fonts/ibm-plex-sans-latin-400-normal.woff2") format("woff2"); }
@font-face { font-family:"IBM Plex Sans"; font-weight:500; font-display:swap;
  src:url("/fonts/ibm-plex-sans-latin-500-normal.woff2") format("woff2"); }
@font-face { font-family:"IBM Plex Sans"; font-weight:600; font-display:swap;
  src:url("/fonts/ibm-plex-sans-latin-600-normal.woff2") format("woff2"); }
@font-face { font-family:"IBM Plex Mono"; font-weight:400; font-display:swap;
  src:url("/fonts/ibm-plex-mono-latin-400-normal.woff2") format("woff2"); }
:root { --bg:#ffffff; --sidebar:#f7f7f5; --surface-2:#f2f2ef; --border:#e6e5e1;
  --border-2:#d4d3ce; --text:#1a1a18; --text-2:#5f5e5a; --accent:#1d6b48;
  --accent-hover:#175a3c; --ok-fg:#1d6b48; --ok-bg:#e8f2ec; --warn-fg:#8a5a00;
  --warn-bg:#fbf1dc; --bad-fg:#b42318; --bad-bg:#fbe9e7; --mark:#fff0a8; color-scheme:light; }
* { box-sizing:border-box; }
body { margin:0; background:var(--sidebar); color:var(--text);
  font:14px/1.5 "IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  font-variant-numeric:tabular-nums; }
.banner { height:28px; line-height:28px; text-align:center; font-size:12.5px; font-weight:500;
  color:var(--warn-fg); background:var(--warn-bg); border-bottom:1px solid var(--border); }
main { max-width:560px; margin:32px auto 48px; padding:0 16px; }
.panel { background:var(--bg); border:1px solid var(--border); border-radius:8px; }
.brand { display:flex; align-items:center; gap:10px; padding:16px 20px;
  border-bottom:1px solid var(--border); }
.mono-mark { display:inline-grid; place-items:center; width:28px; height:28px; border-radius:6px;
  background:var(--accent); color:#fff; font-weight:600; }
.brand b { display:block; font-weight:600; }
.brand small { display:block; color:var(--warn-fg); font-size:12px; }
.section { padding:16px 20px; border-bottom:1px solid var(--border); }
.section:last-child { border-bottom:0; }
h1 { font-size:20px; font-weight:600; margin:0 0 4px; }
h2 { font-size:13px; font-weight:600; margin:0 0 8px; }
p { margin:0; }
.muted { color:var(--text-2); }
.amount { font-size:28px; font-weight:600; line-height:1.2; margin:12px 0 2px; }
.amount span { font-size:16px; font-weight:400; color:var(--text-2); }
.badge { display:inline-flex; align-items:center; height:20px; padding:0 7px; border-radius:4px;
  font-size:12px; font-weight:500; vertical-align:1px; }
.held { color:var(--warn-fg); background:var(--warn-bg); }
.captured { color:var(--ok-fg); background:var(--ok-bg); }
.voided { color:var(--bad-fg); background:var(--bad-bg); }
dl.kv { display:grid; grid-template-columns:120px minmax(0,1fr); gap:6px 12px; margin:12px 0 0; }
dl.kv dt { color:var(--text-2); }
dl.kv dd { margin:0; overflow-wrap:anywhere; }
code, .mono { font:12.5px/1.5 "IBM Plex Mono",ui-monospace,Consolas,monospace; }
table { width:100%; border-collapse:collapse; margin-top:12px; }
td, th { text-align:left; padding:6px 0; border-bottom:1px solid var(--border);
  vertical-align:top; }
th { color:var(--text-2); font-weight:500; font-size:12.5px; }
tr:last-child td { border-bottom:0; }
.num { text-align:right; font-variant-numeric:tabular-nums; }
.page + .page { margin-top:14px; }
.url { font:12.5px/1.5 "IBM Plex Mono",ui-monospace,Consolas,monospace; color:var(--text-2);
  overflow-wrap:anywhere; }
pre.quote { white-space:pre-wrap; overflow-wrap:anywhere; margin:6px 0 8px; padding:10px 12px;
  background:var(--surface-2); border-radius:6px; max-height:280px; overflow:auto;
  font:12.5px/1.55 "IBM Plex Mono",ui-monospace,Consolas,monospace; }
mark { background:var(--mark); color:var(--text); padding:0 1px; }
.why { margin-bottom:6px; }
ul.reasons { margin:0; padding-left:18px; }
ul.reasons li + li { margin-top:4px; }
.actions { display:flex; gap:8px; justify-content:flex-end; flex-wrap:wrap; margin-top:12px; }
button { font:inherit; font-weight:500; height:36px; padding:0 16px; border-radius:6px;
  cursor:pointer; border:1px solid var(--border-2); background:var(--bg); color:var(--text); }
button:hover { background:var(--surface-2); }
button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
button.primary:hover { background:var(--accent-hover); }
button:disabled { opacity:.5; cursor:default; }
button:focus-visible, a:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
#result { margin-top:10px; font-weight:500; }
@media (max-width:520px) {
  main { margin-top:16px; }
  .actions button { flex:1 1 100%; }
  dl.kv { grid-template-columns:96px minmax(0,1fr); }
}
"""


def _mandate_block(mandate: Any) -> str:  # Any: mantis.paypal.models.Mandate
    if mandate is None:
        return '<p class="muted">Mandate not available in this process.</p>'
    scope = mandate.scope
    expires = "no expiry"
    if scope.ttl_seconds:
        expires = (mandate.issued_ts + timedelta(seconds=scope.ttl_seconds)).strftime(
            "%Y-%m-%d %H:%M UTC")
    rows = [
        ("You asked for", mandate.description or "-"),
        ("Categories", ", ".join(scope.categories) or "any"),
        ("Merchants", ", ".join(scope.allowed_merchants) or "any"),
        ("Per order", f"${scope.max_amount:.2f}" if scope.max_amount else "no limit"),
        ("In total", f"${mandate.budget_total:.2f}" if mandate.budget_total else "no limit"),
        ("Valid until", expires),
        ("Mandate", f"{mandate.mandate_id} &middot; <code>{_e(mandate.digest()[:16])}</code>"),
    ]
    body = "".join(f"<dt>{_e(k)}</dt><dd>{v if k == 'Mandate' else _e(v)}</dd>"
                   for k, v in rows)
    return f'<dl class="kv" style="margin-top:0">{body}</dl>'


def _evidence_block(record: dict[str, Any]) -> str:
    flagged = [p for p in record.get("evidence", [])
               if (p.get("judge") or {}).get("injected") or p.get("l3_fired")]
    if not flagged:
        pages = record.get("evidence", [])
        return f'<p class="muted">{len(pages)} page(s) read; none flagged by a text layer.</p>'
    out = []
    for page in flagged:
        judge = page.get("judge") or {}
        span = judge.get("quoted_span", "") if judge.get("injected") else ""
        l3 = ("abstained (vocabulary coverage "
              f"{page.get('coverage', 0):.2f})" if page.get("l3_abstained")
              else f"P(injected) {page.get('l3', 0):.2f}")
        why = ""
        if judge.get("injected"):
            target = judge.get("target_action")
            why = ('<p class="why"><b>The judge&rsquo;s reason:</b> '
                   + (f"it wants the agent to {_e(target)}. " if target else "")
                   + _e(judge.get("rationale", "")) + "</p>")
        out.append(
            f'<div class="page"><div class="url">{_e(page["url"])}</div>'
            f'<pre class="quote">{highlight(page.get("text", ""), span)}</pre>'
            f'{why}<p class="muted">Text classifier: {_e(l3)} &middot; judge: '
            f'{"injection" if judge.get("injected") else "no opinion"}</p></div>'
        )
    return "".join(out)


def render_consent_page(record: dict[str, Any], mandate: Any) -> str:  # Any: Mandate | None
    """The full HTML page for one gate record."""
    order_id = record["order_id"]
    status = record["status"]
    held = status == "held"
    merchant = _MERCHANTS.get(record["merchant_id"], record["merchant_id"])
    items = "".join(
        f"<tr><td>{_e(i['name'])} <span class='muted mono'>{_e(i['sku'])}</span></td>"
        f"<td class='num'>{_e(i['quantity'])}</td><td class='num'>${_e(i['unit_amount'])}</td></tr>"
        for i in record.get("items", [])
    )
    basket = (f"<table><tr><th>Item</th><th class='num'>Qty</th><th class='num'>Each</th></tr>"
              f"{items}</table>") if items else ""
    reasons = "".join(
        f"<li><b>{_e(r['layer'])}</b> <code>{_e(r['code'])}</code> &mdash; {_e(r['detail'])}</li>"
        for r in record.get("all_reasons") or record.get("reasons", [])
        # A layer reporting its own state ("all 9 clauses passed") is not a reason.
        if r.get("code") not in _STATUS_CODES
    ) or '<li class="muted">No layer reported a reason.</li>'
    payload = json.dumps(order_id)
    disabled = "" if held else " disabled"
    label = {"held": "Held", "captured": "Captured", "voided": "Voided"}.get(status, status.title())
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Approve this payment? &middot; MANTIS</title><style>{_STYLE}</style></head>
<body><div class="banner">Test data &middot; PayPal sandbox &middot; no real money moves</div>
<main><div class="panel">
<div class="brand"><span class="mono-mark" aria-hidden="true">M</span>
<span><b>MANTIS</b><small>Sandbox</small></span></div>

<div class="section">
<h1>Approve this payment?</h1>
<p class="muted">Your shopping agent asked PayPal to authorize this order. MANTIS held it for
you: the funds are reserved and nothing has moved.</p>
<div class="amount">${_e(record['amount'])} <span>{_e(record['currency'])}</span></div>
<div>{_e(merchant)} &middot; <span class="badge {_e(status)}" id="status">{_e(label)}</span></div>
<dl class="kv">
<dt>PayPal order</dt><dd class="mono">{_e(order_id)}</dd>
<dt>Authorization</dt><dd class="mono">{_e(record['authorization_id'])}</dd>
<dt>Paid by</dt><dd>{_e(record.get('payment_source'))}</dd></dl>
{basket}</div>

<div class="section"><h2>What you allowed</h2>{_mandate_block(mandate)}</div>

<div class="section"><h2>What the agent read</h2>{_evidence_block(record)}</div>

<div class="section"><h2>Why MANTIS held it</h2>
<ul class="reasons">{reasons}</ul>
<p class="muted" style="margin-top:8px">Decision {_e(record['decision'])} &middot; layers
{_e(' + '.join(record.get('layers_ran', [])))} &middot; scoring {_e(record.get('latency_ms'))} ms
&middot; judge {_e(record.get('judge_ms'))} ms</p></div>

<div class="section">
<p class="muted">Approve captures the payment. Deny voids the authorization and releases
the funds.</p>
<div class="actions">
<button id="deny"{disabled}>Deny and void</button>
<button class="primary" id="approve"{disabled}>Approve and capture</button></div>
<div id="result" role="status"></div></div>
</div>
</main>
<script>
const orderId = {payload};
const LABEL = {{held: "Held", captured: "Captured", voided: "Voided"}};
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
    const st = document.getElementById("status");
    st.textContent = LABEL[body.status] || body.status;
    st.className = "badge " + body.status;
    out.textContent = body.status === "captured"
      ? "Captured. PayPal capture " + body.capture_id
      : "Voided. The authorization is released.";
  }} catch (e) {{ out.textContent = "Failed: " + e.message; }}
}}
document.getElementById("approve").onclick = () => decide(true);
document.getElementById("deny").onclick = () => decide(false);
</script></body></html>"""
