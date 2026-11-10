"""The gate log: every decision the gate makes, as rows an ops dashboard can show.

    GET /api/gate/log?limit=200&offset=0      newest first, with a ``version`` to poll on
    GET /api/gate/log/{order_id}              one row in full (provenance, mandate, receipt)

Where rows come from
--------------------
* **seed** — the recorded Day 2 runs (D/E/F/G) against the real PayPal sandbox,
  committed in ``data/fixtures/gate_log_seed.jsonl``, so the dashboard is never empty
  on a clean clone or a fresh deploy.
* **api / playground / mcp / demo** — appended live. :class:`GateLog` subscribes to
  :attr:`PaymentGate.listeners`, so every decision and every consent lands here.

Rows are appended to ``<runtime dir>/gate_log.jsonl`` (Render's persistent disk).
The reader **tails that file**, so a gate in another process on the same disk — the
MCP proxy, the CLI demo — shows up in the dashboard too. A later line for the same
order supersedes an earlier one (a consent turns ``held`` into ``captured``).

Visitor-authored text
---------------------
Playground pages are written by anonymous visitors. In the shared log their full
text is **withheld**; the row carries the judge's short quote instead, and the full
text is served separately, links stripped, for the "show visitor text" toggle.
The browser escapes everything it renders either way.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from mantis.core.paths import DATA_DIR
from mantis.core.quota import runtime_dir

__all__ = ["LOG", "SEED_PATH", "GateLog", "row_from_record", "strip_links"]

SEED_PATH: Final[Path] = DATA_DIR / "fixtures" / "gate_log_seed.jsonl"

#: Reason codes that report a layer's state rather than a layer firing.
_STATUS_CODES: Final[frozenset[str]] = frozenset(
    {"protocol_ok", "not_loaded", "provenance_clean"})

_LINK: Final[re.Pattern[str]] = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)


def strip_links(text: str) -> str:
    """Visitor text with every URL replaced, so the dashboard links nowhere."""
    return _LINK.sub("[link removed]", text)


def _fired(record: dict[str, Any]) -> list[str]:
    """Layer chips that fired: L0, gate, budget, L3, judge, unvetted, L1, taint."""
    chips: list[str] = []

    def add(chip: str) -> None:
        if chip not in chips:
            chips.append(chip)

    for reason in record.get("all_reasons") or record.get("reasons") or []:
        code, layer = reason.get("code", ""), reason.get("layer", "")
        if code in _STATUS_CODES:
            continue
        if layer == "L0":
            add("L0")
        elif layer == "gate":
            add("budget" if code == "budget_exceeded" else "gate")
        elif layer == "L3":
            add("L3")
        elif layer == "L3-judge":
            add("unvetted" if code == "unvetted_text" else "judge")
        elif layer == "L1" and record.get("decision") != "approve":
            add("L1")
    protocol = any(c in chips for c in ("L0", "gate", "budget"))
    text = any(c in chips for c in ("L3", "judge", "unvetted"))
    if text and not protocol and record.get("decision") in ("review", "decline"):
        # Session taint: the order itself is inside the mandate, but the agent read
        # an injected page on the way to it. Held so a human looks at the basket.
        add("taint")
    return chips


def _risk(record: dict[str, Any], chips: list[str]) -> float:
    """0..1 for the risk bar: protocol breach 1.0, text layers agreeing 0.9, one
    text layer 0.6; an approval shows the worst in-domain L3 page, scaled down."""
    if any(c in chips for c in ("L0", "gate", "budget")):
        return 1.0
    if record.get("decision") == "decline":
        return 0.9
    if record.get("decision") in ("review", "challenge"):
        return 0.6
    l3 = [p.get("l3") for p in record.get("evidence") or [] if p.get("l3") is not None]
    return round(min(0.3, 0.3 * max(l3, default=0.0)) + 0.02, 3)


def row_from_record(record: dict[str, Any], mandate: dict[str, Any] | None = None,
                    *, source: str | None = None) -> dict[str, Any]:
    """One dashboard row from a :class:`~mantis.paypal.gate.GateRecord` dict."""
    chips = _fired(record)
    amount = float(record.get("amount") or 0.0)
    status = record.get("status", "")
    quote = ""
    evidence = []
    for page in record.get("evidence") or []:
        judge = page.get("judge") or {}
        if judge.get("injected") and judge.get("verified") and not quote:
            quote = str(judge.get("quoted_span") or "")
        visitor = bool(page.get("visitor_authored"))
        evidence.append({
            "url": page.get("url"),
            "coverage": page.get("coverage"),
            "l3": page.get("l3"),
            "l3_abstained": page.get("l3_abstained"),
            "l3_fired": page.get("l3_fired"),
            "judge": {k: judge.get(k) for k in ("injected", "verified", "quoted_span",
                                                "target_action", "rationale", "latency_ms")}
            if judge else None,
            "visitor_authored": visitor,
            # Withheld for visitor pages: the shared grid shows the judge's quote.
            "text": None if visitor else page.get("text"),
        })
    history = record.get("history") or []
    ts = record.get("decided_at") or (history[0].get("at") if history else "") or _now()
    items = record.get("items") or []
    return {
        "order_id": record["order_id"],
        "ts": ts,
        "source": source or record.get("source") or "api",
        "agent": "agent-shopper-01" if record.get("agent_ms") is not None else "",
        "merchant_id": record.get("merchant_id", ""),
        "total": amount,
        "currency": record.get("currency", "USD"),
        "n_items": sum(int(i.get("quantity", 1)) for i in items),
        "items": items,
        "decision": record.get("decision", ""),
        "paypal_action": record.get("paypal_action", ""),
        "final_state": status,
        "authorization_id": record.get("authorization_id"),
        "capture_id": record.get("capture_id"),
        "layers_fired": chips,
        "layers_ran": record.get("layers_ran") or [],
        "risk": _risk(record, chips),
        "judge_quote": quote,
        "reasons": record.get("reasons") or [],
        "receipt_hash": record.get("receipt_hash", ""),
        "mandate_id": record.get("mandate_id", ""),
        "mandate": mandate,
        "protected_usd": amount if status == "voided" else 0.0,
        "captured_usd": amount if status == "captured" else 0.0,
        "held_usd": amount if status == "held" else 0.0,
        "latency": {
            "agent_ms": round(float(record.get("agent_ms") or 0.0), 1),
            "judge_ms": round(float(record.get("judge_ms") or 0.0), 1),
            "score_ms": round(float(record.get("latency_ms") or 0.0), 2),
            "paypal_ms": round(float(record.get("paypal_ms") or 0.0), 1),
        },
        "consent": record.get("consent"),
        "evidence": evidence,
    }


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class GateLog:
    """Seed rows plus an append-only JSONL, folded by order id, tailed on read."""

    def __init__(self, path: Path | None = None, seed: Path | None = SEED_PATH) -> None:
        self.path = path or runtime_dir() / "gate_log.jsonl"
        self.seed = seed
        self._rows: dict[str, dict[str, Any]] = {}
        #: Visitor-authored page text by order id, never sent with the shared rows.
        self._visitor_text: dict[str, list[dict[str, str]]] = {}
        self._offset = 0
        self._loaded_seed = False
        self.version = 0
        self._lock = threading.Lock()

    # -- writing ---------------------------------------------------------------- #

    def record(self, record: Any, mandate: dict[str, Any] | None = None) -> None:  # Any: GateRecord
        """Listener for :attr:`PaymentGate.listeners`. Never raises into the gate."""
        raw = record.as_dict() if hasattr(record, "as_dict") else dict(record)
        visitor = [{"url": p.get("url", ""), "text": strip_links(p.get("text") or "")}
                   for p in raw.get("evidence") or [] if p.get("visitor_authored")]
        row = row_from_record(raw, mandate)
        line = {"row": row, "visitor_text": visitor} if visitor else {"row": row}
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(line, ensure_ascii=False) + "\n")
        self.refresh()

    # -- reading ---------------------------------------------------------------- #

    def _ingest(self, raw: dict[str, Any]) -> None:
        row = raw.get("row", raw)
        order_id = row.get("order_id")
        if not order_id:
            return
        previous = self._rows.get(order_id)
        if previous is not None and not row.get("mandate"):
            row["mandate"] = previous.get("mandate")
        self._rows[order_id] = row
        if raw.get("visitor_text"):
            self._visitor_text[order_id] = raw["visitor_text"]
        self.version += 1

    def refresh(self) -> None:
        """Load the seed once, then read whatever the JSONL gained since last time."""
        with self._lock:
            if not self._loaded_seed:
                self._loaded_seed = True
                if self.seed is not None and self.seed.exists():
                    for line in self.seed.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            self._ingest(json.loads(line))
            if not self.path.exists():
                return
            size = self.path.stat().st_size
            if size < self._offset:  # truncated or replaced: start over
                self._offset = 0
            if size == self._offset:
                return
            with self.path.open("r", encoding="utf-8") as handle:
                handle.seek(self._offset)
                chunk = handle.read()
                self._offset = handle.tell()
            for line in chunk.splitlines():
                if line.strip():
                    try:
                        self._ingest(json.loads(line))
                    except ValueError:
                        continue

    def page(self, *, limit: int = 200, offset: int = 0) -> dict[str, Any]:
        self.refresh()
        rows = sorted(self._rows.values(), key=lambda r: r.get("ts") or "", reverse=True)
        return {"version": self.version, "total": len(rows),
                "rows": rows[offset: offset + limit]}

    def get(self, order_id: str) -> dict[str, Any] | None:
        self.refresh()
        return self._rows.get(order_id)

    def visitor_text(self, order_id: str) -> list[dict[str, str]]:
        self.refresh()
        return list(self._visitor_text.get(order_id, []))


#: The process-wide log the API serves and the gates write to.
LOG: Final[GateLog] = GateLog()
