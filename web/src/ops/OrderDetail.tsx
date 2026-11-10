/**
 * One order, explained: provenance with the injected span highlighted, the
 * mandate in plain words, and the receipt with a Verify button.
 *
 * Used twice: as the full-width detail row under an expanded feed row, and as the
 * "Order detail" AG Studio custom widget that follows the selected order.
 *
 * Safety: page text is attacker-controlled. It is rendered as React text nodes
 * (escaped by construction) and the highlight is made by splitting the string,
 * never by injecting HTML. Visitor-written playground pages are not in the shared
 * row at all; "show visitor text" fetches them on demand, links already stripped.
 */

import { useState } from "react";
import type { Evidence, LogRow } from "./store";
import { MERCHANTS, apiBase, money, opsStore, useOps } from "./store";

const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** `text` with the first whitespace- and case-tolerant match of `span` marked. */
function Highlighted({ text, span }: { text: string; span: string | null }) {
  const words = (span ?? "").split(/\s+/).filter(Boolean).map(escapeRe);
  const match = words.length ? new RegExp(words.join("\\s+"), "i").exec(text) : null;
  if (!match) return <>{text}</>;
  const [a, b] = [match.index, match.index + match[0].length];
  return (
    <>
      {text.slice(0, a)}
      <mark className="ops-mark">{text.slice(a, b)}</mark>
      {text.slice(b)}
    </>
  );
}

function Page({ page, orderId }: { page: Evidence; orderId: string }) {
  const [visitor, setVisitor] = useState<string | null>(null);
  const flagged = Boolean(page.judge?.injected) || Boolean(page.l3_fired);
  const span = page.judge?.injected ? page.judge.quoted_span : null;

  async function showVisitorText() {
    try {
      const r = await fetch(`${apiBase()}/gate/log/${encodeURIComponent(orderId)}/visitor-text`);
      const body = (await r.json()) as { pages: { url: string; text: string }[] };
      setVisitor(body.pages.find((p) => p.url === page.url)?.text ?? "(not available)");
    } catch {
      setVisitor("(not available)");
    }
  }

  const text = page.visitor_authored ? visitor : page.text;
  return (
    <div className={`ops-page ${flagged ? "flagged" : ""}`}>
      <div className="ops-page-head">
        <code>{page.url}</code>
        <span className="ops-page-meta">
          {page.l3_abstained
            ? `L3 abstained (coverage ${(page.coverage ?? 0).toFixed(2)})`
            : page.l3 != null
              ? `L3 ${page.l3.toFixed(2)}`
              : ""}
          {page.judge ? ` · judge: ${page.judge.injected ? "INJECTION" : page.judge.injected === false ? "clean" : "no opinion"}` : ""}
          {page.visitor_authored ? " · visitor-written" : ""}
        </span>
      </div>
      {page.judge?.injected && page.judge.target_action && (
        <div className="ops-page-why">
          wants the agent to <b>{page.judge.target_action}</b>
          {page.judge.rationale ? ` — ${page.judge.rationale}` : ""}
        </div>
      )}
      {page.visitor_authored && text == null ? (
        <div className="ops-page-text">
          {span && <p className="ops-quote">“{span}”</p>}
          <button type="button" className="ops-link" onClick={showVisitorText}>
            show visitor text
          </button>
        </div>
      ) : (
        text && (
          <pre className="ops-page-text">
            <Highlighted text={text} span={span} />
          </pre>
        )
      )}
    </div>
  );
}

function VerifyBox({ row }: { row: LogRow }) {
  const result = useOps((s) => s.verify[row.order_id]);
  return (
    <div className="ops-receipt">
      <div>
        receipt <code title={row.receipt_hash}>{row.receipt_hash.slice(0, 20)}…</code>
      </div>
      <button type="button" className="ops-btn" onClick={() => opsStore.verify(row.order_id)}
              disabled={result === "pending"}>
        {result === "pending" ? "Verifying…" : "Verify against PayPal"}
      </button>
      {result && result !== "pending" && (
        result === "error" ? (
          <div className="ops-verify bad">Could not verify (no receipt on this server).</div>
        ) : (
          <div className={`ops-verify ${result.match ? "ok" : "bad"}`}>
            {result.match ? "✔ MATCH" : "✘ MISMATCH"} — custom_id{" "}
            <code>{(result.custom_id ?? "none").slice(0, 16)}…</code> ({result.source}).{" "}
            {result.detail}
          </div>
        )
      )}
    </div>
  );
}

export function OrderDetailView({ row }: { row: LogRow }) {
  return (
    <div className="ops-detail">
      <section>
        <h4>What the agent read</h4>
        {row.evidence.length === 0 && <p className="ops-faint">No pages recorded.</p>}
        {row.evidence.map((page) => (
          <Page key={page.url} page={page} orderId={row.order_id} />
        ))}
      </section>
      <section>
        <h4>What the human signed</h4>
        {row.mandate ? (
          <>
            <p className="ops-faint">“{row.mandate.description}”</p>
            <ul className="ops-can">
              {row.mandate.can.map((c) => <li key={c}>{c}</li>)}
            </ul>
            <ul className="ops-cannot">
              {row.mandate.cannot.map((c) => <li key={c}>{c}</li>)}
            </ul>
          </>
        ) : (
          <p className="ops-faint">Mandate {row.mandate_id}</p>
        )}
        <h4>Why</h4>
        <ul className="ops-reasons">
          {row.reasons.map((r) => (
            <li key={r.code + r.detail}>
              <span className={`ops-chip ops-chip-${r.layer === "L3-judge" ? "judge" : r.layer}`}>{r.layer}</span>{" "}
              <code>{r.code}</code> {r.detail}
            </li>
          ))}
        </ul>
        <h4>The money</h4>
        <p>
          {money(row.total, row.currency)} at {MERCHANTS[row.merchant_id] ?? row.merchant_id} ·{" "}
          {row.decision.toUpperCase()} → {row.paypal_action} → <b>{row.final_state}</b>
          {row.consent ? ` (human ${row.consent.approve ? "approved" : "denied"})` : ""}
        </p>
        <VerifyBox row={row} />
      </section>
    </div>
  );
}

/** The AG Studio custom widget: follows the order selected in the feed. */
export function OrderDetailWidget() {
  const row = useOps((s) => (s.selected ? s.byId.get(s.selected) ?? null : null));
  if (!row) return <div className="ops-empty">Select an order in the feed.</div>;
  return (
    <div className="ops-widget-scroll">
      <div className="ops-detail-title">
        <span className={`ops-pill ops-pill-${row.decision}`}>{row.decision.toUpperCase()}</span>
        <code>{row.order_id}</code>
        {row.judge_quote && <span className="ops-quote">“{row.judge_quote}”</span>}
      </div>
      <OrderDetailView row={row} />
    </div>
  );
}
