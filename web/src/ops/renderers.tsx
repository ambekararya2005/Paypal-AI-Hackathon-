/**
 * Cell renderers for the MANTIS order feed (AG Grid Community, inside an AG Studio
 * custom widget). Each one answers a question an operator asks at a glance:
 * how risky, what was decided, which layer caught it, what PayPal did, and how
 * much money did not move. Status uses the app's badges; layers are plain words;
 * the order id links to its order page.
 */

import type { ICellRendererParams } from "ag-grid-community";
import { LAYER_WORDS } from "../app/data";
import { orderHref } from "../app/router";
import { StatusBadge } from "../app/ui";
import type { LogRow } from "./store";
import { MERCHANTS, money } from "./store";

type P = ICellRendererParams<LogRow>;

/** Layer legend: id, label, tooltip. Order is the order they are shown in. */
export const LAYERS: [string, string, string][] = [
  ["L0", "Mandate rules", "Protocol clause: the order breaks the signed mandate"],
  ["budget", "Budget", "Gate clause: the mandate's total budget is spent"],
  ["gate", "Receipt check", "Gate clause: the order does not match its receipt"],
  ["judge", "Injection judge", "LLM injection judge: an instruction aimed at the agent, quoted"],
  ["L3", "Text classifier", "Text classifier: the page looks like an injection"],
  ["unvetted", "Unvetted page", "A page nobody could vouch for: held, fail closed"],
  ["L1", "Risk model", "Supervised model escalated the decision"],
  ["taint", "Session taint", "Session taint: in scope, but the agent read an injected page"],
];

/** A thin 2 px line: risk as a share of the cell. */
export function RiskBar({ value }: P) {
  const risk = typeof value === "number" ? value : 0;
  const pct = Math.round(risk * 100);
  return (
    <div className="ops-risk" title={`risk ${pct}%`}>
      <span className="ops-risk-num num">{pct}</span>
      <div className="ops-risk-track">
        <div className="ops-risk-fill" style={{ width: `${Math.max(2, pct)}%` }} />
      </div>
    </div>
  );
}

const DECISION: Record<string, string> = { approve: "Approve", review: "Review", challenge: "Challenge", decline: "Decline" };

/** The gate's decision, as a word; the status badge in the PayPal column carries colour. */
export function DecisionPill({ value }: P) {
  const v = String(value ?? "");
  return <span className="ops-decision">{DECISION[v] ?? v}</span>;
}

export function LayerChips({ data }: P) {
  if (!data) return null;
  const fired = new Set(data.layers_fired);
  if (fired.size === 0) return <span className="ops-faint" title="every check passed">—</span>;
  const shown = LAYERS.filter(([id]) => fired.has(id));
  return (
    <span className="ops-layers" title={shown.map(([, , tip]) => tip).join("\n")}>
      {shown.map(([id]) => LAYER_WORDS[id] ?? id).join(", ")}
    </span>
  );
}

/** Status badge (click to expand the row's detail) and the order id, linked to its page. */
export function PayPalBadge({ data, context }: P & { context: { toggle: (id: string) => void } }) {
  if (!data) return null;
  return (
    <span className="ops-pp">
      <button
        type="button"
        className="ops-pp-btn"
        onClick={() => context.toggle(data.order_id)}
        title="Expand provenance, mandate and receipt"
      >
        <StatusBadge state={data.final_state} />
      </button>
      <a className="ops-pp-id mono" href={orderHref(data.order_id)} title={`Open order ${data.order_id}`} onClick={(e) => e.stopPropagation()}>
        …{data.order_id.slice(-6)}
      </a>
    </span>
  );
}

export function Protected({ data }: P) {
  if (!data) return null;
  if (data.protected_usd > 0) {
    return (
      <span className="ops-money num" title="blocked before capture: never moved">
        {money(data.protected_usd, data.currency)}
      </span>
    );
  }
  if (data.captured_usd > 0) return <span className="ops-money num">{money(data.captured_usd, data.currency)}</span>;
  return <span className="ops-money num">{money(data.held_usd, data.currency)}</span>;
}

export function Merchant({ data }: P) {
  if (!data) return null;
  return (
    <span className="ops-merchant">
      {MERCHANTS[data.merchant_id] ?? data.merchant_id}
      <span className="ops-src">{data.source}</span>
    </span>
  );
}

/** A tiny stacked bar: agent think time, judge, MANTIS scoring, PayPal round trips. */
export function LatencySplit({ data }: P) {
  if (!data) return null;
  const { agent_ms, judge_ms, score_ms, paypal_ms } = data.latency;
  const total = agent_ms + judge_ms + score_ms + paypal_ms || 1;
  const parts: [string, number][] = [
    ["agent", agent_ms],
    ["judge", judge_ms],
    ["score", score_ms],
    ["paypal", paypal_ms],
  ];
  const tip = parts.map(([k, v]) => `${k} ${v >= 1000 ? (v / 1000).toFixed(1) + " s" : v.toFixed(0) + " ms"}`).join(" · ");
  return (
    <div className="ops-lat" title={tip}>
      {parts.map(([k, v]) => (
        <span key={k} className={`ops-lat-${k}`} style={{ width: `${(100 * v) / total}%` }} />
      ))}
    </div>
  );
}

export function JudgeQuote({ value }: P) {
  const q = String(value ?? "");
  if (!q) return <span className="ops-faint">—</span>;
  return <span className="ops-quote" title={q}>“{q}”</span>;
}
