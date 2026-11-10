/**
 * Cell renderers for the MANTIS order feed (AG Grid Community, inside an AG Studio
 * custom widget). Each one answers a question an operator asks at a glance:
 * how risky, what was decided, which layer caught it, what PayPal did, and how
 * much money did not move.
 */

import type { ICellRendererParams } from "ag-grid-community";
import type { LogRow } from "./store";
import { MERCHANTS, money } from "./store";

type P = ICellRendererParams<LogRow>;

/** Layer chip legend: label, tooltip. Order is the order they are shown in. */
export const LAYERS: [string, string, string][] = [
  ["L0", "L0", "Protocol clause: the order breaks the signed mandate"],
  ["budget", "budget", "Gate clause: the mandate's total budget is spent"],
  ["gate", "gate", "Gate clause: the order does not match its receipt"],
  ["judge", "judge", "LLM injection judge: an instruction aimed at the agent, quoted"],
  ["L3", "L3", "Text classifier: the page looks like an injection"],
  ["unvetted", "unvetted", "A page nobody could vouch for: held, fail closed"],
  ["L1", "L1", "Supervised model escalated the decision"],
  ["taint", "taint", "Session taint: in scope, but the agent read an injected page"],
];

export function RiskBar({ value }: P) {
  const risk = typeof value === "number" ? value : 0;
  const pct = Math.round(risk * 100);
  const tone = risk >= 0.85 ? "hi" : risk >= 0.5 ? "mid" : "lo";
  return (
    <div className="ops-risk" title={`risk ${pct}%`}>
      <div className={`ops-risk-fill ${tone}`} style={{ width: `${Math.max(4, pct)}%` }} />
      <span className="ops-risk-num">{pct}</span>
    </div>
  );
}

export function DecisionPill({ value }: P) {
  const v = String(value ?? "");
  return <span className={`ops-pill ops-pill-${v}`}>{v.toUpperCase()}</span>;
}

export function LayerChips({ data }: P) {
  if (!data) return null;
  const fired = new Set(data.layers_fired);
  if (fired.size === 0) {
    return <span className="ops-chip ops-chip-clear" title="every check passed">clear</span>;
  }
  return (
    <span className="ops-chips">
      {LAYERS.filter(([id]) => fired.has(id)).map(([id, label, tip]) => (
        <span key={id} className={`ops-chip ops-chip-${id}`} title={tip}>
          {label}
        </span>
      ))}
    </span>
  );
}

const STATE_LABEL: Record<string, string> = {
  captured: "CAPTURED",
  voided: "VOIDED",
  held: "HELD",
  authorized: "AUTHORIZED",
};

export function PayPalBadge({ data, context }: P & { context: { toggle: (id: string) => void } }) {
  if (!data) return null;
  const s = data.final_state;
  return (
    <button
      type="button"
      className={`ops-pp ops-pp-${s}`}
      onClick={() => context.toggle(data.order_id)}
      title={`PayPal sandbox order ${data.order_id} — click for provenance, mandate and receipt`}
    >
      <span className="ops-pp-dot" />
      {STATE_LABEL[s] ?? s.toUpperCase()}
      {/* last 6: the full id is in the tooltip and the detail view */}
      <span className="ops-pp-id">…{data.order_id.slice(-6)}</span>
    </button>
  );
}

export function Protected({ data }: P) {
  if (!data) return null;
  if (data.protected_usd > 0) {
    return <span className="ops-money ops-money-protected" title="blocked: never moved">
      ⛨ {money(data.protected_usd, data.currency)}
    </span>;
  }
  if (data.captured_usd > 0) {
    return <span className="ops-money ops-money-captured">{money(data.captured_usd, data.currency)}</span>;
  }
  return <span className="ops-money ops-money-held">{money(data.held_usd, data.currency)} held</span>;
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
