/**
 * The gate log, as the Ops dashboard holds it.
 *
 * Rows arrive from GET /api/gate/log (polled every 2 s; re-rendered only when the
 * server's `version` moves). AG Studio gets a flattened copy as its data source and
 * owns filtering; the two MANTIS custom widgets read the full rows from here,
 * keyed by order id, and share the selected order through this store.
 */

import { useSyncExternalStore } from "react";

export type Reason = { layer: string; code: string; detail: string };

export type Evidence = {
  url: string;
  coverage: number | null;
  l3: number | null;
  l3_abstained: boolean | null;
  l3_fired: boolean | null;
  judge: {
    injected: boolean | null;
    verified: boolean | null;
    quoted_span: string | null;
    target_action: string | null;
    rationale: string | null;
  } | null;
  visitor_authored: boolean;
  /** null for visitor-written pages: shown only behind "show visitor text". */
  text: string | null;
};

export type LogRow = {
  order_id: string;
  ts: string;
  source: string;
  merchant_id: string;
  total: number;
  currency: string;
  n_items: number;
  items: { name: string; sku: string; quantity: number; unit_amount: string }[];
  decision: "approve" | "review" | "challenge" | "decline" | string;
  paypal_action: string;
  final_state: "captured" | "voided" | "held" | "authorized" | string;
  authorization_id: string | null;
  capture_id: string | null;
  layers_fired: string[];
  layers_ran: string[];
  risk: number;
  judge_quote: string;
  reasons: Reason[];
  receipt_hash: string;
  mandate_id: string;
  mandate: { description: string; can: string[]; cannot: string[]; hash: string } | null;
  protected_usd: number;
  captured_usd: number;
  held_usd: number;
  latency: { agent_ms: number; judge_ms: number; score_ms: number; paypal_ms: number };
  consent: { approve: boolean; signer: string; at: string } | null;
  evidence: Evidence[];
  scenario?: string;
};

export type Verify = {
  order_id: string;
  match: boolean;
  receipt_hash: string;
  custom_id: string | null;
  source: string;
  detail: string;
};

type State = {
  rows: LogRow[];
  byId: Map<string, LogRow>;
  version: number;
  selected: string | null;
  verify: Record<string, Verify | "pending" | "error">;
};

let state: State = { rows: [], byId: new Map(), version: -1, selected: null, verify: {} };
const listeners = new Set<() => void>();

function set(patch: Partial<State>) {
  state = { ...state, ...patch };
  listeners.forEach((l) => l());
}

export const opsStore = {
  get: () => state,
  subscribe(listener: () => void) {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
  setRows(rows: LogRow[], version: number) {
    const byId = new Map(rows.map((r) => [r.order_id, r]));
    const selected = state.selected && byId.has(state.selected) ? state.selected : rows[0]?.order_id ?? null;
    set({ rows, byId, version, selected });
  },
  select(orderId: string | null) {
    if (orderId !== state.selected) set({ selected: orderId });
  },
  async verify(orderId: string) {
    set({ verify: { ...state.verify, [orderId]: "pending" } });
    try {
      const r = await fetch(`${apiBase()}/receipts/${encodeURIComponent(orderId)}/verify`);
      const body = r.ok ? ((await r.json()) as Verify) : "error";
      set({ verify: { ...state.verify, [orderId]: body } });
    } catch {
      set({ verify: { ...state.verify, [orderId]: "error" } });
    }
  },
};

export function useOps<T>(pick: (s: State) => T): T {
  return useSyncExternalStore(opsStore.subscribe, () => pick(state));
}

/** Same-origin API in production; VITE_API_BASE for a split dev setup. */
export function apiBase(): string {
  return (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api";
}

export const money = (v: number, currency = "USD") =>
  v.toLocaleString("en-US", { style: "currency", currency, maximumFractionDigits: 2 });

export const MERCHANTS: Record<string, string> = {
  "m-freshmart": "FreshMart",
  "m-greenleaf": "GreenLeaf Market",
  "m-gifthub": "GiftHub",
  "m-giftly": "Giftly",
};
