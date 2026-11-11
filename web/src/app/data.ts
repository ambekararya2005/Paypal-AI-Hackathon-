/**
 * Data for the app pages. Everything shown comes from the API; when the API is
 * absent or an order is missing, pages show an honest empty state.
 *
 * The gate log is one shared, polled store: every page that needs it subscribes,
 * and it polls only while something is subscribed.
 */

import { useEffect, useState, useSyncExternalStore } from "react";
import { MERCHANTS, apiBase, type LogRow } from "../ops/store";

export type { LogRow };

export type Loaded<T> =
  | { state: "loading" }
  | { state: "ok"; data: T }
  | { state: "error"; status: number | null };

/** GET `${apiBase()}${path}`; re-polls every `everyMs` when given. */
export function useApi<T>(path: string | null, everyMs?: number): Loaded<T> {
  const [out, setOut] = useState<Loaded<T>>({ state: "loading" });
  useEffect(() => {
    if (path == null) return;
    setOut({ state: "loading" });
    let alive = true;
    let timer: number | undefined;
    const load = () => {
      fetch(`${apiBase()}${path}`)
        .then(async (r) => {
          if (!r.ok) throw r.status;
          const data = (await r.json()) as T;
          if (alive) setOut({ state: "ok", data });
        })
        .catch((e: unknown) => {
          if (alive) setOut((prev) => (prev.state === "ok" ? prev : { state: "error", status: typeof e === "number" ? e : null }));
        })
        .finally(() => {
          if (alive && everyMs) timer = window.setTimeout(load, everyMs);
        });
    };
    load();
    return () => {
      alive = false;
      if (timer) window.clearTimeout(timer);
    };
  }, [path, everyMs]);
  return out;
}

// --------------------------------------------------------------- gate log --

type LogState = { status: "loading" | "ok" | "error"; rows: LogRow[]; total: number; version: number };
let logState: LogState = { status: "loading", rows: [], total: 0, version: -1 };
const logListeners = new Set<() => void>();
let logTimer: number | undefined;

async function pollLog() {
  try {
    const r = await fetch(`${apiBase()}/gate/log?limit=500`);
    if (!r.ok) throw new Error(String(r.status));
    const body = (await r.json()) as { rows: LogRow[]; version: number; total?: number };
    if (body.version !== logState.version || logState.status !== "ok") {
      logState = { status: "ok", rows: body.rows, total: body.total ?? body.rows.length, version: body.version };
      logListeners.forEach((l) => l());
    }
  } catch {
    if (logState.status === "loading") {
      logState = { ...logState, status: "error" };
      logListeners.forEach((l) => l());
    }
  }
  if (logListeners.size) logTimer = window.setTimeout(pollLog, 4000);
}

function subscribeLog(listener: () => void) {
  logListeners.add(listener);
  if (logListeners.size === 1) void pollLog();
  return () => {
    logListeners.delete(listener);
    if (!logListeners.size && logTimer) window.clearTimeout(logTimer);
  };
}

export function useGateLog(): LogState {
  return useSyncExternalStore(subscribeLog, () => logState);
}

// ------------------------------------------------------------- limits etc --

export type Limits = {
  live_disabled: boolean;
  caps_enforced: boolean;
  remaining: Record<string, number>;
  caps: Record<string, number>;
};

export type PlaygroundConfig = {
  template: string;
  max_chars: number;
  tasks: { id: string; text: string }[];
  recorded: { id: string; title: string }[];
  live_available: boolean;
  live_unavailable_reason: string;
  per_ip_per_hour: number;
  per_ip_remaining?: number;
  page_preview?: { url: string; title: string; reviews: string[]; insert_at: number };
};

// ---------------------------------------------------------------- words --

export type Status = "Captured" | "Held" | "Voided" | "Authorized";

export function statusOf(finalState: string): Status {
  if (finalState === "captured") return "Captured";
  if (finalState === "held") return "Held";
  if (finalState === "voided") return "Voided";
  return "Authorized";
}

/** Layer ids from the gate log, as plain words. */
export const LAYER_WORDS: Record<string, string> = {
  L0: "Mandate rules",
  gate: "Receipt check",
  budget: "Budget",
  judge: "Injection judge",
  L3: "Text classifier",
  unvetted: "Unvetted page",
  L1: "Risk model",
  taint: "Session taint",
};

/** What each layer means, one sentence. */
export const LAYER_MEANING: Record<string, string> = {
  L0: "The order breaks a clause of the signed mandate: category, merchant, ceiling or expiry.",
  gate: "The order does not match its receipt or its provenance.",
  budget: "The mandate's total budget would be exceeded.",
  judge: "The LLM judge found an instruction aimed at the agent, and quoted it verbatim.",
  L3: "The text classifier scored a page the agent read as an injection.",
  unvetted: "A page the agent read could not be vetted, so the order is held (fail closed).",
  L1: "The supervised risk model escalated the decision.",
  taint: "The order is in scope, but the agent read an injected page earlier in the session.",
};

export const layerWords = (ids: string[]) => (ids.length ? ids.map((l) => LAYER_WORDS[l] ?? l).join(", ") : "—");

export const merchantName = (id: string) => MERCHANTS[id] ?? id;

/** "5YM2…621E". */
export const shortId = (id: string) => (id.length > 10 ? `${id.slice(0, 4)}…${id.slice(-4)}` : id);
export const shortHash = (h: string) => (h.length > 16 ? `${h.slice(0, 8)}…${h.slice(-6)}` : h);

export function utc(ts: string): string {
  const d = new Date(ts);
  return Number.isNaN(d.getTime()) ? ts : `${d.toISOString().slice(0, 19).replace("T", " ")} UTC`;
}

export function relTime(ts: string, now = Date.now()): string {
  const t = new Date(ts).getTime();
  if (Number.isNaN(t)) return "";
  const s = Math.max(0, Math.round((now - t) / 1000));
  if (s < 45) return "just now";
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h} h ago`;
  const d = Math.round(h / 24);
  return d === 1 ? "1 day ago" : `${d} days ago`;
}

export const usd = (v: number, currency = "USD") =>
  v.toLocaleString("en-US", { style: "currency", currency, minimumFractionDigits: 2, maximumFractionDigits: 2 });

export function seconds(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms.toFixed(ms < 10 ? 1 : 0)} ms`;
}

/** The server-rendered consent page for a held order (approve -> capture, deny -> void). */
export const consentHref = (id: string) => `${apiBase()}/gate/${encodeURIComponent(id)}/consent`;

export const GITHUB = "https://github.com/ambekararya2005/Paypal-AI-Hackathon-";

/** Safe localStorage: private windows and blocked storage just forget. */
export const store = {
  get(key: string): string | null {
    try {
      return window.localStorage.getItem(key);
    } catch {
      return null;
    }
  },
  set(key: string, value: string) {
    try {
      window.localStorage.setItem(key, value);
    } catch {
      /* storage unavailable: nothing to remember */
    }
  },
};
