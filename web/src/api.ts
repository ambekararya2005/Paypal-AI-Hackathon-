/**
 * The one place the console talks to the API.
 *
 * `VITE_API_BASE` is baked at build time. Empty in dev, where vite.config.ts
 * proxies `/api` to 127.0.0.1:8000 and CORS never comes up; set to the deployed
 * API's origin for the Vercel build. Nothing else in the app knows a hostname.
 */

const RAW_BASE = import.meta.env.VITE_API_BASE ?? "";
export const API_BASE = RAW_BASE ? RAW_BASE.replace(/\/$/, "") : "/api";

export type Decision = "approve" | "challenge" | "review" | "decline";

export interface LayerScore {
  score: number | null;
  percentile: number | null;
}

export interface Contribution {
  feature: string;
  value: unknown;
  contribution: number;
}

export interface ScoredEvent {
  event_id: string;
  ts: string;
  amount: number;
  currency: string;
  mcc: string | null;
  channel: string;
  entry_mode: string | null;
  customer_id: string | null;
  merchant_id: string | null;
  merchant_country: string | null;
  card_bin: string | null;
  device_id: string | null;
  txn_type: string | null;
  ag_agent_id: string | null;
  ag_agent_platform: string | null;
  ag_mandate_type: string | null;
  ag_human_present: boolean | null;
  ag_delegation_depth: number | null;
  layers: Record<string, LayerScore>;
  l0: { fired: boolean; reason: string | null };
  risk: number | null;
  decision: Decision;
  contributions: Contribution[];
}

export interface Truth {
  is_fraud: boolean;
  attack_id: string | null;
  attack_campaign: string | null;
}

/** One SSE `auth` frame: what the firewall knew, then what was actually true. */
export interface AuthFrame {
  seq: number;
  event: ScoredEvent;
  truth: Truth;
}

export interface StreamMeta {
  run_id: string;
  n_events: number;
  rate: number;
  operating_fpr: number | null;
  sampling_note: string;
  provenance_note: string;
  thresholds: Record<string, number>;
}

export interface StreamDone {
  n_events: number;
  decisions: Record<string, number>;
  caught: number;
  missed: number;
  false_positives: number;
}

export interface AtlasCard {
  id: string;
  name: string;
  family: string;
  status: string;
  rails: string[];
  detected_by: string[];
  has_injector: boolean;
  discovered_by: string;
}

export interface AtlasResponse {
  cards: AtlasCard[];
  families: { family: string; total: number; implemented: number }[];
  total: number;
  implemented: number;
  discovered: AtlasCard[];
}

export interface ResultsResponse {
  generated: string;
  tables: { title: string; header: string[]; rows: string[][] }[];
  layer_performance: Record<string, string>[];
  per_family: Record<string, string>[];
  per_attack: Record<string, string>[];
  decisions: Record<string, string>[];
  zero_day: { detector: string; recall: number }[];
  evasion_curve: number[];
  headline: Record<string, unknown>;
}

export interface ArenaResponse {
  operating_fpr: number;
  cards: string[];
  seconds: number;
  n_background: number;
  evasion_curve: number[];
  generations: {
    generation: number;
    n_variants: number;
    n_events: number;
    mean_evasion: number;
    max_evasion: number;
    per_card: Record<string, number>;
  }[];
  survivors: Record<string, unknown>[];
  zero_day: Record<string, number | string> | null;
}

/** One row of the fidelity scorecard's marginal table. */
export interface MarginalRow {
  feature: string;
  kind: "continuous" | "categorical";
  metric: "KS" | "JSD";
  distance: number;
  band: number;
  ratio: number;
  synthetic_median?: number;
  real_median?: number;
  max_level_delta?: number;
  max_level?: string;
}

export interface FidelityResponse {
  available: boolean;
  note: string;
  generated: string;
  calibration: { source?: string; note?: string };
  reference: {
    available?: boolean;
    provenance?: string;
    note?: string;
    levels?: Record<string, number>;
    level_ratios?: Record<string, number>;
  };
  synthetic: { levels?: Record<string, number>; note?: string; events_compared?: number };
  headline: Record<string, number | string | null>;
  marginals: {
    rows?: MarginalRow[];
    correlation?: {
      frobenius: number;
      rms_off_diagonal: number;
      n_features: number;
      worst_pairs: { pair: string; synthetic: number; real: number; delta: number }[];
    };
  };
  tstr: {
    trtr?: { auc_pr: number; roc_auc: number; baseline: number };
    tstr?: { auc_pr: number; roc_auc: number; baseline: number };
    trts?: { auc_pr: number; roc_auc: number; baseline: number };
    transfer_ratio?: number;
    caveat?: string;
    what_each_learned?: {
      trtr: { feature: string; gain_share: number }[];
      tstr: { feature: string; gain_share: number }[];
    };
  };
  discriminator: {
    auc?: number;
    target?: number;
    separability?: number;
    reading?: string;
    n_per_side?: number;
    per_feature?: { feature: string; alone_auc: number; gain_share: number }[];
  };
  discriminator_ablated: { auc?: number; separability?: number; excluded?: string[] };
  adjudications: {
    feature: string;
    third_quantity: string;
    synthetic: string;
    reference: string;
    verdict: string;
    note: string;
  }[];
  known_divergences: {
    name: string;
    measured: string;
    cause: string;
    why_not_fixed: string;
  }[];
  population: Record<string, unknown>;
}

export interface LatencyResponse {
  available: boolean;
  note: string;
  generated: string;
  n_events: number;
  warm_events: number;
  budget_ms: number;
  within_budget: boolean;
  headroom: number;
  end_to_end_ms: Record<string, number>;
  stages_ms: Record<string, Record<string, number>>;
}

/**
 * Where the console gets its data from.
 *
 * `live` means an API answered `/health` and everything is a real request
 * against a real backend. `static` means nothing answered and the console is
 * reading the frozen artefacts in `public/data/`, replaying the same committed
 * feed on a client-side timer.
 *
 * Both show identical numbers — the frozen files were produced by the API's own
 * route handlers — but they are not the same claim, so the console displays
 * which mode it is in rather than letting a viewer assume a backend exists.
 */
export type Mode = "live" | "static" | "unknown";

let mode: Mode = "unknown";
const listeners = new Set<(m: Mode) => void>();

export function onModeChange(fn: (m: Mode) => void): () => void {
  listeners.add(fn);
  fn(mode);
  return () => listeners.delete(fn);
}

function setMode(next: Mode): void {
  if (mode === next) return;
  mode = next;
  for (const fn of listeners) fn(next);
}

export const getMode = (): Mode => mode;

/** Probe the API once. Everything else keys off the answer. */
let probe: Promise<Mode> | null = null;

export function detectMode(): Promise<Mode> {
  if (probe) return probe;
  probe = (async () => {
    try {
      // A timeout, not just a rejected fetch: a sleeping free-tier dyno accepts
      // the connection and then thinks for thirty seconds, which without this
      // would leave the console blank for exactly as long as a judge is looking
      // at it. Two seconds and we fall back to files that are already loaded.
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 2000);
      const response = await fetch(`${API_BASE}/health`, { signal: controller.signal });
      clearTimeout(timer);
      setMode(response.ok ? "live" : "static");
    } catch {
      setMode("static");
    }
    return mode;
  })();
  return probe;
}

/** The frozen artefacts, served as plain files beside index.html. */
async function getStatic<T>(name: string): Promise<T> {
  const response = await fetch(`${import.meta.env.BASE_URL}data/${name}`);
  if (!response.ok) {
    throw new Error(
      `no API answered and the bundled ${name} is missing — ` +
        `run 'make static' before building the console`,
    );
  }
  return (await response.json()) as T;
}

async function getJSON<T>(path: string): Promise<T> {
  if ((await detectMode()) === "static") {
    return getStatic<T>(`${path.replace(/^\//, "")}.json`);
  }
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    // The API answers 503 with a message naming the make target that produces
    // the missing artefact. Surfacing that verbatim turns "the panel is blank"
    // into "run make firewall", which is the difference between a bug report
    // and a fix.
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* a non-JSON error body is not worth a second failure */
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export const fetchAtlas = () => getJSON<AtlasResponse>("/atlas");
export const fetchResults = () => getJSON<ResultsResponse>("/results");
export const fetchArena = () => getJSON<ArenaResponse>("/arena");
export const fetchFidelity = () => getJSON<FidelityResponse>("/fidelity");

/**
 * Latency, which is allowed to be absent.
 *
 * `make latency` is a five-minute fit and is not part of `make submission`, so a
 * clone can legitimately have no latency.json. Every other fetch in this file
 * throws on a missing artefact because a missing RESULTS.md means the console is
 * broken; here it means one panel says "not measured", which is a true statement
 * and the one this project would rather make than invent a number.
 */
export const fetchLatency = async (): Promise<LatencyResponse | null> => {
  try {
    return await getJSON<LatencyResponse>("/latency");
  } catch {
    return null;
  }
};

/**
 * One card in full, for the atlas drawer.
 *
 * `/atlas/{id}` is a route per card and a static host has no routes, so the
 * offline path reads one frozen map of all 42 instead — loaded once, cached, and
 * produced by the same route handler at build time.
 */
let frozenCards: Promise<Record<string, AtlasCardDetail>> | null = null;

export async function fetchAtlasCard(id: string): Promise<AtlasCardDetail> {
  if ((await detectMode()) === "static") {
    frozenCards ??= getStatic<Record<string, AtlasCardDetail>>("atlas_cards.json");
    const card = (await frozenCards)[id];
    if (!card) throw new Error(`no atlas card ${id} in the frozen set`);
    return card;
  }
  return getJSON<AtlasCardDetail>(`/atlas/${id}`);
}

export interface AtlasCardDetail extends AtlasCard {
  actor: string;
  genai_enabler: string;
  description: string;
  preconditions: string[];
  observable_signals: { signal: string; feature: string | null; layer: string }[];
  mitigations: string[];
  generator: string | null;
  references: string[];
}

export interface SimulateOptions {
  n_events?: number;
  rate?: number;
  family?: string | null;
  offset?: number;
}

export interface RunHandle {
  run_id: string;
  n_events: number;
  rate: number;
  note: string;
}

/** A started replay. `close()` stops it, in either mode. */
export interface StreamHandle {
  close: () => void;
}

export interface StreamHandlers {
  onMeta?: (meta: StreamMeta) => void;
  onAuth?: (frame: AuthFrame) => void;
  onDone?: (done: StreamDone) => void;
  onError?: (message: string) => void;
}

interface FrozenFeed {
  manifest: Record<string, unknown>;
  frames: AuthFrame[];
}

let frozenFeed: Promise<FrozenFeed> | null = null;
const loadFrozenFeed = (): Promise<FrozenFeed> =>
  (frozenFeed ??= getStatic<FrozenFeed>("feed.json"));

export async function startRun(options: SimulateOptions = {}): Promise<RunHandle> {
  const n = options.n_events ?? 200;
  const rate = options.rate ?? 6;

  if ((await detectMode()) === "static") {
    const feed = await loadFrozenFeed();
    return {
      run_id: `static-${options.offset ?? 0}-${n}`,
      n_events: Math.min(n, feed.frames.length),
      rate,
      note: String(feed.manifest.sampling_note ?? ""),
    };
  }

  const response = await fetch(`${API_BASE}/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      n_events: n,
      rate,
      family: options.family ?? null,
      offset: options.offset ?? 0,
    }),
  });
  if (!response.ok) throw new Error(`simulate failed: ${response.status}`);
  const body = await response.json();
  return {
    run_id: body.run_id,
    n_events: body.n_events,
    rate: body.rate,
    note: body.note,
  };
}

/**
 * Start streaming a run, over SSE when there is an API and off a timer when
 * there is not.
 *
 * The caller owns the returned handle, and React's effect cleanup closes it. A
 * stream left open when a component unmounts keeps ticking — over SSE that is a
 * live server coroutine, and off the timer it is a `setTimeout` chain writing
 * into a dead component. A judge clicking between tabs mid-demo does exactly
 * that, so both paths are cancellable by the same call.
 */
export function openStream(
  run: RunHandle,
  handlers: StreamHandlers,
  options: SimulateOptions = {},
): StreamHandle {
  if (getMode() === "static") {
    return replayFrozen(run, handlers, options);
  }

  const source = new EventSource(`${API_BASE}/stream/${run.run_id}`);
  source.addEventListener("meta", (e) =>
    handlers.onMeta?.(JSON.parse((e as MessageEvent).data)),
  );
  source.addEventListener("auth", (e) =>
    handlers.onAuth?.(JSON.parse((e as MessageEvent).data)),
  );
  source.addEventListener("done", (e) => {
    handlers.onDone?.(JSON.parse((e as MessageEvent).data));
    // The server has said its last word. Without this close the browser sees a
    // finished stream as a dropped one and reconnects, replaying the whole feed
    // from the top — which during a demo looks like the console glitching.
    source.close();
  });
  source.onerror = () => handlers.onError?.("stream interrupted — is the API still running?");
  return { close: () => source.close() };
}

/** The no-backend path: the same frames, paced by a timer instead of a server. */
function replayFrozen(
  run: RunHandle,
  handlers: StreamHandlers,
  options: SimulateOptions,
): StreamHandle {
  let timer: ReturnType<typeof setTimeout> | undefined;
  let stopped = false;

  void (async () => {
    let feed: FrozenFeed;
    try {
      feed = await loadFrozenFeed();
    } catch (e) {
      handlers.onError?.(e instanceof Error ? e.message : String(e));
      return;
    }
    if (stopped) return;

    const offset = options.offset ?? 0;
    const all = feed.frames;
    const chosen = Array.from({ length: run.n_events }, (_, i) => all[(offset + i) % all.length]);

    handlers.onMeta?.({
      run_id: run.run_id,
      n_events: chosen.length,
      rate: run.rate,
      operating_fpr: (feed.manifest.operating_fpr as number) ?? null,
      sampling_note: String(feed.manifest.sampling_note ?? ""),
      provenance_note: String(feed.manifest.provenance_note ?? ""),
      thresholds: (feed.manifest.thresholds as Record<string, number>) ?? {},
    });

    const tally: Record<string, number> = {};
    let caught = 0;
    let missed = 0;
    let falsePositives = 0;
    let i = 0;

    const tick = () => {
      if (stopped || i >= chosen.length) {
        if (!stopped) {
          handlers.onDone?.({
            n_events: chosen.length,
            decisions: tally,
            caught,
            missed,
            false_positives: falsePositives,
          });
        }
        return;
      }
      const frame = chosen[i];
      // Renumber: the console keys React rows on seq and expects it dense from
      // zero, and a wrapped offset would otherwise hand it a repeated index.
      const renumbered: AuthFrame = { ...frame, seq: i };
      const decision = frame.event.decision;
      tally[decision] = (tally[decision] ?? 0) + 1;
      const flagged = decision !== "approve";
      if (frame.truth.is_fraud) flagged ? caught++ : missed++;
      else if (flagged) falsePositives++;

      handlers.onAuth?.(renumbered);
      i++;
      timer = setTimeout(tick, 1000 / run.rate);
    };
    tick();
  })();

  return {
    close: () => {
      stopped = true;
      if (timer !== undefined) clearTimeout(timer);
    },
  };
}
