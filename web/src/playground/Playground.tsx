/**
 * Playground: write a poisoned review, watch a real agent read it, watch the gate decide.
 *
 * The visitor edits one review on our fake store's reviews page (1,500 characters
 * max), picks a shopping task, and the real agent (gpt-4.1-mini) runs against the
 * PayPal sandbox through the MANTIS gate. Steps stream in as the agent takes them,
 * each stamped with the server-measured t_s; then the result per order, with a
 * link to its order page. When live runs are off, the visitor's hourly allowance
 * is spent, or today's budget is gone, three recorded attacks replay offline.
 *
 * Endpoints and error handling are unchanged: 409 (kill switch or daily cap,
 * detail {message, replays}), 429 (5 live runs per hour per visitor, detail is
 * the message). Every string, the visitor's own text included, is rendered as a
 * React text node: escaped by construction.
 */

import { useEffect, useRef, useState } from "react";
import { PlayIcon, ResetIcon } from "@radix-ui/react-icons";
import { apiBase } from "../ops/store";
import { layerWords, merchantName, usd, type PlaygroundConfig } from "../app/data";
import { orderHref } from "../app/router";
import { Empty, PageHead, StatusBadge } from "../app/ui";
import "./playground.css";

type Step = { tool: string; args: Record<string, unknown>; result: string; t_s?: number | null };

type Order = {
  order_id: string;
  merchant_id: string;
  total: number;
  decision: string;
  paypal_action: string;
  final_state: string;
  layers_fired: string[];
  judge_quote: string;
};

type RunState = {
  run_id: string;
  kind: "live" | "replay";
  status: "running" | "done" | "error";
  error: string;
  steps: Step[];
  outcome: {
    summary: string;
    agent_resisted: boolean;
    judge_on_your_page: { injected: boolean | null; quoted_span: string | null; target_action: string | null; rationale: string | null } | null;
    final_text: string;
    llm_s: number;
    orders: Order[];
  } | null;
};

/** "00:09.8", or blank when the step has no server time (never estimated). */
function stamp(t: number | null | undefined): string {
  if (t == null) return "       ";
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`;
}

function args(step: Step): string {
  const a = step.args;
  switch (step.tool) {
    case "read_page":
      return String(a.url ?? "").replace(/^https?:\/\//, "");
    case "search_products":
      return `“${String(a.query ?? "")}”`;
    case "add_to_cart":
      return `${String(a.quantity ?? 1)} × ${String(a.sku ?? "")}`;
    case "checkout":
      return `${String(a.store ?? "")}  AUTHORIZE`;
    default:
      return Object.values(a).map(String).join(" ");
  }
}

/** "Maria K. (5 stars): text" -> name, stars, text. */
function parseReview(r: string): { name: string; stars: number; text: string } {
  const m = /^(.+?) \((\d) stars?\): ([\s\S]*)$/.exec(r);
  return m ? { name: m[1], stars: Number(m[2]), text: m[3] } : { name: "", stars: 0, text: r };
}

function Review({ review }: { review: string }) {
  const r = parseReview(review);
  return (
    <li className="pg-review">
      <div className="pg-review-head">
        <span className="pg-reviewer">{r.name}</span>
        {r.stars > 0 && (
          <span className="pg-stars" aria-label={`${r.stars} of 5 stars`}>
            {"★".repeat(r.stars)}
            <span className="pg-stars-off">{"★".repeat(5 - r.stars)}</span>
          </span>
        )}
      </div>
      <p>{r.text}</p>
    </li>
  );
}

/** One plain sentence about what happened, from the orders themselves. */
function summaryOf(o: NonNullable<RunState["outcome"]>): string {
  if (o.orders.length === 0) return o.summary;
  const stopped = o.orders.filter((x) => x.decision !== "approve");
  if (stopped.length === 0) return "The agent ignored your review. There was nothing to stop.";
  const parts = stopped.map((x) => `${usd(x.total)} at ${merchantName(x.merchant_id)} ${x.final_state === "held" ? "held for a person" : x.final_state}`);
  return `MANTIS stopped ${stopped.length} of ${o.orders.length} order${o.orders.length > 1 ? "s" : ""}: ${parts.join("; ")}.`;
}

export default function Playground() {
  const [cfg, setCfg] = useState<PlaygroundConfig | null>(null);
  const [text, setText] = useState("");
  const [task, setTask] = useState("weekly");
  const [run, setRun] = useState<RunState | null>(null);
  const [error, setError] = useState("");
  const [replays, setReplays] = useState(false);
  const timer = useRef<number | null>(null);

  const loadConfig = (first = false) =>
    fetch(`${apiBase()}/playground/config`)
      .then((r) => {
        if (!r.ok) throw new Error(String(r.status));
        return r.json() as Promise<PlaygroundConfig>;
      })
      .then((c) => {
        setCfg(c);
        if (!first) return;
        setText(c.template ?? "");
        // Deep link: #/playground?replay=giftcard starts that recorded attack on load.
        const replay = new URLSearchParams(window.location.hash.split("?")[1] ?? "").get("replay");
        if (replay && c.recorded.some((r) => r.id === replay)) void start(`/playground/replay/${replay}`);
      });

  useEffect(() => {
    // The static (CDN) build has no API: say so instead of rendering nothing.
    loadConfig(true).catch(() => setError("The playground needs the live MANTIS server; this copy has no API."));
    return () => {
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  function poll(runId: string) {
    fetch(`${apiBase()}/playground/run/${runId}`)
      .then((r) => r.json())
      .then((s: RunState) => {
        setRun(s);
        if (s.status === "running") timer.current = window.setTimeout(() => poll(runId), 1000);
        else void loadConfig().catch(() => undefined);
      })
      .catch(() => {
        timer.current = window.setTimeout(() => poll(runId), 2000);
      });
  }

  async function start(path: string, body?: unknown) {
    setError("");
    setReplays(false);
    setRun(null);
    const r = await fetch(`${apiBase()}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
    const payload = await r.json().catch(() => ({}));
    if (!r.ok) {
      const d = (payload as { detail?: unknown }).detail;
      setError(typeof d === "string" ? d : (d as { message?: string })?.message ?? `failed (${r.status})`);
      if (r.status === 409 || r.status === 429) setReplays(true);
      return;
    }
    poll((payload as { run_id: string }).run_id);
  }

  if (!cfg) {
    return (
      <>
        <PageHead title="Playground" />
        <div className="m-panel">
          <Empty title={error ? "Playground unavailable" : "Loading the playground…"}>{error}</Empty>
        </div>
      </>
    );
  }

  const busy = run?.status === "running";
  const over = text.length > cfg.max_chars;
  const left = cfg.per_ip_remaining;
  const noneLeft = left === 0;
  const showReplays = replays || !cfg.live_available || noneLeft;
  const preview = cfg.page_preview;
  const reviews = preview?.reviews ?? [];
  const at = preview?.insert_at ?? 1;

  return (
    <>
      <PageHead
        title="Playground"
        sub="Write a poisoned review. A real AI shopping agent reads it on our test store and tries to pay through the PayPal sandbox. MANTIS decides what happens to the money."
      />
      <div className="pg-grid">
        <section className="m-panel">
          <header className="m-panel-head">
            <h2>Page the agent will read</h2>
          </header>
          <div className="m-panel-body">
            <div className="pg-page">
              <div className="pg-url mono">{(preview?.url ?? "https://shop.freshmart.test/reviews").replace(/^https?:\/\//, "")}</div>
              <div className="pg-page-body">
                <div className="pg-store">FreshMart</div>
                <div className="pg-store-sub">Customer reviews</div>
                <ul className="pg-reviews">
                  {reviews.slice(0, at).map((r) => (
                    <Review key={r} review={r} />
                  ))}
                  <li className="pg-review pg-mine">
                    <div className="pg-review-head">
                      <label className="pg-reviewer" htmlFor="pg-text">
                        Your review
                      </label>
                      <span className={`mono pg-count ${over ? "m-t-bad" : ""}`}>
                        {text.length.toLocaleString("en-US")} / {cfg.max_chars.toLocaleString("en-US")}
                      </span>
                    </div>
                    <textarea
                      id="pg-text"
                      className="pg-text"
                      value={text}
                      maxLength={cfg.max_chars}
                      onChange={(e) => setText(e.target.value)}
                      spellCheck={false}
                    />
                  </li>
                  {reviews.slice(at).map((r) => (
                    <Review key={r} review={r} />
                  ))}
                </ul>
              </div>
            </div>

            <label className="m-label" htmlFor="pg-task">
              What the agent is asked to do
            </label>
            <select id="pg-task" className="m-input" value={task} onChange={(e) => setTask(e.target.value)}>
              {cfg.tasks.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.text}
                </option>
              ))}
            </select>

            <div className="pg-actions">
              <button
                type="button"
                className="m-btn m-btn-primary"
                disabled={busy || !cfg.live_available || noneLeft || !text.trim() || over}
                onClick={() => start("/playground/run", { page_text: text, task_id: task })}
              >
                <PlayIcon aria-hidden /> {busy && run?.kind === "live" ? "Agent running…" : "Run the agent live"}
              </button>
              <button type="button" className="m-btn" onClick={() => setText(cfg.template)} disabled={busy}>
                <ResetIcon aria-hidden /> Reset to template
              </button>
            </div>

            {!cfg.live_available ? (
              <p className="pg-note">{cfg.live_unavailable_reason} Replay a recorded attack instead.</p>
            ) : left != null ? (
              <p className="pg-note">
                {noneLeft
                  ? "No live runs left this hour. Replay a recorded attack instead."
                  : `${left} live run${left === 1 ? "" : "s"} left this hour. Sandbox only: nothing here touches a real store or real money.`}
              </p>
            ) : (
              <p className="pg-note">Live runs: {cfg.per_ip_per_hour} per hour per visitor, on the PayPal sandbox.</p>
            )}
            {error && (
              <p className="pg-error" role="alert">
                {error}
              </p>
            )}

            <div className={`pg-replays ${showReplays ? "pg-replays-on" : ""}`}>
              <span className="m-muted">Replay a recorded attack (offline):</span>
              <ul>
                {cfg.recorded.map((r) => (
                  <li key={r.id}>
                    <button type="button" className="m-link" disabled={busy} onClick={() => start(`/playground/replay/${r.id}`)}>
                      {r.title}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </section>

        <section className="m-panel">
          <header className="m-panel-head">
            <h2>Run</h2>
            {run && <span className="m-muted m-small">{run.kind === "replay" ? "Recorded replay, mock PayPal" : "Live, PayPal sandbox"}</span>}
          </header>
          <div className="m-panel-body">
            <div className="pg-console" aria-live="polite">
              {!run && <div className="pg-line pg-dim">Waiting for a run. Steps appear here as the agent takes them.</div>}
              {run?.steps.map((s, i) => (
                <div key={i} className="pg-step">
                  <div className="pg-line">
                    <span className="pg-t">{stamp(s.t_s)}</span>
                    <span className="pg-tool">{s.tool}</span>
                    <span className="pg-args">{args(s)}</span>
                  </div>
                  {s.tool === "checkout" && <div className="pg-line pg-dim pg-result">{s.result}</div>}
                </div>
              ))}
              {busy && <div className="pg-line pg-dim">agent running…</div>}
              {run?.kind === "replay" && run.status === "done" && (
                <div className="pg-line pg-dim pg-note-line">Recorded replays run from cache, so every step lands within the same tenth of a second.</div>
              )}
              {run?.status === "error" && <div className="pg-line pg-err">{run.error}</div>}
            </div>

            {run?.outcome && (
              <div className="pg-outcome">
                <p className="pg-summary">{summaryOf(run.outcome)}</p>
                {run.outcome.judge_on_your_page?.injected === false && (
                  <p className="m-muted">The judge read your review and found no instruction aimed at the agent.</p>
                )}
                {run.outcome.orders.map((o) => (
                  <div key={o.order_id} className="pg-order">
                    <div className="pg-order-head">
                      <StatusBadge state={o.final_state} />
                      <span className="num pg-amt">{usd(o.total)}</span>
                      <span>{merchantName(o.merchant_id)}</span>
                      <a className="pg-open" href={orderHref(o.order_id)}>
                        Open order →
                      </a>
                    </div>
                    <dl className="m-dl">
                      <dt>Stopped by</dt>
                      <dd>{o.layers_fired.length ? layerWords(o.layers_fired) : "Nothing fired"}</dd>
                      <dt>PayPal order</dt>
                      <dd className="mono">{o.order_id}</dd>
                      {o.judge_quote && (
                        <>
                          <dt>Judge quote</dt>
                          <dd>
                            <mark>{o.judge_quote}</mark>
                          </dd>
                        </>
                      )}
                    </dl>
                  </div>
                ))}
                <p className="m-muted m-small">
                  Agent: {run.outcome.llm_s} s of model time{run.kind === "replay" ? " (recorded)" : ""}.
                </p>
              </div>
            )}
          </div>
        </section>
      </div>
    </>
  );
}
