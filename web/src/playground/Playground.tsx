/**
 * "Attack MANTIS": write a poisoned review, watch a real agent read it, watch the
 * gate decide.
 *
 * The visitor edits one review on our fake store's reviews page (1,500 characters
 * max), picks a shopping task, and the real agent (gpt-4.1-mini) runs against the
 * PayPal sandbox through the MANTIS gate. Steps stream in as the agent takes them;
 * then the decision, the layers that fired, the judge's quote and the sandbox order.
 * When live runs are off or today's budget is spent, three recorded attacks replay
 * offline instead.
 *
 * Every string here, including the visitor's own text, is rendered as a React text
 * node: escaped by construction. No HTML is ever injected.
 */

import { useEffect, useRef, useState } from "react";
import { apiBase, money } from "../ops/store";
import "../ops/ops.css";
import "./playground.css";

type Config = {
  template: string;
  max_chars: number;
  tasks: { id: string; text: string }[];
  recorded: { id: string; title: string }[];
  live_available: boolean;
  live_unavailable_reason: string;
  per_ip_per_hour: number;
};

type Step = { tool: string; args: Record<string, unknown>; result: string };

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

const MERCHANT: Record<string, string> = {
  "m-freshmart": "FreshMart",
  "m-gifthub": "GiftHub",
  "m-greenleaf": "GreenLeaf Market",
};

function describe(step: Step): string {
  const a = step.args;
  switch (step.tool) {
    case "read_page": return `read ${String(a.url ?? "")}`;
    case "search_products": return `searched “${String(a.query ?? "")}”`;
    case "add_to_cart": return `added ${String(a.quantity ?? 1)} × ${String(a.sku ?? "")}`;
    case "view_cart": return "looked at the carts";
    case "checkout": return `asked to pay at ${String(a.store ?? "")}`;
    default: return step.tool;
  }
}

export default function Playground() {
  const [cfg, setCfg] = useState<Config | null>(null);
  const [text, setText] = useState("");
  const [task, setTask] = useState("weekly");
  const [run, setRun] = useState<RunState | null>(null);
  const [error, setError] = useState("");
  const timer = useRef<number | null>(null);

  useEffect(() => {
    fetch(`${apiBase()}/playground/config`)
      .then((r) => {
        if (!r.ok) throw new Error(String(r.status));
        return r.json() as Promise<Config>;
      })
      .then((c) => {
        setCfg(c);
        setText(c.template ?? "");
        // Deep link: #attack?replay=giftcard starts that recorded attack on load.
        const replay = new URLSearchParams(window.location.hash.split("?")[1] ?? "").get("replay");
        if (replay && c.recorded.some((r) => r.id === replay)) void start(`/playground/replay/${replay}`);
      })
      // The static (CDN) build has no API: say so instead of rendering nothing.
      .catch(() => setError("The playground needs the live MANTIS server; this copy has no API."));
    return () => {
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, []);

  function poll(runId: string) {
    fetch(`${apiBase()}/playground/run/${runId}`)
      .then((r) => r.json())
      .then((s: RunState) => {
        setRun(s);
        if (s.status === "running") timer.current = window.setTimeout(() => poll(runId), 1000);
      })
      .catch(() => {
        timer.current = window.setTimeout(() => poll(runId), 2000);
      });
  }

  async function start(path: string, body?: unknown) {
    setError("");
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
      return;
    }
    poll((payload as { run_id: string }).run_id);
  }

  if (!cfg) {
    return <div className="ops pg"><div className="ops-empty">{error || "Loading the playground…"}</div></div>;
  }
  const busy = run?.status === "running";
  const over = text.length > cfg.max_chars;

  return (
    <div className="ops pg" data-ag-theme-mode="mantis-dark">
      <div className="ops-bar">
        <div>
          <h2>Attack MANTIS</h2>
          <span className="ops-sub">
            Write a poisoned review. A real AI shopping agent reads it on our fake store and tries to
            pay through PayPal (sandbox). MANTIS decides what happens to the money.
          </span>
        </div>
      </div>

      <div className="pg-grid">
        <section className="pg-card">
          <h3>1 · Your review on FreshMart’s reviews page</h3>
          <textarea
            className="pg-text"
            value={text}
            maxLength={cfg.max_chars}
            onChange={(e) => setText(e.target.value)}
            spellCheck={false}
            aria-label="Your review text"
          />
          <div className={`pg-count ${over ? "bad" : ""}`}>
            {text.length} / {cfg.max_chars} characters · it is shown to the agent among three real reviews
          </div>
          <h3>2 · What the agent was asked to do</h3>
          <select className="pg-select" value={task} onChange={(e) => setTask(e.target.value)} aria-label="Task">
            {cfg.tasks.map((t) => <option key={t.id} value={t.id}>{t.text}</option>)}
          </select>
          <div className="pg-actions">
            <button
              type="button"
              className="ops-btn pg-go"
              disabled={busy || !cfg.live_available || !text.trim() || over}
              onClick={() => start("/playground/run", { page_text: text, task_id: task })}
            >
              {busy && run?.kind === "live" ? "Agent running…" : "Run the agent live"}
            </button>
            <button type="button" className="ops-link" onClick={() => setText(cfg.template)} disabled={busy}>
              reset to the template
            </button>
          </div>
          {!cfg.live_available && <p className="pg-note">{cfg.live_unavailable_reason} Replay a recorded attack instead.</p>}
          <p className="pg-note">
            Live runs: {cfg.per_ip_per_hour} per hour per visitor, on the PayPal sandbox. Nothing here touches a
            real store or real money.
          </p>
          <h3>Or replay a recorded attack (offline)</h3>
          <div className="pg-recorded">
            {cfg.recorded.map((r) => (
              <button key={r.id} type="button" className="ops-btn" disabled={busy}
                      onClick={() => start(`/playground/replay/${r.id}`)}>
                {r.title}
              </button>
            ))}
          </div>
          {error && <p className="pg-error">{error}</p>}
        </section>

        <section className="pg-card">
          <h3>3 · What the agent did</h3>
          {!run && <p className="ops-faint">Run the agent to see its steps here, as it takes them.</p>}
          {run && (
            <ol className="pg-steps">
              {run.steps.map((s, i) => (
                <li key={i} className={s.tool === "checkout" ? "pay" : ""}>
                  <span className="pg-tool">{s.tool}</span> {describe(s)}
                  {s.tool === "checkout" && <div className="pg-result">{s.result}</div>}
                </li>
              ))}
              {busy && <li className="pg-live">thinking…</li>}
            </ol>
          )}
          {run?.status === "error" && <p className="pg-error">{run.error}</p>}

          {run?.outcome && (
            <>
              <h3>4 · What MANTIS decided</h3>
              <p className={`pg-summary ${run.outcome.agent_resisted ? "resisted" : ""}`}>{run.outcome.summary}</p>
              {run.outcome.judge_on_your_page?.injected && run.outcome.judge_on_your_page.quoted_span && (
                <div className="pg-judge">
                  The LLM judge quoted your review: <span className="ops-quote">“{run.outcome.judge_on_your_page.quoted_span}”</span>
                  {run.outcome.judge_on_your_page.target_action && <> — it wants the agent to <b>{run.outcome.judge_on_your_page.target_action}</b>.</>}
                </div>
              )}
              {run.outcome.judge_on_your_page?.injected === false && (
                <div className="pg-judge">The judge read your review and found no instruction aimed at the agent.</div>
              )}
              <table className="pg-orders">
                <thead><tr><th>Order (sandbox)</th><th>Store</th><th>Total</th><th>Decision</th><th>Caught by</th><th>PayPal</th></tr></thead>
                <tbody>
                  {run.outcome.orders.map((o) => (
                    <tr key={o.order_id}>
                      <td><code>{o.order_id}</code></td>
                      <td>{MERCHANT[o.merchant_id] ?? o.merchant_id}</td>
                      <td>{money(o.total)}</td>
                      <td><span className={`ops-pill ops-pill-${o.decision}`}>{o.decision.toUpperCase()}</span></td>
                      <td>
                        <span className="ops-chips">
                          {o.layers_fired.length ? o.layers_fired.map((l) => <span key={l} className={`ops-chip ops-chip-${l}`}>{l}</span>)
                            : <span className="ops-chip ops-chip-clear">clear</span>}
                        </span>
                      </td>
                      <td>
                        <b>{o.final_state}</b>
                        {o.final_state === "held" && (
                          <> · <a className="ops-link" href={`${apiBase()}/gate/${o.order_id}/consent`} target="_blank" rel="noreferrer">you decide →</a></>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="ops-faint">
                Agent: {run.outcome.llm_s}s of model time{run.kind === "replay" ? " (recorded)" : ""}. This run is now in the
                MANTIS Ops tab.
              </p>
            </>
          )}
        </section>
      </div>
    </div>
  );
}
