/**
 * The live defence console.
 *
 * Authorisations arrive over SSE, one at a time, and each one is scored in front
 * of the judge: a risk index, which layers fired, the decision, and the three
 * features that drove it. Rows go red or green as the outcome lands.
 *
 * Two things this screen is careful about
 * ----------------------------------------
 * **The outcome is revealed, not assumed.** The API sends `event` and `truth` as
 * separate objects and this component renders the decision from `event` alone.
 * The outcome column reads `truth`, and it is the only thing that does. If the
 * two were merged, the console could colour a row before the score arrived and
 * the demo would be a lookup wearing a detector's clothes.
 *
 * **The stream says what it is.** The feed over-samples fraud roughly 25x so
 * that something happens while a judge is watching. That is stated on the screen
 * in the header, not buried in a manifest, because a stream that looks like
 * production traffic and is not is the kind of thing a judge should hear from
 * the console rather than work out afterwards.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  type AuthFrame,
  type Decision,
  type Mode,
  type StreamDone,
  type StreamHandle,
  type StreamMeta,
  onModeChange,
  openStream,
  startRun,
} from "./api";

/** Layers in the order they run. L2e is omitted: RESULTS.md reports it as a
 *  failed experiment, and a bar for it on the live console would present a
 *  below-chance layer as a working one. */
const LAYERS = ["L1", "L2", "L3"] as const;

const LAYER_LABEL: Record<string, string> = {
  L1: "L1 · gradient boosting",
  L2: "L2 · residual monitor",
  L3: "L3 · ingested text",
};

const DECISIONS: Decision[] = ["approve", "challenge", "review", "decline"];

const inr = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 });

function renderValue(value: unknown): string {
  if (value === null || value === undefined) return "absent";
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4).replace(/0+$/, "");
  }
  if (typeof value === "boolean") return value ? "true" : "false";
  return String(value);
}

/** Was this row the right call? Read off `truth`, and only here. */
function outcomeOf(frame: AuthFrame): { cls: string; text: string } {
  const flagged = frame.event.decision !== "approve";
  if (frame.truth.is_fraud) {
    return flagged
      ? { cls: "hit", text: `CAUGHT ${frame.truth.attack_id ?? ""}`.trim() }
      : { cls: "miss", text: `MISSED ${frame.truth.attack_id ?? ""}`.trim() };
  }
  return flagged ? { cls: "fp", text: "FALSE POSITIVE" } : { cls: "ok", text: "legitimate" };
}

function AuthRow({
  frame,
  selected,
  onSelect,
}: {
  frame: AuthFrame;
  selected: boolean;
  onSelect: () => void;
}) {
  const { event } = frame;
  const outcome = outcomeOf(frame);
  const agentic = event.channel === "agentic";
  return (
    <div
      className={`auth d-${event.decision} ${selected ? "sel" : ""}`}
      onClick={onSelect}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && onSelect()}
    >
      <div className={`risk num`} style={{ color: riskColour(event.risk) }}>
        {event.risk === null ? "–" : Math.round(event.risk)}
      </div>
      <div className="who">
        <div className="top">
          <span className={`rail ${agentic ? "agentic" : ""}`}>{event.channel}</span>
          <span className="amt num">
            {inr.format(event.amount)} {event.currency}
          </span>
          <span className="hint">mcc {event.mcc ?? "—"}</span>
        </div>
        <div className="bot num">
          {event.merchant_id ?? "—"} · {event.ag_agent_id ? `agent ${event.ag_agent_id}` : "no agent"}
          {event.l0.fired ? ` · L0 ${event.l0.reason}` : ""}
        </div>
      </div>
      <div className={`badge ${event.decision}`}>{event.decision}</div>
      <div className={`outcome ${outcome.cls}`}>{outcome.text}</div>
    </div>
  );
}

function riskColour(risk: number | null): string {
  if (risk === null) return "var(--ink-faint)";
  if (risk >= 90) return "var(--decline)";
  if (risk >= 75) return "var(--review)";
  if (risk >= 50) return "var(--challenge)";
  return "var(--ink-dim)";
}

function Inspector({ frame }: { frame: AuthFrame | null }) {
  if (!frame) {
    return (
      <div className="panel">
        <div className="panel-head">
          <h2>Alert detail</h2>
        </div>
        <div className="panel-body hint">
          Click any authorisation in the stream to see its per-layer breakdown and the
          three features that drove the decision.
        </div>
      </div>
    );
  }
  const { event } = frame;
  return (
    <div className="panel">
      <div className="panel-head">
        <h2>Alert detail</h2>
        <span className="hint num">{event.event_id}</span>
      </div>
      <div className="panel-body">
        {event.l0.fired && (
          <div className="l0-fired">
            <div className="t">L0 protocol violation — overrides the score</div>
            <div className="r num">{event.l0.reason}</div>
            <div className="hint" style={{ marginTop: 4 }}>
              A deterministic clause. Needs no training data, and is a defensible thing
              to tell a cardholder.
            </div>
          </div>
        )}

        <div style={{ marginBottom: 14 }}>
          {LAYERS.map((name) => {
            const layer = event.layers[name];
            const pct = layer?.percentile;
            const has = pct !== null && pct !== undefined;
            return (
              <div className="layerbar" key={name}>
                <div className="lab">
                  <span>{LAYER_LABEL[name]}</span>
                  <span className="num">
                    {has ? `${(pct * 100).toFixed(1)}%` : "no opinion"}
                  </span>
                </div>
                <div className="track">
                  <div
                    className={`fill ${has ? "" : "none"}`}
                    style={{ width: has ? `${Math.max(pct * 100, 1)}%` : "100%" }}
                  />
                </div>
              </div>
            );
          })}
          <div className="hint">
            Each bar is that layer's rank against legitimate traffic. A hatched bar means
            the layer has no opinion — L3 reads ingested text, and a classic card
            authorisation carries none.
          </div>
        </div>

        <div className="panel-head" style={{ padding: "8px 0", borderTop: "1px solid var(--line)" }}>
          <h2>Why — top 3 contributions</h2>
        </div>
        {event.contributions.map((c) => (
          <div className="contrib" key={c.feature}>
            <span className={`w num ${c.contribution >= 0 ? "pos" : "neg"}`}>
              {c.contribution >= 0 ? "+" : "−"}
              {Math.abs(c.contribution).toFixed(2)}
            </span>
            <span className="f">
              <span className="num">{c.feature}</span>{" "}
              <span className="val num">= {renderValue(c.value)}</span>
            </span>
          </div>
        ))}
        <div className="hint" style={{ marginTop: 8 }}>
          Log-odds of L1's raw margin, from LightGBM's own <span className="num">pred_contrib</span>.
          Positive pushes toward fraud.
        </div>

        <div className="panel-head" style={{ padding: "8px 0", borderTop: "1px solid var(--line)", marginTop: 12 }}>
          <h2>Authorisation</h2>
        </div>
        <dl className="kv">
          <dt>rail</dt>
          <dd>{event.channel}</dd>
          <dt>entry mode</dt>
          <dd>{event.entry_mode ?? "—"}</dd>
          <dt>type</dt>
          <dd>{event.txn_type ?? "—"}</dd>
          <dt>merchant</dt>
          <dd className="num">
            {event.merchant_id ?? "—"} ({event.merchant_country ?? "—"})
          </dd>
          <dt>customer</dt>
          <dd className="num">{event.customer_id ?? "—"}</dd>
          <dt>card BIN</dt>
          <dd className="num">{event.card_bin ?? "—"}</dd>
          <dt>device</dt>
          <dd className="num">{event.device_id ?? "—"}</dd>
          {event.ag_agent_id && (
            <>
              <dt>agent</dt>
              <dd className="num">{event.ag_agent_id}</dd>
              <dt>platform</dt>
              <dd>{event.ag_agent_platform ?? "—"}</dd>
              <dt>mandate</dt>
              <dd>{event.ag_mandate_type ?? "—"}</dd>
              <dt>human present</dt>
              <dd>{event.ag_human_present === null ? "—" : String(event.ag_human_present)}</dd>
              <dt>delegation depth</dt>
              <dd className="num">{event.ag_delegation_depth ?? "—"}</dd>
            </>
          )}
        </dl>
      </div>
    </div>
  );
}

export default function Console() {
  const [frames, setFrames] = useState<AuthFrame[]>([]);
  const [selected, setSelected] = useState<AuthFrame | null>(null);
  const [meta, setMeta] = useState<StreamMeta | null>(null);
  const [done, setDone] = useState<StreamDone | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rate, setRate] = useState(6);
  const [mode, setMode] = useState<Mode>("unknown");
  const sourceRef = useRef<StreamHandle | null>(null);

  useEffect(() => onModeChange(setMode), []);

  const stop = useCallback(() => {
    sourceRef.current?.close();
    sourceRef.current = null;
    setRunning(false);
  }, []);

  // A stream left open when the tab unmounts keeps ticking on the server. A
  // judge clicking between Console and Results during a demo does exactly that.
  useEffect(() => stop, [stop]);

  const start = useCallback(async () => {
    stop();
    setFrames([]);
    setSelected(null);
    setDone(null);
    setError(null);
    try {
      // A fresh offset each run so a second demo does not replay the same first
      // ten authorisations at the same person.
      const offset = Math.floor(Math.random() * 300);
      const run = await startRun({ n_events: 240, rate, offset });
      setRunning(true);
      sourceRef.current = openStream(
        run,
        {
          onMeta: setMeta,
          onAuth: (frame) => {
            // Newest first, capped. An unbounded list is a demo that gets slower
            // the longer a judge watches it.
            setFrames((prior) => [frame, ...prior].slice(0, 120));
            if (frame.event.decision === "decline") setSelected(frame);
          },
          onDone: (d) => {
            setDone(d);
            setRunning(false);
          },
          onError: (message) => {
            setError(message);
            setRunning(false);
          },
        },
        { offset },
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [rate, stop]);

  const tally: Record<string, number> = {};
  let caught = 0;
  let missed = 0;
  let falsePositives = 0;
  for (const frame of frames) {
    tally[frame.event.decision] = (tally[frame.event.decision] ?? 0) + 1;
    const flagged = frame.event.decision !== "approve";
    if (frame.truth.is_fraud) flagged ? caught++ : missed++;
    else if (flagged) falsePositives++;
  }

  return (
    <>
      <div className="panel" style={{ marginBottom: 14 }}>
        <div className="panel-body controls">
          <button className={running ? "btn stop" : "btn"} onClick={running ? stop : start}>
            {running ? "Stop" : "Start authorisation stream"}
          </button>
          {running && (
            <span style={{ display: "flex", alignItems: "center", gap: 7 }}>
              <span className="live-dot" />
              <span className="hint">live</span>
            </span>
          )}
          <label className="field">
            pace
            <input
              type="range"
              min={1}
              max={20}
              value={rate}
              disabled={running}
              onChange={(e) => setRate(Number(e.target.value))}
              style={{ width: 110 }}
            />
            <span className="num">{rate}/s</span>
          </label>
          {mode === "static" && (
            <span className="rail" title="No API answered; replaying the committed feed on a timer.">
              offline replay
            </span>
          )}
          {mode === "live" && <span className="rail agentic">API connected</span>}
          <span className="hint" style={{ marginLeft: "auto", maxWidth: 620, textAlign: "right" }}>
            {meta?.sampling_note ??
              "Curated replay of pre-scored authorisations. Fraud is over-sampled so something happens while you watch."}
          </span>
        </div>
      </div>

      <div className="tallies">
        {DECISIONS.map((d) => (
          <div className={`tally ${d}`} key={d}>
            <div className="k">{d}</div>
            <div className="v num">{tally[d] ?? 0}</div>
          </div>
        ))}
        <div className="tally caught">
          <div className="k">fraud caught</div>
          <div className="v num">{caught}</div>
        </div>
        <div className="tally missed">
          <div className="k">fraud missed</div>
          <div className="v num">{missed}</div>
        </div>
        <div className="tally">
          <div className="k">false positives</div>
          <div className="v num">{falsePositives}</div>
        </div>
      </div>

      {error && <div className="panel"><div className="error">{error}</div></div>}

      <div className="console-grid">
        <div className="panel">
          <div className="panel-head">
            <h2>Authorisation stream</h2>
            <span className="hint num">
              {frames.length ? `${frames.length} shown` : "idle"}
              {meta ? ` · of ${meta.n_events}` : ""}
            </span>
          </div>
          <div className="stream">
            {frames.length === 0 && (
              <div className="loading">
                Press <strong>Start</strong> to stream authorisations through the Mandate
                Firewall.
              </div>
            )}
            {frames.map((frame) => (
              <AuthRow
                key={`${frame.seq}-${frame.event.event_id}`}
                frame={frame}
                selected={selected?.event.event_id === frame.event.event_id}
                onSelect={() => setSelected(frame)}
              />
            ))}
          </div>
          {done && (
            <div className="panel-body" style={{ borderTop: "1px solid var(--line)" }}>
              <span className="hint">
                Run complete — {done.n_events} authorisations, {done.caught} fraud caught,{" "}
                {done.missed} missed, {done.false_positives} false positives.
              </span>
            </div>
          )}
        </div>

        <Inspector frame={selected} />
      </div>
    </>
  );
}
