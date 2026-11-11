/**
 * #/orders/:id: one PayPal order the gate decided, laid out like a payment page.
 *
 * Everything comes from GET /api/gate/log/{id}. Page text is attacker-written and
 * is rendered only as React text nodes; visitor-written playground text is not in
 * the row and is fetched on demand, links already stripped.
 */

import { useState, type FormEvent } from "react";
import { CheckCircledIcon, FileIcon, MagnifyingGlassIcon } from "@radix-ui/react-icons";
import { apiBase, type Evidence, type Verify } from "../../ops/store";
import {
  LAYER_MEANING,
  LAYER_WORDS,
  consentHref,
  merchantName,
  seconds,
  shortHash,
  statusOf,
  usd,
  useApi,
  utc,
  type LogRow,
} from "../data";
import Receipt from "../Receipt";
import { go, orderHref } from "../router";
import { CopyButton, Empty, Highlighted, Id, Panel, Skeleton, StatusBadge, When } from "../ui";

const SOURCE_WORDS: Record<string, string> = {
  "paypal-sandbox": "checked against the live PayPal sandbox",
  "recorded-paypal": "checked against the recorded sandbox order",
  mock: "checked against the mock PayPal",
};

export function VerifyResult({ v }: { v: Verify }) {
  return (
    <div className={`m-verify ${v.match ? "m-verify-ok" : "m-verify-bad"}`} role="status">
      <strong>{v.match ? "Matches PayPal" : "Mismatch"}</strong>
      <span className="m-muted"> · {SOURCE_WORDS[v.source] ?? v.source}</span>
    </div>
  );
}

function VerifyButton({ id }: { id: string }) {
  const [v, setV] = useState<Verify | "pending" | "error" | null>(null);
  async function run() {
    setV("pending");
    try {
      const r = await fetch(`${apiBase()}/receipts/${encodeURIComponent(id)}/verify`);
      setV(r.ok ? ((await r.json()) as Verify) : "error");
    } catch {
      setV("error");
    }
  }
  return (
    <div className="m-verify-wrap">
      <button type="button" className="m-btn" onClick={run} disabled={v === "pending"}>
        <CheckCircledIcon aria-hidden /> {v === "pending" ? "Checking PayPal…" : "Verify receipt"}
      </button>
      {v && v !== "pending" &&
        (v === "error" ? (
          <div className="m-verify m-verify-bad" role="status">No receipt for this order on this server</div>
        ) : (
          <VerifyResult v={v} />
        ))}
    </div>
  );
}

function Timeline({ row }: { row: LogRow }) {
  const l = row.latency;
  const judged = row.layers_ran.includes("L3-judge");
  const status = statusOf(row.final_state);
  const steps: { title: string; detail: string; tone?: string; at?: string }[] = [
    {
      title: "Authorized by agent",
      detail: `${(row as LogRow & { agent?: string }).agent ?? "The agent"} asked PayPal to authorize ${usd(row.total, row.currency)} at ${merchantName(row.merchant_id)}${l.agent_ms ? ` after ${seconds(l.agent_ms)} of model time` : ""}.`,
    },
    {
      title: "Pages read",
      detail: row.evidence.length
        ? `${row.evidence.length} page${row.evidence.length > 1 ? "s" : ""}: ${row.evidence.map((e) => e.url.replace(/^https?:\/\//, "")).join(", ")}`
        : "No pages recorded.",
    },
    { title: `Scored by MANTIS (${seconds(l.score_ms)})`, detail: `Layers run: ${row.layers_ran.join(", ") || "—"}.` },
  ];
  if (judged) {
    steps.push({
      title: `Injection judge (${seconds(l.judge_ms)})`,
      detail: row.layers_fired.includes("judge") ? "Found an instruction aimed at the agent." : "Found no instruction aimed at the agent.",
    });
  }
  steps.push({
    title: status === "Held" ? "Held for a person" : status,
    detail:
      status === "Voided"
        ? "The authorization was voided. Nothing moved."
        : status === "Captured"
          ? `The payment was captured${row.capture_id ? ` (capture ${row.capture_id})` : ""}.`
          : status === "Held"
            ? "Funds stay reserved until a person approves or denies."
            : `PayPal state: ${row.final_state}.`,
    tone: status === "Voided" ? "bad" : status === "Held" ? "warn" : status === "Captured" ? "ok" : undefined,
    at: row.ts,
  });
  if (row.consent) {
    steps.push({
      title: `Person ${row.consent.approve ? "approved" : "denied"} on the consent page`,
      detail: `Signed by ${row.consent.signer}.`,
      at: row.consent.at,
    });
  }
  return (
    <ol className="m-timeline">
      {steps.map((s, i) => (
        <li key={i} className={s.tone ? `m-tl-${s.tone}` : ""}>
          <div className="m-tl-title">
            {s.title}
            {s.at && (
              <span className="m-muted m-small">
                {" "}
                · <When ts={s.at} />
              </span>
            )}
          </div>
          <div className="m-muted">{s.detail}</div>
        </li>
      ))}
      {l.paypal_ms > 0 && <li className="m-tl-note m-muted m-small">PayPal round trips for this order: {seconds(l.paypal_ms)}.</li>}
    </ol>
  );
}

function PageRead({ page, orderId }: { page: Evidence; orderId: string }) {
  const [visitor, setVisitor] = useState<string | null>(null);
  const span = page.judge?.injected ? page.judge.quoted_span : null;
  const flagged = Boolean(page.judge?.injected) || Boolean(page.l3_fired);
  async function showVisitor() {
    try {
      const r = await fetch(`${apiBase()}/gate/log/${encodeURIComponent(orderId)}/visitor-text`);
      const body = (await r.json()) as { pages: { url: string; text: string }[] };
      setVisitor(body.pages.find((p) => p.url === page.url)?.text ?? "(not available)");
    } catch {
      setVisitor("(not available)");
    }
  }
  const text = page.visitor_authored ? visitor : page.text;
  const verdicts = [
    page.l3_abstained ? `Text classifier abstained (vocabulary coverage ${(page.coverage ?? 0).toFixed(2)})` : page.l3 != null ? `Text classifier ${page.l3.toFixed(2)}` : null,
    page.judge ? `Judge: ${page.judge.injected ? "injection" : page.judge.injected === false ? "clean" : "no opinion"}` : null,
    page.visitor_authored ? "Written by a playground visitor" : null,
  ].filter(Boolean);
  return (
    <li className={`m-page ${flagged ? "m-page-flagged" : ""}`}>
      <div className="m-page-head">
        <span className="mono">{page.url}</span>
        {flagged && <span className="m-badge m-badge-bad">Flagged</span>}
      </div>
      <div className="m-muted m-small">{verdicts.join(" · ")}</div>
      {page.visitor_authored && text == null ? (
        <button type="button" className="m-link" onClick={showVisitor}>
          Show the visitor’s text
        </button>
      ) : (
        text && (
          <pre className="m-pre">
            <Highlighted text={text} span={span} />
          </pre>
        )
      )}
    </li>
  );
}

function NotFound({ id }: { id: string }) {
  const [q, setQ] = useState("");
  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (q.trim()) go(orderHref(q.trim().toUpperCase()));
  };
  return (
    <Empty icon={<MagnifyingGlassIcon />} title="No order with that ID">
      <p>
        The gate log on this server has no order <span className="mono">{id}</span>.
      </p>
      <form className="m-inline-form" onSubmit={submit}>
        <input className="m-input mono" value={q} onChange={(e) => setQ(e.target.value)} placeholder="PayPal order ID" aria-label="PayPal order ID" />
        <button type="submit" className="m-btn m-btn-primary">
          Open
        </button>
      </form>
      <p>
        <a href="#/orders">Back to all orders</a>
      </p>
    </Empty>
  );
}

export default function OrderDetail({ id }: { id: string }) {
  const res = useApi<LogRow>(`/gate/log/${encodeURIComponent(id)}`, 5000);
  if (res.state === "loading") {
    return (
      <div className="m-grid-main">
        <div className="m-col"><Skeleton rows={4} /></div>
      </div>
    );
  }
  if (res.state === "error") {
    return res.status === 404 ? (
      <NotFound id={id} />
    ) : (
      <Empty title="The gate log is not reachable">This copy of the site has no API, so there is no order to show.</Empty>
    );
  }
  const row = res.data;
  const held = row.final_state === "held";
  const why = row.layers_fired;
  const reasons = row.reasons.filter((r) => !["protocol_ok", "not_loaded", "provenance_clean"].includes(r.code));
  const injected = row.evidence.find((e) => e.judge?.injected && e.judge.quoted_span);

  return (
    <>
      <div className="m-orderhead">
        <div>
          <div className="m-amount-line">
            <span className="m-amount num">
              {usd(row.total, row.currency)} <span className="m-cur">{row.currency}</span>
            </span>
            <StatusBadge state={row.final_state} />
          </div>
          <div className="m-orderhead-meta">
            <span>{merchantName(row.merchant_id)}</span>
            <span className="m-sep">·</span>
            <Id value={row.order_id} />
            <span className="m-sep">·</span>
            <span className="m-muted">
              <When ts={row.ts} />
            </span>
          </div>
        </div>
        <div className="m-actions">
          <VerifyButton id={row.order_id} />
          {held && (
            <a className="m-btn m-btn-primary" href={consentHref(row.order_id)}>
              Review
            </a>
          )}
        </div>
      </div>

      <div className="m-grid-detail">
        <div className="m-col">
          <Panel title="Timeline">
            <Timeline row={row} />
          </Panel>

          <Panel title="Why MANTIS decided this">
            {why.length === 0 ? (
              <p className="m-muted">Every check passed, so MANTIS captured the payment.</p>
            ) : (
              <ul className="m-why">
                {why.map((l) => (
                  <li key={l}>
                    <strong>{LAYER_WORDS[l] ?? l}</strong>
                    <span className="m-muted"> · {LAYER_MEANING[l] ?? ""}</span>
                  </li>
                ))}
              </ul>
            )}
            {injected?.judge?.quoted_span && (
              <figure className="m-quote">
                <figcaption className="m-muted m-small">
                  The judge quoted this from {injected.url.replace(/^https?:\/\//, "")}
                  {injected.judge.target_action ? `; it wants the agent to ${injected.judge.target_action}` : ""}.
                </figcaption>
                <blockquote>
                  <mark>{injected.judge.quoted_span}</mark>
                </blockquote>
                {injected.judge.rationale && <p className="m-muted">{injected.judge.rationale}</p>}
              </figure>
            )}
            {reasons.length > 0 && (
              <table className="m-table m-table-plain">
                <thead>
                  <tr>
                    <th>Layer</th>
                    <th>Reason</th>
                    <th>Detail</th>
                  </tr>
                </thead>
                <tbody>
                  {reasons.map((r) => (
                    <tr key={r.layer + r.code + r.detail}>
                      <td className="m-nowrap">{r.layer}</td>
                      <td className="mono m-nowrap">{r.code}</td>
                      <td>{r.detail}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Panel>

          <Panel title="Pages the agent read">
            {row.evidence.length === 0 ? (
              <p className="m-muted">No pages recorded for this order.</p>
            ) : (
              <ul className="m-pages">
                {row.evidence.map((p) => (
                  <PageRead key={p.url} page={p} orderId={row.order_id} />
                ))}
              </ul>
            )}
          </Panel>

          <Panel title="Mandate">
            {row.mandate ? (
              <>
                <p className="m-mandate">“{row.mandate.description}”</p>
                <div className="m-two">
                  <div>
                    <h3>Allowed</h3>
                    <ul>
                      {row.mandate.can.map((c) => (
                        <li key={c}>{c}</li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <h3>Not allowed</h3>
                    <ul>
                      {row.mandate.cannot.map((c) => (
                        <li key={c}>{c}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </>
            ) : (
              <p className="m-muted">
                Mandate <span className="mono">{row.mandate_id}</span> is not held by this server.
              </p>
            )}
          </Panel>

          <details className="m-panel m-raw">
            <summary>
              <FileIcon aria-hidden /> Raw event
            </summary>
            <div className="m-raw-body">
              <div className="m-raw-copy">
                <CopyButton value={JSON.stringify(row, null, 2)} label="Copy JSON" />
              </div>
              <pre className="m-pre m-pre-code">{JSON.stringify(row, null, 2)}</pre>
            </div>
          </details>
        </div>

        <aside className="m-col">
          <Panel title="Details">
            <dl className="m-dl">
              <dt>Agent</dt>
              <dd>{(row as LogRow & { agent?: string }).agent ?? "—"}</dd>
              <dt>Mandate ID</dt>
              <dd className="mono">{row.mandate_id}</dd>
              <dt>PayPal intent</dt>
              <dd className="mono">AUTHORIZE</dd>
              {row.authorization_id && (
                <>
                  <dt>Authorization</dt>
                  <dd>
                    <Id value={row.authorization_id} />
                  </dd>
                </>
              )}
              {row.capture_id && (
                <>
                  <dt>Capture</dt>
                  <dd>
                    <Id value={row.capture_id} />
                  </dd>
                </>
              )}
              <dt>Final PayPal state</dt>
              <dd>
                <StatusBadge state={row.final_state} />
              </dd>
              <dt>Created</dt>
              <dd className="mono">{utc(row.ts)}</dd>
              <dt>Source</dt>
              <dd>{row.source}</dd>
              <dt>Receipt sha256</dt>
              <dd>
                <Id value={row.receipt_hash} short={shortHash(row.receipt_hash)} />
              </dd>
            </dl>
            <p className="m-small">
              <a href={`#/verify?id=${encodeURIComponent(row.order_id)}`}>
                Open in Verify receipt →
              </a>
            </p>
          </Panel>
          <Panel title="Printable receipt">
            <Receipt row={row} />
          </Panel>
        </aside>
      </div>
    </>
  );
}
