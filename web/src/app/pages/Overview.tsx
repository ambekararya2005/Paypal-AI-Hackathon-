/** Overview: where a judge lands. Every number is summed from the gate log. */

import { useState } from "react";
import { CheckIcon } from "@radix-ui/react-icons";
import { consentHref, layerWords, merchantName, store, usd, useGateLog } from "../data";
import { orderHref } from "../router";
import { Empty, PageHead, Panel, Skeleton, StatusBadge, TableFoot, When } from "../ui";

const CHECKLIST: [key: string, label: string, href: string][] = [
  ["attack", "Run a test attack", "#/playground"],
  ["review", "Review a held order", "#/review"],
  ["verify", "Verify a receipt against PayPal", "#/verify"],
  ["mcp", "Connect Claude Desktop over MCP", "#/mcp"],
];
const KEY = "mantis.checklist";

function Checklist() {
  const [done, setDone] = useState<Set<string>>(() => {
    try {
      return new Set(JSON.parse(store.get(KEY) ?? "[]") as string[]);
    } catch {
      return new Set();
    }
  });
  const mark = (k: string) => {
    const next = new Set(done).add(k);
    setDone(next);
    store.set(KEY, JSON.stringify([...next]));
  };
  return (
    <Panel title="Getting started" actions={<span className="m-muted">{done.size} of 4 done</span>} flush>
      <ol className="m-check">
        {CHECKLIST.map(([k, label, href]) => (
          <li key={k}>
            <a href={href} onClick={() => mark(k)} className={done.has(k) ? "m-done" : ""}>
              <span className="m-tick" aria-hidden>
                {done.has(k) && <CheckIcon />}
              </span>
              <span>{label}</span>
              {done.has(k) && <span className="sr-only">(done)</span>}
            </a>
          </li>
        ))}
      </ol>
    </Panel>
  );
}

export default function Overview() {
  const log = useGateLog();
  const rows = log.rows;
  const held = rows.filter((r) => r.final_state === "held");
  const sum = (k: "captured_usd" | "protected_usd") => rows.reduce((a, r) => a + r[k], 0);
  const recent = rows.slice(0, 8);

  return (
    <>
      <PageHead
        title="Overview"
        sub="MANTIS sits between AI shopping agents and PayPal. Agents can only authorize; MANTIS captures, voids or holds."
      />

      <div className="m-grid-main">
        <div className="m-col">
          <Checklist />

          <div className="m-strip" aria-label="Summary">
            {log.status === "loading" ? (
              <div className="m-strip-cell"><Skeleton rows={2} /></div>
            ) : log.status === "error" ? (
              <div className="m-strip-cell m-muted">The gate log is not reachable from this copy of the site.</div>
            ) : (
              <>
                <div className="m-strip-cell">
                  <div className="m-stat-label">Orders scored</div>
                  <div className="m-stat num">{log.total}</div>
                </div>
                <div className="m-strip-cell">
                  <div className="m-stat-label">Captured</div>
                  <div className="m-stat num">{usd(sum("captured_usd"))}</div>
                </div>
                <div className="m-strip-cell">
                  <div className="m-stat-label">Voided, blocked before capture</div>
                  <div className="m-stat num">{usd(sum("protected_usd"))}</div>
                </div>
                <div className="m-strip-cell">
                  <div className="m-stat-label">Awaiting review</div>
                  <div className="m-stat num">{held.length}</div>
                </div>
              </>
            )}
          </div>

          <Panel title="Recent orders" actions={<a href="#/orders">View all orders →</a>} flush>
            {log.status === "loading" ? (
              <div className="m-pad"><Skeleton rows={6} /></div>
            ) : recent.length === 0 ? (
              <Empty title="No orders yet">Orders appear here as soon as an agent asks PayPal to authorize one.</Empty>
            ) : (
              <>
                <div className="m-table-wrap">
                  <table className="m-table">
                    <thead>
                      <tr>
                        <th>Time</th>
                        <th>Merchant</th>
                        <th className="m-r">Amount</th>
                        <th>Status</th>
                        <th className="m-hide-sm">Stopped by</th>
                        <th className="m-hide-sm">Order ID</th>
                      </tr>
                    </thead>
                    <tbody>
                      {recent.map((r) => (
                        <tr key={r.order_id} className="m-row-link" onClick={() => (window.location.hash = orderHref(r.order_id))}>
                          <td className="m-nowrap m-muted"><When ts={r.ts} /></td>
                          <td>{merchantName(r.merchant_id)}</td>
                          <td className="m-r num">{usd(r.total, r.currency)}</td>
                          <td><StatusBadge state={r.final_state} /></td>
                          <td className="m-hide-sm m-muted">{layerWords(r.layers_fired)}</td>
                          <td className="m-hide-sm">
                            <a className="mono" href={orderHref(r.order_id)} onClick={(e) => e.stopPropagation()}>
                              {r.order_id}
                            </a>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <TableFoot shown={recent.length} total={log.total} />
              </>
            )}
          </Panel>
        </div>

        <aside className="m-col">
          <Panel title="Review queue" actions={held.length ? <span className="m-count">{held.length}</span> : undefined} flush>
            {log.status === "loading" ? (
              <div className="m-pad"><Skeleton rows={2} /></div>
            ) : held.length === 0 ? (
              <Empty title="Nothing waiting">Held orders appear here when MANTIS isn’t sure.</Empty>
            ) : (
              <ul className="m-list">
                {held.map((r) => (
                  <li key={r.order_id}>
                    <div>
                      <div>
                        {merchantName(r.merchant_id)} · <span className="num">{usd(r.total, r.currency)}</span>
                      </div>
                      <div className="m-muted m-small"><When ts={r.ts} /> · {layerWords(r.layers_fired)}</div>
                    </div>
                    <a className="m-btn m-btn-sm" href={consentHref(r.order_id)}>
                      Review
                    </a>
                  </li>
                ))}
              </ul>
            )}
          </Panel>
        </aside>
      </div>
    </>
  );
}
