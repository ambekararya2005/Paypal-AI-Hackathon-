/**
 * Verify receipt: recompute an order's receipt hash on the server and compare it
 * with the custom_id PayPal holds for that order.
 */

import { useEffect, useState, type FormEvent } from "react";
import { CheckCircledIcon } from "@radix-ui/react-icons";
import { apiBase, type Verify } from "../../ops/store";
import { go, orderHref } from "../router";
import { Empty, Id, PageHead, Panel } from "../ui";
import { VerifyResult } from "./OrderDetail";

type Result = (Verify & { recorded_hash?: string }) | { error: string };

export default function VerifyPage({ id }: { id: string }) {
  const [q, setQ] = useState(id);
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<Result | null>(null);

  async function check(orderId: string) {
    setBusy(true);
    setRes(null);
    try {
      const r = await fetch(`${apiBase()}/receipts/${encodeURIComponent(orderId)}/verify`);
      const body = (await r.json()) as Verify & { detail?: string };
      setRes(r.ok ? body : { error: typeof body.detail === "string" ? body.detail : `Verify failed (${r.status})` });
    } catch {
      setRes({ error: "The verify endpoint is not reachable from this copy of the site." });
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    setQ(id);
    if (id) void check(id);
  }, [id]);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const v = q.trim().toUpperCase();
    if (!v) return;
    if (v === id) void check(v);
    else go(`/verify?id=${encodeURIComponent(v)}`);
  };

  return (
    <>
      <PageHead
        title="Verify receipt"
        sub="Every order MANTIS scores carries the sha256 of its receipt (basket, mandate and the pages the agent read) in the PayPal order's custom_id. This recomputes the hash and asks PayPal what it holds."
      />
      <Panel>
        <form className="m-inline-form" onSubmit={submit}>
          <label className="m-label" htmlFor="verify-id">
            PayPal order ID
          </label>
          <input
            id="verify-id"
            className="m-input mono"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="5YM287057T143621E"
            spellCheck={false}
            autoComplete="off"
          />
          <button type="submit" className="m-btn m-btn-primary" disabled={busy || !q.trim()}>
            <CheckCircledIcon aria-hidden /> {busy ? "Checking PayPal…" : "Verify"}
          </button>
        </form>
      </Panel>

      {res &&
        ("error" in res ? (
          <Panel>
            <Empty title="Could not verify">{res.error}</Empty>
          </Panel>
        ) : (
          <Panel title="Result" actions={<a href={orderHref(res.order_id)}>Open order →</a>}>
            <VerifyResult v={res} />
            <dl className="m-dl m-dl-wide">
              <dt>Order</dt>
              <dd>
                <Id value={res.order_id} href={orderHref(res.order_id)} />
              </dd>
              <dt>Hash we computed</dt>
              <dd>
                <Id value={res.receipt_hash} />
              </dd>
              <dt>Hash PayPal holds</dt>
              <dd>{res.custom_id ? <Id value={res.custom_id} /> : <span className="m-muted">none</span>}</dd>
              {res.recorded_hash && res.recorded_hash !== res.receipt_hash && (
                <>
                  <dt>Hash the gate recorded</dt>
                  <dd>
                    <Id value={res.recorded_hash} />
                  </dd>
                </>
              )}
            </dl>
            <p className="m-muted">{res.detail}</p>
          </Panel>
        ))}
    </>
  );
}
