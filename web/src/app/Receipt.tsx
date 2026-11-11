/**
 * <Receipt row={...}/>: one gate-log row as a printable card-terminal slip.
 *
 * Shown on the order detail page as "Printable receipt". Every value is read off
 * the row; the injected sentence is the judge's verbatim quote, rendered as a
 * React text node on the highlighter.
 */

import { layerWords, merchantName, shortHash, statusOf, usd, utc, type LogRow } from "./data";

const MAX_QUOTE = 250;

/** Whole sentences of `q` up to MAX_QUOTE characters, and whether it was cut. */
function clampQuote(q: string): [string, boolean] {
  if (q.length <= MAX_QUOTE) return [q, false];
  // A sentence ends at . ! or ? followed by whitespace, so a URL's dots do not split it.
  const sentences = q.match(/[\s\S]+?[.!?](?=\s|$)\s*/g) ?? [];
  let out = "";
  for (const s of sentences) {
    if ((out + s).trimEnd().length > MAX_QUOTE) break;
    out += s;
  }
  return [(out || q.slice(0, MAX_QUOTE)).trimEnd(), true];
}

export default function Receipt({ row }: { row: LogRow }) {
  const page = row.evidence.find((e) => e.judge?.injected) ?? null;
  const [quote, cut] = clampQuote(row.judge_quote);
  const agent = (row as LogRow & { agent?: string }).agent;
  const status = statusOf(row.final_state);

  return (
    <figure className="m-slip" aria-label={`Printable receipt for PayPal sandbox order ${row.order_id}`}>
      <div className="m-slip-head">
        <span>MANTIS authorization gate</span>
        <span className="m-muted">{utc(row.ts)}</span>
      </div>
      <hr />
      <dl>
        <dt>Mandate</dt>
        <dd>{row.mandate?.description ?? row.mandate_id}</dd>
        {agent && (
          <>
            <dt>Agent</dt>
            <dd>{agent}</dd>
          </>
        )}
        <dt>Merchant</dt>
        <dd>{merchantName(row.merchant_id)}</dd>
        <dt>Amount</dt>
        <dd>
          {usd(row.total, row.currency)} {row.currency}
        </dd>
      </dl>
      <hr />
      <dl>
        <dt className="m-slip-wide">Agent read</dt>
        <dd className="m-slip-wide">
          {page && <span className="m-muted">{page.url.replace(/^https?:\/\//, "")}</span>}
          {quote ? (
            <q className="m-slip-quote" title={row.judge_quote}>
              <mark>{quote}</mark>
              {cut && " …"}
            </q>
          ) : (
            <span className="m-muted"> nothing flagged</span>
          )}
        </dd>
        <dt>Stopped by</dt>
        <dd>{row.layers_fired.length ? layerWords(row.layers_fired) : "nothing fired"}</dd>
      </dl>
      <hr />
      <dl>
        <dt>PayPal order</dt>
        <dd>{row.order_id}</dd>
        <dt>Result</dt>
        <dd className={status === "Voided" ? "m-t-bad" : status === "Held" ? "m-t-warn" : ""}>
          {status.toUpperCase()}
          {row.consent ? ` (person ${row.consent.approve ? "approved" : "denied"})` : ""}
        </dd>
        <dt>Receipt sha256</dt>
        <dd title={row.receipt_hash}>{shortHash(row.receipt_hash)}</dd>
      </dl>
    </figure>
  );
}
