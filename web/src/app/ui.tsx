/** Small building blocks shared by every page. */

import { useEffect, useState, useSyncExternalStore, type ReactNode } from "react";
import { CheckIcon, CopyIcon } from "@radix-ui/react-icons";
import { relTime, statusOf, utc, type Status } from "./data";

// ----------------------------------------------------------------- toast --

let toastText: string | null = null;
let toastTimer: number | undefined;
const toastListeners = new Set<() => void>();

export function toast(text: string) {
  toastText = text;
  toastListeners.forEach((l) => l());
  if (toastTimer) window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    toastText = null;
    toastListeners.forEach((l) => l());
  }, 2000);
}

export function ToastHost() {
  const text = useSyncExternalStore(
    (l) => {
      toastListeners.add(l);
      return () => toastListeners.delete(l);
    },
    () => toastText,
  );
  return (
    <div className="m-toast-host" role="status" aria-live="polite">
      {text && (
        <div className="m-toast">
          <CheckIcon aria-hidden /> {text}
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------- status etc --

const TONE: Record<Status, string> = { Captured: "ok", Held: "warn", Voided: "bad", Authorized: "neutral" };

export function StatusBadge({ state }: { state: string }) {
  const s = statusOf(state);
  return <span className={`m-badge m-badge-${TONE[s]}`}>{s}</span>;
}

export function CopyButton({ value, label = "Copy" }: { value: string; label?: string }) {
  async function copy() {
    try {
      await navigator.clipboard.writeText(value);
      toast("Copied");
    } catch {
      toast("Copy failed: select the text instead");
    }
  }
  return (
    <button type="button" className="m-copy" onClick={copy} aria-label={`${label} ${value}`} title={label}>
      <CopyIcon aria-hidden />
    </button>
  );
}

/** A mono id with a copy button; links when `href` is given. */
export function Id({ value, href, short }: { value: string; href?: string; short?: string }) {
  const text = short ?? value;
  return (
    <span className="m-id">
      {href ? (
        <a className="mono" href={href} title={value}>
          {text}
        </a>
      ) : (
        <span className="mono" title={value}>
          {text}
        </span>
      )}
      <CopyButton value={value} />
    </span>
  );
}

/** "12 min ago", with the exact UTC time on hover. Re-renders every 30 s. */
export function When({ ts }: { ts: string }) {
  const [, tick] = useState(0);
  useEffect(() => {
    const t = window.setInterval(() => tick((n) => n + 1), 30_000);
    return () => window.clearInterval(t);
  }, []);
  return (
    <time dateTime={ts} title={utc(ts)}>
      {relTime(ts)}
    </time>
  );
}

export function Skeleton({ rows = 3, width }: { rows?: number; width?: string }) {
  return (
    <div className="m-skel" aria-busy="true" aria-label="Loading">
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="m-skel-bar" style={{ width: width ?? `${90 - (i % 3) * 15}%` }} />
      ))}
    </div>
  );
}

export function Empty({ icon, title, children }: { icon?: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="m-empty">
      {icon && <div className="m-empty-icon">{icon}</div>}
      <p className="m-empty-title">{title}</p>
      {children && <div className="m-empty-body">{children}</div>}
    </div>
  );
}

export function PageHead({ title, sub, actions }: { title: string; sub?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="m-pagehead">
      <div>
        <h1>{title}</h1>
        {sub && <p className="m-sub">{sub}</p>}
      </div>
      {actions && <div className="m-actions">{actions}</div>}
    </div>
  );
}

export function Panel({ title, actions, children, flush }: { title?: ReactNode; actions?: ReactNode; children: ReactNode; flush?: boolean }) {
  return (
    <section className={`m-panel ${flush ? "m-panel-flush" : ""}`}>
      {(title || actions) && (
        <header className="m-panel-head">
          {title && <h2>{title}</h2>}
          {actions}
        </header>
      )}
      <div className="m-panel-body">{children}</div>
    </section>
  );
}

/** `text` with the first whitespace- and case-tolerant match of `span` on the highlighter.
 *  Rendered as React text nodes: escaped by construction, never injected HTML. */
export function Highlighted({ text, span }: { text: string; span: string | null | undefined }) {
  const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const words = (span ?? "").split(/\s+/).filter(Boolean).map(esc);
  const match = words.length ? new RegExp(words.join("\\s+"), "i").exec(text) : null;
  if (!match) return <>{text}</>;
  const [a, b] = [match.index, match.index + match[0].length];
  return (
    <>
      {text.slice(0, a)}
      <mark>{text.slice(a, b)}</mark>
      {text.slice(b)}
    </>
  );
}

export function TableFoot({ shown, total }: { shown: number; total: number }) {
  return <div className="m-tfoot">{total === 0 ? "0 results" : `1–${shown} of ${total}`}</div>;
}
