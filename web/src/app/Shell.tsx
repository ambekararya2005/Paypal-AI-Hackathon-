/**
 * The app shell: test-mode banner, sidebar navigation (a drawer below 768 px),
 * and a per-page top bar with breadcrumbs, order search and "Run test attack".
 *
 * Keyboard: "/" focuses search, "g" then "o" opens Orders.
 */

import { useEffect, useRef, useState, type FormEvent, type ReactNode } from "react";
import {
  BarChartIcon,
  CheckCircledIcon,
  CodeIcon,
  Cross1Icon,
  DashboardIcon,
  ExternalLinkIcon,
  HamburgerMenuIcon,
  MagnifyingGlassIcon,
  PlayIcon,
  TableIcon,
  TimerIcon,
} from "@radix-ui/react-icons";
import { apiBase } from "../ops/store";
import { GITHUB, useApi, useGateLog, type Limits, type PlaygroundConfig } from "./data";
import { go, orderHref, type Route } from "./router";
import { ToastHost } from "./ui";

export type Crumb = [label: string, href?: string];

type NavItem = { href: string; label: string; icon: ReactNode; active: boolean; badge?: number; external?: boolean };

function NavLink({ item, onNavigate }: { item: NavItem; onNavigate: () => void }) {
  return (
    <a
      href={item.href}
      className="m-nav-item"
      aria-current={item.active ? "page" : undefined}
      onClick={onNavigate}
      {...(item.external ? { target: "_blank", rel: "noreferrer" } : {})}
    >
      <span className="m-nav-icon" aria-hidden>
        {item.icon}
      </span>
      <span className="m-nav-label">{item.label}</span>
      {item.badge ? <span className="m-count" aria-label={`${item.badge} waiting`}>{item.badge}</span> : null}
      {item.external && <ExternalLinkIcon className="m-nav-ext" aria-hidden />}
    </a>
  );
}

function Usage() {
  const cfg = useApi<PlaygroundConfig>("/playground/config", 30_000);
  const limits = useApi<Limits>("/limits", 30_000);
  if (cfg.state !== "ok" && limits.state !== "ok") {
    return <div className="m-usage m-muted">{cfg.state === "loading" ? "Loading usage…" : "No API on this copy"}</div>;
  }
  const c = cfg.state === "ok" ? cfg.data : null;
  const live = limits.state === "ok" ? !limits.data.live_disabled : c?.live_available ?? false;
  const left = c?.per_ip_remaining;
  const of = c?.per_ip_per_hour;
  return (
    <div className="m-usage">
      {left != null && of != null && (
        <>
          <div className="m-usage-line">
            Live runs: <span className="num">{left}</span> of <span className="num">{of}</span> left this hour
          </div>
          <div className="m-meter" aria-hidden>
            <div className="m-meter-fill" style={{ width: `${(100 * left) / Math.max(1, of)}%` }} />
          </div>
        </>
      )}
      <div className="m-usage-line">
        Live agent: <span className={live ? "m-on" : "m-off"}>{live ? "on" : "off"}</span>
      </div>
    </div>
  );
}

function Sidebar({ route, onNavigate }: { route: Route; onNavigate: () => void }) {
  const log = useGateLog();
  const held = log.rows.filter((r) => r.final_state === "held").length;
  const main: NavItem[] = [
    { href: "#/overview", label: "Overview", icon: <DashboardIcon />, active: route.name === "overview" },
    { href: "#/orders", label: "Orders", icon: <TableIcon />, active: route.name === "orders" || route.name === "order" },
    { href: "#/review", label: "Review queue", icon: <TimerIcon />, active: route.name === "review", badge: held },
    { href: "#/playground", label: "Playground", icon: <PlayIcon />, active: route.name === "playground" },
    { href: "#/verify", label: "Verify receipt", icon: <CheckCircledIcon />, active: route.name === "verify" },
    { href: "#/mcp", label: "Claude Desktop (MCP)", icon: <CodeIcon />, active: route.name === "mcp" },
  ];
  const research: NavItem[] = [
    { href: "#/research/console", label: "Offline research", icon: <BarChartIcon />, active: route.name === "research" },
  ];
  return (
    <div className="m-side-in">
      <a className="m-brand" href="#/overview" onClick={onNavigate}>
        <span className="m-mono" aria-hidden>
          M
        </span>
        <span>
          <span className="m-brand-name">MANTIS</span>
          <span className="m-env">Sandbox</span>
        </span>
      </a>
      <nav className="m-nav" aria-label="Main">
        {main.map((i) => (
          <NavLink key={i.href} item={i} onNavigate={onNavigate} />
        ))}
        <div className="m-nav-group">Research</div>
        {research.map((i) => (
          <NavLink key={i.href} item={i} onNavigate={onNavigate} />
        ))}
      </nav>
      <Usage />
      <div className="m-credits">
        Arya Ambekar, Aaryaman Mishra, Ishan Ambekar ·{" "}
        <a href={GITHUB} target="_blank" rel="noreferrer">
          GitHub
        </a>
      </div>
    </div>
  );
}

function Search({ inputRef }: { inputRef: React.RefObject<HTMLInputElement> }) {
  const [q, setQ] = useState("");
  const [msg, setMsg] = useState("");
  const log = useGateLog();
  async function submit(e: FormEvent) {
    e.preventDefault();
    const id = q.trim().toUpperCase();
    if (!id) return;
    setMsg("");
    if (log.rows.some((r) => r.order_id === id)) {
      setQ("");
      go(orderHref(id));
      return;
    }
    try {
      const r = await fetch(`${apiBase()}/gate/log/${encodeURIComponent(id)}`);
      if (r.ok) {
        setQ("");
        go(orderHref(id));
      } else setMsg("No order with that ID");
    } catch {
      setMsg("No order with that ID");
    }
  }
  return (
    <form className="m-search" role="search" onSubmit={submit}>
      <MagnifyingGlassIcon className="m-search-icon" aria-hidden />
      <input
        ref={inputRef}
        value={q}
        onChange={(e) => {
          setQ(e.target.value);
          setMsg("");
        }}
        placeholder="Search by PayPal order ID"
        aria-label="Search by PayPal order ID"
        spellCheck={false}
        autoComplete="off"
      />
      <kbd className="m-kbd" aria-hidden>
        /
      </kbd>
      {msg && (
        <div className="m-pop" role="alert">
          {msg}
        </div>
      )}
    </form>
  );
}

export default function Shell({ route, crumbs, children }: { route: Route; crumbs: Crumb[]; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const search = useRef<HTMLInputElement>(null);

  useEffect(() => setOpen(false), [route]);

  useEffect(() => {
    let g = 0;
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "/") {
        e.preventDefault();
        search.current?.focus();
      } else if (e.key === "g") {
        g = Date.now();
      } else if (e.key === "o" && Date.now() - g < 1000) {
        go("/orders");
      } else if (e.key === "Escape") {
        setOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="m-app">
      <div className="m-banner">Test data · PayPal sandbox · no real money moves</div>
      <div className="m-frame">
        <aside className="m-side" aria-label="Sidebar">
          <Sidebar route={route} onNavigate={() => undefined} />
        </aside>
        {open && (
          <div className="m-drawer-wrap" onClick={() => setOpen(false)}>
            <aside className="m-drawer" aria-label="Menu" onClick={(e) => e.stopPropagation()}>
              <button type="button" className="m-icon-btn m-drawer-close" onClick={() => setOpen(false)} aria-label="Close menu">
                <Cross1Icon />
              </button>
              <Sidebar route={route} onNavigate={() => setOpen(false)} />
            </aside>
          </div>
        )}
        <div className="m-main">
          <div className="m-mobilebar">
            <button type="button" className="m-icon-btn" onClick={() => setOpen(true)} aria-label="Open menu">
              <HamburgerMenuIcon />
            </button>
            <a className="m-brand m-brand-sm" href="#/overview">
              <span className="m-mono" aria-hidden>
                M
              </span>
              <span className="m-brand-name">MANTIS</span>
              <span className="m-env">Sandbox</span>
            </a>
          </div>
          <header className="m-top">
            <nav className="m-crumbs" aria-label="Breadcrumb">
              {crumbs.map(([label, href], i) => (
                <span key={i} className="m-crumb">
                  {i > 0 && <span className="m-crumb-sep">/</span>}
                  {href && i < crumbs.length - 1 ? <a href={href}>{label}</a> : <span aria-current="page">{label}</span>}
                </span>
              ))}
            </nav>
            <Search inputRef={search} />
            <a className="m-btn m-btn-primary" href="#/playground">
              <PlayIcon aria-hidden /> Run test attack
            </a>
          </header>
          <main className="m-content">{children}</main>
          <footer className="m-foot">
            <span>Arya Ambekar, Aaryaman Mishra, Ishan Ambekar</span>
            <span>Vault flow built; live demo uses the card path</span>
            <span>
              <a href={GITHUB} target="_blank" rel="noreferrer">
                GitHub
              </a>{" "}
              · MIT licence
            </span>
          </footer>
        </div>
      </div>
      <ToastHost />
    </div>
  );
}
