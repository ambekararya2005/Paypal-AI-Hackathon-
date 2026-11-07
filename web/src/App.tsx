/**
 * Shell and tab router.
 *
 * Four screens, one per judging criterion that has a visual answer: the live
 * console (efficacy, watched happening), the results (efficacy and novelty, as
 * numbers), the atlas (diversity), and the fidelity scorecard (fidelity, plus the
 * latency figure that finishes the feasibility argument).
 *
 * Still deliberately not a router library — four tabs and a piece of state is the
 * whole requirement, and CLAUDE.md §5 says nothing heavier than the chart library.
 * The tab order is the order the argument is made in: here is the thing working,
 * here is what it scores, here is the taxonomy it scores against, here is why you
 * should believe the data underneath any of it.
 */

import { useEffect, useState } from "react";
import Atlas from "./Atlas";
import Console from "./Console";
import Fidelity from "./Fidelity";
import Results from "./Results";
import { API_BASE } from "./api";

type Tab = "console" | "results" | "atlas" | "fidelity";

const TABS: [Tab, string][] = [
  ["console", "Live console"],
  ["results", "Results"],
  ["atlas", "Attack atlas"],
  ["fidelity", "Fidelity"],
];

const SCREENS: Record<Tab, () => JSX.Element> = {
  console: Console,
  results: Results,
  atlas: Atlas,
  fidelity: Fidelity,
};

export default function App() {
  const [tab, setTab] = useState<Tab>("console");
  const [health, setHealth] = useState<{ atlas_cards: number; schema_version: string } | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <h1>MANTIS</h1>
          <span className="sub">MANDATE FIREWALL · AGENTIC COMMERCE</span>
        </div>
        {health && (
          <span className="hint num">
            schema v{health.schema_version} · {health.atlas_cards} attack cards
          </span>
        )}
        <nav className="tabs">
          {TABS.map(([id, label]) => (
            <button
              key={id}
              className={`tab ${tab === id ? "on" : ""}`}
              onClick={() => setTab(id)}
            >
              {label}
            </button>
          ))}
        </nav>
      </header>
      <main className="main">{(() => {
        const Screen = SCREENS[tab];
        return <Screen />;
      })()}</main>
    </div>
  );
}
