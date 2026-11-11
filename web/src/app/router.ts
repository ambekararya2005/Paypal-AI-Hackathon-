/**
 * Hash router with real paths: #/overview, #/orders/5YM287057T143621E, …
 *
 * Every route is deep-linkable and the back button works because navigation is
 * plain `location.hash` assignment. The links the old console handed out
 * (#ops, #attack, #console, #results, #atlas, #fidelity, #home, #how,
 * #research), and #/developers/mcp from before the MCP page moved, are
 * rewritten in place with replaceState, so a bookmark or a link in the README
 * lands on the new page without an extra history entry.
 */

import { useEffect, useState } from "react";

export type ResearchTab = "console" | "results" | "atlas" | "fidelity";
export const RESEARCH_TABS: ResearchTab[] = ["console", "results", "atlas", "fidelity"];

export type Route =
  | { name: "overview" }
  | { name: "orders" }
  | { name: "order"; id: string }
  | { name: "review" }
  | { name: "playground" }
  | { name: "verify"; id: string }
  | { name: "mcp" }
  | { name: "research"; tab: ResearchTab }
  | { name: "notfound"; path: string };

/** Old single-word hashes -> new paths. The query string is kept (#attack?replay=x). */
const LEGACY: Record<string, string> = {
  "": "/overview",
  home: "/overview",
  how: "/overview",
  ops: "/orders",
  attack: "/playground",
  research: "/research/console",
  console: "/research/console",
  results: "/research/results",
  atlas: "/research/atlas",
  fidelity: "/research/fidelity",
};

/** The new hash for a legacy one, or null when it is already a path. */
export function redirectFor(hash: string): string | null {
  const raw = hash.replace(/^#/, "");
  // The MCP page moved from #/developers/mcp to #/mcp.
  if (/^\/developers(\/mcp)?\/?($|\?)/.test(raw)) return "#/mcp";
  if (raw.startsWith("/")) return null;
  const [head, query] = raw.split("?");
  const target = LEGACY[head];
  if (target === undefined) return null;
  return `#${target}${query ? `?${query}` : ""}`;
}

export function parse(hash: string): Route {
  const raw = hash.replace(/^#/, "");
  const [path, query = ""] = raw.split("?");
  const parts = path.split("/").filter(Boolean).map(decodeURIComponent);
  const params = new URLSearchParams(query);
  switch (parts[0]) {
    case undefined:
    case "overview":
      return { name: "overview" };
    case "orders":
      return parts[1] ? { name: "order", id: parts[1].trim() } : { name: "orders" };
    case "review":
      return { name: "review" };
    case "playground":
      return { name: "playground" };
    case "verify":
      return { name: "verify", id: (params.get("id") ?? "").trim() };
    case "mcp":
      return { name: "mcp" };
    case "research": {
      const tab = RESEARCH_TABS.find((t) => t === parts[1]) ?? "console";
      return { name: "research", tab };
    }
    default:
      return { name: "notfound", path };
  }
}

export function useRoute(): Route {
  const read = () => {
    const to = redirectFor(window.location.hash);
    if (to) window.history.replaceState(null, "", to);
    return parse(window.location.hash);
  };
  const [route, setRoute] = useState<Route>(read);
  useEffect(() => {
    const on = () => setRoute(read());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

export const go = (path: string) => {
  window.location.hash = path.startsWith("#") ? path : `#${path}`;
};

export const orderHref = (id: string) => `#/orders/${encodeURIComponent(id)}`;
