/**
 * MANTIS Ops — an AG Studio dashboard over the gate log.
 *
 * AG Studio owns the layout, the KPI tiles, the charts, the filters and the theme;
 * two MANTIS custom widgets live inside it:
 *
 *   mantisOrderFeed    AG Grid Community + custom renderers + full-width detail rows
 *   mantisOrderDetail  the selected order: provenance with the injected span marked,
 *                      the mandate as CAN / CANNOT, the receipt and Verify
 *
 * Licence: the AG Studio key comes from /api/client-config at runtime (Render env
 * var AG_STUDIO_LICENSE_KEY), falling back to VITE_AG_STUDIO_LICENSE_KEY in local
 * dev. With neither, Studio still renders (with its trial watermark). No AG Grid or
 * AG Charts Enterprise feature is used anywhere: the Studio licence does not cover
 * Enterprise inside custom widgets, and we do not have that licence.
 */

import { useEffect, useMemo, useState } from "react";
import { AgStudio, AgStudioProvider, createWidgets } from "ag-studio-react";
import type { AgReportState, AgWidgetParams } from "ag-studio";
import { OrderDetailWidget } from "./OrderDetail";
import { OrderFeedWidget } from "./OrderFeed";
import { LAYERS } from "./renderers";
import { MERCHANTS, apiBase, money, opsStore, useOps, type LogRow } from "./store";
import { mantisStudioTheme, type Mode } from "./theme";
import "./ops.css";

const POLL_MS = 2000;
const ICON = { url: "data:image/svg+xml;utf8," + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path d="M8 1l6 3v4c0 4-3 6-6 7-3-1-6-3-6-7V4z" fill="#22d3ee"/></svg>') };

/** Flat rows for Studio's data engine: one per order, one per (order, layer) catch. */
function toSources(rows: LogRow[]) {
  const orders = rows.map((r) => {
    const d = new Date(r.ts);
    const ok = !Number.isNaN(d.getTime());
    return {
      order_id: r.order_id,
      day: ok ? d.toISOString().slice(0, 10) : "",
      hour: ok ? `${d.toISOString().slice(5, 10)} ${d.toISOString().slice(11, 13)}:00` : "",
      source: r.source,
      merchant: MERCHANTS[r.merchant_id] ?? r.merchant_id,
      decision: r.decision,
      final_state: r.final_state,
      total_usd: r.total,
      protected_usd: r.protected_usd,
      captured_usd: r.captured_usd,
      held_usd: r.held_usd,
      risk: r.risk,
      approved: r.decision === "approve" ? 1 : 0,
      held: r.decision === "review" || r.decision === "challenge" ? 1 : 0,
      declined: r.decision === "decline" ? 1 : 0,
      caught_by: r.layers_fired[0] ?? "clear",
      judge_s: r.latency.judge_ms / 1000,
      agent_s: r.latency.agent_ms / 1000,
    };
  });
  const catches = rows.flatMap((r) =>
    r.layers_fired.map((layer) => ({
      order_id: r.order_id,
      layer: LAYERS.find(([id]) => id === layer)?.[1] ?? layer,
      protected_usd: r.protected_usd,
    })),
  );
  return {
    sources: [
      { id: "orders", name: "Orders", data: orders, fields: ORDER_FIELDS },
      { id: "catches", name: "Catches", data: catches, fields: CATCH_FIELDS },
    ],
  };
}

const USD = { format: new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }) };
const text = (id: string, name: string) => ({ id, name, format: "textFormat" });
const usd = (id: string, name: string) => ({ id, name, format: "currencyFormat", formatOptions: USD });

/** Typed fields: names for filters and axes, real currency formatting on money. */
const ORDER_FIELDS = [
  text("order_id", "Order"),
  text("day", "Day"),
  text("hour", "Hour (UTC)"),
  text("source", "Source"),
  text("merchant", "Merchant"),
  text("decision", "Decision"),
  text("final_state", "PayPal state"),
  text("caught_by", "Caught by"),
  usd("total_usd", "Order total"),
  usd("protected_usd", "$ protected"),
  usd("captured_usd", "$ captured"),
  usd("held_usd", "$ held"),
  { id: "risk", name: "Risk", format: "percentageFormat" },
  { id: "approved", name: "Approved", format: "integerFormat" },
  { id: "held", name: "Held for a human", format: "integerFormat" },
  { id: "declined", name: "Declined", format: "integerFormat" },
  { id: "judge_s", name: "Judge (s)", format: "decimalFormat" },
  { id: "agent_s", name: "Agent (s)", format: "decimalFormat" },
];
const CATCH_FIELDS = [
  text("order_id", "Order"),
  text("layer", "Layer"),
  usd("protected_usd", "$ protected"),
];

const label = (t: string) => ({ style: { label: { enabled: true, text: t } } });
const titled = (t: string) => ({ title: { enabled: true, text: t } });
/** Title plus a bottom legend, for charts whose colours carry meaning. */
const titledLegend = (t: string) => ({
  ...titled(t),
  style: { theme: { common: { legend: { enabled: true, position: "bottom" } } } },
});

const DESKTOP: AgReportState = {
  pages: [
    {
      id: "ops",
      widgets: {
        kOrders: { type: "value", dataMapping: { value: [{ id: "orders.order_id", aggregation: "count" }] }, format: label("Orders scored") },
        kProtected: { type: "value", dataMapping: { value: [{ id: "orders.protected_usd", aggregation: "sum" }] }, format: label("$ protected — never moved") },
        kCaptured: { type: "value", dataMapping: { value: [{ id: "orders.captured_usd", aggregation: "sum" }] }, format: label("$ captured — approved") },
        kHeld: { type: "value", dataMapping: { value: [{ id: "orders.held_usd", aggregation: "sum" }] }, format: label("$ held for a human") },
        cTime: {
          type: "column-chart-stacked",
          format: titledLegend("Decisions over time"),
          dataMapping: {
            categoryKey: [{ id: "orders.hour" }],
            // One series per decision, in a fixed order, so the palette's green /
            // amber / red always mean approve / held / declined. A legendKey would
            // colour them in data order, which once painted "approve" red.
            valueKey: [
              { id: "orders.approved", aggregation: "sum" },
              { id: "orders.held", aggregation: "sum" },
              { id: "orders.declined", aggregation: "sum" },
            ],
          },
        },
        cMoney: {
          type: "column-chart-grouped",
          format: titledLegend("$ protected vs $ captured, by merchant"),
          dataMapping: {
            categoryKey: [{ id: "orders.merchant" }],
            valueKey: [
              { id: "orders.protected_usd", aggregation: "sum" },
              { id: "orders.captured_usd", aggregation: "sum" },
            ],
          },
        },
        cLayers: {
          type: "bar-chart-grouped",
          format: titled("Catches by layer"),
          dataMapping: {
            categoryKey: [{ id: "catches.layer" }],
            valueKey: [{ id: "catches.order_id", aggregation: "count" }],
          },
        },
        feed: { type: "mantisOrderFeed", format: titled("Order feed — click a PayPal badge to expand"), dataMapping: { order: [{ id: "orders.order_id" }] } },
        detail: { type: "mantisOrderDetail", format: titled("Order detail"), dataMapping: {} },
      },
      widgetLayout: {
        kOrders: { xTrack: 0, yTrack: 0, xSpan: 6, ySpan: 6 },
        kProtected: { xTrack: 6, yTrack: 0, xSpan: 6, ySpan: 6 },
        kCaptured: { xTrack: 12, yTrack: 0, xSpan: 6, ySpan: 6 },
        kHeld: { xTrack: 18, yTrack: 0, xSpan: 6, ySpan: 6 },
        cTime: { xTrack: 0, yTrack: 6, xSpan: 8, ySpan: 14 },
        cMoney: { xTrack: 8, yTrack: 6, xSpan: 8, ySpan: 14 },
        cLayers: { xTrack: 16, yTrack: 6, xSpan: 8, ySpan: 14 },
        feed: { xTrack: 0, yTrack: 20, xSpan: 17, ySpan: 30 },
        detail: { xTrack: 17, yTrack: 20, xSpan: 7, ySpan: 30 },
      },
    },
  ],
  selectedPageId: "ops",
  panels: { filters: { collapsed: true } },
} as unknown as AgReportState;

/** Phones: one column, KPIs 2 x 2, everything else full width and stacked. */
const PHONE: AgReportState = (() => {
  const page = (DESKTOP as unknown as { pages: { widgetLayout: Record<string, unknown> }[] }).pages[0];
  const stack: [string, number, number, number, number][] = [
    ["kOrders", 0, 0, 12, 6], ["kProtected", 12, 0, 12, 6],
    ["kCaptured", 0, 6, 12, 6], ["kHeld", 12, 6, 12, 6],
    ["cTime", 0, 12, 24, 14], ["cMoney", 0, 26, 24, 14], ["cLayers", 0, 40, 24, 14],
    ["feed", 0, 54, 24, 28], ["detail", 0, 82, 24, 30],
  ];
  const widgetLayout = Object.fromEntries(
    stack.map(([id, x, y, w, h]) => [id, { xTrack: x, yTrack: y, xSpan: w, ySpan: h }]),
  );
  return {
    ...DESKTOP,
    pages: [{ ...page, widgetLayout, layout: { minWidth: 320 } }],
  } as unknown as AgReportState;
})();

const IS_PHONE = typeof window !== "undefined" && window.innerWidth < 760;
const INITIAL = IS_PHONE ? PHONE : DESKTOP;

/** `[key, ready]`. Studio is mounted only once `ready`: the provider applies the
 *  licence at mount, so a key that arrives later would leave the watermark on. */
function useLicenceKey(): [string | undefined, boolean] {
  const built = (import.meta.env.VITE_AG_STUDIO_LICENSE_KEY as string | undefined) || undefined;
  const [key, setKey] = useState<string | undefined>(built);
  const [ready, setReady] = useState(false);
  useEffect(() => {
    fetch(`${apiBase()}/client-config`)
      .then((r) => (r.ok ? r.json() : null))
      .then((c: { ag_studio_license_key?: string } | null) => {
        if (c?.ag_studio_license_key) setKey(c.ag_studio_license_key);
      })
      .catch(() => undefined)
      .finally(() => setReady(true));
  }, []);
  return [key, ready];
}

function usePollLog() {
  useEffect(() => {
    let stop = false;
    const tick = async () => {
      try {
        const r = await fetch(`${apiBase()}/gate/log?limit=500`);
        if (r.ok) {
          const body = (await r.json()) as { version: number; rows: LogRow[] };
          if (body.version !== opsStore.get().version) opsStore.setRows(body.rows, body.version);
        }
      } catch {
        /* offline: keep the last rows */
      }
      if (!stop) window.setTimeout(tick, POLL_MS);
    };
    void tick();
    return () => {
      stop = true;
    };
  }, []);
}

/** `?theme=light|dark` wins; otherwise the visitor's OS preference. */
function initialMode(): Mode {
  const q = new URLSearchParams(window.location.search).get("theme");
  if (q === "light" || q === "dark") return `mantis-${q}`;
  return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "mantis-light" : "mantis-dark";
}

export default function OpsDashboard() {
  usePollLog();
  const [licence, licenceReady] = useLicenceKey();
  const rows = useOps((s) => s.rows);
  const version = useOps((s) => s.version);
  const [mode, setMode] = useState<Mode>(initialMode);

  const data = useMemo(() => toSources(rows), [version]); // eslint-disable-line react-hooks/exhaustive-deps

  const widgets = useMemo(
    () =>
      createWidgets({
        additionalTypes: [
          {
            id: "mantisOrderFeed",
            label: "MANTIS order feed",
            icon: ICON,
            dataMapping: {
              // A category (dimension) field: Studio returns the distinct order ids
              // that pass the page's filters, with no aggregation applied.
              order: { type: "field", supportedRoles: ["category"], intent: "category", required: true },
            },
            form: (p: { createDefaults: (o: unknown) => unknown }) =>
              p.createDefaults({ dataMappingItems: [{ key: "order", label: "Order id" }] }),
            comp: OrderFeedWidget as (p: AgWidgetParams) => JSX.Element,
            defaultSize: { width: 900, height: 520 },
            minSize: { width: 480, height: 280 },
          },
          {
            id: "mantisOrderDetail",
            label: "MANTIS order detail",
            icon: ICON,
            dataMapping: {},
            form: (p: { createDefaults: (o: unknown) => unknown }) => p.createDefaults({ dataMappingItems: [] }),
            comp: OrderDetailWidget as unknown as (p: AgWidgetParams) => JSX.Element,
            defaultSize: { width: 420, height: 520 },
            minSize: { width: 320, height: 280 },
          },
        ],
      } as never),
    [],
  );

  const totals = useMemo(() => {
    const sum = (k: "protected_usd" | "captured_usd") => rows.reduce((a, r) => a + r[k], 0);
    return { protected: sum("protected_usd"), captured: sum("captured_usd") };
  }, [rows]);

  return (
    // `ag-theme-mode` marks the ancestor whose data-ag-theme-mode Studio and the
    // grids follow; without the class the attribute is ignored.
    <div className="ops ag-theme-mode" data-ag-theme-mode={mode}>
      <div className="ops-bar">
        <div>
          <h2>MANTIS Ops</h2>
          <span className="ops-sub">
            every authorization the gate decided · {rows.length} orders · {money(totals.protected)} protected ·{" "}
            {money(totals.captured)} captured · live, every 2 s
          </span>
        </div>
        <button type="button" className="ops-btn" onClick={() => setMode(mode === "mantis-dark" ? "mantis-light" : "mantis-dark")}>
          {mode === "mantis-dark" ? "☀ Light" : "☾ Dark"}
        </button>
      </div>
      <div className="ops-studio">
        {version < 0 || !licenceReady ? (
          // Studio resolves initialState's field references against the data it is
          // mounted with; mounted on an empty source, every widget is dropped.
          <div className="ops-empty">Loading the gate log…</div>
        ) : (
          <AgStudioProvider licenseKey={licence}>
            <AgStudio
              theme={mantisStudioTheme}
              data={data as never}
              widgets={widgets}
              initialState={INITIAL}
              mode="view"
              style={{ height: "100%" }}
            />
          </AgStudioProvider>
        )}
      </div>
    </div>
  );
}
