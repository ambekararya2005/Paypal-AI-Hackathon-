/**
 * The order feed: an AG Studio custom widget wrapping AG Grid **Community**.
 *
 * Studio owns filtering. The widget declares one data mapping (the order id) and
 * asks Studio for the ids that pass the page's filters and cross-filters via
 * `widgetApi.getData`, with `onDataUpdated` so a filter change or a new log row
 * re-filters the feed with no polling here. Full rows come from the Ops store.
 *
 * Detail without Enterprise: clicking a row's PayPal badge inserts a full-width
 * row beneath it (`isFullWidthRow` + `fullWidthCellRenderer`, both Community),
 * holding provenance, the mandate and the receipt. No master-detail module, no
 * licence needed, no watermark.
 */

import { useEffect, useMemo, useState } from "react";
import { AgGridReact } from "ag-grid-react";
import {
  AllCommunityModule,
  ModuleRegistry,
  type ColDef,
  type GetRowIdParams,
  type ICellRendererParams,
  type IsFullWidthRowParams,
  type RowClickedEvent,
  type RowHeightParams,
} from "ag-grid-community";
import type { AgWidgetParams } from "ag-studio";
import { OrderDetailView } from "./OrderDetail";
import {
  DecisionPill,
  JudgeQuote,
  LatencySplit,
  LayerChips,
  Merchant,
  PayPalBadge,
  Protected,
  RiskBar,
} from "./renderers";
import { mantisGridTheme } from "./theme";
import { opsStore, useOps, type LogRow } from "./store";

ModuleRegistry.registerModules([AllCommunityModule]);

type DetailRow = { __detail: true; parent: LogRow };
type FeedRow = LogRow | DetailRow;
const isDetail = (r: FeedRow | undefined): r is DetailRow => Boolean(r && "__detail" in r);

function DetailCell({ data }: ICellRendererParams<FeedRow>) {
  if (!isDetail(data)) return null;
  return (
    <div className="ops-fullwidth">
      <OrderDetailView row={data.parent} />
    </div>
  );
}

const time = (ts: string) => {
  const d = new Date(ts);
  return Number.isNaN(d.getTime()) ? ts : d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
};

/** Ids passing Studio's filters, or null before the first answer (show everything). */
function useStudioFilteredIds(params: AgWidgetParams): Set<string> | null {
  const [ids, setIds] = useState<Set<string> | null>(null);
  // Our own mapping key; the default registry types dataMapping with no keys.
  const fields = (params.dataMapping as Record<string, unknown> | undefined)?.order;
  useEffect(() => {
    if (!Array.isArray(fields) || fields.length === 0) return;
    const controller = new AbortController();
    const take = (rows: Record<string, unknown>[]) =>
      setIds(new Set(rows.map((r) => String(Object.values(r)[0] ?? ""))));
    params.widgetApi
      .getData({ fields }, { signal: controller.signal, onDataUpdated: (res) => take(res.results.rows) })
      .then((res) => take(res.results.rows))
      .catch(() => undefined);
    return () => controller.abort();
  }, [params.widgetApi, fields]);
  return ids;
}

export function OrderFeedWidget(params: AgWidgetParams) {
  const rows = useOps((s) => s.rows);
  const selected = useOps((s) => s.selected);
  const allowed = useStudioFilteredIds(params);
  const [open, setOpen] = useState<Set<string>>(new Set());

  const toggle = (id: string) => {
    opsStore.select(id);
    setOpen((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const rowData = useMemo<FeedRow[]>(() => {
    const out: FeedRow[] = [];
    for (const r of rows) {
      if (allowed && !allowed.has(r.order_id)) continue;
      out.push(r);
      if (open.has(r.order_id)) out.push({ __detail: true, parent: r });
    }
    return out;
  }, [rows, allowed, open]);

  const columnDefs = useMemo<ColDef<FeedRow>[]>(() => [
    { headerName: "Time", field: "ts" as never, width: 96, valueFormatter: (p) => time(String(p.value ?? "")), sort: "desc" },
    { headerName: "Merchant", field: "merchant_id" as never, width: 132, cellRenderer: Merchant },
    { headerName: "Decision", field: "decision" as never, width: 112, cellRenderer: DecisionPill },
    { headerName: "Risk", field: "risk" as never, width: 84, cellRenderer: RiskBar },
    { headerName: "Caught by", field: "layers_fired" as never, width: 146, cellRenderer: LayerChips, sortable: false, cellDataType: false },
    { headerName: "PayPal", field: "final_state" as never, width: 176, cellRenderer: PayPalBadge },
    { headerName: "Money", field: "total" as never, width: 118, cellRenderer: Protected, type: "rightAligned" },
    { headerName: "Latency", headerTooltip: "agent · judge · score · PayPal", field: "latency" as never, width: 120, cellRenderer: LatencySplit, sortable: false, cellDataType: false },
    { headerName: "Judge quote", field: "judge_quote" as never, flex: 1, minWidth: 170, cellRenderer: JudgeQuote },
  ], []);

  return (
    <div className="ops-feed">
      <AgGridReact<FeedRow>
        theme={mantisGridTheme}
        rowData={rowData}
        columnDefs={columnDefs}
        defaultColDef={{ sortable: true, resizable: true }}
        context={{ toggle }}
        getRowId={(p: GetRowIdParams<FeedRow>) => (isDetail(p.data) ? `d-${p.data.parent.order_id}` : p.data.order_id)}
        isFullWidthRow={(p: IsFullWidthRowParams<FeedRow>) => isDetail(p.rowNode.data)}
        fullWidthCellRenderer={DetailCell}
        getRowHeight={(p: RowHeightParams<FeedRow>) => (isDetail(p.data) ? 340 : 40)}
        rowClassRules={{ "ops-row-selected": (p) => !isDetail(p.data) && p.data?.order_id === selected }}
        onRowClicked={(e: RowClickedEvent<FeedRow>) => {
          if (!isDetail(e.data) && e.data) opsStore.select(e.data.order_id);
        }}
        suppressCellFocus
        animateRows
      />
    </div>
  );
}
