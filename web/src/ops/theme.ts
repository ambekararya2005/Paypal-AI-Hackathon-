/**
 * The MANTIS theme for AG Studio and for the AG Grid inside our custom widgets.
 *
 * Built with the Theming API (`studioTheme.withParams`, `themeQuartz.withParams`),
 * one parameter set per mode. The Ops tab switches modes by setting
 * `data-ag-theme-mode` on its own container, so Studio, its charts and the
 * Community grid in the custom widgets all change together.
 *
 * Palette: the console's own — cyan for the defence, amber for the adversary —
 * with decision colours that read the same in both modes.
 */

import { studioTheme } from "ag-studio";
import { themeQuartz } from "ag-grid-community";

export type Mode = "mantis-dark" | "mantis-light";

const DARK = {
  bg: "#07090d",
  panel: "#0d1117",
  panel2: "#131a24",
  line: "#1f2836",
  ink: "#e6edf6",
  dim: "#8b9bb0",
  accent: "#22d3ee",
};
const LIGHT = {
  bg: "#f4f6fa",
  panel: "#ffffff",
  panel2: "#eef2f7",
  line: "#dde3ec",
  ink: "#0f1722",
  dim: "#5a6a80",
  accent: "#0e7490",
};

/** Decision colours, shared by charts and renderers. */
export const DECISION_COLOURS: Record<string, string> = {
  approve: "#16a34a",
  review: "#d97706",
  challenge: "#d97706",
  decline: "#dc2626",
};

/** Series 1-3 are approve / held / declined on the decisions chart: green, amber, red. */
const palette = ["#16a34a", "#f59e0b", "#dc2626", "#22d3ee", "#a78bfa", "#64748b"];

function studioParams(p: typeof DARK, scheme: "dark" | "light") {
  return {
    browserColorScheme: scheme,
    fontFamily: "Inter, -apple-system, Segoe UI, Roboto, sans-serif",
    fontSize: 13,
    backgroundColor: p.panel,
    foregroundColor: p.ink,
    textColor: p.ink,
    subtleTextColor: p.dim,
    borderColor: p.line,
    accentColor: p.accent,
    rowHoverColor: p.panel2,

    // Default 600 px: on a phone it pushes the canvas past the screen edge.
    studioMinWidth: 320,
    studioWrapperBorder: false,
    studioWrapperBackgroundColor: p.bg,
    studioCanvasBackgroundColor: p.bg,
    studioCanvasBorder: false,
    studioCanvasShadow: false,
    studioCanvasPadding: 16,
    studioCanvasRowHeight: 18,

    studioWidgetBackgroundColor: p.panel,
    studioWidgetBorder: { width: 1, color: p.line },
    studioWidgetBorderRadius: 10,
    studioWidgetPadding: 10,
    studioWidgetTitleFontSize: 13,
    studioWidgetTitleFontWeight: 600,
    studioWidgetTitleTextColor: p.ink,
    studioWidgetSubtitleTextColor: p.dim,

    gridHeaderBackgroundColor: p.panel2,
    gridHeaderTextColor: p.dim,
    gridHeaderFontWeight: 600,
    gridRowBorder: { width: 1, color: p.line },

    ...Object.fromEntries(
      palette.flatMap((c, i) => [
        [`chartPaletteFills${i + 1}Color`, c],
        [`chartPaletteStrokes${i + 1}Color`, c],
      ]),
    ),
  };
}

// One params object per named mode. The palette keys are generated, so the object
// is typed as the documented parameter record rather than spelled out by hand.
type StudioParams = Parameters<typeof studioTheme.withParams>[0];

export const mantisStudioTheme = studioTheme
  .withParams(studioParams(DARK, "dark") as StudioParams, "mantis-dark")
  .withParams(studioParams(LIGHT, "light") as StudioParams, "mantis-light");

function gridParams(p: typeof DARK, scheme: "dark" | "light") {
  return {
    browserColorScheme: scheme,
    fontFamily: "Inter, -apple-system, Segoe UI, Roboto, sans-serif",
    fontSize: 13,
    backgroundColor: p.panel,
    foregroundColor: p.ink,
    headerBackgroundColor: p.panel2,
    headerTextColor: p.dim,
    headerFontWeight: 600,
    borderColor: p.line,
    rowHoverColor: p.panel2,
    selectedRowBackgroundColor: `color-mix(in srgb, ${p.accent} 14%, transparent)`,
    accentColor: p.accent,
    rowHeight: 40,
    headerHeight: 34,
    wrapperBorder: false,
  };
}

export const mantisGridTheme = themeQuartz
  .withParams(gridParams(DARK, "dark"), "mantis-dark")
  .withParams(gridParams(LIGHT, "light"), "mantis-light");
