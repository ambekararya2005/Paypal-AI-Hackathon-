/**
 * The MANTIS theme for AG Studio and for the AG Grid inside our custom widgets.
 *
 * Built with the Theming API (`studioTheme.withParams`, `themeQuartz.withParams`),
 * one parameter set per mode. The Orders page switches modes by setting
 * `data-ag-theme-mode` on its own container, so Studio, its charts and the
 * Community grid in the custom widgets all change together. Light is the default.
 *
 * Colours are the app tokens (web/src/tokens.css). The chart palette is the
 * status colours, in the order the decision chart's series are declared:
 * ok (approved / captured), warn (held), bad (declined / voided), then text-3
 * and border-2 for anything else.
 */

import { studioTheme } from "ag-studio";
import { themeQuartz } from "ag-grid-community";

export type Mode = "mantis-dark" | "mantis-light";

const LIGHT = {
  bg: "#ffffff",
  panel: "#ffffff",
  panel2: "#f7f7f5",
  hover: "#f2f2ef",
  line: "#e6e5e1",
  ink: "#1a1a18",
  dim: "#5f5e5a",
  accent: "#1d6b48",
};
/** Warm neutral dark; no navy. */
const DARK = {
  bg: "#141413",
  panel: "#1c1c1a",
  panel2: "#232321",
  hover: "#2a2a27",
  line: "#34332f",
  ink: "#ececea",
  dim: "#a3a29c",
  accent: "#4fa37a",
};

export const STATUS = { ok: "#1d6b48", warn: "#8a5a00", bad: "#b42318", text3: "#8f8e89", border2: "#d4d3ce" };

/** Series 1-3 are approve / held / declined on the decisions chart. */
const palette = [STATUS.ok, STATUS.warn, STATUS.bad, STATUS.text3, STATUS.border2];

const FONT = '"IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif';

function studioParams(p: typeof LIGHT, scheme: "dark" | "light") {
  return {
    browserColorScheme: scheme,
    fontFamily: FONT,
    fontSize: 13,
    backgroundColor: p.panel,
    foregroundColor: p.ink,
    textColor: p.ink,
    subtleTextColor: p.dim,
    borderColor: p.line,
    accentColor: p.accent,
    rowHoverColor: p.hover,

    // Default 600 px: on a phone it pushes the canvas past the screen edge.
    studioMinWidth: 320,
    studioWrapperBorder: false,
    studioWrapperBackgroundColor: p.bg,
    studioCanvasBackgroundColor: p.bg,
    studioCanvasBorder: false,
    studioCanvasShadow: false,
    studioCanvasPadding: 0,
    studioCanvasRowHeight: 18,

    studioWidgetBackgroundColor: p.panel,
    studioWidgetBorder: { width: 1, color: p.line },
    studioWidgetBorderRadius: 8,
    studioWidgetPadding: 12,
    studioWidgetTitleFontSize: 13,
    studioWidgetTitleFontWeight: 500,
    studioWidgetTitleTextColor: p.ink,
    studioWidgetSubtitleTextColor: p.dim,

    gridHeaderBackgroundColor: p.panel2,
    gridHeaderTextColor: p.dim,
    gridHeaderFontWeight: 500,
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
  .withParams(studioParams(LIGHT, "light") as StudioParams, "mantis-light")
  .withParams(studioParams(DARK, "dark") as StudioParams, "mantis-dark");

function gridParams(p: typeof LIGHT, scheme: "dark" | "light") {
  return {
    browserColorScheme: scheme,
    fontFamily: FONT,
    fontSize: 13,
    backgroundColor: p.panel,
    foregroundColor: p.ink,
    headerBackgroundColor: p.panel2,
    headerTextColor: p.dim,
    headerFontWeight: 500,
    borderColor: p.line,
    rowHoverColor: p.hover,
    selectedRowBackgroundColor: scheme === "light" ? "#e8f2ec" : "#22352b",
    accentColor: p.accent,
    rowHeight: 40,
    headerHeight: 34,
    wrapperBorder: false,
  };
}

export const mantisGridTheme = themeQuartz
  .withParams(gridParams(LIGHT, "light"), "mantis-light")
  .withParams(gridParams(DARK, "dark"), "mantis-dark");
