"""Does the synthetic population actually track the reference? Measure, then draw.

This is the honest half of Pillar 2 and a down-payment on the fidelity
scorecard. Every number here is computed against the *same* calibration object
the simulator drew from, so the comparison is "did the sampler reproduce its own
target", not "does this look plausible". Both are worth knowing; only the first
is checkable.

What is measured
----------------
* **Amount** — Kolmogorov-Smirnov distance between the realised amounts and the
  analytic mixture-of-log-normals implied by the MCC profiles. This number is
  deliberately *not* zero: round-number snapping shifts mass onto multiples of
  50/100/500 on purpose, and KS charges us for it. Reporting a small non-zero
  distance with a known cause is worth more than reporting zero.
* **Hour of day** — total-variation distance between the realised and target
  24-bin curves.
* **MCC mix** and **per-MCC median ticket** — maximum absolute deviation.
* **Merchant popularity** — the Zipf exponent recovered by OLS on the
  rank-frequency curve, against the exponent that was asked for.

The agentic block has no reference and never will until an agentic-payments
panel exists. The figure says so on its face rather than quietly implying the
whole thing was calibrated.

Matplotlib is imported lazily and its absence is survivable: the metrics are the
point, the picture is the evidence. A clean clone without matplotlib still
generates a population and still prints the numbers.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Final

import numpy as np

from mantis.core.paths import DOCS_DIR, ensure_dir
from mantis.foundry.base.reference import ReferenceStats

__all__ = ["CALIBRATION_PNG", "calibration_report", "format_report", "plot_calibration"]

CALIBRATION_PNG: Final[Path] = DOCS_DIR / "population_calibration.png"

# Palette: one accent for the synthetic draw, recessive ink for the reference
# target. The reference is a baseline, not a peer series, so it is drawn in muted
# grey *and* dashed/open — identity never rests on colour alone.
_SYNTH: Final[str] = "#2a78d6"
_REF: Final[str] = "#898781"
_ACCENT_2: Final[str] = "#eb6834"
_INK: Final[str] = "#0b0b0b"
_INK_2: Final[str] = "#52514e"
_MUTED: Final[str] = "#898781"
_GRID: Final[str] = "#e1e0d9"
_SURFACE: Final[str] = "#fcfcfb"

_TOP_MCCS: Final[int] = 14


def _mixture_cdf(stats: ReferenceStats, x: np.ndarray) -> np.ndarray:
    """CDF of the reference amount distribution: a weight-mixed set of log-normals."""
    from scipy.stats import norm

    out = np.zeros_like(x, dtype=float)
    log_x = np.log(np.maximum(x, 1e-9))
    for profile in stats.mcc_profiles:
        z = (log_x - profile.log_amount_mu) / profile.log_amount_sigma
        out += profile.weight * norm.cdf(z)
    return out


def _mixture_log10_pdf(stats: ReferenceStats, y: np.ndarray) -> np.ndarray:
    """Density of ``log10(amount)`` under the reference mixture.

    Change of variables from the natural-log parameterisation: with
    ``x = 10**y``, the Jacobian collapses neatly and leaves a plain sum of
    Gaussians in ``ln(x)`` scaled by ``ln(10)``.
    """
    ln10 = math.log(10.0)
    log_x = y * ln10
    out = np.zeros_like(y, dtype=float)
    for profile in stats.mcc_profiles:
        sigma = profile.log_amount_sigma
        z = (log_x - profile.log_amount_mu) / sigma
        out += profile.weight * ln10 * np.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))
    return out


def _zipf_exponent(counts: np.ndarray, min_count: int = 5, max_rank: int = 2_000) -> float:
    """Recover the rank-frequency exponent by OLS on the resolved log-log head.

    Only ranks whose count is above ``min_count`` are fitted. The deep tail of a
    Zipf sample is dominated by whether a merchant happened to get one
    transaction or two, and including it biases the recovered slope toward zero —
    that would be an artefact of the estimator, not a property of the population.
    """
    counts = np.sort(counts)[::-1]
    head = counts[: min(max_rank, counts.size)]
    head = head[head >= min_count]
    if head.size < 20:
        return float("nan")
    rank = np.log(np.arange(1, head.size + 1))
    slope = np.polyfit(rank, np.log(head), 1)[0]
    return float(-slope)


def calibration_report(frame: Any, stats: ReferenceStats) -> dict[str, Any]:
    """Compare a generated frame against the calibration it was drawn from.

    Args:
        frame: The flat population frame from ``simulate_frame``.
        stats: The calibration object used to generate it.

    Returns:
        A JSON-serialisable dict of distances and realised summaries. Goes
        straight into the run manifest, so the numbers on the figure and the
        numbers in the manifest can never drift apart.
    """
    from scipy.stats import kstest

    amount = frame["amount"].to_numpy(dtype=float)
    ks = kstest(amount, lambda x: _mixture_cdf(stats, np.asarray(x)))

    hours = frame["ts"].dt.hour.to_numpy()
    hour_synth = np.bincount(hours, minlength=24).astype(float)
    hour_synth /= hour_synth.sum()
    hour_ref = np.asarray(stats.hour_weights, dtype=float)
    # The reference hour curve is the *human* curve; the agentic subset is drawn
    # from a curve blended toward uniform, so the marginal is the volume-weighted
    # mix of the two. Comparing against the human curve alone would flag a
    # deviation the simulator was told to produce.
    share = float((frame["channel"] == "agentic").mean())
    blend = stats.agentic_hour_uniform_blend
    agent_ref = (1.0 - blend) * hour_ref + blend / 24.0
    agent_ref /= agent_ref.sum()
    hour_target = (1.0 - share) * hour_ref + share * agent_ref

    mcc_synth = frame["mcc"].value_counts(normalize=True)
    mcc_ref = {p.mcc: p.weight for p in stats.mcc_profiles}
    mcc_delta = {m: float(mcc_synth.get(m, 0.0) - w) for m, w in mcc_ref.items()}

    median_synth = frame.groupby("mcc")["amount"].median()
    median_delta = {
        p.mcc: float(median_synth.get(p.mcc, float("nan")) / math.exp(p.log_amount_mu) - 1.0)
        for p in stats.mcc_profiles
    }

    merchant_counts = frame["merchant_id"].value_counts().to_numpy()

    return {
        "n_events": len(frame),
        "amount_ks_distance": float(ks.statistic),
        "amount_median": float(np.median(amount)),
        "amount_mean": float(np.mean(amount)),
        "amount_p99": float(np.percentile(amount, 99)),
        "hour_total_variation": float(0.5 * np.abs(hour_synth - hour_target).sum()),
        "mcc_mix_max_abs_delta": float(max(abs(v) for v in mcc_delta.values())),
        "mcc_median_max_abs_rel_delta": float(
            max(abs(v) for v in median_delta.values() if not math.isnan(v))
        ),
        "zipf_exponent_target": float(stats.merchant_zipf_exponent),
        "zipf_exponent_realised": _zipf_exponent(merchant_counts),
        "agentic_share": share,
        "channel_mix": {
            str(k): float(v) for k, v in frame["channel"].value_counts(normalize=True).items()
        },
        "distinct_merchants": int(frame["merchant_id"].nunique()),
        "distinct_customers": int(frame["customer_id"].nunique()),
        "geo_missing_rate": float(frame["lat"].isna().mean()),
    }


def format_report(report: dict[str, Any], stats: ReferenceStats) -> str:
    """Render the calibration report as the block the CLI prints."""
    lines = [
        "calibration vs reference",
        f"  amount KS distance      : {report['amount_ks_distance']:.4f}  "
        "(non-zero by design: round-number snapping)",
        f"  hour-of-day TV distance : {report['hour_total_variation']:.4f}",
        f"  mcc mix max |delta|     : {report['mcc_mix_max_abs_delta']:.4f}",
        f"  mcc median max |rel|    : {report['mcc_median_max_abs_rel_delta']:.3f}",
        f"  merchant zipf exponent  : {report['zipf_exponent_realised']:.3f} "
        f"realised vs {report['zipf_exponent_target']:.3f} target",
        f"  amount median / mean    : {stats.currency} {report['amount_median']:,.2f} / "
        f"{stats.currency} {report['amount_mean']:,.2f}",
        f"  agentic share           : {report['agentic_share']:.2%}",
        f"  geo missing (all rows)  : {report['geo_missing_rate']:.2%}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #


def _style_axes(ax: Any, title: str, xlabel: str = "", ylabel: str = "") -> None:
    """Recessive chrome: hairline grid, no top/right spines, muted ticks."""
    ax.set_title(title, color=_INK, fontsize=11, fontweight="600", loc="left", pad=9)
    ax.set_xlabel(xlabel, color=_INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=_INK_2, fontsize=9)
    ax.tick_params(colors=_MUTED, labelsize=8, length=3, width=0.8)
    ax.grid(True, color=_GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
        ax.spines[side].set_linewidth(0.8)
    ax.set_facecolor(_SURFACE)


def _note(ax: Any, text: str, xy: tuple[float, float] = (0.98, 0.94)) -> None:
    """A small measured-value caption pinned inside the panel.

    Position is a parameter because the lollipop panels sort their data into a
    diagonal, so the free corner moves from panel to panel and a fixed corner
    would land the caption on top of a data point.
    """
    ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha="right" if xy[0] > 0.5 else "left",
        va="top" if xy[1] > 0.5 else "bottom",
        fontsize=8,
        color=_INK_2,
        zorder=6,
        bbox={"facecolor": _SURFACE, "edgecolor": _GRID, "boxstyle": "round,pad=0.35"},
    )


def plot_calibration(
    frame: Any, stats: ReferenceStats, report: dict[str, Any], out: Path | None = None
) -> Path | None:
    """Draw the six-panel calibration figure. Returns ``None`` if matplotlib is absent.

    Five panels carry a reference curve. The sixth is the agentic rail, which has
    no reference and is labelled as modelled — that panel is the one that keeps
    the figure honest.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    target = CALIBRATION_PNG if out is None else out
    ensure_dir(target.parent)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "figure.facecolor": _SURFACE,
            "savefig.facecolor": _SURFACE,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.6))
    fig.suptitle(
        "MANTIS legitimate population vs. reference calibration",
        color=_INK,
        fontsize=14,
        fontweight="700",
        x=0.008,
        ha="left",
        y=0.985,
    )
    fig.text(
        0.008,
        0.945,
        f"{report['n_events']:,} synthetic authorisations   |   reference: {stats.source}   |   "
        f"amounts in {stats.currency}",
        color=_INK_2,
        fontsize=9.5,
        ha="left",
    )

    # -- 1. amount -------------------------------------------------------------- #
    ax = axes[0][0]
    amount = frame["amount"].to_numpy(dtype=float)
    log_amount = np.log10(np.maximum(amount, 1.0))
    grid = np.linspace(0.0, 5.6, 400)
    ax.hist(
        log_amount,
        bins=90,
        range=(0.0, 5.6),
        density=True,
        color=_SYNTH,
        alpha=0.85,
        label="synthetic",
    )
    ax.plot(
        grid,
        _mixture_log10_pdf(stats, grid),
        color=_REF,
        linewidth=2.0,
        linestyle="--",
        label="reference mixture",
    )
    _style_axes(ax, "Amount distribution", f"amount ({stats.currency}, log scale)", "density")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["10", "100", "1k", "10k", "100k"])
    ax.legend(frameon=False, fontsize=8.5, labelcolor=_INK_2, loc="upper left")
    _note(ax, f"KS = {report['amount_ks_distance']:.4f}")

    # -- 2. hour of day ---------------------------------------------------------- #
    ax = axes[0][1]
    hours = frame["ts"].dt.hour.to_numpy()
    hour_synth = np.bincount(hours, minlength=24).astype(float)
    hour_synth /= hour_synth.sum()
    hour_ref = np.asarray(stats.hour_weights, dtype=float)
    blend = stats.agentic_hour_uniform_blend
    agent_ref = (1.0 - blend) * hour_ref + blend / 24.0
    agent_ref /= agent_ref.sum()
    share = report["agentic_share"]
    hour_target = (1.0 - share) * hour_ref + share * agent_ref

    # Synthetic first, reference dashed on top: they overlap almost exactly, and
    # whichever is drawn second is the only one you can see.
    ax.plot(
        range(24),
        hour_synth,
        color=_SYNTH,
        linewidth=2.4,
        marker="o",
        markersize=5,
        label="synthetic",
        zorder=2,
    )
    ax.plot(
        range(24),
        hour_target,
        color=_REF,
        linewidth=1.8,
        linestyle=(0, (5, 3)),
        marker="o",
        markersize=5,
        markerfacecolor=_SURFACE,
        markeredgewidth=1.4,
        label="reference",
        zorder=3,
    )
    _style_axes(ax, "Hour-of-day activity (IST)", "hour", "share of volume")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(frameon=False, fontsize=8.5, labelcolor=_INK_2, loc="upper left")
    _note(ax, f"total variation = {report['hour_total_variation']:.4f}", (0.98, 0.05))

    # -- 3. per-MCC median ticket -------------------------------------------------- #
    ax = axes[0][2]
    top = sorted(stats.mcc_profiles, key=lambda p: -p.weight)[:_TOP_MCCS]
    y = np.arange(len(top))
    ref_median = np.array([math.exp(p.log_amount_mu) for p in top])
    syn_median = frame.groupby("mcc")["amount"].median()
    got = np.array([float(syn_median.get(p.mcc, np.nan)) for p in top])
    ax.hlines(y, ref_median, got, color=_GRID, linewidth=1.6, zorder=1)
    ax.scatter(
        ref_median,
        y,
        s=46,
        facecolors="none",
        edgecolors=_REF,
        linewidths=1.8,
        label="reference",
        zorder=2,
    )
    ax.scatter(got, y, s=34, color=_SYNTH, label="synthetic", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{p.mcc}  {p.label[:16].rstrip(' &')}" for p in top], fontsize=7.5)
    ax.set_xscale("log")
    ax.invert_yaxis()
    _style_axes(ax, "Median ticket by category", f"median amount ({stats.currency})")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=_INK_2, loc="upper left")
    _note(ax, f"max |rel delta| = {report['mcc_median_max_abs_rel_delta']:.3f}", (0.98, 0.16))

    # -- 4. category mix ------------------------------------------------------------ #
    ax = axes[1][0]
    mcc_synth = frame["mcc"].value_counts(normalize=True)
    ref_share = np.array([p.weight for p in top])
    got_share = np.array([float(mcc_synth.get(p.mcc, 0.0)) for p in top])
    ax.hlines(y, ref_share, got_share, color=_GRID, linewidth=1.6, zorder=1)
    ax.scatter(
        ref_share,
        y,
        s=46,
        facecolors="none",
        edgecolors=_REF,
        linewidths=1.8,
        label="reference",
        zorder=2,
    )
    ax.scatter(got_share, y, s=34, color=_SYNTH, label="synthetic", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([p.mcc for p in top], fontsize=8)
    ax.invert_yaxis()
    _style_axes(ax, "Category volume mix", "share of events")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=_INK_2, loc="upper left")
    _note(ax, f"max |delta| = {report['mcc_mix_max_abs_delta']:.4f}", (0.98, 0.16))

    # -- 5. merchant popularity ------------------------------------------------------ #
    ax = axes[1][1]
    counts = np.sort(frame["merchant_id"].value_counts().to_numpy())[::-1]
    rank = np.arange(1, counts.size + 1)
    realised = report["zipf_exponent_realised"]
    ax.plot(rank, counts, color=_SYNTH, linewidth=2.4, label="synthetic", zorder=2)
    # Compare against the *realised* OLS fit, not the per-pool sampling exponent.
    # Merchants are drawn Zipf within a category-and-metro pool, so the national
    # curve is legitimately flatter than the pool exponent: a small grocer in
    # Guwahati holds real share of Guwahati volume however tiny it is nationally.
    # Plotting the pool exponent here would read as a calibration miss when it is
    # actually the locality model doing its job.
    fit = counts[0] * rank.astype(float) ** (-realised)
    ax.plot(
        rank,
        fit,
        color=_REF,
        linewidth=1.8,
        linestyle=(0, (5, 3)),
        label=f"OLS fit  -{realised:.2f}",
        zorder=3,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style_axes(ax, "Merchant popularity (rank-frequency)", "merchant rank", "transactions")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=_INK_2, loc="lower left")
    _note(
        ax,
        f"per-pool Zipf a = {stats.merchant_zipf_exponent:.2f}\n"
        f"national curve flatter: merchant\nchoice is locality-conditioned",
        (0.98, 0.97),
    )

    # -- 6. the agentic rail: modelled, no reference exists --------------------------- #
    ax = axes[1][2]
    agentic = frame[frame["channel"] == "agentic"]
    entropy = agentic["ag_cursor_entropy"].to_numpy(dtype=float)
    human = agentic["ag_human_present"].to_numpy(dtype=bool)
    bins = np.linspace(0, 6, 60)
    ax.hist(
        entropy[~human],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.0,
        color=_SYNTH,
        label="human_present = False",
    )
    ax.hist(
        entropy[human],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.0,
        color=_ACCENT_2,
        label="human_present = True",
    )
    _style_axes(ax, "Agentic telemetry: cursor entropy", "cursor entropy", "density")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=_INK_2, loc="upper right")
    ax.text(
        0.97,
        0.60,
        "MODELLED, NOT CALIBRATED\nNo public agentic-payments panel exists.\n"
        "This gap is what F1-09 has to forge.",
        transform=ax.transAxes,
        fontsize=8,
        color=_INK_2,
        va="top",
        ha="right",
        bbox={"facecolor": "#f4f1ea", "edgecolor": _GRID, "boxstyle": "round,pad=0.4"},
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.935))
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target


def main() -> None:
    """Regenerate the figure from the committed parquet, if one exists."""
    import pandas as pd

    from mantis.core.paths import POPULATION_PARQUET
    from mantis.foundry.base.reference import load_reference_stats

    if not POPULATION_PARQUET.is_file():
        print(f"no population at {POPULATION_PARQUET}")
        print("run: python -m mantis.foundry.base --n 200000 --seed 7")
        return

    stats = load_reference_stats()
    frame = pd.read_parquet(POPULATION_PARQUET)
    report = calibration_report(frame, stats)
    print(format_report(report, stats))
    written = plot_calibration(frame, stats, report)
    print(f"figure: {written}" if written else "matplotlib unavailable; metrics only")


if __name__ == "__main__":
    main()
