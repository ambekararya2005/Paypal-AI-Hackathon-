"""Fit calibration shape parameters from a reference transaction CSV.

    python scripts/fit_reference.py

Drop the Kaggle ``kartik2112/fraud-detection`` (Sparkov) CSVs into
``data/reference/`` and run this. It writes ``data/reference/reference_stats.json``,
which the foundry then prefers over its built-in Indian-market priors.

**If no CSV is there, this exits cleanly and changes nothing.** That is the
default state of a clean clone and it is not an error — HARD RULE 4 says the repo
runs with no Kaggle token, so the fitted path is an upgrade, never a dependency.

What is fitted, and what deliberately is not
--------------------------------------------
Sparkov is US data, and is itself synthetic. Some of its parameters transfer to
an Indian card population and some emphatically do not, so this script only
touches the first group:

===========================  ==========================================
Fitted from the CSV          Why it transfers
===========================  ==========================================
hour-of-day curve            Diurnal retail shape is a human constant,
                             and it is measured in local time either way.
day-of-week curve            Same.
per-category log-amount      Dispersion is scale-free: the *spread* of
sigma                        fuel spend relative to its own median does
                             not care what currency it is in.
merchant Zipf exponent       Rank-frequency structure of an acceptance
                             estate, dimensionless.
per-customer velocity        Transactions per cardholder per day.
session burst rate           Share of transactions following another
                             within the session window.
===========================  ==========================================

===========================  ==========================================
NOT fitted                   Why it does not transfer
===========================  ==========================================
MCC volume mix               Sparkov's category mix is a property of its
                             own generator and of the US market. Fitting
                             the Indian mix from it would be strictly
                             worse than the prior.
amount *location* (median)   Needs PPP and basket composition, not an FX
                             rate. Available behind
                             ``--fit-amount-location``, off by default,
                             and it warns when you use it.
geography, BINs, 3DS,        US geography, US issuers, US authentication
channel mix                  rules. None of it applies.
everything agentic           There is no agentic-payments panel to fit
                             against. That absence is the project.
===========================  ==========================================

Only ``is_fraud == 0`` rows are used. Calibrating the *legitimate* population
against a file that includes fraud would bake the thing we are trying to detect
into the background, and every downstream metric would quietly understate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

# Allow `python scripts/fit_reference.py` from a clone with no editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mantis.core.paths import REFERENCE_DIR, REFERENCE_STATS_JSON, ensure_dir
from mantis.foundry.base.reference import ReferenceStats

#: Sparkov spending categories mapped onto the MCCs the foundry actually models.
#: Several Sparkov categories collapse onto one MCC (``grocery_net`` and
#: ``grocery_pos`` are both 5411); amounts are pooled across them.
_CATEGORY_TO_MCC: Final[dict[str, str]] = {
    "entertainment": "7832",
    "food_dining": "5812",
    "gas_transport": "5541",
    "grocery_net": "5411",
    "grocery_pos": "5411",
    "health_fitness": "7997",
    "home": "5999",
    "kids_pets": "5945",
    "misc_net": "5999",
    "misc_pos": "5999",
    "personal_care": "5977",
    "shopping_net": "5651",
    "shopping_pos": "5311",
    "travel": "4722",
}

_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "is_fraud",
)

#: Two transactions by the same cardholder inside this window count as one
#: session, which is what ``burst_probability`` measures.
_SESSION_WINDOW_S: Final[int] = 1_500

_DEFAULT_FX: Final[float] = 83.0


def _find_csvs(directory: Path) -> list[Path]:
    """Reference CSVs, preferring the canonical Sparkov filenames."""
    if not directory.is_dir():
        return []
    named = [p for p in (directory / "fraudTrain.csv", directory / "fraudTest.csv") if p.is_file()]
    return named or sorted(directory.glob("*.csv"))


def _load(paths: list[Path]) -> pd.DataFrame:
    """Read only the columns we fit from, and keep only legitimate rows."""
    frames: list[pd.DataFrame] = []
    for path in paths:
        head = pd.read_csv(path, nrows=1)
        missing = [c for c in _REQUIRED_COLUMNS if c not in head.columns]
        if missing:
            raise ValueError(f"{path.name}: missing expected column(s) {missing}")
        frames.append(pd.read_csv(path, usecols=list(_REQUIRED_COLUMNS)))

    frame = pd.concat(frames, ignore_index=True)
    frame["ts"] = pd.to_datetime(frame["trans_date_trans_time"])
    legit = frame[frame["is_fraud"] == 0].copy()
    return legit.sort_values(["cc_num", "ts"], ignore_index=True)


def _fit_hour_dow(frame: pd.DataFrame) -> tuple[list[float], list[float]]:
    """Diurnal and weekly activity curves, normalised."""
    hour = np.bincount(frame["ts"].dt.hour.to_numpy(), minlength=24).astype(float)
    dow = np.bincount(frame["ts"].dt.dayofweek.to_numpy(), minlength=7).astype(float)
    return list(hour / hour.sum()), list(dow / dow.sum())


def _fit_amount_by_mcc(frame: pd.DataFrame) -> dict[str, tuple[float, float, int]]:
    """Log-amount ``(mu, sigma, n)`` per mapped MCC, pooling Sparkov categories."""
    frame = frame.copy()
    frame["mcc"] = frame["category"].map(_CATEGORY_TO_MCC)
    frame = frame[frame["mcc"].notna() & (frame["amt"] > 0)]
    log_amt = np.log(frame["amt"].to_numpy(dtype=float))

    out: dict[str, tuple[float, float, int]] = {}
    for mcc, idx in frame.groupby("mcc").indices.items():
        values = log_amt[idx]
        if values.size < 200:
            continue
        out[str(mcc)] = (float(values.mean()), float(values.std(ddof=1)), int(values.size))
    return out


def _fit_zipf(frame: pd.DataFrame, min_count: int = 5, max_rank: int = 2_000) -> float:
    """Merchant rank-frequency exponent by OLS on the resolved head."""
    counts = np.sort(frame["merchant"].value_counts().to_numpy())[::-1]
    head = counts[: min(max_rank, counts.size)]
    head = head[head >= min_count]
    if head.size < 20:
        return float("nan")
    rank = np.log(np.arange(1, head.size + 1))
    return float(-np.polyfit(rank, np.log(head), 1)[0])


def _fit_velocity(frame: pd.DataFrame) -> tuple[float, float]:
    """Per-customer daily rate as a gamma ``(shape, mean)``, by moments."""
    span_days = max((frame["ts"].max() - frame["ts"].min()).days, 1)
    per_customer = frame.groupby("cc_num").size().to_numpy(dtype=float) / span_days
    mean = float(per_customer.mean())
    var = float(per_customer.var(ddof=1))
    shape = mean * mean / var if var > 0 else 1.6
    return float(np.clip(shape, 0.2, 20.0)), mean


def _fit_burst(frame: pd.DataFrame) -> float:
    """Share of transactions that follow the same cardholder's previous one closely."""
    same = frame["cc_num"].to_numpy()[1:] == frame["cc_num"].to_numpy()[:-1]
    gap = np.diff(frame["ts"].to_numpy().astype("datetime64[s]").astype(np.int64))
    return float((same & (gap > 0) & (gap <= _SESSION_WINDOW_S)).mean())


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python scripts/fit_reference.py",
        description="Fit calibration shape parameters from a reference CSV.",
    )
    parser.add_argument("--input-dir", type=Path, default=REFERENCE_DIR)
    parser.add_argument("--out", type=Path, default=REFERENCE_STATS_JSON)
    parser.add_argument(
        "--fx",
        type=float,
        default=_DEFAULT_FX,
        help=f"INR per USD, only used with --fit-amount-location (default {_DEFAULT_FX})",
    )
    parser.add_argument(
        "--fit-amount-location",
        action="store_true",
        help="also move per-MCC median tickets by FX. Off by default: converting US "
        "ticket sizes to Indian ones needs PPP and basket composition, not a spot rate.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Fit and write, or explain what is missing and exit cleanly."""
    args = _parse_args(argv)
    paths = _find_csvs(args.input_dir)

    if not paths:
        print("No reference CSV found. Using built-in Indian-market priors -- nothing to do.")
        print()
        print(f"  looked in : {args.input_dir}")
        print("  wanted    : fraudTrain.csv / fraudTest.csv (or any *.csv)")
        print("  source    : Kaggle dataset 'kartik2112/fraud-detection' (Sparkov)")
        print()
        print("This is the normal state of a clean clone. The foundry runs either way:")
        print("  python -m mantis.foundry.base --n 200000 --seed 7")
        return 0

    print(f"reading {len(paths)} file(s): {', '.join(p.name for p in paths)}")
    try:
        frame = _load(paths)
    except (ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Expected the Kaggle 'kartik2112/fraud-detection' schema.", file=sys.stderr)
        return 1

    print(
        f"  {len(frame):,} legitimate rows "
        f"({frame['ts'].min():%Y-%m-%d} to {frame['ts'].max():%Y-%m-%d})"
    )
    print()

    stats = ReferenceStats()
    hour, dow = _fit_hour_dow(frame)
    amounts = _fit_amount_by_mcc(frame)
    zipf = _fit_zipf(frame)
    shape, rate_mean = _fit_velocity(frame)
    burst = _fit_burst(frame)

    fitted_mccs: list[str] = []
    for profile in stats.mcc_profiles:
        got = amounts.get(profile.mcc)
        if got is None:
            continue
        mu, sigma, _n = got
        profile.log_amount_sigma = sigma
        if args.fit_amount_location:
            profile.log_amount_mu = mu + math.log(args.fx)
        fitted_mccs.append(profile.mcc)

    stats.hour_weights = hour
    stats.dow_weights = dow
    stats.merchant_zipf_exponent = zipf
    stats.customer_rate_gamma_shape = shape
    stats.customer_rate_mean_per_day = rate_mean
    stats.burst_probability = burst
    stats.source = f"fitted:{'+'.join(p.stem for p in paths)}"

    kept = sorted({p.mcc for p in stats.mcc_profiles} - set(fitted_mccs))
    stats.provenance = {
        "fitted_from": ", ".join(p.name for p in paths),
        "rows_used": f"{len(frame)} rows with is_fraud == 0",
        "fitted": "hour curve, day-of-week curve, merchant Zipf exponent, per-customer "
        "velocity (gamma), session burst rate, per-MCC log-amount sigma.",
        "amount_location": (
            f"FITTED via FX {args.fx} INR/USD -- treat with suspicion, an FX rate is not PPP."
            if args.fit_amount_location
            else "NOT fitted; per-MCC median tickets remain Indian-market priors."
        ),
        "mcc_sigma_fitted": ", ".join(fitted_mccs) or "none",
        "mcc_sigma_kept_as_prior": ", ".join(kept) or "none",
        "not_fitted": "MCC volume mix, geography, card BINs, 3DS outcomes, channel mix "
        "(all US-specific), and the entire agentic block (no panel exists).",
    }

    ensure_dir(args.out.parent)
    args.out.write_text(
        json.dumps(stats.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )

    summary: dict[str, Any] = {
        "hour peak": f"{int(np.argmax(hour)):02d}:00 at {max(hour):.4f} share",
        "zipf exponent": f"{zipf:.3f}",
        "velocity": f"mean {rate_mean:.3f}/day, gamma shape {shape:.2f}",
        "burst rate": f"{burst:.4f}",
        "mcc sigmas fitted": f"{len(fitted_mccs)} of {len(stats.mcc_profiles)}",
    }
    for key, value in summary.items():
        print(f"  {key:<20} {value}")
    print()
    print(f"wrote {args.out}")
    print(f"  MCCs kept at prior sigma: {', '.join(kept) or 'none'}")
    print("  the foundry will now prefer this file; delete it to fall back to priors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
