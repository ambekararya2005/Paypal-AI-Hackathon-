"""Adversarial audit of the generated population. Assume it is wrong; try to prove it.

    python scripts/audit_population.py [--parquet PATH] [--quick]

This runs *before* injectors, because every downstream number inherits whatever
is wrong here. The bar is not "looks plausible" but "I tried to break it and
could not". Five checks, each printing PASS or FAIL:

1. **Determinism.** Same seed twice, identical frames. Different seed, different
   frames. Re-run under several ``PYTHONHASHSEED`` values, because a set of
   strings iterates in hash order and CPython randomises that per process --
   a real bug this audit caught on its first run.
2. **Agentic realism.** Legitimate agentic traffic must carry *near-miss*
   variation: mandate ages spread across their TTLs, mixed mandate types, varied
   provenance lengths, delegation depth above zero, and a scope ceiling that is
   not a fixed multiple of the amount. Uniform legitimate traffic would make
   every L0 rule fire at exactly 0% on legit and 100% on attack, which turns the
   "near-zero false positive" claim into an artefact of the generator.
3. **No accidental separator.** Best achievable single-column AUC against the
   agentic flag. Anything above the threshold that is not definitionally agentic
   means rail identity is leaking into an unrelated field. A random 1% label is
   scored the same way, as a null baseline.
4. **Distribution sanity.** Per-MCC median/p95/skew against the reference, a
   real diurnal curve rather than uniform, and a Zipf-ish merchant tail.
5. **Entity coherence.** Stable card/device maps with bounded churn, and geo
   clustered near each customer's home point.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mantis.core.events import LABEL_COLUMNS  # noqa: E402
from mantis.core.paths import POPULATION_PARQUET  # noqa: E402
from mantis.foundry.base.entities import MAX_AGENTS, MAX_CARDS, MAX_DEVICES  # noqa: E402
from mantis.foundry.base.reference import load_reference_stats  # noqa: E402

# Three tiers, because "any column above 0.7 is a leak" is the wrong test for a
# rail that announces itself. `channel == "agentic"` literally names the rail and
# every `ag_*` column is non-null exactly on it, so no amount of care makes the
# rail unknowable. What matters is whether a column separates *more sharply than
# its stated causal story allows* -- that is the signature of a generator
# artefact, and each tier below carries the reason it sits where it does.

#: Tier 1: the rail is self-announcing here. A perfect AUC is correct.
#: ``device_id`` belongs here on reflection: a hosted agent runtime genuinely is
#: a distinct device that only ever transacts agentically, and surfacing that is
#: precisely what Know-Your-Agent exists to do. Modelling it away would describe
#: a worse world, not a more honest one.
DEFINITIONAL: Final[frozenset[str]] = frozenset(
    {"channel", "entry_mode", "terminal_id", "device_id"}
)

#: Tier 2: correlated for a stated reason. Bounded so we notice if one goes
#: *perfect*, which would mean the reason stopped being the mechanism.
#:   ip             -- a function of customer and device
#:   customer_id    -- agent adoption genuinely varies by person
#:   threeds_result -- the authentication path differs by rail
#:   mcc, merchant_id -- agentic_affinity is deliberate: agents book flights, not fuel
#:   lat, lon       -- remote rails carry geo less often
BY_DESIGN: Final[frozenset[str]] = frozenset(
    {"ip", "customer_id", "threeds_result", "mcc", "merchant_id", "lat", "lon"}
)
BY_DESIGN_AUC: Final[float] = 0.85

#: Tier 3: nothing about the rail should be readable from these at all.
LEAK_AUC: Final[float] = 0.70

_results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    """Log one check outcome and print it as it happens."""
    _results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _digest(frame: pd.DataFrame) -> str:
    """Order-sensitive digest of a frame, ignoring list-valued columns."""
    import hashlib

    flat = frame.drop(columns=[c for c in frame.columns if frame[c].map(type).eq(list).any()])
    return hashlib.sha256(
        pd.util.hash_pandas_object(flat, index=False).values.tobytes()
    ).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# 1. Determinism
# --------------------------------------------------------------------------- #

_SUBPROCESS_PROBE: Final[str] = """
import sys, hashlib
sys.path.insert(0, {root!r})
import pandas as pd
from mantis.foundry.base.simulator import SimulationConfig, simulate_frame
d = simulate_frame(SimulationConfig(n_events=4000, seed=7, n_customers=500, n_merchants=1200))
d = d.drop(columns=[c for c in d.columns if d[c].map(type).eq(list).any()])
print(hashlib.sha256(pd.util.hash_pandas_object(d, index=False).values.tobytes()).hexdigest()[:16])
"""


def check_determinism(quick: bool) -> None:
    head("1. DETERMINISM")
    from mantis.foundry.base.simulator import SimulationConfig, simulate_frame

    cfg: dict[str, int] = {"n_events": 4_000, "n_customers": 500, "n_merchants": 1_200}
    a = simulate_frame(SimulationConfig(seed=7, **cfg))
    b = simulate_frame(SimulationConfig(seed=7, **cfg))
    c = simulate_frame(SimulationConfig(seed=8, **cfg))

    try:
        pd.testing.assert_frame_equal(a, b)
        same = True
    except AssertionError:
        same = False
    record("same seed -> identical frame", same, f"digest {_digest(a)}")
    record("different seed -> different frame", _digest(a) != _digest(c), f"seed 8 -> {_digest(c)}")

    if quick:
        record("stable across PYTHONHASHSEED", True, "skipped (--quick)")
        return

    # The hash-order trap: iterating a set of strings is not stable between
    # processes, and any loop that consumes the RNG inherits that instability.
    probe = _SUBPROCESS_PROBE.format(root=str(REPO_ROOT))
    digests: list[str] = []
    for hash_seed in ("0", "1", "12345"):
        out = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            check=True,
        )
        digests.append(out.stdout.strip())
    record(
        "stable across PYTHONHASHSEED",
        len(set(digests)) == 1,
        f"{len(set(digests))} distinct digest(s) over 3 hash seeds",
    )


# --------------------------------------------------------------------------- #
# 2. Is the agentic share real?
# --------------------------------------------------------------------------- #


def check_agentic_realism(df: pd.DataFrame) -> None:
    head("2. AGENTIC REALISM (near-miss variation, not a uniform block)")
    ag = df[df["channel"] == "agentic"]
    share = len(ag) / len(df)
    record("agentic share in 12-18%", 0.12 <= share <= 0.18, f"{share:.2%}")

    mix = ag["ag_mandate_type"].value_counts(normalize=True)
    print(f"       mandate_type mix: {mix.round(4).to_dict()}")
    record(
        "mandate_type is a genuine mix",
        len(mix) >= 3 and mix.max() < 0.80 and mix.min() > 0.02,
        f"{len(mix)} types, largest share {mix.max():.2%}",
    )

    # Mandate age as a fraction of its own TTL. All-near-zero would mean every
    # legitimate mandate is freshly minted, so an "almost expired" rule would
    # never fire on legit traffic and its false-positive rate would be fiction.
    age = (ag["ts"] - ag["ag_mandate_issued_ts"]).dt.total_seconds()
    frac = (age / ag["ag_mandate_ttl_seconds"]).to_numpy(dtype=float)
    q = np.percentile(frac, [5, 25, 50, 75, 95])
    print(f"       mandate age / TTL   p5/25/50/75/95: {np.round(q, 3).tolist()}")
    record(
        "mandate ages spread across TTL",
        q[0] < 0.10 and q[4] > 0.60 and float(np.std(frac)) > 0.15,
        f"p95 {q[4]:.2f}, sd {np.std(frac):.3f}",
    )
    record("no legit mandate used past its TTL", bool((frac < 1.0).all()), f"max {frac.max():.3f}")

    plen = ag["ag_provenance_chain"].map(len)
    print(f"       provenance length   counts: {plen.value_counts().sort_index().to_dict()}")
    record("provenance length varies", plen.nunique() >= 4, f"{plen.nunique()} distinct lengths")

    depth = ag["ag_delegation_depth"].astype(int)
    print(f"       delegation_depth    counts: {depth.value_counts().sort_index().to_dict()}")
    record(
        "delegation_depth varies and is >= 1",
        depth.nunique() >= 2 and int(depth.min()) >= 1,
        f"min {depth.min()}, max {depth.max()}",
    )

    ratio = (ag["amount"] / ag["ag_scope_max_amount"]).to_numpy(dtype=float)
    qr = np.percentile(ratio, [1, 25, 50, 75, 99])
    print(f"       amount/scope_max    p1/25/50/75/99: {np.round(qr, 3).tolist()}")
    record(
        "scope ceiling is not a fixed multiple",
        float(np.std(ratio)) > 0.05 and qr[4] > 0.90,
        f"sd {np.std(ratio):.3f}, p99 {qr[4]:.3f}, max {ratio.max():.3f}",
    )
    record(
        "no legit spend over its own ceiling", bool((ratio <= 1.0).all()), f"max {ratio.max():.4f}"
    )

    # The messy tails that stop L0 being a perfect oracle on legitimate traffic.
    kya = 1.0 - float(ag["ag_kya_registered"].mean())
    consent = 1.0 - float(ag["ag_consent_sig_valid"].mean())
    print(f"       legit unregistered KYA {kya:.2%}, invalid consent sig {consent:.2%}")
    record("legit KYA-unregistered tail exists", 0.005 < kya < 0.10, f"{kya:.2%}")
    record("legit invalid-consent tail exists", 0.0005 < consent < 0.02, f"{consent:.2%}")

    hp = ag.groupby("ag_human_present")["ag_cursor_entropy"].median()
    record(
        "human_present telemetry gap is real",
        len(hp) == 2 and float(hp.max()) / float(hp.min()) > 2.0,
        f"cursor entropy median {hp.round(3).to_dict()}",
    )


# --------------------------------------------------------------------------- #
# 3. Accidental separator hunt
# --------------------------------------------------------------------------- #


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    """Direction-agnostic AUC: a perfectly inverted column leaks just as hard."""
    from sklearn.metrics import roc_auc_score

    ok = np.isfinite(score)
    if ok.sum() < 100 or len(np.unique(y[ok])) < 2 or len(np.unique(score[ok])) < 2:
        return 0.5
    auc = float(roc_auc_score(y[ok], score[ok]))
    return max(auc, 1.0 - auc)


def _best_single_column_auc(col: pd.Series, y: np.ndarray) -> tuple[float, str]:
    """Strongest AUC any single-column probe can extract. Deliberately generous.

    Three probes: a depth-1 stump on numerics, out-of-fold target encoding for
    categoricals (out-of-fold so a unique-per-row id cannot fake a perfect score
    by memorising itself), and a plain is-null indicator -- which is how a column
    leaks a rail without leaking a value.
    """
    from sklearn.tree import DecisionTreeClassifier

    best, how = 0.5, "-"

    null = col.isna().to_numpy()
    if 0 < int(null.sum()) < len(col):
        auc = _auc(y, null.astype(float))
        if auc > best:
            best, how = auc, "is-null"

    values = col.dropna()
    if values.empty:
        return best, how

    first = values.iloc[0]
    if isinstance(first, (list, np.ndarray)):
        length = col.map(lambda v: len(v) if isinstance(v, (list, np.ndarray)) else np.nan)
        auc = _auc(y, np.nan_to_num(length.to_numpy(dtype=float), nan=-1.0))
        return (auc, "list-length") if auc > best else (best, how)

    if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
        x = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
        fill = float(np.nanmin(x)) - 1.0 if np.isfinite(x).any() else 0.0
        x = np.nan_to_num(x, nan=fill).reshape(-1, 1)
        stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(x, y)
        auc = _auc(y, stump.predict_proba(x)[:, 1])
        return (auc, "depth-1 stump") if auc > best else (best, how)

    # Out-of-fold target encoding. Two folds is enough to kill the memorisation
    # artefact that would otherwise give every unique id a perfect score.
    codes = col.astype("string").fillna("<NA>").astype("category").cat.codes.to_numpy()
    score = np.full(len(codes), np.nan)
    half = len(codes) // 2
    for fit_idx, apply_idx in (
        (np.arange(half), np.arange(half, len(codes))),
        (np.arange(half, len(codes)), np.arange(half)),
    ):
        rate = pd.Series(y[fit_idx]).groupby(codes[fit_idx]).mean()
        score[apply_idx] = pd.Series(codes[apply_idx]).map(rate).to_numpy(dtype=float)
    auc = _auc(y, np.nan_to_num(score, nan=float(y.mean())))
    return (auc, "OOF target-encode") if auc > best else (best, how)


def _tier(col: str) -> str:
    """Which expectation tier a column falls into. See the constants above."""
    if col.startswith("ag_") or col in DEFINITIONAL:
        return "definitional"
    return "by-design" if col in BY_DESIGN else "neutral"


def _rank_columns(df: pd.DataFrame, y: np.ndarray, cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in cols:
        auc, how = _best_single_column_auc(df[col], y)
        rows.append(
            {
                "column": col,
                "auc": round(auc, 4),
                "probe": how,
                "kind": _tier(col),
            }
        )
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def check_separators(df: pd.DataFrame) -> None:
    head("3. ACCIDENTAL SEPARATOR HUNT")
    for label in LABEL_COLUMNS:
        assert df[label].nunique(dropna=False) == 1, f"{label} varies in a legitimate population"
    print(f"       label columns constant as expected: {', '.join(LABEL_COLUMNS)}")

    cols = [c for c in df.columns if c not in LABEL_COLUMNS]
    y = (df["channel"] == "agentic").to_numpy().astype(int)
    ranked = _rank_columns(df, y, cols)

    print("\n       vs is_agentic -- top 12 by best single-column AUC:")
    print(ranked.head(12).to_string(index=False))

    neutral = ranked[(ranked["kind"] == "neutral") & (ranked["auc"] > LEAK_AUC)]
    record(
        f"neutral columns stay below AUC {LEAK_AUC}",
        neutral.empty,
        "clean" if neutral.empty else f"leaking: {neutral['column'].tolist()}",
    )
    designed = ranked[(ranked["kind"] == "by-design") & (ranked["auc"] > BY_DESIGN_AUC)]
    record(
        f"by-design columns stay below AUC {BY_DESIGN_AUC}",
        designed.empty,
        "clean"
        if designed.empty
        else f"sharper than their story allows: {designed['column'].tolist()}",
    )

    # The check that actually protects Day 2. Rail identity is unhideable, so the
    # thing that stops a fraud model cheating is that the *shared* columns behave
    # the same on both rails. If agentic amounts drifted away from classic ones
    # within a category, then once attacks land -- and attacks skew agentic --
    # the model would learn "unusual amount => agentic => fraud" and report a
    # recall that is really just rail detection wearing a costume.
    ag_mask = df["channel"] == "agentic"
    worst_mcc, worst_ks = "-", 0.0
    for mcc, grp in df.groupby("mcc"):
        left = grp.loc[grp["channel"] == "agentic", "amount"].to_numpy()
        right = grp.loc[grp["channel"] != "agentic", "amount"].to_numpy()
        if len(left) < 200 or len(right) < 200:
            continue
        from scipy.stats import ks_2samp

        stat = float(ks_2samp(left, right).statistic)
        if stat > worst_ks:
            worst_mcc, worst_ks = str(mcc), stat
    record(
        "amount is rail-independent given MCC",
        worst_ks < 0.08,
        f"worst per-MCC KS {worst_ks:.4f} (mcc {worst_mcc})",
    )
    hour_ks = float(
        np.abs(
            np.bincount(df.loc[ag_mask, "ts"].dt.hour, minlength=24) / int(ag_mask.sum())
            - np.bincount(df.loc[~ag_mask, "ts"].dt.hour, minlength=24) / int((~ag_mask).sum())
        ).sum()
        * 0.5
    )
    record(
        "agentic hour curve differs only as modelled",
        0.02 < hour_ks < 0.30,
        f"TV(agentic, classic) = {hour_ks:.4f} -- agents do not sleep, by design",
    )

    rng = np.random.default_rng(1337)
    y_null = (rng.random(len(df)) < 0.01).astype(int)
    ranked_null = _rank_columns(df, y_null, cols)
    print("\n       vs random 1% label (null baseline) -- top 5:")
    print(ranked_null.head(5).to_string(index=False))
    record(
        "null-label baseline stays near chance",
        float(ranked_null["auc"].max()) < 0.60,
        f"max AUC {ranked_null['auc'].max():.4f}",
    )


# --------------------------------------------------------------------------- #
# 4. Distribution sanity
# --------------------------------------------------------------------------- #


def check_distributions(df: pd.DataFrame) -> None:
    head("4. DISTRIBUTION SANITY")
    from scipy.stats import skew

    stats = load_reference_stats()
    ref = {p.mcc: (math.exp(p.log_amount_mu), p.label) for p in stats.mcc_profiles}

    rows: list[dict[str, Any]] = []
    for mcc, grp in df.groupby("mcc"):
        target, label = ref[str(mcc)]
        median = float(grp["amount"].median())
        rows.append(
            {
                "mcc": mcc,
                "label": label[:26],
                "n": len(grp),
                "median": round(median, 1),
                "ref": round(target, 1),
                "rel": round(median / target - 1.0, 3),
                "p95": round(float(grp["amount"].quantile(0.95)), 1),
                "skew": round(float(skew(grp["amount"])), 2),
                "log_skew": round(float(skew(np.log(grp["amount"]))), 2),
            }
        )
    table = pd.DataFrame(rows).sort_values("n", ascending=False)
    print(table.to_string(index=False))

    absurd = table[(table["median"] < 20) | (table["median"] > 100_000)]
    record(
        "no implausible per-MCC median",
        absurd.empty,
        "all medians in INR 20-100k" if absurd.empty else str(absurd["mcc"].tolist()),
    )
    drift = table[table["rel"].abs() > 0.25]
    record(
        "per-MCC median within 25% of reference",
        drift.empty,
        f"max |rel| {table['rel'].abs().max():.3f}",
    )
    record(
        "amounts right-skewed, log-normal-ish",
        bool((table["skew"] > 1.0).all()) and bool((table["log_skew"].abs() < 1.0).all()),
        f"min raw skew {table['skew'].min():.2f}, "
        f"max |log skew| {table['log_skew'].abs().max():.2f}",
    )

    hour = df["ts"].dt.hour.value_counts(normalize=True).sort_index().to_numpy()
    tv_uniform = 0.5 * float(np.abs(hour - 1.0 / 24.0).sum())
    record(
        "hour-of-day is diurnal, not uniform",
        tv_uniform > 0.15,
        f"TV from uniform {tv_uniform:.4f}, peak/trough {hour.max() / hour.min():.1f}x",
    )

    counts = np.sort(df["merchant_id"].value_counts().to_numpy())[::-1]
    resolved = counts[counts >= 5][:2000]
    log_rank = np.log(np.arange(1, len(resolved) + 1))
    log_freq = np.log(resolved)
    slope, intercept = np.polyfit(log_rank, log_freq, 1)
    resid = log_freq - (slope * log_rank + intercept)
    r2 = 1.0 - float(np.var(resid) / np.var(log_freq))
    record(
        "merchant popularity is Zipf-ish",
        0.6 < -slope < 1.6 and r2 > 0.90,
        f"exponent {-slope:.3f}, R2 {r2:.4f} over {len(resolved)} resolved ranks",
    )


# --------------------------------------------------------------------------- #
# 5. Entity coherence
# --------------------------------------------------------------------------- #


def check_entities(df: pd.DataFrame) -> None:
    head("5. ENTITY COHERENCE")
    per = df.groupby("customer_id").agg(
        n=("event_id", "size"),
        cards=("card_bin", "nunique"),
        devices=("device_id", "nunique"),
        merchants=("merchant_id", "nunique"),
    )
    print(f"       per-customer events  p50 {per['n'].median():.0f}  max {per['n'].max()}")
    print(f"       distinct cards       max {per['cards'].max()}  (cap {MAX_CARDS})")
    print(
        f"       distinct devices     max {per['devices'].max()}  "
        f"(cap {MAX_DEVICES} personal + {MAX_AGENTS} agent)"
    )
    record(
        "cards per customer within cap",
        int(per["cards"].max()) <= MAX_CARDS,
        f"max {per['cards'].max()}",
    )
    record(
        "devices per customer within cap",
        int(per["devices"].max()) <= MAX_DEVICES + MAX_AGENTS,
        f"max {per['devices'].max()} -- no customer with 400 devices",
    )

    # Churn: a portfolio where everyone uses exactly one card forever is as
    # unrealistic as one where everyone uses forty.
    share_multi = float((per["cards"] > 1).mean())
    record(
        "card churn is realistic",
        0.20 < share_multi < 0.80,
        f"{share_multi:.1%} of customers use more than one card",
    )

    geo = df.dropna(subset=["lat", "lon"]).copy()
    home = (
        geo.groupby("customer_id")[["lat", "lon"]]
        .median()
        .rename(columns={"lat": "hlat", "lon": "hlon"})
    )
    geo = geo.join(home, on="customer_id")
    dlat = (geo["lat"] - geo["hlat"]) * 111.0
    dlon = (geo["lon"] - geo["hlon"]) * 111.0 * np.cos(np.radians(geo["hlat"]))
    dist = np.sqrt(dlat**2 + dlon**2)
    q = np.percentile(dist, [50, 90, 99])
    far = float((dist > 200).mean())
    print(f"       km from home point   p50 {q[0]:.1f}  p90 {q[1]:.1f}  p99 {q[2]:.1f}")
    record(
        "geo clusters near home",
        q[0] < 15 and q[1] < 60,
        f"p50 {q[0]:.1f} km, p90 {q[1]:.1f} km",
    )
    record(
        "travel tail small but present",
        0.005 < far < 0.10,
        f"{far:.2%} of events beyond 200 km",
    )

    ag = df[df["channel"] == "agentic"]
    per_agent = ag.groupby("ag_agent_id")["customer_id"].nunique()
    record(
        "each agent belongs to exactly one customer",
        int(per_agent.max()) == 1,
        f"max customers per agent {per_agent.max()}",
    )


def main() -> int:
    """Run the audit. Exit code 1 if any check failed, 2 if there is no population."""
    parser = argparse.ArgumentParser(description="Adversarial audit of the population.")
    parser.add_argument("--parquet", type=Path, default=POPULATION_PARQUET)
    parser.add_argument("--quick", action="store_true", help="skip the subprocess hash-seed sweep")
    args = parser.parse_args()

    if not args.parquet.is_file():
        print(f"no population at {args.parquet}")
        print("run: python -m mantis.foundry.base --n 200000 --seed 7")
        return 2

    df = pd.read_parquet(args.parquet)
    print(f"audit target: {args.parquet}  ({len(df):,} rows x {df.shape[1]} cols)")

    check_determinism(args.quick)
    check_agentic_realism(df)
    check_separators(df)
    check_distributions(df)
    check_entities(df)

    head("SUMMARY")
    width = max(len(name) for name, _, _ in _results)
    for name, ok, detail in _results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    failed = [name for name, ok, _ in _results if not ok]
    print(f"\n  {len(_results) - len(failed)}/{len(_results)} checks passed")
    if failed:
        print("  FAILED: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
