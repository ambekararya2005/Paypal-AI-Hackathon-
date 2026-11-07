"""End-to-end per-event scoring latency through the full fused stack.

    python scripts/latency_bench.py                  # (make latency)
    python scripts/latency_bench.py --n 1000 --budget 50

Criterion 5 is real-world feasibility, and ``RESULTS.md`` currently ends with an
admission: *"Latency is not measured here. The feature pass is 0.052 ms/row and
the graph pass 0.021 ms/row, but an end-to-end p99 for the whole firewall is a
Day 7 number and is not quoted before it is measured."* This script is that
measurement.

The budget is **50 ms**. That is roughly what an issuer's authorisation host has
to spare inside a Mastercard timeout after the network, the HSM and the account
lookup have taken their share; a scorer that misses it does not get to decline
the transaction, it gets bypassed.

Two paths, because a firewall is not one deadline
--------------------------------------------------
Not every layer has to answer inside the authorisation window, and pretending
they do is what makes the budget look unmeetable.

**Inline (L0 + L1).** What must return before the issuer answers the acquirer,
because it is what changes the *authorisation decision*: the deterministic
protocol clauses, and the supervised score they escalate to. Both need only the
event and backward-looking state.

**Fast-follow (L3 text, L4 graph, fusion, policy).** What informs an action taken
**after** the authorisation has been answered. The two actions that matter on the
agentic rail are both post-authorisation in any real deployment:

* **mandate revocation** — a mandate that has been used to read a poisoned page
  is revoked so the *next* authorisation under it fails L0. Revocation is a
  credential-lifecycle operation against the mandate registry, not a field in
  the authorisation response.
* **agent quarantine** — an agent whose identity component has fused with a ring
  is suspended, which changes every future authorisation it attempts and none of
  the one in flight.

Both are seconds-to-minutes operations by nature. Holding an authorisation open
while a TF-IDF classifier reads eleven web pages, to take an action that cannot
be applied to that authorisation anyway, would be an architectural error rather
than a performance one.

So this script reports **both** paths, separately, and the inline one is the
number the 50 ms budget actually governs. The full-stack number stays in the
document because a reader is entitled to know what everything costs.

What is measured, precisely
----------------------------
**One event at a time, against warm state.** Not a batch divided by its row
count. The distinction matters: a batch of 300,000 amortises every per-call cost
in Python, pandas and LightGBM across 300,000 rows, and an issuer scoring one
authorisation gets none of that amortisation. Batch-throughput numbers are the
usual way a latency claim turns out to be false in production, so this harness
calls every stage with exactly one event:

===========================  ===========================================
stage                        how it is called
===========================  ===========================================
velocity                     ``RollingStore.observe`` - already the
                             per-event API the offline pass uses
graph (L4)                   ``EntityGraph.observe`` - likewise
transaction / entity /       their real functions, on a one-row frame
mandate features
L0 clauses                   ``l0.evaluate`` on a one-row frame
L1 (LightGBM)                ``score`` on a one-row matrix
L2 (isolation forest)        ``score`` on a one-row matrix
L3 (text)                    ``score`` on a one-row frame
fusion + policy              on one event's four layer scores
===========================  ===========================================

The state is **warmed first** by replaying the whole training window through the
velocity store and the graph, so the measured events are scored against the
history an issuer would actually hold: populated dictionaries, real eviction
work, real component sizes. Timing against empty stores would report the cost of
the easy case.

What is *not* in the number, stated so it is not over-claimed
--------------------------------------------------------------
The glue that concatenates the five feature blocks into one row vector is not
timed separately; it is array assembly measured in microseconds, and it is the
only part of the path this harness does not put a clock on. Everything else -
every model, every clause, both stateful stores - is timed on a genuine
single-event call.

Nor is this a claim about a production implementation. It is Python, pandas and
scikit-learn, and a real deployment would call LightGBM's C++ predictor on a
float array without a DataFrame in sight. That makes this number an **upper
bound** on what the same model costs, which is the useful direction for a budget
claim to err in.

Both numbers, because one of them alone is misleading
------------------------------------------------------
The harness therefore also times the identical stages **in batch** and reports
the per-row cost beside the single-event cost. The ratio between them is the
per-call overhead, and printing it is what stops this section from being read as
either of the two available lies:

* Quoting only the **batch** number ("0.06 ms per event!") is the lie of
  omission. No issuer scores 100,000 authorisations at once.
* Quoting only the **single-event** number, without saying that two stages spend
  their time in ``pandas.Series.map`` overhead that scales with the size of a
  lookup table rather than with the work, invites the reader to conclude the
  *model* is too slow. It is not; the calling convention is.

The measured example: ``entity_features`` costs about 46 ms on a one-row frame
and **0.0022 ms per row** in batch, a factor of twenty thousand. The cause is
that ``Series.map(dict)`` materialises the dictionary into an index on every
call, so a fourteen-feature block against profile tables of several thousand
entries pays that cost fourteen times to look up fourteen values. The named fix
is a plain dictionary lookup on the single-event path, and it is **not** applied
here: the feature builder is shared with the offline pass that produced every
pinned number in ``RESULTS.md``, and three days out this project records a
finding rather than re-rolls a table for it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # so the script runs without an editable install
    sys.path.insert(0, str(REPO_ROOT))

from mantis.core.paths import GENERATED_DIR, ensure_dir  # noqa: E402
from mantis.defense.experiment import TRAIN_SHARE  # noqa: E402
from mantis.defense.features import FeatureBuilder  # noqa: E402
from mantis.defense.features.spec import VELOCITY_KEYS  # noqa: E402
from mantis.defense.features.state import RollingStore, as_epoch  # noqa: E402
from mantis.defense.features.transaction import transaction_features  # noqa: E402
from mantis.defense.fusion import FusionModel  # noqa: E402
from mantis.defense.l0_rules import rules as l0  # noqa: E402
from mantis.defense.l1_gbdt import L1Model  # noqa: E402
from mantis.defense.l2_novelty import L2Model  # noqa: E402
from mantis.defense.l3_text import L3Model  # noqa: E402
from mantis.defense.l4_graph.graph import EntityGraph  # noqa: E402
from mantis.defense.policy import PolicyThresholds, decide  # noqa: E402

POOL: Final[Path] = GENERATED_DIR / "pool_2seed.parquet"
LATENCY_JSON: Final[Path] = GENERATED_DIR / "latency.json"
SEED: Final[int] = 1337

#: The authorisation-host budget this is measured against. See the docstring.
DEFAULT_BUDGET_MS: Final[float] = 50.0

#: Stages that must complete before the issuer answers the acquirer, because
#: they are what changes the authorisation decision. L0's clauses need the event;
#: L1 needs the event's features, which is every feature block plus the two
#: stateful stores.
INLINE_STAGES: Final[tuple[str, ...]] = (
    "velocity",
    "graph",
    "transaction",
    "entity",
    "mandate",
    "L0",
    "L1",
)

#: Stages whose output drives a **post-authorisation** action -- mandate
#: revocation, agent quarantine -- and which therefore do not sit inside the
#: authorisation window. See the module docstring.
#:
#: `graph` appears in INLINE_STAGES rather than here because L1 consumes the 28
#: `gph_` features: the graph pass is inline as a *feature source* even though
#: L4's own ring findings drive a fast-follow action.
#:
#: `L2` is here because CLAUDE.md demotes it to a residual monitor and drift
#: canary rather than a detector -- "has the shape of legitimate traffic moved"
#: is a question about the stream, not about the authorisation in flight.
#:
#: `fusion+policy` is here because the fused score is what the *review queue* and
#: the revocation logic read. The inline decision is L0's clauses plus L1's own
#: threshold; a fused score that waits on L2 and L3 cannot be inline by
#: definition, since two of its inputs are not.
FOLLOW_STAGES: Final[tuple[str, ...]] = ("L2", "L3", "fusion+policy")

#: Stages in the order an online scorer runs them. The order is the report's
#: order too, because a reader should see where the time goes along the path.
STAGES: Final[tuple[str, ...]] = (
    "velocity",
    "graph",
    "transaction",
    "entity",
    "mandate",
    "L0",
    "L1",
    "L2",
    "L3",
    "fusion+policy",
)


def _percentiles(samples: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(samples)),
        "p50": float(np.percentile(samples, 50)),
        "p95": float(np.percentile(samples, 95)),
        "p99": float(np.percentile(samples, 99)),
        "max": float(np.max(samples)),
    }


def _warm(frame: pd.DataFrame) -> tuple[RollingStore, EntityGraph]:
    """Replay the training window so the stores hold the history a host would.

    This is the expensive part of the script and it is not optional. Scoring
    against empty dictionaries would skip eviction entirely, would find every
    key absent and return NaN without touching a window, and would report a
    latency for a case that never occurs after the first hour of a deployment.
    """
    store = RollingStore(VELOCITY_KEYS)
    graph = EntityGraph()

    epoch = as_epoch(frame["ts"].dt.tz_localize(None) if frame["ts"].dt.tz else frame["ts"])
    amount = frame["amount"].to_numpy(dtype=float)
    from mantis.core.events import DECLINE_RESPONSES

    declined = np.isin(frame["auth_response"].to_numpy(), tuple(DECLINE_RESPONSES))
    refund = frame["txn_type"].to_numpy() == "refund"
    lag = frame["settlement_lag_hours"].to_numpy(dtype=float)
    key_columns = {
        spec.name: [frame[c].to_numpy() for c in spec.columns] for spec in VELOCITY_KEYS
    }
    customer = frame["customer_id"].to_numpy()
    device = frame["device_id"].to_numpy()
    merchant = frame["merchant_id"].to_numpy()
    agent = frame["ag_agent_id"].to_numpy()
    card_bin = frame["card_bin"].to_numpy()

    for i in range(len(frame)):
        keys = {
            spec.name: spec.key_of(tuple(column[i] for column in key_columns[spec.name]))
            for spec in VELOCITY_KEYS
        }
        store.observe(
            keys,
            ts=float(epoch[i]),
            amount=float(amount[i]),
            declined=bool(declined[i]),
            outcome_known=True,
            refund=bool(refund[i]),
            settlement_lag=None if np.isnan(lag[i]) else float(lag[i]),
        )
        graph.observe(
            ts=float(epoch[i]),
            customer=customer[i],
            device=device[i],
            merchant=merchant[i],
            agent=agent[i],
            card_bin=card_bin[i],
            amount=float(amount[i]),
            declined=bool(declined[i]),
        )
    return store, graph


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure per-event scoring latency.")
    parser.add_argument("--n", type=int, default=1000, help="events to time, one at a time")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_MS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    if not POOL.exists():
        raise SystemExit(
            f"missing {POOL}\n"
            "  build it with: python -c \"from mantis.defense.pool import build_pool; "
            f"build_pool(seeds=(1337, 7)).to_parquet(r'{POOL}', index=False)\""
        )

    print("MANTIS latency bench")
    print("=" * 78)
    started = time.time()

    pool = pd.read_parquet(POOL).sort_values("ts", kind="stable").reset_index(drop=True)
    cut = pool["ts"].quantile(TRAIN_SHARE)
    train_mask = pool["ts"] <= cut
    tr, te = train_mask.to_numpy(), ~train_mask.to_numpy()
    print(f"  pool          {len(pool):,} events, train {tr.sum():,} / test {te.sum():,}")

    print("  fitting the stack (once, offline - not part of the measurement)...")
    builder = FeatureBuilder()
    X = builder.fit_transform_stream(pool, train_mask)
    y = pool["is_fraud"].to_numpy(dtype=bool)
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]

    l1 = L1Model(seed=args.seed).fit(X_tr, y_tr, timestamps=pool.loc[tr, "ts"])
    legit = X_tr[~y_tr]
    l2 = L2Model(seed=args.seed).fit(legit, np.zeros(len(legit), dtype=bool))
    del legit
    l3 = L3Model(seed=args.seed).fit()

    s1_te, s2_te = l1.score(X_te), l2.score(X_te)
    s3_te = l3.score(pool[te])
    fusion = FusionModel(seed=args.seed).fit(
        {"L1": l1.score(X_tr), "L2": l2.score(X_tr), "L3": l3.score(pool[tr])}, y_tr
    )
    thresholds = PolicyThresholds.fit(
        fusion.score({"L1": s1_te, "L2": s2_te, "L3": s3_te}), y_te
    )

    print(f"  warming state on {tr.sum():,} training events...")
    store, graph = _warm(pool[tr].reset_index(drop=True))
    print(f"    velocity keys held  {sum(store.size().values()):,}")

    # ---------------------------------------------------------------- time ---
    test = pool[te].reset_index(drop=True)
    matrix = X_te.reset_index(drop=True)
    n = min(args.n, len(test))
    rng = np.random.default_rng(args.seed)
    # A contiguous block, not a random sample: the stores are stateful and
    # sampling scattered events would fold a different history in than the one
    # the events actually had. The block's start is randomised instead.
    start = int(rng.integers(0, max(1, len(test) - n)))
    print(f"  timing {n:,} events one at a time, from test row {start:,}")

    from mantis.core.events import DECLINE_RESPONSES

    per_stage: dict[str, list[float]] = {name: [] for name in STAGES}
    totals: list[float] = []
    inline_totals: list[float] = []

    for offset in range(n):
        i = start + offset
        row = test.iloc[[i]]
        vector = matrix.iloc[[i]]
        ts = float(as_epoch(row["ts"].dt.tz_localize(None) if row["ts"].dt.tz else row["ts"])[0])
        amount = float(row["amount"].iloc[0])
        declined = bool(row["auth_response"].iloc[0] in DECLINE_RESPONSES)
        refund = bool(row["txn_type"].iloc[0] == "refund")
        lag = row["settlement_lag_hours"].iloc[0]
        keys = {
            spec.name: spec.key_of(tuple(row[c].iloc[0] for c in spec.columns))
            for spec in VELOCITY_KEYS
        }

        event_started = time.perf_counter()

        mark = time.perf_counter()
        store.observe(
            keys,
            ts=ts,
            amount=amount,
            declined=declined,
            outcome_known=True,
            refund=refund,
            settlement_lag=None if pd.isna(lag) else float(lag),
        )
        per_stage["velocity"].append(time.perf_counter() - mark)

        mark = time.perf_counter()
        graph.observe(
            ts=ts,
            customer=row["customer_id"].iloc[0],
            device=row["device_id"].iloc[0],
            merchant=row["merchant_id"].iloc[0],
            agent=row["ag_agent_id"].iloc[0],
            card_bin=row["card_bin"].iloc[0],
            amount=amount,
            declined=declined,
        )
        per_stage["graph"].append(time.perf_counter() - mark)

        mark = time.perf_counter()
        transaction_features(row)
        per_stage["transaction"].append(time.perf_counter() - mark)

        mark = time.perf_counter()
        from mantis.defense.features.entity import entity_features

        entity_features(row, builder.profiles)
        per_stage["entity"].append(time.perf_counter() - mark)

        mark = time.perf_counter()
        from mantis.defense.features.mandate import mandate_features

        mandate_features(row, builder.baselines)
        per_stage["mandate"].append(time.perf_counter() - mark)

        mark = time.perf_counter()
        l0.evaluate(row)
        per_stage["L0"].append(time.perf_counter() - mark)

        mark = time.perf_counter()
        one1 = l1.score(vector)
        per_stage["L1"].append(time.perf_counter() - mark)

        mark = time.perf_counter()
        one2 = l2.score(vector)
        per_stage["L2"].append(time.perf_counter() - mark)

        mark = time.perf_counter()
        one3 = l3.score(row)
        per_stage["L3"].append(time.perf_counter() - mark)

        mark = time.perf_counter()
        fused = fusion.score({"L1": one1, "L2": one2, "L3": one3})
        decide(fused, thresholds, human_present=row["ag_human_present"].to_numpy())
        per_stage["fusion+policy"].append(time.perf_counter() - mark)

        totals.append(time.perf_counter() - event_started)
        # Summed from the per-stage clocks rather than wrapped in its own timer:
        # the inline path is a subset of the same single pass, and re-timing it
        # would double-count the work.
        inline_totals.append(
            sum(per_stage[name][-1] for name in INLINE_STAGES)
        )

    # --------------------------------------------------------- batch timing ---
    # The same stages over the same events, called once with the whole block.
    # This is not the deployable number; it is the denominator that turns the
    # single-event number into a statement about *where* the time goes.
    print("  timing the same stages in batch, for the per-call overhead ratio...")
    block = test.iloc[start : start + n]
    block_matrix = matrix.iloc[start : start + n]
    batch_ms: dict[str, float] = {}

    from mantis.defense.features.entity import entity_features as _entity
    from mantis.defense.features.mandate import mandate_features as _mandate

    for name, call in (
        ("transaction", lambda: transaction_features(block)),
        ("entity", lambda: _entity(block, builder.profiles)),
        ("mandate", lambda: _mandate(block, builder.baselines)),
        ("L0", lambda: l0.evaluate(block)),
        ("L1", lambda: l1.score(block_matrix)),
        ("L2", lambda: l2.score(block_matrix)),
        ("L3", lambda: l3.score(block)),
    ):
        mark = time.perf_counter()
        call()
        batch_ms[name] = (time.perf_counter() - mark) * 1000.0 / n

    block_s1, block_s2 = l1.score(block_matrix), l2.score(block_matrix)
    block_s3 = l3.score(block)
    mark = time.perf_counter()
    block_fused = fusion.score({"L1": block_s1, "L2": block_s2, "L3": block_s3})
    decide(block_fused, thresholds, human_present=block["ag_human_present"].to_numpy())
    batch_ms["fusion+policy"] = (time.perf_counter() - mark) * 1000.0 / n

    # velocity and graph have no batch form that is any cheaper: their offline
    # pass is this same per-event loop, event by event, because the state has to
    # be read before the event is folded in. Their single-event cost IS their
    # amortised cost, which is why they show no overhead ratio -- and why they
    # are the two numbers a deployable estimate rests on.
    batch_ms["velocity"] = float(np.mean(per_stage["velocity"])) * 1000.0
    batch_ms["graph"] = float(np.mean(per_stage["graph"])) * 1000.0

    # -------------------------------------------------------------- report ---
    total_ms = np.asarray(totals) * 1000.0
    end_to_end = _percentiles(total_ms)
    # The split must be a partition of STAGES. Computing fast-follow by
    # subtraction would let a stage in neither tuple vanish into it silently,
    # which is how L2 -- 40% of the clock -- spent one run mislabelled.
    assert set(INLINE_STAGES) | set(FOLLOW_STAGES) == set(STAGES), (
        "INLINE_STAGES and FOLLOW_STAGES must partition STAGES; "
        f"unassigned: {set(STAGES) - set(INLINE_STAGES) - set(FOLLOW_STAGES)}"
    )
    assert not set(INLINE_STAGES) & set(FOLLOW_STAGES), "a stage cannot be in both paths"

    inline = _percentiles(np.asarray(inline_totals) * 1000.0)
    follow = _percentiles(
        np.asarray(
            [sum(per_stage[name][i] for name in FOLLOW_STAGES) for i in range(n)]
        )
        * 1000.0
    )
    stages = {
        name: _percentiles(np.asarray(values) * 1000.0) for name, values in per_stage.items()
    }

    batch_total = sum(batch_ms.values())

    print()
    print(
        f"  {'stage':<16} {'mean':>9} {'p50':>9} {'p95':>9} {'p99':>9} "
        f"{'share':>7} {'in batch':>10} {'overhead':>10}"
    )
    print(
        f"  {'-' * 16} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 9} "
        f"{'-' * 7} {'-' * 10} {'-' * 10}"
    )
    for name in STAGES:
        row = stages[name]
        share = row["mean"] / end_to_end["mean"] if end_to_end["mean"] else 0.0
        per_row = batch_ms.get(name, float("nan"))
        factor = row["mean"] / per_row if per_row > 0 else float("nan")
        factor_text = "-" if abs(factor - 1.0) < 0.01 else f"{factor:>9,.0f}x"
        print(
            f"  {name:<16} {row['mean']:>8.3f}ms {row['p50']:>8.3f}ms "
            f"{row['p95']:>8.3f}ms {row['p99']:>8.3f}ms {share:>7.1%} "
            f"{per_row:>9.4f}ms {factor_text:>10}"
        )
    print(
        f"  {'-' * 16} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 9} "
        f"{'-' * 7} {'-' * 10} {'-' * 10}"
    )
    print(
        f"  {'END TO END':<16} {end_to_end['mean']:>8.3f}ms {end_to_end['p50']:>8.3f}ms "
        f"{end_to_end['p95']:>8.3f}ms {end_to_end['p99']:>8.3f}ms {'':>7} "
        f"{batch_total:>9.4f}ms"
    )
    print()

    passed = end_to_end["p99"] <= args.budget
    inline_passed = inline["p99"] <= args.budget
    headroom = args.budget / end_to_end["p99"] if end_to_end["p99"] else float("inf")
    inline_headroom = args.budget / inline["p99"] if inline["p99"] else float("inf")

    print("  Two paths, because a firewall is not one deadline")
    print("  " + "-" * 48)
    print(f"  {'path':<34} {'p50':>9} {'p95':>9} {'p99':>9}")
    print(f"  {'-' * 34} {'-' * 9} {'-' * 9} {'-' * 9}")
    print(
        f"  {'INLINE  L0+L1, pre-auth':<34} {inline['p50']:>8.2f}ms "
        f"{inline['p95']:>8.2f}ms {inline['p99']:>8.2f}ms"
    )
    print(
        f"  {'FAST-FOLLOW  L2+L3+fusion, post-auth':<34} {follow['p50']:>8.2f}ms "
        f"{follow['p95']:>8.2f}ms {follow['p99']:>8.2f}ms"
    )
    print(
        f"  {'FULL STACK':<34} {end_to_end['p50']:>8.2f}ms "
        f"{end_to_end['p95']:>8.2f}ms {end_to_end['p99']:>8.2f}ms"
    )
    print()
    print(f"  budget {args.budget:.0f} ms, and it governs the INLINE path")
    print(
        f"  inline p99                   {inline['p99']:>9.2f} ms   "
        f"{'WITHIN BUDGET' if inline_passed else 'OVER BUDGET'}"
        + (f"  ({inline_headroom:.1f}x headroom)" if inline_passed else "")
    )
    print(
        f"  full-stack p99               {end_to_end['p99']:>9.2f} ms   "
        f"{'WITHIN BUDGET' if passed else 'OVER BUDGET'}"
        + (f"  ({headroom:.1f}x headroom)" if passed else "")
    )
    print(f"  same stages, per row in batch{batch_total:>9.4f} ms")
    print(f"  max observed                 {end_to_end['max']:>9.2f} ms over {n:,} events")
    print()
    print("  The fast-follow layers drive mandate revocation and agent quarantine, both")
    print("  of which are post-authorisation actions against the mandate registry and the")
    print("  agent directory - they cannot be applied to the authorisation in flight, so")
    print("  holding it open for them would be an architectural error, not a slow one.")
    print()

    # The stage that dominates the clock, not the one with the largest ratio. A
    # 255x overhead on a stage costing 0.03 ms per row is not what misses a
    # budget; 47 ms on one stage is.
    heaviest = max(STAGES, key=lambda name: stages[name]["mean"])
    factor = stages[heaviest]["mean"] / batch_ms[heaviest] if batch_ms.get(heaviest) else 1.0
    streaming = stages["velocity"]["p99"] + stages["graph"]["p99"]

    print("  Read the last two columns together.")
    print(
        f"  Most of the clock goes to {heaviest}, which costs "
        f"{stages[heaviest]['mean']:.1f} ms called with one row and "
        f"{batch_ms[heaviest]:.4f} ms per row"
    )
    print(
        f"  called with many - a factor of {factor:,.0f}. That is per-call framework "
        "overhead in"
    )
    print("  pandas and scikit-learn, not model work, and the named fix is in this")
    print("  script's docstring. It is NOT applied: the feature builder is shared with the")
    print("  offline pass behind every pinned number in RESULTS.md.")
    print()
    print("  The two stages that genuinely cannot be batched - the stateful stores, which")
    print("  must read state before folding the event in - cost")
    print(
        f"    velocity {stages['velocity']['p99']:.3f} ms p99 + "
        f"graph {stages['graph']['p99']:.3f} ms p99 = {streaming:.3f} ms."
    )
    print(f"  The p99 of {end_to_end['p99']:.1f} ms above is what THIS implementation does")
    print("  today, and it is reported as measured rather than as an estimate of what a")
    print("  rewritten scoring path would do.")

    payload: dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_events": n,
        "budget_ms": args.budget,
        "within_budget": bool(passed),
        "headroom": float(headroom),
        "end_to_end_ms": end_to_end,
        "inline_ms": inline,
        "fast_follow_ms": follow,
        "inline_stages": list(INLINE_STAGES),
        "follow_stages": list(FOLLOW_STAGES),
        "inline_within_budget": bool(inline_passed),
        "inline_headroom": float(inline_headroom),
        "stages_ms": stages,
        "batch_per_row_ms": batch_ms,
        "batch_total_per_row_ms": batch_total,
        "streaming_stages_ms": {
            "velocity_p99": stages["velocity"]["p99"],
            "graph_p99": stages["graph"]["p99"],
        },
        "pool": POOL.name,
        "warm_events": int(tr.sum()),
        "note": (
            "One event at a time against warm velocity and graph state. Python/pandas "
            "throughout, so an upper bound on the same models' cost: the same stages "
            f"cost {batch_total:.4f} ms per row in batch, and the gap is per-call "
            "framework overhead rather than model work."
        ),
    }
    ensure_dir(GENERATED_DIR)
    LATENCY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote {LATENCY_JSON}  ({time.time() - started:.0f}s total)")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
