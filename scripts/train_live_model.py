"""Fit the live gate's L1 + fusion artefact from a pooled population file.

    python scripts/train_live_model.py          # ~1 min on the 2-world pool

Reads ``data/generated/pool_2seed.parquet`` by default — the 404,240-event
two-world pool the latency bench and the adversary loop use. The five-world
pool behind RESULTS.md (``--pool data/generated/pool_5seed.parquet``) needs a
1.7 GiB single allocation for the feature matrix and ran out of memory on the
15 GB build laptop; the two-world artefact measures within a couple of points of
the pinned L1 numbers (see live_l1.json). Writes, both gitignored:

* ``data/models/live_l1.pkl``  — the fitted builder, L1, L1+L3 fusion, thresholds
* ``data/models/live_l1.json`` — what it is and how it measured, on its own test
  window, labelled as the live artefact and **not** the RESULTS.md numbers

Reads the pool, writes only to ``data/models/``. It cannot move a pinned number:
RESULTS.md and ``data/generated/`` are never opened for writing.

Recipe, identical to ``mantis.defense.experiment`` except that fusion is L1 + L3
(L2 is the drift canary; see mantis/defense/live_l1.py):

1. sort by time, 70/30 split;
2. ``FeatureBuilder.fit_transform_stream`` — one continuous velocity pass;
3. inner 80/20 split of the training window: L1 fitted on the first 80%, scored
   on the last 20%, and the fusion weights fitted there, where L1 has not looked;
4. L1 refitted on the whole training window;
5. policy thresholds placed at the 1.0 / 0.5 / 0.1% FPR budgets on the test
   window's legitimate fused scores, as ``PolicyThresholds.fit`` always does.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mantis.core.events import SCHEMA_VERSION  # noqa: E402
from mantis.core.paths import GENERATED_DIR  # noqa: E402
from mantis.defense.features import FeatureBuilder  # noqa: E402
from mantis.defense.fusion.model import FusionModel  # noqa: E402
from mantis.defense.l1_gbdt import L1Model  # noqa: E402
from mantis.defense.l3_text.model import L3Model  # noqa: E402
from mantis.defense.live import LIVE_MODEL_DIR  # noqa: E402
from mantis.defense.live_l1 import ARTEFACT_NAME, OnlineL1  # noqa: E402
from mantis.defense.metrics import FPR_GRID, score_report  # noqa: E402
from mantis.defense.policy.decide import PolicyThresholds  # noqa: E402

POOL: Final[Path] = GENERATED_DIR / "pool_2seed.parquet"
TRAIN_SHARE: Final[float] = 0.70
INNER_SHARE: Final[float] = 0.80
SEED: Final[int] = 1337


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pool", type=Path, default=POOL)
    parser.add_argument("--out", type=Path, default=LIVE_MODEL_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    if not args.pool.exists():
        print(f"missing {args.pool}; build it with mantis.defense.pool.build_pool "
              "(or copy pool_5seed.parquet in). The gate runs L0 + L3 without it.",
              file=sys.stderr)
        return 2

    started = time.time()

    def log(message: str) -> None:
        print(f"  [{time.time() - started:6.0f}s] {message}", flush=True)

    print("MANTIS live gate model: L1 + (L1, L3) fusion")
    pool = pd.read_parquet(args.pool).sort_values("ts", kind="stable").reset_index(drop=True)
    cut = pool["ts"].quantile(TRAIN_SHARE)
    train_mask = pool["ts"] <= cut
    tr, te = train_mask.to_numpy(), ~train_mask.to_numpy()
    y = pool["is_fraud"].to_numpy(dtype=bool)
    log(f"pool {len(pool):,} events ({y.sum():,} fraud); train {tr.sum():,} / test {te.sum():,}")

    builder = FeatureBuilder()
    X = builder.fit_transform_stream(pool, train_mask)
    log(f"features built: {X.shape[1]} columns")
    X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
    ts_tr = pool.loc[tr, "ts"]

    l3 = L3Model(seed=args.seed).fit()
    s3 = l3.score(pool)
    s3_tr, s3_te = s3[tr], s3[te]
    log("L3 fitted and scored")

    inner = (ts_tr <= ts_tr.quantile(INNER_SHARE)).to_numpy()
    l1_inner = L1Model(seed=args.seed).fit(X_tr[inner], y_tr[inner], timestamps=ts_tr[inner])
    fusion = FusionModel(seed=args.seed).fit(
        {"L1": l1_inner.score(X_tr[~inner]), "L3": s3_tr[~inner]}, y_tr[~inner]
    )
    del l1_inner
    log("fusion weights fitted on the inner 20% slice")

    l1 = L1Model(seed=args.seed).fit(X_tr, y_tr, timestamps=ts_tr)
    s1_te = l1.score(X_te)
    fused_te = fusion.score({"L1": s1_te, "L3": s3_te})
    thresholds = PolicyThresholds.fit(fused_te, y_te)
    log("L1 refitted on the whole training window; thresholds placed")

    metrics = {}
    for name, scores in (("L1", s1_te), ("fused(L1,L3)", fused_te)):
        reports = {f"{fpr:.3%}": score_report(scores, y_te, fpr) for fpr in FPR_GRID}
        first = next(iter(reports.values()))
        metrics[name] = {
            "auc_pr": round(first.auc_pr, 4),
            "recall_at_fpr": {k: round(r.recall, 4) for k, r in reports.items()},
        }
        print(f"    {name:<14} AUC-PR {first.auc_pr:.4f}  recall@0.1%FPR "
              f"{reports['0.100%'].recall:.4f}")

    manifest = {
        "artefact": "MANTIS live gate model (L1 + L1/L3 fusion)",
        "note": "Separate artefact from RESULTS.md: fusion is L1+L3 only. Metrics below are "
                "this artefact's own test window, not the pinned numbers.",
        "built": datetime.now(UTC).isoformat(timespec="seconds"),
        "schema_version": SCHEMA_VERSION,
        "pool": args.pool.name,
        "n_events": len(pool),
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "n_features": int(X.shape[1]),
        "seed": args.seed,
        "thresholds": {"challenge@1.0%FPR": thresholds.challenge,
                       "review@0.5%FPR": thresholds.review,
                       "decline@0.1%FPR": thresholds.decline},
        "fusion_weights": fusion.weights().to_dict(orient="records"),
        "test_metrics": metrics,
        "seconds": round(time.time() - started, 1),
    }
    artefact = OnlineL1(builder=builder, l1=l1, fusion=fusion, thresholds=thresholds,
                        l1_legit_reference=np.sort(s1_te[~y_te]), manifest=manifest)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ARTEFACT_NAME).open("wb") as handle:
        pickle.dump(artefact, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (args.out / "live_l1.json").write_text(json.dumps(manifest, indent=2, default=float) + "\n",
                                           encoding="utf-8")
    size = (args.out / ARTEFACT_NAME).stat().st_size / 1e6
    log(f"wrote {args.out / ARTEFACT_NAME} ({size:.1f} MB) and live_l1.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
