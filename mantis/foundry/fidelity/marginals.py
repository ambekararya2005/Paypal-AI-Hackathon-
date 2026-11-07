"""Per-feature distances between the synthetic population and the reference panel.

Three questions, three answers:

* **Does each feature have the right distribution on its own?** Two-sample KS for
  the continuous features, Jensen-Shannon for the categorical ones. Each against
  the band that pure sampling noise would produce at these sample sizes, because
  a distance without a band is a number nobody can act on.
* **Do the features have the right relationships to each other?** The Frobenius
  distance between the two Spearman correlation matrices. A generator can match
  every marginal perfectly and still draw each column independently; this is the
  metric that notices.
* **Which feature is worst, and by how much?** The table is sorted by the ratio
  to the band, so the first row is the divergence a judge would find, and it is
  therefore the one this project has to name first.

Spearman rather than Pearson for the correlation matrix: ``log_amount_z`` and
``seconds_since_prior_z`` both have heavy tails on both panels, and a Pearson
coefficient between two heavy-tailed columns mostly reports where the biggest
outlier landed.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import pandas as pd

from mantis.foundry.fidelity.common import (
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    SHAPE_FEATURES,
)
from mantis.foundry.fidelity.metrics import (
    jsd,
    jsd_null_band,
    ks_two_sample,
    ks_two_sample_band,
    ratio,
    share_table,
)

__all__ = ["correlation_distance", "marginal_rows"]

#: A distance more than this many times its own sampling-noise band is called
#: out. Not a pass/fail gate: the scorecard ranks and names, it does not grade.
#: One is the point at which a difference stops being explicable as noise, and
#: three is where it stops being arguable.
FLAG_RATIO: Final[float] = 3.0


def marginal_rows(
    synthetic: pd.DataFrame, real: pd.DataFrame, *, seed: int = 1337
) -> list[dict[str, Any]]:
    """One row per shape feature: distance, band, ratio, and the two moments.

    ``synthetic`` and ``real`` are shape matrices from
    :func:`mantis.foundry.fidelity.common.to_shape`.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for name in CONTINUOUS_FEATURES:
        a = synthetic[name].to_numpy(dtype=float)
        b = real[name].to_numpy(dtype=float)
        distance = ks_two_sample(a, b)
        band = ks_two_sample_band(len(a), len(b))
        rows.append(
            {
                "feature": name,
                "kind": "continuous",
                "metric": "KS",
                "distance": distance,
                "band": band,
                "ratio": ratio(distance, band),
                "synthetic_median": float(np.median(a)),
                "real_median": float(np.median(b)),
                "synthetic_iqr": float(np.subtract(*np.percentile(a, [75, 25]))),
                "real_iqr": float(np.subtract(*np.percentile(b, [75, 25]))),
            }
        )

    for name in CATEGORICAL_FEATURES:
        a_raw = synthetic[name].to_numpy()
        b_raw = real[name].to_numpy()
        levels = sorted({str(v) for v in a_raw} | {str(v) for v in b_raw})
        p = share_table(a_raw, levels)
        q = share_table(b_raw, levels)
        distance = jsd(p, q)
        # The band is computed against the **real** side, which is the thing being
        # matched, and at the smaller of the two sample sizes, which is the one
        # that limits how tightly they could possibly agree.
        band = jsd_null_band(q, min(len(a_raw), len(b_raw)), rng)
        rows.append(
            {
                "feature": name,
                "kind": "categorical",
                "metric": "JSD",
                "distance": distance,
                "band": band,
                "ratio": ratio(distance, band),
                "levels": len(levels),
                "max_level_delta": float(np.max(np.abs(p - q))),
                "max_level": levels[int(np.argmax(np.abs(p - q)))],
            }
        )

    rows.sort(key=lambda r: (-r["ratio"], r["feature"]))
    return rows


def correlation_distance(synthetic: pd.DataFrame, real: pd.DataFrame) -> dict[str, Any]:
    """Frobenius distance between the two Spearman correlation matrices.

    Normalised by the number of **off-diagonal** entries and reported as a root
    mean square, so the answer reads as "the average correlation is wrong by
    this much" rather than as a number whose size depends on how many features
    there happen to be. The diagonal is excluded because it is 1.0 on both sides
    by construction and would otherwise dilute the statistic toward zero.
    """
    columns = list(SHAPE_FEATURES)
    a = synthetic[columns].corr(method="spearman").to_numpy()
    b = real[columns].corr(method="spearman").to_numpy()
    delta = np.nan_to_num(a - b)

    off = ~np.eye(len(columns), dtype=bool)
    frobenius = float(np.sqrt(np.sum(delta[off] ** 2)))
    rms = float(np.sqrt(np.mean(delta[off] ** 2)))

    flat = np.argsort(np.abs(delta[off]))[::-1]
    pairs = [(i, j) for i in range(len(columns)) for j in range(len(columns)) if i != j]
    worst = []
    seen: set[frozenset[str]] = set()
    for index in flat:
        i, j = pairs[index]
        key = frozenset({columns[i], columns[j]})
        if key in seen:
            continue
        seen.add(key)
        worst.append(
            {
                "pair": f"{columns[i]} x {columns[j]}",
                "synthetic": float(a[i, j]),
                "real": float(b[i, j]),
                "delta": float(delta[i, j]),
            }
        )
        if len(worst) == 5:
            break

    return {
        "frobenius": frobenius,
        "rms_off_diagonal": rms,
        "n_features": len(columns),
        "worst_pairs": worst,
    }


def format_marginals(rows: list[dict[str, Any]]) -> str:
    """The table as printed by the scorecard CLI."""
    lines = [
        f"  {'feature':<26} {'metric':<7} {'distance':>10} {'band':>10} {'x band':>8}",
        f"  {'-' * 26} {'-' * 7} {'-' * 10} {'-' * 10} {'-' * 8}",
    ]
    for row in rows:
        mark = "  <<" if row["ratio"] > FLAG_RATIO else ""
        lines.append(
            f"  {row['feature']:<26} {row['metric']:<7} "
            f"{row['distance']:>10.4f} {row['band']:>10.4f} {row['ratio']:>8.1f}{mark}"
        )
    return "\n".join(lines)
