"""Distances, and the null bands that make them mean something.

Day 4's ``scripts/drift_check.py`` needed exactly these four functions to compare
the realised population against its own specification. Day 7's scorecard needs
the same four to compare it against **real** data. They live here so there is one
implementation rather than two that drift apart — ``drift_check`` imports them
from this module, and the numbers it printed on Day 4 are unchanged, because the
formulae moved without being touched.

Why every distance carries a null band
--------------------------------------
"A JSD under 0.02 is fine" is a number somebody made up. Every distance in this
package is reported against a **band** instead: draw ``n`` samples from the target
distribution itself, ``BOOTSTRAP`` times, and take the 99th percentile of the
distance that pure sampling noise produces at that ``n`` and that support size. A
realised distance inside the band is indistinguishable from noise; one outside it
is a real deviation whose size can then be argued about on its merits.

This matters because supports differ wildly — ``txn_type`` has 5 levels and
``mcc`` has 24, and 200,000 draws give far tighter agreement on the first than
the second. One flat threshold would flag one and excuse the other for reasons
that have nothing to do with fidelity.

The two distances
-----------------
* **Categorical** — Jensen-Shannon divergence, base 2, so 0 is identical and 1 is
  disjoint support. Symmetric and finite even when a level is missing from one
  side, which KL is not.
* **Continuous** — the two-sample Kolmogorov-Smirnov statistic. The 99% critical
  value for two samples of size ``n`` and ``m`` is
  ``1.63 * sqrt((n + m) / (n * m))``.
"""

from __future__ import annotations

from typing import Final

import numpy as np

__all__ = [
    "BOOTSTRAP",
    "KS_99",
    "NULL_Q",
    "entity_jsd_null_band",
    "jsd",
    "jsd_null_band",
    "ks_two_sample",
    "ks_two_sample_band",
    "ratio",
    "share_table",
]

#: Bootstrap replicates for a null band. 200 is enough for a 99th percentile to
#: be stable to the third decimal and keeps a whole scorecard run under a minute.
BOOTSTRAP: Final[int] = 200

#: Quantile of the null distribution a realised distance must stay under.
NULL_Q: Final[float] = 0.99

#: KS 99% critical coefficient: ``P(sqrt(n) D > 1.63) ~= 0.01``.
KS_99: Final[float] = 1.63


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits. 0 identical, 1 disjoint."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        keep = a > 0
        return float(np.sum(a[keep] * np.log2(a[keep] / b[keep])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def jsd_null_band(target: np.ndarray, n: int, rng: np.random.Generator) -> float:
    """The JSD that ``n`` honest draws from ``target`` produce, at ``NULL_Q``.

    This is the whole point of the module: a distance is only evidence if it is
    bigger than what perfect sampling would have given you anyway.
    """
    target = np.asarray(target, dtype=float)
    target = target / target.sum()
    if n <= 0:
        return float("inf")
    if (target > 0).sum() <= 1:
        # A degenerate target -- entry_mode|moto is 100% ecom_keyed, threeds on
        # card_present is 100% not_applicable -- has no sampling noise at all, so
        # there is no band to compute. Any deviation is a real one.
        return 0.0
    draws = rng.multinomial(n, target, size=BOOTSTRAP).astype(float)
    dists = [jsd(row / n, target) for row in draws]
    return float(np.quantile(dists, NULL_Q))


def entity_jsd_null_band(
    weights: np.ndarray, target: np.ndarray, rng: np.random.Generator
) -> float:
    """Null band for a value drawn **per entity** and observed **per event**.

    ``merchant_country`` is drawn once per merchant; ``card_bin`` once per card;
    ``ag_agent_platform`` once per agent. The events then inherit it. Because
    merchant popularity is Zipf-distributed, one lucky draw on a head merchant
    moves the event-level marginal by far more than 1/n, and a null band computed
    at ``n = 200,000`` is wrong by an order of magnitude -- it will call ordinary
    sampling noise "drift" every single time.

    The correct null re-draws the value for every entity from the target and
    re-weights by that entity's **realised** event count, which is exactly what
    this does. ``weights`` is one event count per entity.
    """
    target = np.asarray(target, dtype=float)
    target = target / target.sum()
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 0 or (target > 0).sum() <= 1:
        return 0.0
    dists = []
    for _ in range(BOOTSTRAP):
        assigned = rng.choice(len(target), size=weights.size, p=target)
        mass = np.bincount(assigned, weights=weights, minlength=len(target)) / total
        dists.append(jsd(mass, target))
    return float(np.quantile(dists, NULL_Q))


def ratio(distance: float, band: float) -> float:
    """Distance as a multiple of its null band, with the degenerate case handled.

    ``entry_mode | moto`` is 100% ``ecom_keyed`` and ``threeds | card_present``
    is 100% ``not_applicable``. A one-level target has no sampling noise, so its
    band is zero -- and dividing by it turned six exactly-correct distributions
    into ``inf DRIFT``. A zero distance against a zero band is a perfect match.
    """
    if band > 0:
        return distance / band
    return 0.0 if distance <= 0 else float("inf")


def ks_two_sample(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS statistic: the largest gap between two empirical CDFs.

    Day 4 compared a sample against the reference's own *analytic* CDF, because
    the reference **was** the specification. Day 7 has no analytic form for the
    real data — only a second sample of it — so the two-sample statistic is the
    one that applies. Both are the same quantity: sup |F(x) - G(x)|.
    """
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / a.size
    fb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(fa - fb)))


def ks_two_sample_band(n: int, m: int) -> float:
    """The 99% critical value for a two-sample KS at these two sample sizes.

    Reported beside every KS distance for the same reason the JSD carries a
    bootstrap band: at n = m = 200,000 this is 0.0052, so a KS of 0.02 is a real
    difference and a KS of 0.004 is two samples of the same thing.
    """
    if n <= 0 or m <= 0:
        return float("inf")
    return float(KS_99 * np.sqrt((n + m) / (n * m)))


def share_table(values: np.ndarray, levels: list[str]) -> np.ndarray:
    """Share of each level in ``levels``, in order, from a categorical array.

    Levels absent from ``values`` come back as 0.0 rather than being dropped, so
    two tables built against the same ``levels`` are always aligned and can be
    handed straight to :func:`jsd`.
    """
    counts = {}
    unique, freq = np.unique(np.asarray(values).astype(str), return_counts=True)
    counts = dict(zip(unique, freq, strict=True))
    total = float(freq.sum()) or 1.0
    return np.array([counts.get(level, 0) / total for level in levels], dtype=float)
