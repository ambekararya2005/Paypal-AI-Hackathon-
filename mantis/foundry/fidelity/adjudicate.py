"""Which side of a divergence is the anomalous one?

A discriminator that separates two panels tells you a difference exists. It does
not tell you which panel is wrong, and the temptation — three days before a
submission — is to assume it is the reference and move on. So the rule this
module enforces is that **an adjudication must carry a measurement**, not a
preference:

    A divergence may be attributed to the reference panel only when a *third*
    quantity, independent of both panels and stated in advance, says the
    reference is the side that departs from it.

The third quantity here is domain structure that payments people agree on before
either dataset exists: retail spend has a diurnal curve, and an acceptance estate
has a heavy-tailed merchant-popularity curve. Both are checkable against each
panel separately, which is what :func:`adjudicate` does, and both are stated with
the number that decided them.

The two findings, and why they are not special pleading
--------------------------------------------------------
The Sparkov reference panel turns out to have **neither** structure:

* Its hour-of-day distribution is a two-level step: ~3.3% of volume in each hour
  before noon and ~5.1% in each hour after, a peak-to-trough ratio of **1.64**.
  Real card traffic has a pronounced overnight trough. This population's ratio is
  **22.5**, with the trough at 03:00 and the peak at 19:00.
* Its 693 merchants are close to uniformly popular — the busiest takes 6.4x the
  volume of the quietest, and the top 10% of merchants carry **14.6%** of spend
  against the 10% that perfect uniformity would give. Acceptance estates are
  Zipf: this population's top 10% carry **66.0%**.

Neither is a defect in Kaggle's dataset for its own purpose. Sparkov exists to
benchmark fraud classifiers, and a flat time curve does not hurt that. It does
mean these two axes cannot be used to measure this project's fidelity, and the
scorecard therefore reports the discriminator **twice**: once on everything, and
once with the adjudicated axes removed, so a reader can see exactly how much of
the separation rests on a judgement call and decide about it themselves.

What this module refuses to do
-------------------------------
It does not adjudicate a feature it has no third quantity for. ``log_amount_z``,
``amount_vs_customer``, ``gap_ratio_log``, ``burst_1h``, ``dow`` and
``category_shift`` have no such reference structure available here, so their
distances stand as measured, unexcused, and they are the ones that carry the
honest part of section 2.
"""

from __future__ import annotations

from typing import Any, Final

import pandas as pd

__all__ = ["ADJUDICATED_FEATURES", "adjudicate"]

#: Features whose divergence is attributed to the reference panel, and therefore
#: excluded from the ablated discriminator. Anything not in this tuple stands as
#: measured.
ADJUDICATED_FEATURES: Final[tuple[str, ...]] = ("hour", "merchant_rank_pct")


def _diurnal_ratio(frame: pd.DataFrame) -> float:
    """Peak hour share divided by trough hour share. 1.0 is no diurnal curve."""
    shares = pd.to_datetime(frame["ts"]).dt.hour.value_counts(normalize=True)
    return float(shares.max() / shares.min()) if shares.min() > 0 else float("inf")


def _estate_concentration(frame: pd.DataFrame) -> tuple[float, float]:
    """Share of volume taken by the busiest 10% of merchants, and the max/min ratio.

    0.10 is perfect uniformity. A real acceptance estate is heavy-tailed and sits
    far above it, which is what makes this a usable third quantity.
    """
    shares = frame["merchant_id"].value_counts(normalize=True).sort_values(ascending=False)
    head = max(1, len(shares) // 10)
    spread = float(shares.iloc[0] / shares.iloc[-1]) if shares.iloc[-1] > 0 else float("inf")
    return float(shares.iloc[:head].sum()), spread


def adjudicate(synthetic_common: pd.DataFrame, real_common: pd.DataFrame) -> list[dict[str, Any]]:
    """Return one verdict per adjudicated feature, each carrying its evidence."""
    syn_diurnal = _diurnal_ratio(synthetic_common)
    real_diurnal = _diurnal_ratio(real_common)
    syn_head, syn_spread = _estate_concentration(synthetic_common)
    real_head, real_spread = _estate_concentration(real_common)

    return [
        {
            "feature": "hour",
            "third_quantity": "retail spend has a diurnal curve (overnight trough)",
            "synthetic": f"peak/trough {syn_diurnal:.1f}x",
            "reference": f"peak/trough {real_diurnal:.1f}x",
            "verdict": "REFERENCE" if real_diurnal < syn_diurnal else "SYNTHETIC",
            "note": (
                "The reference panel's hour curve is a two-level step - a flat rate "
                "before noon and a slightly higher flat rate after - rather than a "
                "diurnal curve. That is a property of the Sparkov generator and it is "
                "harmless for the benchmark it was built for, but it means this axis "
                "cannot measure whether our time-of-day model is right."
            ),
        },
        {
            "feature": "merchant_rank_pct",
            "third_quantity": "an acceptance estate is Zipf, not uniform (top 10% >> 10%)",
            "synthetic": f"top 10% carry {syn_head:.1%}, max/min {syn_spread:,.0f}x",
            "reference": f"top 10% carry {real_head:.1%}, max/min {real_spread:,.0f}x",
            "verdict": "REFERENCE" if abs(real_head - 0.10) < abs(syn_head - 0.10) else "SYNTHETIC",
            "note": (
                "The reference panel's 693 merchants are close to uniformly popular. "
                "Merchant volume in a real estate is heavy-tailed, which is why the "
                "foundry draws it from a Zipf curve at all, and it is the single "
                "feature the discriminator leans on hardest - so the ablated "
                "discriminator below is the number to read beside the full one."
            ),
        },
    ]


def format_adjudications(rows: list[dict[str, Any]]) -> str:
    """The block the scorecard CLI prints under the discriminator."""
    lines = []
    for row in rows:
        lines.append(f"  {row['feature']}  ->  divergence attributed to the {row['verdict']}")
        lines.append(f"    third quantity   {row['third_quantity']}")
        lines.append(f"    synthetic        {row['synthetic']}")
        lines.append(f"    reference        {row['reference']}")
        lines.append(f"    {row['note']}")
        lines.append("")
    return "\n".join(lines)
