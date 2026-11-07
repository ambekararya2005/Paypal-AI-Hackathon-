"""F4-28 — threshold probing: value split into transactions that hug a boundary.

The attack
----------
Risk policy uses amount thresholds, and a threshold is discoverable by anyone
who can observe outcomes. Once located, value is split into several
authorisations that sit deliberately just below it, distributed across
merchants so no single relationship carries the whole total. Each transaction is
unremarkable by construction; the tell is the unnatural density of amounts
hugging a line that means nothing to a genuine customer.

Modelling decisions worth defending
-----------------------------------
* **Six thresholds, weighted toward the low ones.** Indian card and UPI policy
  has meaningful boundaries at ₹1,000 (low-value / UPI Lite), ₹2,000 (PIN-less
  contactless), ₹5,000, ₹10,000, ₹25,000 and ₹50,000. Using only the high ones
  would put every attack row above the population's 90th percentile and hand a
  detector the ``amount`` column for free. Weighting toward the low boundaries
  keeps the bulk of the attack inside the dense part of the amount distribution
  -- the population's median ticket is ₹782.
* **The boundary is picked per customer, not per campaign.** An operator who has
  located five thresholds uses all five. Drawing one per campaign gave the
  weights nothing to express and left the realised amounts far higher than
  intended.
* **Categories are filtered to ones where the threshold is a plausible ticket.**
  A ₹49,000 bus fare is a generator artefact. For each campaign the target
  threshold must sit between the category's 35th and 97th percentile, so the
  amount looks native to the merchant even while it hugs the boundary.
* **Splits are round-snapped some of the time.** The base population snaps 16%
  of amounts (46% in fuel, recharge and P2P) to round rupee steps, so a
  structuring pattern that never snapped would be *less* natural than the
  background, not more.
* **The distributed half.** Splits land at different merchants inside the
  category and, across campaigns, across different BINs and rails — so the
  evidence is a per-entity aggregate, not a repeated merchant pair.

How the boundary gets located (amendment 1.1.0)
-----------------------------------------------
Structuring presupposes that the operator *knows* where the line is, and until
Day 3 this injector simply granted them that knowledge. With ``auth_response``
in the schema, the discovery step is now modelled: each campaign opens with a
short **calibration phase** of attempts placed deliberately *above* a candidate
boundary, most of which come back ``declined_risk``. Those declines are the
measurement. Everything after them sits just underneath the line that was found.

Two properties this buys, neither of which existed before:

* The attack now has a **temporal shape** — a small cluster of high, declined
  attempts followed by a long tail of lower, approved ones — rather than being a
  static amount distribution. That is a sequence feature, which is L1 and L4
  territory, not something a single column can express.
* ``declined_risk`` appears in the data at an elevated rate for exactly the
  reason the card names. The background declines for risk on ~0.4% of events;
  inside a calibration burst it is the modal outcome. The *whole-campaign* rate
  stays modest, because calibration is a small share of the volume.

Realism check (measured, not asserted)
--------------------------------------
Best single-feature depth-1 stump AUC: **see the probe table**. ``amount`` remains
the strongest single column — real and expected, since structuring does push
tickets above a Rs782 median — and it is still not separation: the
boundary-hugging pattern lives in the *density* of amounts just under each
threshold, which a single split cannot express.

Measured by ``mantis.foundry.injectors.probe`` against a 200k-event background at
seed 1337, over every column an issuer can read off one authorisation message.
Re-measure with ``python -m mantis.foundry --attacks F4-28``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mantis.foundry.injectors.base import (
    BaseAttack,
    campaign_id,
    card_entry_point,
    demo_main,
    register,
    split_count,
)

__all__ = ["ThresholdProbeAttack", "inject", "main"]

#: Policy boundaries an operator can locate by observing outcomes, and how often
#: a campaign targets each. Weighted low: most structuring is small.
_THRESHOLDS: tuple[int, ...] = (1_000, 2_000, 5_000, 10_000, 25_000, 50_000)
_THRESHOLD_P: tuple[float, ...] = (0.22, 0.30, 0.22, 0.14, 0.08, 0.04)

#: Where inside the "just under" region a split lands. Never above 0.995 — an
#: operator who touches the line loses the card.
_RATIO_RANGE: tuple[float, float] = (0.84, 0.995)

#: Rails a split can ride. Recurring and MOTO are excluded: a recurring mandate
#: cannot be re-cut into five payments on demand, so structuring there would be
#: an artefact of the generator rather than a technique.
_RAILS: tuple[str, ...] = ("ecom", "agentic", "upi_p2m", "card_present")

#: Share of a campaign spent locating the boundary rather than exploiting it.
#: Small: calibration is cheap and needs few samples, and a campaign that was
#: mostly declines would be caught by velocity long before structuring mattered.
_CALIBRATION_SHARE: float = 0.14

#: How far above the boundary a calibration attempt is placed. Close enough that
#: an approval would be informative, far enough that a decline is unambiguous.
_OVERSHOOT_RANGE: tuple[float, float] = (1.02, 1.35)

#: Chance a calibration attempt above the boundary is refused. Not 1.0: a
#: threshold is a risk score input, not a hard wall, and an operator who saw
#: deterministic refusal would have located it in one attempt.
_OVERSHOOT_DECLINE_P: float = 0.71


def _sequence_within(owner: np.ndarray) -> np.ndarray:
    """Index of each row inside its (sorted) owner group."""
    starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
    group_start = np.repeat(starts, np.diff(np.r_[starts, owner.size]))
    return np.arange(owner.size) - group_start


@register
class ThresholdProbeAttack(BaseAttack):
    """Just-under-threshold structuring across several merchants and rails."""

    card_id = "F4-28"
    base_events = 150
    # Nine, not five. Structuring campaigns are short, so a handful of them
    # leaves whole weekdays untouched and the probe reads ``ts_dow`` at 0.69 --
    # a calendar artefact rather than a property of structuring.
    base_campaigns = 9

    def inject(
        self, population: pd.DataFrame, intensity: float, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Emit campaigns that split value into near-boundary authorisations."""
        view = self.view
        # Per-category quantiles decide which thresholds are plausible tickets.
        q35 = {mcc: float(np.quantile(a, 0.35)) for mcc, a in view.amounts_by_mcc.items()}
        q97 = {mcc: float(np.quantile(a, 0.97)) for mcc, a in view.amounts_by_mcc.items()}

        active = view.customers[view.customers["n_events"] >= 6].index.to_numpy()
        counts = split_count(self.n_events(intensity), self.n_campaigns(intensity), rng)
        starts = view.spread_epochs(len(counts), rng)

        blocks: list[pd.DataFrame] = []
        for c, n in enumerate(counts):
            n = int(n)
            n_customers = int(np.clip(round(n / 3.2), 4, 20))
            cohort = rng.choice(active, size=min(n_customers, active.size), replace=False)
            owner = np.sort(rng.integers(0, cohort.size, n))
            rows = view.clone(view.source_rows(cohort[owner], rng, channels=_RAILS))

            # The boundary is chosen per customer, not per campaign. An operator
            # who has located five thresholds uses all five; drawing one per
            # campaign gave the weights no chance to express themselves and left
            # the realised amount distribution far higher than intended.
            per_customer = rng.choice(_THRESHOLDS, size=cohort.size, p=_THRESHOLD_P)
            threshold = per_customer[owner]

            plausible = {
                int(t): (
                    [m for m in view.amounts_by_mcc if q35[m] <= t <= q97[m]]
                    or [m for m in view.amounts_by_mcc if q97[m] >= t]
                    or ["5999"]
                )
                for t in np.unique(per_customer)
            }
            pools = {
                m: view.merchants_in_mcc(m, popularity=(0.2, 1.0))
                for group in plausible.values()
                for m in group
            }
            mcc_choice = np.asarray(
                [plausible[int(t)][rng.integers(0, len(plausible[int(t)]))] for t in threshold]
            )
            merchants = np.asarray(
                [pools[m][rng.integers(0, pools[m].size)] for m in mcc_choice], dtype=object
            )
            view.retarget(rows, merchants)

            # The opening calibration attempts overshoot the boundary; the rest
            # of the campaign hugs it from below.
            calibrating = np.zeros(n, dtype=bool)
            calibrating[: max(1, round(n * _CALIBRATION_SHARE))] = True

            ratio = np.where(
                calibrating,
                rng.uniform(*_OVERSHOOT_RANGE, n),
                rng.uniform(*_RATIO_RANGE, n),
            )
            amount = threshold * ratio
            # Match the population's own round-number habit rather than emitting
            # a smooth band, which would itself be a tell.
            step = np.select([amount < 1_000, amount < 10_000], [50.0, 100.0], default=500.0)
            snapped = np.maximum(step, np.floor(amount / step) * step)
            amount = np.where(rng.random(n) < 0.35, snapped, np.round(amount, 2))
            rows["amount"] = np.clip(amount, 1.0, None)

            # A split runs inside one sitting: minutes to a few hours apart.
            seq = _sequence_within(owner)
            gap = rng.integers(18 * 60, 5 * 3_600, n)
            # One session start per customer, so a split is a burst rather than
            # a scatter of unrelated events.
            sitting = rng.integers(0, 30 * 3_600, cohort.size)[owner]
            view.set_timestamps(rows, starts[c] + sitting + seq * gap, rng=rng, groups=owner)

            # An attempt over the boundary is usually refused, and that refusal
            # is the measurement the rest of the campaign is built on.
            declined = calibrating & (rng.random(n) < _OVERSHOOT_DECLINE_P)
            view.decline(rows, declined, "declined_risk")

            blocks.append(
                view.finalise(
                    rows,
                    card_id=self.card_id,
                    campaigns=np.full(n, campaign_id(self.card_id, c), dtype=object),
                    rng=rng,
                )
            )
        return pd.concat(blocks, ignore_index=True)


inject = card_entry_point(ThresholdProbeAttack)


def main() -> None:
    """Print a sample campaign.

    Run: ``python -m mantis.foundry.injectors.f4_28_threshold_probe``.
    """
    demo_main(ThresholdProbeAttack)


if __name__ == "__main__":
    main()
