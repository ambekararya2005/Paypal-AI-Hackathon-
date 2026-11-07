"""F1-02 — intent-mandate scope inflation. **HARD bucket.**

The attack
----------
A human delegates a *goal* rather than a basket: "book me a flight under 40k",
"keep the house stocked". The mandate carries an envelope — categories, a
ceiling, a merchant allow-list, a TTL — and the agent is granted interpretive
latitude by design. It then transacts outside the envelope: an adjacent category
billed against the mandate, or an amount that creeps under the ceiling and then
steps over it.

Because the mandate is genuine and unexpired, everything except the scope check
passes. That is why this is the highest-frequency F1 vector: it needs no
cryptographic failure, no injection and no model access — only latitude.

Bucket: HARD, and unapologetically so
---------------------------------------
Both signals are single-message checks the network can make with certainty:

* the settled MCC is absent from the category list the mandate carried, and
* the amount is at or above the scope ceiling.

**L0 should catch this at near-zero false positive rate.** The Day 1 population
was built so that it can: ``tests/test_population.py`` asserts that *every*
legitimate agentic authorisation sits inside its own mandate scope, on all four
clauses. There are zero background violations, so the rule has nothing to trade
against and the recall claim is clean.

That is precisely why F1-01 and F1-03 exist in the CLEAN bucket. If this were the
only kind of agentic attack we generated, the entire firewall would be a policy
engine and the ML story would be theatre.

Modelling decisions worth defending
-----------------------------------
* **Inflation is a ramp, not a step.** Within a campaign the ratio to the ceiling
  climbs across the sequence, most events sitting just under it and the tail
  crossing. That is the salami shape the card's mitigation ("decrement a
  cumulative budget across the mandate lifetime, not just per transaction")
  is written against, and it means a per-transaction ceiling check catches only
  the last few events — the cumulative one catches the campaign.
* **The two violation shapes are mixed, not merged.** Roughly half the events
  drift category and half inflate amount, with a minority doing both. An
  injector where every row tripped every clause would let a single rule score
  perfect recall and would tell us nothing about which clause is load-bearing.
* **Category drift is to a genuine sibling.** The settled MCC is a real category
  the customer plausibly transacts in, just not one the mandate named. Drifting
  to something absurd would make ``mcc`` itself separating.
* **The mandate is an intent mandate**, so the allow-list is empty — that is the
  population's own convention for intent mandates, and it is why this card
  attacks the category clause rather than the merchant clause.
* **No injected content, no collapsed deliberation.** This attack does not need
  them, and adding them would blur the bucket split that Day 4 asserts on.

Realism check (measured, not asserted)
--------------------------------------
See the probe table from ``python -m mantis.foundry --attacks F1-02``, quoting
the rail-conditioned column. A high number on ``ag_scope_categories``-derived or
amount-ratio features is *expected and correct* here: the attack is supposed to
be visible, and the honest claim is that L0 catches it, not that it is subtle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mantis.core.events import MandateType
from mantis.foundry.injectors.agentic import AgenticAttack, Bucket, agentic_pool
from mantis.foundry.injectors.base import (
    campaign_id,
    card_entry_point,
    demo_main,
    register,
    split_count,
)

__all__ = ["ScopeInflationAttack", "inject", "main"]

#: Share of events whose settled category is outside the mandate's category list.
_CATEGORY_DRIFT_SHARE: float = 0.55

#: Share whose amount reaches or exceeds the scope ceiling.
_AMOUNT_INFLATION_SHARE: float = 0.58

#: Where the ratio to the ceiling starts and ends across a campaign. Below 1.0
#: for most of the ramp: the creep is the technique, the breach is the payoff.
_RAMP_START: tuple[float, float] = (0.74, 0.88)
_RAMP_END: tuple[float, float] = (1.04, 1.9)

#: How many sibling categories an inflated mandate names. Small: a mandate
#: listing fifteen categories would not be a constraint at all.
_SCOPE_SIZE: tuple[int, int] = (1, 4)


@register
class ScopeInflationAttack(AgenticAttack):
    """Spend outside the envelope the human signed, on a genuine intent mandate."""

    card_id = "F1-02"
    bucket = Bucket.HARD
    base_events = 120
    base_campaigns = 6

    def inject(
        self, population: pd.DataFrame, intensity: float, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Emit in-force intent mandates spent outside their own scope."""
        view = self.view
        pool = agentic_pool(view)
        all_mccs = np.asarray(sorted(view.amounts_by_mcc), dtype=object)

        counts = split_count(self.n_events(intensity), self.n_campaigns(intensity), rng)
        starts = view.spread_epochs(len(counts), rng)

        blocks: list[pd.DataFrame] = []
        for c, n in enumerate(counts):
            n = int(n)
            rows = view.clone(rng.choice(pool, size=n, replace=True))

            rows["amount"] = view.draw_amounts(rows["mcc"].to_numpy(), 0.45, 0.95, rng)
            rows["ag_mandate_type"] = MandateType.INTENT.value

            # A mandate runs for a while; the campaign is the spend under it.
            order = np.argsort(rng.random(n))
            elapsed = np.sort(rng.integers(0, 20 * 86_400, n))[np.argsort(order)]
            view.set_timestamps(rows, starts[c] + elapsed, rng=rng)

            block = view.finalise(
                rows,
                card_id=self.card_id,
                campaigns=np.full(n, campaign_id(self.card_id, c), dtype=object),
                rng=rng,
            )
            self._inflate(block, all_mccs, order, rng)
            blocks.append(block)

        return pd.concat(blocks, ignore_index=True)

    def _inflate(
        self,
        block: pd.DataFrame,
        all_mccs: np.ndarray,
        order: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        """Undo ``finalise``'s scope repair, deliberately and in named ways.

        ``finalise`` makes the mandate describe the transaction, which is right
        for every other injector and is exactly wrong here: the gap between the
        signed envelope and the settled purchase *is* this attack. The two
        excursions are re-opened explicitly rather than by skipping the repair,
        so what is violated is a written decision and not an omission.
        """
        n = len(block)

        # An intent mandate names no merchant. That is the population's own
        # convention, and it is why this card attacks the category clause.
        block["ag_scope_allowed_merchants"] = pd.Series(
            [[] for _ in range(n)], index=block.index, dtype=object
        )

        # -- excursion 1: the settled category is outside the signed list ----- #
        drift = rng.random(n) < _CATEGORY_DRIFT_SHARE
        mcc = block["mcc"].to_numpy()
        categories = block["ag_scope_categories"].to_numpy().copy()
        for i in np.flatnonzero(drift):
            size = int(rng.integers(*_SCOPE_SIZE))
            # Siblings the human plausibly did authorise -- just not this one.
            picks: list[str] = []
            while len(picks) < size:
                candidate = str(all_mccs[rng.integers(0, all_mccs.size)])
                if candidate != str(mcc[i]) and candidate not in picks:
                    picks.append(candidate)
            categories[i] = picks
        block["ag_scope_categories"] = categories

        # -- excursion 2: the amount creeps up to the ceiling, then over ------ #
        # A ramp across the campaign's own sequence, so the breach is the tail of
        # a trend rather than an isolated row.
        progress = order / max(1, n - 1)
        lo = rng.uniform(*_RAMP_START, n)
        hi = rng.uniform(*_RAMP_END, n)
        ratio = lo + (hi - lo) * progress

        inflate = rng.random(n) < _AMOUNT_INFLATION_SHARE
        # Rows not on the inflation half keep a ceiling that still covers them,
        # so the category drift is measurable on its own.
        ratio = np.where(inflate, ratio, np.minimum(ratio, 0.95))
        block["ag_scope_max_amount"] = np.round(block["amount"].to_numpy() / ratio, 2)


inject = card_entry_point(ScopeInflationAttack)


def main() -> None:
    """Print a sample campaign.

    Run: ``python -m mantis.foundry.injectors.f1_02_scope_inflation``.
    """
    demo_main(ScopeInflationAttack)


if __name__ == "__main__":
    main()
