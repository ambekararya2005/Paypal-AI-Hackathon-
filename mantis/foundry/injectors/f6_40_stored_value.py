"""F6-40 — stored-value cash-out ring: convert authority into instruments.

The attack
----------
Compromised payment authority is converted into stored-value instruments that
can be resold or redeemed away from the rail. Once value sits on an instrument
the payment network has nothing left to trace, so the detectable window is the
purchase itself: a burst of denominated buys, from customers with no history of
that category, converging on a handful of redeeming merchants.

Why this card and not chargeback abuse
--------------------------------------
The Day 2 attack list asked for a chargeback/refund abuse ring. ``TxEvent`` is an
authorisation message: it carries no refund flag, no dispute outcome and no
authorisation response code, so a refund-shaped attack cannot be *labelled* in
this schema without either inventing a field — the frozen schema forbids it,
CLAUDE.md §4 — or faking the signal the detector is supposed to find. The same
constraint is why F1-03 (refund-logic hijack) is deliberately held at
``status: mapped``. This card is the monetisation ring the schema *can* carry
honestly: same actor, same ring topology, same "convert authority into value
fast" objective, and every signal it claims is present in the data.

Modelling decisions worth defending
-----------------------------------
* **Denominations, not amounts.** Purchases land on ₹500 / ₹1,000 / ₹2,000 /
  ₹5,000 / ₹10,000, weighted toward the low end. The population already snaps
  16% of its amounts to round steps, so roundness alone is not separating —
  it is roundness *combined with* category novelty and burst timing, i.e.
  ``amount_round_number_flag`` earning its place next to other features rather
  than on its own.
* **Category novelty is per-customer, not global.** The ring buys in ordinary
  retail, department-store and electronics categories that plenty of people use.
  What is odd is that *these* customers never have. That is
  ``mcc_novelty_for_customer``, and it is invisible to any single column.
* **Convergence, not volume.** Ten to twenty unrelated customers, two to four
  redeeming merchants, inside a few days. Any one purchase is a gift card; the
  set of them is a cash-out.
* **Bursts of two to four purchases** inside half an hour to three hours, which
  is what ``customer_velocity_1h`` is there to catch.

Realism check (measured, not asserted)
--------------------------------------
Best single-feature depth-1 stump AUC: **0.676** (``amount``) — the denominational
structure is deliberately invisible to a threshold, which is the correct outcome:
a stump cannot express "is a round number". The ring is found by convergence and
by per-customer category novelty.

Measured by ``mantis.foundry.injectors.probe`` against a 200k-event background at
seed 1337, over every column an issuer can read off one authorisation message.
Re-measure with ``python -m mantis.foundry --attacks F6-40``.
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

__all__ = ["StoredValueAttack", "inject", "main"]

#: Categories that carry stored-value and easily-resold goods in this estate.
_REDEEM_MCCS: tuple[str, ...] = ("5999", "5311", "5732", "5734", "5651")

#: Instrument denominations and how often each is bought.
_DENOMINATIONS: tuple[int, ...] = (500, 1_000, 2_000, 5_000, 10_000)
_DENOMINATION_P: tuple[float, ...] = (0.26, 0.31, 0.24, 0.14, 0.05)

#: Stored value is bought at a till as readily as online, and excluding
#: card-present made "not card-present" a free 0.64-AUC feature.
_RAILS: tuple[str, ...] = ("ecom", "agentic", "upi_p2m", "card_present")


@register
class StoredValueAttack(BaseAttack):
    """Unrelated customers converging on a few redeemers, in denominated bursts."""

    card_id = "F6-40"
    base_events = 130
    base_campaigns = 5

    def inject(
        self, population: pd.DataFrame, intensity: float, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Emit denominated purchase bursts converging on a handful of merchants."""
        view = self.view
        active = view.customers[view.customers["n_events"] >= 4].index.to_numpy()
        redeem_pools = {m: view.merchants_in_mcc(m, popularity=(0.05, 0.55)) for m in _REDEEM_MCCS}

        counts = split_count(self.n_events(intensity), self.n_campaigns(intensity), rng)
        starts = view.spread_epochs(len(counts), rng)

        blocks: list[pd.DataFrame] = []
        for c, n in enumerate(counts):
            n = int(n)
            ring_size = int(min(np.clip(round(n / 2.6), 10, 20), active.size))
            ring = rng.choice(active, size=ring_size, replace=False)
            owner = np.sort(rng.integers(0, ring.size, n))
            seq = self._sequence(owner)
            rows = view.clone(view.source_rows(ring[owner], rng, channels=_RAILS))

            # Two to four redeemers, spread over a couple of categories: the
            # convergence a genuine gift-card buyer never produces.
            chosen_mccs = rng.choice(_REDEEM_MCCS, size=int(rng.integers(2, 4)), replace=False)
            redeemers = np.concatenate(
                [
                    rng.choice(redeem_pools[m], size=max(1, 4 // len(chosen_mccs)), replace=False)
                    for m in chosen_mccs
                ]
            )
            view.retarget(rows, rng.choice(redeemers, size=n).astype(object))

            rows["amount"] = rng.choice(_DENOMINATIONS, size=n, p=_DENOMINATION_P).astype(float)

            # A burst per customer: half an hour to three hours end to end.
            session = starts[c] + rng.integers(0, 4 * 86_400, ring.size)[owner]
            gap = rng.integers(8 * 60, 55 * 60, n)
            view.set_timestamps(rows, session + seq * gap, rng=rng, groups=owner)

            blocks.append(
                view.finalise(
                    rows,
                    card_id=self.card_id,
                    campaigns=np.full(n, campaign_id(self.card_id, c), dtype=object),
                    rng=rng,
                )
            )
        return pd.concat(blocks, ignore_index=True)

    @staticmethod
    def _sequence(owner: np.ndarray) -> np.ndarray:
        """Position of each purchase inside its customer's burst."""
        starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
        group_start = np.repeat(starts, np.diff(np.r_[starts, owner.size]))
        return np.arange(owner.size) - group_start


inject = card_entry_point(StoredValueAttack)


def main() -> None:
    """Print a sample ring. Run: ``python -m mantis.foundry.injectors.f6_40_stored_value``."""
    demo_main(StoredValueAttack)


if __name__ == "__main__":
    main()
