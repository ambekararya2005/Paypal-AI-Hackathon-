"""F6-39 — transaction laundering: real volume pushed through a miscoded merchant.

The attack
----------
The attacker controls the merchant of record. Proceeds are pushed through it and
emerge as ordinary settlement for goods that were never shipped. Every
authorisation is well-formed, and the customers may even be genuine. What is
wrong is the merchant: it is registered under one category and processing the
ticket profile of a completely different one.

Modelling decisions worth defending
-----------------------------------
* **The tell is ``amount`` *given* ``mcc``, never ``amount`` alone.** Ticket
  sizes are resampled from a high-ticket reference category — electronics,
  hotels, travel — while the merchant stays registered under a low-ticket one:
  dining, quick service, books, cinema, small retail. A ₹4,200 authorisation is
  utterly ordinary on the network and completely wrong at a quick-service
  restaurant. That is a two-column interaction by construction, which is exactly
  why a depth-1 stump cannot find it and an L1 model can.
* **The reference band is the middle of the high-ticket distribution, not its
  tail.** Laundering wants throughput, not attention. Drawing from the top decile
  would have pushed every row above the population's 95th percentile and made
  the amount column sufficient on its own.
* **Shell merchants come from the genuine long tail.** Real transaction
  laundering runs through a merchant that already exists and already has an
  acquiring relationship. Inventing a merchant id would have made
  ``merchant_novelty`` a perfect detector.
* **A narrow, repetitive customer base.** Twelve to twenty-four customers,
  transacting repeatedly at the same one or two merchants over weeks — a
  bipartite pattern no genuine retailer of that size sustains, and the reason
  the card names ``merchant_customer_bipartite_anomaly``.
* **Amounts are unusually round and unusually uniform**, because they are
  settlement figures rather than baskets.

Realism check (measured, not asserted)
--------------------------------------
Best single-feature depth-1 stump AUC: **0.792** (``amount``) — the ticket uplift, seen
without its category. Nothing in the single-column view sees the *miscoding*,
which is the attack: ₹3,100 is an unremarkable authorisation and an absurd
quick-service restaurant bill, and only the pair says so.

Measured by ``mantis.foundry.injectors.probe`` against a 200k-event background at
seed 1337, over every column an issuer can read off one authorisation message.
Re-measure with ``python -m mantis.foundry --attacks F6-39``.
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

__all__ = ["ShellMerchantAttack", "inject", "main"]

#: Categories a shell merchant registers under: low ticket, high volume, dull.
_DECLARED_MCCS: tuple[str, ...] = ("5812", "5814", "5942", "7832", "5999", "5977")

#: Categories whose ticket profile the realised volume actually matches.
_REFERENCE_MCCS: tuple[str, ...] = ("5732", "7011", "4511", "5311", "5651")

#: Band of the reference category resampled for ticket size. Mid, not tail:
#: laundering wants throughput, not attention.
_TICKET_BAND: tuple[float, float] = (0.10, 0.52)

#: A merchant of record the attacker controls can hold a terminal as easily as
#: a checkout page, so card-present is in scope. Leaving it out gave every row
#: the same "never card-present" tell.
_RAILS: tuple[str, ...] = ("ecom", "agentic", "upi_p2m", "card_present")


@register
class ShellMerchantAttack(BaseAttack):
    """Attacker-controlled merchant of record accepting miscoded volume."""

    card_id = "F6-39"
    base_events = 150
    base_campaigns = 5

    def inject(
        self, population: pd.DataFrame, intensity: float, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Emit narrow, repetitive volume through long-tail miscoded merchants."""
        view = self.view
        active = view.customers[view.customers["n_events"] >= 4].index.to_numpy()
        shell_pools = {m: view.merchants_in_mcc(m, popularity=(0.0, 0.30)) for m in _DECLARED_MCCS}

        counts = split_count(self.n_events(intensity), self.n_campaigns(intensity), rng)
        starts = view.spread_epochs(len(counts), rng, pad_days=6.0)

        blocks: list[pd.DataFrame] = []
        for c, n in enumerate(counts):
            n = int(n)
            declared = str(rng.choice(_DECLARED_MCCS))
            reference = str(rng.choice(_REFERENCE_MCCS))
            pool = shell_pools[declared]
            shells = rng.choice(pool, size=max(1, min(2, pool.size)), replace=False)

            base = rng.choice(
                active, size=int(min(np.clip(round(n / 7.0), 12, 24), active.size)), replace=False
            )
            owner = rng.integers(0, base.size, n)
            rows = view.clone(view.source_rows(base[owner], rng, channels=_RAILS))
            view.retarget(rows, rng.choice(shells, size=n).astype(object))

            # The miscoding: the ticket profile of the reference category, filed
            # under the declared one.
            amount = view.draw_amounts(np.full(n, reference, dtype=object), *_TICKET_BAND, rng)
            step = np.where(amount < 5_000, 100.0, 500.0)
            amount = np.where(
                rng.random(n) < 0.58, np.maximum(step, np.round(amount / step) * step), amount
            )
            rows["amount"] = np.round(amount, 2)

            # Settlement volume runs steadily for weeks, not in a burst.
            view.set_timestamps(
                rows, starts[c] + rng.integers(0, int(rng.integers(18, 40)) * 86_400, n), rng=rng
            )

            blocks.append(
                view.finalise(
                    rows,
                    card_id=self.card_id,
                    campaigns=np.full(n, campaign_id(self.card_id, c), dtype=object),
                    rng=rng,
                )
            )
        return pd.concat(blocks, ignore_index=True)


inject = card_entry_point(ShellMerchantAttack)


def main() -> None:
    """Print a sample shell. Run: ``python -m mantis.foundry.injectors.f6_39_shell_merchant``."""
    demo_main(ShellMerchantAttack)


if __name__ == "__main__":
    main()
