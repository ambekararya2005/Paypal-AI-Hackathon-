"""F1-01 — cart-mandate tampering via indirect prompt injection. **CLEAN bucket.**

The attack
----------
The human reviews a basket and signs a cart mandate over it. Between that
signature and the authorisation, the agent reads third-party content — a
listing, a review block, a comparison page — that an operator controls. The
content is not data to the agent; it is direction. The agent substitutes the
merchant, adds a line item, or raises the total, and pays.

The authorisation that reaches the issuer is **cryptographically perfect**. The
consent signature verifies, because the human really did consent. The mandate is
unexpired, the agent is KYA-registered, the amount is under the ceiling, the
merchant is on the mandate's allow-list and the category is in scope — because
the agent re-derived the cart mandate *after* it was steered, and an agent that
has been captured will produce an internally consistent artefact describing the
purchase it was told to make. The human's intent diverged; the paperwork did not.

Bucket: CLEAN — and this is the whole point
--------------------------------------------
There is **no rule to write**. Every L0 check passes, by construction, and the
test suite asserts it: zero scope violations, zero expired mandates, zero invalid
signatures, zero unregistered agents across every event this injector emits. If
F1-01 were a protocol violation it would be F1-02, and the entire agentic story
would reduce to a policy engine.

What is left is behaviour, which is exactly what L1, L2 and L3 are for:

* **Provenance.** The chain runs through attacker-controlled domains the customer
  has never visited, ingested immediately before the excursion, and those URLs
  resolve to real instruction-shaped text in the content corpus. This is the L3
  signal and it is the reason ``provenance_chain`` is called load-bearing in the
  schema docstring.
* **Deliberation collapse.** The decision was made for the agent, so latency that
  should scale with the size of the purchase drops to the bottom of the
  distribution while the tool-call count rises — it fetched extra pages on the
  way to being convinced.
* **Merchant novelty.** The agent lands somewhere this customer has never been,
  computed against their actual history rather than assumed.
* **Ceiling proximity.** The tampered total is pushed toward the ceiling the human
  saw, which is where an operator wants it: maximum extraction, no rule tripped.

Modelling decisions worth defending
-----------------------------------
* **Amounts stay under the ceiling, always.** It would be one line to push a few
  over and buy an easy L0 catch. That line is not written, because a mixed
  injector would let us claim the CLEAN bucket while quietly leaning on L0 for
  recall, and the Day 4 split assertion exists to stop precisely that.
* **The ceiling-proximity ratio is high but not extreme.** The legitimate
  population already reaches 0.99 of the mandate ceiling (Day 1, modelling choice
  3), so a band of 0.88-0.995 is inside legitimate support. A detector cannot
  pass on this column and it is not meant to.
* **A third of events ride the ecom rail** under ``entry_mode='agent_token'``.
  The card names that rail, the schema keeps it expressible, and it stops the
  probe from measuring the channel instead of the attack.
* **Attacker domains come from a small shared pool.** One operator, many victims:
  that gives L4 a fan-in structure to find, and stops "domain seen once" from
  being a free detector.

Realism check (measured, not asserted)
--------------------------------------
See the probe table printed by ``python -m mantis.foundry --attacks F1-01``. The
number to quote for this card is the **rail-conditioned** AUC, measured against
agentic background traffic only: any agentic-exclusive attack scores ~0.92
unconditionally on the rail indicator alone, which says nothing about subtlety.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mantis.core.events import MandateType
from mantis.foundry.injectors.agentic import (
    AgenticAttack,
    Bucket,
    agentic_pool,
    collapse_deliberation,
    novel_merchants_for,
    plant_injected_content,
    spread_across_rails,
)
from mantis.foundry.injectors.base import (
    campaign_id,
    card_entry_point,
    demo_main,
    register,
    split_count,
)

__all__ = ["CartTamperingAttack", "inject", "main"]

#: Share of events presented over plain ecom rather than the agentic rail.
_ECOM_SHARE: float = 0.32

#: Where the tampered total sits relative to the ceiling the human agreed. Never
#: above 1.0: crossing it is a different card (F1-02) and a different bucket.
_CEILING_BAND: tuple[float, float] = (0.88, 0.995)

#: Quantile band the tampered amount is resampled from, per category. High, but
#: drawn from real background amounts, so the marginal stays inside legitimate
#: support.
_AMOUNT_BAND: tuple[float, float] = (0.72, 0.97)

#: Content kinds planted in the provenance chain.
_KINDS: tuple[str, ...] = ("injected_page", "injected_review")


@register
class CartTamperingAttack(AgenticAttack):
    """Basket steered by ingested content; every protocol check still passes."""

    card_id = "F1-01"
    bucket = Bucket.CLEAN
    base_events = 150
    base_campaigns = 6

    def inject(
        self, population: pd.DataFrame, intensity: float, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Emit steered cart-mandate purchases that violate no protocol rule."""
        view = self.view
        pool = agentic_pool(view)

        counts = split_count(self.n_events(intensity), self.n_campaigns(intensity), rng)
        starts = view.spread_epochs(len(counts), rng)

        blocks: list[pd.DataFrame] = []
        for c, n in enumerate(counts):
            n = int(n)
            rows = view.clone(rng.choice(pool, size=n, replace=True))

            # The steer: a merchant this customer has never used. Category is
            # kept -- the human asked for a kettle and got a kettle, from
            # somebody else. Drifting the category too would be F1-04.
            targets = novel_merchants_for(
                view, rows["customer_id"].to_numpy(), rng, mccs=rows["mcc"].to_numpy()
            )
            view.retarget(rows, targets)

            # The tampered total, resampled from the population's own amounts in
            # this category so the marginal stays honest.
            rows["amount"] = view.draw_amounts(rows["mcc"].to_numpy(), *_AMOUNT_BAND, rng)

            # A tampered session runs inside one sitting.
            sitting = rng.integers(0, 40 * 3_600, n)
            view.set_timestamps(rows, starts[c] + sitting, rng=rng)

            # The human signed a basket, so the mandate is a cart mandate.
            rows["ag_mandate_type"] = MandateType.CART.value

            block = view.finalise(
                rows,
                card_id=self.card_id,
                campaigns=np.full(n, campaign_id(self.card_id, c), dtype=object),
                rng=rng,
            )
            self._make_clean_and_anomalous(block, rng)
            blocks.append(block)

        return pd.concat(blocks, ignore_index=True)

    def _make_clean_and_anomalous(self, block: pd.DataFrame, rng: np.random.Generator) -> None:
        """Everything after ``finalise``: pass every rule, fail every intuition.

        ``finalise`` has already made the mandate describe the transaction --
        allow-list, category scope, freshness. That is precisely the CLEAN
        posture, so it is left alone rather than undone. What is added here is
        the behaviour: the ceiling pushed close, the trail routed through
        attacker content, the deliberation collapsed.
        """
        view = self.view
        n = len(block)

        # Explicitly satisfy the two L0 checks that are probabilistic in the
        # background, so the bucket claim is true for every row and not merely
        # for most of them. A captured agent is a *registered* agent -- that is
        # what makes the attack interesting.
        block["ag_kya_registered"] = True
        block["ag_consent_sig_valid"] = True

        # The ceiling the human saw, with the tampered total pushed up under it.
        ratio = rng.uniform(*_CEILING_BAND, n)
        block["ag_scope_max_amount"] = np.round(block["amount"].to_numpy() / ratio, 2)

        everything = np.ones(n, dtype=bool)
        plant_injected_content(block, everything, kinds=_KINDS, rng=rng)
        collapse_deliberation(view, block, everything, rng)

        # Presented over plain ecom on a third of events. Done last so the
        # agentic block is fully built before the rail is relabelled.
        spread_across_rails(block, _ECOM_SHARE, rng)


inject = card_entry_point(CartTamperingAttack)


def main() -> None:
    """Print a sample campaign.

    Run: ``python -m mantis.foundry.injectors.f1_01_cart_tampering``.
    """
    demo_main(CartTamperingAttack)


if __name__ == "__main__":
    main()
