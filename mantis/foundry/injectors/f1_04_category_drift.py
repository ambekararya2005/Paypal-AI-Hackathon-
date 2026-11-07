"""F1-04 — intent-mandate category drift. **HARD bucket.**

Note on the card id
-------------------
The Day 3 brief named F1-04 "merchant-endpoint impersonation via forged agent
cards". In the frozen atlas, F1-04 is **intent-mandate category drift**, and
F1-05 is delegation-chain laundering. This injector is written against the card
that actually exists, for the same reason four Day 2 injectors were remapped: an
injector that generates something other than what its card describes is exactly
the overclaim the registry assertion exists to prevent. Endpoint impersonation
has no card in the frozen 42 and would need one written first.

The attack
----------
The human authorises spend within a set of categories. The agent settles in a
category just *outside* that set, on a merchant whose acceptance code is
adjacent to the permitted ones. Every individual step is defensible: the amount
is comfortably within the ceiling, the mandate is unexpired and correctly signed,
the merchant is unconstrained because an intent mandate names no merchant. Only
the category is wrong.

Drift is a lower-yield attack than outright substitution, which is precisely why
it survives review. Nobody escalates a slightly-off purchase.

How this differs from F1-02, and why both exist
------------------------------------------------
F1-02 is the aggressive sibling: it drifts category *and* ramps the amount over
the ceiling across a campaign. F1-04 does **only** the category, and does it
quietly:

* the amount sits in the middle of the customer's own range, never near the
  ceiling — ``amount_to_scope_max_ratio`` is deliberately uninformative here;
* the drift is a single step to a genuinely adjacent category, not a scatter;
* the settled category is one the customer has **never transacted in**, computed
  against their real history, which is what makes ``mcc_novelty_for_customer``
  a true signal rather than a likely one.

Having both is the point. F1-02 shows what a greedy operator looks like; F1-04
shows what the same excursion looks like when the operator is patient. A detector
tuned on the first and evaluated on the second is the honest test, and it is
available because the two are separate cards with separate injectors.

Bucket: HARD
------------
``mandate_scope_violation`` is a single-message check: the settled MCC is not in
the list the mandate carried. The background contains zero such rows, so L0
should catch this at near-zero false positive rate.

What is left for the deeper layers is *ranking*: this is the lowest-value F1
attack, so an issuer that blocks every scope excursion outright will block a lot
of merely-sloppy agents too. ``scope_category_distance`` (how far the settled
category sits from the permitted set) and ``rare_feature_combination_score`` are
what separate "drifted to something adjacent" from "drifted to something
unrelated", and that distinction is a triage decision, not a block decision.

Realism check (measured, not asserted)
--------------------------------------
See the probe table from ``python -m mantis.foundry --attacks F1-04``, quoting
the in-slice column. A visible number is expected and correct: the attack is
meant to be catchable, and the claim is that L0 catches it cheaply.
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

__all__ = ["CategoryDriftAttack", "inject", "main"]

#: How many categories the signed mandate names. A real intent mandate is not a
#: single code -- "keep the house stocked" covers several -- and a one-category
#: scope would make the violation trivially wide.
_SCOPE_SIZE: tuple[int, int] = (2, 5)

#: Where the drifted amount sits in the category's own distribution. Middling on
#: purpose: this attack must not be catchable on amount, or it is F1-02.
_AMOUNT_BAND: tuple[float, float] = (0.30, 0.78)

#: Headroom left under the ceiling. Generous, so ``amount_to_scope_max_ratio``
#: carries no information here and the category clause has to do the work.
_CEILING_HEADROOM: tuple[float, float] = (1.35, 3.2)


@register
class CategoryDriftAttack(AgenticAttack):
    """A quiet single-step excursion outside the signed category list."""

    card_id = "F1-04"
    bucket = Bucket.HARD
    base_events = 100
    base_campaigns = 5

    def _novel_categories(
        self, customer_ids: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, list[list[str]]]:
        """A category each customer has never used, plus their habitual ones.

        The habitual set becomes the mandate's scope and the novel one becomes
        where the money went. Deriving both from the customer's actual history
        is what makes the drift real rather than asserted: a randomly chosen
        "outside" category would frequently be one they shop in every week.
        """
        view = self.view
        mcc_col = view.frame["mcc"].to_numpy()
        all_mccs = np.asarray(sorted(view.amounts_by_mcc), dtype=object)

        settled = np.empty(len(customer_ids), dtype=object)
        scopes: list[list[str]] = []
        for i, cid in enumerate(customer_ids):
            history = mcc_col[view.rows_by_customer[str(cid)]]
            seen = list(dict.fromkeys(str(m) for m in history))
            unseen = [m for m in map(str, all_mccs) if m not in seen]
            settled[i] = (
                unseen[rng.integers(0, len(unseen))]
                if unseen
                else str(all_mccs[rng.integers(0, all_mccs.size)])
            )
            # The mandate names what they habitually buy -- and not the one the
            # agent drifted into.
            size = int(rng.integers(*_SCOPE_SIZE))
            scope = [m for m in seen if m != settled[i]][:size]
            while len(scope) < 2:
                filler = str(all_mccs[rng.integers(0, all_mccs.size)])
                if filler != settled[i] and filler not in scope:
                    scope.append(filler)
            scopes.append(scope)
        return settled, scopes

    def inject(
        self, population: pd.DataFrame, intensity: float, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Emit single-step excursions outside a genuine intent mandate's scope."""
        view = self.view
        pool = agentic_pool(view)

        counts = split_count(self.n_events(intensity), self.n_campaigns(intensity), rng)
        starts = view.spread_epochs(len(counts), rng)

        blocks: list[pd.DataFrame] = []
        for c, n in enumerate(counts):
            n = int(n)
            rows = view.clone(rng.choice(pool, size=n, replace=True))

            settled, scopes = self._novel_categories(rows["customer_id"].to_numpy(), rng)
            # Retarget into the drifted category, on an ordinary merchant there.
            merchants = np.asarray(
                [
                    view.merchants_in_mcc(str(m), popularity=(0.15, 0.95))[
                        rng.integers(0, view.merchants_in_mcc(str(m), popularity=(0.15, 0.95)).size)
                    ]
                    for m in settled
                ],
                dtype=object,
            )
            view.retarget(rows, merchants)
            rows["amount"] = view.draw_amounts(rows["mcc"].to_numpy(), *_AMOUNT_BAND, rng)

            rows["ag_mandate_type"] = MandateType.INTENT.value
            view.set_timestamps(rows, starts[c] + rng.integers(0, 16 * 86_400, n), rng=rng)

            block = view.finalise(
                rows,
                card_id=self.card_id,
                campaigns=np.full(n, campaign_id(self.card_id, c), dtype=object),
                rng=rng,
            )
            self._drift(block, scopes, rng)
            blocks.append(block)

        return pd.concat(blocks, ignore_index=True)

    def _drift(
        self, block: pd.DataFrame, scopes: list[list[str]], rng: np.random.Generator
    ) -> None:
        """Restore the human's category list, which ``finalise`` had repaired.

        ``finalise`` folds the settled MCC into the scope so that ordinary
        injectors do not trip L0 for free. Here the excursion *is* the attack, so
        the signed list is put back — the categories the human actually named,
        which do not include where the money went.
        """
        n = len(block)
        block["ag_scope_categories"] = pd.Series(scopes, index=block.index, dtype=object)
        # An intent mandate names no merchant.
        block["ag_scope_allowed_merchants"] = pd.Series(
            [[] for _ in range(n)], index=block.index, dtype=object
        )
        # Ample headroom: the ceiling must not be what gives this away.
        headroom = rng.uniform(*_CEILING_HEADROOM, n)
        block["ag_scope_max_amount"] = np.round(block["amount"].to_numpy() * headroom, 2)


inject = card_entry_point(CategoryDriftAttack)


def main() -> None:
    """Print a sample campaign.

    Run: ``python -m mantis.foundry.injectors.f1_04_category_drift``.
    """
    demo_main(CategoryDriftAttack)


if __name__ == "__main__":
    main()
