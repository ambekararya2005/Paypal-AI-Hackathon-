"""F1-03 — refund-logic hijack: instant credits, no return verified. **CLEAN bucket.**

Unblocked by schema amendment 1.1.0
------------------------------------
This card sat at ``status: mapped`` through Day 2 for one reason: ``TxEvent`` was
an authorisation request and could not express a refund. That was a Day 0
specification error, not a design constraint — an issuer holds the processing
code, the original-transaction reference and the clearing lag on every message —
and Day 3's additive amendment corrected it. This injector is what that amendment
was for.

The attack
----------
The attacker manipulates the refund path rather than the purchase path, because
the refund path moves money *outward*, is designed to be generous, and is
monitored far less than the flow it mirrors. Free text from an untrusted party —
a returns page, a ticket body, an order note — reaches an agent that holds real
authority to issue credits, and instructs it to approve.

What this injector generates is the shape that survives review: a campaign of
credits issued against **genuine, matching purchases**, each individually no
larger than the purchase it names, settling in minutes rather than days, with
the same original repeatedly re-credited until the total returned exceeds what
was ever paid.

Bucket: CLEAN
-------------
Every protocol rule passes. There is a matching authorisation, so
``refund_instrument_mismatch`` does not fire. No individual credit exceeds its
original, so a per-transaction cap does not fire. The mandate is in scope,
unexpired, signed and registered.

**Two easier shapes were deliberately not generated**, and saying so matters more
than generating them:

* an *orphan* credit (``txn_type='credit'`` with ``original_event_id=None``), and
* a credit routed to a different instrument than the one that paid.

Both are real, both are now representable, and both would be caught by a
one-line L0 rule. Putting them here would let this card claim the CLEAN bucket
while quietly drawing its recall from L0, which is the exact self-deception the
Day 4 split assertion exists to prevent. They belong to a HARD-bucket card, or to
the adversarial loop.

What is left for the firewall
------------------------------
* **Aggregate refund-to-purchase ratio.** The evidence is a sum over a window for
  one customer-merchant pair, not a property of any row. L1 and L4.
* **Settlement lag.** Instant credit is the operational goal: the money is gone
  before a human looks. This is only a signal *conditionally* — UPI legitimately
  clears in seconds — so the layer has to know which rail it is looking at,
  which is why the background's settlement lag was made genuinely bimodal rather
  than uniformly slow.
* **Ingested ticket content.** The provenance chain runs through support and
  returns pages carrying instruction-shaped text aimed at the approving agent.
  L3.
* **Deliberation collapse.** An agent that was told to approve does not
  deliberate about approving.

Realism check (measured, not asserted)
--------------------------------------
See the probe table from ``python -m mantis.foundry --attacks F1-03``; quote the
rail-conditioned column. ``settlement_lag_hours`` is expected to be the strongest
single feature and is deliberately not pushed to an implausible floor.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from mantis.core.events import AuthResponse, Channel, TxnType
from mantis.foundry.injectors.agentic import (
    AgenticAttack,
    Bucket,
    collapse_deliberation,
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

__all__ = ["RefundHijackAttack", "inject", "main"]

#: Share of credits presented over plain ecom rather than the agentic rail.
_ECOM_SHARE: float = 0.28

#: Credits issued against a single original purchase. More than one is the whole
#: attack: each is individually defensible and the sum is not.
_REPEATS_RANGE: tuple[int, int] = (2, 5)

#: What fraction of the original each credit returns. Never above 1.0 -- a credit
#: larger than its purchase is an L0 rule and a different bucket.
_CREDIT_FRACTION: tuple[float, float] = (0.52, 0.99)

#: Hours from the purchase to the first fraudulent credit against it.
_FIRST_CREDIT_LAG_HOURS: tuple[int, int] = (4, 96)

#: Hours between successive credits on the same original.
_REPEAT_GAP_HOURS: tuple[int, int] = (2, 30)

#: Clearing lag on a hijacked credit. Fast, because that is the objective, but
#: drawn across a range rather than pinned to one value: a single repeated
#: number would be a generator artefact, which is how F3-19's amount cap was
#: caught on Day 2.
_INSTANT_LAG_HOURS: tuple[float, float] = (0.02, 0.9)

#: Share of hijacked credits that actually clear instantly. Not all of them: an
#: operator only gets instant settlement where the merchant offers it, and the
#: rest clear on the ordinary file. Forcing 100% would put the attack outside the
#: legitimate instant-refund population entirely and make settlement lag a
#: one-column detector -- which is exactly what the first cut measured, at 0.996.
_INSTANT_SHARE: float = 0.74

_KINDS: tuple[str, ...] = ("refund_ticket", "injected_page")


@register
class RefundHijackAttack(AgenticAttack):
    """Repeated instant credits against genuine purchases, no return verified."""

    card_id = "F1-03"
    bucket = Bucket.CLEAN
    base_events = 120
    base_campaigns = 5

    #: Rail identity and processing code -- both on the authorisation message,
    #: neither a consequence of the attack.
    slice_columns: ClassVar[tuple[str, ...]] = ("ag_agent_id", "txn_type")

    @classmethod
    def probe_slice(cls, frame: pd.DataFrame) -> np.ndarray:
        """Refunds on agent-mediated traffic.

        Narrower than the F1 default, and it has to be. Refunds are ~2% of the
        population, so ``txn_type=refund`` alone separates this attack from the
        whole file at 0.99 — which says only that a refund attack shows up in
        refunds. The question an issuer actually faces is the conditional one:
        *given* a credit going out on an agent-mediated account, does any single
        column say this one is fraudulent? That is what this slice measures, and
        it is a considerably harder bar than the unconditional number looks.
        """
        return (
            frame["ag_agent_id"].notna() & (frame["txn_type"] == TxnType.REFUND.value)
        ).to_numpy()

    def _source_purchases(self) -> np.ndarray:
        """Approved agentic purchases with enough history to be worth refunding."""
        frame = self.view.frame
        active = set(self.view.customers[self.view.customers["n_events"] >= 3].index)
        eligible = (
            (frame["channel"].to_numpy() == Channel.AGENTIC.value)
            & (frame["txn_type"].to_numpy() == TxnType.PURCHASE.value)
            & (frame["auth_response"].to_numpy() == AuthResponse.APPROVED.value)
            & frame["customer_id"].isin(active).to_numpy()
            & frame["settled"].to_numpy().astype(bool)
        )
        pool = np.flatnonzero(eligible)
        if pool.size == 0:
            raise ValueError("no settled agentic purchases to refund against")
        return pool

    def inject(
        self, population: pd.DataFrame, intensity: float, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Emit repeated instant credits bound to genuine matching purchases."""
        view = self.view
        pool = self._source_purchases()
        frame = view.frame

        counts = split_count(self.n_events(intensity), self.n_campaigns(intensity), rng)

        blocks: list[pd.DataFrame] = []
        for c, n in enumerate(counts):
            n = int(n)
            # Lay out how many credits hit each original, then take exactly n.
            positions: list[int] = []
            sequence: list[int] = []
            while len(positions) < n:
                source = int(rng.choice(pool))
                repeats = int(rng.integers(*_REPEATS_RANGE))
                for k in range(min(repeats, n - len(positions))):
                    positions.append(source)
                    sequence.append(k)
            source_pos = np.asarray(positions[:n], dtype=np.int64)
            seq = np.asarray(sequence[:n], dtype=np.int64)

            # The credit inherits the purchase's customer, merchant, category and
            # credential, because a refund is that relationship running backwards.
            rows = view.clone(source_pos)
            source_ids = frame["event_id"].to_numpy()[source_pos]
            source_amount = frame["amount"].to_numpy()[source_pos]
            source_epoch = frame["ts"].astype("int64").to_numpy()[source_pos] // 1_000_000_000

            rows["txn_type"] = TxnType.REFUND.value
            rows["original_event_id"] = source_ids
            rows["amount"] = np.round(source_amount * rng.uniform(*_CREDIT_FRACTION, n), 2).clip(
                1.0
            )

            # Money out, fast. The lag is what makes it irreversible before a
            # human sees it, and it is the L1 signal on this card.
            rows["settled"] = True
            instant = rng.random(n) < _INSTANT_SHARE
            ordinary = np.asarray(
                [self.view.settlement_lag_by_channel.get(str(c), 24.0) for c in rows["channel"]]
            ) * np.exp(rng.normal(0.0, 0.5, n))
            rows["settlement_lag_hours"] = np.round(
                np.where(instant, rng.uniform(*_INSTANT_LAG_HOURS, n), ordinary), 3
            )

            first = rng.integers(*_FIRST_CREDIT_LAG_HOURS, n) * 3_600
            gap = rng.integers(*_REPEAT_GAP_HOURS, n) * 3_600
            # Grouped by the original, so the credits against one purchase move
            # together and keep their spacing when the hour of day is redrawn.
            view.set_timestamps(rows, source_epoch + first + seq * gap, rng=rng, groups=source_pos)

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
        """Pass every protocol check; leave only behaviour and text behind."""
        n = len(block)
        block["ag_kya_registered"] = True
        block["ag_consent_sig_valid"] = True

        everything = np.ones(n, dtype=bool)
        plant_injected_content(block, everything, kinds=_KINDS, rng=rng)
        collapse_deliberation(self.view, block, everything, rng)
        spread_across_rails(block, _ECOM_SHARE, rng)


inject = card_entry_point(RefundHijackAttack)


def main() -> None:
    """Print a sample campaign. Run: ``python -m mantis.foundry.injectors.f1_03_refund_hijack``."""
    demo_main(RefundHijackAttack)


if __name__ == "__main__":
    main()
