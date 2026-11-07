"""F6-38 — mule network: many small transfers fan in, then disperse.

The attack
----------
Proceeds enter a small number of collection accounts from a wide set of mules,
dwell for hours rather than days, and are pushed straight back out as ordinary
purchases. Every hop is a small transfer that would never be flagged alone. The
structure is only visible *as a structure*, which is why this card is the
clearest argument in the atlas for having a graph layer at all.

Modelling decisions worth defending
-----------------------------------
* **Fan-in, then dispersal, with a short dwell between them.** Collection rows
  land inside a two-day window on two or three long-tail person-to-person
  accounts; dispersal rows follow within hours, at a different, smaller set of
  cash-out customers and merchants. ``hop_dwell_time_seconds`` and
  ``funds_flow_fanout_ratio`` exist because of that ordering.
* **Only about three fifths of the volume is on the person-to-person category.**
  A network that transacted exclusively on MCC 6012 would make one category
  indicator a near-perfect classifier; a real cash-out leg buys things. The
  remainder is ordinary retail and top-up spend at long-tail merchants.
* **Mule accounts are existing customers with real histories.** Mules are
  recruited, not manufactured — the account was genuine before it was rented,
  and modelling it as brand new would hand the detector a novelty shortcut.
* **Transfer amounts are more regular than genuine traffic** — a narrow mid
  band, heavily round-snapped — which is the ``interarrival_regularity`` and
  round-number evidence, without ever leaving the population's own support.

Realism check (measured, not asserted)
--------------------------------------
Best single-feature depth-1 stump AUC: **0.787** (``mcc=6012``) — category
concentration on the person-to-person rail. That is a genuine property of mule
traffic rather than a generator artefact, and it is why only three fifths of the
volume rides that category. ``amount`` reaches 0.690. The recall has to come from
the topology.

Measured by ``mantis.foundry.injectors.probe`` against a 200k-event background at
seed 1337, over every column an issuer can read off one authorisation message.
Re-measure with ``python -m mantis.foundry --attacks F6-38``.
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

__all__ = ["MuleNetworkAttack", "inject", "main"]

_P2P_MCC: str = "6012"

#: Share of campaign volume on the collection (fan-in) leg.
_COLLECTION_SHARE: float = 0.62

#: Amount band for a hop: small enough to be beneath notice, large enough to be
#: worth moving.
_HOP_BAND: tuple[float, float] = (0.30, 0.72)

#: Categories the cash-out leg spends into.
_CASHOUT_MCCS: tuple[str, ...] = ("5999", "5311", "4814", "5732", "5651", "5734")


@register
class MuleNetworkAttack(BaseAttack):
    """Wide fan-in to a few collectors, then rapid narrow dispersal."""

    card_id = "F6-38"
    base_events = 180
    base_campaigns = 5

    def inject(
        self, population: pd.DataFrame, intensity: float, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Emit fan-in / dispersal topologies with short dwell at each hop."""
        view = self.view
        active = view.customers[view.customers["n_events"] >= 4].index.to_numpy()
        collectors_pool = view.merchants_in_mcc(_P2P_MCC, popularity=(0.0, 0.40))
        cashout_pools = {m: view.merchants_in_mcc(m, popularity=(0.0, 0.75)) for m in _CASHOUT_MCCS}

        counts = split_count(self.n_events(intensity), self.n_campaigns(intensity), rng)
        starts = view.spread_epochs(len(counts), rng)

        blocks: list[pd.DataFrame] = []
        for c, n in enumerate(counts):
            n = int(n)
            n_collect = max(1, round(n * _COLLECTION_SHARE))
            n_disperse = max(1, n - n_collect)
            n = n_collect + n_disperse

            # Wide at the top, narrow at the bottom: that ratio is the topology.
            mules = rng.choice(
                active,
                size=int(min(np.clip(round(n_collect / 1.8), 10, 45), active.size)),
                replace=False,
            )
            cashout = rng.choice(mules, size=max(2, mules.size // 6), replace=False)

            collect_owner = rng.integers(0, mules.size, n_collect)
            disperse_owner = rng.integers(0, cashout.size, n_disperse)
            customers = np.concatenate([mules[collect_owner], cashout[disperse_owner]])
            rows = view.clone(
                view.source_rows(customers, rng, channels=("upi_p2p", "upi_p2m", "ecom", "agentic"))
            )

            collectors = rng.choice(
                collectors_pool, size=max(2, min(3, collectors_pool.size)), replace=False
            )
            cashout_mccs = rng.choice(_CASHOUT_MCCS, size=n_disperse)
            merchants = np.concatenate(
                [
                    rng.choice(collectors, size=n_collect),
                    np.asarray(
                        [
                            cashout_pools[m][rng.integers(0, cashout_pools[m].size)]
                            for m in cashout_mccs
                        ],
                        dtype=object,
                    ),
                ]
            ).astype(object)
            view.retarget(rows, merchants)

            mcc = rows["mcc"].to_numpy()
            amount = view.draw_amounts(mcc, *_HOP_BAND, rng)
            # Layering hops are dictated by a script, so they snap hard.
            step = np.where(amount < 10_000, 500.0, 1_000.0)
            amount = np.where(
                rng.random(n) < 0.55, np.maximum(step, np.round(amount / step) * step), amount
            )
            rows["amount"] = np.round(amount, 2)

            # Collection over about two days; dispersal within hours of it.
            collect_ts = starts[c] + rng.integers(0, 2 * 86_400, n_collect)
            dwell = rng.integers(20 * 60, 10 * 3_600, n_disperse)
            disperse_ts = int(collect_ts.max()) - rng.integers(0, 6 * 3_600, n_disperse) + dwell
            groups = np.concatenate([np.arange(n_collect), np.full(n_disperse, n_collect)])
            view.set_timestamps(
                rows, np.concatenate([collect_ts, disperse_ts]), rng=rng, groups=groups
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


inject = card_entry_point(MuleNetworkAttack)


def main() -> None:
    """Print a sample network. Run: ``python -m mantis.foundry.injectors.f6_38_mule_fanout``."""
    demo_main(MuleNetworkAttack)


if __name__ == "__main__":
    main()
