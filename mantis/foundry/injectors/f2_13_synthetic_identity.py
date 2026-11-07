"""F2-13 — GenAI synthetic identity onboarding, seen from the payment stream.

The attack
----------
A batch of fabricated customers passes onboarding and starts transacting. The
document-forensics half of this fraud sits upstream of the authorisation message
and this injector does not pretend otherwise. What the stream *does* show is a
cohort whose first transactions arrive suspiciously soon after they appear,
whose devices and network paths overlap far more than unrelated strangers'
should, and whose spend ramps on a curve no organic customer follows.

Modelling decisions worth defending
-----------------------------------
* **The cohort is drawn from the thin-history tail of the real population, not
  invented.** ``TxEvent`` has no account-open date, so "new account" has to be
  inferred the way an issuer infers it: from first-seen-in-window. The injector
  picks customers whose earliest observed authorisation is late in the window
  and few in number, then treats them as the synthetic batch. Minting fresh
  customer ids would have made ``customer_id`` novelty a perfect detector and
  taught the model nothing.
* **Infrastructure overlap is the load-bearing signal.** Roughly half the
  cohort's events are stamped with a device id belonging to another cohort
  member, and a third with that member's IP prefix. Device reuse across
  customers with no other relationship is the one thing a batch of individually
  unique synthetics cannot avoid — it is why the card names ``device_reuse_count``
  and ``identity_cluster_synthetic_score``.
* **The ramp is monotone but noisy.** Spend climbs from the 5th toward the 80th
  percentile of each category over the cultivation window, with enough jitter
  that a single amount threshold cannot find it. Organic customers wander;
  this cohort only goes up.
* **The agentic slice runs unregistered more often than the population does**
  (about a third, against a 2.8% legitimate base rate) — a synthetic identity
  paired with an agent has no real KYA relationship to present. It is a minority
  of the attack's rows, so it does not turn ``ag_kya_registered`` into the
  answer on its own.

Realism check (measured, not asserted)
--------------------------------------
Best single-feature depth-1 stump AUC: **0.639** (``channel=card_present``) — and that column is
not behaviour at all, it is the fact that a cohort sharing a device transacts
remotely. The strongest behavioural column is ``amount`` at 0.57. Detecting this
needs cohort-level features: device reuse across unrelated customers, and the
shape of the ramp. That is the card's whole argument.

Measured by ``mantis.foundry.injectors.probe`` against a 200k-event background at
seed 1337, over every column an issuer can read off one authorisation message.
Re-measure with ``python -m mantis.foundry --attacks F2-13``.
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

__all__ = ["SyntheticIdentityAttack", "inject", "main"]

#: Rails that carry a device id. Card-present is excluded on purpose: the
#: authorisation has a terminal, not a device, so it cannot show device reuse.
_RAILS: tuple[str, ...] = ("ecom", "agentic", "upi_p2m", "upi_p2p", "recurring")

#: Share of cohort events stamped with a shared device / shared IP prefix.
_SHARED_DEVICE_P: float = 0.48
_SHARED_IP_P: float = 0.32

#: Cultivation ramp: quantile band of the category amount distribution at the
#: start and at the end of the window.
_RAMP_START: tuple[float, float] = (0.03, 0.22)
_RAMP_END: tuple[float, float] = (0.55, 0.88)

#: Fraction of the cohort's agentic events with no valid KYA registration.
_UNREGISTERED_P: float = 0.34


@register
class SyntheticIdentityAttack(BaseAttack):
    """A correlated batch of fabricated identities cultivating limit."""

    card_id = "F2-13"
    base_events = 170
    base_campaigns = 4

    def inject(
        self, population: pd.DataFrame, intensity: float, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Emit cohorts with overlapping infrastructure and a monotone spend ramp."""
        view = self.view
        # "Looks newly onboarded" = thin history, measured the way an issuer
        # would. Deliberately *not* filtered on a late first-seen date: an
        # earlier draft did that, and it pushed the whole cohort into the back
        # half of the window, which the probe read as a 0.81-AUC ``ts_epoch``
        # signal. Fraud that only happens in August is a generator artefact.
        thin = view.customers.nsmallest(max(40, int(len(view.customers) * 0.30)), "n_events")
        candidates = thin.index.to_numpy()

        merchant_ids = view.merchants.index.to_numpy()
        merchant_p = view.merchants["n_events"].to_numpy(dtype=float)
        merchant_p /= merchant_p.sum()

        counts = split_count(self.n_events(intensity), self.n_campaigns(intensity), rng)
        blocks: list[pd.DataFrame] = []
        for c, n in enumerate(counts):
            n = int(n)
            cohort_size = int(np.clip(round(n / 9.0), 8, 22))
            cohort = rng.choice(candidates, size=min(cohort_size, candidates.size), replace=False)
            owner = rng.integers(0, cohort.size, n)
            rows = view.clone(view.source_rows(cohort[owner], rng, channels=_RAILS))

            merchants = merchant_ids[rng.choice(merchant_ids.size, size=n, p=merchant_p)]
            view.retarget(rows, merchants)

            # Cultivation runs from each identity's own first appearance, so the
            # gap between "account opened" and "first transaction" is short by
            # construction — the account_age_at_first_txn signal.
            first_seen = view.customers.loc[cohort, "first_seen"].to_numpy()[owner]
            cultivation = int(rng.integers(24, 46)) * 86_400
            progress = np.sort(rng.random(n))
            view.set_timestamps(
                rows,
                first_seen
                + rng.integers(2 * 3_600, 3 * 86_400, n)
                + (progress * cultivation).astype(np.int64),
                rng=rng,
            )

            # The ramp: interpolate the resampling band upward with progress.
            mcc = rows["mcc"].to_numpy()
            lo = _RAMP_START[0] + progress * (_RAMP_END[0] - _RAMP_START[0])
            hi = _RAMP_START[1] + progress * (_RAMP_END[1] - _RAMP_START[1])
            amount = np.empty(n, dtype=float)
            for k in range(n):
                amount[k] = view.draw_amounts(mcc[k : k + 1], float(lo[k]), float(hi[k]), rng)[0]
            rows["amount"] = amount

            self._share_infrastructure(rows, cohort, owner, rng)

            agentic = (rows["channel"] == "agentic").to_numpy()
            unregistered = agentic & (rng.random(n) < _UNREGISTERED_P)
            if unregistered.any():
                rows.loc[unregistered, "ag_kya_registered"] = False
                rows.loc[unregistered, "ag_kya_token"] = None

            blocks.append(
                view.finalise(
                    rows,
                    card_id=self.card_id,
                    campaigns=np.full(n, campaign_id(self.card_id, c), dtype=object),
                    rng=rng,
                )
            )
        return pd.concat(blocks, ignore_index=True)

    def _share_infrastructure(
        self,
        rows: pd.DataFrame,
        cohort: np.ndarray,
        owner: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        """Stamp one cohort member's device and IP prefix across the batch.

        This is the linkage that survives when every fabricated identity is
        individually unique: the documents differ, the infrastructure does not.
        """
        view = self.view
        anchor = str(cohort[rng.integers(0, cohort.size)])
        anchor_rows = view.frame.iloc[view.rows_by_customer[anchor]]
        devices = anchor_rows["device_id"].dropna().unique()
        ips = anchor_rows["ip"].dropna().unique()
        if devices.size == 0 or ips.size == 0:
            return

        device = str(devices[rng.integers(0, devices.size)])
        prefix = ".".join(str(ips[rng.integers(0, ips.size)]).split(".")[:3])

        n = len(rows)
        has_device = rows["device_id"].notna().to_numpy()
        share_device = has_device & (rng.random(n) < _SHARED_DEVICE_P) & (owner != 0)
        rows.loc[share_device, "device_id"] = device

        has_ip = rows["ip"].notna().to_numpy()
        share_ip = has_ip & (rng.random(n) < _SHARED_IP_P)
        if share_ip.any():
            hosts = rng.integers(1, 254, int(share_ip.sum()))
            rows.loc[share_ip, "ip"] = [f"{prefix}.{h}" for h in hosts]


inject = card_entry_point(SyntheticIdentityAttack)


def main() -> None:
    """Print a sample cohort.

    Run: ``python -m mantis.foundry.injectors.f2_13_synthetic_identity``.
    """
    demo_main(SyntheticIdentityAttack)


if __name__ == "__main__":
    main()
