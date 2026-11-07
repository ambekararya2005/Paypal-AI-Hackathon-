"""F4-27 — adaptive BIN attack: probe wide and cheap, then concentrate value.

The attack
----------
The operator holds credentials across a small number of BIN ranges and can see
which authorisations approve. Rather than maximising value per attempt, they
*search*: many low-value authorisations spread thinly across the BIN and across
an implausibly wide set of unrelated merchants, reading the outcome pattern to
locate where the decision boundary sits, then pushing higher-value attempts
through the merchants and categories that just approved.

Modelling decisions worth defending
-----------------------------------
* **Probe amounts are resampled from the population's own low quantile band**
  for whatever category the probe landed in, not drawn from an attacker-chosen
  distribution. A ₹40 probe at a jeweller would be a cartoon; a ₹40 probe at a
  bus operator is Tuesday. Sampling per-MCC keeps every probe individually
  ordinary, which is the attack's entire objective function.
* **Escalation reuses the merchants that approved.** The high-value tail goes
  back only through merchants whose probe came back approved — that is what
  "escalating on success" means when the only feedback channel is the
  approve/decline oracle. It also produces the merchant fan-in the L4 layer is
  supposed to see. Measured: **100% of escalation events land on a merchant that
  approved during that campaign's own probe phase.**

  This was wrong until Day 4 and worth recording rather than quietly fixing.
  The declines for a whole campaign were drawn in one vectorised pass *after*
  the escalation targets had been chosen, so escalation sampled from every
  merchant the probe had *touched*, approved or not — 65% of escalation events
  landed on a merchant that had approved, which is very close to what uniform
  resampling from a 44%-approval pool gives you by chance. The oracle existed in
  the data and was never read. The fix is an ordering one: decide the probe
  outcome first, then let it choose the targets.
* **Two to three BINs per campaign, different ones per campaign.** A campaign
  concentrated on one BIN would make a single BIN indicator a near-perfect
  classifier, which is a property of a lazy generator rather than of BIN attacks.

The oracle, now that the schema can carry it (amendment 1.1.0)
--------------------------------------------------------------
Until Day 3 this injector modelled only half the attack. ``TxEvent`` carried no
response code, so the *feedback channel* — the thing that makes this a search
rather than a spree — had nowhere to live, and the card's ``decline_ratio_1h``
signal was unmeasurable in our own data. The amendment fixed that, and the
attack is now shaped the way it really is:

* **The probe phase mostly fails.** Around 56% of search attempts are declined,
  overwhelmingly ``do_not_honor`` and ``insufficient_funds`` — the operator is
  spraying a finite set of stolen credentials across merchants and reading which
  combinations come back approved. The background declines at roughly 9%, so
  this is a real elevation without being a categorical giveaway. Measured on the
  200k gate run at seed 1337: **48.5% campaign-wide against an 8.8% background
  (5.5x), 51-66% inside the probe phase, 8% inside escalation** — decline-heavy,
  which is what card testing looks like, and the probe/escalation gap is now a
  causal consequence of the oracle rather than a coincidence of two constants.

  *Why 56% and not the 90% naive card testing shows.* Because this card is the
  **adaptive** BIN attack, and the adaptation is precisely the decision to stay
  under the decline-ratio and velocity rules every issuer already runs. An
  operator burning 90% of their attempts describes a campaign that is shut off
  within the hour and never reaches an escalation phase at all — there would be
  nothing left for a detector to be interesting about. A 5x elevation over
  background is high enough to be the signal the card names and low enough to
  survive the rule, which is the trade-off a competent operator actually makes.
  The naive high-decline shape is a *different* attack: it is F4-32
  (decision-boundary mapping through probe transactions), still ``mapped``.
* **The escalation phase mostly succeeds**, because it goes back through the
  merchants that just approved. That asymmetry *is* the signal: neither the
  decline rate nor the approval rate alone says much, but a burst of declines
  inside one BIN followed by high-value approvals through the same merchants is
  a shape no legitimate cardholder produces.
* Declines never settle, so ``settlement_lag_hours`` is null across most of the
  probe phase. That nullity is a real, free consequence, not a plant.

Realism check (measured, not asserted)
--------------------------------------
Best single-feature depth-1 stump AUC: **see the module-level table printed by
the probe**. The decline elevation is now the strongest single column, and it is
deliberately kept to a ~6x lift over background rather than a categorical tell:
a single declined authorisation is unremarkable, and only the per-BIN aggregate
is evidence. That is the claim the card makes.

Measured by ``mantis.foundry.injectors.probe`` against a 200k-event background at
seed 1337, over every column an issuer can read off one authorisation message.
Re-measure with ``python -m mantis.foundry --attacks F4-27``.
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

__all__ = ["AdaptiveBinAttack", "inject", "main"]

#: Rails an operator can submit card-not-present probes on. ``card_present`` is
#: included because probing a stolen credential at an unattended terminal is a
#: real technique, not because it is convenient.
_RAILS: tuple[str, ...] = ("ecom", "agentic", "card_present")

#: Share of a campaign's volume spent searching rather than extracting.
_PROBE_SHARE: float = 0.78

#: Quantile bands the two phases resample amounts from, per category.
_PROBE_BAND: tuple[float, float] = (0.02, 0.30)
_ESCALATION_BAND: tuple[float, float] = (0.70, 0.95)

#: Share of each phase that comes back declined. The gap between the two is the
#: attack: the operator is buying information with the probe phase and spending
#: it in the escalation phase.
_PROBE_DECLINE_P: float = 0.56
_ESCALATION_DECLINE_P: float = 0.09

#: What the issuer says when it refuses a probe. A stolen or guessed credential
#: fails generically far more often than it trips a risk model -- an operator who
#: saw ``declined_risk`` on every attempt would know they had been spotted.
_DECLINE_REASONS: tuple[str, ...] = (
    "declined_do_not_honor",
    "declined_insufficient_funds",
    "declined_invalid_cvv",
    "declined_risk",
)
_DECLINE_REASON_P: tuple[float, ...] = (0.47, 0.28, 0.17, 0.08)


@register
class AdaptiveBinAttack(BaseAttack):
    """Wide low-value search across a BIN range, then value concentration."""

    card_id = "F4-27"
    base_events = 200
    base_campaigns = 5

    def inject(
        self, population: pd.DataFrame, intensity: float, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Emit probe-then-escalate campaigns across two or three BINs each."""
        view = self.view
        card_bin = view.frame["card_bin"].to_numpy()
        rail_ok = np.isin(view.frame["channel"].to_numpy(), _RAILS)
        bins = np.unique(card_bin)

        merchant_ids = view.merchants.index.to_numpy()
        merchant_p = view.merchants["n_events"].to_numpy(dtype=float)
        merchant_p /= merchant_p.sum()

        counts = split_count(self.n_events(intensity), self.n_campaigns(intensity), rng)
        starts = view.spread_epochs(len(counts), rng)

        blocks: list[pd.DataFrame] = []
        for c, n in enumerate(counts):
            n = int(n)
            n_probe = max(1, round(n * _PROBE_SHARE))
            n_escalate = max(0, n - n_probe)

            target_bins = rng.choice(bins, size=int(rng.integers(2, 4)), replace=False)
            pool = np.flatnonzero(np.isin(card_bin, target_bins) & rail_ok)
            # A finite stolen-credential set: attempts are spread thinly over it,
            # which is what makes per-card velocity useless and per-BIN velocity
            # the feature that works.
            n_credentials = int(min(pool.size, rng.integers(28, 48)))
            credentials = rng.choice(pool, size=n_credentials, replace=False)
            rows = view.clone(rng.choice(credentials, size=n, replace=True))

            probe_merchants = merchant_ids[
                rng.choice(merchant_ids.size, size=n_probe, p=merchant_p)
            ]

            # THE ORACLE, READ IN THE RIGHT ORDER.
            # The probe outcome has to be decided *before* the escalation targets
            # are chosen, because the outcome is what chooses them. Deciding the
            # whole campaign's declines in one draw afterwards -- which is what
            # this injector did until Day 4 -- produced an escalation phase that
            # went back through merchants the campaign had merely *touched*,
            # approved or not. That makes the feedback channel decorative: the
            # decline sequence is generated but never read, and "adaptive" is a
            # claim in the docstring rather than a property of the data.
            probe_declined = rng.random(n_probe) < _PROBE_DECLINE_P
            approved_probe = probe_merchants[~probe_declined]
            if approved_probe.size == 0:
                # A campaign whose every probe failed has learned nothing and has
                # nowhere to escalate to. Rare at 56%, but it must not crash, and
                # it must not silently escalate anyway.
                approved_probe = probe_merchants
            escalate_merchants = (
                rng.choice(approved_probe, size=n_escalate)
                if n_escalate
                else np.empty(0, dtype=object)
            )
            view.retarget(rows, np.concatenate([probe_merchants, escalate_merchants]))

            mcc = rows["mcc"].to_numpy()
            amount = np.empty(n, dtype=float)
            amount[:n_probe] = view.draw_amounts(mcc[:n_probe], *_PROBE_BAND, rng)
            if n_escalate:
                amount[n_probe:] = view.draw_amounts(mcc[n_probe:], *_ESCALATION_BAND, rng)
            rows["amount"] = amount

            # The search runs hot for a day or so; the payoff follows it.
            search_seconds = int(rng.integers(20, 40) * 3_600)
            ts = np.empty(n, dtype=np.int64)
            ts[:n_probe] = starts[c] + rng.integers(0, search_seconds, n_probe)
            if n_escalate:
                ts[n_probe:] = starts[c] + search_seconds + rng.integers(0, 14 * 3_600, n_escalate)
            # Each attempt is independent, so each gets its own hour of day.
            view.set_timestamps(rows, ts, rng=rng)

            # Search fails most of the time; extraction, going back through the
            # merchants that just approved, mostly does not. The probe half of
            # this array was already drawn above, because the escalation targets
            # were selected from it.
            declined = np.r_[probe_declined, rng.random(n_escalate) < _ESCALATION_DECLINE_P]
            # An invalid-CVV decline is impossible where no CVV was presented.
            reasons = np.asarray(_DECLINE_REASONS, dtype=object)[
                rng.choice(len(_DECLINE_REASONS), size=n, p=_DECLINE_REASON_P)
            ]
            keyed = rows["entry_mode"].to_numpy() == "ecom_keyed"
            reasons[~keyed & (reasons == "declined_invalid_cvv")] = "declined_do_not_honor"
            view.decline(rows, declined, reasons[declined])

            blocks.append(
                view.finalise(
                    rows,
                    card_id=self.card_id,
                    campaigns=np.full(n, campaign_id(self.card_id, c), dtype=object),
                    rng=rng,
                )
            )
        return pd.concat(blocks, ignore_index=True)


inject = card_entry_point(AdaptiveBinAttack)


def main() -> None:
    """Print a sample campaign. Run: ``python -m mantis.foundry.injectors.f4_27_adaptive_bin``."""
    demo_main(AdaptiveBinAttack)


if __name__ == "__main__":
    main()
