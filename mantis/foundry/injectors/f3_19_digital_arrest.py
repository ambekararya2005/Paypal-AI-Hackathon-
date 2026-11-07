"""F3-19 — digital-arrest scam: the victim authorises every payment themselves.

The attack
----------
A crew impersonating law enforcement holds a victim in a prolonged coercive
session and walks them through transfers to accounts described as "verification"
custody. The victim authenticates normally, on their usual device, with their
genuine credential. Nothing in the authentication layer will ever flag this.

Modelling decisions worth defending
-----------------------------------
* **Anomalous for that victim, not in absolute terms.** This is the whole card,
  and it is the constraint that shapes every line below. Transfer sizes are
  multiples of the *victim's own* 99th percentile, not of a population
  threshold, and each is then capped by a value resampled from the receiving
  category's own upper band. Victims come from the lower half of the spend
  distribution, so a transfer worth several times a victim's personal ceiling is
  still an utterly ordinary number on the network. A detector that flags "large
  payment" will miss this; one that models the customer will not.
* **Escalation inside one session, opening small.** Two to five transfers over
  three to nine hours. The first is a modest "verification" payment inside the
  victim's ordinary range — which is what the pattern actually looks like, and
  what stops the label from collapsing into "big transfer". A single transfer
  would be indistinguishable from a house deposit.
* **Beneficiaries are shared across victims in a campaign** and are drawn from
  the long tail of the person-to-person estate — accounts with almost no prior
  volume that suddenly collect from strangers. That fan-in is the receiving-side
  signal the card argues is the only part the crew cannot hide.
* **Rails follow the category's own mix.** Source rows are cloned from real
  person-to-person authorisations, so the channel and entry-mode distribution is
  the population's rather than the injector's. A fifth of the volume goes to a
  long-tail storefront instead, which is the "pay this fine online" variant.

The agent-mediated variant, and what is deferred
------------------------------------------------
The card also claims an L3 signal: coercion and authority-impersonation language
in the agent's ingested content. That belongs to the text layer and to the LLM
content foundry, neither of which exists yet — so this injector emits the
tabular footprint only, and the L3 side lands with ``foundry/llm``. Saying so
here rather than quietly emitting a keyword is the difference between a claim
and a demo.

Realism check (measured, not asserted)
--------------------------------------
Best single-feature depth-1 stump AUC: **0.872** (``amount``) — the highest of the eight,
and honestly so: a coerced transfer really does sit at the top of its victim's
range, and no amount of modelling makes that untrue. What the number does *not*
support is a threshold: the population's 87th percentile is ordinary traffic, so
an issuer cutting there would drown. Recall has to come from
``amount_vs_customer_p99`` and from beneficiary fan-in, not from ``amount``.

Measured by ``mantis.foundry.injectors.probe`` against a 200k-event background at
seed 1337, over every column an issuer can read off one authorisation message.
Re-measure with ``python -m mantis.foundry --attacks F3-19``.
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

__all__ = ["DigitalArrestAttack", "inject", "main"]

#: Person-to-person category: the custody accounts victims are told to pay.
_P2P_MCC: str = "6012"

#: The "pay the fine on this portal" variant. A third of the volume, not a
#: token slice: with 80% of rows on MCC 6012 the category indicator alone reached
#: 0.90 AUC, which is category concentration masquerading as detection.
_PORTAL_MCC: str = "5999"
_PORTAL_SHARE: float = 0.35

#: A coerced session opens with a small "verification" payment — well inside the
#: victim's ordinary range — before the demands escalate. Modelling that opener
#: is both true to the pattern and the thing that keeps the amount column from
#: carrying the attack on its own.
_OPENING_MULTIPLE: tuple[float, float] = (0.25, 1.0)

#: How far above the victim's own 99th percentile the demands then run, and how
#: much each further transfer in the session escalates.
_FIRST_MULTIPLE: tuple[float, float] = (1.15, 3.2)
_ESCALATION: tuple[float, float] = (1.15, 1.9)

#: Absolute ceiling, resampled per row from this quantile band of the receiving
#: category. Keeps every transfer unremarkable network-wide however small the
#: victim's own history is, without pinning the campaign to one repeated number.
_CATEGORY_CAP_BAND: tuple[float, float] = (0.72, 0.985)

#: Share of sessions where the victim's own agent is drawn into the pretext.
_AGENT_MEDIATED_SHARE: float = 0.12


@register
class DigitalArrestAttack(BaseAttack):
    """Coerced, victim-authorised push payments escalating inside one session."""

    card_id = "F3-19"
    base_events = 110
    base_campaigns = 6

    def inject(
        self, population: pd.DataFrame, intensity: float, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Emit escalating transfer bursts that are outliers only against self."""
        view = self.view
        # Victims: real customers with enough history to have a personal ceiling,
        # drawn from the lower two thirds of the spend distribution so that a
        # multiple of their ceiling is still an ordinary number on the network.
        eligible = view.customers[view.customers["n_events"] >= 8]
        eligible = eligible[eligible["amount_p99"] <= eligible["amount_p99"].quantile(0.45)]
        victims_pool = eligible.index.to_numpy()

        p2p_beneficiaries = view.merchants_in_mcc(_P2P_MCC, popularity=(0.0, 0.35))
        portal_beneficiaries = view.merchants_in_mcc(_PORTAL_MCC, popularity=(0.0, 0.30))

        counts = split_count(self.n_events(intensity), self.n_campaigns(intensity), rng)
        starts = view.spread_epochs(len(counts), rng)

        blocks: list[pd.DataFrame] = []
        for c, n in enumerate(counts):
            n = int(n)
            n_victims = int(np.clip(round(n / 3.2), 3, 14))
            victims = rng.choice(
                victims_pool, size=min(n_victims, victims_pool.size), replace=False
            )
            # Transfers per victim: a coerced session, not a single payment.
            per_victim = rng.integers(2, 6, victims.size)
            owner = np.repeat(np.arange(victims.size), per_victim)[:n]
            if owner.size < n:  # top up if the draw came in short
                owner = np.concatenate([owner, rng.integers(0, victims.size, n - owner.size)])
            owner = np.sort(owner)
            seq = self._sequence(owner)

            # A minority of sessions run through the victim's own agent, which
            # is the variant the L3 provenance signal will eventually read. The
            # tabular footprint is identical; only the rail differs.
            through_agent = rng.random(n) < _AGENT_MEDIATED_SHARE
            source = np.empty(n, dtype=np.int64)
            for mask, rails in (
                (through_agent, ("agentic",)),
                (~through_agent, ("upi_p2p", "upi_p2m")),
            ):
                if mask.any():
                    source[mask] = view.source_rows(victims[owner][mask], rng, channels=rails)
            rows = view.clone(source)

            portal = rng.random(n) < _PORTAL_SHARE
            pool_p2p = rng.choice(
                p2p_beneficiaries, size=max(2, min(4, p2p_beneficiaries.size)), replace=False
            )
            pool_portal = rng.choice(
                portal_beneficiaries, size=max(1, min(2, portal_beneficiaries.size)), replace=False
            )
            merchants = np.where(
                portal,
                rng.choice(pool_portal, size=n),
                rng.choice(pool_p2p, size=n),
            ).astype(object)
            view.retarget(rows, merchants)

            ceiling = view.customers.loc[victims, "amount_p99"].to_numpy()[owner]
            multiple = np.where(
                seq == 0,
                rng.uniform(*_OPENING_MULTIPLE, n),
                rng.uniform(*_FIRST_MULTIPLE, n) * rng.uniform(*_ESCALATION, n) ** seq,
            )
            # The absolute ceiling is drawn per row from the receiving category's
            # own upper band rather than being one fixed quantile. A single cap
            # would pin most of the campaign to one repeated amount -- which is
            # both a cartoon and, ironically, a far stronger signal than the
            # attack itself.
            cap = view.draw_amounts(rows["mcc"].to_numpy(), *_CATEGORY_CAP_BAND, rng)
            amount = np.minimum(ceiling * multiple, cap)
            # Coerced transfers are dictated over a phone call: round numbers.
            step = np.where(amount < 10_000, 500.0, 1_000.0)
            amount = np.where(
                rng.random(n) < 0.62, np.maximum(step, np.round(amount / step) * step), amount
            )
            rows["amount"] = np.round(np.clip(amount, 100.0, None), 2)

            # One sustained session per victim, three to nine hours long.
            session = starts[c] + rng.integers(0, 6 * 86_400, victims.size)[owner]
            step_seconds = rng.integers(25 * 60, 3 * 3_600, n)
            view.set_timestamps(rows, session + seq * step_seconds, rng=rng, groups=owner)

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
        """Position of each transfer inside its victim's session."""
        starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
        group_start = np.repeat(starts, np.diff(np.r_[starts, owner.size]))
        return np.arange(owner.size) - group_start


inject = card_entry_point(DigitalArrestAttack)


def main() -> None:
    """Print a sample session. Run: ``python -m mantis.foundry.injectors.f3_19_digital_arrest``."""
    demo_main(DigitalArrestAttack)


if __name__ == "__main__":
    main()
