"""F1-09 — human-present spoofing for liability shift. **HARD bucket.**

The attack
----------
The authorisation asserts ``human_present``. Nobody is watching. The claim buys
three things at once: step-up friction is skipped, approval rates improve, and —
the part that makes this economically important out of all proportion to its
sophistication — **liability moves**. ``human_present`` is a liability flag as
much as a risk flag, so spoofing it does not merely evade a control, it
relocates the loss onto the issuer or the cardholder.

Presence used to be evidenced by telemetry that was expensive to fake
convincingly: cursor paths, dwell times, hesitation before commit. A generative
model reproduces those distributions for nothing, which downgrades "a human was
watching" from evidence to an unverified assertion.

Bucket: HARD — but only on one half
-------------------------------------
This is the card the brief calls the consent/KYA case, and the deterministic half
is real: a majority of events carry ``consent_sig_valid=False`` or
``kya_registered=False``. An operator willing to lie about presence is, in
practice, operating outside the attestation regime — and those are the two flags
an issuer can check with certainty on a single message. **L0 should catch that
half at near-zero false positive rate**, and the background supports the claim:
the legitimate tails are 0.3% and 2.8% respectively.

The other half is deliberately clean on both flags and only forges telemetry.
Splitting the campaign this way is what keeps the card honest: if every spoof
also failed a signature check, the behavioural work would be unnecessary and we
would be reporting L0's recall as if it were the model's.

Two spoofing sub-shapes, because both exist
--------------------------------------------
* **Lazy (about 40%).** The flag is flipped and the telemetry is not touched, so
  a machine-like cursor entropy sits next to a claim of human oversight. This is
  the mismatch a two-column rule finds.
* **Forged (about 60%).** Telemetry is resampled from the *human* half of the
  background's own distribution, but from a **narrow band** inside it. Real
  people hesitate irregularly; generators do not. The tell is that the forged
  population is distributionally too clean — a variance argument, not a
  threshold — which is L1 and L2 work, not a rule.

Why the lazy half is not a free win, and what we changed to keep it that way
-----------------------------------------------------------------------------
The obvious rule is *"human_present asserted, cursor entropy machine-like"*. On
the Day 2 population that rule was **perfect**, because every supervised session
had hands on the device. That was a property of our generator, not of spoofing,
and reporting recall from it would have been dishonest.

Day 3 added a passive tail to the background (``human_present_passive_share``,
11%): sessions where the person genuinely is watching and never touches the
device — reading on a second screen while the agent drives. Those are honestly
``human_present=True`` with machine-like telemetry. The mismatch rule now has a
real false-positive population to trade against, which is the only condition
under which a recall number from it means anything.

Modelling decisions worth defending
-----------------------------------
* **3-D Secure is left alone.** The card names "presence asserted while no
  step-up was performed" as an L0 signal, and in a deployed system it is one.
  In *this* population it is not usable: 82% of legitimate agentic
  authorisations already clear frictionless or under a mandate exemption, so
  forcing the attack to look that way would add nothing and pinning it there
  would be a generator artefact. The card keeps the signal because an issuer
  with a step-up-capable rail has it; the foundry does not pretend to.
* **Forged telemetry is resampled from real values.** Same discipline as
  ``draw_amounts``: a forged number drawn from an attacker-chosen distribution
  would be detectable on the column alone.

Realism check (measured, not asserted)
--------------------------------------
See the probe table from ``python -m mantis.foundry --attacks F1-09``, and quote
the rail-conditioned column. The consent-signature indicator is expected to be
the strongest single feature, which is correct and is the point of the HARD
bucket: it is what L0 is *for*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mantis.foundry.injectors.agentic import (
    AgenticAttack,
    Bucket,
    agentic_pool,
    resample_band,
    spread_across_rails,
)
from mantis.foundry.injectors.base import (
    campaign_id,
    card_entry_point,
    demo_main,
    register,
    split_count,
)

__all__ = ["PresenceSpoofAttack", "inject", "main"]

_ECOM_SHARE: float = 0.30

#: Share of events where the flag is flipped and the telemetry is not touched.
_LAZY_SHARE: float = 0.40

#: Share carrying an unverified consent signature (background tail: 0.3%).
_CONSENT_INVALID_SHARE: float = 0.52

#: Share from an agent absent from the KYA registry (background tail: 2.8%).
_KYA_UNREGISTERED_SHARE: float = 0.34

#: Quantile band, inside the *human* half of the background's telemetry, that
#: forged values are resampled from. Narrow: that narrowness is the tell, and it
#: is a distributional argument rather than a threshold one.
_FORGED_CURSOR_BAND: tuple[float, float] = (0.62, 0.78)
_FORGED_DWELL_BAND: tuple[float, float] = (0.60, 0.76)

#: Jitter on forged telemetry. Small on purpose -- a generator that reproduced
#: the full human variance would not be spoofing detectably, and the card's
#: claim is precisely that it does not.
_FORGED_JITTER: float = 0.035


@register
class PresenceSpoofAttack(AgenticAttack):
    """Asserted human oversight, with telemetry either untouched or too clean."""

    card_id = "F1-09"
    bucket = Bucket.HARD
    base_events = 130
    base_campaigns = 5

    def inject(
        self, population: pd.DataFrame, intensity: float, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Emit authorisations claiming human oversight that was not there."""
        view = self.view
        pool = agentic_pool(view)

        counts = split_count(self.n_events(intensity), self.n_campaigns(intensity), rng)
        starts = view.spread_epochs(len(counts), rng)

        blocks: list[pd.DataFrame] = []
        for c, n in enumerate(counts):
            n = int(n)
            rows = view.clone(rng.choice(pool, size=n, replace=True))

            # Presence is claimed to buy reduced friction on purchases worth
            # having, so the amounts sit in the upper half rather than at the
            # extreme -- an operator harvesting a liability shift wants volume,
            # not a single spectacular authorisation.
            rows["amount"] = view.draw_amounts(rows["mcc"].to_numpy(), 0.55, 0.93, rng)

            view.set_timestamps(rows, starts[c] + rng.integers(0, 60 * 3_600, n), rng=rng)

            block = view.finalise(
                rows,
                card_id=self.card_id,
                campaigns=np.full(n, campaign_id(self.card_id, c), dtype=object),
                rng=rng,
            )
            self._spoof_presence(block, rng)
            blocks.append(block)

        return pd.concat(blocks, ignore_index=True)

    def _spoof_presence(self, block: pd.DataFrame, rng: np.random.Generator) -> None:
        """Flip the claim; forge or neglect the telemetry that should back it."""
        view = self.view
        n = len(block)

        block["ag_human_present"] = True

        # The deterministic half. An operator lying about presence is usually
        # also outside the attestation regime, and these are the two things an
        # issuer can verify with certainty on one message.
        block["ag_consent_sig_valid"] = rng.random(n) >= _CONSENT_INVALID_SHARE
        unregistered = rng.random(n) < _KYA_UNREGISTERED_SHARE
        block["ag_kya_registered"] = ~unregistered
        # A KYA token without a registration is not a thing. Dropping it keeps
        # the null pattern coherent with the flag rather than leaving a token
        # that would contradict it.
        block.loc[unregistered, "ag_kya_token"] = None

        # The behavioural half.
        forged = rng.random(n) >= _LAZY_SHARE
        count = int(forged.sum())
        if count:
            cursor = resample_band(
                view, "ag_cursor_entropy", count, _FORGED_CURSOR_BAND, rng, jitter=_FORGED_JITTER
            )
            dwell = resample_band(
                view, "ag_dwell_time_ms", count, _FORGED_DWELL_BAND, rng, jitter=_FORGED_JITTER
            )
            block.loc[forged, "ag_cursor_entropy"] = np.round(cursor, 3)
            block.loc[forged, "ag_dwell_time_ms"] = np.maximum(40, dwell.round()).astype(int)
        # The lazy remainder keeps whatever the cloned row had, which is
        # machine-like far more often than not -- and is exactly the mismatch the
        # population's passive-human tail stops from being a perfect rule.

        spread_across_rails(block, _ECOM_SHARE, rng)


inject = card_entry_point(PresenceSpoofAttack)


def main() -> None:
    """Print a sample campaign. Run: ``python -m mantis.foundry.injectors.f1_09_presence_spoof``."""
    demo_main(PresenceSpoofAttack)


if __name__ == "__main__":
    main()
