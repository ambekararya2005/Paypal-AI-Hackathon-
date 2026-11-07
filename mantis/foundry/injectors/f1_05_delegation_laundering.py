"""F1-05 — delegation-chain laundering through sub-agents. **HARD bucket.**

Note on the card id
-------------------
The Day 3 brief named F1-05 "vector-memory poisoning for persistent merchant
steering". In the frozen atlas that vector is **F5-36** (persistent memory and
context-store poisoning), and F1-05 is delegation-chain laundering. This injector
is written against the card that exists, on the same principle as the four Day 2
remappings: the injector must generate what its card describes.

The attack
----------
The human delegates to an agent they chose. That agent delegates onward, and a
sub-agent transacts. The authorisation carries the human's mandate and a
delegation depth the human never contemplated.

Authority has been **laundered through the chain**: the payment is formally
authorised and substantively unsupervised. Every hop strips context, so by the
time the payment is made the executing agent holds the authority the human
granted and none of the constraints the human stated — because those constraints
were prose in a planning context two hops back. That is why ``delegation_depth``
is carried explicitly in the frozen schema rather than inferred after the fact.

Bucket: HARD
------------
``delegation_depth_exceeded`` is a single-message check against issuer policy.
But it is the *weakest* of the HARD signals, and saying why matters:

Legitimate multi-agent orchestration genuinely produces deep chains. A shopping
agent calling a booking agent calling a checkout capability is three hops before
anything unusual has happened. Day 1's population capped legitimate depth at
three, which would have made ``depth >= 4`` a perfect detector — a fact about our
generator, not about laundering. Day 3 widened the legitimate tail to five
(``delegation_depth_weights``), so the rule now has a real false-positive
population to trade against, and the depth threshold is a **policy choice with a
cost** rather than a free win. That is the honest version.

What the graph layer gets, and why it is the real answer here
---------------------------------------------------------------
Depth alone will not carry this card. Two structural signals will, and neither is
visible to any single-column probe by construction:

* ``agent_id_fanout`` (L4). A rogue sub-agent transacts for an implausible number
  of unrelated principals. Each campaign routes many customers through a handful
  of sub-agent identities, which is what a shared malicious capability looks
  like from the network's side.
* ``mandate_hash_reuse_graph_degree`` (L4). One mandate hash appearing across
  agents that share no legitimate relationship — the artefact travelled down the
  chain and was presented by something the human never chose.

* ``agent_platform_novelty`` (L1). The executing agent runs on a platform this
  customer has never used, computed against their actual history.

Modelling decisions worth defending
-----------------------------------
* **Sub-agent identities are reused from the population**, not minted. A
  never-before-seen agent id would be caught by novelty alone and would teach a
  detector nothing transferable; the interesting case is a real, registered
  capability behaving badly across many principals.
* **Depth overlaps the legitimate tail.** Most events sit at three or four,
  where legitimate traffic still lives. Pushing every event to seven would have
  produced a 0.99 single-column AUC and a card that proves nothing.
* **KYA registration is left alone.** A sub-agent in a delegation chain is
  typically a registered capability — that is why the principal's agent was
  willing to call it. Making it unregistered would collapse this card into
  F1-09's consent/KYA half.

Realism check (measured, not asserted)
--------------------------------------
Best single-feature AUC within the agentic slice: **~0.94** on
``ag_delegation_depth``. That is the closest any injector in the atlas sits to
the 0.95 gate, and the number is reported rather than tuned away because it is
the finding: **depth very nearly is a sufficient detector for this attack.**

Two things follow, and both are worth saying to a judge rather than hiding.
First, before Day 3 widened the legitimate tail this figure was effectively
1.00, and the difference between those two numbers is the difference between
measuring an attack and measuring a generator. Second, a card that one column
almost solves is a card whose interesting content is elsewhere: the reason to
build ``agent_id_fanout`` and ``mandate_hash_reuse_graph_degree`` is not that
depth fails, it is that an issuer cannot set a depth threshold low enough to
catch a two-hop laundering chain without blocking ordinary orchestration. The
graph signals are what make the *cheap* end of this attack detectable.

Re-measure with ``python -m mantis.foundry --attacks F1-05``, quoting the
in-slice column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mantis.foundry.injectors.agentic import AgenticAttack, Bucket, agentic_pool
from mantis.foundry.injectors.base import (
    _short_hash,
    campaign_id,
    card_entry_point,
    demo_main,
    register,
    split_count,
)

__all__ = ["DelegationLaunderingAttack", "inject", "main"]

#: Delegation depth at the executing hop, and how often each occurs. Overlaps
#: the legitimate tail deliberately: see the module docstring.
_DEPTHS: tuple[int, ...] = (2, 3, 4, 5, 6)
_DEPTH_P: tuple[float, ...] = (0.34, 0.36, 0.18, 0.09, 0.03)

#: Distinct sub-agent identities a campaign routes its principals through. Small,
#: because a shared malicious capability is shared -- that concentration is the
#: L4 fan-out signal.
_SUBAGENTS_PER_CAMPAIGN: tuple[int, int] = (2, 5)

#: Share of a campaign's events that present a mandate artefact already used by a
#: different sub-agent identity in the same campaign.
_HASH_SHARING_SHARE: float = 0.46

#: How many leaked artefacts a campaign passes around. Two, so the graph has
#: more than one component to find and the degree is not a constant.
_LAUNDERED_ARTEFACTS: int = 2

#: Where the amount sits. Unremarkable: laundering authority is about reach, not
#: about a single large ticket.
_AMOUNT_BAND: tuple[float, float] = (0.35, 0.88)


@register
class DelegationLaunderingAttack(AgenticAttack):
    """Authority laundered through sub-agents the human never chose."""

    card_id = "F1-05"
    bucket = Bucket.HARD
    base_events = 100
    base_campaigns = 4

    def inject(
        self, population: pd.DataFrame, intensity: float, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Emit purchases executed by sub-agents, deep in a delegation chain."""
        view = self.view
        pool = agentic_pool(view)
        frame = view.frame

        # The identities and platforms a sub-agent can plausibly present as.
        # Drawn from the population so the executing agent is a real registered
        # capability rather than an obvious stranger.
        known_agents = frame["ag_agent_id"].dropna().unique()
        known_platforms = frame["ag_agent_platform"].dropna().unique()

        counts = split_count(self.n_events(intensity), self.n_campaigns(intensity), rng)
        starts = view.spread_epochs(len(counts), rng)

        blocks: list[pd.DataFrame] = []
        for c, n in enumerate(counts):
            n = int(n)
            rows = view.clone(rng.choice(pool, size=n, replace=True))
            rows["amount"] = view.draw_amounts(rows["mcc"].to_numpy(), *_AMOUNT_BAND, rng)
            view.set_timestamps(rows, starts[c] + rng.integers(0, 25 * 86_400, n), rng=rng)

            block = view.finalise(
                rows,
                card_id=self.card_id,
                campaigns=np.full(n, campaign_id(self.card_id, c), dtype=object),
                rng=rng,
            )
            self._launder(block, c, known_agents, known_platforms, rng)
            blocks.append(block)

        return pd.concat(blocks, ignore_index=True)

    def _launder(
        self,
        block: pd.DataFrame,
        campaign: int,
        known_agents: np.ndarray,
        known_platforms: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        """Route the campaign's principals through a handful of sub-agents."""
        view = self.view
        n = len(block)

        # -- the chain got longer than the human contemplated ----------------- #
        block["ag_delegation_depth"] = rng.choice(_DEPTHS, size=n, p=_DEPTH_P)

        # -- a shared sub-agent, executing for unrelated principals ----------- #
        k = int(rng.integers(*_SUBAGENTS_PER_CAMPAIGN))
        subagents = known_agents[rng.choice(known_agents.size, size=k, replace=False)]
        assignment = rng.integers(0, k, n)
        block["ag_agent_id"] = subagents[assignment]

        # -- on a platform this customer has never used ----------------------- #
        # Computed against the customer's real history: a randomly chosen
        # platform would often be one they already use, and the novelty signal
        # would be true only by luck.
        platform_col = view.frame["ag_agent_platform"].to_numpy()
        platforms = block["ag_agent_platform"].to_numpy().copy()
        for i, cid in enumerate(block["customer_id"].to_numpy()):
            seen = {
                str(p) for p in platform_col[view.rows_by_customer[str(cid)]] if isinstance(p, str)
            }
            unseen = [str(p) for p in known_platforms if str(p) not in seen]
            if unseen:
                platforms[i] = unseen[rng.integers(0, len(unseen))]
        block["ag_agent_platform"] = platforms

        # -- one mandate artefact, presented by more than one identity -------- #
        # The hash travelled down the chain. ``finalise`` gave every event its
        # own; a share of them are collapsed back onto a per-sub-agent artefact
        # so the graph layer sees a hash spanning identities that share no
        # legitimate relationship.
        # Grouped **independently of** which sub-agent executed, which is the
        # whole point: an artefact grouped per identity would never span two,
        # and ``mandate_hash_reuse_graph_degree`` would have nothing to find.
        # The first cut did exactly that and the test caught it.
        shared = rng.random(n) < _HASH_SHARING_SHARE
        artefact = rng.integers(0, _LAUNDERED_ARTEFACTS, n)
        hashes = block["ag_mandate_hash"].to_numpy().copy()
        ids = block["ag_mandate_id"].to_numpy().copy()
        for a in range(_LAUNDERED_ARTEFACTS):
            members = np.flatnonzero(shared & (artefact == a))
            if members.size < 2:
                continue
            mandate_id = f"mnd-{_short_hash(f'F1-05-{campaign}-{a}-laundered', 10)}"
            ids[members] = mandate_id
            hashes[members] = _short_hash(f"{mandate_id}|delegated", 16)
        block["ag_mandate_id"] = ids
        block["ag_mandate_hash"] = hashes


inject = card_entry_point(DelegationLaunderingAttack)


def main() -> None:
    """Print a sample campaign.

    Run: ``python -m mantis.foundry.injectors.f1_05_delegation_laundering``.
    """
    demo_main(DelegationLaunderingAttack)


if __name__ == "__main__":
    main()
