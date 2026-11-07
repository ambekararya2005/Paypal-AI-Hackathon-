"""F1-10 — mandate replay and TTL abuse. **HARD bucket.**

The attack
----------
A previously valid mandate is presented again: after its TTL has expired, more
times than its scope allows, or by an agent it was never issued to. Every
signature verifies, because the artefact is authentic. The failure is one of
**freshness and single-use**, not of cryptography.

This is the classic bearer-token weakness reappearing in a new setting, and it is
the cheapest F1 vector to run: no injection, no social engineering, no model
access. Agent architectures pass mandates between planners, tools and sub-agents
as ordinary context, so a signed authorisation ends up in caches, transcripts and
logs that were never designed to hold a bearer credential.

Bucket: HARD
------------
Two single-message checks with certainty behind them:

* ``mandate_expired`` — the authorisation arrives after issuance plus TTL. The
  background contains **zero** expired mandates (asserted in
  ``tests/test_population.py``), so the rule has no legitimate population to
  trade against.
* ``mandate_hash_reuse_count`` — strictly, a check against a replay cache rather
  than against the message alone, but every issuer already runs that kind of
  cache for other reasons and the card names it L0 for that reason.

**L0 should catch this at near-zero false positive rate.** If it does not, the
replay cache is not wired up.

What the deeper layers still get
---------------------------------
* ``mandate_replay_interarrival`` (L1). Replays arrive on a **machine-regular
  cadence** — a loop with a fixed sleep — where genuine repeat purchases by one
  customer are irregular. That regularity is modelled explicitly: gaps within a
  replay burst are one interval plus a few percent of jitter, not a fresh draw
  per event.
* ``mandate_hash_shared_agents`` (L4). A third of campaigns spread one mandate
  hash across two or three distinct ``agent_id`` values. A leaked artefact is
  portable, and the graph layer seeing one hash presented by unrelated agents is
  the structural evidence that it leaked.

Modelling decisions worth defending
-----------------------------------
* **The first presentation of each mandate is legitimate and unexpired.** A burst
  where even the first use was already expired would be a different, dumber
  attack. Replay means re-*use*: the artefact was valid once.
* **Not every replay is expired.** Roughly a third arrive inside the TTL and are
  caught only by the reuse counter, not by the clock. Making all of them expired
  would let one rule take the whole card and would hide the fact that the two
  signals are independent.
* **Amounts are resampled from the mandate's own original band**, because a
  replayed mandate authorises a purchase of the shape it was issued for. An
  operator who replayed a grocery mandate for a jeweller would be caught on
  scope, which is F1-02's job, not this one.

Realism check (measured, not asserted)
--------------------------------------
See the probe table from ``python -m mantis.foundry --attacks F1-10``, quoting
the rail-conditioned column. ``mandate_age_seconds`` is expected to be the
strongest single feature — correctly, because a mandate used long after issuance
is what this attack *is*. The reuse count is invisible to a single-column probe
by construction: it is a cross-row join, which is the point.
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

__all__ = ["MandateReplayAttack", "inject", "main"]

#: How many times one captured mandate is presented.
_REPLAYS_RANGE: tuple[int, int] = (3, 8)

#: Share of replays that arrive after the mandate's TTL has run out. The rest are
#: inside the window and are caught only by the reuse counter.
_EXPIRED_SHARE: float = 0.66

#: Multiples of the mandate TTL that an expired replay arrives at.
_EXPIRY_OVERSHOOT: tuple[float, float] = (1.15, 9.0)

#: Machine-regular cadence: the gap between replays, in seconds, and the jitter
#: on it. A few percent, not a fresh draw -- the regularity is the L1 signal.
_REPLAY_INTERVAL_S: tuple[int, int] = (20 * 60, 6 * 3_600)
_REPLAY_JITTER: float = 0.04

#: Share of campaigns where the captured mandate is presented by more than one
#: agent identity -- the structural evidence that the artefact leaked.
_SHARED_AGENT_SHARE: float = 0.34
_SHARED_AGENT_COUNT: tuple[int, int] = (2, 4)


@register
class MandateReplayAttack(AgenticAttack):
    """One authentic mandate, presented again and again."""

    card_id = "F1-10"
    bucket = Bucket.HARD
    base_events = 110
    base_campaigns = 5

    def inject(
        self, population: pd.DataFrame, intensity: float, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Emit bursts that re-present one captured mandate artefact."""
        view = self.view
        pool = agentic_pool(view)

        counts = split_count(self.n_events(intensity), self.n_campaigns(intensity), rng)
        starts = view.spread_epochs(len(counts), rng)

        blocks: list[pd.DataFrame] = []
        for c, n in enumerate(counts):
            n = int(n)

            # Lay out the bursts: which captured mandate, and which presentation
            # within it. Presentation 0 is the legitimate original.
            source_rows: list[int] = []
            burst: list[int] = []
            sequence: list[int] = []
            b = 0
            while len(source_rows) < n:
                origin = int(rng.choice(pool))
                replays = int(rng.integers(*_REPLAYS_RANGE))
                for k in range(min(replays, n - len(source_rows))):
                    source_rows.append(origin)
                    burst.append(b)
                    sequence.append(k)
                b += 1
            positions = np.asarray(source_rows[:n], dtype=np.int64)
            burst_id = np.asarray(burst[:n], dtype=np.int64)
            seq = np.asarray(sequence[:n], dtype=np.int64)

            rows = view.clone(positions)
            rows["amount"] = view.draw_amounts(rows["mcc"].to_numpy(), 0.35, 0.92, rng)

            # A machine-regular cadence: one interval per burst, jittered by a
            # few percent, not redrawn per event.
            interval = rng.integers(*_REPLAY_INTERVAL_S, burst_id.max() + 1)[burst_id]
            jitter = 1.0 + rng.normal(0.0, _REPLAY_JITTER, n)
            offsets = (seq * interval * jitter).astype(np.int64)
            # The whole burst shifts together so the spacing survives the
            # hour-of-day redraw.
            view.set_timestamps(rows, starts[c] + offsets, rng=rng, groups=burst_id)

            block = view.finalise(
                rows,
                card_id=self.card_id,
                campaigns=np.full(n, campaign_id(self.card_id, c), dtype=object),
                rng=rng,
            )
            self._replay(block, burst_id, seq, rng)
            blocks.append(block)

        return pd.concat(blocks, ignore_index=True)

    def _replay(
        self,
        block: pd.DataFrame,
        burst_id: np.ndarray,
        seq: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        """Collapse each burst onto one mandate artefact, and age it past its TTL.

        ``finalise`` gives every event its own mandate id and hash, which is what
        every other injector wants and the exact opposite of what this one is.
        The identity is re-collapsed here, deliberately: one hash, one issuance
        time, many authorisations.
        """
        n = len(block)
        ts = block["ts"].astype("int64").to_numpy() // 1_000_000_000
        ttl = block["ag_mandate_ttl_seconds"].to_numpy(dtype=float)
        agent = block["ag_agent_id"].to_numpy().copy()

        mandate_ids = block["ag_mandate_id"].to_numpy().copy()
        mandate_hashes = block["ag_mandate_hash"].to_numpy().copy()
        issued = np.empty(n, dtype=np.int64)

        for b in np.unique(burst_id):
            members = np.flatnonzero(burst_id == b)
            first = members[np.argmin(seq[members])]

            # One artefact for the whole burst, named after its first, genuine,
            # presentation. That is what a leaked mandate is.
            origin_event = str(block["event_id"].iloc[first])
            mandate_id = f"mnd-{_short_hash(origin_event + '-captured', 10)}"
            mandate_ids[members] = mandate_id
            mandate_hashes[members] = _short_hash(f"{mandate_id}|captured", 16)

            # Issued just before its first use, which was legitimate.
            burst_ttl = float(np.nan_to_num(ttl[members[0]], nan=900.0))
            issued_at = int(ts[first] - rng.integers(5, max(6, int(burst_ttl * 0.4))))
            issued[members] = issued_at

            # A share of the *replays* (never the original) arrive past the TTL.
            for m in members:
                if seq[m] == 0:
                    continue
                if rng.random() < _EXPIRED_SHARE:
                    overshoot = rng.uniform(*_EXPIRY_OVERSHOOT)
                    ts[m] = int(issued_at + burst_ttl * overshoot)

            # A leaked artefact is portable. A third of bursts are presented by
            # more than one agent identity -- the L4 structural signal.
            if rng.random() < _SHARED_AGENT_SHARE:
                identities = np.unique(agent)
                pick = identities[
                    rng.choice(
                        identities.size,
                        size=min(int(rng.integers(*_SHARED_AGENT_COUNT)), identities.size),
                        replace=False,
                    )
                ]
                agent[members] = pick[rng.integers(0, pick.size, members.size)]

        block["ag_mandate_id"] = mandate_ids
        block["ag_mandate_hash"] = mandate_hashes
        block["ag_agent_id"] = agent
        block["ts"] = pd.to_datetime(
            np.clip(ts, self.view.start_epoch, self.view.end_epoch), unit="s", utc=True
        ).tz_convert(block["ts"].dt.tz)
        block["ag_mandate_issued_ts"] = pd.to_datetime(issued, unit="s", utc=True).tz_convert(
            block["ag_mandate_issued_ts"].dt.tz
        )


inject = card_entry_point(MandateReplayAttack)


def main() -> None:
    """Print a sample campaign. Run: ``python -m mantis.foundry.injectors.f1_10_mandate_replay``."""
    demo_main(MandateReplayAttack)


if __name__ == "__main__":
    main()
