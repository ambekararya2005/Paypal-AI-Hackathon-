"""F2-16 — agent-farm bust-out: cultivate impeccably, then draw down together.

The attack
----------
A portfolio of accounts, each paired with its own agent identity, behaves
perfectly for a cultivation period, accrues limit, and then draws down inside a
single window. The bust-out itself is trivial to see and far too late to matter,
so this injector spends most of its rows on the phase that is actually
detectable: the cultivation.

Modelling decisions worth defending
-----------------------------------
* **Seventy per cent of the attack's rows are the boring half.** If an injector
  emitted only the drawdown, the label would be "large purchase" and the whole
  exercise would collapse into an amount threshold. Labelling the cultivation as
  fraud is the honest choice — it *is* part of the campaign — and it is what
  forces the detector to find the portfolio rather than the payday.
* **Cultivation is too regular, not too odd.** Inter-arrival times come from a
  tight per-account period with a few per cent of jitter, and amounts sit in a
  narrow mid band. The card's ``interarrival_regularity`` signal exists because
  abnormally low variance is the tell; the population's own customers are
  bursty and irregular by comparison.
* **The drawdown lands in cash-convertible categories** — electronics, travel,
  hotels, department stores — inside a 30-hour window shared across the
  portfolio, and at amounts drawn from the high band of those categories rather
  than from an attacker-chosen ceiling.
* **Accounts are existing customers with real history.** A bust-out portfolio
  that had no prior footprint would be separable on novelty alone.

Realism check (measured, not asserted)
--------------------------------------
Best single-feature depth-1 stump AUC: **0.627** (``ts_epoch``) — campaign timing, not
behaviour. ``amount`` reaches only 0.569 despite a third of the rows being the
drawdown, because the other two thirds are the cultivation. The portfolio is
invisible one transaction at a time, which is exactly the point.

Measured by ``mantis.foundry.injectors.probe`` against a 200k-event background at
seed 1337, over every column an issuer can read off one authorisation message.
Re-measure with ``python -m mantis.foundry --attacks F2-16``.
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

__all__ = ["BustOutAttack", "inject", "main"]

#: Share of campaign volume spent building history rather than spending it.
_CULTIVATION_SHARE: float = 0.70

#: Amount band during cultivation: unremarkable, tightly held.
_CULTIVATION_BAND: tuple[float, float] = (0.18, 0.48)

#: Amount band during the drawdown, inside the target categories.
_DRAWDOWN_BAND: tuple[float, float] = (0.80, 0.98)

#: Categories that convert to value quickly. The drawdown concentrates here.
_CASHOUT_MCCS: tuple[str, ...] = ("5732", "4511", "7011", "5311", "5651", "4722", "5999")

#: Everyday categories a cultivating account transacts in.
_CULTIVATION_MCCS: tuple[str, ...] = (
    "5411",
    "5812",
    "5814",
    "4814",
    "5912",
    "4121",
    "5999",
    "4900",
    "5734",
)

_RAILS: tuple[str, ...] = ("ecom", "agentic", "upi_p2m", "card_present")


@register
class BustOutAttack(BaseAttack):
    """Portfolio cultivation followed by a coordinated drawdown."""

    card_id = "F2-16"
    base_events = 200
    base_campaigns = 4

    def inject(
        self, population: pd.DataFrame, intensity: float, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Emit portfolios that behave impeccably and then stop, together."""
        view = self.view
        active = view.customers[view.customers["n_events"] >= 5].index.to_numpy()
        cultivation_pools = {m: view.merchants_in_mcc(m) for m in _CULTIVATION_MCCS}
        cashout_pools = {m: view.merchants_in_mcc(m, popularity=(0.1, 0.95)) for m in _CASHOUT_MCCS}

        counts = split_count(self.n_events(intensity), self.n_campaigns(intensity), rng)
        # The drawdown must fit inside the window, so campaigns start early.
        starts = view.start_epoch + rng.integers(
            0, max(1, int(0.35 * view.window_seconds)), len(counts)
        )

        blocks: list[pd.DataFrame] = []
        for c, n in enumerate(counts):
            n = int(n)
            n_cultivate = max(1, round(n * _CULTIVATION_SHARE))
            n_draw = max(1, n - n_cultivate)
            n = n_cultivate + n_draw

            portfolio = rng.choice(
                active, size=int(min(np.clip(round(n / 14.0), 6, 16), active.size)), replace=False
            )
            owner = np.concatenate(
                [
                    np.sort(rng.integers(0, portfolio.size, n_cultivate)),
                    rng.integers(0, portfolio.size, n_draw),
                ]
            )
            rows = view.clone(view.source_rows(portfolio[owner], rng, channels=_RAILS))

            mcc_choice = np.concatenate(
                [
                    rng.choice(_CULTIVATION_MCCS, size=n_cultivate),
                    rng.choice(_CASHOUT_MCCS, size=n_draw),
                ]
            )
            pools = {**cultivation_pools, **cashout_pools}
            merchants = np.asarray(
                [pools[m][rng.integers(0, pools[m].size)] for m in mcc_choice], dtype=object
            )
            view.retarget(rows, merchants)

            mcc = rows["mcc"].to_numpy()
            amount = np.empty(n, dtype=float)
            amount[:n_cultivate] = view.draw_amounts(mcc[:n_cultivate], *_CULTIVATION_BAND, rng)
            amount[n_cultivate:] = view.draw_amounts(mcc[n_cultivate:], *_DRAWDOWN_BAND, rng)
            rows["amount"] = amount

            # Shifting cultivation rows independently would destroy the
            # machine-regular cadence that is this attack's whole signal, so
            # each account moves as a unit and the drawdown as a block.
            groups = np.concatenate([owner[:n_cultivate], np.full(n_draw, int(owner.max()) + 1)])
            view.set_timestamps(
                rows,
                self._schedule(starts[c], owner, n_cultivate, n_draw, rng),
                rng=rng,
                groups=groups,
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

    def _schedule(
        self,
        start: int,
        owner: np.ndarray,
        n_cultivate: int,
        n_draw: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Machine-regular cultivation, then a shared drawdown window.

        Each account gets its own base period with a few per cent of jitter. That
        low variance is the signal: real cardholders cluster into bursts and go
        quiet for a week, and this portfolio never does.
        """
        n_accounts = int(owner.max()) + 1
        period = rng.integers(2 * 86_400, 6 * 86_400, n_accounts)
        phase = rng.integers(0, 86_400, n_accounts)

        cultivate_owner = owner[:n_cultivate]
        seq = np.zeros(n_cultivate, dtype=np.int64)
        seen: dict[int, int] = {}
        for i, account in enumerate(cultivate_owner):
            seq[i] = seen.get(int(account), 0)
            seen[int(account)] = seq[i] + 1
        jitter = rng.normal(0.0, 0.045, n_cultivate) * period[cultivate_owner]
        cultivation = (
            start + phase[cultivate_owner] + seq * period[cultivate_owner] + jitter.astype(np.int64)
        )

        # The drawdown is coordinated: one window, whole portfolio.
        payday = int(cultivation.max()) + int(rng.integers(2 * 86_400, 9 * 86_400))
        drawdown = payday + rng.integers(0, 30 * 3_600, n_draw)
        return np.concatenate([cultivation, drawdown])


inject = card_entry_point(BustOutAttack)


def main() -> None:
    """Print a sample portfolio. Run: ``python -m mantis.foundry.injectors.f2_16_bust_out``."""
    demo_main(BustOutAttack)


if __name__ == "__main__":
    main()
