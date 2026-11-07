"""Injector framework — the machinery that makes the atlas *executable*.

Why this module exists
----------------------
Pillar 1 (the 42-card atlas) is only worth a diversity score if it is a
dependency of Pillar 2 rather than a slide deck sitting next to it. That is
enforced here, at import time, by :func:`validate_registry`:

* every atlas card with ``status: implemented`` **must** have a registered
  injector class, and
* every registered injector's ``card_id`` **must** name a real card whose status
  is ``implemented``, and
* the card's declared ``generator`` path **must** resolve to a callable living in
  the injector's own module.

Fail any of those and ``import mantis.foundry.injectors`` raises. There is no
way to claim a card is implemented without the code existing, and no way to
write an injector for a card that does not exist. That is the whole point.

What an injector returns
------------------------
``inject()`` returns **only the new, labelled attack rows**, with exactly the
frozen ``ALL_COLUMNS`` schema. It never mutates the background frame, and never
relabels an existing row. Three reasons, all of which we have to be able to
defend:

1. The Day 1 calibration (amount KS 0.0051, hour TV 0.0066) was measured on the
   background. If injectors edited it in place, every fidelity number would
   silently drift as attacks were added, and the scorecard would be measuring
   the attacks rather than the simulation.
2. Per-attack accounting — counts, per-family recall, the zero-day holdout —
   needs the attack rows to be separable by construction, not by a join.
3. Injectors compose. Eight injectors run against one untouched background and
   the results concatenate, so a run is reproducible attack-by-attack.

How attack rows are built, and why it matters for realism
---------------------------------------------------------
Attack events are **clones of real background rows**, retargeted. An injector
picks source rows belonging to the customers it wants to use, copies them, and
then overrides merchant, amount and timestamp. Everything it does not override —
card BIN, device id, IP, geo, entry mode, 3DS outcome, the shape of the agentic
block — stays exactly as the legitimate population drew it.

This is not a shortcut, it is the realism mechanism. Fraud that only ever
touches freshly-minted entities is trivially detectable: a model learns
"never-before-seen device" and scores 0.99 without understanding anything. Real
card fraud rides on real cardholders' real credentials. Cloning guarantees that
every attack row belongs to a customer who already exists, on a device that
already exists, with a nullity pattern (no ``device_id`` on card-present, missing
geo on a share of remote rails) that already matches the population.

Amounts follow the same principle: :meth:`PopulationView.draw_amounts` resamples
*actual* background amounts from the target MCC inside a quantile band, so the
marginal amount distribution of an attack is, by construction, a slice of the
legitimate one rather than a fresh log-normal with attacker-chosen parameters.

Safety posture
--------------
Per CLAUDE.md HARD RULE 5 these injectors emit **tabular authorisation records
only** — amounts, categories, merchants, timings. Nothing here is operational
tooling: there is no payload, no target, no working technique. The output exists
to be caught by ``mantis.defense``.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, ClassVar, Final

import numpy as np
import pandas as pd

from mantis.atlas.loader import ATLAS
from mantis.atlas.schema import Status
from mantis.core.events import (
    ALL_COLUMNS,
    AgenticContext,
    AuthResponse,
    Channel,
    DisputeOutcome,
    EntryMode,
    MandateScope,
    MandateType,
    ThreeDSResult,
    TxEvent,
    TxnType,
)
from mantis.foundry.base.simulator import IST

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mantis.atlas.schema import AttackCard

__all__ = [
    "REFERENCE_BACKGROUND",
    "REGISTRY",
    "BaseAttack",
    "InjectorError",
    "PopulationView",
    "campaign_id",
    "card_entry_point",
    "demo_main",
    "events_from_frame",
    "get_injector",
    "register",
    "run_injector",
    "split_count",
    "stable_seed",
    "validate_attack_frame",
    "validate_registry",
]

#: Scratch column carrying the cloned row's mandate age, dropped in ``finalise``.
_MANDATE_AGE_COL: Final[str] = "_mandate_age_s"

#: Channels whose authorisation is located at the merchant, not the cardholder.
_PHYSICAL_CHANNELS: Final[frozenset[str]] = frozenset({"card_present", "upi_p2m", "upi_p2p"})

#: Seconds between UTC and IST. Hour-of-day means nothing without it.
_IST_OFFSET_SECONDS: Final[int] = 5 * 3_600 + 30 * 60

#: How far an attack's hour-of-day curve is pulled toward uniform. Fraud really
#: does run later than legitimate traffic, but not flat-uniform: the first draft
#: of these injectors scheduled uniformly and the probe caught ``ts_hour`` as the
#: strongest single feature on three of the eight attacks.
_DIURNAL_UNIFORM_BLEND: Final[float] = 0.18

#: Background size the ``base_events`` figures are calibrated against.
REFERENCE_BACKGROUND: Final[int] = 200_000

#: Page slugs reused when a retargeted provenance chain is rebuilt.
_PAGES: Final[tuple[str, ...]] = ("search", "browse", "reviews", "compare", "list")


class InjectorError(RuntimeError):
    """Raised when the injector registry and the atlas disagree."""


def _short_hash(text: str, width: int = 12) -> str:
    """Deterministic short digest. Matches the convention in the base simulator."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:width]


def stable_seed(text: str) -> int:
    """A reproducible 32-bit seed from a string.

    Python's ``hash()`` is randomised per process unless ``PYTHONHASHSEED`` is
    pinned — the exact defect the Day 1 audit found in the base simulator. Every
    seed derived from a card id goes through here instead.
    """
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def campaign_id(card_id: str, index: int) -> str:
    """Stable campaign identifier, e.g. ``cmp-F4-27-003``."""
    return f"cmp-{card_id}-{index:03d}"


def split_count(
    total: int, parts: int, rng: np.random.Generator, *, spread: float = 0.45
) -> np.ndarray:
    """Split ``total`` events across ``parts`` campaigns, unevenly.

    Real campaigns are not the same size. An even split would make campaign size
    itself a constant of the generator, and the L4 layer would learn a component
    size rather than a topology.
    """
    if parts < 1:
        raise ValueError(f"parts must be positive, got {parts}")
    weights = np.clip(rng.normal(1.0, spread, parts), 0.25, None)
    counts = np.maximum(1, np.floor(total * weights / weights.sum()).astype(int))
    # Hand the rounding remainder to random campaigns so it does not always
    # land on the first one.
    while counts.sum() < total:
        counts[rng.integers(0, parts)] += 1
    while counts.sum() > total and counts.max() > 1:
        candidates = np.flatnonzero(counts > 1)
        counts[candidates[rng.integers(0, candidates.size)]] -= 1
    return counts


# --------------------------------------------------------------------------- #
# The read-only view an injector gets of the background population
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class PopulationView:
    """Derived indices over the background frame, built once and shared.

    An injector sees the population exactly as an issuer does: a flat stream of
    authorisations. It gets no access to the generative ``Population`` object,
    which means "account tenure" has to be inferred from first-seen timestamps
    the way a real risk team would, rather than read off a hidden ground-truth
    column. That constraint is deliberate — an injector that could see the
    generator's internals would produce attacks a detector could not.
    """

    frame: pd.DataFrame
    customers: pd.DataFrame
    merchants: pd.DataFrame
    rows_by_customer: dict[str, np.ndarray]
    rows_by_mcc: dict[str, np.ndarray]
    amounts_by_mcc: dict[str, np.ndarray]
    hour_weights: np.ndarray
    settlement_lag_by_channel: dict[str, float]
    start_epoch: int
    end_epoch: int

    # -- construction -------------------------------------------------------- #

    @classmethod
    def build(cls, frame: pd.DataFrame) -> PopulationView:
        """Index a background frame. Under a second on 200k rows; do it once."""
        missing = set(ALL_COLUMNS) - set(frame.columns)
        if missing:
            raise InjectorError(f"background frame is missing columns: {sorted(missing)}")
        if bool(frame["is_fraud"].any()):
            raise InjectorError("background frame already contains fraud; injectors expect clean")

        epoch = frame["ts"].astype("int64").to_numpy() // 1_000_000_000
        amount = frame["amount"].to_numpy()

        by_customer = {str(k): v for k, v in frame.groupby("customer_id").indices.items()}
        by_mcc = {str(k): v for k, v in frame.groupby("mcc").indices.items()}
        amounts_by_mcc = {mcc: np.sort(amount[idx]) for mcc, idx in by_mcc.items()}

        by_cust = frame.groupby("customer_id")
        customers = pd.DataFrame(
            {
                "first_seen": by_cust["ts"].min().astype("int64") // 1_000_000_000,
                "last_seen": by_cust["ts"].max().astype("int64") // 1_000_000_000,
                "n_events": by_cust.size(),
                "amount_p50": by_cust["amount"].median(),
                "amount_p99": by_cust["amount"].quantile(0.99),
                "amount_max": by_cust["amount"].max(),
                "agentic_share": by_cust["channel"].apply(lambda s: float((s == "agentic").mean())),
            }
        )

        by_mer = frame.groupby("merchant_id")
        merchants = pd.DataFrame(
            {
                "mcc": by_mer["mcc"].first(),
                "merchant_country": by_mer["merchant_country"].first(),
                "n_events": by_mer.size(),
                "lat": by_mer["lat"].median(),
                "lon": by_mer["lon"].median(),
                "amount_p50": by_mer["amount"].median(),
            }
        )
        # Merchant web domains are not on the authorisation message; they are
        # reconstructed from the id under the naming convention in
        # ``foundry.base.entities`` (``mer-<slug>`` <-> ``shop.<slug>.test``) so
        # that a retargeted provenance chain still ends where the money went.
        merchants["domain"] = "shop." + merchants.index.str.slice(4) + ".test"

        # The population's own hour-of-day curve. Attack timestamps are drawn
        # against this rather than uniformly -- see ``set_timestamps``.
        local_hour = ((epoch + _IST_OFFSET_SECONDS) % 86_400) // 3_600
        hour_weights = np.bincount(local_hour, minlength=24).astype(float)
        hour_weights /= hour_weights.sum()

        # Per-rail clearing lag, read off the background rather than assumed.
        # A clone whose source was declined has no lag of its own, and inventing
        # one from a global constant would put card refunds on UPI timings.
        lag_median = frame.groupby("channel")["settlement_lag_hours"].median()
        settlement_lag = {
            str(k): (float(v) if np.isfinite(v) else 24.0) for k, v in lag_median.items()
        }

        return cls(
            frame=frame,
            customers=customers,
            merchants=merchants,
            rows_by_customer=by_customer,
            rows_by_mcc=by_mcc,
            amounts_by_mcc=amounts_by_mcc,
            hour_weights=hour_weights,
            settlement_lag_by_channel=settlement_lag,
            start_epoch=int(epoch.min()),
            end_epoch=int(epoch.max()),
        )

    # -- sampling helpers ---------------------------------------------------- #

    @property
    def window_seconds(self) -> int:
        """Length of the observation window, in seconds."""
        return self.end_epoch - self.start_epoch

    def source_rows(
        self,
        customer_ids: np.ndarray,
        rng: np.random.Generator,
        *,
        channels: tuple[str, ...] | None = None,
    ) -> np.ndarray:
        """One source row position per requested customer, filtered by channel.

        Falls back to any row for that customer when they have none on the
        requested channels — a customer with no agentic history transacting
        agentically is itself realistic, and refusing to model it would bias the
        attack cohort toward heavy agent adopters, which would make
        ``agentic_share`` a giveaway.
        """
        allowed = None if channels is None else list(channels)
        channel_col = self.frame["channel"].to_numpy()
        out = np.empty(len(customer_ids), dtype=np.int64)
        for i, cid in enumerate(customer_ids):
            rows = self.rows_by_customer[str(cid)]
            if allowed is not None:
                filtered = rows[np.isin(channel_col[rows], allowed)]
                if filtered.size:
                    rows = filtered
            out[i] = rows[rng.integers(0, rows.size)]
        return out

    def draw_amounts(
        self,
        mccs: np.ndarray,
        lo_q: float,
        hi_q: float,
        rng: np.random.Generator,
        *,
        jitter: float = 0.06,
    ) -> np.ndarray:
        """Resample real background amounts from each MCC inside a quantile band.

        This is the single most important realism device in the foundry. An
        attack that draws its own amounts from an attacker-chosen distribution
        is detectable on the ``amount`` column alone, which is exactly the
        cartoon we are trying not to build. Resampling the population's own
        empirical amounts — restricted to the band the attack cares about, with
        a few percent of multiplicative jitter so values are not literal
        duplicates — keeps the marginal inside the legitimate support.
        """
        mccs = np.asarray(mccs, dtype=object)
        out = np.empty(len(mccs), dtype=float)
        for mcc in np.unique(mccs):
            sorted_amounts = self.amounts_by_mcc[str(mcc)]
            lo = int(lo_q * (sorted_amounts.size - 1))
            hi = max(lo + 1, int(hi_q * (sorted_amounts.size - 1)))
            band = sorted_amounts[lo : hi + 1]
            mask = mccs == mcc
            count = int(mask.sum())
            picked = band[rng.integers(0, band.size, count)]
            out[mask] = picked * np.exp(rng.normal(0.0, jitter, count))
        return np.round(np.clip(out, 1.0, None), 2)

    def merchants_in_mcc(
        self, mcc: str, *, popularity: tuple[float, float] = (0.0, 1.0)
    ) -> np.ndarray:
        """Merchant ids in one category, restricted to a popularity quantile band.

        ``popularity=(0.0, 0.4)`` returns the long tail — the small merchants a
        laundering or cash-out ring actually recruits, rather than the flagship
        retailer nobody can quietly take over.
        """
        here = self.merchants[self.merchants["mcc"] == mcc]
        if here.empty:
            raise InjectorError(f"no merchant in the background carries mcc {mcc!r}")
        ranked = here.sort_values("n_events")
        lo = int(popularity[0] * len(ranked))
        hi = max(lo + 1, int(popularity[1] * len(ranked)))
        return ranked.index.to_numpy()[lo:hi]

    def spread_epochs(
        self, n: int, rng: np.random.Generator, *, pad_days: float = 3.0
    ) -> np.ndarray:
        """Campaign start times spread across the window, away from its edges.

        **Stratified**, not uniform: the window is cut into ``n`` equal strata
        and one start is drawn inside each, then shuffled. An unstratified draw
        of five starts leaves gaps by chance, and the probe reads those gaps as
        a ``ts_epoch`` signal — a generator artefact that would let a model
        "detect" fraud by learning which fortnight it happened in.
        """
        pad = int(pad_days * 86_400)
        lo = self.start_epoch + pad
        hi = max(lo + 2, self.end_epoch - pad)
        edges = np.linspace(lo, hi, n + 1)
        starts = np.array(
            [
                rng.integers(int(edges[i]), max(int(edges[i]) + 1, int(edges[i + 1])))
                for i in range(n)
            ]
        )
        return rng.permutation(starts)

    # -- row construction ----------------------------------------------------- #

    def clone(self, positions: np.ndarray) -> pd.DataFrame:
        """Copy background rows as the seed for attack events.

        The copy is shallow with respect to the list-valued columns
        (``ag_provenance_chain`` and friends), so callers must **replace** those
        lists rather than mutate them in place. ``finalise`` always replaces.

        The **transaction lifecycle is reset** to an approved, settled purchase.
        A clone inherits its source's whole row, and since Day 3 that row might
        be a decline, a reversal, or a refund pointing at some third party's
        purchase — none of which the injector asked for, and the last of which
        would leave a dangling ``original_event_id``. Resetting here means every
        injector starts from the same neutral lifecycle and has to *opt in* to a
        decline or a refund, which is exactly how F4-27, F4-28 and F1-03 use it.
        """
        rows = self.frame.iloc[positions].copy()
        rows.reset_index(drop=True, inplace=True)
        rows[_MANDATE_AGE_COL] = (rows["ts"] - rows["ag_mandate_issued_ts"]).dt.total_seconds()
        self.reset_lifecycle(rows)
        return rows

    def reset_lifecycle(self, rows: pd.DataFrame) -> None:
        """Make cloned rows approved, settled purchases with no history."""
        rows["txn_type"] = TxnType.PURCHASE.value
        rows["auth_response"] = AuthResponse.APPROVED.value
        rows["original_event_id"] = None
        rows["dispute_outcome"] = None
        rows["dispute_raised_ts"] = pd.NaT
        rows["settled"] = True
        fallback = rows["channel"].map(self.settlement_lag_by_channel).astype(float)
        rows["settlement_lag_hours"] = rows["settlement_lag_hours"].astype(float).fillna(fallback)

    def decline(
        self,
        rows: pd.DataFrame,
        mask: np.ndarray,
        reasons: np.ndarray | str,
    ) -> None:
        """Mark rows declined, and keep the clearing columns coherent.

        A declined authorisation never settles. The schema enforces that, but
        enforcing it *here* means an injector cannot accidentally emit a frame
        that only fails three steps later inside ``events_from_frame``.
        """
        mask = np.asarray(mask, dtype=bool)
        rows.loc[mask, "auth_response"] = reasons
        rows.loc[mask, "settled"] = False
        rows.loc[mask, "settlement_lag_hours"] = np.nan

    def retarget(self, rows: pd.DataFrame, merchant_ids: np.ndarray) -> None:
        """Point cloned rows at different merchants, in place, keeping geo honest.

        Category, acquirer country and — on the rails where the authorisation
        physically happens at the merchant — location all follow the merchant.
        A remote authorisation keeps the cardholder's location, because that is
        what the issuer actually observes.
        """
        merchant_ids = np.asarray(merchant_ids, dtype=object)
        meta = self.merchants.loc[merchant_ids]
        rows["merchant_id"] = merchant_ids
        rows["mcc"] = meta["mcc"].to_numpy()
        rows["merchant_country"] = meta["merchant_country"].to_numpy()

        physical = rows["channel"].isin(_PHYSICAL_CHANNELS).to_numpy()
        m_lat = meta["lat"].to_numpy(dtype=float)
        m_lon = meta["lon"].to_numpy(dtype=float)
        movable = physical & np.isfinite(m_lat)
        if movable.any():
            rows.loc[movable, "lat"] = m_lat[movable]
            rows.loc[movable, "lon"] = m_lon[movable]

        # A terminal id belongs to the merchant it sits in. Card-present rows
        # keep a terminal; every other rail carries None and must keep carrying
        # None, or the nullity pattern alone would name the attack.
        card_present = (rows["channel"] == "card_present").to_numpy()
        if card_present.any():
            lanes = np.arange(len(rows)) % 12
            new_terminals = np.asarray(
                [
                    f"trm-{str(m)[4:]}-{lane:03d}"
                    for m, lane in zip(merchant_ids, lanes, strict=True)
                ],
                dtype=object,
            )
            rows.loc[card_present, "terminal_id"] = new_terminals[card_present]

    def set_timestamps(
        self,
        rows: pd.DataFrame,
        epochs: np.ndarray,
        *,
        rng: np.random.Generator | None = None,
        groups: np.ndarray | None = None,
    ) -> None:
        """Set event times from epoch seconds, clipped into the observed window.

        With ``rng``, each event's **time of day** is redrawn from the
        background's own hour-of-day curve (blended slightly toward uniform, so
        the attack still runs a little later than legitimate traffic) while its
        date is left alone. Without this the injectors scheduled uniformly across
        the 24 hours, and against a diurnal population that made ``ts_hour`` the
        single strongest separating column on three of the eight attacks — a
        pure generator artefact, and exactly the kind of thing the probe exists
        to catch.

        ``groups`` preserves burst structure: every row sharing a group id is
        shifted by the **same** delta, so a four-transfer coerced session keeps
        its 25-minute gaps and its ordering while the session as a whole moves to
        a plausible hour. Rows with no group move independently, which is right
        for attacks whose events are genuinely unrelated attempts.
        """
        epochs = np.asarray(epochs, dtype="int64")
        if rng is not None:
            epochs = epochs + self._diurnal_shift(epochs, rng, groups)
        clipped = np.clip(epochs, self.start_epoch, self.end_epoch).astype("int64")
        rows["ts"] = pd.to_datetime(clipped, unit="s", utc=True).tz_convert(IST)

    def _diurnal_shift(
        self, epochs: np.ndarray, rng: np.random.Generator, groups: np.ndarray | None
    ) -> np.ndarray:
        """Per-row seconds to add so each group lands at a plausible local hour."""
        weights = (1.0 - _DIURNAL_UNIFORM_BLEND) * self.hour_weights + (
            _DIURNAL_UNIFORM_BLEND / 24.0
        )
        weights /= weights.sum()

        keys = np.arange(epochs.size) if groups is None else np.asarray(groups)
        unique, inverse = np.unique(keys, return_inverse=True)
        anchor = np.full(unique.size, np.iinfo(np.int64).max, dtype="int64")
        np.minimum.at(anchor, inverse, epochs)

        target_hour = rng.choice(24, size=unique.size, p=weights)
        target = target_hour * 3_600 + rng.integers(0, 3_600, unique.size)
        current = (anchor + _IST_OFFSET_SECONDS) % 86_400
        return (target - current).astype("int64")[inverse]

    def finalise(
        self,
        rows: pd.DataFrame,
        *,
        card_id: str,
        campaigns: np.ndarray,
        rng: np.random.Generator,
        repair_scope: bool = True,
    ) -> pd.DataFrame:
        """Stamp identity and ground truth, and re-cohere the agentic block.

        Everything an attack changed upstream of here — merchant, amount, time —
        leaves the cloned mandate describing a purchase that no longer happened.
        Repairing it matters: a mandate whose ceiling sits below the amount, or
        whose allow-list names the wrong merchant, is a *different attack*
        (F1-01/F1-02) and would let L0 catch these eight for free. Unless an
        injector explicitly wants that violation, the mandate is made consistent
        with the transaction it authorised.
        """
        campaigns = np.asarray(campaigns, dtype=object)
        rows["event_id"] = [
            f"atk-{card_id.lower()}-{str(c).rsplit('-', 1)[-1]}-{i:05d}"
            for i, c in enumerate(campaigns)
        ]
        rows["is_fraud"] = True
        rows["attack_id"] = card_id
        rows["attack_campaign"] = campaigns

        # Keyed on the agentic *block*, not the rail. Since Day 3 an F1 injector
        # can present an agent-originated purchase over a plain ecom rail (legal
        # under the frozen schema when entry_mode='agent_token', and named by
        # three F1 cards), and those rows need their mandate repairing exactly
        # like the ones that stayed on the agentic channel. For every Day 2
        # injector this mask is identical to the old one.
        agentic = rows["ag_agent_id"].notna().to_numpy()
        if agentic.any():
            self._repair_agentic(rows, agentic, rng=rng, repair_scope=repair_scope)

        rows.drop(columns=[_MANDATE_AGE_COL], inplace=True, errors="ignore")
        return rows[list(ALL_COLUMNS)].reset_index(drop=True)

    def _repair_agentic(
        self,
        rows: pd.DataFrame,
        mask: np.ndarray,
        *,
        rng: np.random.Generator,
        repair_scope: bool,
    ) -> None:
        """Make the cloned mandate describe the transaction it now authorises."""
        idx = np.flatnonzero(mask)
        amount = rows["amount"].to_numpy()
        merchant = rows["merchant_id"].to_numpy()
        mcc = rows["mcc"].to_numpy()
        event_id = rows["event_id"].to_numpy()

        # Mandate identity is a function of the event, so replay detection has
        # something real to key on and two attack events never share a hash.
        mandate_ids = rows["ag_mandate_id"].to_numpy().copy()
        mandate_hashes = rows["ag_mandate_hash"].to_numpy().copy()
        for i in idx:
            mandate_ids[i] = f"mnd-{_short_hash(f'{event_id[i]}-mandate', 10)}"
            mandate_hashes[i] = _short_hash(f"{mandate_ids[i]}|{merchant[i]}|{amount[i]}", 16)
        rows["ag_mandate_id"] = mandate_ids
        rows["ag_mandate_hash"] = mandate_hashes

        # Issued-at keeps the cloned row's mandate age, so the TTL relationship
        # the population drew is preserved against the new timestamp.
        age = rows[_MANDATE_AGE_COL].to_numpy(dtype=float)
        issued = rows["ts"] - pd.to_timedelta(np.nan_to_num(age, nan=60.0), unit="s")
        rows["ag_mandate_issued_ts"] = rows["ag_mandate_issued_ts"].where(~mask, issued)

        if repair_scope:
            headroom = 1.0 + rng.gamma(2.2, 0.28, len(rows))
            new_max = np.round(amount * headroom, 2)
            rows.loc[mask, "ag_scope_max_amount"] = new_max[mask]

        cats = rows["ag_scope_categories"].to_numpy().copy()
        allowed = rows["ag_scope_allowed_merchants"].to_numpy().copy()
        chains = rows["ag_provenance_chain"].to_numpy().copy()
        content = rows["ag_ingested_content_ids"].to_numpy().copy()
        domains = self.merchants["domain"]
        for i in idx:
            existing = list(cats[i]) if cats[i] is not None else []
            cats[i] = existing if str(mcc[i]) in existing else [str(mcc[i]), *existing[:3]]
            if allowed[i] is not None and len(allowed[i]) > 0:
                allowed[i] = [str(merchant[i])]
            # Rebuild the trail on the customer's own habitual domains, ending at
            # the merchant that was actually paid. Keeping the habit is the point:
            # the L3 novel-domain ratio is measured against it.
            old = list(chains[i]) if chains[i] is not None else []
            hosts = [url.split("/")[2] for url in old[:-1]] or ["search.example.test"]
            new_chain = [
                f"https://{host}/{_PAGES[k % len(_PAGES)]}"
                f"?ref={_short_hash(f'{event_id[i]}{k}', 6)}"
                for k, host in enumerate(hosts)
            ]
            domain = str(domains.get(str(merchant[i]), "shop.unknown.test"))
            new_chain.append(f"https://{domain}/product/{_short_hash(domain, 8)}")
            chains[i] = new_chain
            content[i] = [f"sha256:{_short_hash(url)}" for url in new_chain]
        rows["ag_scope_categories"] = cats
        rows["ag_scope_allowed_merchants"] = allowed
        rows["ag_provenance_chain"] = chains
        rows["ag_ingested_content_ids"] = content


# --------------------------------------------------------------------------- #
# The injector contract
# --------------------------------------------------------------------------- #


class BaseAttack(ABC):
    """One atlas card, made executable.

    Subclasses set :attr:`card_id` (which must name an ``implemented`` card),
    :attr:`base_events` (how many authorisations the attack emits at
    ``intensity=1.0``) and :attr:`base_campaigns` (how many independent rings
    that volume is split across), then implement :meth:`inject`.
    """

    #: Atlas card this injector implements. Checked against the atlas at import.
    card_id: ClassVar[str]

    #: Events emitted at ``intensity == 1.0``. Chosen so the whole atlas run
    #: lands well under a 1% overall fraud rate on a 200k background.
    base_events: ClassVar[int] = 120

    #: Independent campaigns that volume is split across. More than one, always:
    #: a single ring makes every graph feature a memorisation of one component.
    base_campaigns: ClassVar[int] = 4

    def __init__(self, view: PopulationView) -> None:
        self.view = view

    # -- sizing --------------------------------------------------------------- #

    def n_events(self, intensity: float) -> int:
        """Event count for this intensity, scaled to the background size.

        ``base_events`` is calibrated against a 200k background. Scaling by the
        actual background keeps **prevalence** — not raw count — the invariant,
        so a 40k smoke run and a 200k gate run both land near six fraud events
        per thousand. Prevalence is the number that decides whether AUC-PR and
        recall@0.1%FPR mean anything; letting it swing with ``--n`` would make
        every metric incomparable between runs. Floored at a fifth so that a
        small demo still produces campaigns with visible structure.
        """
        scale = max(len(self.view.frame) / REFERENCE_BACKGROUND, 0.2)
        return max(12, round(self.base_events * intensity * scale))

    def n_campaigns(self, intensity: float) -> int:
        """Campaign count for this intensity, at least one.

        Sub-linear in intensity on purpose: turning the dial up should make each
        ring bigger and noisier, not spawn a proportional number of tiny ones.
        """
        return max(1, round(self.base_campaigns * min(intensity, 4.0) ** 0.5))

    @property
    def card(self) -> AttackCard:
        """The atlas card this injector implements."""
        return ATLAS[self.card_id]

    #: Columns :meth:`probe_slice` is allowed to read. Empty when no slice is
    #: declared. Every name must appear in ``probe.SLICE_ALLOWED_COLUMNS``, and
    #: ``tests/test_probe_slices.py`` proves the mask really is a function of
    #: these columns alone -- a slice that secretly read ``amount`` or the
    #: provenance chain would be conditioning on the attack's own footprint, and
    #: the conditional AUC it produced would mean nothing.
    slice_columns: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def probe_slice(cls, frame: pd.DataFrame) -> np.ndarray | None:
        """The background rows this attack should be measured *against*.

        Return ``None`` (the default) to be measured against the whole
        population. Return a boolean mask to declare that the attack lives
        inside a **definitional slice** of traffic, and that the honest question
        is whether one column separates it *within* that slice.

        Why this exists, and why it is not a way of grading ourselves gently
        ---------------------------------------------------------------------
        Some properties of an attack are not evidence, they are its definition.
        Every F1 attack carries an agentic block, and only ~15% of the
        population does, so *any* F1 injector scores ~0.92 on the nullity of
        *any* ``ag_`` column before it has done anything at all. F1-03 is a
        refund attack, and refunds are ~2% of traffic, so ``txn_type=refund``
        alone scores ~0.99. Neither number says the attack is easy to detect;
        both say the attack happens on a rail we already knew it happened on. An
        issuer does not congratulate itself for noticing that a refund is a
        refund.

        This is the same reasoning the Day 1 audit settled on for the rail
        (CLAUDE.md, "rail identity is unhideable, and that is fine"): columns
        that are definitional get no bound, and the test that protects us is the
        *conditional* one. Declaring the slice here makes that conditioning an
        explicit, reviewable property of each injector rather than a special
        case buried in the probe.

        The gate applies to the **conditional** number when a slice is declared.
        The unconditional number is still computed and still printed, so nobody
        can accuse us of quietly moving the denominator; it simply is not the
        number that decides whether an attack is a cartoon.

        **The constraint that makes this legitimate.** A slice may condition only
        on facts a detector already knows before it scores, and which are not
        consequences of the attack -- rail, processing code, category. It may not
        condition on anything the attack itself produced. Slicing F1-01 to
        "agent-mediated" is fair; slicing it to "agent-mediated and provenance
        chain longer than three" would be marking our own homework. An injector
        declaring its own slice is self-assessment unless that rule is enforced,
        so it is: name the columns in :attr:`slice_columns`, and
        ``tests/test_probe_slices.py`` checks both that they are on the
        allow-list and that the returned mask genuinely depends on nothing else.
        """
        return None

    @abstractmethod
    def inject(
        self, population: pd.DataFrame, intensity: float, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Emit labelled attack events against ``population``.

        Args:
            population: The clean background frame. **Never mutated.**
            intensity: Volume multiplier; 1.0 is the calibrated default.
            rng: Seeded generator. All randomness must come from here.

        Returns:
            A frame with exactly ``ALL_COLUMNS``, every row ``is_fraud=True``,
            ``attack_id == self.card_id``, and ``attack_campaign`` grouping rows
            into rings.
        """


REGISTRY: Final[dict[str, type[BaseAttack]]] = {}


def register(cls: type[BaseAttack]) -> type[BaseAttack]:
    """Class decorator adding an injector to the registry."""
    card_id = getattr(cls, "card_id", None)
    if not card_id:
        raise InjectorError(f"{cls.__name__} does not set card_id")
    existing = REGISTRY.get(card_id)
    if existing is not None:
        if existing.__qualname__ != cls.__qualname__:
            raise InjectorError(
                f"two injectors claim card {card_id!r}: "
                f"{existing.__module__}.{existing.__qualname__} and "
                f"{cls.__module__}.{cls.__qualname__}"
            )
        # Same class, re-executed. ``python -m mantis.foundry.injectors.f4_27_...``
        # runs the module a second time under the name ``__main__`` after the
        # package has already imported it. Keeping the first, correctly-packaged
        # registration is what lets every injector module stay runnable on its
        # own (CLAUDE.md §5) without the registry seeing a phantom duplicate.
        return cls
    REGISTRY[card_id] = cls
    return cls


def get_injector(card_id: str) -> type[BaseAttack]:
    """Look up an injector class by atlas card id."""
    try:
        return REGISTRY[card_id]
    except KeyError:
        raise InjectorError(
            f"no injector registered for {card_id!r}; registered: {sorted(REGISTRY)}"
        ) from None


def validate_registry() -> None:
    """Assert the atlas and the injector set agree. Called at package import.

    This is the assertion that keeps Pillar 1 honest. Four things are checked,
    and every one of them is a way a hackathon project quietly overclaims:

    1. An injector naming a card that does not exist.
    2. An injector for a card the atlas still calls ``mapped`` — code the
       writeup does not know about.
    3. A card claiming ``implemented`` with nothing behind it — the overclaim
       that matters, because it is the number that goes on the slide.
    4. A ``generator`` path in the card that does not resolve to a callable in
       the injector's own module, i.e. documentation that has drifted from code.
    """
    for card_id, cls in sorted(REGISTRY.items()):
        card = ATLAS.get(card_id)
        if card is None:
            raise InjectorError(
                f"{cls.__module__}.{cls.__name__} claims card {card_id!r}, "
                "which is not in the atlas"
            )
        if card.status is not Status.IMPLEMENTED:
            raise InjectorError(
                f"{card_id} has an injector ({cls.__name__}) but the atlas still calls it "
                f"{card.status.value!r}; promote the card or drop the injector"
            )
        if card.generator is None:
            raise InjectorError(f"{card_id} is implemented but names no generator path")
        module_path, _, func_name = card.generator.partition(":")
        if module_path != cls.__module__:
            raise InjectorError(
                f"{card_id} declares generator module {module_path!r} but its injector "
                f"lives in {cls.__module__!r}"
            )
        entry = getattr(import_module(module_path), func_name, None)
        if not callable(entry):
            raise InjectorError(
                f"{card_id} declares generator {card.generator!r}, which does not resolve "
                "to a callable"
            )

    claimed = {c.id for c in ATLAS.values() if c.status is Status.IMPLEMENTED}
    missing = sorted(claimed - set(REGISTRY))
    if missing:
        raise InjectorError(
            f"atlas cards claim status='implemented' with no registered injector: {missing}. "
            "The atlas must not promise generation it cannot do -- either write the injector "
            "or set the card back to status='mapped'."
        )


# --------------------------------------------------------------------------- #
# Output contract
# --------------------------------------------------------------------------- #


def validate_attack_frame(rows: pd.DataFrame, card_id: str, background: pd.DataFrame) -> None:
    """Assert an injector honoured the contract. Cheap; run it on every call."""
    if list(rows.columns) != list(ALL_COLUMNS):
        raise InjectorError(f"{card_id}: columns are not ALL_COLUMNS in order")
    if rows.empty:
        raise InjectorError(f"{card_id}: injector produced no rows")
    if not bool(rows["is_fraud"].all()):
        raise InjectorError(f"{card_id}: injector emitted an unlabelled row")
    if not bool((rows["attack_id"] == card_id).all()):
        raise InjectorError(f"{card_id}: attack_id does not match the injector's card")
    if bool(rows["attack_campaign"].isna().any()):
        raise InjectorError(f"{card_id}: every attack row must belong to a campaign")
    if bool(rows["event_id"].duplicated().any()):
        raise InjectorError(f"{card_id}: duplicate event_id within the attack")
    if bool(rows["event_id"].isin(background["event_id"]).any()):
        raise InjectorError(f"{card_id}: attack event_id collides with the background")

    # The lifecycle block must describe something that can happen. The frozen
    # schema checks this too, but only once the frame is rebuilt into events;
    # catching it here names the injector that did it.
    declined = rows["auth_response"] != AuthResponse.APPROVED.value
    if bool((declined & rows["settled"].astype(bool)).any()):
        raise InjectorError(f"{card_id}: a declined authorisation is marked settled")
    unsettled = ~rows["settled"].astype(bool)
    if bool((unsettled & rows["settlement_lag_hours"].notna()).any()):
        raise InjectorError(f"{card_id}: settlement lag on an unsettled authorisation")
    inbound = rows["txn_type"].isin([TxnType.PURCHASE.value, TxnType.PREAUTH.value])
    if bool((inbound & rows["original_event_id"].notna()).any()):
        raise InjectorError(f"{card_id}: original_event_id on a purchase or pre-authorisation")
    linked = rows["original_event_id"].dropna()
    if len(linked):
        known = set(background["event_id"]) | set(rows["event_id"])
        unknown = [e for e in linked if e not in known]
        if unknown:
            raise InjectorError(
                f"{card_id}: {len(unknown)} rows reference an original_event_id that is in "
                "neither the background nor the attack; a refund must name a real purchase"
            )

    # Entities must be reused, not invented. Fraud that only ever touches
    # never-before-seen customers and merchants is trivially separable, and a
    # detector trained on it learns nothing transferable.
    for column in ("customer_id", "merchant_id"):
        unknown = ~rows[column].isin(background[column])
        if bool(unknown.any()):
            raise InjectorError(
                f"{card_id}: {int(unknown.sum())} rows use a {column} absent from the "
                "background; injectors must reuse existing entities"
            )


def card_entry_point(cls: type[BaseAttack]):
    """Build the module-level ``inject`` the atlas ``generator`` path points at.

    The atlas names a *function*, not a class, because a card should be able to
    say "here is how you generate me" without knowing how the foundry is
    structured internally. This factory produces that function.

    It is the convenience path, not the fast path: it rebuilds the
    :class:`PopulationView` on every call. Batch runs go through
    :func:`run_injector` with a view built once.
    """

    def inject(
        population: pd.DataFrame,
        intensity: float = 1.0,
        rng: np.random.Generator | None = None,
        *,
        seed: int = 1337,
    ) -> pd.DataFrame:
        """Generate this card's attack events against ``population``."""
        view = PopulationView.build(population)
        if rng is None:
            rng = np.random.default_rng([seed, stable_seed(cls.card_id)])
        rows = cls(view).inject(population, intensity, rng)
        validate_attack_frame(rows, cls.card_id, population)
        return rows

    inject.__doc__ = f"Generate {cls.card_id} attack events against a background frame."
    return inject


def demo_main(cls: type[BaseAttack], *, n_events: int = 20_000) -> None:
    """Print a readable sample of one injector's output.

    Every injector module is runnable as ``python -m`` (CLAUDE.md §5). It builds
    its own small background so it works from a clean clone with nothing in
    ``data/generated``.
    """
    from mantis.foundry.base.reference import load_reference_stats
    from mantis.foundry.base.simulator import SimulationConfig, simulate_frame

    cfg = SimulationConfig(n_events=n_events, seed=7, n_customers=900, n_merchants=2_200)
    background = simulate_frame(cfg, load_reference_stats())
    view = PopulationView.build(background)
    rows = run_injector(cls, view, seed=7)

    card = ATLAS[cls.card_id]
    print(f"{card.id}  {card.name}   [{card.family_name}]")
    print(f"  background {len(background):,} events -> {len(rows):,} attack events")
    print(f"  campaigns  {rows['attack_campaign'].nunique()}")
    print(
        f"  customers  {rows['customer_id'].nunique()}   merchants {rows['merchant_id'].nunique()}"
    )
    print(f"  rails      {dict(rows['channel'].value_counts())}")
    print(f"  txn types  {dict(rows['txn_type'].value_counts())}")
    declined = rows["auth_response"] != "approved"
    reasons = dict(rows.loc[declined, "auth_response"].value_counts())
    print(f"  declined   {declined.mean():.1%}  {reasons}")
    print(
        f"  amount     median {rows['amount'].median():,.0f}  "
        f"p90 {rows['amount'].quantile(0.9):,.0f}  max {rows['amount'].max():,.0f}"
    )
    print()
    sample = rows.sort_values("ts").head(10)
    for _, row in sample.iterrows():
        print(
            f"  {row['ts']:%Y-%m-%d %H:%M} {row['channel']:<13} mcc={row['mcc']} "
            f"{row['amount']:>10,.2f}  {row['merchant_id']:<22} {row['attack_campaign']}"
        )


def _opt(value: object) -> object:
    """Collapse pandas' several flavours of missing into ``None``."""
    if value is None:
        return None
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except (TypeError, ValueError):  # arrays and lists are never NA here
        pass
    return value


def _as_list(value: object) -> list[str]:
    """List-valued columns come back from parquet as arrays, not lists."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [str(item) for item in value]  # type: ignore[union-attr]


def events_from_frame(frame: pd.DataFrame) -> Iterator[TxEvent]:
    """Rebuild validated ``TxEvent`` objects from a flat frame.

    ``flatten`` has no inverse in ``core.events`` — the schema is frozen and the
    foundry is the only thing that needs to travel backwards — so the inverse
    lives here. It is what lets the gate *prove*, rather than assert by eyeball,
    that injected rows satisfy every validator on the frozen schema: the 4-digit
    MCC, the ISO currency and country codes, rail consistency between ``channel``
    and the agentic block, and label integrity between ``is_fraud`` and
    ``attack_id``. An injector that produced a malformed event fails here rather
    than three days later in the defense layer.
    """
    for row in frame.to_dict("records"):
        agentic = None
        if _opt(row["ag_agent_id"]) is not None:
            scope = MandateScope(
                categories=_as_list(row["ag_scope_categories"]),
                max_amount=_opt(row["ag_scope_max_amount"]),  # type: ignore[arg-type]
                max_items=(
                    None
                    if _opt(row["ag_scope_max_items"]) is None
                    else int(row["ag_scope_max_items"])
                ),
                allowed_merchants=_as_list(row["ag_scope_allowed_merchants"]),
                ttl_seconds=(
                    None
                    if _opt(row["ag_scope_ttl_seconds"]) is None
                    else int(row["ag_scope_ttl_seconds"])
                ),
            )
            agentic = AgenticContext(
                agent_id=str(row["ag_agent_id"]),
                agent_platform=str(row["ag_agent_platform"]),
                kya_token=_opt(row["ag_kya_token"]),  # type: ignore[arg-type]
                kya_registered=bool(row["ag_kya_registered"]),
                mandate_type=MandateType(str(row["ag_mandate_type"])),
                mandate_id=_opt(row["ag_mandate_id"]),  # type: ignore[arg-type]
                mandate_hash=_opt(row["ag_mandate_hash"]),  # type: ignore[arg-type]
                mandate_issued_ts=_opt(row["ag_mandate_issued_ts"]),  # type: ignore[arg-type]
                mandate_ttl_seconds=(
                    None
                    if _opt(row["ag_mandate_ttl_seconds"]) is None
                    else int(row["ag_mandate_ttl_seconds"])
                ),
                mandate_scope=scope,
                human_present=bool(row["ag_human_present"]),
                consent_sig_valid=_opt(row["ag_consent_sig_valid"]),  # type: ignore[arg-type]
                delegation_depth=int(row["ag_delegation_depth"]),
                provenance_chain=_as_list(row["ag_provenance_chain"]),
                ingested_content_ids=_as_list(row["ag_ingested_content_ids"]),
                tool_call_count=int(row["ag_tool_call_count"]),
                deliberation_latency_ms=(
                    None
                    if _opt(row["ag_deliberation_latency_ms"]) is None
                    else int(row["ag_deliberation_latency_ms"])
                ),
                cursor_entropy=_opt(row["ag_cursor_entropy"]),  # type: ignore[arg-type]
                dwell_time_ms=(
                    None if _opt(row["ag_dwell_time_ms"]) is None else int(row["ag_dwell_time_ms"])
                ),
            )
        yield TxEvent(
            event_id=str(row["event_id"]),
            ts=row["ts"],
            amount=float(row["amount"]),
            currency=str(row["currency"]),
            mcc=str(row["mcc"]),
            channel=Channel(str(row["channel"])),
            entry_mode=EntryMode(str(row["entry_mode"])),
            customer_id=str(row["customer_id"]),
            card_bin=str(row["card_bin"]),
            merchant_id=str(row["merchant_id"]),
            merchant_country=str(row["merchant_country"]),
            terminal_id=_opt(row["terminal_id"]),  # type: ignore[arg-type]
            device_id=_opt(row["device_id"]),  # type: ignore[arg-type]
            ip=_opt(row["ip"]),  # type: ignore[arg-type]
            lat=_opt(row["lat"]),  # type: ignore[arg-type]
            lon=_opt(row["lon"]),  # type: ignore[arg-type]
            threeds_result=ThreeDSResult(str(row["threeds_result"])),
            txn_type=TxnType(str(row["txn_type"])),
            auth_response=AuthResponse(str(row["auth_response"])),
            original_event_id=_opt(row["original_event_id"]),  # type: ignore[arg-type]
            dispute_outcome=(
                None
                if _opt(row["dispute_outcome"]) is None
                else DisputeOutcome(str(row["dispute_outcome"]))
            ),
            dispute_raised_ts=_opt(row["dispute_raised_ts"]),  # type: ignore[arg-type]
            settled=bool(row["settled"]),
            settlement_lag_hours=_opt(row["settlement_lag_hours"]),  # type: ignore[arg-type]
            agentic=agentic,
            is_fraud=bool(row["is_fraud"]),
            attack_id=_opt(row["attack_id"]),  # type: ignore[arg-type]
            attack_campaign=_opt(row["attack_campaign"]),  # type: ignore[arg-type]
        )


def run_injector(
    cls: type[BaseAttack],
    view: PopulationView,
    *,
    intensity: float = 1.0,
    seed: int = 1337,
) -> pd.DataFrame:
    """Instantiate, run and validate one injector with a card-derived stream.

    The generator is seeded on ``(seed, card_id)`` rather than drawn from a
    shared stream, so adding or removing an injector cannot perturb any other
    injector's output. A judge re-running with ``--attacks F4-27`` gets exactly
    the F4-27 rows a full run produced.
    """
    rng = np.random.default_rng([seed, stable_seed(cls.card_id)])
    before = len(view.frame)
    rows = cls(view).inject(view.frame, intensity, rng)
    if len(view.frame) != before:
        raise InjectorError(f"{cls.card_id}: injector mutated the background frame")
    validate_attack_frame(rows, cls.card_id, view.frame)
    return rows
