"""Applying a genome to the rows an injector emitted.

The contract this module holds
--------------------------------
A mutated variant must still be a **valid, schema-conformant, labelled instance
of its card**. Everything the injector framework guarantees — clones of real
background rows, amounts inside the population's own per-MCC band, non-negative
amounts, ``ALL_COLUMNS`` exactly — has to survive the transformation, or the loop
would be manufacturing training data that no injector could have produced and the
evasion curve would be measuring the mutator's bugs.

So each transformation is written to preserve those properties:

* amounts are scaled and then **clipped back into the background's own per-MCC
  quantile band**, so ``amount_scale`` cannot walk a variant off the end of the
  legitimate distribution and win by being absurd;
* timestamps stay inside the observation window;
* re-drawn merchants come from the background's merchant table and re-drawn
  devices from the background's own devices, so a variant never invents an
  entity — the same rule that makes the base injectors realistic;
* the provenance chain stays **length-preserving** when pages are cleaned, which
  is the Day 3 fix that stopped ``ag_provenance_chain_len`` from being a 0.96
  detector. A mutation that reintroduced it would let the loop "evade" by
  breaking a property of the generator rather than of the attack.

``mutate_rows`` is deliberately total: any genome in the box produces a frame
that :func:`~mantis.foundry.injectors.base.validate_attack_frame` accepts, and
the arena asserts that on every variant it generates.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from mantis.foundry.injectors.base import PopulationView, campaign_id, stable_seed
from mantis.foundry.llm.corpus import ContentStore, load_content_store

__all__ = ["mutate_rows"]

#: Quantile band the background's per-MCC amounts are clipped to. Wide enough
#: that ``amount_scale`` has somewhere to move, narrow enough that it cannot
#: escape the population.
_AMOUNT_BAND: Final[tuple[float, float]] = (0.02, 0.995)


def _scale_amounts(
    rows: pd.DataFrame, view: PopulationView, factor: float
) -> np.ndarray:
    """Scale, then clip into each MCC's own observed band."""
    amount = rows["amount"].to_numpy(dtype=float) * factor
    mcc = rows["mcc"].to_numpy()
    out = amount.copy()
    for code in np.unique(mcc):
        pool = view.amounts_by_mcc.get(str(code))
        if pool is None or pool.size == 0:
            continue
        low, high = np.quantile(pool, _AMOUNT_BAND)
        mask = mcc == code
        out[mask] = np.clip(out[mask], low, high)
    return np.maximum(out, 1.0)


def _respread_time(
    rows: pd.DataFrame, view: PopulationView, spread: float, hour_shift: float
) -> pd.Series:
    """Stretch each campaign around its own midpoint, then shift the clock.

    Stretching about the midpoint rather than the start keeps the campaign in the
    same part of the calendar, so ``time_spread`` changes the *pace* of the ring
    and not when it happened. Letting it change both would confound the gene with
    a plain calendar shift, and the arena could not attribute an evasion to it.
    """
    epoch = rows["ts"].astype("int64").to_numpy() // 1_000_000_000
    out = epoch.astype(float)
    for campaign in pd.unique(rows["attack_campaign"]):
        mask = (rows["attack_campaign"] == campaign).to_numpy()
        if not mask.any():
            continue
        centre = out[mask].mean()
        out[mask] = centre + (out[mask] - centre) * spread
    out = out + hour_shift * 3_600.0
    out = np.clip(out, view.start_epoch, view.end_epoch)
    return pd.to_datetime(np.round(out).astype("int64"), unit="s", utc=True).tz_convert(
        rows["ts"].dt.tz
    )


def _refanout(
    rows: pd.DataFrame, card_id: str, fanout: float, rng: np.random.Generator
) -> np.ndarray:
    """Split each existing campaign into ``fanout`` sub-rings.

    The operator's answer to being rolled up: run four small rings instead of one
    big one. Sub-rings keep their parent's rows contiguous in time where they can,
    because a ring that interleaves randomly with three others is not four rings,
    it is one ring with a relabelled column.
    """
    parts = max(1, round(fanout))
    if parts == 1:
        return rows["attack_campaign"].to_numpy()
    out = rows["attack_campaign"].astype(str).to_numpy().copy()
    for index, campaign in enumerate(pd.unique(rows["attack_campaign"])):
        mask = (rows["attack_campaign"] == campaign).to_numpy()
        positions = np.flatnonzero(mask)
        if positions.size <= parts:
            continue
        order = positions[np.argsort(rows["ts"].to_numpy()[positions], kind="stable")]
        blocks = np.array_split(order, parts)
        for block_index, block in enumerate(blocks):
            out[block] = campaign_id(card_id, index * 10 + block_index)
    _ = rng
    return out


def _redraw(
    values: np.ndarray, pool: np.ndarray, share: float, rng: np.random.Generator
) -> np.ndarray:
    """Replace ``share`` of ``values`` with draws from ``pool``, leaving NaN alone."""
    if share <= 0.0 or pool.size == 0:
        return values
    out = values.copy()
    eligible = np.flatnonzero(pd.notna(values))
    if eligible.size == 0:
        return out
    chosen = eligible[rng.random(eligible.size) < share]
    out[chosen] = pool[rng.integers(0, pool.size, chosen.size)]
    return out


def _clean_provenance(
    rows: pd.DataFrame, share: float, rng: np.random.Generator, store: ContentStore
) -> np.ndarray:
    """Swap a share of injected pages back to benign content, length-preserving.

    The gene aimed at L3. Each injected content id has a probability ``share`` of
    being replaced by a benign artefact's id, and the chain keeps its length and
    its terminating merchant page — so this is a variant that read *less* hostile
    text, not a variant with a differently-shaped trail.
    """
    contents = rows["ag_ingested_content_ids"].to_numpy().copy()
    if share <= 0.0:
        return contents
    benign = store.benign_pool
    if not benign:
        return contents
    # Content ids are digests of URLs; a "cleaned" slot is bound to a benign
    # artefact under a fresh synthetic id so the store still resolves it.
    for i, chain in enumerate(contents):
        if chain is None or len(chain) == 0:
            continue
        chain = list(chain)
        changed = False
        for position, content_id in enumerate(chain[:-1]):  # never the merchant page
            artifact = store.resolve(str(content_id))
            if artifact is None or not artifact.is_injected:
                continue
            if rng.random() >= share:
                continue
            replacement = benign[int(rng.integers(0, len(benign)))]
            # stable_seed, not hash(): see AttackGenome.label.
            token = stable_seed(f"{content_id}|{replacement}|{position}")
            fresh = f"sha256:{token % (16**12):012x}"
            store.bind(fresh, replacement)
            chain[position] = fresh
            changed = True
        if changed:
            contents[i] = chain
    return contents


def mutate_rows(
    rows: pd.DataFrame,
    genome: object,
    view: PopulationView,
    rng: np.random.Generator,
    *,
    store: ContentStore | None = None,
) -> pd.DataFrame:
    """Apply ``genome`` to an injector's output. Returns a new frame.

    Args:
        rows: Attack rows straight out of
            :func:`~mantis.foundry.injectors.base.run_injector`. Not mutated.
        genome: An :class:`~mantis.loop.genome.AttackGenome`.
        view: The background these rows were injected against.
        rng: Seeded generator.
        store: Content store the provenance gene rebinds into. The process-wide
            one by default.
    """
    out = rows.copy()
    store = store if store is not None else load_content_store()

    out["amount"] = _scale_amounts(out, view, float(genome.amount_scale))  # type: ignore[attr-defined]
    out["attack_campaign"] = _refanout(
        out, str(genome.card_id), float(genome.campaign_fanout), rng  # type: ignore[attr-defined]
    )
    out["ts"] = _respread_time(
        out, view, float(genome.time_spread), float(genome.hour_shift)  # type: ignore[attr-defined]
    )

    merchant_pool = view.merchants.index.to_numpy()
    out["merchant_id"] = _redraw(
        out["merchant_id"].to_numpy(), merchant_pool, float(genome.merchant_spread), rng  # type: ignore[attr-defined]
    )
    device_pool = pd.unique(view.frame["device_id"].dropna()).astype(object)
    out["device_id"] = _redraw(
        out["device_id"].to_numpy(), device_pool, float(genome.device_rotate), rng  # type: ignore[attr-defined]
    )

    agentic = out["ag_agent_id"].notna().to_numpy()
    if agentic.any():
        depth = out["ag_delegation_depth"].to_numpy(dtype=float)
        shifted = np.clip(depth + round(float(genome.delegation_delta)), 1.0, 8.0)  # type: ignore[attr-defined]
        out["ag_delegation_depth"] = np.where(agentic, shifted, depth)

        latency = out["ag_deliberation_latency_ms"].to_numpy(dtype=float)
        out["ag_deliberation_latency_ms"] = np.where(
            agentic, latency * float(genome.deliberation_scale), latency  # type: ignore[attr-defined]
        )
        out["ag_ingested_content_ids"] = _clean_provenance(
            out, float(genome.provenance_clean), rng, store  # type: ignore[attr-defined]
        )

    return out.sort_values("ts", kind="stable").reset_index(drop=True)
