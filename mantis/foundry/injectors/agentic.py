"""Shared machinery for the F1 (mandate & delegation) injectors.

The design constraint this module exists to enforce
----------------------------------------------------
The easy version of "agentic fraud" is five attacks that each break a protocol
rule, five deterministic L0 checks that catch them, and an ML story that
collapses to *"we wrote some if-statements"*. If every agentic attack is a clean
protocol violation, L1 and L2 have nothing to do on the rail that the whole
project is about.

So the F1 injectors are deliberately split into two buckets, declared on the
class and asserted by ``tests/test_agentic_injectors.py``:

``HARD``
    The attack breaks a rule the network can check on a single authorisation:
    an expired mandate, a replayed hash, a category outside the signed scope, an
    unverified consent signature. **L0 should catch these at near-zero false
    positive rate**, and if it does not, L0 is broken. F1-02, F1-10 and the
    consent/KYA half of F1-09.

``CLEAN``
    The attack satisfies **every** protocol rule. The mandate is genuine,
    unexpired, in scope, correctly signed, from a registered agent, and the
    amount is under the ceiling the human agreed. There is no rule to write. The
    only evidence is behavioural — where the agent went, how long it thought,
    what it read — which is L1, L2 and L3 territory by construction. F1-01 and
    F1-03.

The CLEAN bucket is the one that matters. It is the honest answer to "what
happens when the attacker does not make a mistake", and it is the reason the
firewall needs five layers instead of one.

What lives here
---------------
Four things every F1 injector needs and none of the Day 2 injectors did:

* :func:`spread_across_rails` — F1-01, F1-03 and F1-09 name ``ecom`` as well as
  ``agentic`` on their cards, and putting a share of events there is not
  cosmetic. An attack that is 100% agentic sits at ~0.92 single-feature AUC on
  ``ag_agent_id_isnull`` before it does anything at all, because the rail is only
  15% of the population. Splitting the rail is what makes the probe measure the
  attack instead of the channel.
* :func:`plant_injected_content` — rewrites the provenance chain to run through
  attacker-controlled domains and binds those URLs to real text in the content
  store, so ``ingested_content_ids`` resolves to something L3 can classify.
* :func:`collapse_deliberation` — resamples the *population's own* deliberation
  latencies from a low quantile band, rather than inventing a small number. Same
  principle as ``PopulationView.draw_amounts``: an attacker-chosen distribution
  is detectable on the column alone, and that is the cartoon we are avoiding.
* :class:`AgenticAttack` — the base class carrying the bucket declaration.
"""

from __future__ import annotations

from typing import ClassVar, Final

import numpy as np
import pandas as pd

from mantis.core.events import Channel, EntryMode
from mantis.foundry.injectors.base import BaseAttack, PopulationView, _short_hash
from mantis.foundry.llm.corpus import ContentStore, content_id_for_url, load_content_store

__all__ = [
    "ATTACKER_DOMAINS",
    "AgenticAttack",
    "Bucket",
    "agentic_pool",
    "collapse_deliberation",
    "novel_merchants_for",
    "plant_injected_content",
    "spread_across_rails",
]


class Bucket:
    """Which half of the L0 split an injector belongs to. See the module docstring."""

    HARD: Final[str] = "hard"
    CLEAN: Final[str] = "clean"


#: The attacker's content infrastructure. A **small, shared** pool on purpose:
#: one operator serves many victims from the same handful of hosts, so the L4
#: layer sees domain fan-in across unrelated customers. Giving every attack event
#: a freshly-minted domain would destroy that graph structure and would also make
#: "domain seen exactly once" a free detector.
ATTACKER_DOMAINS: Final[tuple[str, ...]] = tuple(
    f"{slug}.{tld}.test"
    for slug, tld in (
        ("deals-verified", "offers"),
        ("price-compare-now", "offers"),
        ("stock-notice", "cdn"),
        ("listing-update", "cdn"),
        ("bestbuy-guide", "reviews"),
        ("trusted-picks", "reviews"),
        ("returns-desk", "support"),
        ("order-help", "support"),
        ("refund-status", "support"),
        ("checkout-assist", "pay"),
        ("secure-basket", "pay"),
        ("vendor-switch", "pay"),
    )
)

#: Page slugs on the attacker's hosts. Ordinary-looking: the URL is not the tell,
#: the *content behind it* is, which is precisely why L3 has to exist.
_ATTACK_PAGES: Final[tuple[str, ...]] = (
    "listing",
    "offer",
    "notice",
    "update",
    "guide",
    "ticket",
)


class AgenticAttack(BaseAttack):
    """An F1 injector, carrying its L0 bucket declaration.

    Subclasses must set :attr:`bucket`. ``tests/test_agentic_injectors.py``
    asserts both that the field is set and that the *behaviour* matches it: a
    ``CLEAN`` injector that emits a scope violation, an expired mandate or an
    invalid consent signature fails the test. The declaration is a contract, not
    a comment.
    """

    #: ``Bucket.HARD`` or ``Bucket.CLEAN``.
    bucket: ClassVar[str]

    #: Rail identity only. See :meth:`BaseAttack.probe_slice`.
    slice_columns: ClassVar[tuple[str, ...]] = ("ag_agent_id",)

    @classmethod
    def probe_slice(cls, frame: pd.DataFrame) -> np.ndarray:
        """Agent-mediated traffic. See :meth:`BaseAttack.probe_slice`.

        Keyed on the presence of the agentic **block**, not on
        ``channel == 'agentic'``: an F1 attack may present over a plain ecom
        rail under ``entry_mode='agent_token'``, and it still carries every
        ``ag_`` column when it does. Conditioning on the channel would leave the
        nullity of those columns doing the separating, which is the artefact
        this slice exists to remove.
        """
        return frame["ag_agent_id"].notna().to_numpy()


# --------------------------------------------------------------------------- #
# Rail selection
# --------------------------------------------------------------------------- #


def agentic_pool(view: PopulationView, *, min_events: int = 3) -> np.ndarray:
    """Row positions of agentic authorisations by reasonably active customers.

    Attacks need a customer with a history, because most of what the firewall
    will use — novelty, velocity, deviation from a baseline — is undefined for a
    customer whose only transaction is the attack.
    """
    frame = view.frame
    active = set(view.customers[view.customers["n_events"] >= min_events].index)
    is_agentic = frame["channel"].to_numpy() == Channel.AGENTIC.value
    known = frame["customer_id"].isin(active).to_numpy()
    pool = np.flatnonzero(is_agentic & known)
    if pool.size == 0:
        raise ValueError("no agentic rows in the background; cannot inject an F1 attack")
    return pool


def spread_across_rails(
    rows: pd.DataFrame, share_ecom: float, rng: np.random.Generator
) -> np.ndarray:
    """Move a share of agentic rows onto the ecom rail, keeping the agent block.

    An agent-originated purchase presented over a plain ecom rail is a real
    pattern and the frozen schema keeps it expressible: ``AgenticContext`` on a
    classic rail is legal exactly when ``entry_mode='agent_token'``. Cards F1-01,
    F1-03 and F1-09 all name ``ecom`` for this reason — the agent is not always
    announcing itself.

    It also has a measurement consequence we depend on. A 100%-agentic attack
    scores ~0.92 on ``ag_agent_id_isnull`` before it has done anything, because
    the rail is 15% of the file. Splitting it means the probe measures the
    attack.

    Returns the boolean mask of rows that were moved.
    """
    moved = rng.random(len(rows)) < share_ecom
    if moved.any():
        rows.loc[moved, "channel"] = Channel.ECOM.value
        rows.loc[moved, "entry_mode"] = EntryMode.AGENT_TOKEN.value
    return moved


def novel_merchants_for(
    view: PopulationView,
    customer_ids: np.ndarray,
    rng: np.random.Generator,
    *,
    mccs: np.ndarray | None = None,
) -> np.ndarray:
    """One merchant per customer that the customer has never transacted with.

    Merchant novelty is a real signal on F1-01 (the agent was steered somewhere
    new), and computing it against the customer's actual history — rather than
    picking a random merchant and hoping — is what makes it true rather than
    likely.
    """
    frame = view.frame
    merchant_col = frame["merchant_id"].to_numpy()
    out = np.empty(len(customer_ids), dtype=object)
    for i, cid in enumerate(customer_ids):
        seen = set(merchant_col[view.rows_by_customer[str(cid)]])
        if mccs is not None:
            candidates = view.merchants_in_mcc(str(mccs[i]), popularity=(0.05, 0.95))
        else:
            candidates = view.merchants.index.to_numpy()
        # Ten attempts, then take whatever came up. A customer who has been to
        # most of a small category is not a reason to fail the run.
        pick = candidates[rng.integers(0, candidates.size)]
        for _ in range(10):
            if str(pick) not in seen:
                break
            pick = candidates[rng.integers(0, candidates.size)]
        out[i] = pick
    return out


# --------------------------------------------------------------------------- #
# Provenance and content
# --------------------------------------------------------------------------- #


def plant_injected_content(
    rows: pd.DataFrame,
    mask: np.ndarray,
    *,
    kinds: tuple[str, ...],
    rng: np.random.Generator,
    store: ContentStore | None = None,
    n_pages: tuple[int, int] = (1, 3),
) -> int:
    """Route the provenance chain through attacker content, and bind the text.

    For every masked row this **replaces** the chain's penultimate entries with
    URLs on :data:`ATTACKER_DOMAINS`, binds each of those URLs to a real
    adversarial artefact in the content store, and rewrites
    ``ag_ingested_content_ids`` to match.

    Three properties this has to have, and does:

    1. **The chain still ends at the merchant that was paid.** The trail is
       causal or it is decoration; the injected pages sit *between* the
       customer's habitual browsing and the checkout, which is where an indirect
       injection actually lands.
    2. **The content ids are computed exactly as the simulator computes them**
       (:func:`~mantis.foundry.llm.corpus.content_id_for_url`), so a defender
       joining the parquet to the corpus finds the text with no special case for
       attack rows.
    3. **Benign ids resolve too.** The store assigns any unbound id into the
       benign pool deterministically, so L3 cannot pass by checking whether an
       id resolves at all — it has to read the words.

    Returns the number of content artefacts planted.
    """
    store = store or load_content_store()
    mask = np.asarray(mask, dtype=bool)
    idx = np.flatnonzero(mask)
    if not idx.size:
        return 0

    chains = rows["ag_provenance_chain"].to_numpy().copy()
    contents = rows["ag_ingested_content_ids"].to_numpy().copy()
    event_ids = rows["event_id"].to_numpy()
    planted = 0

    for i in idx:
        chain = list(chains[i]) if chains[i] is not None else []
        if not chain:
            continue
        tail = chain[-1]  # the merchant page: the trail must still end there
        head = chain[:-1]

        # Length-preserving: the injected pages *replace* habitual browsing
        # rather than extending the trail. The first cut of this grew the chain
        # by one to three entries and the probe read ag_provenance_chain_len at
        # 0.96 -- the attack was detectable by counting URLs without looking at
        # one of them, which is precisely the cartoon the probe exists to catch.
        n = int(rng.integers(n_pages[0], n_pages[1] + 1))
        n = max(1, min(n, len(head)))
        injected: list[str] = []
        for k in range(n):
            domain = ATTACKER_DOMAINS[rng.integers(0, len(ATTACKER_DOMAINS))]
            page = _ATTACK_PAGES[rng.integers(0, len(_ATTACK_PAGES))]
            url = f"https://{domain}/{page}?id={_short_hash(f'{event_ids[i]}{k}', 8)}"
            kind = kinds[rng.integers(0, len(kinds))]
            artifact = store.pick(kind, int(rng.integers(0, 10_000)))
            store.bind(content_id_for_url(url), artifact.artifact_id)
            injected.append(url)
            planted += 1

        new_chain = [*head[: len(head) - n], *injected, tail]
        chains[i] = new_chain
        contents[i] = [content_id_for_url(url) for url in new_chain]

    rows["ag_provenance_chain"] = chains
    rows["ag_ingested_content_ids"] = contents
    return planted


# --------------------------------------------------------------------------- #
# Behavioural telemetry
# --------------------------------------------------------------------------- #

#: Cache of the background's own agentic telemetry, keyed by ``id(view.frame)``.
#: Sorting 30,000 values per injector call is wasteful and the frames are stable
#: for the life of a run.
_TELEMETRY_CACHE: dict[int, dict[str, np.ndarray]] = {}


def _telemetry(view: PopulationView) -> dict[str, np.ndarray]:
    key = id(view.frame)
    cached = _TELEMETRY_CACHE.get(key)
    if cached is not None:
        return cached
    agentic = view.frame[view.frame["channel"] == Channel.AGENTIC.value]
    cached = {
        column: np.sort(agentic[column].dropna().to_numpy(dtype=float))
        for column in (
            "ag_deliberation_latency_ms",
            "ag_cursor_entropy",
            "ag_dwell_time_ms",
            "ag_tool_call_count",
        )
    }
    _TELEMETRY_CACHE[key] = cached
    return cached


def resample_band(
    view: PopulationView,
    column: str,
    n: int,
    band: tuple[float, float],
    rng: np.random.Generator,
    *,
    jitter: float = 0.08,
) -> np.ndarray:
    """Resample the background's own values for ``column`` from a quantile band.

    The same device as ``PopulationView.draw_amounts`` and for the same reason:
    a forged value drawn from an attacker-chosen distribution is detectable on
    the column alone. Every number this returns is a real value some legitimate
    agentic session produced, nudged by a few percent.
    """
    values = _telemetry(view)[column]
    lo = int(band[0] * (values.size - 1))
    hi = max(lo + 1, int(band[1] * (values.size - 1)))
    picked = values[lo : hi + 1][rng.integers(0, hi - lo + 1, n)]
    return picked * np.exp(rng.normal(0.0, jitter, n))


def collapse_deliberation(
    view: PopulationView,
    rows: pd.DataFrame,
    mask: np.ndarray,
    rng: np.random.Generator,
    *,
    band: tuple[float, float] = (0.02, 0.26),
) -> None:
    """Shorten deliberation on masked rows, and raise the tool-call count.

    The signature of an agent acting on an injected instruction is that it stops
    reasoning: the decision was made for it, so the latency that should scale
    with the size of the purchase collapses toward the floor, while the *number*
    of tool calls goes up because it fetched extra pages on the way.

    The pair is what matters. ``deliberation_latency_z`` on its own is a real but
    weak signal — plenty of legitimate small purchases are fast. Fast *and*
    expensive *and* preceded by extra fetches is the shape, and no single column
    expresses it.
    """
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if not n:
        return
    latency = resample_band(view, "ag_deliberation_latency_ms", n, band, rng)
    rows.loc[mask, "ag_deliberation_latency_ms"] = np.maximum(120, latency.round()).astype(int)

    # Tool calls are resampled from the *upper band of the background's own*
    # distribution, not added to the cloned row's value. Adding pushed the count
    # past the legitimate maximum and the probe read it at 0.96 -- an attack
    # detectable by counting tool calls is not an attack, it is a tell.
    calls = resample_band(view, "ag_tool_call_count", n, (0.62, 0.97), rng, jitter=0.05)
    rows.loc[mask, "ag_tool_call_count"] = np.maximum(1, calls.round()).astype(int)
