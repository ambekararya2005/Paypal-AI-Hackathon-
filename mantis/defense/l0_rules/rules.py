"""L0 — deterministic protocol-integrity clauses.

What L0 is for
---------------
Every clause here answers a question with a **provable** answer: was this
authorisation inside the envelope the human signed? Not "does it look unusual" —
that is L1's job and L1 is a probability. L0 returns a named reason, and the
reason is either true of the message or it is not.

That makes L0 the only layer an issuer could deploy tomorrow. It needs no
training data, no labels and no history; it reads one authorisation and checks it
against the mandate it carries. It is also the only layer whose false positives
are arguments rather than errors: if L0 fires ``mandate_expired``, the mandate
*was* expired, and whether to decline on that is policy.

What L0 is deliberately not for
--------------------------------
Anything probabilistic. A clause that fires on "amount is in the top percentile"
would be a bad model wearing a rule's clothes: no threshold makes it true, and
its false positives are real errors rather than policy choices.

The clause that is declared and switched OFF, and why that matters
-------------------------------------------------------------------
``provenance_untrusted_domain`` — "the agent read a page on a domain that is not
on the reputation allow-list" — is implemented, measured, and **excluded from the
operative rule set**. Measured on the gate dataset it catches **100% of F1-01
and 100% of F1-03 at a 0.00% false-positive rate**, which is precisely why it
cannot be trusted:

* The foundry draws attacker URLs from a fixed pool of twelve hosts that appear
  **nowhere** in legitimate traffic. The clause is not detecting an attack, it is
  detecting a partition the generator created.
* Real deployments do not get that partition. 4,803 of the 4,827 domains in this
  file's legitimate provenance chains are per-merchant shop hosts seen once or
  twice, and in reality attacker infrastructure is *also* new. "Unseen domain" is
  separable here and is not separable on a real stream.
* Most importantly, it would make L3 redundant *and* let L3 post a fake number.
  The foundry's own comment on the attack page slugs says it outright: "the URL
  is not the tell, the content behind it is, which is precisely why L3 has to
  exist." A layer that reads the text has to earn its recall by reading the text.

This is the same class of artefact Day 3 caught twice — provenance *length* at
0.96 AUC, tool-call *count* at 0.96 — one level up: not how many URLs, but which
hosts. Keeping the clause implemented and printing its inflated number next to
the reason it is excluded is more useful than deleting it, because the next
person to have this idea can see it was had and measured.

What replaced it is a real invariant: :func:`provenance_merchant_mismatch`
checks that the trail **terminates at the merchant that was paid**. That is
protocol integrity rather than reputation, it is checkable on one message, and
the foundry preserves it deliberately on attack rows — so it correctly does not
fire on the CLEAN attacks either.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlparse

import numpy as np
import pandas as pd

__all__ = [
    "CLAUSES",
    "DECLARED_ONLY",
    "Clause",
    "L0Result",
    "evaluate",
    "trusted_domains",
]

#: Delegation hops above which a chain is rejected outright.
#:
#: Legitimate multi-agent orchestration reaches five (a shopping agent calling a
#: booking agent calling a checkout capability is three before anything unusual
#: has happened), and the reference's tail was widened to five on Day 3 for
#: exactly that reason. So the clause fires at **six**: deep enough that no
#: modelled legitimate chain reaches it, which is what keeps this a rule instead
#: of a threshold somebody tuned.
MAX_DELEGATION_DEPTH: Final[int] = 5

#: Merchant shop hosts follow this shape, so the paid merchant can be recovered
#: from the last URL in the chain.
_SHOP_HOST: Final[re.Pattern[str]] = re.compile(r"^shop\.(?P<merchant>[a-z0-9_-]+)\.test$")


@dataclass(frozen=True, slots=True)
class Clause:
    """One deterministic check. ``fn`` returns a boolean mask over the frame."""

    name: str
    reason: str
    fn: Callable[[pd.DataFrame], np.ndarray]
    #: Present in the rule set but excluded from the fired verdict. See the
    #: module docstring; the only member is the domain-reputation clause.
    declared_only: bool = False


def _agentic(frame: pd.DataFrame) -> np.ndarray:
    """Rows carrying an agentic block. Every clause below is scoped to these."""
    return frame["ag_agent_id"].notna().to_numpy()


def _present(values: np.ndarray) -> np.ndarray:
    """Mask of entries that carry a value at all: not ``None``, not ``NaN``.

    Written as ``v == v`` rather than ``pd.notna`` because these are object
    columns holding a mix of Python bools, numpy bools and ``None``, and only the
    NaN-is-never-equal-to-itself trick is safe across all three.
    """
    return np.array([v is not None and v == v for v in values], dtype=bool)


def _as_bool(series: pd.Series) -> np.ndarray:
    """Tri-state object column to a numpy bool, with ``None`` as False.

    ``bool(v)`` rather than ``v is True``. The identity check looks tighter and is
    a trap: these columns hold ``numpy.bool_`` in memory and Python ``bool`` after
    a parquet round-trip, and ``np.True_ is True`` is **False**. Written with
    ``is``, ``kya_unregistered`` fired on 100% of the CLEAN attacks when run
    against an in-memory frame and on 0% when run against the same data reloaded
    from disk -- a bug that a test reading the parquet would never have found.
    """
    values = series.to_numpy()
    return np.array(
        [bool(v) if (v is not None and v == v) else False for v in values], dtype=bool
    )


def _is_false(series: pd.Series) -> np.ndarray:
    """True only where the value is explicitly false; ``None`` is not a breach.

    An absent verification result means the check did not run, which is a
    different fact from the check failing, and L0 must not decline on it.
    """
    values = series.to_numpy()
    return _present(values) & ~_as_bool(series)


# --------------------------------------------------------------------------- #
# The clauses
# --------------------------------------------------------------------------- #


def cart_outside_intent_scope(frame: pd.DataFrame) -> np.ndarray:
    """The category paid for is not one the mandate's scope names.

    Skipped where the scope names no categories: an empty list means
    unconstrained, per the schema. Reading it as "nothing is permitted" would
    fire on every open-scope mandate and put L0's false-positive rate through
    the roof — the single most common way a rules layer gets switched off in
    production.
    """
    out = np.zeros(len(frame), dtype=bool)
    mccs = frame["mcc"].to_numpy()
    for i, scope in enumerate(frame["ag_scope_categories"].to_numpy()):
        if scope is None or not hasattr(scope, "__len__") or len(scope) == 0:
            continue
        out[i] = str(mccs[i]) not in {str(s) for s in scope}
    return out & _agentic(frame)


def merchant_outside_allow_list(frame: pd.DataFrame) -> np.ndarray:
    """The merchant paid is not on the mandate's allow-list. Empty list = open."""
    out = np.zeros(len(frame), dtype=bool)
    merchants = frame["merchant_id"].to_numpy()
    for i, allowed in enumerate(frame["ag_scope_allowed_merchants"].to_numpy()):
        if allowed is None or not hasattr(allowed, "__len__") or len(allowed) == 0:
            continue
        out[i] = str(merchants[i]) not in {str(a) for a in allowed}
    return out & _agentic(frame)


def amount_over_cap(frame: pd.DataFrame) -> np.ndarray:
    """The amount exceeds the per-transaction ceiling the human agreed."""
    ceiling = frame["ag_scope_max_amount"].to_numpy(dtype=float)
    amount = frame["amount"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        return (ceiling > 0) & (amount > ceiling) & _agentic(frame)


def mandate_expired(frame: pd.DataFrame) -> np.ndarray:
    """The mandate's TTL had elapsed before the authorisation was presented."""
    age = (frame["ts"] - frame["ag_mandate_issued_ts"]).dt.total_seconds().to_numpy(dtype=float)
    ttl = frame["ag_mandate_ttl_seconds"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        return (ttl > 0) & (age > ttl) & _agentic(frame)


def delegation_too_deep(frame: pd.DataFrame) -> np.ndarray:
    """More agent-to-sub-agent hops between the human and the payment than allowed."""
    depth = frame["ag_delegation_depth"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        return (depth > MAX_DELEGATION_DEPTH) & _agentic(frame)


def consent_signature_invalid(frame: pd.DataFrame) -> np.ndarray:
    """The consent signature failed cryptographic verification."""
    return _is_false(frame["ag_consent_sig_valid"]) & _agentic(frame)


def kya_unregistered(frame: pd.DataFrame) -> np.ndarray:
    """The agent is not in the Know-Your-Agent registry."""
    return ~_as_bool(frame["ag_kya_registered"]) & _agentic(frame)


def mandate_missing(frame: pd.DataFrame) -> np.ndarray:
    """An agent-mediated payment presented with no mandate at all."""
    kind = frame["ag_mandate_type"].to_numpy()
    return ((kind == "none") | pd.isna(kind)) & _agentic(frame)


def provenance_merchant_mismatch(frame: pd.DataFrame) -> np.ndarray:
    """The trail does not end at the merchant that was paid.

    The clause that replaced domain reputation. A causal provenance chain
    terminates at the merchant's own page — the agent read the listing, then
    bought the thing. A chain ending somewhere else describes an agent that
    decided to pay one merchant while looking at another, which is a protocol
    integrity failure rather than a judgement about anybody's reputation.

    Rows whose last URL is not a recognisable shop host are skipped rather than
    flagged: the clause can only assert a mismatch it can actually read.
    """
    out = np.zeros(len(frame), dtype=bool)
    merchants = frame["merchant_id"].to_numpy()
    for i, chain in enumerate(frame["ag_provenance_chain"].to_numpy()):
        if chain is None or not hasattr(chain, "__len__") or len(chain) == 0:
            continue
        match = _SHOP_HOST.match(urlparse(str(chain[-1])).netloc)
        if match is None:
            continue
        out[i] = not str(merchants[i]).endswith(match.group("merchant"))
    return out & _agentic(frame)


def trusted_domains(train: pd.DataFrame, min_events: int = 20) -> frozenset[str]:
    """Reputation allow-list fitted from legitimate training provenance chains.

    Only used by the declared-only clause. See the module docstring for why the
    number it produces cannot be believed.
    """
    counts: dict[str, int] = {}
    legit = train[~train["is_fraud"].to_numpy()] if "is_fraud" in train else train
    for chain in legit["ag_provenance_chain"].to_numpy():
        if chain is None or not hasattr(chain, "__len__"):
            continue
        for url in chain:
            host = urlparse(str(url)).netloc
            counts[host] = counts.get(host, 0) + 1
    return frozenset(host for host, n in counts.items() if n >= min_events)


def make_untrusted_domain_clause(allow: frozenset[str]) -> Clause:
    """Build the declared-only reputation clause against a fitted allow-list."""

    def _fn(frame: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(frame), dtype=bool)
        for i, chain in enumerate(frame["ag_provenance_chain"].to_numpy()):
            if chain is None or not hasattr(chain, "__len__"):
                continue
            for url in chain:
                host = urlparse(str(url)).netloc
                # A merchant's own shop host is trusted for its own purchase,
                # without which every first-time merchant fires the clause.
                if host in allow or _SHOP_HOST.match(host):
                    continue
                out[i] = True
                break
        return out & _agentic(frame)

    return Clause(
        name="provenance_untrusted_domain",
        reason="agent read a page on a domain outside the reputation allow-list",
        fn=_fn,
        declared_only=True,
    )


#: The operative rule set. Order is the order the table prints in.
CLAUSES: Final[tuple[Clause, ...]] = (
    Clause("scope_mcc", "cart category outside the signed intent scope", cart_outside_intent_scope),
    Clause(
        "scope_merchant", "merchant outside the mandate allow-list", merchant_outside_allow_list
    ),
    Clause("amount_over_cap", "amount exceeds the mandate ceiling", amount_over_cap),
    Clause("mandate_expired", "mandate TTL elapsed before presentation", mandate_expired),
    Clause("delegation_depth", f"delegation deeper than {MAX_DELEGATION_DEPTH} hops",
           delegation_too_deep),
    Clause("consent_invalid", "consent signature failed verification", consent_signature_invalid),
    Clause("kya_unregistered", "agent absent from the KYA registry", kya_unregistered),
    Clause("mandate_missing", "agent-mediated payment with no mandate", mandate_missing),
    Clause("provenance_mismatch", "trail does not end at the merchant paid",
           provenance_merchant_mismatch),
)

#: Names of clauses that are measured and printed but never fire a verdict.
DECLARED_ONLY: Final[tuple[str, ...]] = ("provenance_untrusted_domain",)


@dataclass(slots=True)
class L0Result:
    """Per-clause masks plus the fired verdict."""

    masks: dict[str, np.ndarray]
    #: True where at least one **operative** clause fired.
    fired: np.ndarray
    #: First operative clause name per row, or ``""``.
    reason: np.ndarray

    def clause_names(self) -> list[str]:
        return list(self.masks)


def evaluate(frame: pd.DataFrame, clauses: tuple[Clause, ...] = CLAUSES) -> L0Result:
    """Run every clause over ``frame``.

    Declared-only clauses are evaluated and returned in ``masks`` but excluded
    from ``fired`` and ``reason``, so a caller cannot accidentally take credit
    for one.
    """
    masks: dict[str, np.ndarray] = {}
    fired = np.zeros(len(frame), dtype=bool)
    reason = np.full(len(frame), "", dtype=object)

    for clause in clauses:
        mask = np.asarray(clause.fn(frame), dtype=bool)
        masks[clause.name] = mask
        if clause.declared_only:
            continue
        fresh = mask & ~fired
        reason[fresh] = clause.name
        fired |= mask
    return L0Result(masks=masks, fired=fired, reason=reason)
