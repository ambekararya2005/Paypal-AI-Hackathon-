"""Score ONE authorisation, now. The online counterpart of ``python -m mantis.defense``.

    python -m mantis.defense.live          # load once, score three demo events

Everything else in ``mantis/defense`` scores a million rows offline. This module
scores a single :class:`~mantis.core.events.TxEvent` inside a request, which
changes three things and nothing else:

1. **Load once.** :class:`LiveScorer` fits L3 (seconds, from the committed corpus)
   and loads an L1 artefact if one exists, in its constructor. A request never
   fits anything.
2. **Plain lookups on the hot path.** L3 is a dict of artefact scores plus one
   TF-IDF transform when the agent forwarded the page text; L0 runs the *same*
   clause functions the measured tables came from, on a one-row frame, so a
   clause cannot mean one thing offline and another here.
3. **Honest about what ran.** ``layers_ran`` lists the layers that actually
   produced a score for this event. Without ``data/models/`` the gate runs
   **L0 + L3** and says so; it never pads the list.

How the decision is made
--------------------------
The policy is :func:`mantis.defense.policy.decide.decide`, unchanged:

* an L0 clause firing — or a gate-level integrity failure such as a receipt hash
  that does not match the order PayPal holds — is a **protocol violation** and
  declines outright;
* otherwise L3's score gets **one** boundary: above every benign page in the
  corpus. With an L1 artefact, the fused score is compared against boundaries
  fitted at FPR budgets on legitimate traffic, and the decision is the **more
  severe** of the two: the supervised layer can escalate, never veto the
  label-free ones (measured reason in :meth:`LiveScorer.score`). L3 is never
  allowed to decline on its own, because
  its *threshold* does not transfer to text it did not come from (CLAUDE.md: 90%
  false positives on hand-authored controls) even though its *ordering* does. So
  a high L3 score is a ``CHALLENGE``, which ``decide`` escalates to ``REVIEW``
  when no human is present — i.e. the order is **held for human consent**.

Reasons
---------
Every decision carries its top three reasons, most severe first: violated clauses
with the numbers that violated them, then the page that drove L3 with the phrases
that drove *that*, then L1. An approval gets reasons too — which checks passed —
because "approved, no reason" is not an audit trail.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Protocol

import numpy as np
import pandas as pd

from mantis.core.events import TxEvent, flatten
from mantis.core.paths import DATA_DIR
from mantis.defense.l0_rules.rules import CLAUSES
from mantis.defense.l3_text.model import L3Model
from mantis.defense.policy.decide import Decision, PolicyThresholds, decide, escalate

__all__ = ["LIVE_MODEL_DIR", "LiveScore", "LiveScorer", "Reason"]

LIVE_MODEL_DIR: Final[Path] = DATA_DIR / "models"

#: Reasons returned per decision.
TOP_REASONS: Final[int] = 3


@dataclass(frozen=True, slots=True)
class Reason:
    """One line of an explanation. ``layer`` is L0 / L1 / L3 / gate."""

    layer: str
    code: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"layer": self.layer, "code": self.code, "detail": self.detail}


@dataclass(slots=True)
class LiveScore:
    decision: Decision
    reasons: list[Reason]
    layers_ran: list[str]
    scores: dict[str, float]
    latency_ms: float
    #: Every reason, before truncation to the top three. For the audit log.
    all_reasons: list[Reason] = field(default_factory=list)


class OnlineLayer(Protocol):
    """An optional supervised layer (L1 + fusion) loaded from ``data/models/``."""

    name: str
    thresholds: PolicyThresholds

    def score_one(self, event: TxEvent, *, l3_score: float) -> tuple[float, list[Reason]]: ...


def _one_row(event: TxEvent) -> pd.DataFrame:
    """The event as the one-row frame the L0 clauses were written against.

    Timestamps are coerced explicitly: a tz-aware ``datetime`` next to ``None``
    lands in an object column, and ``mandate_expired`` needs ``.dt``.
    """
    frame = pd.DataFrame([flatten(event)])
    for column in ("ts", "ag_mandate_issued_ts"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def _l0_detail(name: str, event: TxEvent, fallback: str) -> str:
    """The clause's reason, with the numbers that tripped it."""
    ag = event.agentic
    scope = ag.mandate_scope if ag is not None else None
    if name == "amount_over_cap" and scope is not None and scope.max_amount is not None:
        return (f"amount {event.amount:.2f} {event.currency} exceeds the mandate ceiling "
                f"{scope.max_amount:.2f}")
    if name == "scope_mcc" and scope is not None:
        return f"MCC {event.mcc} is outside the signed categories {scope.categories}"
    if name == "scope_merchant" and scope is not None:
        return (f"merchant {event.merchant_id} is not on the mandate allow-list "
                f"{scope.allowed_merchants}")
    if name == "mandate_expired" and ag is not None and ag.expires_at() is not None:
        return f"mandate expired at {ag.expires_at():%Y-%m-%d %H:%M} UTC"
    return fallback


class LiveScorer:
    """Load once, score one event at a time."""

    def __init__(self, *, seed: int = 1337, model_dir: Path = LIVE_MODEL_DIR,
                 l1: OnlineLayer | None = None, use_l1: bool = True) -> None:
        self.seed = seed
        self.l3 = L3Model(seed=seed).fit()
        store = self.l3._resolved_store()
        benign = [s for a, s in self.l3.artifact_scores.items()
                  if not store.artifacts[a].is_injected]
        #: Hold line: above every benign page in the corpus. Derived, not tuned.
        self.l3_hold_at = float(max(benign))
        self._terms = np.asarray(self.l3.vectoriser.get_feature_names_out())
        self._coef = np.asarray(self.l3.classifier.coef_).ravel()
        #: ``use_l1=False`` pins the clean-clone behaviour (L0 + L3) whatever is
        #: in the gitignored ``data/models/``; the tests use it for that reason.
        self.l1 = l1 if l1 is not None else (self._load_l1(model_dir) if use_l1 else None)

    @staticmethod
    def _load_l1(model_dir: Path) -> OnlineLayer | None:
        """The persisted L1 + fusion bundle, if ``scripts/train_live_model.py`` ran."""
        try:
            from mantis.defense.live_l1 import load_online_l1
        except ImportError:
            return None
        return load_online_l1(model_dir)

    # -- L3 ------------------------------------------------------------------ #

    def _page_scores(self, event: TxEvent,
                     texts: dict[str, str] | None) -> list[tuple[str, float, str, str | None]]:
        """``(url, P(injected), source, artefact id)`` for every page read.

        Text the agent forwarded is classified directly; otherwise the URL's
        content id resolves through the committed store, exactly as offline.
        """
        ag = event.agentic
        if ag is None:
            return []
        store = self.l3._resolved_store()
        out: list[tuple[str, float, str, str | None]] = []
        for url, content_id in zip(ag.provenance_chain, ag.ingested_content_ids, strict=False):
            text = (texts or {}).get(url)
            if text:
                matrix = self.l3.vectoriser.transform([text])
                p = float(self.l3.classifier.predict_proba(matrix)[0, 1])
                out.append((url, p, "text", None))
                continue
            artifact = store.resolve(content_id)
            if artifact is not None:
                out.append((url, self.l3.artifact_scores.get(artifact.artifact_id, 0.0),
                            "store", artifact.artifact_id))
        return out

    def _phrases(self, text: str, top: int = 3) -> list[str]:
        """The terms in this page pushing it toward "injected", by contribution."""
        row = self.l3.vectoriser.transform([text])
        contrib = row.multiply(self._coef).tocsr()
        idx, vals = contrib.indices, contrib.data
        order = np.argsort(vals)[::-1]
        return [str(self._terms[idx[i]]) for i in order[:top] if vals[i] > 0]

    # -- scoring ---------------------------------------------------------------- #

    def score(self, event: TxEvent, *, page_texts: dict[str, str] | None = None,
              gate_violations: list[Reason] | None = None) -> LiveScore:
        """Score one event and map it to a decision. See the module docstring."""
        started = time.perf_counter()
        violations: list[Reason] = list(gate_violations or [])
        layers: list[str] = []
        scores: dict[str, float] = {}

        # L0 -------------------------------------------------------------------
        frame = _one_row(event)
        for clause in CLAUSES:
            if clause.declared_only:
                continue
            if bool(np.asarray(clause.fn(frame), dtype=bool)[0]):
                violations.append(
                    Reason("L0", clause.name, _l0_detail(clause.name, event, clause.reason))
                )
        layers.append("L0")
        scores["l0_violations"] = float(sum(r.layer == "L0" for r in violations))

        # L3 -------------------------------------------------------------------
        pages = self._page_scores(event, page_texts)
        l3_reasons: list[Reason] = []
        l3_score = float("nan")
        if pages:
            ranked = sorted(pages, key=lambda p: p[1], reverse=True)
            runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
            l3_score = ranked[0][1] + 0.02 * runner_up  # the offline tiebreak, verbatim
            layers.append("L3")
            scores["l3"] = l3_score
            url, p, _source, artifact_id = ranked[0]
            if p > self.l3_hold_at:
                text = (page_texts or {}).get(url)
                if text is None and artifact_id is not None:
                    text = self.l3._resolved_store().artifacts[artifact_id].text
                phrases = self._phrases(text) if text else []
                said = f"; phrases: {', '.join(repr(t) for t in phrases)}" if phrases else ""
                l3_reasons.append(Reason(
                    "L3", "injected_content",
                    f"agent read {url} scoring P(injected)={p:.2f} "
                    f"(> {self.l3_hold_at:.2f}, every benign page){said}",
                ))
            else:
                l3_reasons.append(Reason(
                    "L3", "provenance_clean",
                    f"worst of {len(pages)} pages read scores {p:.2f}, "
                    f"under the hold line {self.l3_hold_at:.2f}",
                ))

        ag = event.agentic
        human = np.array([bool(ag.human_present) if ag else False])
        l0_fired = np.array([bool(violations)])

        # L0 + L3: the label-free decision. One L3 boundary, and it can only
        # hold, never decline (its threshold does not transfer; its ordering does).
        l3_only = PolicyThresholds(challenge=self.l3_hold_at, review=np.inf, decline=np.inf)
        # Decision(...) because the object array can hand back the plain string.
        decision = Decision(decide(
            np.array([l3_score if np.isfinite(l3_score) else 0.0]),
            l3_only, l0_violation=l0_fired, human_present=human,
        )[0])

        # L1 + fusion (optional): may ESCALATE that decision, never lower it.
        # Measured on scenario C: the fused weights, learned on the synthetic pool
        # where L3's percentile carries a negative coefficient, scored an injected
        # page at 0.061 against a 0.862 review line and would have captured it.
        # Supervised layers do not get a veto over the label-free ones -- the same
        # rule that lets L0 outrank every score.
        l1_reasons: list[Reason] = []
        if self.l1 is not None:
            fused, l1_reasons = self.l1.score_one(event, l3_score=l3_score)
            layers.append(self.l1.name)
            scores["fused"] = fused
            fused_decision = Decision(decide(
                np.array([fused]), self.l1.thresholds,
                l0_violation=l0_fired, human_present=human,
            )[0])
            decision = escalate(decision, fused_decision)

        ordered = violations + l3_reasons + l1_reasons
        if not violations:
            ordered.insert(0, Reason("L0", "protocol_ok",
                                     f"all {sum(not c.declared_only for c in CLAUSES)} "
                                     "mandate clauses passed"))
            if decision is not Decision.APPROVE:
                ordered = [r for r in ordered if r.code != "protocol_ok"] + [ordered[0]]
        if self.l1 is None:
            ordered.append(Reason("L1", "not_loaded",
                                  "no data/models/ artefact; scored on L0 + L3 only"))
        return LiveScore(
            decision=decision,
            reasons=ordered[:TOP_REASONS],
            layers_ran=layers,
            scores=scores,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            all_reasons=ordered,
        )


def main() -> None:
    from datetime import UTC, datetime, timedelta

    from mantis.core.events import (
        AgenticContext,
        Channel,
        EntryMode,
        MandateScope,
        MandateType,
    )
    from mantis.foundry.llm.corpus import content_id_for_url

    t0 = time.perf_counter()
    scorer = LiveScorer()
    print(f"LiveScorer loaded in {time.perf_counter() - t0:.1f}s; "
          f"L3 hold line {scorer.l3_hold_at:.3f}; L1 {'loaded' if scorer.l1 else 'absent'}")

    now = datetime.now(UTC)
    store = scorer.l3._resolved_store()

    def event(amount: float, mcc: str, urls: list[str]) -> TxEvent:
        return TxEvent(
            event_id=f"demo-{amount}-{mcc}", ts=now, amount=amount, currency="USD", mcc=mcc,
            channel=Channel.AGENTIC, entry_mode=EntryMode.AGENT_TOKEN, customer_id="c",
            card_bin="411111", merchant_id="m-freshmart", merchant_country="US",
            agentic=AgenticContext(
                agent_id="agent-shopper-01", agent_platform="demo", kya_registered=True,
                mandate_type=MandateType.INTENT, mandate_id="m", mandate_hash="h",
                mandate_issued_ts=now - timedelta(minutes=5), mandate_ttl_seconds=86_400,
                mandate_scope=MandateScope(categories=["5411"], max_amount=100.0),
                consent_sig_valid=True, provenance_chain=urls,
                ingested_content_ids=[content_id_for_url(u) for u in urls],
            ),
        )

    clean = ["https://recipes.test/week", "https://shop.freshmart.test/basket"]
    bad = ["https://deals.test/blog", "https://shop.freshmart.test/basket"]
    texts = {bad[0]: store.artifacts["injected_page-023"].text}
    for label, ev, pages in (("in scope", event(32.0, "5411", clean), None),
                             ("gift card", event(400.0, "6540", clean), None),
                             ("injected", event(32.0, "5411", bad), texts)):
        result = scorer.score(ev, page_texts=pages)
        print(f"\n  {label:<10} -> {result.decision.value.upper():<9} "
              f"layers={'+'.join(result.layers_ran)}  {result.latency_ms:.1f} ms")
        for r in result.reasons:
            print(f"     [{r.layer}] {r.code}: {r.detail}")


if __name__ == "__main__":
    main()
