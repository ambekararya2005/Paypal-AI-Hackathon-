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
from typing import TYPE_CHECKING, Final, Protocol

import numpy as np
import pandas as pd

from mantis.core.events import TxEvent, flatten
from mantis.core.paths import DATA_DIR
from mantis.defense.l0_rules.rules import CLAUSES
from mantis.defense.l3_text.model import L3Model
from mantis.defense.policy.decide import Decision, PolicyThresholds, decide, escalate

if TYPE_CHECKING:
    from mantis.defense.l3_judge import InjectionJudge, JudgeVerdict

__all__ = ["LIVE_MODEL_DIR", "LiveScore", "LiveScorer", "PageEval", "Reason"]

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
class PageEval:
    """One page the agent read, as the text layers saw it."""

    url: str
    text: str | None
    #: Share of the page's words in L3's vocabulary.
    coverage: float
    #: L3's P(injected), or ``None`` when L3 abstained (coverage below its floor).
    l3: float | None
    artifact_id: str | None
    verdict: JudgeVerdict | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "url": self.url,
            "coverage": round(self.coverage, 3),
            "l3": None if self.l3 is None else round(self.l3, 3),
            "l3_abstained": self.l3 is None,
            "artifact_id": self.artifact_id,
            "judge": None if self.verdict is None else self.verdict.as_dict(),
            "text": (self.text or "")[:4000],
        }


@dataclass(slots=True)
class LiveScore:
    decision: Decision
    reasons: list[Reason]
    layers_ran: list[str]
    scores: dict[str, float]
    #: MANTIS scoring, excluding the judge. The judge is reported separately.
    latency_ms: float
    #: Every reason, before truncation to the top three. For the audit log.
    all_reasons: list[Reason] = field(default_factory=list)
    #: Wall-clock of the judge phase (parallel calls; replayed from cache in mock).
    judge_ms: float = 0.0
    #: Per page: coverage, L3 score or abstention, judge verdict, text.
    evidence: list[dict[str, object]] = field(default_factory=list)


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
                 l1: OnlineLayer | None = None, use_l1: bool = True,
                 judge: InjectionJudge | None = None) -> None:
        self.seed = seed
        self.judge = judge
        self.l3 = L3Model(seed=seed).fit()
        store = self.l3._resolved_store()
        self._analyzer = self.l3.vectoriser.build_analyzer()
        self._vocab = self.l3.vectoriser.vocabulary_
        #: L3's domain floor: the least familiar text it was validated on (its
        #: held-out phrasings). Below this it abstains. Derived, not tuned.
        self.l3_min_coverage = float(min(
            self.coverage(store.artifacts[a].text) for a in self.l3.held_out
        ))
        benign = [s for a, s in self.l3.artifact_scores.items()
                  if not store.artifacts[a].is_injected]
        #: Hold line: above every benign page in the corpus. Derived, not tuned.
        self.l3_hold_at = float(max(benign))
        #: Judge line: the benign 99th percentile. Pages above it get a second
        #: opinion from the LLM judge; pages below it never cost an LLM call.
        self.l3_judge_at = float(np.quantile(benign, 0.99))
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

    def coverage(self, text: str) -> float:
        """Share of the text's words that are in L3's vocabulary (unigrams)."""
        words = [w for w in self._analyzer(text) if " " not in w]
        return sum(w in self._vocab for w in words) / max(1, len(words))

    def page_score(self, text: str) -> float:
        """P(injected) for one page of text. One TF-IDF transform, ~1.5 ms."""
        matrix = self.l3.vectoriser.transform([text])
        return float(self.l3.classifier.predict_proba(matrix)[0, 1])

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
        violations: list[Reason] = []
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
        # Gate-level live clauses (receipt, amount, budget) after the nine L0
        # clauses: protocol violations lead the explanation.
        violations.extend(gate_violations or [])
        layers.append("L0")
        scores["l0_violations"] = float(sum(r.layer == "L0" for r in violations))

        # L3 + L3-judge ----------------------------------------------------------
        pages = self._evaluate_pages(event, page_texts)
        in_domain = [p for p in pages if p.l3 is not None]
        l3_score = float("nan")
        if in_domain:
            ranked = sorted((p.l3 for p in in_domain), reverse=True)  # type: ignore[misc]
            l3_score = ranked[0] + 0.02 * (ranked[1] if len(ranked) > 1 else 0.0)
            layers.append("L3")
            scores["l3"] = l3_score
        scores["l3_abstained_pages"] = float(sum(p.l3 is None for p in pages))

        judge_started = time.perf_counter()
        self._run_judge(pages)
        judge_wall_ms = (time.perf_counter() - judge_started) * 1000.0
        judged = [p for p in pages if p.verdict is not None]
        # Reported judge latency: the wall clock when a live call was made; on a
        # cache replay, the recorded latency of the slowest call (they run in
        # parallel), so a replayed demo still shows what the judge costs live.
        recorded = [p.verdict.latency_ms for p in judged  # type: ignore[union-attr]
                    if p.verdict.source == "cache"]  # type: ignore[union-attr]
        live_judge = any(p.verdict.source == "live" for p in judged)  # type: ignore[union-attr]
        judge_ms = judge_wall_ms if live_judge else max(recorded, default=judge_wall_ms)
        if any(p.verdict.source in ("cache", "live") for p in judged):  # type: ignore[union-attr]
            layers.append("L3-judge")
        scores["judge_pages"] = float(len(judged))

        l3_fired = {p.url for p in in_domain if p.l3 > self.l3_hold_at}  # type: ignore[operator]
        judge_fired = {p.url for p in judged if p.verdict.injected}  # type: ignore[union-attr]
        unvetted = [p for p in pages if p.l3 is None
                    and (p.verdict is None or p.verdict.injected is None)]
        agree = l3_fired & judge_fired
        text_reasons = self._text_reasons(pages, l3_fired, judge_fired, agree, unvetted)

        ag = event.agentic
        human = np.array([bool(ag.human_present) if ag else False])
        l0_fired = np.array([bool(violations)])

        # The label-free decision (PayPal Day 2 policy table):
        #   L0 or a gate clause fired           -> DECLINE
        #   L3 and the judge agree on a page    -> DECLINE
        #   exactly one of them fired           -> REVIEW (hold for human consent)
        #   a page nobody could vouch for       -> REVIEW (fail closed)
        #   neither                             -> APPROVE; L1 may only tighten
        # Without a judge configured (Day 1), L3 alone holds and never declines.
        if violations or agree:
            decision = Decision.DECLINE
        elif l3_fired or judge_fired or unvetted:
            decision = Decision.REVIEW
        else:
            decision = Decision.APPROVE

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

        ordered = violations + text_reasons + l1_reasons
        if not violations:
            ok = Reason("L0", "protocol_ok",
                        f"all {sum(not c.declared_only for c in CLAUSES)} mandate clauses passed")
            ordered = [ok, *ordered] if decision is Decision.APPROVE else [*ordered, ok]
        if self.l1 is None:
            ordered.append(Reason("L1", "not_loaded",
                                  "no data/models/ artefact; scored without L1"))
        total_ms = (time.perf_counter() - started) * 1000.0
        return LiveScore(
            decision=decision,
            reasons=ordered[:TOP_REASONS],
            layers_ran=layers,
            scores=scores,
            latency_ms=total_ms - judge_wall_ms,
            all_reasons=ordered,
            judge_ms=judge_ms,
            evidence=[{**p.as_dict(), "l3_fired": p.url in l3_fired} for p in pages],
        )

    # -- the text layers, per page ------------------------------------------------ #

    def _evaluate_pages(self, event: TxEvent,
                        texts: dict[str, str] | None) -> list[PageEval]:
        """Every page the agent read, with L3's score or L3's abstention.

        Forwarded text is scored directly — unless its vocabulary coverage is below
        the floor of L3's own held-out validation set, in which case L3 abstains
        (``l3=None``). A URL with no forwarded text resolves through the committed
        content store, as offline, and is in-domain by construction.
        """
        ag = event.agentic
        if ag is None:
            return []
        store = self.l3._resolved_store()
        out: list[PageEval] = []
        for url, content_id in zip(ag.provenance_chain, ag.ingested_content_ids, strict=False):
            text = (texts or {}).get(url)
            if text:
                coverage = self.coverage(text)
                l3 = self.page_score(text) if coverage >= self.l3_min_coverage else None
                out.append(PageEval(url, text, coverage, l3, None))
                continue
            artifact = store.resolve(content_id)
            if artifact is not None:
                out.append(PageEval(url, artifact.text, 1.0,
                                    self.l3.artifact_scores.get(artifact.artifact_id, 0.0),
                                    artifact.artifact_id))
        return out

    def _run_judge(self, pages: list[PageEval]) -> None:
        """Ask the judge about abstained pages and pages above the judge line."""
        if self.judge is None:
            return
        todo = [p for p in pages
                if p.text and (p.l3 is None or p.l3 > self.l3_judge_at)]
        if not todo:
            return
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(4, len(todo))) as pool:
            verdicts = list(pool.map(lambda p: self.judge.judge(p.url, p.text), todo))  # type: ignore[union-attr]
        for page, verdict in zip(todo, verdicts, strict=True):
            page.verdict = verdict

    def _text_reasons(self, pages: list[PageEval], l3_fired: set[str], judge_fired: set[str],
                      agree: set[str], unvetted: list[PageEval]) -> list[Reason]:
        out: list[Reason] = []
        by_url = {p.url: p for p in pages}
        for url in [*agree, *(judge_fired - agree)]:
            v = by_url[url].verdict
            assert v is not None
            out.append(Reason(
                "L3-judge", "injection_confirmed" if url in agree else "injection_flagged",
                f'agent read {url}: "{v.quoted_span[:180]}" -> wants the agent to '
                f"{v.target_action or 'act outside its task'}"
                + (" (L3 agrees)" if url in agree else " (L3 abstained or below its line)"),
            ))
        for url in l3_fired - agree:
            page = by_url[url]
            phrases = self._phrases(page.text) if page.text else []
            said = f"; phrases: {', '.join(repr(t) for t in phrases)}" if phrases else ""
            out.append(Reason(
                "L3", "injected_content",
                f"agent read {url} scoring P(injected)={page.l3:.2f} "
                f"(> {self.l3_hold_at:.2f}, every benign page){said}",
            ))
        for page in unvetted:
            why = page.verdict.rationale if page.verdict is not None else "no judge configured"
            out.append(Reason("L3-judge", "unvetted_text",
                              f"{page.url}: outside L3's vocabulary "
                              f"(coverage {page.coverage:.2f}) and no judge verdict ({why})"))
        if not out and pages:
            judged = sum(p.verdict is not None for p in pages)
            abstained = sum(p.l3 is None for p in pages)
            worst = max((p.l3 for p in pages if p.l3 is not None), default=None)
            parts = [f"{len(pages)} pages read"]
            if worst is not None:
                parts.append(f"L3 worst {worst:.2f} < hold line {self.l3_hold_at:.2f}")
            if abstained:
                parts.append(f"L3 abstained on {abstained} (vocabulary coverage "
                             f"< {self.l3_min_coverage:.2f})")
            if judged:
                parts.append(f"judge cleared {judged}")
            out.append(Reason("L3", "provenance_clean", "; ".join(parts)))
        return out


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
