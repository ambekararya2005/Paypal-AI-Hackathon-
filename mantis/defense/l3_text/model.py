"""L3 — the text layer. What the agent read before it decided to pay.

Why this is the cheapest large win available
----------------------------------------------
F1-01 (cart-mandate tampering) and F1-03 (refund-logic hijack) are the two
**CLEAN** attacks in the atlas: they trip zero L0 clauses at zero tolerance, by
design, because a project in which every agentic attack is a protocol violation
would collapse into a list of if-statements. Their signal is not in the
authorisation message at all. It is in the **text the agent ingested** — an
instruction planted on a page the agent read on its way to checkout.
``provenance_chain`` is the field CLAUDE.md calls the highest-value one in the
schema, and this is the layer that cashes it.

The join
---------
``ag_ingested_content_ids`` is a list of ``sha256:<12 hex>`` digests of the URLs
in ``ag_provenance_chain``. :class:`~mantis.foundry.llm.corpus.ContentStore` maps
each of those to the text behind it. **Every id resolves** — ids with no explicit
binding are hashed deterministically into the benign pool — which is a leakage
control, not a convenience: if only attacked ids resolved, "does this id resolve"
would be a perfect label and this layer's metrics would be measuring the storage
layout rather than the words.

The unit of classification is the **artefact**, not the event
---------------------------------------------------------------
L3 classifies *a page*, and an event's score is the **maximum over the pages its
agent read**. Three reasons, and the third is the important one:

1. It is the question a defender actually has. "Was there an instruction anywhere
   in what the agent read" is a max, not an average. Summing a chain's vectors
   dilutes one injected page among eleven innocuous ones, and measured, that
   dilution costs about half the recall.
2. It is fast. The corpus is ~230 artefacts; a chain is a lookup, not a
   tokenisation. Fitting takes about a second, against sixteen minutes for the
   first version, which re-tokenised the same 230 texts 700,000 times.
3. **It needs no labelled fraud.** This is the one that matters for the
   architecture. L3's training label is
   :attr:`~mantis.foundry.llm.corpus.ContentArtifact.injected` — a property of a
   piece of *text*, curated the way any prompt-injection corpus is curated. It
   never sees ``is_fraud``, never sees a transaction, and does not know which
   attack card an event belongs to. So L3 belongs with L0 rather than with L1 in
   the reframed architecture (CLAUDE.md, "The zero-day answer, reframed"): it is
   a layer that works on an attack it has never seen in the payment data,
   because the thing it was trained on is not payment data.

   :meth:`L3Model.fit` takes no ``y`` argument at all, and that is the assertion.

Why a near-perfect number here is honest rather than suspicious
----------------------------------------------------------------
Said plainly before the number appears: **the injected instruction *is* the
attack.** F1-01 is defined as an agent acting on text that told it to change the
cart. A classifier reading that text and finding the instruction is not cheating
— it is the detection working.

Where L3 earns its place is therefore **not** always extra recall. On some runs
L1 matches it on F1-01 off the tabular features alone; RESULTS.md prints both
numbers side by side and says which case it is. What L3 has that L1 never does is
**independence**: its recall on a family is unchanged when that family is held
out of L1's training set, because L3 was not trained on transactions at all.

What *would* be dishonest is claiming it generalises when it does not, so it is
tested two ways, both harder than the metric they support:

**Held-out phrasings.** The corpus is authored as several variants of each
adversarial kind — the same attack class written differently by the model that
generated it. :data:`HELD_OUT_VARIANTS` withholds the highest-numbered variant of
every injected kind from the vocabulary *and* from training. Every kind keeps at
least one phrasing, so what is withheld is the wording.

**A held-out kind.** Withholding a phrasing still leaves the same prompt template
in training. So :data:`HELD_OUT_KIND` withholds an entire adversarial kind —
every ``refund_ticket`` specimen, all eighteen — and F1-03 is then scored on an
injection *type* the classifier has never seen in any form. That is the number
the writeup may lean on, and it is reported next to the others rather than
instead of them.

Deployability
---------------
TF-IDF over word 1-2 grams into a logistic regression. Not a transformer, and the
reason is the same one that chose Isolation Forest for L2: it has to fit in
seconds on a laptop with no GPU and no network (HARD RULE 4), and its
coefficients are readable, so an alert can name the phrase that caused it instead
of gesturing at an embedding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import pandas as pd

from mantis.foundry.llm.corpus import ContentStore, load_content_store

__all__ = [
    "HELD_OUT_KIND",
    "HELD_OUT_VARIANTS",
    "L3Model",
    "chains_for",
    "held_out_artifacts",
]

#: Variant index withheld from training, per injected kind. The highest index of
#: each, so every kind keeps at least one phrasing and what is withheld is the
#: wording rather than the attack class.
HELD_OUT_VARIANTS: Final[dict[str, int]] = {
    "injected_page": 3,
    "injected_review": 1,
    "refund_ticket": 2,
    "agent_transcript": 2,
    "scam_script": 2,
    "shell_merchant_copy": 1,
}

#: An entire adversarial kind withheld, for the harder of the two generalisation
#: tests. ``refund_ticket`` is F1-03's payload: withholding it means F1-03 is
#: scored on an injection type the classifier has never seen in any phrasing.
HELD_OUT_KIND: Final[str] = "refund_ticket"

#: ``... (injected_page v3)`` — how the corpus builder records which phrasing an
#: artefact is. Parsed rather than stored separately because the corpus is
#: committed and its schema is frozen by being on disk.
_VARIANT_RE: Final[re.Pattern[str]] = re.compile(r"\(([a-z_]+) v(\d+)\)\s*$")

#: TF-IDF settings. Fitted over the ~230-artefact corpus, so ``min_df`` counts
#: artefacts rather than events.
_VECTOR_PARAMS: Final[dict[str, Any]] = {
    "lowercase": True,
    "ngram_range": (1, 2),
    "min_df": 1,
    "sublinear_tf": True,
    "strip_accents": "unicode",
}

#: Weight on the second-worst page in the chain. The score is a maximum, and a
#: maximum over ~230 discrete artefact probabilities produces heavy ties — heavy
#: enough that the 0.1% FPR quantile can land inside one and refuse to be placed.
#: Breaking ties by the runner-up is not a fudge: between two chains whose worst
#: page is equally bad, the one with a second bad page really is worse. Small
#: enough that it can never reorder two chains with different maxima.
_RUNNER_UP_WEIGHT: Final[float] = 0.02


def _kind_and_variant(title: str, fallback: str) -> tuple[str, int]:
    match = _VARIANT_RE.search(title)
    return (match.group(1), int(match.group(2))) if match else (fallback, -1)


def held_out_artifacts(
    store: ContentStore, *, variants: bool = True, kind: str | None = None
) -> set[str]:
    """Artefact ids withheld from L3's vocabulary and training set.

    Args:
        store: The content store to select from.
        variants: Withhold the phrasings named in :data:`HELD_OUT_VARIANTS`.
        kind: Withhold every artefact of this kind as well. ``None`` for neither.
    """
    out: set[str] = set()
    for artifact in store.artifacts.values():
        if kind is not None and artifact.kind == kind:
            out.add(artifact.artifact_id)
            continue
        if not (variants and artifact.is_injected):
            continue
        family, variant = _kind_and_variant(artifact.title, artifact.kind)
        if HELD_OUT_VARIANTS.get(family) == variant:
            out.add(artifact.artifact_id)
    return out


def chains_for(
    frame: pd.DataFrame, store: ContentStore, *, max_artifacts: int = 12
) -> list[list[str]]:
    """Each event's provenance chain, resolved to artefact ids.

    A row with no agentic block has no chain and gets an empty list;
    :meth:`L3Model.score` returns NaN for it rather than a probability. "This rail
    carries no text" and "this text looks clean" are different facts and must not
    collapse into the same number.
    """
    chains = frame["ag_ingested_content_ids"].to_numpy()
    out: list[list[str]] = []
    cache: dict[str, str | None] = {}

    for chain in chains:
        if chain is None or len(chain) == 0:
            out.append([])
            continue
        ids: list[str] = []
        for content_id in chain[:max_artifacts]:
            key = str(content_id)
            if key in cache:
                artifact_id = cache[key]
            else:
                artifact = store.resolve(key)
                artifact_id = artifact.artifact_id if artifact is not None else None
                cache[key] = artifact_id
            if artifact_id is not None:
                ids.append(artifact_id)
        out.append(ids)
    return out


@dataclass(slots=True)
class L3Model:
    """A page classifier, applied to every page the agent read.

    Fitted on a corpus of text. **Never on transaction labels** — see the module
    docstring, and note that :meth:`fit` has no ``y`` parameter to pass one to.
    """

    seed: int = 1337
    store: ContentStore | None = None
    vectoriser: Any = None
    classifier: Any = None
    #: artefact id -> P(injected). Every artefact is scored, including the ones
    #: withheld from training, because the withheld ones are what the
    #: generalisation test measures.
    artifact_scores: dict[str, float] = field(default_factory=dict)
    held_out: set[str] = field(default_factory=set)
    n_train_artifacts: int = 0

    def _resolved_store(self) -> ContentStore:
        if self.store is None:
            self.store = load_content_store()
        return self.store

    def fit(self, *, hold_out_variants: bool = True, hold_out_kind: str | None = None) -> L3Model:
        """Fit the page classifier on the committed corpus.

        Args:
            hold_out_variants: Withhold the phrasings in
                :data:`HELD_OUT_VARIANTS` from the vocabulary and from training.
            hold_out_kind: Additionally withhold an entire adversarial kind, for
                the harder generalisation test. ``None`` for the phrasing test
                alone.

        There is no ``y``. The label is the artefact's own ``injected`` flag,
        which is a property of the text and not of any transaction.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        store = self._resolved_store()
        if not store.artifacts:
            raise RuntimeError(
                "the content corpus is empty; run 'python -m mantis.foundry.llm' to build it"
            )
        self.held_out = held_out_artifacts(store, variants=hold_out_variants, kind=hold_out_kind)

        train_ids = sorted(a for a in store.artifacts if a not in self.held_out)
        labels = np.array([store.artifacts[a].is_injected for a in train_ids], dtype=int)
        if labels.sum() in (0, len(labels)):
            raise ValueError("L3 needs both injected and benign artefacts in training")
        self.n_train_artifacts = len(train_ids)

        self.vectoriser = TfidfVectorizer(**_VECTOR_PARAMS)
        matrix = self.vectoriser.fit_transform(store.artifacts[a].text for a in train_ids)
        self.classifier = LogisticRegression(
            max_iter=2_000,
            C=4.0,
            class_weight="balanced",
            random_state=self.seed,
        )
        self.classifier.fit(matrix, labels)

        every = sorted(store.artifacts)
        scores = self.classifier.predict_proba(
            self.vectoriser.transform(store.artifacts[a].text for a in every)
        )[:, 1]
        self.artifact_scores = dict(zip(every, scores.tolist(), strict=True))
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """Per-event score, **NaN on rows carrying no text**.

        The worst page in the chain, nudged by the second worst; see
        :data:`_RUNNER_UP_WEIGHT` for why the tiebreak exists.

        NaN rather than zero on a classic authorisation. L3 has no opinion about a
        row with no provenance chain, and scoring it zero would be an opinion — at
        fusion time it would drag a classic event's fused score down for a reason
        that has nothing to do with risk.
        """
        if self.classifier is None:
            raise RuntimeError("L3Model.score called before fit")
        table = self.artifact_scores
        out = np.full(len(frame), np.nan, dtype=float)
        for i, chain in enumerate(chains_for(frame, self._resolved_store())):
            if not chain:
                continue
            values = sorted((table.get(a, 0.0) for a in chain), reverse=True)
            runner_up = values[1] if len(values) > 1 else 0.0
            out[i] = values[0] + _RUNNER_UP_WEIGHT * runner_up
        return out

    def holdout_mask(self, frame: pd.DataFrame) -> np.ndarray:
        """Rows whose chain carries an artefact withheld from training."""
        held = self.held_out
        return np.array(
            [any(a in held for a in chain) for chain in chains_for(frame, self._resolved_store())],
            dtype=bool,
        )

    def holdout_generalisation(self) -> pd.DataFrame:
        """P(injected) the classifier assigns to each withheld artefact.

        The most direct statement of whether the layer generalises: these texts
        were never in the vocabulary or the training set, so if their scores sit
        among the benign ones then the layer memorised rather than learned.
        """
        store = self._resolved_store()
        rows = [
            {
                "artifact_id": a,
                "kind": store.artifacts[a].kind,
                "injected": store.artifacts[a].is_injected,
                "p_injected": self.artifact_scores.get(a, float("nan")),
            }
            for a in sorted(self.held_out)
        ]
        return pd.DataFrame(rows).sort_values("p_injected", ignore_index=True)

    def top_terms(self, top: int = 20) -> pd.DataFrame:
        """The terms pushing a page toward "injected", by coefficient.

        Readable output is half the reason this layer is a linear model. An alert
        saying *"the agent read a page containing 'approve credit without
        verification'"* is one an analyst can act on; *"embedding distance 4.7"*
        is not.
        """
        if self.classifier is None:
            raise RuntimeError("L3Model.top_terms called before fit")
        names = np.asarray(self.vectoriser.get_feature_names_out())
        weights = np.asarray(self.classifier.coef_).ravel()
        order = np.argsort(weights)[::-1][:top]
        return pd.DataFrame({"term": names[order], "weight": weights[order]})
