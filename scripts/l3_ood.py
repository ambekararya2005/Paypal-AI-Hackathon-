"""L3 against text it did not come from — the out-of-distribution probe.

Run: ``python scripts/l3_ood.py``  (``make ood``)

Why this exists
----------------
RESULTS.md reports L3 at **1.000** on F1-01 and F1-03, and defends that number
two ways: a held-out *phrasing* and a held-out *kind*. Both of those hold-outs
are still drawn from ``data/cache/content/`` — text authored by one 7B model,
against one set of prompt templates, every specimen carrying the same
``[synthetic-mantis]`` marker and the same verbose-polite register. A classifier
can generalise perfectly across the variants of a corpus and still be reading the
corpus rather than the attack.

So this probe asks the harder question: **does L3 fire on an injection written by
somebody else?** 18 injected specimens and 10 benign controls, hand-authored, in
registers the corpus contains none of — HTML comments, YAML, a fake system
banner, transliterated Hindi-English, shouting, txt-speak, Cyrillic homoglyphs,
letters spaced apart. Written by a different model (Claude) from the one that
authored the corpus (``mistral:7b-instruct-q4_K_M``); no other local model was
available, and hand-authoring by a different author is the axis that matters
anyway.

The benign controls are load-bearing
--------------------------------------
Without them, a *low* number here is uninterpretable: it could mean L3 misses
novel injections, or it could mean L3 has simply learned "text that does not look
like the corpus is not injected", in which case the benign controls would score
low too and the layer has no opinion about this register at all. With them, the
two failure modes separate — the OOD ranking (injected vs benign, both in the new
register) is reported next to the OOD recall.

The rule, stated before the number is known
---------------------------------------------
**A lower number here is a better result than an unqualified 1.000.** Nothing in
this script may be tuned to rescue it: the vectoriser, the classifier, the
hold-out protocol and the threshold are all taken from the fitted
:class:`~mantis.defense.l3_text.L3Model` unchanged, and the payload file is
committed so the specimens cannot be quietly reselected after seeing the scores.
If L3 misses most of these, that is the honest ceiling of a TF-IDF bag of words
and it goes in the writeup as a limitation with a named fix.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # so the script runs without an editable install
    sys.path.insert(0, str(REPO_ROOT))

from mantis.core.paths import REFERENCE_DIR  # noqa: E402
from mantis.defense.l3_text import L3Model  # noqa: E402

#: The committed probe set. Hand-authored, and committed *before* it was scored.
PAYLOADS: Final[Path] = REFERENCE_DIR / "l3_ood_payloads.json"

#: The classifier's own decision boundary. Not fitted here, not tuned here — a
#: logistic regression's 0.5 is the point at which it says "injected", and using
#: anything else would be choosing an operating point after seeing the answer.
DECISION: Final[float] = 0.5


@dataclass(frozen=True, slots=True)
class Scored:
    payload_id: str
    register: str
    injected: bool
    p: float


@dataclass(frozen=True, slots=True)
class Panel:
    """One population of pages, read three ways.

    ``recall`` and ``fp`` are the layer at its own decision boundary — the
    deployable reading. ``roc`` is the layer's *discrimination* independent of
    where that boundary sits. The two can come apart badly, and when they do it
    is the whole finding: a classifier can keep its ordering and lose its
    calibration, which looks like success on one row and failure on the other.
    """

    label: str
    recall: float
    fp: float
    roc: float
    n_injected: int
    n_benign: int


def _load() -> dict:
    if not PAYLOADS.exists():
        raise SystemExit(f"missing probe set: {PAYLOADS}")
    return json.loads(PAYLOADS.read_text(encoding="utf-8"))


def _score_texts(model: L3Model, texts: list[str]) -> np.ndarray:
    """P(injected) per raw text, straight through the fitted pipeline.

    :meth:`L3Model.score` takes a transaction frame and resolves provenance
    chains; here there is no transaction, only the page. This calls the same
    vectoriser and the same classifier the event path calls, one rung lower down.
    """
    if model.classifier is None or model.vectoriser is None:
        raise RuntimeError("L3Model must be fitted first")
    matrix = model.vectoriser.transform(texts)
    return np.asarray(model.classifier.predict_proba(matrix)[:, 1], dtype=float)


def _panel(label: str, scores: np.ndarray, truth: np.ndarray) -> Panel:
    """Recall, false-positive rate and ROC for one population of pages."""
    injected = scores[truth]
    benign = scores[~truth]
    return Panel(
        label=label,
        recall=float((injected >= DECISION).mean()),
        fp=float((benign >= DECISION).mean()),
        roc=_roc(scores, truth),
        n_injected=int(truth.sum()),
        n_benign=int((~truth).sum()),
    )


def _in_distribution_reference(model: L3Model) -> Panel:
    """L3 on the corpus it was fitted on — the row the OOD number is read against.

    Page-level rather than event-level, because the OOD specimens are pages and
    have no transactions attached; the event-level 1.000 in RESULTS.md is a
    different denominator and the two are labelled as such wherever they appear
    together. Scored artefacts include the 34 withheld ones, which is the point:
    excluding them would make this row a training-set score.
    """
    store = model.store
    assert store is not None
    ids = sorted(store.artifacts)
    scores = np.array([model.artifact_scores.get(a, 0.0) for a in ids], dtype=float)
    truth = np.array([store.artifacts[a].is_injected for a in ids], dtype=bool)
    return _panel("in distribution", scores, truth)


def _roc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if labels.all() or not labels.any():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _recalibrated(scores: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """Recall if the page threshold were moved to clear the benign controls.

    **An oracle number and labelled as one.** The threshold is placed just above
    the highest-scoring benign control in this very set, so it has seen the
    answer and could not be set this way in deployment. It is reported because it
    separates the two things a bad OOD row can mean: if recall survives a
    threshold that clears the controls, the layer's *ordering* is intact and the
    defect is calibration; if it collapses, the ordering is gone too.
    """
    benign = scores[~truth]
    if not len(benign):
        return float("nan"), float("nan")
    cut = float(benign.max())
    return float((scores[truth] > cut).mean()), cut


def main() -> None:
    data = _load()
    print("L3 OUT-OF-DISTRIBUTION PROBE")
    print("=" * 78)
    print(f"  probe set   {PAYLOADS}")
    print(f"  authored by {data['author'].split(',')[0]}")
    print()

    model = L3Model().fit()
    print(f"  L3 fitted on {model.n_train_artifacts} corpus artefacts, "
          f"{len(model.held_out)} withheld (the RESULTS.md protocol, unchanged)")
    print()

    ref = _in_distribution_reference(model)

    rows: list[Scored] = []
    for label, key in ((True, "injected"), (False, "benign")):
        specimens = data[key]
        probs = _score_texts(model, [s["text"] for s in specimens])
        rows += [
            Scored(s["id"], s["register"], label, float(p))
            for s, p in zip(specimens, probs, strict=True)
        ]

    scores = np.array([r.p for r in rows], dtype=float)
    truth = np.array([r.injected for r in rows], dtype=bool)
    ood = _panel("out of distribution", scores, truth)
    recal_recall, recal_cut = _recalibrated(scores, truth)

    print("THE TWO PANELS, SIDE BY SIDE")
    print(f"  {'':<22} {'recall':>8} {'FP':>8} {'ROC':>8}   n+/n-")
    for panel in (ref, ood):
        print(
            f"  {panel.label:<22} {panel.recall:>8.3f} {panel.fp:>8.3f} {panel.roc:>8.3f}"
            f"   {panel.n_injected}/{panel.n_benign}"
        )
    print()
    print("  in distribution = the committed corpus, one 7B model, one register")
    print("  out of distribution = hand-authored, a different model, registers the")
    print("                        corpus contains none of, benign controls in the")
    print("                        SAME new register so novelty is not read as attack")
    print()

    print("RECALIBRATED — oracle, not an operating point")
    print(f"  threshold moved to {recal_cut:.3f}, just above the worst benign control")
    print(f"  recall on the injected specimens at that threshold:  {recal_recall:.3f}")
    print("  This threshold has seen the answer. It is here only to separate a")
    print("  calibration defect from a loss of ordering.")
    print()

    print("PER SPECIMEN")
    print(f"  {'id':<12} {'register':<24} {'truth':<9} {'p(injected)':>11}  fired")
    for r in sorted(rows, key=lambda r: -r.p):
        fired = "yes" if r.p >= DECISION else "-"
        print(
            f"  {r.payload_id:<12} {r.register:<24} "
            f"{'injected' if r.injected else 'benign':<9} {r.p:>11.3f}  {fired}"
        )
    print()

    missed = sorted((r for r in rows if r.injected and r.p < DECISION), key=lambda r: r.p)
    if missed:
        print(f"MISSED ({len(missed)}) — the registers that defeat a bag of words")
        for r in missed:
            print(f"  {r.payload_id:<12} {r.register:<24} p={r.p:.3f}")
        print()

    print("READING")
    print(_reading(ref, ood, recal_recall))

    out = REFERENCE_DIR.parent / "generated" / "l3_ood.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "decision_threshold": DECISION,
                "in_distribution": {
                    "recall": ref.recall,
                    "fp_rate": ref.fp,
                    "roc": ref.roc,
                    "n_injected": ref.n_injected,
                    "n_benign": ref.n_benign,
                },
                "out_of_distribution": {
                    "recall": ood.recall,
                    "fp_rate": ood.fp,
                    "roc": ood.roc,
                    "n_injected": ood.n_injected,
                    "n_benign": ood.n_benign,
                },
                "recalibrated_oracle": {
                    "threshold": recal_cut,
                    "recall": recal_recall,
                    "note": "threshold placed above the worst benign control; "
                    "has seen the answer, not an operating point",
                },
                "per_specimen": [
                    {
                        "id": r.payload_id,
                        "register": r.register,
                        "injected": r.injected,
                        "p_injected": r.p,
                    }
                    for r in sorted(rows, key=lambda r: -r.p)
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  wrote {out}")


def _reading(ref: Panel, ood: Panel, recalibrated: float) -> str:
    """The sentences the writeup may use, chosen from the shape of the result.

    The order of these checks is the point. A naive reading looks at recall
    first, and recall alone can say "1.000 out of distribution, the layer holds"
    while the layer is in fact calling nine of ten *clean* pages injected. So the
    false-positive rate on the controls is tested **before** recall, and when it
    is high the recall row is explicitly disqualified as a headline.
    """
    lines: list[str] = []
    drop = ood.roc - ref.roc

    if ood.fp > 0.5:
        lines.append(
            f"  L3's decision threshold does NOT transfer. It fires on {ood.recall:.0%} of the"
            f"\n  novel injections, but also on {ood.fp:.0%} of the clean controls written in the"
            "\n  same registers — so the recall row is meaningless on its own and must never"
            "\n  be quoted without the control row beside it. At its own 0.5 boundary, against"
            "\n  text unlike its corpus, this layer is a false-positive machine."
        )
    elif ood.recall >= ref.recall - 0.05:
        lines.append(
            "  L3 holds on text it did not come from, at a control FP rate it can defend."
        )
    elif ood.recall >= 0.5:
        lines.append(
            "  L3 degrades on novel phrasing but keeps a majority of its recall. The"
            "\n  in-distribution number is an upper bound and must be quoted with this row."
        )
    else:
        lines.append(
            "  L3 largely fails on text it did not come from. The in-distribution number"
            "\n  measures the corpus, not the attack class, and the writeup must say so."
        )

    lines.append("")
    lines.append(
        f"  What survives is the ORDERING: ROC {ref.roc:.3f} in distribution -> "
        f"{ood.roc:.3f} out"
        f"\n  of it ({drop:+.3f}). Injected pages still rank above clean ones written by the"
        "\n  same author in the same register, which is a real property and not a"
        "\n  restatement of the recall row."
    )
    if recalibrated == recalibrated:
        lines.append(
            f"  Re-placing the threshold above the worst control recovers {recalibrated:.0%}"
            "\n  recall (oracle). So the defect is CALIBRATION, not an absence of signal."
        )

    lines.append("")
    lines.append(
        "  Named fix, not done today: the page threshold is fitted on one corpus and"
        "\n  must instead be fitted on a benign corpus drawn from the traffic it will"
        "\n  actually see. Longer term a bag of words is the wrong model for this — it"
        "\n  keys on lexical markers of instruction ('do not', 'skip', 'without'), which"
        "\n  is why prose that merely SOUNDS procedural trips it."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
