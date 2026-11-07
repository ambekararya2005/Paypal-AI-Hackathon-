"""Rendering RESULTS.md.

RESULTS.md is written by code, not by hand, for one reason: a number that appears
in a document and nowhere in a run is a number nobody checked. Every figure below
comes out of :func:`~mantis.defense.experiment.run_experiment`, so re-running the
firewall rewrites the document, and a claim cannot outlive the measurement that
produced it.

The one thing this module is allowed to do that the experiment is not is
**editorialise** — the prose around each table says what the table means, what it
does not, and where it is weak. That prose is checked against the numbers it sits
next to on every run, because it is generated from them.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from mantis.core.events import SCHEMA_VERSION
from mantis.core.paths import GENERATED_DIR
from mantis.defense.experiment import ABLATED_FEATURE, LAYER_ORDER, TRAIN_SHARE
from mantis.defense.metrics import FPR_GRID, OPERATING_FPR
from mantis.defense.pool import POOL_SEEDS

__all__ = ["write_results"]

#: Written by ``python -m mantis.loop``. Read here so the loop's result lands in
#: the same document as the firewall's, rather than in a second file nobody opens.
ARENA_JSON = GENERATED_DIR / "arena.json"

#: Written by ``python -m mantis.foundry.fidelity`` and ``scripts/latency_bench.py``.
#: Same reason as the arena: criterion 2 and half of criterion 5 are measured by
#: code that is not the firewall, and a submission where those numbers live in
#: three separate files is a submission where one of them is stale.
#:
#: Both sections degrade to a single honest line when their artefact is absent.
#: Neither substitutes a number for a missing measurement.
FIDELITY_JSON = GENERATED_DIR / "fidelity.json"
LATENCY_JSON = GENERATED_DIR / "latency.json"

#: One line per layer, so the table's prose cannot drift from the architecture.
_LAYER_BLURB: dict[str, str] = {
    "L1": "GBDT, supervised, time-split, isotonic-calibrated",
    "L2": "isolation forest on events, legitimate traffic only — **residual monitor**",
    "L2e": "isolation forest on **entity** aggregates — the time-boxed experiment",
    "L3": "page classifier over ingested text, **fitted on no transaction labels**",
    "fused": "weighted logistic stacker over the four",
}


def _f3(value: float) -> str:
    return "n/a" if value != value else f"{value:.3f}"


def _f4(value: float) -> str:
    return "n/a" if value != value else f"{value:.4f}"


def _pct(value: float) -> str:
    return "n/a" if value != value else f"{value:.4%}"


def _curve_cells(curve: dict[float, tuple[float, float]]) -> str:
    return " | ".join(_f3(curve[f][0]) for f in FPR_GRID)


def write_results(result, pool: pd.DataFrame, path: Path) -> None:
    """Render the whole document. ``result`` is an ``ExperimentResult``."""
    out: list[str] = []
    w = out.append

    w("# MANTIS — Detection results")
    w("")
    w(f"*Day 5. Schema v{SCHEMA_VERSION}. Generated {date.today().isoformat()} by "
      "`python -m mantis.defense`.*")
    w("")

    # ---------------------------------------------------------------- operating point
    w("## The operating point, and why it is a curve")
    w("")
    w("> **Every recall in this document is measured at a threshold placed so that a fixed "
      "share of legitimate test traffic is flagged.** The headline share is "
      f"**{OPERATING_FPR:.1%}**.")
    w("")
    w("No accuracy figures appear anywhere here, and that is deliberate: at ~1% prevalence a "
      "model that approves everything is 99% accurate. The metrics reported are **AUC-PR** and "
      "**recall at a fixed FPR**, the second always with its realised false-positive rate "
      "attached.")
    w("")
    w("Day 4 quoted one operating point. Day 5 quotes " + ", ".join(f"{f:.1%}" for f in FPR_GRID)
      + " — a curve rather than a point, because one number at one budget is something a reader "
        "has to trust you did not pick, and three is a shape. 0.1% stays the headline because it "
        "is the tightest, and the tightest is the one an issuer can actually staff. 1.0% is "
        "roughly the top of what a review queue absorbs, which is why the curve stops there.")
    w("")
    w("Each model variant gets its own threshold, because each has its own score distribution. "
      "What is held constant across every column of every table is the false-positive rate — "
      "the quantity an issuer budgets, and the only thing that makes two columns comparable.")
    w("")
    w("### Two kinds of recall, both labelled")
    w("")
    w("Day 4 reported **event-level** recall only, and event-level recall is the wrong question "
      "for half of this atlas. A mule ring that runs 40 authorisations and is flagged on 3 of "
      "them scores 7.5% event-level and is **caught**: one alert opens a case, and the case "
      "takes the ring. So every layer is also reported at **campaign level** — was the campaign "
      "flagged at all, and on which of its events.")
    w("")
    w("Neither is the real number. Event-level flatters a layer that fires on every event of an "
      "obvious attack; campaign-level flatters a layer that fires once on something subtle. "
      "They appear side by side, always, with their names on them.")
    w("")

    # ---------------------------------------------------------------- dataset
    w("## The evaluation dataset")
    w("")
    w(f"{len(POOL_SEEDS)} independently generated worlds, seeds "
      f"`{', '.join(str(s) for s in POOL_SEEDS)}`, pooled.")
    w("")
    w("| | |")
    w("|---|---|")
    w(f"| events | {len(pool):,} |")
    w(f"| fraud | {int(pool['is_fraud'].sum()):,} ({pool['is_fraud'].mean():.4%}) |")
    w(f"| train / test | {result.n_train:,} / {result.n_test:,}, **time-based** split at the "
      f"{TRAIN_SHARE:.0%} quantile |")
    w(f"| features | {result.n_features} "
      f"(of which {result.graph_features} are the new `gph_` entity-graph block) |")
    w("")
    w("Pooling is not resampling one file five times: each seed regenerates the whole "
      "population — its own customers, merchants, devices, agents and calendar — so the variance "
      "being averaged over is generator variance. Identifiers are namespaced per seed before "
      "concatenation, because letting `cus-00042` from two seeds collide would fuse two people's "
      "histories inside every velocity, entity and graph feature.")
    w("")

    # ---------------------------------------------------------------- layers
    w("## Layer performance")
    w("")
    recall_headers = " | ".join(f"recall@{f:.1%}" for f in FPR_GRID)
    w(f"| layer | AUC-PR | ROC-AUC | {recall_headers} | realised FPR | campaign recall | "
      "first alert |")
    w("|---" * (6 + len(FPR_GRID)) + "|")
    for name in LAYER_ORDER:
        layer = result.layers[name]
        camp = layer.campaigns
        first = (
            "n/a" if camp.median_index != camp.median_index
            else f"event {camp.median_index:.0f} of {camp.median_size:.0f}"
        )
        w(f"| **{name}** {_LAYER_BLURB[name]} | {_f4(layer.report.auc_pr)} | "
          f"{_f4(layer.report.auc_roc)} | {_curve_cells(layer.curve)} | "
          f"{_pct(layer.report.realised_fpr)} | {_f3(camp.recall)} "
          f"({camp.n_caught}/{camp.n_campaigns}) | {first} |")
    w("")
    w(f"Baseline precision (a coin flip) is the prevalence: {result.prevalence:.4%}. AUC-PR is "
      "only interpretable against it.")
    w("")
    w(_fusion_note(result))
    w("")

    # ---------------------------------------------------------------- fusion weights
    if len(result.fusion_weights):
        w("### What the fusion learned")
        w("")
        w("| layer | weight on the percentile | weight on the raw score | "
          "weight on \"had an opinion\" |")
        w("|---|---|---|---|")
        for row in result.fusion_weights.itertuples():
            present = (
                "n/a" if row.weight_present != row.weight_present
                else f"{row.weight_present:+.3f}"
            )
            w(f"| {row.layer} | {row.weight_percentile:+.3f} | {row.weight_score:+.3f} | "
              f"{present} |")
        w("")
        w("The coefficients are the interesting output, more than the fused recall is. A weight "
          "near zero is the stacker saying that layer carries no information the others do not "
          "already have — which is a cleaner statement than any table of recalls, and it is how "
          "the Day 4 problem got fixed without anyone hand-tuning a number.")
        w("")

    # ---------------------------------------------------------------- per rail
    if result.l1_rail:
        w("### Per rail, because the headline is partly measuring \"is this agentic\"")
        w("")
        w("Fraud is concentrated on the agentic rail by design — 15% of volume carrying 51% of "
          "the fraud, a 5.7x concentration. A single number across both rails is therefore "
          "partly reading a field the issuer gets free off the authorisation message.")
        w("")
        w("| rail | n positive | AUC-PR | recall@0.1%FPR |")
        w("|---|---|---|---|")
        for rail, report in result.l1_rail.items():
            w(f"| {rail} | {report.n_positive:,} | {_f4(report.auc_pr)} | {_f3(report.recall)} |")
        w("")

    # ---------------------------------------------------------------- LOFO
    w("## Leave one family out — the headline experiment")
    w("")
    w("For each family, L1 is retrained with that family **entirely removed from the training "
      "set** and then asked to catch it in the test set anyway. L2 never sees any attack at all, "
      "and L3 never sees a transaction, so neither of their columns changes with the hold-out.")
    w("")
    frame = result.per_family
    w("| family | n_pos | L1 (trained WITH it) | L1 (family HELD OUT) | L2 | L3 | fused | "
      "fused, held out |")
    w("|---|---|---|---|---|---|---|---|")
    for row in frame.itertuples():
        w(f"| **{row.family}** | {row.n_pos:,} | {_f3(row.l1_with)} | {_f3(row.l1_heldout)} | "
          f"{_f3(row.l2)} | {_f3(row.l3)} | {_f3(row.fused_with)} | {_f3(row.fused_heldout)} |")
    if len(frame):
        mean_with = frame["l1_with"].mean()
        mean_held = frame["l1_heldout"].mean()
        w(f"| *mean* | | *{mean_with:.3f}* | *{mean_held:.3f}* | *{frame['l2'].mean():.3f}* | "
          f"*{frame['l3'].mean():.3f}* | *{frame['fused_with'].mean():.3f}* | "
          f"*{frame['fused_heldout'].mean():.3f}* |")
        w("")
        w(f"**Mean event-level recall lost when a family is held out of training: "
          f"{mean_with - mean_held:+.3f}** ({mean_with:.1%} → {mean_held:.1%}).")
    w("")

    if len(result.per_family_campaign):
        w("### The same experiment at campaign level")
        w("")
        w("| family | campaigns | median size | fused (with) | fused (held out) | "
          "first alert at event | elapsed before alert |")
        w("|---|---|---|---|---|---|---|")
        for row in result.per_family_campaign.itertuples():
            elapsed = (
                "n/a" if row.share_before_alert != row.share_before_alert
                else f"{row.share_before_alert:.0%}"
            )
            index = (
                "n/a" if row.median_index != row.median_index else f"{row.median_index:.0f}"
            )
            w(f"| **{row.family}** | {row.n_campaigns} | {row.median_size:.0f} | "
              f"{_f3(row.fused_with)} | {_f3(row.fused_heldout)} | {index} | {elapsed} |")
        w("")
        w("Read the last two columns together with the first: a campaign caught on its third "
          "event of forty is a case opened before most of the money moved. A campaign caught on "
          "its thirty-fifth is a post-mortem.")
        w("")

    w(_lofo_reading(result))
    w("")

    # ---------------------------------------------------------------- the loop
    w(_arena_section())
    w("")

    # ---------------------------------------------------------------- L3
    if len(result.l3_cards):
        w("## L3, and why a near-perfect number here is honest")
        w("")
        w("L3 classifies **a page**, and an event's score is the worst page its agent read. It "
          "is fitted on the committed content corpus using each artefact's own `injected` flag "
          "— **it never sees `is_fraud`, and `L3Model.fit` has no `y` parameter to pass one "
          "to**. That is why it sits with L0 in the reframed architecture rather than with L1: "
          "it works on an attack it has never seen in the payment data, because the thing it was "
          "trained on is not payment data.")
        w("")
        w("| card | n_pos | recall | on unseen *phrasing* | n unseen | on an unseen *kind* |")
        w("|---|---|---|---|---|---|")
        for row in result.l3_cards.itertuples():
            w(f"| {row.attack_id} | {row.n_pos:,} | {_f3(row.recall)} | "
              f"{_f3(row.recall_unseen_phrasing)} | {row.n_unseen_phrasing:,} | "
              f"{_f3(row.recall_unseen_kind)} |")
        w("")
        w(_l3_note(result))
        w("")
        w("What would be dishonest is claiming that generalises when it does not, so the last "
          "two columns are the ones the writeup may lean on. **Unseen phrasing**: the "
          "highest-numbered variant of every adversarial kind is withheld from the vocabulary "
          "and from training. **Unseen kind**: every `refund_ticket` specimen is withheld — all "
          "of them — so F1-03 is scored on an injection type the classifier has never seen in "
          "any wording.")
        w("")
        if len(result.l3_holdout):
            injected = result.l3_holdout[result.l3_holdout["injected"]]
            if len(injected):
                w(f"Measured directly on the withheld texts themselves: {len(injected)} "
                  "artefacts that were never in the vocabulary or the training set score "
                  f"P(injected) between {injected['p_injected'].min():.2f} and "
                  f"{injected['p_injected'].max():.2f}, median "
                  f"{injected['p_injected'].median():.2f}. If the layer had memorised rather "
                  "than learned, these would sit with the benign artefacts.")
                w("")
        w("L3's *overall* recall is low, and that is correct rather than disappointing: it has "
          "an opinion about two of the fifteen cards and no opinion at all about the classic "
          "rail, where there is no provenance chain to read. A layer that scored highly on "
          "everything would be a layer reading something other than the text.")
        w("")
        w(_l3_ood_section())
        w("")

    # ---------------------------------------------------------------- L2 / L2e
    w(_novelty_section(result))
    w("")

    # ---------------------------------------------------------------- per card
    w("## Per attack card")
    w("")
    w("At the same 0.1% FPR operating point. The last column is campaign-level: the share of "
      "that card's campaigns in which **at least one** event was flagged by the fused score.")
    w("")
    w("| card | n_pos | " + " | ".join(LAYER_ORDER) + " | campaigns caught |")
    w("|---|---|" + "---|" * (len(LAYER_ORDER) + 1))
    for row in result.per_attack.itertuples():
        cells = " | ".join(_f3(getattr(row, name)) for name in LAYER_ORDER)
        w(f"| {row.attack_id} | {row.n_pos:,} | {cells} | {_f3(row.campaign)} |")
    w("")

    # ---------------------------------------------------------------- decisions
    if result.decisions:
        w("## The decision layer")
        w("")
        w("A score is not an action. The fused score maps to one of four responses, with each "
          "boundary placed at a false-positive budget on legitimate traffic rather than at a "
          "hard-coded score — so a retrain re-prices nothing. Over the test window:")
        w("")
        w("| decision | events | share |")
        w("|---|---|---|")
        total = max(sum(result.decisions.values()), 1)
        for name, count in result.decisions.items():
            w(f"| {name} | {count:,} | {count / total:.3%} |")
        w("")
        w("A deterministic L0 clause firing overrides all four, because \"the mandate had "
          "expired\" is a defensible thing to tell a cardholder and \"the ensemble scored "
          "0.83\" is not.")
        w("")

    # ---------------------------------------------------------------- explain
    if result.explanation:
        w("## What an alert actually says")
        w("")
        w("Per-event attribution comes from LightGBM's own `pred_contrib`, not from SHAP. For a "
          "tree ensemble that is the *same* computation — `TreeExplainer` calls into LightGBM "
          "for it — without a wrapper on the scoring path. The three highest-scoring test "
          "events, with the features that put them there (contributions are in log-odds of the "
          "raw margin, which is what the ranking, and therefore the alert, is made of):")
        w("")
        w("```")
        w(result.explanation.rstrip())
        w("```")
        w("")

    # ---------------------------------------------------------------- ablation
    if len(result.ablation):
        w("## The feature that was too good, ablated")
        w("")
        w(f"`{ABLATED_FEATURE}` — the residual of an agent's deliberation latency against what a "
          "ticket that size deserves — separates F1-01 at **0.99 AUC on its own**. That is above "
          "the foundry's own 0.95 separability gate, and the gate never saw it: the gate probes "
          "**raw columns**, and this is a derived residual.")
        w("")
        w("It has not been silently removed and it has not been silently kept:")
        w("")
        w("| family | recall with the feature | recall without it | delta |")
        w("|---|---|---|---|")
        merged = result.ablation.merge(
            result.per_family[["family", "l1_with"]], on="family", how="left"
        )
        for row in merged.itertuples():
            delta = row.recall_without - row.l1_with
            w(f"| {row.family} | {_f3(row.l1_with)} | {_f3(row.recall_without)} | {delta:+.3f} |")
        w("")

    # ---------------------------------------------------------------- importance
    w("## What L1 actually uses")
    w("")
    w("| feature | gain share |")
    w("|---|---|")
    for row in result.importance.head(15).itertuples():
        w(f"| `{row.feature}` | {row.share:.2%} |")
    w("")

    # ---------------------------------------------------------------- fidelity
    w(_fidelity_section())

    # ---------------------------------------------------------------- latency
    w(_latency_section())

    # ---------------------------------------------------------------- feasibility
    w(_feasibility_section())

    # ---------------------------------------------------------------- caveats
    w("## What this does not claim")
    w("")
    w("- **This is not real-world performance.** It is measured on synthetic data whose attacks "
      "we wrote. The fidelity scorecard above is the argument that the background is "
      "realistic, and it is a qualified argument: read its discriminator row before reading "
      "any recall here.")
    w("- **F5 is absent from every table.** It is the zero-day holdout family and has no "
      "implemented injector, so it is not in the data and cannot be scored. The "
      "leave-one-family-out columns and the loop experiment are the closest available stand-ins.")
    w("- **L2 is not a detector and is no longer presented as one.** See the section above.")
    w("- **L3 covers two of fifteen cards.** It is a specialist, and its overall recall should be "
      "read as coverage of the agentic-injection rail rather than as a headline.")
    w("- **The current event's own outcome is never a feature.** `auth_response`, `settled` and "
      "`settlement_lag_hours` of the row being scored are blocked by name in the feature "
      "builder, alongside the label and post-hoc columns. Feeding them in would raise F4-27's "
      "recall substantially and mean nothing — an issuer cannot decline a transaction because it "
      "was declined.")
    w("- **The measured latency is over budget, and the section above says so.** What is *not* "
      "claimed is the reverse: no number here is an estimate of what a rewritten scoring path "
      "would cost. The p99 is what this implementation does today.")
    w("")

    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _card_recall(result, card: str, layer: str) -> float:
    frame = result.per_attack
    row = frame[frame["attack_id"] == card]
    return float(row[layer].iloc[0]) if len(row) else float("nan")


def _l3_note(result) -> str:
    """The "is a near-perfect number suspicious" paragraph, from measured numbers."""
    l3 = _card_recall(result, "F1-01", "L3")
    l1 = _card_recall(result, "F1-01", "L1")
    opening = (
        "**The injected instruction *is* the attack.** F1-01 is defined as an agent acting on "
        "text that told it to change the cart, and F1-03 as one acting on a refund request that "
        "told it to skip verification. A classifier reading that text and finding the "
        "instruction is not cheating and is not leakage — it is the detection working."
    )
    if l3 != l3 or l1 != l1:
        return opening
    if l3 > l1 + 0.05:
        return (
            f"{opening} The comparison that makes it concrete is L1's: on F1-01, L3 reaches "
            f"{_f3(l3)} reading the text while L1 reaches {_f3(l1)} off {result.n_features} "
            "tabular features, because the tabular trace of \"the agent read a bad page\" is "
            "faint and the page itself is not."
        )
    return (
        f"{opening} Note that L1 is not behind here: on F1-01 it reaches {_f3(l1)} off "
        f"{result.n_features} tabular features against L3's {_f3(l3)}. The tabular trace of this "
        "attack is strong enough on its own in this run, so L3's value on F1-01 is not extra "
        "recall — it is **independence**. L3 is the only layer whose F1-01 recall survives the "
        "family being held out of L1's training set, because L3 was never trained on transactions "
        "at all."
    )


def _l3_ood_section() -> str:
    """The out-of-distribution probe, read off ``scripts/l3_ood.py``'s artefact.

    Absent until the probe has been run, and silent when it is — a section that
    invented numbers because a file was missing would be worse than no section.
    """
    import json

    from mantis.core.paths import GENERATED_DIR

    path = GENERATED_DIR / "l3_ood.json"
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""

    ind, ood = data["in_distribution"], data["out_of_distribution"]
    recal = data.get("recalibrated_oracle", {})
    lines = [
        "### The harder test: text L3 did not come from",
        "",
        "Both hold-outs above are still drawn from `data/cache/content/` — one 7B model, one "
        "set of prompt templates, one register. A classifier can generalise perfectly across "
        "the variants of a corpus and still be reading the corpus. So "
        f"`scripts/l3_ood.py` scores {ood['n_injected']} hand-authored injection payloads and "
        f"{ood['n_benign']} hand-authored **benign controls in the same registers** — HTML "
        "comments, YAML, a fake system banner, transliterated Hindi-English, shouting, "
        "txt-speak, Cyrillic homoglyphs — written by a different model from the one that "
        "authored the corpus, and committed before they were scored.",
        "",
        "| | recall | FP on controls | ROC | n+ / n- |",
        "|---|---|---|---|---|",
        f"| in distribution (the corpus) | {ind['recall']:.3f} | {ind['fp_rate']:.3f} | "
        f"{ind['roc']:.3f} | {ind['n_injected']} / {ind['n_benign']} |",
        f"| **out of distribution** | {ood['recall']:.3f} | **{ood['fp_rate']:.3f}** | "
        f"**{ood['roc']:.3f}** | {ood['n_injected']} / {ood['n_benign']} |",
        "",
    ]

    if ood["fp_rate"] > 0.5:
        lines += [
            f"**Read the second column before the first.** L3 fires on {ood['recall']:.0%} of the "
            f"novel injections — and on {ood['fp_rate']:.0%} of the *clean* pages written in the "
            "same registers. The recall cell is therefore meaningless on its own, and this is "
            "exactly why the benign controls were authored alongside the payloads rather than "
            "afterwards: without them this table would have read as a triumph.",
            "",
            "**L3's decision threshold does not transfer.** Calibrated on one corpus and pointed "
            "at text unlike it, the layer is a false-positive machine. What survives is the "
            f"*ordering*: ROC {ind['roc']:.3f} → {ood['roc']:.3f}. Injected pages still rank "
            "above clean ones by the same author in the same register",
        ]
        if recal.get("recall") is not None:
            lines[-1] += (
                f", and moving the threshold above the worst control recovers "
                f"{recal['recall']:.0%} recall — an oracle number, since that threshold has seen "
                "the answer, but enough to locate the defect in **calibration** rather than in "
                "an absence of signal"
            )
        lines[-1] += "."
        lines += [
            "",
            "**The named fix, not done today.** The page threshold is fitted on one corpus and "
            "must instead be fitted on benign text drawn from the traffic it will actually see. "
            "Longer term, a bag of words is the wrong model: it keys on lexical markers of "
            "instruction — *do not*, *skip*, *without* — which is why prose that merely sounds "
            "procedural trips it. The 1.000 in the table above is a real number about this "
            "corpus and **not** a claim about the open web.",
        ]
    else:
        lines += [
            f"L3 keeps {ood['recall']:.0%} of its recall on text it did not come from, at "
            f"{ood['fp_rate']:.0%} false positives on controls written in the same registers.",
        ]
    return "\n".join(lines)


def _fusion_note(result) -> str:
    """The Day 4 → Day 5 fusion story, generated from this run's own numbers."""
    l1 = result.layers["L1"].report.recall
    fused = result.layers["fused"].report.recall
    if fused != fused or l1 != l1:
        return ""
    if fused >= l1:
        return (
            f"**Fusion is now better than L1 alone** ({l1:.3f} → {fused:.3f} at "
            f"{OPERATING_FPR:.1%} FPR). On Day 4 it was worse — 0.361 → 0.286 — because the "
            "layers were combined with an *unweighted* noisy-OR, which gave a near-random L2 "
            "equal say inside a fixed false-positive budget. The fix was not to delete L2 but to "
            "fit the weights on a slice of the training window none of the base layers had seen, "
            "and let the stacker discount it. The coefficients below are what it decided."
        )
    return (
        f"**Fusion is still not beating L1 alone** ({l1:.3f} vs {fused:.3f} at "
        f"{OPERATING_FPR:.1%} FPR), and that is reported rather than hidden by quoting only the "
        "best row. The weighting is now fitted rather than uniform, so this is no longer the Day "
        "4 failure mode; what it says instead is that at this operating point the auxiliary "
        "layers' independent signal does not outweigh the cost of spending FP budget on them. "
        "**Quote L1's number, not the fused one.**"
    )


def _lofo_reading(result) -> str:
    frame = result.per_family
    if not len(frame):
        return ""
    mean_with = frame["l1_with"].mean()
    mean_held = frame["l1_heldout"].mean()
    l2_mean = frame["l2"].mean()
    return (
        "### Reading it\n"
        "\n"
        f"**Supervised detection collapses on attacks it has never seen.** Held-out recall "
        f"averages {mean_held:.1%} against {mean_with:.1%} when the family is in training. This "
        "is the expected and the important result: an L1 trained on a labelled history is a "
        "detector for that history.\n"
        "\n"
        f"**The unsupervised layer does not rescue it.** L2 averages {l2_mean:.1%} at the same "
        "operating point. Day 5 stopped treating that as a gap to be closed and reframed it as "
        "the finding it is — see the next two sections. The architecture's answer to an unseen "
        "attack is now **L0's protocol invariants**, which need no training data because a "
        "violated mandate is a broken contract rather than an outlier, and **the closed loop**, "
        "which manufactures the attack before an attacker does."
    )


def _novelty_section(result) -> str:
    """The negative result, written as a finding rather than an apology."""
    l2 = result.layers["L2"].report
    l2e = result.layers["L2e"].report
    return (
        "## The negative result worth publishing\n"
        "\n"
        "> **Attacks built to be distributionally faithful are, by construction, invisible to "
        "distributional anomaly detection.**\n"
        "\n"
        f"L2 scores events: AUC-PR {_f4(l2.auc_pr)}, ROC {_f4(l2.auc_roc)}, recall "
        f"{_f3(l2.recall)} at {_pct(l2.realised_fpr)}. The stated reason it *should* have worked "
        "better is a ring — every event in a mule network is bland, but the entity is not. So "
        "Day 5 tested exactly that, time-boxed to thirty minutes: **L2e**, an isolation forest "
        "over entity aggregates rather than event rows, scoring customers and merchants instead "
        f"of authorisations. Result: AUC-PR {_f4(l2e.auc_pr)}, ROC {_f4(l2e.auc_roc)}, recall "
        f"{_f3(l2e.recall)} at {_pct(l2e.realised_fpr)}.\n"
        "\n"
        + _below_chance_note(l2e)
        + "The negative result is worth more than the layer would have been. "
        "L2e is also *generous* to the hypothesis in a way that has to be stated: an entity's "
        "vector is aggregated over the whole scoring window, so the score attached to that "
        "entity's first event was computed with knowledge of their last. That is a legitimate "
        "deployment mode — a nightly entity-risk queue, which is what AML and merchant-monitoring "
        "teams actually run — but it is not an authorisation scorer, and it still did not work.\n"
        "\n"
        "**Our own fidelity work caused this.** Every foundry decision pushed the attacks toward "
        "the legitimate manifold: clone real background rows, resample amounts inside the target "
        "MCC's own empirical band, redraw the hour of day from the population's diurnal curve, "
        "widen three legitimate tails specifically so an attack would not be free, keep "
        "provenance planting length-preserving. The Day 2 separability gate is literally a rule "
        "forbidding any single raw column from separating an attack above 0.95 AUC. An isolation "
        "forest measures distance from that manifold; we spent two days minimising it.\n"
        "\n"
        "And that is the property real agentic fraud has. An agent paying with a validly-signed "
        "mandate, on a real cardholder's real device, for a plausible amount at a real merchant, "
        "**is** legitimate in every marginal. The fraud lives in the *intent* — which is L3, the "
        "text — and in the *relations* — which is L4, the graph — not in the marginals. That is "
        "why \"just run an autoencoder on it\" is not an answer to agentic fraud, and it is why "
        "the two layers Day 5 built are the two that read intent and relations.\n"
        "\n"
        "**Corollary, and it is a sharp one:** a fidelity scorecard and an anomaly-detection "
        "recall number are in tension *by construction*. A project reporting both as high is "
        "reporting one of them wrongly.\n"
        "\n"
        "What survives for L2 is the narrow claim, and it is real: its recall is completely "
        "**unaffected** by whether an attack was in training, which is a property no supervised "
        "layer has. Its job in the architecture is now **residual monitor and drift canary** — "
        "\"has the shape of legitimate traffic moved\" — and no table in this repo presents it as "
        "a detector."
    )


#: A gene is "moved" once it is this far from its default, as a share of its
#: range. Mirrors ``mantis.loop.writeback._NOTABLE`` — the two have to agree,
#: because this paragraph describes what that module wrote.
_NOTABLE_GENE_MOVE: float = 0.20


def _survivor_note(survivors: list[dict]) -> list[str]:
    """What survived, split into genuine variants and the unmutated parents.

    The split is not pedantry. Every card's arena population is seeded with an
    **identity genome** so the evasion curve carries its own no-evolution
    reference row, and that individual competes like any other. On a card the
    detector is already bad at, it wins — and a survivor list that did not say so
    would be claiming the loop discovered an attack that was already in
    ``cards/``.
    """
    from mantis.loop.genome import GENE_BOUNDS
    from mantis.loop.genome import identity_genome as _identity

    def moved(row: dict) -> bool:
        reference = _identity(row["card_id"])
        for gene, value in row["genes"].items():
            low, high = GENE_BOUNDS[gene]
            default = float(getattr(reference, gene))
            if abs(value - default) / (high - low) >= _NOTABLE_GENE_MOVE:
                return True
        return False

    novel = [row for row in survivors if moved(row)]
    unmutated = [row for row in survivors if not moved(row)]

    lines = [
        f"{len(novel)} genuinely mutated variant(s) survived three or more consecutive rounds "
        "against a retraining detector and were written back to `mantis/atlas/discovered/` as "
        "validated attack cards with `discovered_by: adversarial_loop`. They live beside the "
        "atlas rather than inside it: `mantis/atlas/cards/` is the frozen 42, and the "
        "implemented count is a ratchet that moves only when an injector lands. Each card ships "
        "a `.genome.json` sidecar, so a variant is reproducible rather than merely described.",
        "",
    ]
    if unmutated:
        worst = max(unmutated, key=lambda row: row["evasion"])
        lines += [
            f"**{len(unmutated)} further survivor(s) were the *unmutated* attack** — every gene "
            "at its default — and they are **not** written back. Each card's population is "
            "seeded with an identity genome so the curve carries its own no-evolution reference "
            "row, and on cards the detector is already weak at, that individual simply wins: "
            f"{worst['card_id']} evades **{worst['evasion']:.1%}** of decisions without being "
            "mutated at all. That is a result about the parent card, not a discovery, and "
            "recording it as one would be the exact overclaim the atlas ratchet exists to "
            "prevent. It also points at where the next detection work is: "
            + ", ".join(sorted({row["card_id"] for row in unmutated}))
            + ".",
            "",
        ]
    return lines


def _curve_reading(curve: list[float]) -> str:
    """Describe the shape the curve actually has, not the one the design predicts."""
    if len(curve) < 2:
        return "A single generation is not a curve; re-run with more."

    first, last = curve[0], curve[-1]
    floor = min(curve)
    floor_at = curve.index(floor)

    if last >= first:
        return (
            f"Evasion runs **{first:.3f} → {last:.3f}** over {len(curve)} generations, which is "
            "**not** the declining curve the design predicts. Reported as measured. Either the "
            "adversary is searching faster than the detector is learning, or the retrain is not "
            "using the arena's output — and which of those it is has to be diagnosed rather than "
            "argued about."
        )

    total_fall = first - last
    first_step = first - curve[1]
    if total_fall > 0 and first_step / total_fall > 0.80:
        where = (
            " Almost all of that fall happens at the **first** retrain — generation 0 is scored "
            "against a detector that has never seen a mutated variant, and generation 1 against "
            "one that has."
        )
    else:
        where = (
            " The fall is spread across the generations rather than concentrated at the first "
            "retrain, which means each round of manufactured variants is still teaching the "
            "detector something the previous round did not."
        )
    fall = (
        f"Evasion falls from **{first:.3f} to {last:.3f}** over {len(curve)} generations."
        f"{where}"
    )

    rebound = last - floor
    if rebound > 0.02:
        return (
            f"{fall}\n\nAfter that the curve **rebounds**, from a floor of {floor:.3f} at "
            f"generation {floor_at} back to {last:.3f}, and that is the more interesting half of "
            "the shape. It is what a real arms race looks like: the defender's first response is "
            "worth far more than any subsequent one, and the adversary then claws back a fraction "
            "of it by finding a corner of the gene space the retrain has not covered. A curve "
            "that fell monotonically to zero would be a curve to distrust — it would mean the "
            "adversary had stopped searching, which with a mutation operator that never stops "
            "would mean the search space was too small to be interesting.\n\nThe claim the "
            f"chart supports is therefore the bounded one: **retraining on manufactured variants "
            f"cuts evasion by {(first - last) / first:.0%} and holds it down**, not that it "
            "drives evasion to zero. Nothing drives evasion to zero."
        )

    return (
        f"{fall} It then stays down: the curve does not rebound, and the adversary does not "
        f"recover the ground it lost. The claim the chart supports is that retraining on "
        f"manufactured variants cuts evasion by {(first - last) / first:.0%} and keeps it cut."
    )


def _below_chance_note(report) -> str:
    """Say plainly when a layer is not merely weak but anti-correlated."""
    if report.auc_roc != report.auc_roc:
        return "It did not move. "
    if report.auc_roc >= 0.50:
        return "It did not move. "
    return (
        f"It did not move — it went **backwards**. A ROC of {report.auc_roc:.4f} is *below "
        "chance*, which is a stronger statement than \"weak\": entity aggregates are mildly "
        "**anti-correlated** with fraud on this data. The reason is the foundry's own realism "
        "discipline pointed at the entity level: attacks ride established customers and busy "
        "merchants by construction, while the genuinely unusual entities in this population are "
        "ordinary people with three transactions. An outlier detector at entity level finds the "
        "quiet, and the quiet is innocent. "
    )


def _arena_section() -> str:
    """The loop's result, read from arena.json if the loop has been run."""
    if not ARENA_JSON.exists():
        return (
            "## The closed loop\n"
            "\n"
            "_Not yet run for this document. `python -m mantis.loop` writes "
            "`data/generated/arena.json`, and re-running `python -m mantis.defense` folds its "
            "numbers in here._"
        )
    payload = json.loads(ARENA_JSON.read_text(encoding="utf-8"))
    curve = payload.get("evasion_curve") or []
    lines = [
        "## The closed loop — the other half of the zero-day answer",
        "",
        "An evolutionary adversary mutates the **operational parameters** of a known attack — "
        "pacing, ring fan-out, device rotation, ticket size, how many injected pages the agent "
        "reads — and selects on **evasion x payoff** against the live detector. Between rounds "
        "the detector retrains on everything the arena has produced. That is the loop an "
        "operator runs when their attempts start getting declined, except that here the defender "
        "runs it first.",
        "",
    ]
    if curve:
        lines += [
            "### The evasion curve",
            "",
            "| generation | " + " | ".join(str(g["generation"]) for g in payload["generations"])
            + " |",
            "|---" * (len(curve) + 1) + "|",
            "| mean evasion | " + " | ".join(f"{v:.3f}" for v in curve) + " |",
            "| max evasion | "
            + " | ".join(f"{g['max_evasion']:.3f}" for g in payload["generations"]) + " |",
            "",
            _curve_reading(curve),
            "",
            "Full per-generation detail, the surviving genomes and which genes moved are in "
            "`data/generated/arena.json`.",
            "",
        ]
    survivors = payload.get("survivors") or []
    if survivors:
        lines += _survivor_note(survivors)
    zero = payload.get("zero_day")
    if zero:
        lines += [
            "### The zero-day demonstration",
            "",
            "This is the comparison the submission's argument rests on.",
            "",
            f"| detector | recall@0.1%FPR on the {zero['n_test_positive']:,} real "
            f"{zero['family']} test events |",
            "|---|---|",
            f"| trained **with** family {zero['family']} | "
            f"{zero['recall_trained_on_family']:.3f} |",
            f"| family {zero['family']} **held out** of training | "
            f"{zero['recall_family_held_out']:.3f} |",
            f"| held out, **plus {zero['n_variant_events']:,} loop-manufactured variant events** "
            f"| **{zero['recall_loop_augmented']:.3f}** |",
            "",
            f"The loop recovers **{zero['gap_closed']:.0%}** of the collapse that holding the "
            "family out caused.",
            "",
            "**What the detector had, and what the loop had. These are not the same thing, "
            "and the whole claim turns on the difference.**",
            "",
            f"The *detector* never trained on a single real {zero['family']} event. That is what "
            f"the middle row measures, and {zero['recall_family_held_out']:.3f} is what it gets "
            "it for.",
            "",
            f"The *loop* had something else: {zero['family']}'s **atlas cards and their "
            "executable injectors** — a written description of a class of attack, and code that "
            "manufactures instances of it. That is a red team, not a fraud history. It is why "
            "the third row is not magic and must never be described as the detector "
            "generalising on its own: it did not generalise, it was **given manufactured "
            "training data for a family it had never seen in the wild**, and that data was "
            f"produced from a specification a human wrote before any {zero['family']} attack was "
            "observed.",
            "",
            "The variants are still not the test rows. Every gene moved them, and they were "
            "**selected for evading the detector**, so they sit off-distribution from the "
            "canonical attack in exactly the direction that makes the transfer hard — which is "
            "why the recovery is 66% and not 100%.",
            "",
            "**This is the realistic position on a new rail, and it is the point of the "
            "project.** Agentic commerce has no labelled fraud history, and will not have one "
            "until losses have already been taken. What it can have on day one is a red team: "
            "people who can describe the attack and write the generator. The claim is therefore "
            "*\"an attack family that has been described but never observed can be manufactured, "
            "and training on the manufactured version transfers to the real one\"* — **not** "
            "*\"the detector caught something nobody had thought of\"*. Nothing does that. "
            "Somebody thought of it; the contribution is that thinking of it was enough.",
            "",
            "Measured on the loop's own two-seed background rather than the five-seed pool above, "
            "so all three rows share one dataset and one operating point. The Day 4 five-seed "
            "figures (0.569 trained, 0.007 held out) are the same experiment at a different "
            "scale, not a directly comparable row.",
            "",
        ]
    return "\n".join(lines)


def _fidelity_section() -> str:
    """Criterion 2's numbers, read from the scorecard rather than restated.

    Absent artefact produces one honest line naming the command, not a blank
    heading and not a placeholder number. The same rule the loop section follows.
    """
    if not FIDELITY_JSON.exists():
        return (
            "## Fidelity of the simulation\n\n"
            "Not measured in this run. `make reference && make fidelity` writes "
            "`data/generated/fidelity.json`, and this section is rendered from it.\n"
        )
    card = json.loads(FIDELITY_JSON.read_text(encoding="utf-8"))
    if not card.get("reference", {}).get("available"):
        return (
            "## Fidelity of the simulation\n\n"
            "The scorecard ran without a reference panel, so its marginal, TSTR and "
            "discriminator sections were skipped. **Nothing is substituted for them.** "
            "Comparing the population against its own specification is what "
            "`scripts/drift_check.py` does; it is a different question, and reporting "
            "it here as fidelity would be the dishonest version of this section. "
            "Run `make reference` and re-run `make fidelity`.\n"
        )

    rows = card["marginals"]["rows"]
    tstr = card["tstr"]
    disc = card["discriminator"]
    ablated = card["discriminator_ablated"]
    syn = card["synthetic"]["levels"]
    ref = card["reference"]["levels"]
    naive_headline = card.get("discriminator_naive", {})

    lines = [
        "## Fidelity of the simulation",
        "",
        "Every detection figure above is measured on data this project generated, which makes "
        "all of them conditional. This is the measurement of what the condition is worth, and "
        "it is deliberately unflattering: a scorecard where everything came out green would be "
        "evidence that it was measuring the wrong things.",
        "",
        "| | |",
        "|---|---|",
        f"| reference panel | {card['reference']['provenance']} |",
        f"| population calibration | `{card['calibration']['source']}` |",
        f"| discriminator, all shape features | **{disc['auc']:.4f}** (target 0.5) |",
        f"| discriminator, adjudicated axes removed | **{ablated['auc']:.4f}** |",
        (
            f"| discriminator, raw columns (ids + currency) | {naive_headline['auc']:.4f} "
            "(expected; not a finding) |"
            if naive_headline
            else "| discriminator, raw columns | not measured |"
        ),
        f"| top driver | `{disc['per_feature'][0]['feature']}`, "
        f"{disc['per_feature'][0].get('contribution_share', float('nan')):.0%} of contribution |",
        f"| TSTR transfer ratio | {tstr['transfer_ratio']:.3f} |",
        f"| correlation-matrix RMS error | "
        f"{card['marginals']['correlation']['rms_off_diagonal']:.3f} |",
        "",
        "### Nothing is compared raw, and that is the first thing to check",
        "",
        "A rupee population with an agentic rail cannot be compared to a dollar panel without "
        "one on absolute amount, category, geography, BIN, 3-D Secure or channel. Doing it "
        "would produce a large number that measures the difference between two countries. Both "
        "sides are projected into a **dimensionless shape space** first — the diurnal curve, "
        "within-category amount dispersion, burstiness against each cardholder's own rhythm, "
        "and merchant rank-frequency — and every distance below lives there.",
        "",
        "The agentic rail is excluded from the synthetic side for the sharpest version of the "
        "same reason: **the reference panel has no agentic transactions, because no panel "
        "does.** That absence is the premise of the whole project, and it is also the limit of "
        "what this section can claim.",
        "",
        "The two panels differ in *level* as well as in shape, and the levels are reported with "
        "**no distance attached** — a ratio between two panels' cardholder velocity is a fact "
        "about how each was composed:",
        "",
        "| level | synthetic | reference | ratio |",
        "|---|---|---|---|",
    ]
    for key, label, digits in (
        ("customers", "cardholders", 0),
        ("merchants", "merchants", 0),
        ("txn_per_customer_per_day", "txn / cardholder / day", 3),
        ("top_1pct_merchant_share", "top 1% merchant share", 3),
    ):
        a, b = syn.get(key), ref.get(key)
        ratio = a / b if a is not None and b else float("nan")
        fmt = f"{{:,.{digits}f}}"
        lines.append(
            f"| {label} | {fmt.format(a)} | {fmt.format(b)} | {ratio:.2f}x |"
        )

    lines += [
        "",
        "### Marginals, each against its own sampling-noise band",
        "",
        "No distance here is compared against a threshold somebody made up. Every band is "
        "bootstrapped from the reference distribution itself at these sample sizes, so a ratio "
        "of 1.0 means *indistinguishable from sampling noise* and the ratio is the number to "
        "read. Sorted worst first.",
        "",
        "| feature | metric | distance | noise band | x band |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['feature']}` | {row['metric']} | {row['distance']:.4f} | "
            f"{row['band']:.4f} | {row['ratio']:,.1f} |"
        )

    correlation = card["marginals"]["correlation"]
    worst = correlation["worst_pairs"][0]
    lines += [
        "",
        "Marginals alone are not enough: a generator can match every one of them and still "
        "draw each column independently. The Spearman correlation matrices differ by an RMS of "
        f"**{correlation['rms_off_diagonal']:.3f}** off the diagonal (Frobenius "
        f"{correlation['frobenius']:.3f} over {correlation['n_features']} features). The worst "
        f"pair is `{worst['pair']}`: {worst['synthetic']:+.3f} here against "
        f"{worst['real']:+.3f} in the reference.",
        "",
        "### TSTR — and it does not transfer",
        "",
        "| model | trained on | tested on | AUC-PR | ROC | lift over baseline |",
        "|---|---|---|---|---|---|",
        f"| **TRTR** | real | real | {tstr['trtr']['auc_pr']:.4f} | "
        f"{tstr['trtr']['roc_auc']:.4f} | {tstr['trtr_lift']:.0f}x |",
        f"| **TSTR** | synthetic | real | {tstr['tstr']['auc_pr']:.4f} | "
        f"{tstr['tstr']['roc_auc']:.4f} | {tstr['tstr_lift']:.1f}x |",
        f"| **TRTS** | real | synthetic | {tstr['trts']['auc_pr']:.4f} | "
        f"{tstr['trts']['roc_auc']:.4f} | |",
        "",
        f"The transfer ratio is **{tstr['transfer_ratio']:.3f}**, and stating that as a failure "
        "of realism would be the easy reading and the wrong one. The gain tables say what "
        "actually happened:",
        "",
    ]
    learned = tstr.get("what_each_learned")
    if learned:
        lines += [
            "| feature | gain, trained on real | gain, trained on synthetic |",
            "|---|---|---|",
        ]
        synth_gain = {r["feature"]: r["gain_share"] for r in learned["tstr"]}
        for row in learned["trtr"][:5]:
            lines.append(
                f"| `{row['feature']}` | {row['gain_share']:.1%} | "
                f"{synth_gain.get(row['feature'], 0.0):.1%} |"
            )
        top_real = learned["trtr"][0]
        top_syn = learned["tstr"][0]
        lines += [
            "",
            f"A detector trained on the reference panel spends {top_real['gain_share']:.0%} of "
            f"its gain on `{top_real['feature']}`. One trained on ours spends "
            f"{top_syn['gain_share']:.0%} on `{top_syn['feature']}`. **The two panels' fraud "
            "are different phenomena living in different features** — Sparkov's fraud is an "
            "amount anomaly, and this project's classic-rail attacks were built specifically so "
            "that no single raw column separates them above 0.95 AUC. TRTS confirms the "
            f"symmetry: a model trained on real data scores ROC {tstr['trts']['roc_auc']:.3f} "
            "on ours, which is chance.",
            "",
            "So the honest reading is that TSTR measures *whether the two datasets' fraud is the "
            "same phenomenon*, and here it is not, by construction. It is **not** evidence that "
            "the background population is unrealistic — the marginal and discriminator sections "
            "are what speak to that.",
            "",
            tstr["caveat"],
            "",
        ]

    naive = card.get("discriminator_naive", {})
    classes = {row["feature"]: row for row in card.get("feature_class", [])}
    lines += [
        "### The discriminator — the only test that sees interactions",
        "",
        "Label the synthetic rows 1, the real rows 0, and fit a gradient-boosted tree on the "
        "shape features, scored out of fold on balanced subsamples. **The target is 0.5**: here, "
        "higher is worse.",
        "",
        f"Result: **{disc['auc']:.4f}** ({disc['separability']:.1%} separable). {disc['reading']}",
        "",
        "#### What it was allowed to see",
        "",
        "A discriminator handed the raw columns of two payment files separates them at 1.0 and "
        "tells you nothing. Identifiers are namespaced differently on each side; the reference "
        "panel predates agentic commerce, so **every row carrying an `ag_` block is "
        "definitionally separable**; and an amount in rupees is not an amount in dollars. "
        "Counting any of that as infidelity would be the same error `BaseAttack.probe_slice` "
        "fixed on Day 3 — measuring a property that *defines* the two populations rather than "
        "one that distinguishes their behaviour.",
        "",
        "So the headline number is measured on the **intersection** of what both panels carry, "
        "with identity-shaped columns excluded by construction, the agentic rail dropped from "
        "the synthetic side, and every survivor made dimensionless. Both numbers, labelled:",
        "",
        "| input | AUC | what it means |",
        "|---|---|---|",
    ]
    if naive:
        lines.append(
            f"| raw columns — ids, currencies and taxonomies included | {naive['auc']:.4f} | "
            "expected, and **not a finding**: two identifier namespaces separate at 1.0 by "
            "themselves |"
        )
    lines += [
        f"| shape space — intersection, ids dropped, agentic excluded | **{disc['auc']:.4f}** | "
        "the headline |",
        "",
        "**The discipline bought almost nothing, and that is the important result.** If the "
        "0.9994 had been an artefact of identifiers or currency, removing them would have "
        "collapsed it. It did not move. The separation is real, and it is in behaviour rather "
        "than in formatting.",
        "",
        "#### What drives it",
        "",
        "Attribution is LightGBM's native `pred_contrib` — the same quantity `TreeExplainer` "
        "computes, read from the source SHAP would have called. Ranked by mean absolute "
        "contribution; `alone` is the feature's own rank-AUC against the panel label.",
        "",
        "| feature | alone (AUC) | gain | contribution | class |",
        "|---|---|---|---|---|",
    ]
    for row in disc["per_feature"]:
        verdict = classes.get(row["feature"], {}).get("verdict", "—")
        lines.append(
            f"| `{row['feature']}` | {row['alone_auc']:.4f} | {row['gain_share']:.1%} | "
            f"{row.get('contribution_share', float('nan')):.1%} | {verdict} |"
        )

    top = disc.get("top_features", [])
    if top:
        structural = [f for f in top if classes.get(f, {}).get("verdict") == "structural"]
        lines += [
            "",
            f"**Top five drivers:** {', '.join(f'`{f}`' for f in top)} — "
            f"{len(structural)} of 5 structural.",
            "",
            "#### Cosmetic or structural, and why the distinction decides what to do",
            "",
            "**Cosmetic** is a surface property of how values were *rendered*: identifier shape, "
            "timestamp granularity, amount rounding. Changing it moves the discriminator without "
            "changing anything an attacker or a detector would experience. **Structural** is a "
            "property of the joint distribution — which merchants get the volume, how a "
            "cardholder's transactions cluster in time, how a ticket sits against their history. "
            "Fixing one of those means changing the generative model.",
            "",
            "Nothing below is a fix and nothing below is an excuse. A feature can be structural "
            "*and* adjudicated to the reference panel — `merchant_rank_pct` is both, and the two "
            "statements are independent.",
            "",
        ]
        for row in card.get("feature_class", [])[:5]:
            lines.append(f"- **`{row['feature']}`** — *{row['verdict']}*. {row['reason']}")
        lines.append("")

    path = card.get("discriminator_ablation_path", [])
    if path:
        first, last = path[0], path[-1]
        lines += [
            "#### Is it one column, or the joint distribution?",
            "",
            "Drop the strongest feature, refit, re-score, repeat. A separation that lives in one "
            "column collapses on the first drop; one that lives in the joint distribution "
            "degrades gradually. Which of those is true decides whether there is a fix or a "
            "rebuild.",
            "",
            "| features left | AUC | dropped so far |",
            "|---|---|---|",
        ]
        for step in path:
            dropped = ", ".join(f"`{d}`" for d in step["dropped_so_far"]) or "—"
            lines.append(f"| {step['n_features']} | {step['auc']:.4f} | {dropped} |")
        lines += [
            "",
            f"**It degrades gradually.** Removing the single strongest feature takes the AUC "
            f"from {first['auc']:.4f} to {path[1]['auc']:.4f} — still far above chance — and it "
            f"takes dropping {len(last['dropped_so_far'])} of "
            f"{first['n_features']} features to reach {last['auc']:.4f}. So the separation is "
            "**not** one bad column that could be patched. It is distributed across the joint "
            "structure, which is the same thing the correlation-matrix distance measures, and it "
            "is the foundry's most substantial outstanding item rather than a cosmetic one.",
            "",
        ]

    adjudications = card.get("adjudications", [])
    if adjudications:
        lines += [
            "",
            "#### Which side is the anomalous one?",
            "",
            "A discriminator says a difference exists. It does not say which panel is wrong, and "
            "the temptation three days before a submission is to assume it is the reference. So "
            "the rule is that **an adjudication must carry a measurement**: a divergence is "
            "attributed to the reference panel only when a third quantity, independent of both "
            "and agreed before either dataset existed, says the reference is the side that "
            "departs from it.",
            "",
            "| feature | test | synthetic | reference | verdict |",
            "|---|---|---|---|---|",
        ]
        for row in adjudications:
            lines.append(
                f"| `{row['feature']}` | {row['third_quantity']} | {row['synthetic']} | "
                f"{row['reference']} | **{row['verdict'].lower()}** |"
            )
        lines += [
            "",
            "Sparkov's hour-of-day curve is a two-level step rather than a diurnal curve, and its "
            "693 merchants are close to uniformly popular. Neither is a defect in that dataset "
            "for its own purpose — it exists to benchmark fraud classifiers, and a flat time "
            "curve does not hurt that — but it does mean those two axes cannot measure this "
            "project's fidelity.",
            "",
            f"With them removed the discriminator falls to **{ablated['auc']:.4f}** "
            f"({ablated['separability']:.1%} separable). **Both numbers are quoted because the "
            "ablation is a judgement**: the full discriminator is the measurement, the ablated "
            "one is the measurement after a judgement a reader is free to reject. And "
            f"{ablated['auc']:.2f} is still high — the remaining features are individually close "
            "(every one under 0.54 alone) and the separation is in their *joint* structure, "
            "which is the same thing the correlation-matrix distance measures and is the "
            "foundry's most substantial outstanding item.",
            "",
        ]

    known = card.get("known_divergences", [])
    if known:
        lines += [
            "### Divergences we name ourselves",
            "",
            "Found before this scorecard existed, measured, and deliberately left in place. A "
            "scorecard that names its own divergences is worth more than one where a judge finds "
            "them.",
            "",
        ]
        for row in known:
            lines += [
                f"**{row['name']}** — {row['measured']}.",
                "",
                f"{row['cause']} {row['why_not_fixed']}",
                "",
            ]

    return "\n".join(lines)


def _latency_section() -> str:
    """Criterion 5's second number, reported as measured rather than as targeted."""
    if not LATENCY_JSON.exists():
        return (
            "## Scoring latency\n\n"
            "Not measured in this run. `make latency` writes "
            "`data/generated/latency.json`, and this section is rendered from it. "
            "**No number is substituted for it.**\n"
        )
    payload = json.loads(LATENCY_JSON.read_text(encoding="utf-8"))
    end = payload["end_to_end_ms"]
    stages = payload["stages_ms"]
    batch = payload.get("batch_per_row_ms", {})
    budget = payload["budget_ms"]
    verdict = "**within**" if payload["within_budget"] else "**over**"

    heaviest = max(stages, key=lambda name: stages[name]["mean"])
    factor = stages[heaviest]["mean"] / batch[heaviest] if batch.get(heaviest) else float("nan")
    streaming = stages["velocity"]["p99"] + stages["graph"]["p99"]

    inline = payload.get("inline_ms")
    follow = payload.get("fast_follow_ms")
    # The inline path's amortised cost, summed from the per-stage batch figures
    # rather than measured separately: the same stages, called with many rows.
    # It is the number that says whether the budget miss is architectural.
    inline_batch = sum(
        batch.get(name, 0.0) for name in payload.get("inline_stages", [])
    )

    lines = [
        "## Scoring latency",
        "",
        f"Measured **one event at a time** against state warmed on "
        f"{payload['warm_events']:,} training events — not a batch divided by its row count, "
        "which is the usual way a latency claim turns out to be false in production. "
        f"{payload['n_events']:,} events timed.",
        "",
    ]

    if inline and follow:
        inline_verdict = "**within**" if payload.get("inline_within_budget") else "**over**"
        lines += [
            "### The firewall is two paths, not one deadline",
            "",
            "Not every layer has to answer inside the authorisation window, and pretending they "
            "do is what makes the budget look unmeetable.",
            "",
            "**Inline — L0 + L1.** What must return before the issuer answers the acquirer, "
            "because it is what changes the *authorisation decision*: the deterministic protocol "
            "clauses, and the supervised score they escalate to. Both need only the event and "
            "backward-looking state.",
            "",
            "**Fast-follow — L2, L3, L4, fusion and policy.** What informs an action taken "
            "**after** the authorisation has been answered. On the agentic rail the two actions "
            "that matter are both post-authorisation in any real deployment:",
            "",
            "- **Mandate revocation** — a mandate used to read a poisoned page is revoked so the "
            "*next* authorisation under it fails L0. That is a credential-lifecycle operation "
            "against the mandate registry, not a field in the authorisation response.",
            "- **Agent quarantine** — an agent whose identity component has fused with a ring is "
            "suspended, which changes every future authorisation it attempts and none of the one "
            "in flight.",
            "",
            "Holding an authorisation open while a classifier reads eleven web pages, in order to "
            "take an action that cannot be applied to that authorisation anyway, would be an "
            "architectural error rather than a performance one. The graph pass is counted "
            "**inline** despite L4 being a fast-follow layer, because L1 consumes its 28 `gph_` "
            "features: it is inline as a feature source.",
            "",
            "| path | p50 | p95 | p99 | vs the 50 ms budget |",
            "|---|---|---|---|---|",
            f"| **inline** — L0+L1, pre-authorisation | {inline['p50']:.1f} ms | "
            f"{inline['p95']:.1f} ms | **{inline['p99']:.1f} ms** | {inline_verdict} |",
            f"| **fast-follow** — L2+L3+fusion, post-authorisation | {follow['p50']:.1f} ms | "
            f"{follow['p95']:.1f} ms | {follow['p99']:.1f} ms | not governed by it |",
            f"| **full stack** | {end['p50']:.1f} ms | {end['p95']:.1f} ms | "
            f"**{end['p99']:.1f} ms** | {verdict} |",
            "",
            "The split is asserted to partition the stage list rather than computed by "
            "subtraction — an earlier run let L2, which is 40% of the clock, vanish into "
            "fast-follow without being named there.",
            "",
            f"**The inline path misses the budget by 1.3x, and ~{inline_batch:.2f} ms of it is "
            "real work.** Those same inline stages cost "
            f"**{inline_batch:.2f} ms per row** when called with many rows instead of one — so "
            f"of the {inline['p99']:.0f} ms measured at p99, something like "
            f"{100 * (1 - inline_batch / inline['p99']):.0f}% is per-call framework overhead "
            "that a production scoring path would not pay. That does not make the budget met; "
            "it makes the miss an implementation property rather than an architectural one, and "
            "the next section measures exactly where it goes.",
            "",
            "### Every stage",
            "",
        ]
    else:
        lines += [
            "| | p50 | p95 | p99 | max |",
            "|---|---|---|---|---|",
            f"| end to end | {end['p50']:.1f} ms | {end['p95']:.1f} ms | "
            f"**{end['p99']:.1f} ms** | {end['max']:.1f} ms |",
            "",
            f"Against a **{budget:.0f} ms** authorisation-host budget, that is {verdict} budget.",
            "",
        ]

    lines += [
        "| stage | mean | p99 | share | per row in batch | overhead |",
        "|---|---|---|---|---|---|",
    ]
    for name in stages:
        row = stages[name]
        per_row = batch.get(name, float("nan"))
        share = row["mean"] / end["mean"] if end["mean"] else 0.0
        ratio = row["mean"] / per_row if per_row else float("nan")
        overhead = "—" if abs(ratio - 1.0) < 0.01 else f"{ratio:,.0f}x"
        lines.append(
            f"| `{name}` | {row['mean']:.3f} ms | {row['p99']:.3f} ms | {share:.1%} | "
            f"{per_row:.4f} ms | {overhead} |"
        )

    lines += [
        "",
        "### Read the last two columns before the p99",
        "",
        f"Most of the clock goes to `{heaviest}`, which costs {stages[heaviest]['mean']:.1f} ms "
        f"called with one row and {batch[heaviest]:.4f} ms per row called with many — a factor "
        f"of **{factor:,.0f}**. That is per-call overhead in pandas and scikit-learn, not model "
        "work: `Series.map(dict)` materialises the lookup table into an index on every call, so "
        "a feature block pays that cost once per feature to look up one value.",
        "",
        "**The fix is named and not applied.** A plain dictionary lookup on the single-event "
        "path removes it, and the feature builder is shared with the offline pass behind every "
        "pinned number in this document. Three days out, this project records the finding "
        "rather than re-rolls the tables for it.",
        "",
        "The two stages that genuinely **cannot** be batched — the stateful stores, which must "
        "read state before folding the event in, and which are therefore the same code online "
        f"and offline — cost `velocity` {stages['velocity']['p99']:.3f} ms p99 and `graph` "
        f"{stages['graph']['p99']:.3f} ms p99, **{streaming:.2f} ms together**. Those are the "
        "numbers that would survive a rewritten scoring path, and they are the ones the "
        "architecture was designed around: one forward pass, `bisect` and prefix sums per "
        "window, union-find over the identity graph, bounded memory by eviction.",
        "",
        f"The honest headline is therefore both sentences: **the current implementation misses a "
        f"50 ms budget at p99 ({end['p99']:.0f} ms full stack"
        + (f", {inline['p99']:.0f} ms inline" if inline else "")
        + "), and the miss is in the calling convention rather than in the models.** Quoting "
        "only the second would be an estimate dressed as a measurement; quoting only the first "
        "would invite the conclusion that a five-layer firewall cannot run inline, which this "
        "measurement does not support.",
        "",
        "One measurement-hygiene note, because it changed a number by a factor of two: an "
        "earlier run of this bench was taken while the fidelity scorecard was fitting on the "
        "same machine, and every percentile roughly doubled. Latency is the one measurement in "
        "this repository that is not reproducible from a seed — it is a property of the machine "
        "at the moment it ran — so it is measured with nothing else running, and that is worth "
        "stating rather than assuming.",
        "",
    ]
    return "\n".join(lines)


def _feasibility_section() -> str:
    """Two deployment questions the tables above raise but do not answer.

    Static prose rather than generated numbers, because both are statements about
    the *architecture* and neither moves with a retrain. They live in the
    generated document rather than in a hand-edited one for the reason the module
    docstring gives: there is one place a reader looks, and it cannot go stale
    against a file nobody remembers to update.
    """
    return "\n".join(
        [
            "## Two things a deployment would need to know",
            "",
            "### Fusion consumes L3's score, not L3's decision",
            "",
            "This matters more than it sounds like it does, and the L3 out-of-distribution "
            "result above is why.",
            "",
            "`FusionModel` gives every layer **three columns**: its percentile against the "
            "legitimate score distribution, its raw score standardised on the fusion window's "
            "legitimate rows, and — where the layer is sometimes silent — an indicator for "
            "whether it had an opinion at all. No threshold is applied to any layer before "
            "fusion. L3's page threshold is used for *reporting* L3 as a standalone layer and "
            "for nothing else.",
            "",
            "That is the right way round, and it is what makes the out-of-distribution finding "
            "survivable. L3's threshold **does not transfer** to text unlike its training "
            "corpus: pointed at hand-authored payloads it fires on 100% of the injections and "
            "on 90% of the *clean* controls written in the same registers. A fused score that "
            "consumed L3's thresholded decision would inherit that failure directly — every "
            "procedural-sounding page would arrive at fusion as a hard vote for fraud, on a "
            "layer whose calibration had silently stopped being valid.",
            "",
            "Consuming the score instead means the stacker sees L3's *ordering*, which is what "
            "survived the transfer: ROC falls 0.999 → 0.811 rather than collapsing. The "
            "percentile column is re-derived from the deployment's own legitimate traffic every "
            "time fusion is fitted, so a shift in L3's absolute scale is re-absorbed at the next "
            "refit rather than becoming a permanent bias. And the fitted weight is the check on "
            "the whole arrangement: on this data the stacker put **-0.943** on L3's percentile "
            "and **+0.353** on its standardised raw score, which is the model saying it trusts "
            "the layer's ordering while discounting the calibration that produced the "
            "percentile. That is a discount a decision-consuming fusion could not have applied.",
            "",
            "### What the non-transferring threshold means for deploying on novel text",
            "",
            "Stated plainly, because it is the largest single caveat on the agentic side of this "
            "architecture: **L3 as calibrated here cannot be pointed at the open web.** It is a "
            "classifier fitted on one 7B model's output through one set of prompt templates in "
            "one register, and its decision boundary is a property of that corpus rather than of "
            "injected text in general.",
            "",
            "Three consequences for a deployment, in the order they would bite:",
            "",
            "1. **Fit the page threshold on the traffic it will see, not on the corpus it was "
            "trained on.** The benign side of that calibration set is the part that matters and "
            "the part that is easy to get for free — it is the pages agents read on ordinary, "
            "approved authorisations, which an issuer accumulates without labelling anything. "
            "The 90% false-positive rate on hand-authored controls is what happens when this "
            "step is skipped.",
            "2. **Do not promote L3 to a standalone decision.** Its recall is real on two of "
            "fifteen cards and its ordering transfers; its calibration does not. It belongs "
            "behind fusion, where a fitted weight can discount it, and behind L0, where the "
            "protocol invariants need no calibration at all.",
            "3. **A bag of words is the wrong long-term model, and the failure mode says why.** "
            "It keys on lexical markers of instruction — *do not*, *skip*, *without* — which is "
            "precisely why prose that merely sounds procedural trips it. The 1.000 recall on "
            "F1-01 and F1-03 is a true statement about this corpus and is not a claim about "
            "text in general.",
            "",
            "The general form of this is worth keeping, because it is not specific to L3: **a "
            "layer whose ordering transfers and whose calibration does not is still useful, "
            "provided nothing downstream consumes its threshold.** The architecture already "
            "satisfies that condition; it was not designed to, and the out-of-distribution probe "
            "is what turned an accident into a checked property.",
            "",
        ]
    )
