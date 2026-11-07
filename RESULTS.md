# MANTIS — Detection results

*Day 5. Schema v1.1.0. Generated 2026-08-25 by `python -m mantis.defense`.*

## The operating point, and why it is a curve

> **Every recall in this document is measured at a threshold placed so that a fixed share of legitimate test traffic is flagged.** The headline share is **0.1%**.

No accuracy figures appear anywhere here, and that is deliberate: at ~1% prevalence a model that approves everything is 99% accurate. The metrics reported are **AUC-PR** and **recall at a fixed FPR**, the second always with its realised false-positive rate attached.

Day 4 quoted one operating point. Day 5 quotes 0.1%, 0.5%, 1.0% — a curve rather than a point, because one number at one budget is something a reader has to trust you did not pick, and three is a shape. 0.1% stays the headline because it is the tightest, and the tightest is the one an issuer can actually staff. 1.0% is roughly the top of what a review queue absorbs, which is why the curve stops there.

Each model variant gets its own threshold, because each has its own score distribution. What is held constant across every column of every table is the false-positive rate — the quantity an issuer budgets, and the only thing that makes two columns comparable.

### Two kinds of recall, both labelled

Day 4 reported **event-level** recall only, and event-level recall is the wrong question for half of this atlas. A mule ring that runs 40 authorisations and is flagged on 3 of them scores 7.5% event-level and is **caught**: one alert opens a case, and the case takes the ring. So every layer is also reported at **campaign level** — was the campaign flagged at all, and on which of its events.

Neither is the real number. Event-level flatters a layer that fires on every event of an obvious attack; campaign-level flatters a layer that fires once on something subtle. They appear side by side, always, with their names on them.

## The evaluation dataset

5 independently generated worlds, seeds `1337, 7, 11, 23, 41`, pooled.

| | |
|---|---|
| events | 1,010,600 |
| fraud | 10,600 (1.0489%) |
| train / test | 707,420 / 303,180, **time-based** split at the 70% quantile |
| features | 232 (of which 28 are the new `gph_` entity-graph block) |

Pooling is not resampling one file five times: each seed regenerates the whole population — its own customers, merchants, devices, agents and calendar — so the variance being averaged over is generator variance. Identifiers are namespaced per seed before concatenation, because letting `cus-00042` from two seeds collide would fuse two people's histories inside every velocity, entity and graph feature.

## Layer performance

| layer | AUC-PR | ROC-AUC | recall@0.1% | recall@0.5% | recall@1.0% | realised FPR | campaign recall | first alert |
|---|---|---|---|---|---|---|---|---|
| **L1** GBDT, supervised, time-split, isotonic-calibrated | 0.5903 | 0.9248 | 0.450 | 0.572 | 0.623 | 0.1000% | 0.916 (120/131) | event 3 of 23 |
| **L2** isolation forest on events, legitimate traffic only — **residual monitor** | 0.0151 | 0.5785 | 0.002 | 0.009 | 0.021 | 0.1000% | 0.031 (4/131) | event 12 of 23 |
| **L2e** isolation forest on **entity** aggregates — the time-boxed experiment | 0.0098 | 0.4421 | 0.000 | 0.003 | 0.005 | 0.1881% | 0.008 (1/131) | event 1 of 23 |
| **L3** page classifier over ingested text, **fitted on no transaction labels** | 0.1687 | 0.6919 | 0.138 | 0.149 | 0.167 | 0.1024% | 0.214 (28/131) | event 1 of 23 |
| **fused** weighted logistic stacker over the four | 0.5881 | 0.9119 | 0.483 | 0.580 | 0.606 | 0.1000% | 0.908 (119/131) | event 2 of 23 |

Baseline precision (a coin flip) is the prevalence: 1.0941%. AUC-PR is only interpretable against it.

**Fusion is now better than L1 alone** (0.450 → 0.483 at 0.1% FPR). On Day 4 it was worse — 0.361 → 0.286 — because the layers were combined with an *unweighted* noisy-OR, which gave a near-random L2 equal say inside a fixed false-positive budget. The fix was not to delete L2 but to fit the weights on a slice of the training window none of the base layers had seen, and let the stacker discount it. The coefficients below are what it decided.

### What the fusion learned

| layer | weight on the percentile | weight on the raw score | weight on "had an opinion" |
|---|---|---|---|
| L1 | +5.723 | +0.381 | n/a |
| L2 | +0.789 | -0.225 | n/a |
| L2e | +0.428 | +0.233 | n/a |
| L3 | -0.943 | +0.353 | -0.268 |

The coefficients are the interesting output, more than the fused recall is. A weight near zero is the stacker saying that layer carries no information the others do not already have — which is a cleaner statement than any table of recalls, and it is how the Day 4 problem got fixed without anyone hand-tuning a number.

### Per rail, because the headline is partly measuring "is this agentic"

Fraud is concentrated on the agentic rail by design — 15% of volume carrying 51% of the fraud, a 5.7x concentration. A single number across both rails is therefore partly reading a field the issuer gets free off the authorisation message.

| rail | n positive | AUC-PR | recall@0.1%FPR |
|---|---|---|---|
| agentic | 1,702 | 0.8088 | 0.606 |
| classic | 1,615 | 0.3912 | 0.318 |

## Leave one family out — the headline experiment

For each family, L1 is retrained with that family **entirely removed from the training set** and then asked to catch it in the test set anyway. L2 never sees any attack at all, and L3 never sees a transaction, so neither of their columns changes with the hold-out.

| family | n_pos | L1 (trained WITH it) | L1 (family HELD OUT) | L2 | L3 | fused | fused, held out |
|---|---|---|---|---|---|---|---|
| **F1** | 1,356 | 0.681 | 0.004 | 0.005 | 0.338 | 0.786 | 0.011 |
| **F2** | 433 | 0.176 | 0.016 | 0.000 | 0.000 | 0.157 | 0.014 |
| **F3** | 158 | 0.595 | 0.405 | 0.000 | 0.000 | 0.544 | 0.361 |
| **F4** | 567 | 0.086 | 0.028 | 0.000 | 0.002 | 0.086 | 0.023 |
| **F6** | 803 | 0.437 | 0.137 | 0.000 | 0.000 | 0.415 | 0.127 |
| *mean* | | *0.395* | *0.118* | *0.001* | *0.068* | *0.398* | *0.107* |

**Mean event-level recall lost when a family is held out of training: +0.277** (39.5% → 11.8%).

### The same experiment at campaign level

| family | campaigns | median size | fused (with) | fused (held out) | first alert at event | elapsed before alert |
|---|---|---|---|---|---|---|
| **F1** | 60 | 22 | 1.000 | 0.183 | 1 | 0% |
| **F2** | 14 | 32 | 0.714 | 0.214 | 8 | 24% |
| **F3** | 9 | 17 | 0.889 | 1.000 | 3 | 9% |
| **F4** | 22 | 23 | 0.818 | 0.227 | 12 | 54% |
| **F6** | 26 | 29 | 0.885 | 0.615 | 3 | 9% |

Read the last two columns together with the first: a campaign caught on its third event of forty is a case opened before most of the money moved. A campaign caught on its thirty-fifth is a post-mortem.

### Reading it

**Supervised detection collapses on attacks it has never seen.** Held-out recall averages 11.8% against 39.5% when the family is in training. This is the expected and the important result: an L1 trained on a labelled history is a detector for that history.

**The unsupervised layer does not rescue it.** L2 averages 0.1% at the same operating point. Day 5 stopped treating that as a gap to be closed and reframed it as the finding it is — see the next two sections. The architecture's answer to an unseen attack is now **L0's protocol invariants**, which need no training data because a violated mandate is a broken contract rather than an outlier, and **the closed loop**, which manufactures the attack before an attacker does.

## The closed loop — the other half of the zero-day answer

An evolutionary adversary mutates the **operational parameters** of a known attack — pacing, ring fan-out, device rotation, ticket size, how many injected pages the agent reads — and selects on **evasion x payoff** against the live detector. Between rounds the detector retrains on everything the arena has produced. That is the loop an operator runs when their attempts start getting declined, except that here the defender runs it first.

### The evasion curve

| generation | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| mean evasion | 0.626 | 0.334 | 0.338 | 0.363 | 0.381 |
| max evasion | 0.998 | 0.797 | 0.777 | 0.767 | 0.777 |

Evasion falls from **0.626 to 0.381** over 5 generations. Almost all of that fall happens at the **first** retrain — generation 0 is scored against a detector that has never seen a mutated variant, and generation 1 against one that has.

After that the curve **rebounds**, from a floor of 0.334 at generation 1 back to 0.381, and that is the more interesting half of the shape. It is what a real arms race looks like: the defender's first response is worth far more than any subsequent one, and the adversary then claws back a fraction of it by finding a corner of the gene space the retrain has not covered. A curve that fell monotonically to zero would be a curve to distrust — it would mean the adversary had stopped searching, which with a mutation operator that never stops would mean the search space was too small to be interesting.

The claim the chart supports is therefore the bounded one: **retraining on manufactured variants cuts evasion by 39% and holds it down**, not that it drives evasion to zero. Nothing drives evasion to zero.

Full per-generation detail, the surviving genomes and which genes moved are in `data/generated/arena.json`.

6 genuinely mutated variant(s) survived three or more consecutive rounds against a retraining detector and were written back to `mantis/atlas/discovered/` as validated attack cards with `discovered_by: adversarial_loop`. They live beside the atlas rather than inside it: `mantis/atlas/cards/` is the frozen 42, and the implemented count is a ratchet that moves only when an injector lands. Each card ships a `.genome.json` sidecar, so a variant is reproducible rather than merely described.

**3 further survivor(s) were the *unmutated* attack** — every gene at its default — and they are **not** written back. Each card's population is seeded with an identity genome so the curve carries its own no-evolution reference row, and on cards the detector is already weak at, that individual simply wins: F2-16 evades **77.8%** of decisions without being mutated at all. That is a result about the parent card, not a discovery, and recording it as one would be the exact overclaim the atlas ratchet exists to prevent. It also points at where the next detection work is: F1-01, F1-05, F2-16.

### The zero-day demonstration

This is the comparison the submission's argument rests on.

| detector | recall@0.1%FPR on the 544 real F1 test events |
|---|---|
| trained **with** family F1 | 0.811 |
| family F1 **held out** of training | 0.013 |
| held out, **plus 4,442 loop-manufactured variant events** | **0.539** |

The loop recovers **66%** of the collapse that holding the family out caused.

**What the detector had, and what the loop had. These are not the same thing, and the whole claim turns on the difference.**

The *detector* never trained on a single real F1 event. That is what the middle row measures, and 0.013 is what it gets it for.

The *loop* had something else: F1's **atlas cards and their executable injectors** — a written description of a class of attack, and code that manufactures instances of it. That is a red team, not a fraud history. It is why the third row is not magic and must never be described as the detector generalising on its own: it did not generalise, it was **given manufactured training data for a family it had never seen in the wild**, and that data was produced from a specification a human wrote before any F1 attack was observed.

The variants are still not the test rows. Every gene moved them, and they were **selected for evading the detector**, so they sit off-distribution from the canonical attack in exactly the direction that makes the transfer hard — which is why the recovery is 66% and not 100%.

**This is the realistic position on a new rail, and it is the point of the project.** Agentic commerce has no labelled fraud history, and will not have one until losses have already been taken. What it can have on day one is a red team: people who can describe the attack and write the generator. The claim is therefore *"an attack family that has been described but never observed can be manufactured, and training on the manufactured version transfers to the real one"* — **not** *"the detector caught something nobody had thought of"*. Nothing does that. Somebody thought of it; the contribution is that thinking of it was enough.

Measured on the loop's own two-seed background rather than the five-seed pool above, so all three rows share one dataset and one operating point. The Day 4 five-seed figures (0.569 trained, 0.007 held out) are the same experiment at a different scale, not a directly comparable row.


## L3, and why a near-perfect number here is honest

L3 classifies **a page**, and an event's score is the worst page its agent read. It is fitted on the committed content corpus using each artefact's own `injected` flag — **it never sees `is_fraud`, and `L3Model.fit` has no `y` parameter to pass one to**. That is why it sits with L0 in the reframed architecture rather than with L1: it works on an attack it has never seen in the payment data, because the thing it was trained on is not payment data.

| card | n_pos | recall | on unseen *phrasing* | n unseen | on an unseen *kind* |
|---|---|---|---|---|---|
| F1-01 | 249 | 1.000 | 1.000 | 134 | 1.000 |
| F1-02 | 205 | 0.020 | n/a | 0 | 0.020 |
| F1-03 | 201 | 1.000 | 1.000 | 93 | 1.000 |
| F1-05 | 174 | 0.017 | n/a | 0 | 0.017 |

**The injected instruction *is* the attack.** F1-01 is defined as an agent acting on text that told it to change the cart, and F1-03 as one acting on a refund request that told it to skip verification. A classifier reading that text and finding the instruction is not cheating and is not leakage — it is the detection working. The comparison that makes it concrete is L1's: on F1-01, L3 reaches 1.000 reading the text while L1 reaches 0.759 off 232 tabular features, because the tabular trace of "the agent read a bad page" is faint and the page itself is not.

What would be dishonest is claiming that generalises when it does not, so the last two columns are the ones the writeup may lean on. **Unseen phrasing**: the highest-numbered variant of every adversarial kind is withheld from the vocabulary and from training. **Unseen kind**: every `refund_ticket` specimen is withheld — all of them — so F1-03 is scored on an injection type the classifier has never seen in any wording.

Measured directly on the withheld texts themselves: 34 artefacts that were never in the vocabulary or the training set score P(injected) between 0.24 and 0.91, median 0.76. If the layer had memorised rather than learned, these would sit with the benign artefacts.

L3's *overall* recall is low, and that is correct rather than disappointing: it has an opinion about two of the fifteen cards and no opinion at all about the classic rail, where there is no provenance chain to read. A layer that scored highly on everything would be a layer reading something other than the text.

### The harder test: text L3 did not come from

Both hold-outs above are still drawn from `data/cache/content/` — one 7B model, one set of prompt templates, one register. A classifier can generalise perfectly across the variants of a corpus and still be reading the corpus. So `scripts/l3_ood.py` scores 18 hand-authored injection payloads and 10 hand-authored **benign controls in the same registers** — HTML comments, YAML, a fake system banner, transliterated Hindi-English, shouting, txt-speak, Cyrillic homoglyphs — written by a different model from the one that authored the corpus, and committed before they were scored.

| | recall | FP on controls | ROC | n+ / n- |
|---|---|---|---|---|
| in distribution (the corpus) | 0.917 | 0.000 | 0.999 | 96 / 138 |
| **out of distribution** | 1.000 | **0.900** | **0.811** | 18 / 10 |

**Read the second column before the first.** L3 fires on 100% of the novel injections — and on 90% of the *clean* pages written in the same registers. The recall cell is therefore meaningless on its own, and this is exactly why the benign controls were authored alongside the payloads rather than afterwards: without them this table would have read as a triumph.

**L3's decision threshold does not transfer.** Calibrated on one corpus and pointed at text unlike it, the layer is a false-positive machine. What survives is the *ordering*: ROC 0.999 → 0.811. Injected pages still rank above clean ones by the same author in the same register, and moving the threshold above the worst control recovers 50% recall — an oracle number, since that threshold has seen the answer, but enough to locate the defect in **calibration** rather than in an absence of signal.

**The named fix, not done today.** The page threshold is fitted on one corpus and must instead be fitted on benign text drawn from the traffic it will actually see. Longer term, a bag of words is the wrong model: it keys on lexical markers of instruction — *do not*, *skip*, *without* — which is why prose that merely sounds procedural trips it. The 1.000 in the table above is a real number about this corpus and **not** a claim about the open web.

## The negative result worth publishing

> **Attacks built to be distributionally faithful are, by construction, invisible to distributional anomaly detection.**

L2 scores events: AUC-PR 0.0151, ROC 0.5785, recall 0.002 at 0.1000%. The stated reason it *should* have worked better is a ring — every event in a mule network is bland, but the entity is not. So Day 5 tested exactly that, time-boxed to thirty minutes: **L2e**, an isolation forest over entity aggregates rather than event rows, scoring customers and merchants instead of authorisations. Result: AUC-PR 0.0098, ROC 0.4421, recall 0.000 at 0.1881%.

It did not move — it went **backwards**. A ROC of 0.4421 is *below chance*, which is a stronger statement than "weak": entity aggregates are mildly **anti-correlated** with fraud on this data. The reason is the foundry's own realism discipline pointed at the entity level: attacks ride established customers and busy merchants by construction, while the genuinely unusual entities in this population are ordinary people with three transactions. An outlier detector at entity level finds the quiet, and the quiet is innocent. The negative result is worth more than the layer would have been. L2e is also *generous* to the hypothesis in a way that has to be stated: an entity's vector is aggregated over the whole scoring window, so the score attached to that entity's first event was computed with knowledge of their last. That is a legitimate deployment mode — a nightly entity-risk queue, which is what AML and merchant-monitoring teams actually run — but it is not an authorisation scorer, and it still did not work.

**Our own fidelity work caused this.** Every foundry decision pushed the attacks toward the legitimate manifold: clone real background rows, resample amounts inside the target MCC's own empirical band, redraw the hour of day from the population's diurnal curve, widen three legitimate tails specifically so an attack would not be free, keep provenance planting length-preserving. The Day 2 separability gate is literally a rule forbidding any single raw column from separating an attack above 0.95 AUC. An isolation forest measures distance from that manifold; we spent two days minimising it.

And that is the property real agentic fraud has. An agent paying with a validly-signed mandate, on a real cardholder's real device, for a plausible amount at a real merchant, **is** legitimate in every marginal. The fraud lives in the *intent* — which is L3, the text — and in the *relations* — which is L4, the graph — not in the marginals. That is why "just run an autoencoder on it" is not an answer to agentic fraud, and it is why the two layers Day 5 built are the two that read intent and relations.

**Corollary, and it is a sharp one:** a fidelity scorecard and an anomaly-detection recall number are in tension *by construction*. A project reporting both as high is reporting one of them wrongly.

What survives for L2 is the narrow claim, and it is real: its recall is completely **unaffected** by whether an attack was in training, which is a property no supervised layer has. Its job in the architecture is now **residual monitor and drift canary** — "has the shape of legitimate traffic moved" — and no table in this repo presents it as a detector.

## Per attack card

At the same 0.1% FPR operating point. The last column is campaign-level: the share of that card's campaigns in which **at least one** event was flagged by the fused score.

| card | n_pos | L1 | L2 | L2e | L3 | fused | campaigns caught |
|---|---|---|---|---|---|---|---|
| F1-01 | 249 | 0.759 | 0.000 | 0.000 | 1.000 | 0.984 | 1.000 |
| F1-02 | 205 | 0.693 | 0.000 | 0.000 | 0.020 | 0.688 | 1.000 |
| F1-03 | 201 | 0.443 | 0.010 | 0.000 | 1.000 | 0.940 | 1.000 |
| F1-04 | 164 | 0.982 | 0.000 | 0.000 | 0.006 | 0.982 | 1.000 |
| F1-05 | 174 | 0.868 | 0.000 | 0.000 | 0.017 | 0.839 | 1.000 |
| F1-09 | 202 | 0.356 | 0.000 | 0.000 | 0.000 | 0.312 | 1.000 |
| F1-10 | 161 | 0.745 | 0.031 | 0.000 | 0.000 | 0.752 | 1.000 |
| F2-13 | 185 | 0.232 | 0.000 | 0.000 | 0.000 | 0.205 | 1.000 |
| F2-16 | 248 | 0.133 | 0.000 | 0.000 | 0.000 | 0.121 | 0.429 |
| F3-19 | 158 | 0.595 | 0.000 | 0.000 | 0.000 | 0.544 | 0.889 |
| F4-27 | 336 | 0.036 | 0.000 | 0.000 | 0.003 | 0.033 | 0.875 |
| F4-28 | 231 | 0.160 | 0.000 | 0.004 | 0.000 | 0.165 | 0.786 |
| F6-38 | 340 | 0.632 | 0.000 | 0.000 | 0.000 | 0.618 | 1.000 |
| F6-39 | 252 | 0.139 | 0.000 | 0.000 | 0.000 | 0.135 | 0.700 |
| F6-40 | 211 | 0.479 | 0.000 | 0.000 | 0.000 | 0.422 | 1.000 |

## The decision layer

A score is not an action. The fused score maps to one of four responses, with each boundary placed at a false-positive budget on legitimate traffic rather than at a hard-coded score — so a retrain re-prices nothing. Over the test window:

| decision | events | share |
|---|---|---|
| approve | 298,170 | 98.348% |
| challenge | 1,515 | 0.500% |
| review | 1,593 | 0.525% |
| decline | 1,902 | 0.627% |

A deterministic L0 clause firing overrides all four, because "the mandate had expired" is a defensible thing to tell a cardholder and "the ensemble scored 0.83" is not.

## What an alert actually says

Per-event attribution comes from LightGBM's own `pred_contrib`, not from SHAP. For a tree ensemble that is the *same* computation — `TreeExplainer` calls into LightGBM for it — without a wrapper on the scoring path. The three highest-scoring test events, with the features that put them there (contributions are in log-odds of the raw margin, which is what the ranking, and therefore the alert, is made of):

```
  #1  score 1.0000  upi_p2m  4,000 INR  [F3-19]
        + 3.289  mcc                                    = 6012
        + 1.503  ent_merchant_customers                 = absent
        + 1.477  ent_customer_n_events                  = 14
        + 1.129  vel_customer_count_24h                 = 4
        + 0.993  vel_customer_seconds_since_prior       = 2,251
        + 0.771  ent_merchant_refund_ratio              = absent

  #2  score 1.0000  agentic  6,502 INR  [F1-02]
        + 5.285  mnd_mcc_in_scope                       = 0
        + 2.238  mnd_amount_over_ceiling                = 1.192
        + 1.782  mnd_ceiling_breached                   = 1
        + 0.637  ent_mcc_amount_z                       = 1.422
        + 0.552  mnd_deliberation_residual_z            = -1.884
        + 0.509  vel_agent_amount_vs_mean               = 2.072

  #3  score 1.0000  agentic  1,358 INR  [F1-05]
        + 2.303  vel_mandate_hash_count_1h              = 5
        + 1.755  gph_agent_customers                    = 11
        + 1.441  gph_new_edge_share                     = 0.6667
        + 1.042  vel_bin_seconds_since_prior            = 0
        + 0.840  vel_agent_seconds_since_prior          = 0
        + 0.763  ent_customer_n_events                  = 21
```

## The feature that was too good, ablated

`mnd_deliberation_residual_z` — the residual of an agent's deliberation latency against what a ticket that size deserves — separates F1-01 at **0.99 AUC on its own**. That is above the foundry's own 0.95 separability gate, and the gate never saw it: the gate probes **raw columns**, and this is a derived residual.

It has not been silently removed and it has not been silently kept:

| family | recall with the feature | recall without it | delta |
|---|---|---|---|
| F1 | 0.681 | 0.664 | -0.018 |
| F2 | 0.176 | 0.173 | -0.002 |
| F3 | 0.595 | 0.538 | -0.057 |
| F4 | 0.086 | 0.085 | -0.002 |
| F6 | 0.437 | 0.437 | +0.000 |

## What L1 actually uses

| feature | gain share |
|---|---|
| `mcc` | 5.68% |
| `mnd_mcc_in_scope` | 4.98% |
| `ent_merchant_customers` | 4.77% |
| `mnd_amount_over_ceiling` | 4.06% |
| `ent_mcc_amount_z` | 3.28% |
| `ent_customer_n_events` | 2.98% |
| `vel_agent_amount_vs_mean` | 2.98% |
| `txn_round_score` | 2.88% |
| `ent_merchant_refund_ratio` | 2.85% |
| `channel` | 2.80% |
| `mnd_deliberation_residual_z` | 2.37% |
| `gph_device_customers` | 2.36% |
| `vel_merchant_count_7d` | 2.36% |
| `ent_merchant_n_events` | 2.26% |
| `vel_mandate_hash_count_1h` | 2.09% |

## Fidelity of the simulation

Every detection figure above is measured on data this project generated, which makes all of them conditional. This is the measurement of what the condition is worth, and it is deliberately unflattering: a scorecard where everything came out green would be evidence that it was measuring the wrong things.

| | |
|---|---|
| reference panel | Kaggle kartik2112/fraud-detection (Sparkov): fraudTrain.csv, fraudTest.csv. 1,852,394 rows, cut to 200,051 over 2019-12-17..2020-03-15. US panel, USD, itself synthetic - see mantis/foundry/fidelity/real.py. |
| population calibration | `indian-market-priors` |
| discriminator, all shape features | **0.9994** (target 0.5) |
| discriminator, adjudicated axes removed | **0.8399** |
| discriminator, raw columns (ids + currency) | 0.9999 (expected; not a finding) |
| top driver | `merchant_rank_pct`, 71% of contribution |
| TSTR transfer ratio | 0.030 |
| correlation-matrix RMS error | 0.076 |

### Nothing is compared raw, and that is the first thing to check

A rupee population with an agentic rail cannot be compared to a dollar panel without one on absolute amount, category, geography, BIN, 3-D Secure or channel. Doing it would produce a large number that measures the difference between two countries. Both sides are projected into a **dimensionless shape space** first — the diurnal curve, within-category amount dispersion, burstiness against each cardholder's own rhythm, and merchant rank-frequency — and every distance below lives there.

The agentic rail is excluded from the synthetic side for the sharpest version of the same reason: **the reference panel has no agentic transactions, because no panel does.** That absence is the premise of the whole project, and it is also the limit of what this section can claim.

The two panels differ in *level* as well as in shape, and the levels are reported with **no distance attached** — a ratio between two panels' cardholder velocity is a fact about how each was composed:

| level | synthetic | reference | ratio |
|---|---|---|---|
| cardholders | 4,947 | 918 | 5.39x |
| merchants | 10,590 | 693 | 15.28x |
| txn / cardholder / day | 0.369 | 2.449 | 0.15x |
| top 1% merchant share | 0.322 | 0.019 | 16.58x |

### Marginals, each against its own sampling-noise band

No distance here is compared against a threshold somebody made up. Every band is bootstrapped from the reference distribution itself at these sample sizes, so a ratio of 1.0 means *indistinguishable from sampling noise* and the ratio is the number to read. Sorted worst first.

| feature | metric | distance | noise band | x band |
|---|---|---|---|---|
| `hour` | JSD | 0.0721 | 0.0000 | 1,649.1 |
| `dow` | JSD | 0.0115 | 0.0000 | 643.0 |
| `burst_1h` | JSD | 0.0026 | 0.0000 | 438.2 |
| `merchant_rank_pct` | KS | 0.5234 | 0.0054 | 96.1 |
| `category_shift` | JSD | 0.0005 | 0.0000 | 74.7 |
| `gap_ratio_log` | KS | 0.0909 | 0.0054 | 16.7 |
| `amount_vs_customer` | KS | 0.0828 | 0.0054 | 15.2 |
| `log_amount_z` | KS | 0.0325 | 0.0054 | 6.0 |

Marginals alone are not enough: a generator can match every one of them and still draw each column independently. The Spearman correlation matrices differ by an RMS of **0.076** off the diagonal (Frobenius 0.568 over 8 features). The worst pair is `amount_vs_customer x merchant_rank_pct`: -0.004 here against -0.259 in the reference.

### TSTR — and it does not transfer

| model | trained on | tested on | AUC-PR | ROC | lift over baseline |
|---|---|---|---|---|---|
| **TRTR** | real | real | 0.8353 | 0.9878 | 121x |
| **TSTR** | synthetic | real | 0.0249 | 0.6385 | 3.6x |
| **TRTS** | real | synthetic | 0.0098 | 0.5175 | |

The transfer ratio is **0.030**, and stating that as a failure of realism would be the easy reading and the wrong one. The gain tables say what actually happened:

| feature | gain, trained on real | gain, trained on synthetic |
|---|---|---|
| `log_amount_z` | 57.6% | 17.1% |
| `amount_vs_customer` | 16.5% | 16.0% |
| `hour` | 13.0% | 6.7% |
| `merchant_rank_pct` | 6.3% | 35.8% |
| `gap_ratio_log` | 4.9% | 15.8% |

A detector trained on the reference panel spends 58% of its gain on `log_amount_z`. One trained on ours spends 36% on `merchant_rank_pct`. **The two panels' fraud are different phenomena living in different features** — Sparkov's fraud is an amount anomaly, and this project's classic-rail attacks were built specifically so that no single raw column separates them above 0.95 AUC. TRTS confirms the symmetry: a model trained on real data scores ROC 0.517 on ours, which is chance.

So the honest reading is that TSTR measures *whether the two datasets' fraud is the same phenomenon*, and here it is not, by construction. It is **not** evidence that the background population is unrealistic — the marginal and discriminator sections are what speak to that.

TSTR is measured on classic-rail fraud only. The reference panel contains no agentic transactions, so no number here is evidence about the agentic attacks.

### The discriminator — the only test that sees interactions

Label the synthetic rows 1, the real rows 0, and fit a gradient-boosted tree on the shape features, scored out of fold on balanced subsamples. **The target is 0.5**: here, higher is worse.

Result: **0.9994** (99.9% separable). Separable at 100%. On these features the synthetic population is not passing for the reference panel, and the scorecard says so.

#### What it was allowed to see

A discriminator handed the raw columns of two payment files separates them at 1.0 and tells you nothing. Identifiers are namespaced differently on each side; the reference panel predates agentic commerce, so **every row carrying an `ag_` block is definitionally separable**; and an amount in rupees is not an amount in dollars. Counting any of that as infidelity would be the same error `BaseAttack.probe_slice` fixed on Day 3 — measuring a property that *defines* the two populations rather than one that distinguishes their behaviour.

So the headline number is measured on the **intersection** of what both panels carry, with identity-shaped columns excluded by construction, the agentic rail dropped from the synthetic side, and every survivor made dimensionless. Both numbers, labelled:

| input | AUC | what it means |
|---|---|---|
| raw columns — ids, currencies and taxonomies included | 0.9999 | expected, and **not a finding**: two identifier namespaces separate at 1.0 by themselves |
| shape space — intersection, ids dropped, agentic excluded | **0.9994** | the headline |

**The discipline bought almost nothing, and that is the important result.** If the 0.9994 had been an artefact of identifiers or currency, removing them would have collapsed it. It did not move. The separation is real, and it is in behaviour rather than in formatting.

#### What drives it

Attribution is LightGBM's native `pred_contrib` — the same quantity `TreeExplainer` computes, read from the source SHAP would have called. Ranked by mean absolute contribution; `alone` is the feature's own rank-AUC against the panel label.

| feature | alone (AUC) | gain | contribution | class |
|---|---|---|---|---|
| `merchant_rank_pct` | 0.8288 | 79.3% | 70.7% | structural |
| `amount_vs_customer` | 0.5170 | 5.4% | 7.0% | structural |
| `hour` | 0.5643 | 7.3% | 6.0% | cosmetic |
| `gap_ratio_log` | 0.5302 | 3.2% | 5.6% | structural |
| `burst_1h` | 0.5211 | 1.7% | 4.7% | structural |
| `log_amount_z` | 0.5002 | 2.7% | 3.4% | cosmetic |
| `dow` | 0.5056 | 0.3% | 2.1% | cosmetic |
| `category_shift` | 0.5075 | 0.1% | 0.5% | structural |

**Top five drivers:** `merchant_rank_pct`, `amount_vs_customer`, `hour`, `gap_ratio_log`, `burst_1h` — 4 of 5 structural.

#### Cosmetic or structural, and why the distinction decides what to do

**Cosmetic** is a surface property of how values were *rendered*: identifier shape, timestamp granularity, amount rounding. Changing it moves the discriminator without changing anything an attacker or a detector would experience. **Structural** is a property of the joint distribution — which merchants get the volume, how a cardholder's transactions cluster in time, how a ticket sits against their history. Fixing one of those means changing the generative model.

Nothing below is a fix and nothing below is an excuse. A feature can be structural *and* adjudicated to the reference panel — `merchant_rank_pct` is both, and the two statements are independent.

- **`merchant_rank_pct`** — *structural*. Which merchants carry the volume. Ours is Zipf (top 10% take 66%), the reference is near-uniform (14.6%). A property of the generative model, not of formatting -- and separately adjudicated to the reference, because real acceptance estates are heavy-tailed.
- **`amount_vs_customer`** — *structural*. How a ticket sits against the cardholder's own history -- a joint property of the amount draw and the customer assignment, and the single most transferable fraud signal there is.
- **`hour`** — *cosmetic*. Timestamp granularity at the coarsest scale: which hour a transaction is stamped with. The reference has no diurnal curve at all, ours does. Nothing about behaviour changes if the curve is reshaped -- it is when the generator chose to place events, and it is adjudicated to the reference for having no curve to compare against.
- **`gap_ratio_log`** — *structural*. How a cardholder's transactions cluster in time relative to their own rhythm. Burstiness is a behavioural property and the arrival process that produces it is part of the generative model.
- **`burst_1h`** — *structural*. The same arrival process read as a probability. 12.4% of our events follow another within the hour against 16.6% of the reference's.

#### Is it one column, or the joint distribution?

Drop the strongest feature, refit, re-score, repeat. A separation that lives in one column collapses on the first drop; one that lives in the joint distribution degrades gradually. Which of those is true decides whether there is a fix or a rebuild.

| features left | AUC | dropped so far |
|---|---|---|
| 8 | 0.9994 | — |
| 7 | 0.8833 | `merchant_rank_pct` |
| 6 | 0.8416 | `merchant_rank_pct`, `gap_ratio_log` |
| 5 | 0.7076 | `merchant_rank_pct`, `gap_ratio_log`, `amount_vs_customer` |
| 4 | 0.6172 | `merchant_rank_pct`, `gap_ratio_log`, `amount_vs_customer`, `hour` |
| 3 | 0.5883 | `merchant_rank_pct`, `gap_ratio_log`, `amount_vs_customer`, `hour`, `dow` |
| 2 | 0.5231 | `merchant_rank_pct`, `gap_ratio_log`, `amount_vs_customer`, `hour`, `dow`, `log_amount_z` |

**It degrades gradually.** Removing the single strongest feature takes the AUC from 0.9994 to 0.8833 — still far above chance — and it takes dropping 6 of 8 features to reach 0.5231. So the separation is **not** one bad column that could be patched. It is distributed across the joint structure, which is the same thing the correlation-matrix distance measures, and it is the foundry's most substantial outstanding item rather than a cosmetic one.


#### Which side is the anomalous one?

A discriminator says a difference exists. It does not say which panel is wrong, and the temptation three days before a submission is to assume it is the reference. So the rule is that **an adjudication must carry a measurement**: a divergence is attributed to the reference panel only when a third quantity, independent of both and agreed before either dataset existed, says the reference is the side that departs from it.

| feature | test | synthetic | reference | verdict |
|---|---|---|---|---|
| `hour` | retail spend has a diurnal curve (overnight trough) | peak/trough 22.5x | peak/trough 1.6x | **reference** |
| `merchant_rank_pct` | an acceptance estate is Zipf, not uniform (top 10% >> 10%) | top 10% carry 66.0%, max/min 2,980x | top 10% carry 14.6%, max/min 6x | **reference** |

Sparkov's hour-of-day curve is a two-level step rather than a diurnal curve, and its 693 merchants are close to uniformly popular. Neither is a defect in that dataset for its own purpose — it exists to benchmark fraud classifiers, and a flat time curve does not hurt that — but it does mean those two axes cannot measure this project's fidelity.

With them removed the discriminator falls to **0.8399** (68.0% separable). **Both numbers are quoted because the ablation is a judgement**: the full discriminator is the measurement, the ablated one is the measurement after a judgement a reader is free to reject. And 0.84 is still high — the remaining features are individually close (every one under 0.54 alone) and the separation is in their *joint* structure, which is the same thing the correlation-matrix distance measures and is the foundry's most substantial outstanding item.

### Divergences we name ourselves

Found before this scorecard existed, measured, and deliberately left in place. A scorecard that names its own divergences is worth more than one where a judge finds them.

**decline_reason remapping** — invalid_cvv 0.130 -> 0.036, expired 0.080 -> 0.033; 302x its band.

Reasons are remapped where the entry mode makes them impossible - invalid_cvv becomes do_not_honor where no CVV was presented, expired becomes insufficient_funds where the mode cannot expire. Only ~27% of declines are on a CVV-bearing entry mode, so 73% of drawn invalid_cvv gets remapped, which is far larger than the code comment claimed. Conservative for detection: it raises the background rate of the reasons F4-27 farms, which makes that attack's lift smaller rather than larger. Re-tuning it re-rolls every pinned calibration number three days before submission.

**realised decline rate above prior** — moto 2.32x (0.325 vs 0.140), upi_p2p 1.44x, ecom 1.33x, agentic 1.24x; overall 0.088 against a mix-weighted nominal 0.074.

decline_amount_tilt multiplies the per-channel rate by exp(0.55 z), whose expectation is exp(0.55^2/2) = 1.16. That is Jensen's inequality, not a redistribution: the tilt should be mean-preserving per channel and is not. Same reason, and the same direction: a higher decline background makes the card-testing attacks harder to catch, not easier. Recorded as a Day 7 scorecard item rather than a silent edit.

## Scoring latency

Measured **one event at a time** against state warmed on 282,968 training events — not a batch divided by its row count, which is the usual way a latency claim turns out to be false in production. 400 events timed.

### The firewall is two paths, not one deadline

Not every layer has to answer inside the authorisation window, and pretending they do is what makes the budget look unmeetable.

**Inline — L0 + L1.** What must return before the issuer answers the acquirer, because it is what changes the *authorisation decision*: the deterministic protocol clauses, and the supervised score they escalate to. Both need only the event and backward-looking state.

**Fast-follow — L2, L3, L4, fusion and policy.** What informs an action taken **after** the authorisation has been answered. On the agentic rail the two actions that matter are both post-authorisation in any real deployment:

- **Mandate revocation** — a mandate used to read a poisoned page is revoked so the *next* authorisation under it fails L0. That is a credential-lifecycle operation against the mandate registry, not a field in the authorisation response.
- **Agent quarantine** — an agent whose identity component has fused with a ring is suspended, which changes every future authorisation it attempts and none of the one in flight.

Holding an authorisation open while a classifier reads eleven web pages, in order to take an action that cannot be applied to that authorisation anyway, would be an architectural error rather than a performance one. The graph pass is counted **inline** despite L4 being a fast-follow layer, because L1 consumes its 28 `gph_` features: it is inline as a feature source.

| path | p50 | p95 | p99 | vs the 50 ms budget |
|---|---|---|---|---|
| **inline** — L0+L1, pre-authorisation | 61.2 ms | 64.9 ms | **66.3 ms** | **over** |
| **fast-follow** — L2+L3+fusion, post-authorisation | 53.5 ms | 58.1 ms | 62.1 ms | not governed by it |
| **full stack** | 115.7 ms | 121.1 ms | **125.0 ms** | **over** |

The split is asserted to partition the stage list rather than computed by subtraction — an earlier run let L2, which is 40% of the clock, vanish into fast-follow without being named there.

**The inline path misses the budget by 1.3x, and ~0.41 ms of it is real work.** Those same inline stages cost **0.41 ms per row** when called with many rows instead of one — so of the 66 ms measured at p99, something like 99% is per-call framework overhead that a production scoring path would not pay. That does not make the budget met; it makes the miss an implementation property rather than an architectural one, and the next section measures exactly where it goes.

### Every stage

| stage | mean | p99 | share | per row in batch | overhead |
|---|---|---|---|---|---|
| `velocity` | 0.132 ms | 0.654 ms | 0.1% | 0.1319 ms | — |
| `graph` | 0.116 ms | 0.381 ms | 0.1% | 0.1163 ms | — |
| `transaction` | 6.971 ms | 8.553 ms | 6.0% | 0.0192 ms | 364x |
| `entity` | 45.909 ms | 49.781 ms | 39.7% | 0.1128 ms | 407x |
| `mandate` | 4.745 ms | 5.239 ms | 4.1% | 0.0134 ms | 355x |
| `L0` | 0.779 ms | 0.951 ms | 0.7% | 0.0028 ms | 275x |
| `L1` | 3.086 ms | 3.448 ms | 2.7% | 0.0183 ms | 169x |
| `L2` | 43.174 ms | 50.464 ms | 37.3% | 0.1481 ms | 292x |
| `L3` | 0.041 ms | 0.084 ms | 0.0% | 0.0014 ms | 30x |
| `fusion+policy` | 10.801 ms | 11.708 ms | 9.3% | 0.0285 ms | 380x |

### Read the last two columns before the p99

Most of the clock goes to `entity`, which costs 45.9 ms called with one row and 0.1128 ms per row called with many — a factor of **407**. That is per-call overhead in pandas and scikit-learn, not model work: `Series.map(dict)` materialises the lookup table into an index on every call, so a feature block pays that cost once per feature to look up one value.

**The fix is named and not applied.** A plain dictionary lookup on the single-event path removes it, and the feature builder is shared with the offline pass behind every pinned number in this document. Three days out, this project records the finding rather than re-rolls the tables for it.

The two stages that genuinely **cannot** be batched — the stateful stores, which must read state before folding the event in, and which are therefore the same code online and offline — cost `velocity` 0.654 ms p99 and `graph` 0.381 ms p99, **1.04 ms together**. Those are the numbers that would survive a rewritten scoring path, and they are the ones the architecture was designed around: one forward pass, `bisect` and prefix sums per window, union-find over the identity graph, bounded memory by eviction.

The honest headline is therefore both sentences: **the current implementation misses a 50 ms budget at p99 (125 ms full stack, 66 ms inline), and the miss is in the calling convention rather than in the models.** Quoting only the second would be an estimate dressed as a measurement; quoting only the first would invite the conclusion that a five-layer firewall cannot run inline, which this measurement does not support.

One measurement-hygiene note, because it changed a number by a factor of two: an earlier run of this bench was taken while the fidelity scorecard was fitting on the same machine, and every percentile roughly doubled. Latency is the one measurement in this repository that is not reproducible from a seed — it is a property of the machine at the moment it ran — so it is measured with nothing else running, and that is worth stating rather than assuming.

## Two things a deployment would need to know

### Fusion consumes L3's score, not L3's decision

This matters more than it sounds like it does, and the L3 out-of-distribution result above is why.

`FusionModel` gives every layer **three columns**: its percentile against the legitimate score distribution, its raw score standardised on the fusion window's legitimate rows, and — where the layer is sometimes silent — an indicator for whether it had an opinion at all. No threshold is applied to any layer before fusion. L3's page threshold is used for *reporting* L3 as a standalone layer and for nothing else.

That is the right way round, and it is what makes the out-of-distribution finding survivable. L3's threshold **does not transfer** to text unlike its training corpus: pointed at hand-authored payloads it fires on 100% of the injections and on 90% of the *clean* controls written in the same registers. A fused score that consumed L3's thresholded decision would inherit that failure directly — every procedural-sounding page would arrive at fusion as a hard vote for fraud, on a layer whose calibration had silently stopped being valid.

Consuming the score instead means the stacker sees L3's *ordering*, which is what survived the transfer: ROC falls 0.999 → 0.811 rather than collapsing. The percentile column is re-derived from the deployment's own legitimate traffic every time fusion is fitted, so a shift in L3's absolute scale is re-absorbed at the next refit rather than becoming a permanent bias. And the fitted weight is the check on the whole arrangement: on this data the stacker put **-0.943** on L3's percentile and **+0.353** on its standardised raw score, which is the model saying it trusts the layer's ordering while discounting the calibration that produced the percentile. That is a discount a decision-consuming fusion could not have applied.

### What the non-transferring threshold means for deploying on novel text

Stated plainly, because it is the largest single caveat on the agentic side of this architecture: **L3 as calibrated here cannot be pointed at the open web.** It is a classifier fitted on one 7B model's output through one set of prompt templates in one register, and its decision boundary is a property of that corpus rather than of injected text in general.

Three consequences for a deployment, in the order they would bite:

1. **Fit the page threshold on the traffic it will see, not on the corpus it was trained on.** The benign side of that calibration set is the part that matters and the part that is easy to get for free — it is the pages agents read on ordinary, approved authorisations, which an issuer accumulates without labelling anything. The 90% false-positive rate on hand-authored controls is what happens when this step is skipped.
2. **Do not promote L3 to a standalone decision.** Its recall is real on two of fifteen cards and its ordering transfers; its calibration does not. It belongs behind fusion, where a fitted weight can discount it, and behind L0, where the protocol invariants need no calibration at all.
3. **A bag of words is the wrong long-term model, and the failure mode says why.** It keys on lexical markers of instruction — *do not*, *skip*, *without* — which is precisely why prose that merely sounds procedural trips it. The 1.000 recall on F1-01 and F1-03 is a true statement about this corpus and is not a claim about text in general.

The general form of this is worth keeping, because it is not specific to L3: **a layer whose ordering transfers and whose calibration does not is still useful, provided nothing downstream consumes its threshold.** The architecture already satisfies that condition; it was not designed to, and the out-of-distribution probe is what turned an accident into a checked property.

## What this does not claim

- **This is not real-world performance.** It is measured on synthetic data whose attacks we wrote. The fidelity scorecard above is the argument that the background is realistic, and it is a qualified argument: read its discriminator row before reading any recall here.
- **F5 is absent from every table.** It is the zero-day holdout family and has no implemented injector, so it is not in the data and cannot be scored. The leave-one-family-out columns and the loop experiment are the closest available stand-ins.
- **L2 is not a detector and is no longer presented as one.** See the section above.
- **L3 covers two of fifteen cards.** It is a specialist, and its overall recall should be read as coverage of the agentic-injection rail rather than as a headline.
- **The current event's own outcome is never a feature.** `auth_response`, `settled` and `settlement_lag_hours` of the row being scored are blocked by name in the feature builder, alongside the label and post-hoc columns. Feeding them in would raise F4-27's recall substantially and mean nothing — an issuer cannot decline a transaction because it was declined.
- **The measured latency is over budget, and the section above says so.** What is *not* claimed is the reverse: no number here is an estimate of what a rewritten scoring path would cost. The p99 is what this implementation does today.

