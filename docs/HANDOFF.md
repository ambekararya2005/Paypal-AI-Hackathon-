# MANTIS — factual handoff

Written 2026-08-26 from the repository at commit `692cc73`. Every number below was
either read out of a committed artefact or re-measured today; the command, dataset,
seed and operating point are stated inline. Where a figure in `CLAUDE.md` or
`RESULTS.md` disagrees with what the repository produces today, both are given and
the disagreement is named.

---

## 1. Identity and artifacts

| | |
|---|---|
| Project name | MANTIS |
| One line | A red-team / blue-team lab for agentic-payment fraud: an executable 42-card attack atlas, a foundry that injects those attacks into a calibrated legitimate population, a five-layer Mandate Firewall, and an evolutionary adversary that closes the loop. |
| GitHub | https://github.com/ambekararya2005/Mastercard-Innovation-Hackathon- |
| Live prototype | https://aryaambekar-mantis.static.hf.space — note `.static.hf.space`; the plain `.hf.space` form 404s |
| Commit SHA | `692cc733069fdca037f63664157fcb587252d625` ("Latency Result"), branch `main`, working tree clean |
| Tests | **302 passed, 0 failed** — `python -m pytest`, run 2026-08-26. *`CLAUDE.md` §8 says 287; that figure is stale.* |
| ruff | `python -m ruff check .` → **All checks passed!** (line length 100, target py311) |
| Python | declared `requires-python = ">=3.11"`; measured on CPython **3.13.1** |
| Package version | `mantis` 0.1.0 |

### LOC by top-level package (`.py` only, `__pycache__` excluded)

| package | lines | files |
|---|---|---|
| `mantis/core` | 760 | 3 |
| `mantis/atlas` | 427 | 4 |
| `mantis/foundry` | 12,930 | 45 |
| `mantis/defense` | 7,122 | 34 |
| `mantis/loop` | 1,602 | 6 |
| `mantis/api` | 1,228 | 6 |
| `scripts/` | 4,827 | 12 |
| `tests/` | 4,030 | 15 |
| `web/src` (ts/tsx/css) | 3,330 | 9 |

Tests per file: `test_population` 44, `test_injectors` 38, `test_schema` 35,
`test_probe_slices` 32, `test_agentic_injectors` 28, `test_api` 22,
`test_features` 19, `test_atlas` 17, `test_fidelity` 15, `test_l0_rules` 13,
`test_loop` 12, `test_fusion_policy` 11, `test_l4_graph` 9, `test_l3_text` 7.

### Repo tree, 2 levels

```
./            CLAUDE.md RESULTS.md README.md DEPLOY.md Makefile Dockerfile
              pyproject.toml railway.json render.yaml .dockerignore
data/         cache/  generated/  reference/
deploy/       hf/
docs/         MANTIS_submission.docx  HANDOFF.md  drift_check.json
              fidelity_scorecard.png  population.md  population_calibration.png
mantis/       api/  atlas/  core/  defense/  foundry/  loop/
scripts/      audit_population.py  audit_probe_slices.py  build_console_feed.py
              build_docx.py  build_static_site.py  deploy_hf.py  drift_check.py
              fetch_reference.py  fit_reference.py  l3_ood.py  latency_bench.py
              probe_derived.py
tests/        15 test modules
web/          src/  public/  dist/
```

Third level that matters: `mantis/atlas/{cards,discovered,discovery}`;
`mantis/foundry/{base,injectors,llm,fidelity}`;
`mantis/defense/{features,l0_rules,l1_gbdt,l2_novelty,l3_text,l4_graph,fusion,policy,explain}`.

---

## 2. Problem framing as the code actually states it

### 2.1 The pitch — verbatim, `CLAUDE.md` §1

> Agentic commerce — AI agents that pay on a human's behalf via **Mastercard Agent
> Pay / AP2 mandates** — has created a live payment rail with **zero labelled
> fraud data**. You cannot train a fraud detector on data that does not exist. So
> you *manufacture* it, adversarially.
>
> MANTIS is a red-team / blue-team lab in three pillars:
>
> 1. **IDENTIFY** — an *executable* atlas of 42 GenAI payment-fraud vectors as
>    YAML cards that the generator **imports**. Not documentation: a dependency.
> 2. **GENERATE** — a foundry that synthesises a calibrated legitimate payment
>    population and injects those attacks into it, with a **measured** fidelity
>    scorecard (not a vibe check).
> 3. **DEFEND** — a five-layer **Mandate Firewall** that scores every
>    authorisation and reports **recall at a fixed 0.1% false-positive rate**.
>
> Then the closed loop that makes it a *lab* rather than a demo: an evolutionary
> adversary reads the detector's own SHAP output, mutates to evade it, the
> detector retrains, and the **evasion rate falls**. That curve is the money shot.

### 2.2 "The zero-day answer, reframed (Day 5)" — verbatim, `CLAUDE.md`

> **L2 is demoted.** It was designed and documented as *the zero-day layer* — the
> answer to "what about the attacks you did not think of". Day 4 measured it at
> **0.4% mean per-family recall at 0.1% FPR, 0.62 ROC**, and Day 5 did not chase
> that number. L2's job in the architecture is now:
>
> > **L2 is a residual monitor and a drift canary.** It answers "has the shape of
> > legitimate traffic moved", and it flags the residue that no other layer
> > claims. It is not a detector, and no table may present it as one.
>
> The architecture's actual answer to an unseen attack is now two things, both of
> which need **no labelled example of that attack**:
>
> 1. **L0 protocol invariants.** A mandate that is expired, out of scope, over its
>    ceiling, replayed, or whose provenance trail does not terminate at the
>    merchant that was paid, is a **violation of the AP2 contract**, not a
>    statistical outlier. Nine deterministic clauses, no training data, deployable
>    today. An attack that has never been seen still has to break the protocol to
>    move money, and the clauses do not care whether it is novel.
> 2. **The closed loop** (`mantis/loop/`). Rather than hoping an unsupervised layer
>    generalises to an attack we never wrote, we **manufacture the attack before an
>    attacker does**: the evolutionary adversary mutates known cards into variants
>    that evade the current detector, those variants are labelled by construction,
>    and L1 retrains on them. The zero-day answer is not "a layer that generalises"
>    — it is "a generator that gets there first."
>
> **Why L2 failed, stated as a finding rather than hidden as a weakness.** This is
> the most interesting negative result in the project and it belongs in the
> writeup:
>
> > **Attacks built to be distributionally faithful are, by construction, invisible
> > to distributional anomaly detection.**
>
> Every design decision in the foundry pushed the attacks *toward* the legitimate
> manifold — clone real background rows, resample amounts from the target MCC's own
> empirical band, redraw the hour of day from the population's diurnal curve,
> widen three legitimate tails specifically so an attack would not be free, keep
> provenance planting length-preserving. The Day 2/Day 3 separability gate is
> literally a rule that says *no single raw column may separate an attack above
> 0.95 AUC*. An isolation forest measures distance from the legitimate manifold. We
> spent two days minimising exactly that distance. **Our own fidelity work caused
> this result**, and it is not an artefact — it is the property real GenAI-driven
> fraud has, because an agent that pays with a validly-signed mandate on a real
> cardholder's real device for a plausible amount at a real merchant *is* a
> legitimate-looking transaction in every marginal. The fraud is in the **intent
> and the relations**, not in the marginals. That is why the layers that work are
> the ones reading relations (L4, entity-level) and intent (L3, the ingested text),
> and it is why "just run an autoencoder on it" is not an answer to agentic fraud.
>
> Corollary for the writeup: a fidelity scorecard and an anomaly-detection recall
> number are **in tension by construction**. A project that reports both high is
> reporting one of them wrongly.

### 2.3 The loop-vs-held-out wording fix

**There is no section in the repository labelled "Day 6 Task 0b".** `CLAUDE.md`
contains no Day 6 section at all — it jumps Day 5 → Day 7. The load-bearing
wording lives in `RESULTS.md` under "The zero-day demonstration". Verbatim:

> **What the detector had, and what the loop had. These are not the same thing, and the whole claim turns on the difference.**
>
> The *detector* never trained on a single real F1 event. That is what the middle row measures, and 0.013 is what it gets it for.
>
> The *loop* had something else: F1's **atlas cards and their executable injectors** — a written description of a class of attack, and code that manufactures instances of it. That is a red team, not a fraud history. It is why the third row is not magic and must never be described as the detector generalising on its own: it did not generalise, it was **given manufactured training data for a family it had never seen in the wild**, and that data was produced from a specification a human wrote before any F1 attack was observed.
>
> The variants are still not the test rows. Every gene moved them, and they were **selected for evading the detector**, so they sit off-distribution from the canonical attack in exactly the direction that makes the transfer hard — which is why the recovery is 66% and not 100%.
>
> **This is the realistic position on a new rail, and it is the point of the project.** Agentic commerce has no labelled fraud history, and will not have one until losses have already been taken. What it can have on day one is a red team: people who can describe the attack and write the generator. The claim is therefore *"an attack family that has been described but never observed can be manufactured, and training on the manufactured version transfers to the real one"* — **not** *"the detector caught something nobody had thought of"*. Nothing does that. Somebody thought of it; the contribution is that thinking of it was enough.

---

## 3. Event schema

`SCHEMA_VERSION = "1.1.0"` — `mantis/core/events.py:131`. Frozen; amendments
additive-only. `tests/test_schema.py` (35 tests) pins the first seventeen classic
columns literally, so "additive" is enforced rather than intended.

### 3.1 `TxEvent` — classic authorisation block, wire order

| field | type | default | purpose |
|---|---|---|---|
| `event_id` | `str` | required | Unique authorisation id |
| `ts` | `datetime` | required | Authorisation timestamp, tz-aware (IST in the population) |
| `amount` | `float` `ge=0` | required | Amount in `currency` major units. **Non-negative on every `txn_type`** |
| `currency` | `str` | required | ISO 4217 alpha-3, validated uppercase |
| `mcc` | `str` | required | ISO 18245, exactly 4 digits, validated |
| `channel` | `Channel` | required | Acceptance rail: card_present, ecom, moto, recurring, upi_p2p, upi_p2m, agentic |
| `entry_mode` | `EntryMode` | required | chip, contactless, magstripe, ecom_keyed, credential_on_file, network_token, qr_scan, agent_token |
| `customer_id` | `str` | required | Cardholder |
| `card_bin` | `str` | required | Issuer BIN, 6–8 digits |
| `merchant_id` | `str` | required | Merchant |
| `merchant_country` | `str` | required | ISO 3166-1 alpha-2, validated |
| `terminal_id` | `str \| None` | `None` | `None` off card-present rails |
| `device_id` | `str \| None` | `None` | Device |
| `ip` | `str \| None` | `None` | Source IP |
| `lat` | `float \| None` `[-90,90]` | `None` | Latitude |
| `lon` | `float \| None` `[-180,180]` | `None` | Longitude |
| `threeds_result` | `ThreeDSResult` | `not_applicable` | frictionless, challenge_passed, challenge_failed, attempted, unavailable |

### 3.2 `TxEvent` — transaction-lifecycle block (Amendment 1.1.0; all defaulted)

| field | type | default | purpose | wire mapping |
|---|---|---|---|---|
| `txn_type` | `TxnType` | `purchase` | purchase / refund / reversal / preauth / credit | **ISO 8583 DE 3** processing code |
| `auth_response` | `AuthResponse` | `approved` | approved + declined_{insufficient_funds, do_not_honor, invalid_cvv, expired, risk} | **ISO 8583 DE 39** response code |
| `original_event_id` | `str \| None` | `None` | Original-transaction reference on refund/reversal/credit; its **absence** on an outbound flow is itself the F1-03 signal | **DE 90** / the retrieval reference number an acquirer echoes on a credit |
| `dispute_outcome` | `DisputeOutcome \| None` | `None` | none / raised / won_merchant / won_cardholder | issuer dispute system, joined by case management |
| `dispute_raised_ts` | `datetime \| None` | `None` | When the cardholder disputed | same |
| `settled` | `bool` | `True` | Reached clearing | clearing file |
| `settlement_lag_hours` | `float \| None` `ge=0` | `None` | Hours authorisation → clearing | clearing file |

Other ISO references in `events.py`: `currency` ISO 4217, `mcc` ISO 18245,
`merchant_country` ISO 3166-1 alpha-2.

### 3.3 `AgenticContext` — `None` on classic rails; prefixed `ag_` when flattened

| field | type | default | purpose |
|---|---|---|---|
| `agent_id` | `str` | required | Stable transacting-agent identifier |
| `agent_platform` | `str` | required | Vendor / runtime hosting the agent |
| `kya_token` | `str \| None` | `None` | Know-Your-Agent credential |
| `kya_registered` | `bool` | `False` | Agent present in the KYA registry |
| `mandate_type` | `MandateType` | `none` | none / intent / cart / payment — authority and blast radius decrease left to right |
| `mandate_id` | `str \| None` | `None` | Mandate identifier |
| `mandate_hash` | `str \| None` | `None` | Hash of the signed mandate artefact; reuse is replay |
| `mandate_issued_ts` | `datetime \| None` | `None` | Issue time |
| `mandate_ttl_seconds` | `int \| None` `ge=0` | `None` | Validity window |
| `mandate_scope` | `MandateScope \| None` | `None` | The constraint envelope (§3.4) |
| `human_present` | `bool` | — | Human oversight claimed |
| `consent_sig_valid` | `bool \| None` | `None` | Consent-signature verification result |
| `delegation_depth` | `int` | — | Hops from the human to the transacting agent |
| `provenance_chain` | `list[str]` | `[]` | **Ordered URLs / content the agent read before deciding.** Input to L3; the field that turns indirect prompt injection from describable into detectable |
| `ingested_content_ids` | `list[str]` | `[]` | Keys into `ContentStore` |
| `tool_call_count` | `int` `ge=0` | `0` | Tool invocations before authorisation |
| `deliberation_latency_ms` | `int \| None` `ge=0` | `None` | Reasoning time before commit |
| `cursor_entropy` | `float \| None` `ge=0` | `None` | Human-input telemetry |
| `dwell_time_ms` | `int \| None` `ge=0` | `None` | Human-input telemetry |

### 3.4 `MandateScope`

| field | type | default | purpose |
|---|---|---|---|
| `categories` | `list[str]` | `[]` | Permitted MCC groups |
| `max_amount` | `float \| None` `ge=0` | `None` | Per-transaction ceiling |
| `max_items` | `int \| None` `ge=0` | `None` | Max basket line items |
| `allowed_merchants` | `list[str]` | `[]` | Merchant allow-list |
| `ttl_seconds` | `int \| None` `ge=0` | `None` | Scope validity window |

### 3.5 Column contracts and why post-hoc is excluded

| tuple | members | rule |
|---|---|---|
| `LABEL_COLUMNS` | `is_fraud`, `attack_id`, `attack_campaign` | Ground truth. Never a feature (HARD RULE 1) |
| `POST_HOC_COLUMNS` | `dispute_outcome`, `dispute_raised_ts` | **Resolve days to months after the authorisation.** Using them at scoring time is temporal leakage: the model would be reading the future. Carried for evaluation, cost modelling and the console; dropped alongside labels in the feature builder |
| `FUTURE_COLUMNS` (`defense/features/spec.py`) | `auth_response`, `settled`, `settlement_lag_hours` **of the event being scored** | The issuer's own decision on this message. Does not exist when the firewall runs |
| `FORBIDDEN_COLUMNS` | union of the three — 8 columns | One containment assertion on every `transform` |
| `DECLINE_RESPONSES` | every non-approval `AuthResponse` | Named once so decline-ratio features cannot drift |

Exports: `SCHEMA_VERSION`, `LABEL_COLUMNS`, `POST_HOC_COLUMNS`,
`DECLINE_RESPONSES`, `CLASSIC_COLUMNS`, `flatten(ev) -> dict`.

Leakage assertion, fired deliberately (`python -m mantis.defense.features`):
`is_fraud` REJECTED, `attack_id` REJECTED, `auth_response` REJECTED,
`dispute_outcome` REJECTED — three tiers, 8 columns.

### 3.6 What the 1.0.0 → 1.1.0 amendment unblocked

Day 0 modelled only the authorisation *request*. That made **F1-03 (refund-logic
hijack) unrepresentable** — it stayed `status: mapped` through Days 1–2 — and left
**F4-27 / F4-28 half-modelled**, because the approve/decline oracle those attacks
farm had nowhere to live. `events.py` calls this a specification error, not a
design constraint. What it made buildable: the decline-ratio velocity windows (per
card, BIN, merchant, device over 1h/24h/7d), refund-to-purchase ratios per customer
and merchant, merchant settlement-lag deviation from its rail's mode, and the
orphan-credit indicator. Direction of value is carried by `txn_type`, never by the
sign of `amount`, because a signed amount would break every quantile, KS distance
and log-amount feature calibrated on Day 1.

---

## 4. The attack atlas — all 42

Source: `mantis/atlas/cards/*.yaml`, read 2026-08-26. `actor` and `genai_enabler`
are compressed to a phrase each; the full prose is on the card.
Generator paths are relative to `mantis.foundry.injectors`.

| id | name | fam | status | rails | actor | genai_enabler | generator | detected_by | discovered_by |
|---|---|---|---|---|---|---|---|---|---|
| F1-01 | Cart-mandate tampering via indirect prompt injection | F1 | implemented | agentic, ecom | controls third-party content an agent reads | agent cannot separate retrieved content from instructions | `f1_01_cart_tampering:inject` | L0, L1, L3 | human |
| F1-02 | Intent-mandate scope inflation | F1 | implemented | agentic | influences the agent's planning context | intent mandates delegate a goal, not a basket | `f1_02_scope_inflation:inject` | L0, L1, L2 | human |
| F1-03 | Refund-logic hijack | F1 | implemented | agentic, ecom | places content before a buyer- or merchant-side refund agent | merchants automate returns with LLM agents holding real authority | `f1_03_refund_hijack:inject` | L0, L1, L3, L4 | human |
| F1-04 | Intent-mandate category drift | F1 | implemented | agentic | influences what the agent considers "equivalent" | the agent must interpret category boundaries | `f1_04_category_drift:inject` | L0, L1, L2 | human |
| F1-05 | Delegation-chain laundering through sub-agents | F1 | implemented | agentic | controls a sub-agent a primary agent delegates to | delegation is cheap and each hop strips context | `f1_05_delegation_laundering:inject` | L0, L1, L4 | human |
| F1-06 | Consent-signature stripping / authentication downgrade | F1 | mapped | agentic, ecom | sits between the agent and the acquirer | agents negotiate checkout paths dynamically and optimise for completion | — | L0, L1, L2 | human |
| F1-07 | Time-of-check to time-of-use mandate mutation | F1 | mapped | agentic | influences agent state between approval and settlement | the approve→settle gap is no longer milliseconds | — | L0, L1, L3 | human |
| F1-08 | Cross-mandate credential reuse | F1 | mapped | agentic | access to a runtime holding several live mandates | agents hold many concurrent authorisations | — | L0, L1, L4 | human |
| F1-09 | Human-present spoofing for liability shift | F1 | implemented | agentic, ecom | agent/platform operator wanting the liability shift | presence telemetry is now cheap to synthesise | `f1_09_presence_spoof:inject` | L0, L1, L2 | human |
| F1-10 | Mandate replay and TTL abuse | F1 | implemented | agentic | observes or retains a valid mandate artefact | mandates are passed between planners and tools as ordinary context | `f1_10_mandate_replay:inject` | L0, L1, L4 | human |
| F1-11 | Silent mandate renewal / TTL extension abuse | F1 | mapped | agentic, recurring | controls the mandate-refresh component | agents are built not to interrupt the human | — | L0, L1, L2 | human |
| F1-12 | Multi-agent collusion under a single mandate | F1 | mapped | agentic | runs coordinated agents sharing one mandate | coordinating against a shared constraint is a scheduling problem | — | L1, L2, L4 | human |
| F2-13 | GenAI synthetic identity onboarding | F2 | implemented | ecom, upi_p2m, agentic | builds portfolios of fabricated customers | per-identity cost collapsed | `f2_13_synthetic_identity:inject` | L0, L1, L4 | human |
| F2-14 | Unregistered / forged Know-Your-Agent credentials | F2 | mapped | agentic, ecom | stands up unregistered or forged agent identities | agent identity is new and inconsistently enforced | — | L0, L1, L2, L4 | human |
| F2-15 | Deepfake liveness bypass at video KYC | F2 | mapped | ecom, agentic | onboards through remote video KYC under a synthetic identity | real-time face/voice synthesis is commodity | — | L1, L2, L4 | human |
| F2-16 | Agent-farm account bootstrapping for bust-out | F2 | implemented | ecom, agentic, upi_p2m | portfolio of synthetic accounts, each with its own agent | removes the human labour of maintaining plausible accounts | `f2_16_bust_out:inject` | L1, L2, L4 | human |
| F2-17 | Generated-document merchant onboarding | F2 | mapped | ecom, agentic, upi_p2m | onboards a merchant on fabricated documents | document generation at scale in correct regional formats | — | L1, L4 | human |
| F2-18 | Synthetic-agent reputation farming | F2 | mapped | agentic | builds agent identities with manufactured standing | any reputation system that rewards history invites manufacturing it | — | L1, L2, L4 | human |
| F3-19 | Digital-arrest scam | F3 | implemented | upi_p2p, upi_p2m, ecom, agentic | crew impersonating law enforcement, at scale | voice cloning plus real-time translation | `f3_19_digital_arrest:inject` | L1, L3, L4 | human |
| F3-20 | Voice-clone authorised push payment fraud | F3 | mapped | upi_p2p, upi_p2m, ecom | impersonates a family member or bank official | voice cloning from seconds of public audio | — | L1, L2, L4 | human |
| F3-21 | Task / investment scam payout orchestration | F3 | mapped | upi_p2p, upi_p2m, ecom | runs a task or investment scam | personalised multi-week conversation at scale | — | L1, L2, L4 | human |
| F3-22 | Conversational romance-scam escalation | F3 | mapped | upi_p2p, ecom | long-horizon relationship-based operator | removes the per-relationship labour bound | — | L1, L2, L4 | human |
| F3-23 | Fake storefront seeded into agent search | F3 | mapped | agentic, ecom | stands up a merchant presence built to be selected by agents | agents select merchants by reading machine-facing content | — | L1, L3, L4 | human |
| F3-24 | Invoice redirection against agent procurement | F3 | mapped | agentic, ecom, moto | impersonates a supplier and substitutes payment details | BEC-grade correspondence is now free to produce | — | L0, L1, L3, L4 | human |
| F3-25 | Deepfake executive authorisation of a corporate mandate | F3 | mapped | agentic, ecom, moto | impersonates a senior executive on synthetic video | real-time synthetic video breaks the escalation channel | — | L1, L2 | human |
| F3-26 | Support-impersonation mandate handover | F3 | mapped | agentic, ecom | impersonates support or an agent-platform vendor | "let my assistant sort that out" is natural in agentic UX | — | L1, L4 | human |
| F4-27 | Adaptive BIN attack | F4 | implemented | ecom, agentic, card_present | operation with a credential corpus and merchant reach | approve/decline is a free oracle an LLM loop can search | `f4_27_adaptive_bin:inject` | L1, L2, L4 | human |
| F4-28 | Threshold probing and just-under structuring | F4 | implemented | agentic, ecom, upi_p2m, card_present | observes outcomes to locate the threshold | threshold discovery is a search problem, and search is what an agent is for | `f4_28_threshold_probe:inject` | L1, L2 | human |
| F4-29 | Attribution-guided feature-space evasion | F4 | mapped | agentic, ecom | has access to detector explanations | attribution methods work as well for an attacker as an auditor | — | L0, L1, L2, L3 | human |
| F4-30 | Velocity shaping against count-window rules | F4 | mapped | agentic, ecom, upi_p2m | schedules against an inferred velocity rule | scheduling many actions against a known constraint | — | L1, L4 | human |
| F4-31 | Feedback-loop label poisoning | F4 | mapped | agentic, ecom | influences the labels a detector learns from | plausible dispute narratives at volume are cheap | — | L1, L2, L3, L4 | human |
| F4-32 | Decision-boundary mapping through probe transactions | F4 | mapped | agentic, ecom | submits low-value probes to map the boundary | model extraction is a query-efficiency problem | — | L1, L2 | human |
| F5-33 | Tool-descriptor poisoning | F5 | mapped | agentic | controls a tool's metadata, or publishes a tool | an agent chooses which tool to call by reading its description | — | L0, L1, L3 | human |
| F5-34 | Agent-platform compromise / mass mandate minting | F5 | mapped | agentic | has compromised an agent platform | platforms concentrate delegated payment authority | — | L1, L2, L4 | human |
| F5-35 | Model supply-chain substitution | F5 | mapped | agentic | substitutes the planning model | the planner arrives as an opaque supply-chain artefact | — | L1, L2, L4 | human |
| F5-36 | Persistent memory / context-store poisoning | F5 | mapped | agentic | writes into an agent's long-term memory | persistent memory is also a persistence mechanism | — | L0, L2, L3 | human |
| F5-37 | Rogue payment connector in the tool chain | F5 | mapped | agentic, ecom | installs or advertises a payment-capable connector | agents extend themselves with new capabilities at runtime | — | L1, L3, L4 | human |
| F6-38 | Agentic mule-network fan-out | F6 | implemented | upi_p2p, upi_p2m, agentic | coordinates a mule network | layering's coordination cost collapsed | `f6_38_mule_fanout:inject` | L1, L4 | human |
| F6-39 | Shell merchant-of-record laundering | F6 | implemented | ecom, agentic, upi_p2m | controls merchant accounts accepting directed volume | operating a plausible merchant presence is near-free | `f6_39_shell_merchant:inject` | L1, L4 | human |
| F6-40 | Stored-value / gift-card cash-out chains | F6 | implemented | ecom, agentic, upi_p2m | converts payment authority into stored value | purchasing and reselling instruments at volume is clerical work | `f6_40_stored_value:inject` | L1, L4 | human |
| F6-41 | Cross-border micro-remittance layering | F6 | mapped | ecom, agentic, upi_p2p | disperses value across borders below thresholds | splitting across corridors while respecting each threshold | — | L1, L4 | human |
| F6-42 | Subscription-farm bleed | F6 | mapped | recurring, agentic, ecom | monetises via many small recurring charges | many plausible subscription products, cheaply | — | L1, L4 | human |

### 4.1 Counts

| family | total | implemented | mapped |
|---|---|---|---|
| F1 | 12 | 7 | 5 |
| F2 | 6 | 2 | 4 |
| F3 | 8 | 1 | 7 |
| F4 | 6 | 2 | 4 |
| F5 | 5 | **0** | 5 |
| F6 | 5 | 3 | 2 |
| **total** | **42** | **15** | **27** |

F5 is empty **on purpose** — it is the zero-day holdout family, and
`tests/test_atlas.py` pins the implemented-family set so emptying or filling it is
a deliberate act.

### 4.2 The exact wording used for the implemented/mapped split

Printed by `python -m mantis.atlas.loader` under the heading `HONEST COUNT`,
verbatim:

```
  HONEST COUNT
    42 vectors mapped; 15 of them have a working injector.
    The remaining 27 are taxonomy: each carries observable signals
    and mitigations, but the foundry does not generate them. Never blur the
    two in a slide -- overclaiming coverage loses a technical room.
```

Definition of `implemented`, enforced at package import by
`injectors.base.validate_registry()`: an injector exists, is registered, its
`card_id` names a card the atlas calls `implemented`, that card names a
`generator` path, and the path resolves to a callable **in that injector's own
module**. Both directions are checked. `tests/test_injectors.py` fires the
assertion deliberately to prove it is load-bearing.

### 4.3 The six cards in `mantis/atlas/discovered/`

All six are `status: mapped`, `discovered_by: adversarial_loop`, `generator: null`
— a variant is its parent's injector plus a genome, not a module of its own. Each
ships a `.genome.json` sidecar. They are **not counted** in the 42.

| card | parent | label | evasion | payoff | survived rounds | n_events |
|---|---|---|---|---|---|---|
| F1-50 | F1-01 | `F1-01~14019` | 0.110 | 0.0373 | 4 | 300 |
| F1-51 | F1-05 | `F1-05~51253` | 0.055 | 0.0516 | 4 | 200 |
| F3-50 | F3-19 | `F3-19~60646` | 0.264 | 0.2473 | 4 | 220 |
| F4-50 | F4-27 | `F4-27~33316` | 0.635 | 0.7278 | 4 | 400 |
| F6-50 | F6-38 | `F6-38~86941` | 0.414 | 0.4254 | 4 | 360 |
| F6-51 | F6-38 | `F6-38~21488` | 0.328 | 0.4278 | 4 | 360 |

All six survived to generation 4 (the final generation of a 5-generation run), i.e.
4 consecutive rounds against a retraining detector. `RESULTS.md` says "three or
more consecutive rounds"; the measured value for every one of the six is 4.

Gene medians across the six: `deliberation_scale` **2.93** (5 of 6 above 2.6),
`merchant_spread` **0.51**. Both are the loop independently finding features the
build already knew were strong — `mnd_deliberation_residual_z` and
`gph_merchant_fanin_per_component`.

### 4.4 The three unmutated survivors, reported separately

Every card's arena population is seeded with an **identity genome** (all genes at
default) so the evasion curve carries its own no-evolution reference row. Three of
the nine survivors were that individual. `writeback.is_novel` filters them and they
are **not** written back, because recording them would claim a discovery for an
attack already in `cards/`.

| label | card | genes | evasion | payoff | fitness |
|---|---|---|---|---|---|
| `F2-16~50310` | F2-16 | all default | **0.7775** | 0.6474 | 0.5033 |
| `F1-05~50310` | F1-05 | all default | 0.2900 | 0.2875 | 0.0834 |
| `F1-01~50310` | F1-01 | all default | 0.2233 | 0.1290 | 0.0288 |

The reading: F1-01, F1-05 and F2-16 are where the detector is weakest, and F2-16's
unmutated bust-out evades 77.8% of decisions without any evolution at all.

### 4.5 Outstanding atlas item — Day 2 cards under-declare their rails

Writing `test_attack_rails_agree_with_the_card` surfaced that several Day 2
injectors clone whatever rail their source row was on and therefore ride rails
their card does not list. Named example: **F2-13 reaches `card_present`,
`recurring` and `upi_p2p` against a card naming `agentic` / `ecom` / `upi_p2m`.**
The subset assertion is therefore **enforced for the Day 3 F1 cards only**.
Widening seven Day 2 cards to make a new test pass would be the wrong way round;
the reconciliation is a Day 2 job that has **not been done**.

---

## 5. Generation

### 5.1 Population simulator

`python -m mantis.foundry.base --n 200000 --seed 7` →
`data/generated/population.parquet` (16.8 MB) + `population.manifest.json` +
`docs/population_calibration.png`. Runtime: build 0.07 s, simulate 7.80 s,
write 0.67 s.

| property | value |
|---|---|
| size | 200,000 events, 90-day window from 2026-05-15 IST |
| entities | 5,000 customers requested / 4,975 realised; 12,000 merchants requested / 10,778 realised |
| seed | 7 for the population gate; 1337 everywhere else (`--seed`, default 1337) |
| calibration source | **`indian-market-priors` — built-in defaults, not a fitted CSV.** `ReferenceStats.source` records this |
| currency | INR |
| amount KS vs prior | 0.00622 (non-zero by design: round-number snapping) |
| amount median / mean / p99 | ₹782.63 / ₹2,395.02 / ₹28,500.19 |
| hour-of-day total variation | 0.00508 |
| MCC mix max abs delta | 0.00110 |
| Zipf exponent | target 1.08, realised 0.886 (flatter, by locality-conditioned merchant choice) |
| agentic share | **0.1503** (target 0.15) |
| geo missing rate | 0.0956 overall; 0.2201 on remote rails against a 0.22 target |

**State plainly:** `data/reference/` today contains `fraudTrain.csv` and
`fraudTest.csv` (Kaggle Sparkov, 501 MB, **gitignored**, fetched by
`scripts/fetch_reference.py` on Day 7). They are used to **measure** the population
in the fidelity scorecard and are **never used to refit it** — refitting would
re-roll every pinned number. `scripts/fit_reference.py` exists and would fit shape
parameters from that CSV, but was not run; every number above comes from the
committed Indian-market priors.

What is fitted / parameterised: per-MCC log-normal amounts (26 MCC profiles, each
with `median`, `sigma`, `agentic_affinity`), the diurnal hour curve blended toward
uniform on the agentic share, merchant popularity as a Zipf rank-frequency curve
(exponent 1.08), per-MCC channel mix, metro-anchored geography, and a standing
entity map (customer / card / device / merchant / agent) built once per seed.
Velocity is not "fitted" — it emerges from the arrival process.

Lifecycle behaviour (Amendment 1.1.0):

| property | prior | realised (seed 1337, n=200,000, `scripts/drift_check.py`) |
|---|---|---|
| decline rate, card_present | 0.028 | 0.0304 (×1.09, n=53,800) |
| decline rate, ecom | 0.115 | 0.1525 (×1.33, n=44,495) |
| decline rate, moto | 0.140 | **0.3254 (×2.32, n=919)** |
| decline rate, recurring | 0.095 | 0.1213 (×1.28) |
| decline rate, upi_p2m | 0.052 | 0.0494 (×0.95) |
| decline rate, upi_p2p | 0.061 | 0.0880 (×1.44) |
| decline rate, agentic | 0.128 | 0.1581 (×1.24) |
| overall | 0.074 mix-weighted nominal | 0.088 |
| refunds | 2.1% share, bound to real earlier purchases, 68% full-value, median lag 62 h | — |
| reversals | 0.6% share, never settle by definition | — |
| credits | 0.16% share, orphan-free | — |
| pre-auths | 2.4% share | — |
| disputes | 0.09% of settled purchases | realised 0.0834% |
| unsettled | 1.2% prior | realised 2.60% (reversals never settle, so realised sits above prior by definition) |
| settlement lag | **bimodal**: card rails median 25–34 h, UPI 0.03–0.09 h, agentic 22 h; `ln` sigma 0.55 | every per-rail KS inside its bootstrapped noise band |

### 5.2 Injector framework

```python
class BaseAttack(ABC):
    card_id: ClassVar[str]
    base_events: ClassVar[int] = 120
    slice_columns: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def probe_slice(cls, frame: pd.DataFrame) -> np.ndarray | None: ...

    @abstractmethod
    def inject(self, population: pd.DataFrame, intensity: float,
               rng: np.random.Generator) -> pd.DataFrame: ...
```

`REGISTRY: dict[str, type[BaseAttack]]`. **`validate_registry()` runs at package
import and fails the import** unless the atlas and the code agree in both
directions (§4.2). `n_events(intensity)` scales `base_events` by the actual
background size against a 200k reference, so prevalence — not raw count — is the
invariant. Injectors return **only new rows** and never mutate the background.
Attack events are **clones of real background rows**, retargeted; amounts are
resampled from the population's own per-MCC empirical band.

Atlas id → module mapping is the card's own `generator` field, e.g.
`F1-01` → `mantis.foundry.injectors.f1_01_cart_tampering:inject`.

### 5.3 Per-implemented-injector table

Provenance: `data/generated/dataset_v1.manifest.json`, produced by
`python -m mantis.foundry --attacks all --out data/generated/dataset_v1.parquet
--show-content`, seed 1337, 200,000 background + 2,120 attack events
(prevalence **1.0489%**). Gate: `probe.GATE_AUC = 0.95` applied to the **in-slice**
number where a slice is declared. `THIN_SLICE_ROWS = 2,000`.

| card | bucket | probe_slice | slice n | in-slice AUC | in-slice feature | uncond. AUC | uncond. feature |
|---|---|---|---|---|---|---|---|
| F1-01 | CLEAN | agentic block present (`ag_agent_id` non-null) | 30,040 | 0.859 | `ag_deliberation_latency_ms` (hi) | 0.968 | `ag_tool_call_count` |
| F1-02 | HARD | agentic block present | 30,040 | 0.849 | `ag_mandate_type=intent` (hi) | 0.977 | `ag_mandate_type=intent` |
| F1-03 | CLEAN | agentic block present **and** `txn_type=refund` | **620 !** | 0.833 | `ag_deliberation_latency_ms` (hi) | 0.989 | `txn_type=refund` |
| F1-04 | HARD | agentic block present | 30,040 | 0.849 | `ag_mandate_type=intent` (hi) | 0.977 | `ag_mandate_type=intent` |
| F1-05 | HARD | agentic block present | 30,040 | **0.939** | `ag_delegation_depth` (hi) | 0.991 | `ag_delegation_depth` |
| F1-09 | HARD | agentic block present | 30,040 | 0.772 | `ag_human_present` (hi) | 0.965 | `ag_human_present` |
| F1-10 | HARD | agentic block present | 30,040 | 0.832 | `mandate_age_seconds` (hi) | 0.974 | `mandate_age_seconds` |
| F2-13 | n/a | none (whole population) | 202,120 | — | — | 0.683 | `ts_epoch` |
| F2-16 | n/a | none | 202,120 | — | — | 0.627 | `ts_epoch` |
| F3-19 | n/a | none | 202,120 | — | — | 0.872 | `amount` |
| F4-27 | n/a | none | 202,120 | — | — | 0.698 | `auth_response=approved` (**lo**) |
| F4-28 | n/a | none | 202,120 | — | — | 0.758 | `amount` |
| F6-38 | n/a | none | 202,120 | — | — | 0.787 | `mcc=6012` |
| F6-39 | n/a | none | 202,120 | — | — | 0.792 | `amount` |
| F6-40 | n/a | none | 202,120 | — | — | 0.677 | `amount` |
| **ALL fraud** | — | none | 202,120 | — | — | **0.686** | `ag_scope_max_amount` |

`!` = thin slice (under 2,000 rows). **F1-03's 0.833 must always be quoted with
its n=620.** Every row passes the gate. Note F4-27's direction is `lo` — the
probe reports `max(a, 1-a)`, so before the `direction` column was added the row
read as "approved more often" when the truth is the reverse.

Rail concentration in the same manifest: agent-mediated volume 31,113 carrying
1,073 fraud (3.45%); classic volume 171,007 carrying 1,047 fraud (0.61%) — a
**5.7× concentration**, 15.4% of volume carrying 50.6% of the fraud.

Runtime: simulate 7.60 s, index 0.43 s, inject 1.72 s, schema-validate 0.08 s
(2,120 events round-tripped), probe 34.0 s.

### 5.4 LLM layer

| property | value |
|---|---|
| model actually used | **`mistral:7b-instruct-q4_K_M`** — not the `qwen2.5:7b` the plan named. `DEFAULT_MODEL` matches deliberately: the cache key is `sha256(model \| prompt \| params)`, so a mismatched default would give a judge a cache miss on every prompt |
| params | `temperature 0.6, top_p 0.9, num_predict 320` |
| host | `OLLAMA_HOST` env or `http://127.0.0.1:11434`; probe timeout is short and every failure is caught |
| cache | `data/cache/llm/` — **234 files, 1.1 MB, committed** |
| corpus | `data/cache/content/corpus.jsonl` — **234 artefacts, 249 KB**: 138 benign / 96 adversarial |
| kinds | benign_page 72, benign_transcript 36, merchant_copy 30, injected_page 24, agent_transcript 18, refund_ticket 18, injected_review 12, scam_script 12, shell_merchant_copy 12 |
| bindings | `data/cache/content/bindings.jsonl` — **467 bindings** at HEAD |
| dependencies | standard library only (`urllib`) |

Three-stage degradation ladder, in order: **live Ollama → committed disk cache →
bundled deterministic corpus.** `generate()` cannot fail, cannot hang, and opens no
socket unless explicitly told it may. Cache-hit verification: every one of the 234
corpus rows carries `source: "cache"` when read back, and `make corpus` runs with
no network.

`ContentStore` joins `ag_ingested_content_ids` to text in two tiers: explicit
bindings for planted payloads, and deterministic assignment into the benign pool
for everything else, so **every id in the parquet resolves** — on attack rows and
legitimate ones alike. Resolve rate is therefore 100% by construction, which is
exactly the property that hid the Day 5 bindings bug (§11) and, as measured today,
still hides a live regression (§10).

### 5.5 Population changes made specifically to stop an attack being free

| change | value | attack it de-trivialises | before → after |
|---|---|---|---|
| `human_present_passive_share` — people who watch the agent work without touching the device | 0.11 (realised 0.1153) | F1-09 | `(human_present=True, machine-like cursor entropy)` was a **perfect** detector → 16.7% recall at 3.7% FP |
| `refund_instant_share` — instant refunds as a real merchant offering | 0.24 (realised 0.2129 on card rails) | F1-03 | `settlement_lag_hours` alone **0.996 → 0.83** |
| `delegation_depth_weights` extended to depth 5 | depth ≥ 4 at 0.015 (realised 0.01508) | F1-05 | `depth >= 4` was a **perfect** detector → 0.94 (still the closest number in the atlas to the 0.95 gate) |
| Provenance planting made **length-preserving** (it originally extended the chain) | — | all F1 | `ag_provenance_chain_len` **0.96 → not the top feature** |
| `collapse_deliberation` resamples tool-call count from the background's own upper band instead of adding to the clone | — | F1-01 | `ag_tool_call_count` **0.96 → 0.968 unconditional / 0.859 in-slice** |
| `set_timestamps` redraws time-of-day from the background hour curve (blended 18% toward uniform), shifting whole bursts via `groups` | — | three Day 2 attacks | `ts_hour` was the strongest single feature → no longer appears |
| `spread_epochs` stratified rather than uniform | — | Day 2 attacks | removed a `ts_epoch` calendar-gap signal |
| Agents run on-device ~50% of the time | — | all agentic | `device_id` rail AUC **1.000 → not a separator** |
| Graded Beta agentic propensity replacing a binary adopter flag | — | all agentic | `customer_id` AUC **0.90 → 0.75** |
| Agentic 3DS mix given a failure tail | — | all agentic | `threeds_result` AUC **0.86 → 0.75** |
| F2-13 cohort no longer filtered to a late first-seen date | — | F2-13 | `ts_epoch` **0.81 → 0.683** |
| F3-19 amount cap varied rather than fixed | — | F3-19 | removed a single repeated number pinning most of a campaign |

### 5.6 The derived-feature probe

`scripts/probe_derived.py` (`make derived`) runs the separability gate over the
**built 232-feature matrix** rather than raw columns, closing the blind spot that
let `mnd_deliberation_residual_z` reach 0.99 on F1-01 unnoticed. It **flags and
ranks rather than passing or failing**; verdicts live in `ADJUDICATED` at the top of
the script, and an unadjudicated finding exits non-zero.

| card | feature | AUC | adjudication |
|---|---|---|---|
| F1-04 | `mnd_mcc_in_scope` | 1.000 | **definitional** — being outside the mandated category *is* category drift |
| F6-39 | `ent_mcc_amount_z` | 0.989 | legitimate mechanism, **narrow generator** — `mcc=7832` alone scores 0.904 off six declared MCCs |
| F1-01 | `mnd_amount_over_ceiling` | 0.980 | **legitimate** — the card's own declared L1 signal, and it is the ratio not the breach flag, so F1-01 stays CLEAN |
| F6-40 | `txn_round_score` | 0.966 | **artefact** — the raw binary flag is unremarkable, the graded score is not; the injector snaps harder than a real ring would |
| F1-05 | `mnd_delegation_depth` | 0.952 | known and already priced; the raw column is 0.94 and the card's docstring concedes depth alone will not carry it |

Two foundry items fall out and are **recorded rather than done**: widen
`f6_39_shell_merchant._DECLARED_MCCS`, and soften F6-40's round-number snapping.
Both re-roll pinned numbers.

---

## 6. Fidelity

Provenance for everything in this section: `python -m mantis.foundry.fidelity`,
seed 1337, written to `data/generated/fidelity.json` at 2026-08-25T14:06:32Z,
rendered into `RESULTS.md` by `report.py`.

| panel | detail |
|---|---|
| reference | Kaggle `kartik2112/fraud-detection` (Sparkov): `fraudTrain.csv` + `fraudTest.csv`, 1,852,394 rows, **cut to a contiguous 90-day window** 2019-12-17…2020-03-15 → 200,051 rows. US panel, USD, itself synthetic |
| synthetic | `dataset_v1.parquet`, 202,120 events total, **162,338 compared** — classic rails, purchases only; the agentic rail is excluded because the reference panel has none |
| comparison space | a dimensionless **shape space** of 8 features; nothing is compared raw |

Levels are reported with **no distance attached**, because a ratio between two
panels' composition is a fact about how each was built:

| level | synthetic | reference | ratio |
|---|---|---|---|
| events compared | 162,338 | 200,051 | 0.81× |
| days | 89 | 89 | 1.00× |
| cardholders | 4,947 | 918 | 5.39× |
| merchants | 10,590 | 693 | 15.28× |
| categories | 26 | 14 | 1.86× |
| txn / cardholder / day | 0.369 | 2.449 | 0.15× |
| median hours between | 27.81 | 4.69 | 5.93× |
| top-1% merchant share | 0.322 | 0.019 | 16.58× |

### 6.1 Marginals, each against its own bootstrapped noise band

**There are only 8 shape features, so "worst 10" does not exist** — 4 continuous
(KS) and 4 categorical (JSD). All of them, worst first. A ratio of 1.0 means
indistinguishable from sampling noise.

| feature | metric | distance | noise band | × band | detail |
|---|---|---|---|---|---|
| `hour` | JSD | 0.07215 | 0.0000438 | **1,649.1** | 24 levels; worst level "11", delta 0.0331 |
| `dow` | JSD | 0.01148 | 0.0000178 | 643.0 | 7 levels; worst level "6", delta 0.0614 |
| `burst_1h` | JSD | 0.00264 | 0.0000060 | 438.2 | 2 levels; delta 0.0426 |
| `merchant_rank_pct` | KS | 0.52336 | 0.005445 | 96.1 | median 0.036 vs 0.383; IQR 0.164 vs 0.448 |
| `category_shift` | JSD | 0.00046 | 0.0000061 | 74.7 | 2 levels; delta 0.0149 |
| `gap_ratio_log` | KS | 0.09094 | 0.005445 | 16.7 | median 0.0 vs 0.0; IQR 1.889 vs 1.807 |
| `amount_vs_customer` | KS | 0.08278 | 0.005445 | 15.2 | median −0.054 vs 0.182; IQR 1.634 vs 1.905 |
| `log_amount_z` | KS | 0.03255 | 0.005445 | 6.0 | median 0.0 vs 0.0; IQR 1.002 vs 0.875 |

Median ratio across the 8: **85.4×**.

**Correlation**: Spearman matrices differ by **RMS 0.0760** off-diagonal,
**Frobenius 0.5684** over 8 features. Worst pairs — `amount_vs_customer ×
merchant_rank_pct` −0.004 vs −0.259 (Δ 0.255); `hour × merchant_rank_pct` −0.003 vs
+0.166; `hour × amount_vs_customer` +0.004 vs −0.158.

### 6.2 TSTR

| model | trained on | tested on | AUC-PR | ROC | n | n_pos | baseline | lift |
|---|---|---|---|---|---|---|---|---|
| **TRTR** | real | real | 0.8353 | 0.9878 | 60,016 | 413 | 0.00688 | 121× |
| **TSTR** | synthetic | real | 0.0249 | 0.6385 | 60,016 | 413 | 0.00688 | 3.6× |
| **TRTS** | real | synthetic | 0.0098 | 0.5175 | 162,338 | 1,137 | 0.00700 | — |

**Transfer ratio 0.0298.** The gain tables say why, and the honest reading is that
TSTR here measures *whether the two panels' fraud is the same phenomenon* — it is
not, by construction:

| feature | gain, trained on real | gain, trained on synthetic |
|---|---|---|
| `log_amount_z` | 57.6% | 17.1% |
| `amount_vs_customer` | 16.5% | 16.0% |
| `hour` | 13.0% | 6.7% |
| `merchant_rank_pct` | 6.3% | 35.8% |
| `gap_ratio_log` | 4.9% | 15.8% |

Sparkov's fraud is an amount anomaly; MANTIS's classic-rail attacks were built so
no single raw column separates them above 0.95 AUC. TRTS at ROC 0.517 confirms the
symmetry. Caveat carried in the JSON: **TSTR is measured on classic-rail fraud
only**, so no number here is evidence about the agentic attacks.

### 6.3 Discriminator

Label synthetic 1 / real 0, gradient-boosted tree, **5 folds, 162,338 per side**,
scored out of fold on balanced subsamples. **Target 0.5; higher is worse.**

| input | AUC | separability | reading |
|---|---|---|---|
| raw columns — ids, currencies, taxonomies included (**the naive rerun**) | 0.99986 | — | expected, **not a finding**: two identifier namespaces separate at 1.0 alone |
| **shape space — intersection columns, ids dropped, agentic excluded** | **0.99941** | 99.9% | the headline |
| shape space, two adjudicated axes removed | **0.83985** | 68.0% | the measurement after a judgement a reader may reject |

**The intersection-columns rerun bought almost nothing, and that is the important
result.** If 0.9994 had been an artefact of identifiers or currency, removing them
would have collapsed it. It moved by 0.00045. Per-column rank-AUC in the naive
run: `amount` 0.965, `customer_code` 0.872, `merchant_code` 0.721,
`category_code` 0.595, `hour` 0.564.

Top-5 drivers, attribution via LightGBM native `pred_contrib`:

| rank | feature | alone (AUC) | gain | contribution | class |
|---|---|---|---|---|---|
| 1 | `merchant_rank_pct` | 0.8288 | 79.3% | 70.7% | **structural** |
| 2 | `amount_vs_customer` | 0.5170 | 5.4% | 7.0% | **structural** |
| 3 | `hour` | 0.5643 | 7.3% | 6.0% | **cosmetic** |
| 4 | `gap_ratio_log` | 0.5302 | 3.2% | 5.6% | **structural** |
| 5 | `burst_1h` | 0.5211 | 1.7% | 4.7% | **structural** |
| 6 | `log_amount_z` | 0.5002 | 2.7% | 3.4% | cosmetic |
| 7 | `dow` | 0.5056 | 0.3% | 2.1% | cosmetic |
| 8 | `category_shift` | 0.5075 | 0.1% | 0.5% | structural |

4 of the top 5 are structural. Definitions: **cosmetic** = a surface property of
how values were *rendered* (identifier shape, timestamp granularity, amount
rounding); **structural** = a property of the joint distribution. Fixing a
structural one means changing the generative model.

Drop-one ablation path — **it degrades gradually, so the separation is not one bad
column**:

| features left | AUC | dropped so far |
|---|---|---|
| 8 | 0.9994 | — |
| 7 | 0.8833 | `merchant_rank_pct` |
| 6 | 0.8416 | + `gap_ratio_log` |
| 5 | 0.7076 | + `amount_vs_customer` |
| 4 | 0.6172 | + `hour` |
| 3 | 0.5883 | + `dow` |
| 2 | 0.5231 | + `log_amount_z` |

**Adjudication rule** (`adjudicate.py`): a divergence may be attributed to the
reference panel only when a **third quantity, independent of both panels and stated
in advance**, says so.

| feature | third quantity | synthetic | reference | verdict |
|---|---|---|---|---|
| `hour` | retail spend has a diurnal curve with an overnight trough | peak/trough **22.5×** | peak/trough **1.6×** | reference |
| `merchant_rank_pct` | an acceptance estate is Zipf, not uniform | top 10% carry **66.0%**, max/min 2,980× | top 10% carry **14.6%**, max/min 6× | reference |

Both discriminator numbers are always reported, because the ablation is a judgement.

### 6.4 The 33-marginal bootstrapped noise-band result

A **separate** artefact from the scorecard: `scripts/drift_check.py` (`make drift`),
`docs/drift_check.json`, n=200,000, seed 1337, reference `indian-market-priors`. It
compares the population against **its own priors**, not against Sparkov. 33
marginals, KS for continuous and JSD for categorical, each against a band
bootstrapped from the prior at that sample size.

**29 of 33 are inside their band.** The four outside:

| marginal | distance | band | × band | note |
|---|---|---|---|---|
| `decline_reason \| declined` | 0.03400 | 0.000113 | **302.0** | reason remapping — see below |
| `amount` | 0.00799 | 0.003645 | 2.19 | non-zero **by design**: round-number snapping |
| `settlement_lag \| upi_p2p` | 0.02448 | 0.019105 | 1.28 | lognormal draw, bimodal across rails |
| `mcc` | 0.0000447 | 0.0000402 | 1.11 | drawn directly from `mcc_profiles` weights |

Three widened Day 3 tails verify inside their targets: instant refunds 0.2129 vs
0.24 on card rails, passive humans 0.1153 vs 0.11, delegation depth ≥ 4 at 0.01508
vs 0.0150. Three measurement bugs **in the script** were found and fixed first:
entity-drawn columns need an entity-level null band (or Zipf popularity makes
ordinary noise read as 339× drift); the day-of-week target must be calendar-weighted
since 90 days is not a whole number of weeks; and the passive-human share must be
deconvolved from the cursor-entropy mixture rather than thresholded.

### 6.5 The two known divergences — measured, named, deferred

**1. `decline_reason` remapping — 302× its band.**
`invalid_cvv` 0.130 → 0.036; `expired` 0.080 → 0.033. Reasons are remapped where
the entry mode makes them impossible: `invalid_cvv` → `do_not_honor` where no CVV
was presented, `expired` → `insufficient_funds` where the mode cannot expire. Only
~27% of declines are on a CVV-bearing entry mode, so **73% of drawn `invalid_cvv`
gets remapped** — far larger than the code comment claimed.
*Why deferred:* conservative for detection — it raises the background rate of the
reasons F4-27 farms, which makes that attack's lift **smaller**, not larger.
Re-tuning re-rolls every pinned calibration number.

**2. Realised decline rates above prior on every rail.**
moto **×2.32** (0.3254 vs 0.140), upi_p2p ×1.44, ecom ×1.33, recurring ×1.28,
agentic ×1.24, card_present ×1.09, upi_p2m ×0.95; overall 0.088 against a
mix-weighted nominal 0.074. Cause: `decline_amount_tilt = 0.55` multiplies the
per-channel rate by `exp(0.55·z)`, whose expectation is `exp(0.55²/2) ≈ 1.16` —
**Jensen's inequality, not a redistribution.** The tilt should be mean-preserving
per channel and is not.
*Why deferred:* same direction and same reason — a higher decline background makes
the card-testing attacks **harder** to catch, and the fix re-rolls every pinned
number. Recorded as an outstanding item, not applied.

---

## 7. Defence

Pipeline: `TxEvent` → feature builder (232 features, 5 groups) → L0/L1/L2/L3/L4 →
fusion → policy → explanation.

Feature groups, from `python -m mantis.defense.features` run 2026-08-26 on
`dataset_v1.parquet`:

| group | prefix | count |
|---|---|---|
| transaction | `txn_` | 20 |
| velocity | `vel_` | 136 |
| entity | `ent_` | 14 |
| mandate | `mnd_` | 27 |
| graph | `gph_` | 28 |
| categorical passthrough | — | 7 |
| **total** | | **232** |

Velocity keys: `customer, card, bin, device, merchant, agent, mandate_hash, ip` ×
windows `1h, 24h, 7d`, over a keyed rolling state store (`features/state.py`) —
one forward pass, `bisect` + prefix sums, bounded memory by eviction. Measured
**0.073–0.075 ms/row** for the whole builder on a 60,636-row transform.

### 7.1 L0 — deterministic protocol clauses

**What it is:** nine boolean clauses over the mandate and the authorisation
message. No model, no training data, no fit step. Returns a **named reason**.
**Consumes:** the raw event. **Catches:** protocol violations, on any attack,
seen or unseen.

Measured on `dataset_v1.parquet` (202,120 events, 2,120 fraud, 31,113 carrying an
agentic block) by `python -m mantis.defense.l0_rules`, re-run 2026-08-26:

| clause | reason | fires | precision | FP rate (all legit) | FP / legit agentic | status |
|---|---|---|---|---|---|---|
| `scope_mcc` | cart category outside the signed intent scope | 153 | 1.000 | 0.00000 | 0.00000 | operative |
| `scope_merchant` | merchant outside the mandate allow-list | 0 | n/a | 0.00000 | 0.00000 | operative |
| `amount_over_cap` | amount exceeds the mandate ceiling | 36 | 1.000 | 0.00000 | 0.00000 | operative |
| `mandate_expired` | mandate TTL elapsed before presentation | 77 | 1.000 | 0.00000 | 0.00000 | operative |
| `delegation_depth` | delegation deeper than 5 hops | 3 | 1.000 | 0.00000 | 0.00000 | operative |
| `consent_invalid` | consent signature failed verification | 165 | 0.394 | 0.00050 | 0.00333 | operative |
| `kya_unregistered` | agent absent from the KYA registry | 922 | 0.084 | 0.00422 | **0.02813** | operative |
| `mandate_missing` | agent-mediated payment with no mandate | 0 | n/a | 0.00000 | 0.00000 | operative |
| `provenance_mismatch` | trail does not end at the merchant paid | 0 | n/a | 0.00000 | 0.00000 | operative |
| `provenance_untrusted_domain` | page outside the reputation allow-list | 270 | 1.000 | 0.00000 | 0.00000 | **DECLARED ONLY — switched off** |
| **ANY operative** | | **1,303** | **0.279** | 0.00470 | 0.03126 | |

`kya_unregistered` at 2.8% of legitimate agentic traffic is the population's
deliberate messy tail, not a defect. `provenance_untrusted_domain` catches 100% of
both CLEAN attacks at 0.00% FP and is **excluded**, because the foundry draws
attacker URLs from twelve hosts that appear nowhere in legitimate traffic — that is
a partition the generator created, and trusting it would let L3 post a recall it
had not earned by reading a word. `MAX_DELEGATION_DEPTH = 5`.

Per-attack L0 recall: F1-04 1.000, F1-10 0.718, F1-09 0.631, F1-02 0.608,
F2-13 0.094, F1-05 0.050, F4-27 0.025, F2-16 0.020; **F1-01 and F1-03 at 0.000**;
F3-19, F4-28, F6-38, F6-39, F6-40 at 0.000.

**The bucket-contract verdict.** Day 3 asserted CLEAN attacks trip zero clauses at
zero tolerance and HARD attacks fire a clause on ≥25% of events.
- CLEAN holds exactly: F1-01 **0.00%**, F1-03 **0.00%** of nine clauses.
- HARD passes on four: F1-04 100.00%, F1-10 70.00%, F1-09 47.69%, F1-02 44.17%.
- **F1-05 FAILS at 3.00%.**

**The contract is wrong, not L0.** The Day 3 test satisfied F1-05 with
`delegation_depth > 2` and never priced it. Priced:

| threshold | F1-05 recall | FP on legit agentic | FP count |
|---|---|---|---|
| depth > 2 | 0.640 | 0.04987 | 1,498 |
| depth > 3 | 0.290 | 0.01508 | 453 |
| depth > 4 | 0.150 | 0.00316 | 95 |
| **depth > 5** | **0.030** | **0.00000** | **0** |

No issuer declines 5% of legitimate agent-mediated authorisations, so `depth > 2`
is not an L0 clause — it is a weak classifier with a rule's syntax. F1-05's own
docstring already conceded depth alone will not carry the card and named L4 as the
real answer. **Neither side was adjusted to agree**;
`tests/test_l0_rules.py` pins the exception so it stays noisy, and the CLI exits 0
on this one documented failure and non-zero on any other.

### 7.2 L1 — supervised GBDT

**What it is:** LightGBM binary classifier, isotonic-calibrated.
**Consumes:** all 232 features. **Catches:** everything it has a labelled history
for; collapses on families it has not (§8.3).

| parameter | value |
|---|---|
| `objective` | binary |
| `learning_rate` | 0.05 |
| `num_leaves` | 31 |
| `max_depth` | 6 |
| `min_child_samples` | 40 |
| `feature_fraction` | 0.7 |
| `bagging_fraction` / `bagging_freq` | 0.8 / 1 |
| `lambda_l2` | 5.0 |
| `n_estimators` | 400 |
| `scale_pos_weight` | **left alone** — natural distribution; the threshold does the work |
| seed | 1337 |

**Split strategy:** time-based at the 70% quantile of `ts` — never random.
**Calibration:** `IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)` fitted
on `CALIBRATION_SHARE = 0.2` taken from the **end** of the training window, for the
same reason the main split is time-based.
**Threshold selection:** `threshold_at_fpr(scores, labels, fpr)` places the
operating point at a quantile of the **legitimate** score distribution. Reporting
grid `FPR_GRID = (0.001, 0.005, 0.010)`.

Top gain shares (five-seed pool): `mcc` 5.68%, `mnd_mcc_in_scope` 4.98%,
`ent_merchant_customers` 4.77%, `mnd_amount_over_ceiling` 4.06%,
`ent_mcc_amount_z` 3.28%, `ent_customer_n_events` 2.98%,
`vel_agent_amount_vs_mean` 2.98%, `txn_round_score` 2.88%,
`ent_merchant_refund_ratio` 2.85%, `channel` 2.80%.

### 7.3 L2 and L2e — the negative result

**L2:** `IsolationForest(n_estimators=300, max_samples=min(50_000, n),
contamination=0.01, random_state=1337, n_jobs=-1)`, fitted on **legitimate
training rows only** (asserted). Columns filtered to
`MIN_PRESENT_SHARE = 0.70` present on legitimate rows, then median-imputed.
**Both the missingness cut and `contamination` are fixed from the missingness
histogram alone and never tuned against recall** — tuning either would make the
layer supervised through the back door.

**L2e:** the time-boxed 30-minute experiment — the same forest over **entity**
aggregates (customers and merchants) rather than event rows.

| layer | AUC-PR | ROC | recall@0.1% | realised FPR | campaign recall |
|---|---|---|---|---|---|
| L2 | 0.0151 | 0.5785 | 0.002 | 0.1000% | 0.031 (4/131) |
| **L2e** | 0.0098 | **0.4421** | 0.000 | 0.1881% | 0.008 (1/131) |

**L2e's ROC of 0.4421 is below chance.** Published explanation, in four facts:
entity aggregates are *anti-correlated* with fraud here; the attacks ride
established customers and busy merchants by construction; the genuinely unusual
entities are ordinary people with three transactions; L2e is also **generous** to
the hypothesis, since an entity's vector is aggregated over the whole scoring
window (a nightly entity-risk queue, not an authorisation scorer) and it still did
not work. On a smaller single-seed run merchant-side entity scoring showed ROC
0.66; it does not survive pooling.

General statement: **attacks built to be distributionally faithful are by
construction invisible to distributional anomaly detection** — so a fidelity
scorecard and an anomaly-detection recall number are in tension by construction.
L2's surviving claim: **its recall is unaffected by whether an attack was in
training**, a property no supervised layer has.

### 7.4 L3 — page classifier over ingested text

**Architecture:** TF-IDF (`lowercase=True, ngram_range=(1,2), min_df=1,
sublinear_tf=True, strip_accents="unicode"`) + `LogisticRegression(max_iter=2000,
C=4.0, class_weight="balanced", random_state=1337)` over the 234-artefact corpus.
An **event's** score is the **worst page its agent read** (max over the chain), with
the runner-up used to break the heavy ties a max over ~230 discrete probabilities
produces. Decision threshold 0.5.

**`L3Model.fit` has no `y` parameter.** Its label is the artefact's own `injected`
flag — a property of text, not of a transaction. It never sees `is_fraud`.

Two hold-out protocols:
1. **Unseen phrasing** — the highest-numbered variant of every adversarial kind is
   withheld from the vocabulary and from training.
2. **Unseen kind** — every `refund_ticket` specimen is withheld, all 18 of them, so
   F1-03 is scored on an injection type never seen in any wording.

| card | n_pos | recall | on unseen phrasing | n unseen | on an unseen kind |
|---|---|---|---|---|---|
| F1-01 | 249 | 1.000 | 1.000 | 134 | 1.000 |
| F1-02 | 205 | 0.020 | n/a | 0 | 0.020 |
| F1-03 | 201 | 1.000 | 1.000 | 93 | 1.000 |
| F1-05 | 174 | 0.017 | n/a | 0 | 0.017 |

34 artefacts never in the vocabulary or the training set score P(injected) 0.24–0.91,
median 0.76.

**Out-of-distribution probe** — `scripts/l3_ood.py`, `data/generated/l3_ood.json`.
18 hand-authored injection payloads and **10 hand-authored benign controls in the
same registers** (HTML comments, YAML, a fake system banner, transliterated
Hindi-English, shouting, txt-speak, Cyrillic homoglyphs), written by a different
model from the one that authored the corpus and committed before they were scored.

| | recall | FP on controls | ROC | n+ / n− |
|---|---|---|---|---|
| in distribution (the corpus) | 0.917 | **0.000** | 0.9986 | 96 / 138 |
| **out of distribution** | 1.000 | **0.900** | **0.8111** | 18 / 10 |

**The benign-controls design is what makes this readable.** Without them the table
reads as a triumph. Stated precisely:

> **L3's decision threshold does not transfer.** Calibrated on one corpus and
> pointed at text unlike it, the layer fires on 100% of the novel injections *and*
> on 90% of the clean controls written in the same registers. What survives is the
> **ordering**: ROC 0.9986 → 0.8111. Re-placing the threshold above the worst
> control (0.7372) recovers 50% recall — an oracle number, since that threshold has
> seen the answer, but enough to locate the defect in **calibration** rather than in
> an absence of signal.

Named fix, **not done**: fit the page threshold on benign text drawn from the
traffic it will see. Longer term a bag of words is the wrong model — it keys on
lexical markers of instruction (*do not*, *skip*, *without*), which is why prose
that merely sounds procedural trips it.

### 7.5 L4 — the entity graph, streamed

**What it is:** 28 streamed graph features, `gph_` prefix. Union-find over an
**identity** graph (customer ↔ device ↔ agent) plus windowed distinct counters for
the merchant and BIN sides. **Read-then-fold per event**, exactly like the velocity
store — every event's features are read off the graph as it stood *strictly
before* it, so it is backward-looking by construction and needs no fit/transform
split. Identical code online and offline.

The 28 features: `gph_agent_customers`, `gph_agent_merchants`,
`gph_bin_customers_1h`, `gph_bin_merchants_1h`, `gph_bin_merchants_24h`,
`gph_component_agentic_share`, `gph_component_amount_mean`,
`gph_component_customers`, `gph_component_decline_ratio`, `gph_component_devices`,
`gph_component_events`, `gph_component_events_per_customer`, `gph_component_nodes`,
`gph_customer_bins`, `gph_customer_devices`, `gph_customer_events`,
`gph_customer_merchants`, `gph_device_customers`, `gph_device_merchants`,
`gph_merchant_components_7d`, `gph_merchant_customers`, `gph_merchant_devices`,
`gph_merchant_fanin_24h`, `gph_merchant_fanin_7d`, `gph_merchant_fanin_burst`,
`gph_merchant_fanin_per_component`, `gph_new_edge_share`,
`gph_pair_customer_merchant_prior`.

**Per-row cost — two different measurements, both real:** the Day 5 batch build
reported **0.021 ms/row**; today's `latency.json` batch column reports
**0.1163 ms/row** and the single-event streamed p99 is **0.381 ms**. The two
benches differ in machine state and batch size; quote the one whose provenance you
state.

**What it lifted:** L1 AUC-PR **0.4910 → 0.5903** and recall@0.1%FPR
**0.3615 → 0.450** — the graph block is worth **+0.09 recall on its own** (Day 4
vs Day 5, same five-seed pool, same operating point).

**Merchants and BINs are excluded from the identity graph on purpose.** With 16
BINs and a Zipf merchant curve, one union through a popular merchant fuses the file
into a single giant component and `component_size` becomes a constant.
`tests/test_l4_graph.py` pins the *property*: the largest identity component must
hold under 10% of nodes. Merchant-side structure is measured as windowed distinct
payers and as the number of distinct identity **components** paying one merchant —
the ratio between those two is the ring detector.

### 7.6 Fusion

**Consumes SCORES, never thresholded DECISIONS.** Each layer contributes up to
three columns:
1. its **percentile** against the legitimate score distribution,
2. its **raw score standardised** on the fusion window's legitimate rows,
3. where the layer is sometimes silent, an **indicator for whether it had an
   opinion**.

A NaN layer score maps to the **median legitimate percentile (0.5)** — "no
opinion", not "clean". No threshold is applied to any layer before fusion; L3's
page threshold is used for *reporting* L3 standalone and for nothing else.

**Weighting method:** `LogisticRegression(max_iter=1000, C=1.0,
class_weight="balanced", random_state=1337)` fitted on an inner slice of the
training window — `INNER_TRAIN_SHARE = 0.80`, so the fusion weights are fitted on
the remaining 20% that **none of the base layers was fitted on**. Weights are
fitted, never hand-set.

**The percentile-saturation defect and its fix.** Day 5's first attempt was a
stacker over layer **percentiles alone**, and the fused score got *worse* than L1
again (0.104 vs 0.553 on the smoke dataset). Cause: a percentile against a finite
reference **saturates** — every score above the largest legitimate score maps to
exactly 1.0, and at a 0.1% FP budget the ranked events are precisely those. Fix:
add the standardised raw score as a second column per layer. Fused then beat L1
(0.565 vs 0.553 on the smoke set). Pinned by `tests/test_fusion_policy.py`.

**Fitted weights (five-seed pool):**

| layer | weight on percentile | weight on raw score | weight on "had an opinion" |
|---|---|---|---|
| L1 | **+5.723** | +0.381 | n/a |
| L2 | +0.789 | −0.225 | n/a |
| L2e | +0.428 | +0.233 | n/a |
| L3 | **−0.943** | **+0.353** | −0.268 |

L3's negative percentile weight and positive raw weight is the stacker saying it
trusts the layer's *ordering* while discounting the calibration that produced the
percentile — a discount a decision-consuming fusion could not have applied. This is
what makes the L3 OOD result survivable.

**Fused vs L1:** fused 0.483 vs L1 0.450 at 0.1% FPR (Day 5). Day 4's unweighted
noisy-OR was 0.286 vs L1's 0.361 — worse.

### 7.7 Policy

`Decision` is a four-level `StrEnum` with a severity order:

| decision | rank | trigger |
|---|---|---|
| `approve` | 0 | below every boundary |
| `challenge` | 1 | score ≥ the **1.0% FPR** boundary. Escalated to `review` when `human_present` is false — an unanswerable challenge has nobody to answer it |
| `review` | 2 | score ≥ the **0.5% FPR** boundary |
| `decline` | 3 | score ≥ the **0.1% FPR** boundary, **or any operative L0 clause fires** (L0 overrides all four) |

Boundaries are placed at **FPR budgets on legitimate traffic**, not at score
values, using the same `FPR_GRID = (0.001, 0.005, 0.010)` the recall curve is
reported over — so RESULTS.md and the policy cannot drift apart, and a retrain
re-prices nothing.

Test-window distribution (five-seed pool, 303,180 events): approve 298,170
(98.348%), challenge 1,515 (0.500%), review 1,593 (0.525%), decline 1,902 (0.627%).

**KYA reputation score: NOT DONE** — no such score exists anywhere in
`mantis/defense/policy/` or `mantis/api/`. `kya_registered` is a boolean schema
field consumed by the `kya_unregistered` L0 clause and by mandate features; there is
no reputation model.

### 7.8 Explanations

Per-event attribution from LightGBM's own `booster.predict(..., pred_contrib=True)`
— **not** a SHAP wrapper. For a tree ensemble this is the *same* computation
`TreeExplainer` performs, read from the source SHAP would have called, without a
wrapper on the scoring path. Contributions are in **log-odds of the raw margin**,
which is what the ranking, and therefore the alert, is made of. Values render as
themselves: categoricals stay strings, and a genuinely missing key prints
`absent`, not `nan`.

---

## 8. Results — every headline number with provenance

Unless stated otherwise, every figure in §8.1–8.5 comes from **one run**:
`python -m mantis.defense`, five pooled seeds, dateline 2026-08-25, rendered into
`RESULTS.md` by `mantis/defense/report.py`. Threshold placed so that a fixed share
of **legitimate test traffic** is flagged; the headline share is **0.1%**.
No accuracy figure appears anywhere.

### 8.1 Evaluation dataset

| | |
|---|---|
| seeds pooled | `1337, 7, 11, 23, 41` — 5 independently generated worlds, identifiers namespaced per seed, calendars offset |
| events | 1,010,600 |
| fraud | 10,600 (**1.0489%**) |
| train / test | 707,420 / 303,180, **time-based** split at the 70% quantile of `ts` |
| test positives | 3,317 → test prevalence **1.0941%**, which is the AUC-PR baseline |
| features | 232 (28 of them the `gph_` block) |
| artefact | `data/generated/pool_5seed.parquet` (84 MB) |

Positives per family **in the test window**:

| family | test n_pos | cards |
|---|---|---|
| F1 | 1,356 | F1-01 249, F1-02 205, F1-03 201, F1-04 164, F1-05 174, F1-09 202, F1-10 161 |
| F2 | 433 | F2-13 185, F2-16 248 |
| F3 | 158 | F3-19 158 |
| F4 | 567 | F4-27 336, F4-28 231 |
| F6 | 803 | F6-38 340, F6-39 252, F6-40 211 |
| F5 | **0** | zero-day holdout family, no injector, absent from every table |

### 8.2 Layer performance, and the recall curve

| layer | AUC-PR | ROC-AUC | recall@0.1% | recall@0.5% | recall@1.0% | realised FPR | campaign recall | first alert |
|---|---|---|---|---|---|---|---|---|
| **L1** GBDT | 0.5903 | 0.9248 | **0.450** | 0.572 | 0.623 | 0.1000% | 0.916 (120/131) | event 3 of 23 |
| **L2** iForest, events | 0.0151 | 0.5785 | 0.002 | 0.009 | 0.021 | 0.1000% | 0.031 (4/131) | event 12 of 23 |
| **L2e** iForest, entities | 0.0098 | 0.4421 | 0.000 | 0.003 | 0.005 | 0.1881% | 0.008 (1/131) | event 1 of 23 |
| **L3** page classifier | 0.1687 | 0.6919 | 0.138 | 0.149 | 0.167 | 0.1024% | 0.214 (28/131) | event 1 of 23 |
| **fused** logistic stacker | 0.5881 | 0.9119 | **0.483** | 0.580 | 0.606 | 0.1000% | 0.908 (119/131) | event 2 of 23 |

**Per rail** — because a headline across both rails is partly reading a field the
issuer gets free off the authorisation message:

| rail | n positive | AUC-PR | recall@0.1%FPR |
|---|---|---|---|
| agentic | 1,702 | 0.8088 | **0.606** |
| classic | 1,615 | 0.3912 | **0.318** |

Per attack card at 0.1% FPR (fused / campaigns caught): F1-01 0.984 / 1.000;
F1-02 0.688 / 1.000; F1-03 0.940 / 1.000; F1-04 0.982 / 1.000; F1-05 0.839 / 1.000;
F1-09 0.312 / 1.000; F1-10 0.752 / 1.000; F2-13 0.205 / 1.000; F2-16 0.121 / 0.429;
F3-19 0.544 / 0.889; F4-27 0.033 / 0.875; F4-28 0.165 / 0.786; F6-38 0.618 / 1.000;
F6-39 0.135 / 0.700; F6-40 0.422 / 1.000.

### 8.3 Leave one family out — all five columns

L1 is retrained with the family **entirely removed from the training set** and then
asked to catch it in the test set anyway. L2 never sees any attack; L3 never sees a
transaction; neither column changes with the hold-out.

| family | n_pos | L1 (trained WITH) | L1 (HELD OUT) | L2 | L3 | fused | fused, held out |
|---|---|---|---|---|---|---|---|
| F1 | 1,356 | 0.681 | 0.004 | 0.005 | 0.338 | 0.786 | 0.011 |
| F2 | 433 | 0.176 | 0.016 | 0.000 | 0.000 | 0.157 | 0.014 |
| F3 | 158 | 0.595 | 0.405 | 0.000 | 0.000 | 0.544 | 0.361 |
| F4 | 567 | 0.086 | 0.028 | 0.000 | 0.002 | 0.086 | 0.023 |
| F6 | 803 | 0.437 | 0.137 | 0.000 | 0.000 | 0.415 | 0.127 |
| **mean** | | **0.395** | **0.118** | 0.001 | 0.068 | 0.398 | 0.107 |

**Mean event-level recall lost when a family is held out: +0.277** (39.5% → 11.8%).

### 8.4 Campaign-level recall and first-alert index

| family | campaigns | median size | fused (with) | fused (held out) | first alert at event | elapsed before alert |
|---|---|---|---|---|---|---|
| F1 | 60 | 22 | 1.000 | 0.183 | 1 | 0% |
| F2 | 14 | 32 | 0.714 | 0.214 | 8 | 24% |
| F3 | 9 | 17 | 0.889 | 1.000 | 3 | 9% |
| F4 | 22 | 23 | 0.818 | 0.227 | 12 | 54% |
| F6 | 26 | 29 | 0.885 | 0.615 | 3 | 9% |

Overall: L1 campaign recall 0.916 (120/131) with median first alert at **event 3 of
23**; fused 0.908 (119/131) at **event 2 of 23**.

### 8.5 The recovery trio

Provenance: `python -m mantis.loop`, `data/generated/arena.json`, measured on the
**loop's own two-seed background** (`pool_2seed.parquet`, n_background 404,240) so
all three rows share one dataset and one operating point. Operating FPR 0.001.

| detector | recall@0.1%FPR on the **544 real F1 test events** |
|---|---|
| trained **with** family F1 | **0.8107** |
| family F1 **held out** of training | **0.0129** |
| held out, **plus 4,442 loop-manufactured F1 variant events** | **0.5386** |

`gap_closed = 0.659` — the loop recovers **65.9%** of the collapse.

**Exact test-set size: 544 real F1 events. Manufactured events: 4,442.**

Precisely what the loop did and did not have access to:

- **Had:** F1's atlas cards and their executable injectors — a written description
  of a class of attack and code that manufactures instances of it. A red team, not
  a fraud history.
- **Did not have:** a single one of the 544 real F1 rows it is then scored on.
- **The variants are not those rows.** Every gene moved them, and they were
  *selected for evading the detector*, so they sit off-distribution from the
  canonical attack in exactly the direction that makes transfer hard — which is why
  the recovery is 66% and not 100%.
- The Day 4 five-seed figures (0.569 trained / 0.007 held out) are the same
  experiment at a different scale, **not a directly comparable row**.

### 8.6 Arena

| | |
|---|---|
| command | `python -m mantis.loop` |
| generations | **5** (0–4) |
| population | **30 variants per generation**, 9,400 events per generation |
| cards | **6** — `F1-01, F1-05, F2-16, F3-19, F4-27, F6-38`, covering all five implemented families |
| background | 2 pooled seeds, 404,240 events |
| operating FPR | 0.001 |
| wall clock | **581.1 s** |
| evasion start → end | **0.626 → 0.381** |

**The default is six cards, not fifteen, and the reason is cost.** The arena's cost
is linear in cards × population × generations; all fifteen cards at population 8
over two pooled seeds runs about **two hours**, which is not a gate anybody re-runs.
`--cards all` is still available. A curve does not get truer by averaging over cards
that behave like the ones already in it.

| generation | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| mean evasion | 0.6264 | 0.3344 | 0.3378 | 0.3632 | 0.3811 |
| max evasion | 0.9975 | 0.7975 | 0.7775 | 0.7675 | 0.7775 |
| mean fitness | 0.4319 | 0.1608 | 0.1827 | 0.1755 | 0.1883 |
| detector threshold | 0.6667 | 0.9934 | 0.9985 | 0.9985 | 1.0000 |

Per card at generation 0 → 4: F1-01 0.4487 → 0.2220; F1-05 0.2390 → 0.0970;
F2-16 0.9350 → 0.6585; F3-19 0.5327 → 0.3073; F4-27 0.9500 → 0.6220;
F6-38 0.6528 → 0.3800.

**The curve falls and then rebounds** (0.626 → 0.334 → 0.338 → 0.363 → 0.381).
Almost the whole fall happens at the **first** retrain. The supported claim is the
bounded one — **retraining on manufactured variants cuts evasion by 39% and holds it
down** — not that it drives evasion to zero. `report._curve_reading` picks its prose
from the shape the curve actually has, so a future run that rises says so.

9 survivors, of which 6 were genuinely mutated and written back (§4.3) and 3 were
the unmutated identity genome and were not (§4.4).

### 8.7 Latency

Provenance: `python scripts/latency_bench.py`, `data/generated/latency.json`,
generated 2026-08-25T19:43. **400 events timed one at a time** against velocity and
graph state warmed on **282,968 training events** from `pool_2seed.parquet`. Budget
**50 ms**. Latency is the one measurement here that is not reproducible from a seed
— it is a property of the machine at that moment, and it was taken with nothing else
running.

| path | stages | p50 | p95 | **p99** | vs the 50 ms budget |
|---|---|---|---|---|---|
| **inline — L0 + L1** (pre-authorisation) | velocity, graph, transaction, entity, mandate, L0, L1 | 61.2 ms | 64.9 ms | **66.3 ms** | **over, by 1.33×** |
| **fast-follow** (post-authorisation) | L2, L3, fusion+policy | 53.5 ms | 58.1 ms | 62.1 ms | not governed by it |
| **full stack** | all ten | 115.7 ms | 121.1 ms | **125.0 ms** | **over, by 2.5×** |

*`CLAUDE.md` §8 "Day 7 gate" quotes p50 117 / p95 153 / p99 171 ms end-to-end. That
is a superseded run; `latency.json` at HEAD and `RESULTS.md` both carry
115.7 / 121.1 / 125.0. Use the latter.*

**The tiering argument, stated:** **inline** is what changes the *authorisation
decision* — the protocol clauses and the supervised score they escalate to, both
needing only the event and backward-looking state. **Fast-follow** informs actions
taken *after* the authorisation is answered; on the agentic rail the two that
matter are both post-authorisation in any real deployment — **mandate revocation**
(a credential-lifecycle operation against the mandate registry, not a field in the
authorisation response) and **agent quarantine** (which changes every future
authorisation the agent attempts and none of the one in flight). The **graph pass
is counted inline** despite L4 being fast-follow, because L1 consumes its 28
`gph_` features — inline as a feature source. The split is **asserted to partition
the stage list**, not computed by subtraction.

| stage | mean | p99 | share | per row in batch | overhead |
|---|---|---|---|---|---|
| `velocity` | 0.132 ms | **0.654 ms** | 0.1% | 0.1319 ms | — (cannot batch) |
| `graph` | 0.116 ms | **0.381 ms** | 0.1% | 0.1163 ms | — (cannot batch) |
| `transaction` | 6.971 ms | 8.553 ms | 6.0% | 0.0192 ms | 364× |
| `entity` | **45.909 ms** | 49.781 ms | **39.7%** | 0.1128 ms | **407×** |
| `mandate` | 4.745 ms | 5.239 ms | 4.1% | 0.0134 ms | 355× |
| `L0` | 0.779 ms | 0.951 ms | 0.7% | 0.0028 ms | 275× |
| `L1` | 3.086 ms | 3.448 ms | 2.7% | 0.0183 ms | 169× |
| `L2` | **43.174 ms** | 50.464 ms | **37.3%** | 0.1481 ms | 292× |
| `L3` | 0.041 ms | 0.084 ms | 0.0% | 0.0014 ms | 30× |
| `fusion+policy` | 10.801 ms | 11.708 ms | 9.3% | 0.0285 ms | 380× |

Batch total **0.5925 ms/row** for the whole stack; the inline stages alone cost
**~0.41 ms/row** in batch. So of the 66 ms measured at p99 inline, roughly 99% is
per-call framework overhead. Cause: `Series.map(dict)` materialises the lookup table
into an index on **every call**, so a fourteen-feature block pays that cost fourteen
times to look up fourteen values. **The fix is a plain dict lookup on the
single-event path, and it is not applied** — the feature builder is shared with the
offline pass behind every pinned number in `RESULTS.md`.

The two stages that genuinely **cannot** be batched, because they must read state
before folding the event in, are the two the architecture was designed around:
**velocity 0.654 ms p99 + graph 0.381 ms p99 = 1.04 ms together.** Those are the
numbers that would survive a rewritten scoring path.

Report both sentences: **the current implementation misses a 50 ms budget at p99
(125 ms full stack, 66 ms inline), and the miss is in the calling convention rather
than in the models.**

### 8.8 Day 4 vs Day 5 — before / after

Same five-seed pool, same 0.1% FPR operating point.

| quantity | Day 4 | Day 5 | delta |
|---|---|---|---|
| features | 204 | **232** (+28 `gph_`) | +28 |
| L1 AUC-PR | 0.4910 | **0.5903** | +0.0993 |
| L1 recall@0.1%FPR | 0.3615 | **0.450** | **+0.089** |
| L1 recall, agentic rail | 0.503 | 0.606 | +0.103 |
| L1 recall, classic rail | 0.242 | 0.318 | +0.076 |
| fused recall@0.1%FPR | 0.286 (**worse than L1**) | **0.483** (beats L1) | +0.197 |
| fusion method | unweighted noisy-OR | logistic stacker on percentile + standardised raw | — |
| mean per-family recall, family in training | 0.308 | 0.395 | +0.087 |
| mean per-family recall, family held out | 0.105 | 0.118 | +0.013 |
| L2 role | "the zero-day layer" | residual monitor / drift canary | reframed |
| L3 | none | 1.000 on F1-01 and F1-03, holding on unseen phrasing and unseen kind | new |
| campaign-level recall | not reported | L1 0.916, fused 0.908 | new axis |
| recall as a curve | single point | 0.1 / 0.5 / 1.0% FPR | new axis |
| tests | 225 | 264 → **302 today** | — |

---

## 9. System and deployment

### 9.1 API

FastAPI, `mantis/api/app.py`, mounted at `/api` by `mantis/api/site.py`.
Response models in `mantis/api/models.py`. 22 tests in `tests/test_api.py`.

| method | path | response shape |
|---|---|---|
| GET | `/health` | `HealthResponse` |
| GET | `/atlas?family=&implemented_only=` | `AtlasResponse` — `cards[AtlasCardSummary]`, `families[{family,total,implemented}]`, `total`, `implemented`, `discovered[AtlasCardSummary]` |
| GET | `/atlas/{card_id}` | `AtlasCardDetail` — summary + `actor`, `genai_enabler`, `description`, `preconditions[]`, `observable_signals[{signal,feature,layer}]`, `mitigations[]`, `generator`, `references[]` |
| GET | `/results` | `ResultsResponse` — `generated`, `tables[{title,header,rows}]`, `layer_performance[]`, `per_family[]`, `per_attack[]`, `decisions[]`, `zero_day[{detector,recall}]`, `evasion_curve[float]`, `headline{}` |
| GET | `/arena` | `ArenaResponse` — `operating_fpr`, `cards[]`, `seconds`, `n_background`, `evasion_curve[]`, `generations[]`, `survivors[]`, `zero_day{}` |
| GET | `/fidelity` | `FidelityResponse` — `available`, `note`, `generated`, `schema_version`, `calibration`, `reference`, `synthetic`, `headline`, `marginals`, `tstr`, `discriminator`, `discriminator_ablated`, `adjudications[]`, `known_divergences[]`, `population` |
| GET | `/latency` | `LatencyResponse` — `available`, `generated`, `n_events`, `warm_events`, `budget_ms`, `within_budget`, `headroom`, `end_to_end_ms{}`, `stages_ms{}` |
| GET | `/figure/{name}` | `FileResponse` (PNG) |
| POST | `/simulate` | `SimulateRequest{n_events≤600, rate≤60, family, offset}` → `SimulateResponse{run_id, n_events, rate, stream_url}` |
| GET | `/stream/{run_id}` | SSE `EventSourceResponse`; frames are `StreamFrame{seq, event: ScoredEvent, truth: Truth}` |

`ScoredEvent` carries the authorisation fields, `layers{name: {score, percentile}}`,
`l0{fired, reason}`, `risk` (0–100 dial), `decision`, and
`contributions[{feature, value, contribution}]`. **`Truth{is_fraud, attack_id,
attack_campaign}` is sent after the decision, never with it** — `ScoredEvent` has no
ground-truth field.

### 9.2 UI screens

**There are four screens, not six** — `web/src/App.tsx` defines
`type Tab = "console" | "results" | "atlas" | "fidelity"`.

| screen | shows |
|---|---|
| **Console** | Live authorisation stream: per-event risk dial (0–100), which layers fired, the L0 verdict, the decision badge, and "Why — top 3 contributions". Alert detail panel with the authorisation fields. A `live` / `offline replay` indicator |
| **Results** | The zero-day recovery trio; the evasion curve; recall as a curve not a point; per-family recall and what holding it out costs; per attack card |
| **Atlas** | "Six families, 42 cards"; the card grid with status and rails; "Found by the adversarial loop" (the six discovered cards) |
| **Fidelity** | What this is measured against; every feature against its own sampling noise; TSTR; the discriminator; divergences we name ourselves; **scoring latency** (both the aggregate and the one-event-at-a-time stage table) |

The latency content lives inside the Fidelity screen rather than on a fifth tab.

### 9.3 Deployment

**What was attempted, in order:**
1. `render.yaml` and `railway.json` exist in the repo — **rejected** because a free
   dyno on either **sleeps after 15 minutes**, and a judge clicking a cold URL sees
   a spinner.
2. **Hugging Face Docker Space.** `mantis/api/site.py` mounts the API unchanged at
   `/api` and serves the built Vite bundle at `/`; `api.ts` already defaults
   `API_BASE` to the relative `/api`, so a same-origin build needs no
   `VITE_API_BASE` and never exercises a CORS header. The `Dockerfile` is two
   stages (node builds `web/dist`, python serves it) and installs only the `serve`
   extra — **verified** that importing `mantis.api.site` loads none of pandas,
   numpy, sklearn, scipy, lightgbm, shap, matplotlib, networkx or pyarrow, taking
   the image from ~1.5 GB to ~200 MB. **Blocked: Hugging Face now bills Docker
   Spaces, and a free account gets `402 Payment Required` on `create_repo`.**
   `scripts/deploy_hf.py` catches the 402 and tells you to re-run with `--static`.
3. **Static Space — what actually shipped.** `--static` uploads `web/dist`:
   **11 files, 1.5 MB**, verified today by `python scripts/deploy_hf.py --check
   --static`. Live at **https://aryaambekar-mantis.static.hf.space**.

**Why static is not a degraded build.** The bundle is self-contained by
construction: Day 6 froze every API response by **calling the real route handlers**,
and the console already replays the committed feed on a timer when nothing answers
`/api/health`. A CDN has no cold start at all, which is a stronger version of the
property the Spaces decision was buying.

**The hostname trap:** a static Space is served from
`<user>-<space>.static.hf.space`. The plain `<user>-<space>.hf.space` form **404s
while the Space itself reports RUNNING**.

**How the frontend gets data.** `web/public/data/` holds the frozen responses,
copied into `web/dist/data/` at build:

| file | size |
|---|---|
| `feed.json` | 726 KB — the pre-scored authorisation stream |
| `atlas_cards.json` | 107 KB — all 42 card details in one map |
| `results.json` | 14.9 KB |
| `fidelity.json` | 14.2 KB |
| `atlas.json` | 10.2 KB |
| `arena.json` | 4.9 KB |
| `latency.json` | 1.9 KB |
| `health.json` | 172 B |

`api.ts` exports `type Mode = "live" | "static" | "unknown"` and probes
`${API_BASE}/health` on load: `live` means an API answered and everything is a real
request; `static` means it reads the frozen artefacts.

**The SSE-vs-timer-replay caveat, and how the UI discloses it.** In `live` mode the
stream is a real server-sent-event coroutine. In `static` mode `replayFrozen()`
drives the same frames off a `setTimeout` chain — **the one thing genuinely given
up is a real SSE stream.** The Console does not let a viewer assume: it renders a
badge reading `API connected` in live mode and **`offline replay`** in static mode,
with the title attribute *"No API answered; replaying the committed feed on a
timer."* A second line states *"Curated replay of pre-scored authorisations. Fraud
is over-sampled so something happens while you watch."*

### 9.4 Reproduction — clean clone to running system

Requires: Python ≥3.11, Node for the web build. **No Kaggle token, no GPU, no API
key, no network.**

| step | command | runtime |
|---|---|---|
| install | `make install` | ~1–2 min |
| **acceptance test** | `make demo` → `schema atlas corpus population dataset features l0 test` | ~5–8 min |
| — schema contract | `python -m mantis.core.events` | <1 s |
| — atlas + registry | `python -m mantis.atlas.loader` | <1 s |
| — LLM corpus, cache-only | `python -m mantis.foundry.llm` | <1 s |
| — legitimate population | `python -m mantis.foundry.base --n 200000 --seed 7` | **8.5 s** |
| — attacks + probe | `python -m mantis.foundry --attacks all --out data/generated/dataset_v1.parquet --show-content` | **~44 s** (probe is 34 s of it) |
| — features | `python -m mantis.defense.features` | **~16 s** |
| — L0 clauses | `python -m mantis.defense.l0_rules` | ~10 s |
| — tests | `python -m pytest` (302) | ~4 min |
| full firewall tables | `make firewall` / `python -m mantis.defense` | **~15 min** |
| evasion curve + zero-day | `make loop` / `python -m mantis.loop` | **581 s** |
| fidelity scorecard | `make reference` (210 MB download, once) then `make fidelity` | download + ~3 min |
| latency bench | `python scripts/latency_bench.py` | ~1 min |
| L3 OOD probe | `make ood` | <1 min |
| console feed | `make feed` | ~2 min |
| web build + API | `make web` then `make api` → `http://127.0.0.1:8000` | ~30 s |
| submission bundle | `make submission` (= ood feed docx web) | — |
| deploy | `python scripts/deploy_hf.py --space <you>/mantis --static` | seconds |

`make reference` is the **only** step that touches the network, it is optional, and
the fetched CSVs are gitignored. Everything else runs from the committed cache.

---

## 10. Limitations, deviations and open items

Consolidated. Nothing here is tidied.

### 10.1 Coverage and scope

1. **15 of 42 cards have injectors.** 27 are taxonomy only: they carry observable
   signals and mitigations, but the foundry does not generate them. Per family:
   F1 7/12, F2 2/6, F3 1/8, F4 2/6, F5 **0/5**, F6 3/5.
2. **F5 is empty on purpose** and is absent from every results table. It is the
   zero-day holdout family. The leave-one-family-out columns and the loop
   experiment are the closest available stand-ins — there is no measured F5 number.
3. **L3 covers 2 of 15 cards** (F1-01, F1-03). Its overall recall of 0.138 should be
   read as coverage of the agentic-injection rail, not as a headline.
4. **Day 2 cards under-declare their rails.** Several Day 2 injectors ride rails
   their card does not list (F2-13 reaches `card_present`, `recurring`, `upi_p2p`).
   The rails subset assertion is enforced **for the Day 3 F1 cards only**. Not
   reconciled.
5. **The six-card arena default.** The evasion curve is measured over 6 of the 15
   implemented cards, 5 generations, population 30, 2 pooled seeds. All fifteen at
   population 8 is ~2 hours. `--cards all` exists; it was not run for the reported
   curve.

### 10.2 Detection claims that are narrower than they look

6. **L2 does not work and is no longer presented as a detector.** AUC-PR 0.0151,
   ROC 0.5785, recall 0.002 at 0.1% FPR. **L2e is below chance at ROC 0.4421.** The
   published explanation is that attacks built to be distributionally faithful are
   by construction invisible to distributional anomaly detection — our own fidelity
   work caused it. L2's surviving claim is narrow: its recall is unaffected by
   whether an attack was in training.
7. **L3's threshold does not transfer.** On hand-authored out-of-distribution text
   it fires on 100% of injections **and 90% of the clean controls**. Only the
   *ordering* transfers (ROC 0.9986 → 0.8111). L3 as calibrated here **cannot be
   pointed at the open web**. The 1.000 recalls on F1-01/F1-03 are true statements
   about this corpus and are not claims about text in general. Named fix — fit the
   page threshold on benign traffic-drawn text — **not done**.
8. **The zero-day recovery is 66%, not detection of the unknown.** The detector had
   no real F1 events; the *loop* had F1's cards and injectors. The correct claim is
   "an attack family described but never observed can be manufactured, and training
   on the manufactured version transfers to the real one" — never "the detector
   caught something nobody had thought of".
9. **The evasion curve rebounds**, 0.626 → 0.334 → 0.338 → 0.363 → 0.381. Almost all
   of the fall is at the first retrain. The supported claim is "cuts evasion by 39%
   and holds it down", not "drives evasion to zero".
10. **Three of nine arena survivors were the unmutated attack.** F2-16 evades 77.8%
    with every gene at default. That is a result about the parent card, not a
    discovery, and it points at where the detection work is: F1-01, F1-05, F2-16.
11. **F4-27 recall is 0.036 event-level.** Card testing is caught at campaign level
    (0.875) and not at event level. Same shape for F6-39 (0.135 / 0.700) and F2-16
    (0.121 / 0.429).
12. **Fraud is 5.7× concentrated on the agentic rail by design.** Any model leans on
    the presence of an agentic block, which is why per-rail recall must be quoted.
13. **F1-03's in-slice AUC of 0.833 rests on n = 620 rows** — a thin slice, flagged
    `!` by the probe. Always quote it with its n.
14. **`mnd_deliberation_residual_z` separates F1-01 at 0.99** — above the foundry's
    own gate, which never saw it because the gate probes raw columns. Ablated and
    reported: removing it costs F1 only 0.681 → 0.664. Kept, not hidden.
15. **F6-40's `txn_round_score` at 0.966 is adjudicated an artefact** — the injector
    snaps to round numbers harder than a real ring would. Fix recorded, not applied.
16. **F6-39's generator is narrow** — `mcc=7832` alone scores 0.904 off six declared
    MCCs. Widening `_DECLARED_MCCS` is recorded, not done.

### 10.3 Fidelity

17. **The discriminator is 0.9994 against a target of 0.5**, and 0.8399 with the two
    adjudicated axes removed. Both numbers are always reported because the ablation
    is a judgement a reader may reject. The separation is **not one bad column** —
    the drop-one path degrades gradually and it takes dropping 6 of 8 features to
    reach 0.5231. This is the foundry's most substantial outstanding item.
18. **The intersection discipline bought almost nothing** (0.99986 raw → 0.99941
    shape space). The separation is real and in behaviour, not formatting.
19. **TSTR is 0.030.** That measures whether the two panels' fraud is the same
    phenomenon — it is not, by construction — and is *not* evidence the background is
    unrealistic. Measured on **classic-rail fraud only**; no number in that section
    is evidence about the agentic attacks.
20. **The reference panel is itself synthetic** (Sparkov), is a US/USD panel, and
    **has no agentic transactions, because no panel does.** That absence is the
    project's premise and also the limit of what the fidelity section can claim.
21. **`decline_reason` remapping runs at 302× its noise band** and **realised decline
    rates sit above prior on every rail (moto ×2.32)** because `decline_amount_tilt`
    is not mean-preserving — Jensen, not redistribution. Both are conservative for
    detection and both are **deferred**, because fixing them re-rolls every pinned
    calibration number.
22. **The population was never fitted to the Sparkov CSV.** `data/reference/` now
    holds it (gitignored), but `ReferenceStats.source` is `indian-market-priors` and
    `scripts/fit_reference.py` was not run.

### 10.4 Engineering

23. **Latency misses the 50 ms budget**: p99 **125.0 ms** full stack and **66.3 ms**
    on the inline L0+L1 path. The miss is in the calling convention — `entity` costs
    45.9 ms on a one-row frame and 0.1128 ms/row in batch, a 407× per-call overhead
    from `Series.map(dict)`. **The fix is named and not applied**, because the
    feature builder is shared with the offline pass behind every pinned number.
24. **The deployed prototype is a static bundle replaying frozen API responses on a
    timer, not a live SSE stream.** Docker Spaces returned `402 Payment Required` on
    a free account. The UI discloses the mode with an `offline replay` badge.
25. **The live prototype URL is `.static.hf.space`.** The plain `.hf.space` form
    404s while the Space reports RUNNING.
26. **`docs/` is the only place the four-screen UI is described as four.** Any deck
    claiming six screens is wrong.
27. **No KYA reputation score exists.** `kya_registered` is a boolean; there is no
    reputation model anywhere in the codebase.
28. **`CLAUDE.md` §8 is stale in two places**: it says 287 tests (302 today) and
    quotes an end-to-end latency of p50 117 / p95 153 / p99 171 ms (superseded by
    115.7 / 121.1 / 125.0).
29. **There is no "Day 6" section in `CLAUDE.md`.** The Day 6 work (API, console,
    results screen, submission document, static build) is recorded only in the git
    log and in `DEPLOY.md`.
30. **The committed corpus was authored with `mistral:7b-instruct-q4_K_M`**, not the
    `qwen2.5:7b` the plan named, because that is the model that was pulled locally.
    `DEFAULT_MODEL` matches deliberately so a judge gets cache hits. Switching is one
    line plus `--live --refresh` to re-author.
31. **`data/cache/content/bindings.jsonl` at HEAD holds 467 bindings, not the 2,343
    `CLAUDE.md` claims.** Verified today: of 18,404 distinct content ids on fraud
    rows in `pool_5seed.parquet`, only **467 resolve through explicit bindings**.
    Per card, the share of events carrying an explicitly-bound injected payload is
    **F1-01 21.2% (159/750)** and **F1-03 21.7% (130/600)**; every other F1 card is
    0%. That is the one-seed-in-five shape of the Day 5 bindings bug (§11.1). The
    cause is ordering, not a code regression: `experiment_result.pkl` was written
    2026-08-25 07:57 and `bindings.jsonl` was overwritten at 08:01 by the
    single-seed `python -m mantis.foundry` run, so the L3 numbers in `RESULTS.md`
    were produced against the pool's full binding set and **a re-run from the
    committed state today would regress L3 to roughly a fifth of its reported
    recall.** The `ContentStore`'s universal-resolution fallback means this fails
    silently. **Fix: re-run `mantis.defense.pool.build_pool` (which now persists the
    store) before re-running `python -m mantis.defense`, and re-commit
    `bindings.jsonl`.** Not done.
32. **`RESULTS.md`'s dateline still reads "Day 5"** although it now carries the Day 7
    fidelity, latency and deployment sections.

---

## 11. Defect log

One line each. "Caught by" is the column worth carrying into the writeup.

| # | defect | would have manifested as | caught by |
|---|---|---|---|
| 1 | `build_pool` never persisted `bindings.jsonl`; only seed 1337's 513 bindings existed | L3 reading innocuous text on 80% of attack rows — recall four fifths too low, silently, because the benign-pool fallback resolves every id | Counting bindings against expected plantings. Still live in the committed artefact — §10.4 #31 |
| 2 | `hash()` on a string in a `set` iterated inside an RNG loop (population simulator) | `--seed 7` giving a different population every process; slide numbers not reproducible | `scripts/audit_population.py` re-running one seed in two processes |
| 3 | `AttackGenome.label()` used `hash()` on a string | Every arena run and written-back card id differing between runs | Same class as #2; now `stable_seed`. `tests/test_loop.py` pins a **literal** label — a same-process comparison passes on the broken code |
| 4 | Provenance rebinding in `mutate.py` used `hash()` on a string | Same as #3, on variant provenance chains | Same fix, same test discipline |
| 5 | `device_id` was a perfect rail separator (AUC 1.000; 0 of 8,796 devices carried both rails) | Any model reading "is this agentic" off the device | Day 1 audit rail-separability tiers. Agents now run on-device ~50% of the time |
| 6 | Binary adopter flag gave 70% of customers a hard-zero agentic probability | `customer_id` AUC 0.90 | Same audit → graded Beta propensity, 0.75 |
| 7 | Agentic 3DS mix had no failure tail | `threeds_result` AUC 0.86 | Same audit → 0.75 |
| 8 | Content planting **extended** the provenance chain | `ag_provenance_chain_len` a 0.96 detector — catchable by counting URLs without reading one | Day 3 separability probe. Planting is now length-preserving |
| 9 | `collapse_deliberation` **added** to the cloned tool-call count, exceeding the legitimate max | `ag_tool_call_count` 0.96 | Same probe. Now resamples from the background's own upper band |
| 10 | Injectors scheduled uniformly across 24 h against a diurnal population | `ts_hour` the strongest single feature on three attacks | Same probe. `set_timestamps` redraws from the background hour curve, shifting bursts via `groups` |
| 11 | Campaign starts drawn uniformly, leaving calendar gaps | A `ts_epoch` signal | Same probe. `spread_epochs` is now stratified |
| 12 | F4-27's oracle was decorative: campaign declines drawn *after* escalation targets were chosen | Escalation through merchants merely *touched* — 65% landed on an approving merchant, about chance | Day 4 review. Reordered so the probe outcome chooses targets: now 100%, probe declines 51–66%, escalation 8% vs an 8.8% background |
| 13 | Probe report was direction-blind — it takes `max(a, 1−a)` | `auth_response=approved 0.69` reading as "approved more often" when the truth is the reverse; a model trained off it learns card testing sign-flipped | Day 3 review. Every probe row now carries `direction` (`hi`/`lo`) |
| 14 | Reference panel taken as a uniform 200k row sample of 1.85M | Every trailing count and inter-arrival gap deflated ~10× | The fidelity discriminator. Panel now cut to a contiguous 90-day window matching the synthetic span |
| 15 | **Percentile saturation in fusion** — the stacker consumed layer percentiles only | Every score above the largest legitimate score maps to 1.0, which is exactly the region ranked at a 0.1% budget; L1's ordering inside its own top 0.1% discarded. Fused 0.104 vs L1 0.553 | Comparing fused against L1 alone. Fix: percentile **plus** standardised raw score per layer. `tests/test_fusion_policy.py` pins it with a layer whose signal lives above every legitimate score |
| 16 | **L2's NaN sentinel** — missing values filled with `-1e9` | iForest split thresholds landing in the empty gap, so trees isolated on the **missingness pattern** (i.e. the rail), not behaviour; 109 of 204 features are >30% missing on legitimate traffic | Day 4 inspection of what L2 split on. Dense-column filter (`MIN_PRESENT_SHARE = 0.70`) + median impute: recall 0.0006 → 0.0054. Cut chosen on missingness alone, never tuned against recall |
| 17 | `_as_bool` written as `v is True` | Silent failure on `numpy.bool_` (`np.True_ is True` is `False`) — `kya_unregistered` firing on 100% of CLEAN attacks in memory, 0% after a parquet round-trip | Running L0 tests in memory. A test reading only the committed parquet would never have found it |
| 18 | z-scored velocity counts as a fidelity shape feature | Discriminator AUC 1.000. The counts are 87.6%/83.4% zero, and standardising maps that **shared** modal atom to −0.370 vs −0.411 — one split separates the panels while the distributions are close | The discriminator caught a bug in its own measurement. Replaced by `gap_ratio_log` + `burst_1h`; `customer_merchant_share` removed rather than fixed. `tests/test_fidelity.py` builds two samples from one distribution and asserts the old transform would have separated them |
| 19 | **`_reading()` heuristic ordering** in `scripts/l3_ood.py` — recall tested before FP | "1.000 out of distribution, the layer holds", while the layer called 9 of 10 clean pages injected | FP-on-controls is now tested **first**, and disqualifies the recall row as a headline when high. Documented in the function docstring |
| 20 | **Risk percentile display bug** — the console dial showed the raw fused percentile | Every decision lives in the top 1% of legitimate (challenge 99th, review 99.5th, decline 99.9th), so an ordinary **approved** authorisation read 0.95. Same saturation as #15 | Reading the console as a judge would. `_risk_index` now interpolates through the policy boundaries (median→0, challenge→50, review→75, decline→90, max→100); monotonic presentation transform, no metric computed on it |
| 21 | Explain layer coerced every value to float before printing | `channel = nan` on the most readable line of an alert; 7 of the matrix columns are categorical | Reading a rendered alert. Values print as themselves; a real NaN prints `absent` |
| 22 | L3 v1 concatenated each event's chain and vectorised 700,000 strings | 16 minutes, and a **lower** score — summing a chain dilutes one injected page among eleven innocuous ones | Wall clock. Classifying the artefact and taking a **max over the chain** takes ~1 s and scores higher |
| 23 | Three measurement bugs in `scripts/drift_check.py` itself: no entity-level null band; unweighted day-of-week target; thresholded rather than deconvolved passive-human share | Three false drift findings (one at **339×**) that would have triggered re-tuning of a correct population | Each implausible on its face and re-derived before any conclusion was drawn |
| 24 | Circular import: `features/builder` → `l4_graph.graph` → `features.state` for `as_epoch` | Import cycle at module scope | `as_epoch` import deferred into the function body; direction stays one-way at module scope |
| 25 | Latency bench run while the fidelity scorecard was fitting on the same machine | Every percentile roughly **doubled** | Re-running on a quiet machine. Latency is the one measurement here not reproducible from a seed |
| 26 | Inline/fast-follow latency split computed by subtraction | L2 — 40% of the clock — vanishing into fast-follow without being named there | The split is now **asserted to partition** the stage list |
| 27 | `THIN_SLICE_ROWS` fitted to the one case it was meant to judge (750) | A thin-slice warning that never fires | `scripts/audit_probe_slices.py`. Raised to 2,000; F1-03 at 620 rows now flagged |
| 28 | Day 3 bucket contract had no false-positive term — any signal reaching 25% recall passed | It was satisfied by `delegation_depth > 2`, which costs **4.99% FP on legitimate agentic traffic** (1,498 / 30,040) | Measuring the contract against a real L0. **The contract was judged wrong, not L0**; neither side was adjusted to agree |
| 29 | Manufactured variants could land in the test period | Variants inflating the velocity and graph state of the real test rows they are evaluated against — measuring the injection, not the transfer | Reasoning about the zero-day claim before running it. Variants are now confined to the training window |
