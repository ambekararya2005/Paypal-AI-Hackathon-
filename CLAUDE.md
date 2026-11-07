# MANTIS — Project Constitution
## 1. The pitch (compressed)

Agentic commerce — AI agents that pay on a human's behalf via **Mastercard Agent
Pay / AP2 mandates** — has created a live payment rail with **zero labelled
fraud data**. You cannot train a fraud detector on data that does not exist. So
you *manufacture* it, adversarially.

MANTIS is a red-team / blue-team lab in three pillars:

1. **IDENTIFY** — an *executable* atlas of 42 GenAI payment-fraud vectors as
   YAML cards that the generator **imports**. Not documentation: a dependency.
2. **GENERATE** — a foundry that synthesises a calibrated legitimate payment
   population and injects those attacks into it, with a **measured** fidelity
   scorecard (not a vibe check).
3. **DEFEND** — a five-layer **Mandate Firewall** that scores every
   authorisation and reports **recall at a fixed 0.1% false-positive rate**.

Then the closed loop that makes it a *lab* rather than a demo: an evolutionary
adversary reads the detector's own SHAP output, mutates to evade it, the
detector retrains, and the **evasion rate falls**. That curve is the money shot.

---

## 2. Judging criteria and the artifact that scores each one

Every hour of work should be traceable to a row in this table. If it is not, it
is a cut candidate.

| # | Criterion | Artifact that scores it | What we must be able to show |
|---|-----------|-------------------------|------------------------------|
| 1 | **Diversity of attacks** | `mantis/atlas/` — 42 cards across families F1–F6, each with `observable_signals` wired to real features | Atlas summary table; count of *implemented* (not merely mapped) injectors; the zero-day holdout family |
| 2 | **Fidelity of simulation** | `mantis/foundry/fidelity/` — the fidelity scorecard | Marginal KS distances, MCC / amount / hour-of-day mixes vs. reference, a TSTR (train-synthetic-test-real) number. Measured, printed, and honestly imperfect. |
| 3 | **Detection efficacy** | `mantis/defense/` — the five-layer Mandate Firewall | AUC-PR and **recall@0.1%FPR** per layer and fused, as a curve over 0.1/0.5/1.0% FPR; per-family recall, **event-level and campaign-level side by side**; the zero-day holdout result |
| 4 | **Novelty** | `mantis/loop/` — evolutionary adversary + retrain harness | Evasion-rate-vs-generation curve; SHAP-guided mutation; cards found by the discovery agent |
| 5 | **Real-world feasibility** | `mantis/core/events.py` — the schema | `TxEvent` is a **superset of a real card authorisation** plus an agentic extension. An issuer could drop this in behind an existing auth stream without re-platforming. Plus: L0 rules are deployable today, and the latency budget is reported. |

**Criterion 5 is won or lost in the schema.** That is why the schema is frozen
(§4) and why `events.py` carries a docstring that argues the case explicitly.

---

## 3. Repo layout

```
mantis/
  mantis/core/          events.py — THE schema, imported by everything
  mantis/atlas/         schema.py, loader.py, cards/*.yaml, discovery/
  mantis/foundry/       base/  injectors/  llm/  fidelity/
                        llm/ = cache.py client.py fallback.py prompts.py
                               corpus.py build.py  (3-stage, cache committed)
  mantis/defense/       features/ l0_rules/ l1_gbdt/ l2_novelty/
                        l3_text/ l4_graph/ fusion/ policy/ explain/
  mantis/loop/          evolutionary adversary + retrain harness
  mantis/api/           FastAPI + SSE (live defense console backend)
  web/                  React + Vite
  data/reference/       calibration tables (committed, small)
  data/generated/       synthetic output (gitignored except .gitkeep)
  data/cache/           LLM output cache — COMMITTED, see HARD RULES
  docs/  tests/  scripts/
```

Dependency direction is one-way and must stay that way:

```
core  <-  atlas  <-  foundry  <-  defense  <-  loop  <-  api  <-  web
```

`core` imports nothing of ours. `atlas` imports `core` (card rails are
validated against the `Channel` enum). Nothing imports `api`. No cycles, ever.

---

## 4. The event-schema contract — FROZEN (v1.1.0)

`mantis/core/events.py` is frozen as of Day 0. Every other module codes against
it. **Do not add, rename, retype, or reorder a field** without doing all four
of: updating `events.py`, updating `tests/test_schema.py`, updating this
section, and grepping every `ag_` string literal in the repo.

`SCHEMA_VERSION` records where we are: `1.0.0` is the Day 0 freeze, `1.1.0` is
the Day 3 transaction-lifecycle amendment below. It goes into every dataset
manifest, so a parquet can never be mistaken for one it is not.

Additive-only escape hatch: if a new attack genuinely needs a field, add it to
`AgenticContext` with a default of `None` so every existing call site keeps
working. Never repurpose an existing field's meaning.

**Classic block** (a real card authorisation): `event_id`, `ts`, `amount`,
`currency`, `mcc` (4 digits, validated), `channel`, `entry_mode`,
`customer_id`, `card_bin`, `merchant_id`, `merchant_country`, `terminal_id`,
`device_id`, `ip`, `lat`, `lon`, `threeds_result`.

**Agentic block** (`AgenticContext`, `None` on classic rails): `agent_id`,
`agent_platform`, `kya_token`, `kya_registered`, `mandate_type`, `mandate_id`,
`mandate_hash`, `mandate_issued_ts`, `mandate_ttl_seconds`, `mandate_scope`
(nested `MandateScope`), `human_present`, `consent_sig_valid`,
`delegation_depth`, `provenance_chain`, `ingested_content_ids`,
`tool_call_count`, `deliberation_latency_ms`, `cursor_entropy`,
`dwell_time_ms`.

**Transaction lifecycle** (Amendment 1.1.0, Day 3 — additive, all defaulted):
`txn_type` (purchase/refund/reversal/preauth/credit, default purchase),
`auth_response` (approved + five decline reasons, default approved),
`original_event_id`, `dispute_outcome`, `dispute_raised_ts`, `settled` (default
True), `settlement_lag_hours`.

Why this **strengthens** criterion 5 rather than diluting it: every one of those
is a field an issuer already holds. `txn_type` is ISO 8583 DE 3, `auth_response`
is DE 39, `original_event_id` is the original-transaction reference an acquirer
echoes on a credit, the dispute pair is ordinary chargeback case state, and the
settlement pair is the gap between the authorisation and the clearing file.
Day 0 modelling only the authorisation request was a **specification error**: it
made F1-03 unrepresentable and left F4-27/F4-28 half-modelled, because the
approve/decline oracle those attacks farm had nowhere to live.

Amounts stay **non-negative on every type**. Direction is carried by `txn_type`,
not by the sign, because that is how the wire does it and because a signed
amount would silently break every quantile, KS distance and log-amount feature
calibrated on Day 1.

**Ground truth** (never a feature — see HARD RULES): `is_fraud`, `attack_id`,
`attack_campaign`.

**Post-hoc** (`POST_HOC_COLUMNS`, never a *scoring* feature): `dispute_outcome`,
`dispute_raised_ts`. They resolve days to months after the authorisation, so
using them at scoring time is temporal leakage — the model would be reading the
future. They are carried for evaluation, cost modelling and the console, and the
feature builder must drop them alongside `LABEL_COLUMNS`.

Exports: `SCHEMA_VERSION`, `LABEL_COLUMNS`, `POST_HOC_COLUMNS`,
`DECLINE_RESPONSES`, `flatten(ev) -> dict` (agentic fields prefixed `ag_`).

**`provenance_chain` is the highest-value field in the schema.** It is the
ordered list of URLs / content the agent read before it decided to transact. It
is what turns indirect prompt injection from something you can *describe* into
something you can *detect*. It is the input to the L3 text layer. Treat it as
load-bearing.

---

## 5. Coding conventions

- **Python 3.11+**, **pydantic v2** (`field_validator` / `model_validator`, not
  v1 `@validator`). Type hints on every signature; no bare `Any` without a
  one-line comment justifying it.
- **ruff-clean** before any commit: `make lint`. Line length 100.
- **No notebook-only code.** Nothing lives in a `.ipynb`. If it is worth running
  twice, it is a module.
- **Every module runnable via `python -m`**, with a `main()` and an
  `if __name__ == "__main__":` block that prints something a human can read.
  This is how the demo is assembled and how you debug at 3am.
- **Determinism**: every stochastic entry point takes `seed: int`, default
  `1337`. A judge re-running the pipeline must get the numbers on the slides.
- Paths via `pathlib`, never string concatenation; repo-relative paths resolved
  from a single helper module, never `os.getcwd()`.
- Frontend: React + Vite + TypeScript. Nothing heavier than the chart library.
  The console is a dashboard, not an app.

---

## 6. HARD RULES

Violating any of these is a build failure, not a style nit.

1. **NEVER let label columns reach a model's feature matrix.**
   `is_fraud`, `attack_id`, `attack_campaign` — and anything derived from them —
   must be dropped *by name* in the feature builder, with an assertion that
   fires if they survive. Leakage is the #1 way hackathon fraud models silently
   cheat, and a judge who sees 0.999 AUC will assume exactly that. Any metric
   that looks too good is leakage until proven otherwise.

2. **NEVER report accuracy.** The class balance makes it meaningless, and
   quoting it signals you do not know the domain. Report **AUC-PR** and
   **recall@0.1%FPR**, always with the operating point stated. Confusion
   matrices only at a declared threshold.

3. **ALWAYS cache LLM output to disk and commit the cache.** A judge cloning
   this repo will not have Ollama running. Every LLM call goes through the cache
   layer in `mantis/foundry/llm/`, keyed by a hash of prompt + model + params.
   A cache miss with no backend available falls back to the committed cache or a
   deterministic stub — never a crash, never a hang, never a network wait.

4. **The repo must run from a clean clone with no Kaggle token, no GPU, and no
   API key.** `make demo` is the acceptance test. Run it against a scratch clone
   before you sleep on the last night. No download-at-runtime, no credentials,
   no environment variables that only exist on the build machine.

5. **Generated attack content is demonstrative, never operational tooling.**
   The atlas describes attack *classes* with observable signals and mitigations
   so that they can be detected. Injection payloads in the foundry are short,
   obviously synthetic, clearly labelled, and exist to be caught. We do not
   write working exploits against real platforms, real merchants, or named
   third-party products. If a card cannot be written defensively, it stays
   `status: mapped` and never gets an injector.

---

## 7. The cut line

When time runs out — and it will — sacrifice in **this order**, top first:

1. **L4 graph layer** (fall back to four layers, and say so out loud)
2. **Adversary loop generations** (3 instead of 10; the curve still has a slope)
3. **Live LLM** (cache-only mode — already the default path)
4. **Discovery agent** (ship the cards it already found, drop the live run)
5. **React frontend** (fall back to Streamlit; the numbers matter, not the CSS)

**Never cut, at any cost:**

- The **live defense console** — a judge must watch a transaction get scored
- **recall@0.1%FPR** — the one number that proves the detector is real
- The **zero-day holdout** — an attack family the detector never trained on
- The **fidelity scorecard** — proves the synthetic data is not a toy
- The **.docx** submission document — no document, no score, regardless of code

---

## 8. Current state

- **Day 0 complete**: constitution, scaffold, frozen `TxEvent` schema, atlas
  schema + loader, first 8 attack cards, schema round-trip tests.
- **Day 1 complete**:
  - **Atlas finished at 42 cards** (F1 12, F2 6, F3 8, F4 6, F5 5, F6 5).
    `python -m mantis.atlas.loader` prints the implemented/mapped split under an
    "HONEST COUNT" heading; `tests/test_atlas.py` locks the numbers so the
    writeup cannot drift from the repo. (Day 1 claimed 15 implemented on the
    strength of planned generator paths; Day 2 made that claim enforceable and
    the count is now **8**. See §8 Day 2.)
  - **Legitimate population simulator** in `mantis/foundry/base/`:
    `reference.py` (calibration + Indian-market priors), `entities.py` (standing
    customer/card/device/merchant/agent map), `simulator.py` (the draw),
    `calibration.py` (metrics + figure), `__main__.py` (CLI).
  - `scripts/fit_reference.py` fits shape parameters from the Kaggle Sparkov CSV
    if one is dropped into `data/reference/`, and exits cleanly when none is.
  - `mantis/core/paths.py` is the single repo-relative path helper (see §5).
- **Day 1 gate (passing)**: `python -m mantis.foundry.base --n 200000 --seed 7`
  writes `data/generated/population.parquet` (~15 MB, ~8 s) plus a manifest and
  `docs/population_calibration.png`; amount KS 0.0051, hour TV 0.0066, MCC mix
  max delta 0.0010; 42 cards load clean; 65 tests pass; ruff clean.
- **Day 2 complete**:
  - **Injector framework** in `mantis/foundry/injectors/base.py`: `BaseAttack`
    (ABC, `inject(population, intensity, rng) -> DataFrame`), a `PopulationView`
    of read-only indices over the background, and the `REGISTRY`.
    **`validate_registry()` runs at package import and fails the import** unless
    the atlas and the code agree in *both* directions - every `implemented` card
    has an injector, every injector names a real `implemented` card, and each
    card's `generator` path resolves to a callable in that injector's own
    module. This is the seam that makes Pillar 1 executable rather than
    decorative, and `tests/test_injectors.py` fires it deliberately to prove it
    is load-bearing.
  - **Eight tabular injectors**, one per card: F4-27 adaptive BIN, F4-28
    threshold probing / just-under structuring, F2-13 synthetic identity
    onboarding, F2-16 synthetic-identity bust-out, F3-19 digital-arrest APP
    scam, F6-38 mule-network fan-in/fan-out, F6-39 transaction laundering via
    miscoded MCC, F6-40 stored-value cash-out ring.
  - **Separability probe** in `injectors/probe.py`: a depth-1 stump on every
    single column (categoricals exploded per level, null patterns and list
    lengths included, raw identifiers excluded with written reasons), reporting
    the stronger of the stump AUC and the column's own rank AUC. Gate is 0.95.
  - `mantis/foundry/__main__.py` is the Day 2 CLI and gate.
- **Day 2 gate (passing)**:
  `python -m mantis.foundry --attacks all --out data/generated/dataset_v1.parquet`
  writes 201,290 rows (~15 MB, ~40 s including the probe) plus a manifest.
  Prevalence **0.6409%** (200,000 legitimate / 1,290 fraud). Best-single-feature
  AUC per attack: F2-13 0.639, F2-16 0.627, F3-19 0.872, F4-27 0.663,
  F4-28 0.801, F6-38 0.787, F6-39 0.792, F6-40 0.676; **all fraud combined
  0.609**. 102 tests pass; ruff clean.
- **Day 3 complete**:
  - **Schema amended to v1.1.0**, additively (see §4). Seven optional
    lifecycle fields; every 1.0.0 call site unchanged; `tests/test_schema.py`
    pins the first seventeen classic columns literally, so "additive" is
    enforced rather than intended.
  - **The population uses all of it**: a channel-dependent decline rate that
    rises with the ticket (8.8% overall; card-present 2.8%, ecom 11.5%),
    legitimate refunds and reversals **bound to real earlier purchases**,
    pre-authorisation holds, orphan-free credits, a 9-basis-point dispute rate
    that is post-hoc by construction, and settlement lag that is genuinely
    **bimodal** — UPI clears in seconds, card rails on tomorrow's file.
  - **`mantis/foundry/llm/`**: three-stage degradation (live Ollama → committed
    disk cache → bundled deterministic corpus), cache keyed on
    `sha256(model|prompt|params)`, standard library only. `generate()` cannot
    fail, cannot hang, and opens no socket unless explicitly told it may.
    234 artefacts (138 benign / 96 adversarial), 231 authored against a live
    7B model, committed under `data/cache/`.
  - **`ContentStore`** joins `ingested_content_ids` to that text. Two-tier:
    explicit bindings for planted payloads, deterministic assignment into the
    benign pool for everything else — so **every** id in the parquet resolves,
    on attack rows and legitimate ones alike.
  - **Seven agentic F1 injectors**, split HARD/CLEAN (see the decisions below):
    F1-01 cart-mandate tampering, F1-02 intent-scope inflation, F1-03
    refund-logic hijack, F1-04 category drift, F1-05 delegation laundering,
    F1-09 human-present spoofing, F1-10 mandate replay.
  - **The atlas ratchet moves 8 → 15**, and F1 is no longer an empty family.
- **Day 3 gate (passing)**:
  `python -m mantis.foundry --attacks all --out data/generated/dataset_v1.parquet --show-content`
  — 15 injectors, prevalence 1.049%, every attack under the 0.95 separability
  gate *inside its declared slice*, 190 tests, ruff clean, 30/30 on the
  population audit. `make corpus` runs with no network.
- **Day 4 complete** — the defence core:
  - **`mantis/defense/features/`**: 204 features in four groups — transaction,
    velocity, entity, mandate. Velocity runs over a **keyed rolling state store**
    (`state.py`), not a groupby: one forward pass, `bisect` + prefix sums per
    window, bounded memory via eviction. **0.052 ms/row**, which is the number
    Day 5's p99 budget is spent against.
  - **A third leakage tier.** HARD RULE 1 names labels; `features/spec.py` adds
    `POST_HOC_COLUMNS` and a new `FUTURE_COLUMNS` — `auth_response`, `settled`,
    `settlement_lag_hours` **of the event being scored**. Those are the issuer's
    own decision on this message and do not exist when the firewall runs.
    Blocked by name, plus a derived-name check, asserted on every `transform`,
    and `tests/test_features.py` fires all three tiers deliberately.
  - **The decline-ratio windows the amendment made buildable**: per card, BIN,
    merchant, device, over 1h/24h/7d. Plus refund-to-purchase ratios per customer
    and merchant, merchant settlement-lag deviation from its rail's mode, and the
    orphan-credit indicator.
  - **`mantis/defense/l0_rules/`**: nine deterministic clauses, each returning a
    named reason, with per-clause precision and FP rate printed. Every clause but
    one fires on **0.000%** of legitimate agentic traffic; `kya_unregistered` sits
    at 2.8% because the population carries that tail on purpose.
  - **L1** (LightGBM, time-based split, isotonic-calibrated) and **L2**
    (isolation forest, legitimate traffic only, asserted).
  - **`mantis/defense/pool.py`**: five independently-generated worlds pooled,
    identifiers namespaced per seed and calendars offset, so per-family positive
    counts land at 550–4,150 instead of ~120.
- **Day 4 gate (passing)**: `python -m mantis.defense` — 1,010,600 events,
  10,600 fraud, time-split 707k/303k. **L1 AUC-PR 0.4910, recall@0.1%FPR
  0.3615**; per rail, agentic 0.503 and classic 0.242. 225 tests, ruff clean.
  Full table in `RESULTS.md`.
- **Day 5 complete** — the layers that read *intent* and *relations*, and the loop:
  - **Framing change first.** L2 demoted to residual monitor and drift canary;
    the zero-day answer is now L0's protocol invariants plus the closed loop.
    See "The zero-day answer, reframed" above — it outranks anything Day 4 said
    about L2 rescuing a held-out family.
  - **`mantis/defense/l4_graph/`**: 28 streamed graph features. Union-find over
    an **identity** graph (customer ↔ device ↔ agent) plus windowed distinct
    counters for the merchant and BIN sides. Read-then-fold per event, exactly
    like the velocity store, so it is backward-looking by construction and needs
    no fit/transform split. **0.021 ms/row.**
  - **`mantis/defense/l3_text/`**: a **page** classifier — TF-IDF + logistic
    regression over the committed content corpus — with an event scored as the
    worst page its agent read. `L3Model.fit` **has no `y` parameter**: its label
    is the artefact's own `injected` flag, a property of text rather than of a
    transaction. Two hold-out protocols, an unseen *phrasing* and an unseen
    *kind*.
  - **`mantis/defense/fusion/`**: a logistic stacker over layer percentiles,
    fitted on an inner 20% slice of the training window that none of the base
    layers was fitted on. Replaces the unweighted noisy-OR that made Day 4's
    fused score worse than L1 alone.
  - **`mantis/defense/policy/`** (four-decision enum, boundaries placed at FPR
    budgets rather than score values) and **`mantis/defense/explain/`**
    (LightGBM `pred_contrib`, which is the same computation `TreeExplainer`
    performs, without a wrapper on the scoring path).
  - **`mantis/loop/`**: genome → mutation → arena → retrain, plus the zero-day
    demonstration. `data/generated/arena.json` is the gate artefact.
  - **Two new reporting axes, both mandatory from here on**: recall as a
    **curve** over 0.1/0.5/1.0% FPR, and **campaign-level** recall next to
    event-level, each labelled.

- **Day 5 gate (passing)**:
  - `python -m mantis.loop` — 6 cards, 5 generations, 2 pooled seeds, 581 s.
    Evasion **0.626 → 0.381** (falling), 9 survivors of which 6 mutated and
    written back to `mantis/atlas/discovered/`. **Zero-day: 0.811 trained /
    0.013 held out / 0.539 held out + manufactured, 65.9% of the gap closed.**
    `data/generated/arena.json` is the artefact.
  - `python -m mantis.defense` — 1,010,600 events, time-split 707k/303k, **232
    features** (28 of them the new `gph_` block). **L1 AUC-PR 0.5903,
    recall@0.1%FPR 0.450** (Day 4: 0.4910 / 0.3615 — the graph block is worth
    **+0.09 recall** on its own). **Fused 0.483, which finally beats L1**, against
    Day 4's 0.286 that did not. Campaign-level: L1 0.916, fused 0.908, median
    first alert on the ring's **2nd or 3rd** event. L3 **1.000 on F1-01 and
    F1-03**, holding at 1.000 on unseen phrasings *and* on an entirely unseen
    injection kind. L2 0.002 and L2e 0.000 — L2e's ROC is **0.442**, below
    chance. Mean recall lost to holding a family out: **+0.277**. 264 tests,
    ruff clean. Full tables in `RESULTS.md`.

- **Day 7 complete** — the third artifact, the scorecard, and the latency number:
  - **One container, one origin.** `mantis/api/site.py` mounts the Day 6 API
    unchanged at `/api` and serves the built Vite bundle at `/`. `api.ts` already
    defaulted `API_BASE` to the relative `/api`, so a same-origin build needs no
    `VITE_API_BASE` and never exercises a CORS header. The `Dockerfile` is now
    two stages (node builds `web/dist`, python serves it) and installs the new
    `serve` extra only — **verified** that importing `mantis.api.site` loads none
    of pandas, numpy, sklearn, scipy, lightgbm, shap, matplotlib, networkx or
    pyarrow, which takes the image from ~1.5 GB to ~200 MB.
    `scripts/deploy_hf.py` pushes it to a Hugging Face Space; Spaces rather than
    Render/Railway because a free dyno on either **sleeps after 15 minutes** and
    a judge clicking a cold URL sees a spinner.
  - **Deployed, and it is the static Space.** Hugging Face now bills Docker
    Spaces — a free account gets `402 Payment Required` on `create_repo` — so
    `--static` uploads `web/dist` instead. That is **not** a degraded build: the
    bundle is self-contained by construction, because Day 6 froze every API
    response by calling the real route handlers, and the console already replays
    the committed feed on a timer when nothing answers `/api/health`. A CDN has
    no cold start at all, which is a stronger version of the property the Spaces
    decision was buying. The one thing given up is a real SSE stream, and the
    console says which mode it is in rather than letting a viewer assume.
    **Live: <https://aryaambekar-mantis.static.hf.space>** — note the
    `.static.hf.space` subdomain; the plain `.hf.space` form 404s while the Space
    itself reports RUNNING, which is exactly as confusing as it sounds.
  - **`data/reference/` had no Sparkov CSV** — Day 1 ran on the committed
    Indian-market priors, as `ReferenceStats.source` said all along. Kaggle's
    public download endpoint needs no token, so `scripts/fetch_reference.py`
    now pulls the 210 MB panel on demand. It is **gitignored**, and it is used to
    *measure* the population, never to refit it: refitting would re-roll every
    pinned number three days out.
  - **`mantis/foundry/fidelity/`** — criterion 2's artefact. Five sections:
    provenance, marginals (KS/JS against bootstrapped noise bands + correlation
    Frobenius), TSTR, a real-vs-synthetic discriminator with target 0.5, and the
    two known divergences named on its own face. `metrics.py` was **lifted out of
    `scripts/drift_check.py`** so there is one implementation, not two —
    `tests/test_fidelity.py` asserts the identity.
  - **`scripts/latency_bench.py`** — per-event p50/p95/p99 through the full fused
    stack against a 50 ms budget, timed **one event at a time against warm
    state**, plus the same stages in batch so the per-call overhead is visible.
  - **Two new console screens**, Atlas and Fidelity, and the API grew `/latency`
    beside a `/fidelity` that now serves the whole scorecard.
  - **`RESULTS.md` gained three generated sections** — fidelity, latency, and the
    two deployment questions — written by `report.py`, not by hand, because the
    document and the console are both rendered from it.

### Day 7 gate

- `python -m mantis.foundry.fidelity` — discriminator **0.9994**, **0.8399** with
  the two adjudicated axes removed; TSTR transfer ratio **0.030**; correlation
  RMS **0.076**. Writes `data/generated/fidelity.json`.
- `python scripts/latency_bench.py` — end-to-end **p50 117 ms / p95 153 ms /
  p99 171 ms**, **over** the 50 ms budget; the same stages cost **0.68 ms/row**
  in batch; the two stages that genuinely cannot be batched cost **1.02 ms p99
  together**.
- 287 tests, ruff clean. `python scripts/deploy_hf.py --check` green.

### Day 7 findings

- **The discriminator caught a bug in the fidelity measurement itself, which is
  the best possible argument for having one.** The first shape space standardised
  the trailing velocity counts on each panel's own mean and spread. Discriminator
  AUC came back **1.000**. The counts are discrete and overwhelmingly zero — 87.6%
  here, 83.4% in the reference — and standardising maps that *shared* modal atom
  to `-0.370` on one panel and `-0.411` on the other. A single tree split
  separates the panels perfectly while the underlying distributions
  (0.876/0.121/0.003 against 0.834/0.142/0.021) are in fact close. **A z-score of
  a mostly-constant discrete variable is a panel fingerprint, not a comparable
  quantity**, and the measurement was reporting its own transform.
  `tests/test_fidelity.py` builds two samples from the *same* distribution and
  asserts the transform would have separated them, so the lesson is executable.
  Replaced by `gap_ratio_log` (each cardholder against their own median gap) and
  `burst_1h`. `customer_merchant_share` was **removed rather than fixed**: its
  floor *is* the history length, so no centring recovers a comparable quantity.
- **A row sample of the reference panel silently destroys every velocity
  feature.** The first loader took 200k rows uniformly out of 1.85M, which
  deflates every trailing count and inter-arrival gap by roughly ten. The panel is
  now cut to a **contiguous 90-day window**, matching the synthetic span, so each
  retained cardholder's history is intact and the two panels cover the same number
  of days. Same class of error as the Day 5 bindings bug: a transformation that
  cannot fail is a transformation that cannot tell you it fired.
- **On both axes where the discriminator separates the panels, *we* are the more
  realistic side — and that claim is only allowed because it carries a third
  measurement.** Sparkov's hour-of-day curve is a two-level step (peak/trough
  **1.6x**; ours 22.5x) and its 693 merchants are close to uniformly popular (top
  10% carry **14.6%** against the 10% uniformity gives; ours carry 66.0%). Real
  retail has a diurnal curve and real acceptance estates are Zipf, so the
  reference is the side that departs from the domain structure.
  `adjudicate.py` enforces the rule that makes this honest rather than
  self-serving: **a divergence may be attributed to the reference only when a
  third quantity, independent of both panels and stated in advance, says so.**
  Both discriminator numbers are reported, always, because the ablation is a
  judgement a reader must be able to reject.
- **TSTR is 0.030 and that is not a fidelity failure — the gain tables say why.**
  A detector trained on the reference spends **57.6%** of its gain on
  `log_amount_z`; one trained on ours spends **35.8%** on `merchant_rank_pct`.
  Sparkov's fraud is an *amount* anomaly; our classic-rail attacks were built so
  that no single raw column separates them above 0.95 AUC. TRTS confirms the
  symmetry — a real-trained model scores **ROC 0.495** on ours, which is chance.
  So TSTR here measures *whether the two datasets' fraud is the same phenomenon*,
  and it is not, by construction. Do not let the writeup read it as "the synthetic
  data is unrealistic"; the marginal and discriminator sections are what speak to
  that.
- **The latency budget is missed, and the miss is in the calling convention.**
  p99 **171 ms** against a 50 ms budget. But `entity` costs **47 ms** on a one-row
  frame and **0.129 ms/row** in batch — a factor of **364** — because
  `Series.map(dict)` materialises the lookup table into an index on *every call*,
  so a fourteen-feature block pays that cost fourteen times to look up fourteen
  values. Every batchable stage shows 200–360x. The two stages that genuinely
  cannot be batched, because they must read state before folding the event in, are
  the two the architecture was designed around: **velocity 0.660 ms p99 + graph
  0.357 ms p99 = 1.02 ms**. The fix is a plain dict lookup on the single-event
  path and it is **not applied** — the feature builder is shared with the offline
  pass behind every pinned number in RESULTS.md. Report both sentences: the
  implementation misses the budget, and the models are not why.
- **Fusion consumes L3's score, not its decision, and that is what makes the OOD
  result survivable.** Every layer gets three columns — percentile against
  legitimate, standardised raw score, and a "had an opinion" indicator — and no
  threshold is applied before fusion. Since L3's threshold does **not** transfer
  (100% recall and 90% FP on hand-authored controls), a fusion consuming a
  thresholded L3 vote would inherit that failure whole. Consuming the score means
  the stacker sees the *ordering*, which does transfer (ROC 0.999 → 0.811), and
  the fitted weights show it doing exactly that: **-0.943** on L3's percentile,
  **+0.353** on its standardised raw score. The general form is worth keeping: **a
  layer whose ordering transfers and whose calibration does not is still useful,
  provided nothing downstream consumes its threshold.**

### Day 5 findings

- **The bindings bug that would have cost L3 four fifths of its recall.**
  `data/cache/content/bindings.jsonl` held 513 bindings — seed 1337's only —
  because `build_pool` never persisted what the injectors planted. The other
  four seeds' payloads fell through to the **benign pool**, which is the
  deliberate universal-resolution behaviour, so nothing errored and nothing
  warned: L3 would have read innocuous text on 80% of the attack rows and posted
  a recall four fifths too low. `build_pool` now writes the store, and the
  committed file holds 2,343 bindings. The lesson generalises: a fallback that
  cannot fail is a fallback that cannot tell you it fired.
- **L3's first design was 200x too slow and half as good.** Concatenating each
  event's chain into one string and handing 700,000 strings to a vectoriser took
  **16 minutes**, because it re-tokenised the same 234 artefacts hundreds of
  thousands of times. Classifying the *artefact* and taking a **max over the
  chain** takes about a second — and scores higher, because summing a chain's
  vectors dilutes one injected page among eleven innocuous ones. The max is also
  the question a defender actually asks.
- **The entity-level novelty experiment failed, and the failure is the finding.**
  Time-boxed to 30 minutes as planned. L2e — an isolation forest over
  customer and merchant aggregates rather than event rows — did not move the
  number, even though it is *generous* to the hypothesis (an entity's vector is
  aggregated over the whole scoring window, which is a batch review queue and
  not an authorisation scorer). On the five-seed pool it lands at **ROC 0.442 —
  below chance**, which is a stronger statement than "it did not work": entity
  aggregates are *mildly anti-correlated* with fraud here, because the attacks
  ride on established customers and busy merchants by construction while the
  genuinely unusual entities are ordinary people with three transactions. On a
  smaller single-seed run merchant-side entity scoring showed the only flicker of
  life (ROC 0.66) and it does not survive pooling. The general statement is in
  §8 above and belongs in the writeup: **attacks built to be distributionally
  faithful are by construction invisible to distributional anomaly detection**,
  and a fidelity scorecard and an anomaly-detection recall number are therefore
  in tension by construction.
- **Merchants and BINs are excluded from the identity graph on purpose.** With
  16 BINs and a Zipf merchant curve, one union through a popular merchant fuses
  the whole file into a single giant component and `component_size` becomes a
  constant — the classic way naive graph features fail. `tests/test_l4_graph.py`
  pins the property rather than the implementation: the largest identity
  component must hold under 10% of nodes. Merchant-side structure is measured
  the right way instead, as windowed distinct payers and as the number of
  distinct identity **components** paying one merchant — the ratio between those
  two is the ring detector, because many payers spanning few components means
  the payers are related to one another.
- **The manufactured variants are confined to the training window**, and this is
  load-bearing for the zero-day claim. A variant landing in the test period sits
  inside the velocity and graph state of the real test-period attack rows it is
  being evaluated against, inflating their counts and making them easier to
  catch for a reason unrelated to what the detector learned. The comparison
  would have been measuring the injection rather than the transfer.
- **Two determinism defects, both the Day 1 bug again.** `AttackGenome.label()`
  and the provenance rebinding in `mutate.py` were both written with `hash()` on
  a string, which CPython randomises per process — so every arena run and every
  written-back card id would have differed between runs. Both now go through
  `stable_seed`. `tests/test_loop.py` pins a literal label rather than comparing
  two calls in the same process, because a same-process comparison passes on
  exactly the broken implementation.
- **Weighted fusion fixed the Day 4 defect, but not on the first attempt — and
  the second bug is the more interesting one.** Replacing the unweighted noisy-OR
  with a fitted logistic stacker over layer **percentiles** made the fused score
  *worse* than L1 alone again (0.104 against 0.553). The cause is that a
  percentile computed against a finite reference **saturates**: every score above
  the largest legitimate score maps to exactly 1.0, and at a 0.1% false-positive
  budget the events being ranked are precisely the ones in that saturated region.
  L1's ordering inside its own top 0.1% *is* the signal, and the transform threw
  it away, leaving the fused ranking inside the tie to be settled by L2. Each
  layer now contributes two columns — its percentile, and its raw score
  standardised on the fusion window's legitimate rows — and the fused score
  finally beats L1 (0.565 against 0.553 on the smoke dataset).
  `tests/test_fusion_policy.py` pins it with a layer whose entire signal lives
  above every legitimate score.
- **The arena's cost is linear in cards x population x generations**, and all
  fifteen cards at population 8 over two pooled seeds runs for about **two
  hours** — not a gate anybody re-runs. The default is now six cards covering all
  five implemented families, which is ~35 minutes; `--cards all` is still there
  for anyone with the time. A curve does not get truer by averaging over cards
  that behave like the ones already in it.
- **An alert that says `channel = nan` is worse than no alert.** Seven of the
  matrix's columns are categorical, and the explain layer coerced every value to
  a float before printing it, so the single most readable line in the block —
  which rail the authorisation was on — rendered as `nan`. Values now print as
  themselves, and a genuine NaN prints as `absent`, because in this matrix NaN
  overwhelmingly means "this key does not apply to this rail" rather than
  "unknown".
- **A circular import that only the graph could have created.**
  `features/builder` imports `l4_graph.graph`, and `graph` wanted `as_epoch`
  from `features.state`. The import of `as_epoch` is deferred into the function
  body; the dependency direction stays one-way at module scope.

- **Three of the nine arena survivors were the *unmutated* attack**, and that is
  a result about the atlas rather than about the loop. Every card's population is
  seeded with an **identity genome** so the evasion curve carries its own "what
  does this attack do without evolution" reference row; that individual competes
  like any other, and on F2-16, F1-05 and F1-01 it won — the unmutated bust-out
  evades **77.8%** of decisions on its own. Writing those into
  `mantis/atlas/discovered/` as "evasive variants" would have claimed a discovery
  for an attack that was already in `cards/`, which is precisely the overclaim the
  status ratchet exists to prevent. `writeback.is_novel` filters them, the CLI
  reports them separately under their own heading, and six genuinely-mutated
  variants were written back.
- **The evasion curve falls and then rebounds**, 0.626 → 0.334 → 0.338 → 0.363 →
  0.381, and the rebound is the more interesting half. Almost the whole fall
  happens at the **first** retrain: generation 0 is scored against a detector
  that has never seen a mutated variant and generation 1 against one that has.
  After that the adversary claws back about a fifth of the ground by finding
  corners of the gene space the retrain has not covered. The claim the chart
  supports is therefore the bounded one — **retraining on manufactured variants
  cuts evasion by 39% and holds it down** — not that it drives evasion to zero.
  A curve that fell monotonically to zero would be a curve to distrust: with a
  mutation operator that never stops searching, it would mean the search space
  was too small to be interesting. `report._curve_reading` picks its prose from
  the shape the curve actually has, so a future run that rises says so.
- **The loop found the feature we already knew was too good, on its own.**
  `genome.py`'s docstring set this up as the check on whether the arena is really
  searching rather than drifting: *"``deliberation_scale`` is aimed squarely at
  ``mnd_deliberation_residual_z``, the feature the Day 4 review found separating
  F1-01 at 0.99 — the loop should discover that one on its own, and whether it
  does is a check on whether the loop works at all."* Across the six written-back
  variants the median `deliberation_scale` is **2.93**, with five of six above
  2.6 — the adversary consistently stretches the agent's deliberation latency,
  which is precisely the move that kills the residual. It did the same to the new
  graph block: median `merchant_spread` **0.51**, i.e. re-drawing half the
  merchant legs to dilute the beneficiary concentration that
  `gph_merchant_fanin_per_component` measures. Two independent confirmations that
  the search is finding real gradient in the detector rather than wandering.
- **The zero-day demonstration works, and this is the submission's argument.**
  On the loop's own two-seed background, at 0.1% FPR on the real F1 test events:
  **0.811** with the family in training, **0.013** with it held out, **0.539**
  held out plus 4,442 loop-manufactured F1 events — **65.9% of the collapse
  recovered**. What the loop had was F1's *atlas cards*: a written description of
  a class of attack and an executable generator for it. What it did not have was
  a single one of the F1 rows it is then scored on, and the variants are not
  those rows — every gene moved them, and they were selected for **evading** the
  detector, so they sit off-distribution from the canonical attack in exactly the
  direction that makes the transfer hard. The honest statement is *"a family
  described in the atlas but never observed in the data can be manufactured, and
  training on the manufactured version transfers to the real one"* — **not**
  "the detector caught something nobody had thought of". Nothing does that, and
  the whole point of the reframing is that we stopped pretending L2 could.

### The derived-feature probe, and what it found (Day 5 QA)

`scripts/probe_derived.py` (`make derived`) runs the separability gate over the
**built feature matrix** instead of raw columns, closing the blind spot that let
`mnd_deliberation_residual_z` reach 0.99 on F1-01 unnoticed. It **flags and
ranks rather than passing or failing**, because a high number here is not
automatically a defect: a feature measuring the attack's *mechanism* is
detection, and only a feature measuring something the *generator* did that the
attack does not require is an artefact. Verdicts live in `ADJUDICATED` at the
top of the script so the list shrinks to genuinely new findings.

Five features above 0.95 inside their declared slices, all adjudicated:

| card | feature | auc | verdict |
|---|---|---|---|
| F1-04 | `mnd_mcc_in_scope` | 1.000 | **definitional** — being outside the mandated category *is* category drift |
| F1-01 | `mnd_amount_over_ceiling` | 0.980 | legitimate — the card's own declared L1 signal, and the ratio not the breach flag, so F1-01 stays CLEAN |
| F6-39 | `ent_mcc_amount_z` | 0.989 | legitimate mechanism, **narrow generator** — `mcc=7832` alone scores 0.904 off six declared MCCs |
| F6-40 | `txn_round_score` | 0.966 | **artefact** — the raw binary flag is unremarkable, the graded score is not; the injector snaps harder than a real ring would |
| F1-05 | `mnd_delegation_depth` | 0.952 | known and already priced; the raw-column number is 0.94 and the card's docstring concedes depth alone will not carry it |

Two outstanding foundry items fall out of that and are recorded rather than
rushed: widen `f6_39_shell_merchant._DECLARED_MCCS`, and soften F6-40's
round-number snapping. Both re-roll pinned numbers, which is why they are Day 7
scorecard items and not Day 5 edits.

- **Next up**: nothing is scheduled. The three artifacts exist. Outstanding
  items, all recorded rather than rushed: widen
  `f6_39_shell_merchant._DECLARED_MCCS`, soften F6-40's round-number
  snapping, make `decline_amount_tilt` mean-preserving per channel, and the
  single-event dict lookup that would take the latency p99 under budget.
  Every one of them re-rolls a pinned number, which is why none was done.

### The zero-day answer, reframed (Day 5) — READ THIS BEFORE WRITING ANY CLAIM

**L2 is demoted.** It was designed and documented as *the zero-day layer* — the
answer to "what about the attacks you did not think of". Day 4 measured it at
**0.4% mean per-family recall at 0.1% FPR, 0.62 ROC**, and Day 5 did not chase
that number. L2's job in the architecture is now:

> **L2 is a residual monitor and a drift canary.** It answers "has the shape of
> legitimate traffic moved", and it flags the residue that no other layer
> claims. It is not a detector, and no table may present it as one.

The architecture's actual answer to an unseen attack is now two things, both of
which need **no labelled example of that attack**:

1. **L0 protocol invariants.** A mandate that is expired, out of scope, over its
   ceiling, replayed, or whose provenance trail does not terminate at the
   merchant that was paid, is a **violation of the AP2 contract**, not a
   statistical outlier. Nine deterministic clauses, no training data, deployable
   today. An attack that has never been seen still has to break the protocol to
   move money, and the clauses do not care whether it is novel.
2. **The closed loop** (`mantis/loop/`). Rather than hoping an unsupervised layer
   generalises to an attack we never wrote, we **manufacture the attack before an
   attacker does**: the evolutionary adversary mutates known cards into variants
   that evade the current detector, those variants are labelled by construction,
   and L1 retrains on them. The zero-day answer is not "a layer that generalises"
   — it is "a generator that gets there first."

**Why L2 failed, stated as a finding rather than hidden as a weakness.** This is
the most interesting negative result in the project and it belongs in the
writeup:

> **Attacks built to be distributionally faithful are, by construction, invisible
> to distributional anomaly detection.**

Every design decision in the foundry pushed the attacks *toward* the legitimate
manifold — clone real background rows, resample amounts from the target MCC's own
empirical band, redraw the hour of day from the population's diurnal curve,
widen three legitimate tails specifically so an attack would not be free, keep
provenance planting length-preserving. The Day 2/Day 3 separability gate is
literally a rule that says *no single raw column may separate an attack above
0.95 AUC*. An isolation forest measures distance from the legitimate manifold. We
spent two days minimising exactly that distance. **Our own fidelity work caused
this result**, and it is not an artefact — it is the property real GenAI-driven
fraud has, because an agent that pays with a validly-signed mandate on a real
cardholder's real device for a plausible amount at a real merchant *is* a
legitimate-looking transaction in every marginal. The fraud is in the **intent
and the relations**, not in the marginals. That is why the layers that work are
the ones reading relations (L4, entity-level) and intent (L3, the ingested text),
and it is why "just run an autoencoder on it" is not an answer to agentic fraud.

Corollary for the writeup: a fidelity scorecard and an anomaly-detection recall
number are **in tension by construction**. A project that reports both high is
reporting one of them wrongly.

### Day 4 findings that change what we claim

- **The leave-one-family-out result is strong, and only half the story we
  wanted.** Held out of training, L1's mean per-family recall falls from
  **30.8% to 10.5%** (F1 collapses 0.569 → 0.007). Supervised detection really
  does collapse on attacks it has never seen, and that half is emphatic. But
  **L2 does not rescue it**: 0.4% mean recall at the same operating point,
  0.62 ROC. The honest claim is therefore the narrow one — L2's recall is
  *unaffected* by whether an attack was in training, which is a property no
  supervised layer has. **Day 5 stopped treating this as a gap to be closed and
  reframed it as the finding it is** — see "The zero-day answer, reframed"
  above. Do not let the writeup say "the unsupervised layer holds up". It does
  not, and it is not supposed to.
- **Fusion is currently worse than L1 alone** (0.286 vs 0.362). Unweighted
  noisy-OR gives a near-random L2 equal say, and at a fixed 0.1% FP budget that
  costs real recall. The fix is weighted fusion fitted on train — Day 6's job,
  not a coefficient to hand-tune now. **Quote L1's number, not the fused one.**
- **L2's first design was broken by its own sentinel.** Filling NaN with -1e9
  meant an Isolation Forest's random split threshold almost always landed in the
  empty gap, so the trees isolated on the missingness pattern — i.e. on the rail
  — instead of on behaviour. 109 of 204 features are >30% missing on legitimate
  traffic. Filtering to dense columns and median-imputing took recall 0.0006 →
  0.0054, a 7x improvement on a number that is still approximately zero. The
  design rule is chosen on missingness alone and **never tuned against recall**,
  because tuning it would make the layer supervised through the back door.
- **`mnd_deliberation_residual_z` separates F1-01 at 0.99 AUC on its own** —
  above the foundry's 0.95 gate, which never saw it because the gate probes
  **raw columns** and this is a derived residual. Cause: `collapse_deliberation`
  resamples latency from the low band *unconditionally*, so a ₹50,000 purchase
  gets a ₹200 purchase's deliberation time. Ablated and reported: removing it
  costs F1 only 0.569 → 0.549, so L1 is **not** leaning on it — but the gate has
  a structural blind spot for derived features and that is worth fixing.
- **F4-27's oracle was decorative until Day 4.** Declines for a whole campaign
  were drawn in one pass *after* the escalation targets had been chosen, so the
  attack escalated through merchants it had merely *touched* — 65% of escalation
  events landed on an approving merchant, about what chance gives you. Reordered
  so the probe outcome chooses the targets: now **100%**, with probe-phase
  declines at 51-66% and escalation at 8% against an 8.8% background.
- **`_as_bool` is why L0 tests must run in memory, not off parquet.** Written as
  `v is True`, it silently failed on `numpy.bool_` (`np.True_ is True` is False),
  firing `kya_unregistered` on 100% of the CLEAN attacks in memory and 0% after
  a parquet round-trip. A test that only read the committed parquet would never
  have found it.

### The Day 3 bucket contract, checked against a real L0 — VERDICT

Day 3 asserted CLEAN attacks trip zero clauses and HARD attacks fire one on
>=25% of events, against a *provisional* L0. Measured against the real one:

- **CLEAN holds, exactly.** F1-01 and F1-03 fire **0.00%** of nine clauses.
- **F1-05 fails**: its best clause fires on **3%**, not 25%.

**The contract is wrong, not L0** — and specifically, the Day 3 test had no
false-positive term. It satisfied F1-05 with `delegation_depth > 2` and never
priced it. Priced: `depth > 2` gets 64% recall at a **4.99% FP rate on legitimate
agentic traffic** (1,498 events per 30,040). No issuer ships that, so it is not
an L0 clause — it is a weak classifier with a rule's syntax. At `depth > 5`, the
only threshold above the legitimate tail Day 3 deliberately widened, F1-05 fires
on 3%. Four of the five HARD cards survive the omission because their clauses
genuinely are free; F1-05 is the one where it was load-bearing, and its own
docstring already conceded that "depth alone will not carry this card" and that
the real answer is L4. **Neither side was adjusted to agree.** `tests/test_l0_rules.py`
pins the exception so it stays noisy.

- **`provenance_untrusted_domain` is implemented, measured, and switched OFF.**
  It catches 100% of both CLEAN attacks at 0.00% FP — because the foundry draws
  attacker URLs from twelve hosts that appear nowhere in legitimate traffic. That
  is a partition the generator created, not a detection. It would also let L3
  post a recall it had not earned by reading a word. Replaced by
  `provenance_merchant_mismatch`, a real invariant: the trail must terminate at
  the merchant that was paid.

### Day 4 carry-over results (Task 0)

- **Probe slices audited** (`scripts/audit_probe_slices.py`, `make slices`): all
  legal, all proved to be a function of their declared columns alone, all
  containing the whole attack. `THIN_SLICE_ROWS` raised 750 → **2,000** — the old
  value had been fitted to the one case it was meant to judge. One thin slice:
  **F1-03 at 620 rows**, flagged; quote its conditional AUC with its n attached.
- **Distribution drift measured against a bootstrapped sampling-noise band**
  (`scripts/drift_check.py`, `make drift`), 33 marginals, KS for continuous and
  JSD for categorical. Everything inside its band except:
  1. **`decline_reason` at 302x its band.** The reason-remapping (invalid_cvv →
     do_not_honor where no CVV was presented, expired → insufficient_funds where
     the mode cannot expire) is far larger than its comment claims: invalid_cvv
     **0.130 → 0.036**, expired **0.080 → 0.033**. Only ~27% of declines are on a
     CVV-bearing entry mode, so 73% of drawn invalid_cvv gets remapped.
  2. **Realised decline rates run above the per-channel priors on every rail** —
     moto **2.32x** (0.325 vs 0.140), upi_p2p 1.44x, ecom 1.33x, agentic 1.24x,
     overall 0.088 vs a mix-weighted nominal 0.074. Cause: `decline_amount_tilt`
     multiplies by `exp(0.55 z)`, whose expectation is `exp(0.55²/2) ≈ 1.16` —
     Jensen, not redistribution. The tilt should be mean-preserving per channel.
  Neither is fixed: both are conservative for detection (they make F4-27's lift
  *smaller* against a higher background), and re-tuning the population on Day 4
  would re-roll every pinned calibration number while the firewall was being
  stood up. Both are Day 7 scorecard items.
  Three measurement bugs in the *script* were found and fixed first: entity-drawn
  columns (`merchant_country`, `card_bin`, `agent_platform`) need an
  entity-level null band or Zipf popularity makes ordinary noise read as 339x
  drift; the day-of-week target must be weighted by the calendar, since 90 days
  is not a whole number of weeks; and the passive-human share must be
  deconvolved from the cursor-entropy mixture rather than thresholded.
  All three widened Day 3 tails verify: instant refunds 0.213 vs 0.24 on card
  rails, passive humans 0.1153 vs 0.11, delegation depth>=4 0.0151 vs 0.0150.

### Outstanding, recorded rather than papered over

- **Day 2 cards under-declare their rails.** Writing
  `test_attack_rails_agree_with_the_card` surfaced that several Day 2 injectors
  clone whatever rail their source row was on and therefore ride rails their
  card does not list — F2-13 reaches `card_present`, `recurring` and `upi_p2p`
  against a card naming `agentic`/`ecom`/`upi_p2m`. The subset assertion is
  therefore enforced for the Day 3 F1 cards only. Quietly widening seven Day 2
  cards to make a new test pass would be the wrong way round; the reconciliation
  is a Day 2 job that has not been done.
- **The committed corpus was authored with `mistral:7b-instruct-q4_K_M`**, not
  the `qwen2.5:7b` the plan named, because that is the model that was pulled
  locally. `DEFAULT_MODEL` matches it deliberately: the cache key is a hash of
  *(model, prompt, params)*, so a default that did not match the committed
  entries would give a judge a cache miss on every prompt and a silent fall
  through to the bundled corpus — HARD RULE 3 satisfied in letter, not in
  substance. Switching is one line plus `--live --refresh` to re-author.

### Day 3 decisions worth not relitigating

- **The HARD/CLEAN split is the whole design, and it is asserted.** If every
  agentic attack were a clean protocol violation, a Day 4 L0 rule would catch
  all of them at near-zero FP, L1/L2 would have nothing to do on the rail the
  project is about, and the ML story would collapse to "we wrote some
  if-statements". So each F1 injector declares `bucket`, and
  `tests/test_agentic_injectors.py` checks the *behaviour* against the
  declaration: **CLEAN attacks (F1-01, F1-03) trip zero L0 clauses at zero
  tolerance** — no scope violation, no expired mandate, no invalid signature, no
  unregistered agent, no ceiling breach — while each HARD attack must fire a
  deterministic clause on ≥25% of its events. Measured: F1-01 and F1-03 at
  0.00 on every clause; F1-02 mcc-outside-scope 0.61, F1-09 invalid-consent
  0.47, F1-10 expired 0.67 with 21 hash reuses.
- **Two easy shapes were deliberately not generated for F1-03**, and saying so
  matters more than generating them: an *orphan* credit (`txn_type='credit'`
  with no `original_event_id`) and a credit routed to a different instrument.
  Both are now representable and both are one-line L0 catches. Putting them in
  the CLEAN exemplar would let it claim behavioural detection while drawing its
  recall from L0.
- **The probe now measures inside a declared definitional slice**
  (`BaseAttack.probe_slice`). Some properties of an attack are its definition,
  not evidence: every F1 attack carries an agentic block and only ~15% of the
  file does, so *any* F1 injector scores ~0.92 on the nullity of *any* `ag_`
  column before doing anything; F1-03 is a refund attack and refunds are ~2% of
  traffic, so `txn_type=refund` alone scores 0.99. **The gate applies to the
  conditional number**; the unconditional one is still printed so the change of
  denominator is visible. This is the Day 1 "rail identity is unhideable"
  decision applied consistently and declared per injector.
- **Three population tails were widened because an attack would otherwise have
  been free**, and each is defensible on its own terms:
  1. `human_present_passive_share` (11%) — people who watch the agent work
     without touching the device. Without it, *(human_present=True, machine-like
     cursor entropy)* was a **perfect** detector for F1-09. It is now
     **16.7% recall at 3.7% FP**, which is a real trade-off rather than a
     generator artefact.
  2. `refund_instant_share` (24%) — instant refunds are a real merchant
     offering. Without it, F1-03 sat at **0.996** on `settlement_lag_hours`
     alone; it is now 0.83.
  3. `delegation_depth_weights` out to 5 — legitimate multi-agent orchestration
     genuinely chains. Without it, `depth >= 4` was a perfect detector for
     F1-05; it is now 0.94, still the closest number in the atlas to the gate,
     and reported as such.
- **Two generator artefacts the probe caught, both fixed at source.** Content
  planting originally *extended* the provenance chain, making
  `ag_provenance_chain_len` a 0.96 detector — an attack detectable by counting
  URLs without reading one. Planting is now length-preserving. And
  `collapse_deliberation` originally *added* to the cloned tool-call count,
  pushing it past the legitimate maximum (0.96); it now resamples from the
  background's own upper band, the same discipline as `draw_amounts`.
- **F1-04 and F1-05 are written against the cards that exist, not the brief's
  names.** The Day 3 plan called F1-04 "merchant-endpoint impersonation" and
  F1-05 "vector-memory poisoning"; in the frozen atlas those ids are
  *intent-mandate category drift* and *delegation-chain laundering*, and memory
  poisoning is F5-36. Same principle as the four Day 2 remappings: an injector
  that generates something other than what its card describes is exactly the
  overclaim the registry assertion exists to prevent.
- **F5 stays empty on purpose.** It is the zero-day holdout family — an attack
  family the detector never trains on — and `tests/test_atlas.py` pins the
  implemented-family set so that emptying or filling it is a deliberate act.

### Day 3 review — four things checked afterwards, and what they found

1. **F4-27 is not inverted; the *probe report* was.** Card testing's canonical
   signature is an elevated decline ratio, and the injector produces one:
   **46.0% campaign-wide against a 9.0% background (5.1x), 50-66% inside the
   probe phase, 0-17% inside escalation.** What was wrong was the reporting. The
   probe takes `max(a, 1-a)` per feature, so the magnitude is **direction-blind**
   — `auth_response=approved 0.69` reads as "approved more often" when the truth
   is the exact reverse. Every probe row now carries `direction` (`hi`/`lo`) and
   the table prints it, because a Day 4 model trained off a misread table would
   learn card testing with the sign flipped. Same fix, same reason, as the
   `is_fraud`-never-a-feature rule: make the mistake impossible to make quietly.

2. **The probe slice is now audited mechanically, not declared.** The rule:
   *a slice may condition only on facts a detector knows before it scores, and
   which are not a consequence of the attack.* Enforced three ways in
   `tests/test_probe_slices.py` — declared columns must be on
   `probe.SLICE_ALLOWED_COLUMNS` (rail, processing code, category; **not**
   amount, auth_response, or any `ag_scope_*`/behavioural column); the returned
   mask is proved to be a **function of the declared columns alone** by grouping
   the background on them and asserting the mask is constant inside every group;
   and the slice must contain every attack row, so it cannot cherry-pick. A
   slice of "agentic AND provenance_len > 3" fails the second check whatever it
   declares. Slices in use: F1-* on `ag_agent_id` nullity, F1-03 on that plus
   `txn_type`.

3. **The slice denominator is printed, and thin ones are flagged.** F1-03's
   slice is agent-mediated refunds — genuinely small (~140 rows at 40k, ~700 at
   200k), so its 0.83 is a fragile number. The table now prints `slice n` and
   marks anything under 750 rows with `!`. The number is surfaced rather than
   engineered away, because refunds *being* rare is a fact about refunds.

4. **Cumulative calibration drift is ~nil, and now pinned by a test.** Between
   Day 1 and Day 3 the population gained declines, refunds, reversals, pre-auths,
   credits, bimodal settlement, a passive-human tail, instant refunds and a
   deeper delegation tail — three of them added specifically to make attacks
   harder. Measured at 200k: amount KS **0.0051 → 0.0062**, hour-of-day TV
   **0.0066 → 0.0051** (improved), MCC mix max delta 0.0010 → 0.0011, median
   ticket ₹782 → ₹782.62. The audit passes **30/30**, with every new lifecycle
   column included and under its tier bound (worst: `settlement_lag_hours` at
   0.62 against a 0.70 neutral bound). Drift is that small because refunds and
   reversals are **conversions** of rows that would otherwise have been
   purchases, copying `(mcc, amount)` from a real earlier row — a conversion
   drawn from the population preserves its marginals. `test_population.py` now
   pins the three distances so Day 7's scorecard cannot be a surprise.

### The number to state before a judge derives it

**Fraud is concentrated on the agentic rail, heavily and by design.** At the
Day 3 gate: agent-mediated traffic is **15.4% of volume and carries 51% of the
fraud** — 3.48% prevalence against 0.61% classic, a **5.7x concentration**.

That is defensible and is the project's thesis in one number: a new rail with
immature controls is where attackers go. But it has a consequence that must be
said out loud rather than discovered: **the presence of an agentic block is by
itself a strong predictor of fraud in this file**, and any model will lean on
it. So the Day 4 firewall must report **recall@0.1%FPR within each rail** as
well as overall — a headline number computed across both rails is partly
measuring "is this agentic", which an issuer already reads free off the
authorisation message. `python -m mantis.foundry` prints this block on every
run and records it in the manifest.

- **Day 1 audit complete** (`scripts/audit_population.py`, 30/30 checks). It is
  adversarial by design and re-runnable: `python scripts/audit_population.py`.
  It found and we fixed three real defects, all now pinned by regression tests:
  1. **Reproducibility was broken.** A set of strings was iterated inside a loop
     that consumes the RNG, so `--seed 7` produced a different population on
     every process (CPython randomises string hashing). The numbers on the
     slides would not have survived a judge re-running the pipeline.
  2. **`device_id` was a perfect rail separator** (AUC 1.000, zero of 8,796
     devices carried both rails). Agents now run on-device about half the time,
     on the cardholder's existing hardware.
  3. **Two more separators from over-clean modelling**: a binary adopter flag
     gave 70% of customers a hard-zero agentic probability (`customer_id`
     AUC 0.90 -> 0.75, now a graded Beta propensity), and the agentic 3DS mix
     had no failure tail (`threeds_result` AUC 0.86 -> 0.75).

### Day 1 decisions worth not relitigating

- **Rail identity is unhideable, and that is fine.** `channel == "agentic"` names
  the rail and every `ag_*` column is non-null exactly on it, so "can a model
  detect the rail" is not a meaningful leakage test. The audit therefore tiers
  columns: definitional (no bound), correlated-by-design (bound 0.85, each with
  a written causal reason), and neutral (bound 0.70). `device_id` sits in tier 1
  on reflection - a hosted agent runtime genuinely is a device that only ever
  transacts agentically, and exposing that is what KYA is *for*. The test that
  actually protects Day 2 is different and now runs: **amount must be
  rail-independent given MCC** (worst per-MCC KS 0.076), so that once attacks
  land - and attacks skew agentic - the detector cannot pass by learning
  "unusual amount -> agentic -> fraud".
- **F1-03 (refund-logic hijack) stays `mapped`.** Refunds are not representable
  in the frozen schema, so it must not carry a generator path.
- **Convention: `implemented` iff `generator` is set.** A `mapped` card names no
  generator, because a path implies an injector that does not exist. Enforced by
  `tests/test_atlas.py`.
- **Legitimate agentic traffic has messy tails on purpose**: ~2.8% not
  KYA-registered, ~0.3% consent signature invalid, `amount/scope_max` reaching
  0.99. Without them L0 would score perfect recall on a generator artefact.
- **Amount KS is non-zero by design** (round-number snapping) and the national
  Zipf curve is flatter than the per-pool exponent (locality-conditioned
  merchant choice). Both are explained on the figure rather than tuned away.

### Day 2 decisions worth not relitigating

- **The implemented count went 15 -> 8, and that is a tightening, not a
  regression.** Day 1's convention was "implemented iff a `generator` path is
  named". Day 2 replaced it with "implemented iff an injector exists, is
  registered, and its declared path resolves" - checked at import. Nine cards
  (F1-01, F1-02, F1-09, F1-10, F2-14, F3-20, F4-29, F5-33, F5-34) named a path
  with no code and went back to `mapped`; F2-16 and F6-40 gained injectors and
  were promoted. The number is now a **ratchet**: it moves only when code lands.
  F1 and F5 have no implemented card today; they return on Day 3 with the
  agentic injectors, and `tests/test_atlas.py` pins the family set so that is a
  deliberate act rather than a drift.
- **Four of the Day 2 attack ids were remapped to the cards that actually carry
  those semantics.** The requested list named F2-17 for bust-out, F2-18 for
  mule fan-in, F5-34 for miscoded-MCC laundering and F4-28 for card
  enumeration; in the frozen atlas those ids are merchant onboarding, agent
  reputation farming, platform compromise and threshold structuring. Injectors
  were written against **F2-16, F6-38, F6-39 and F4-28** - the cards whose
  `description` and `observable_signals` match the attack - because an injector
  that generates something other than what its card describes is exactly the
  overclaim the registry assertion exists to prevent.
- **Chargeback / refund abuse is not representable and was substituted, not
  faked.** `TxEvent` is an authorisation message: no refund flag, no dispute
  outcome, no authorisation response code. F6-40 (stored-value cash-out ring)
  ships in its place - same actor, same ring topology, same objective, every
  claimed signal present in the data. This is the same constraint that keeps
  F1-03 at `mapped`, and it is recorded in `f6_40_stored_value.py`'s docstring
  so the substitution is visible rather than quietly absent.
- **Injectors return only new rows and never mutate the background.** The Day 1
  calibration was measured on that background; editing it in place would make
  the fidelity scorecard measure the attacks instead of the simulation. It also
  makes per-attack accounting and the zero-day holdout separable by
  construction.
- **Attack events are clones of real background rows, retargeted.** Card BIN,
  device, IP, geo, entry mode, 3DS outcome and the nullity pattern all come from
  the legitimate population. Amounts are resampled from the population's *own*
  per-MCC empirical distribution inside a quantile band. Fraud that only touches
  freshly-minted entities is trivially detectable; a test asserts every attack
  customer and merchant already exists.
- **The probe caught two generator artefacts, both now fixed in the framework.**
  Injectors originally scheduled uniformly across the 24 hours, which against a
  diurnal population made `ts_hour` the strongest single feature on three
  attacks; `set_timestamps` now redraws each event's time of day from the
  background's own hour curve (blended 18% toward uniform), shifting whole
  bursts together via a `groups` argument so escalation gaps and cultivation
  cadence survive intact. Campaign starts were also drawn uniformly and left
  calendar gaps the probe read as a `ts_epoch` signal; `spread_epochs` is now
  stratified. Two attack-level artefacts went the same way: F2-13's cohort was
  filtered to a late first-seen date (0.81 AUC on `ts_epoch`) and F3-19 used one
  fixed amount cap that pinned most of a campaign to a single repeated number.
- **F3-19 at 0.872 on `amount` is the honest ceiling, not a bug.** A coerced
  transfer really does sit at the top of its victim's range. What the number
  does not support is a threshold - the population's 87th percentile is ordinary
  traffic - so recall must come from `amount_vs_customer_p99` and beneficiary
  fan-in. Every injector records its measured number in its module docstring.
- **Prevalence, not raw count, is the invariant.** `BaseAttack.n_events` scales
  `base_events` by the actual background size against a 200k reference, so a
  40k smoke run and a 200k gate run both land near 0.64%. Letting prevalence
  swing with `--n` would make AUC-PR incomparable between runs.
