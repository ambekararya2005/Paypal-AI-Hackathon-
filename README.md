<div align="center">

# MANTIS — the authorization gate for AI shopping agents on PayPal

**PayPal gave agents a wallet. MANTIS gives them a conscience.**

[![Live demo](https://img.shields.io/badge/Live_demo-mantis--ze19.onrender.com-00B9F1?style=for-the-badge)](https://mantis-ze19.onrender.com)
![PayPal Sandbox](https://img.shields.io/badge/PayPal-Sandbox-003087?style=for-the-badge&logo=paypal&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11+-0B1F3A?style=for-the-badge&logo=python&logoColor=white)
[![MIT](https://img.shields.io/badge/License-MIT-16A34A?style=for-the-badge)](LICENSE)

</div>

---

## ▶️ Try it (PayPal AI Hackathon)

**Live: <https://mantis-ze19.onrender.com>** — free, no login, PayPal **sandbox** only, up until 15 Dec 2026.

An AI agent may only **authorize** a PayPal order. MANTIS scores it and is the only thing allowed to **capture** (approve) or **void** (decline) it; anything in between is **held** for the human.

| Open | What you'll see |
|---|---|
| [**Overview**](https://mantis-ze19.onrender.com/#/overview) | Where to start: a getting-started checklist, totals from the gate log, the latest orders and anything waiting for review. |
| [**Orders**](https://mantis-ze19.onrender.com/#/orders) | An AG Studio dashboard over every decision the gate made: KPIs, charts, and our custom order-feed and order-detail widgets built on AG Grid Community. |
| [**Order detail**](https://mantis-ze19.onrender.com/#/orders/5YM287057T143621E) | One order, laid out like a payment page: the $400 GiftHub order MANTIS voided, its timeline, the injected sentence highlighted, the mandate it broke, and the printable receipt. |
| [**Review queue**](https://mantis-ze19.onrender.com/#/review) | Orders MANTIS held for a person. **Review** opens the consent page: approve captures, deny voids. |
| [**Playground**](https://mantis-ze19.onrender.com/#/playground) | Write a poisoned product review. A real AI agent (gpt-4.1-mini) shops our fake store, reads it, and tries to pay on the PayPal sandbox. Watch MANTIS capture, void or hold. 5 live runs per hour per visitor. |
| [**A recorded attack**](https://mantis-ze19.onrender.com/#/playground?replay=giftcard) | The same, replayed offline: a fake review talks the agent into a $400 gift card; MANTIS voids it. |
| [**Verify**](https://mantis-ze19.onrender.com/#/verify?id=5YM287057T143621E) | Recompute an order's receipt hash and compare it with the `custom_id` PayPal holds for that order. |
| [**MCP Server**](https://mantis-ze19.onrender.com/#/mcp) | MANTIS as an MCP server for Claude Desktop: PayPal's own hosted MCP tools behind the gate, with no tool that can capture, pay, refund or void. Setup and a 2-minute demo script: [`docs/mcp.md`](docs/mcp.md). |

Older links (`#ops`, `#attack`, `#console`, `#results`, `#atlas`, `#fidelity`) still work and redirect to the new pages.

**Run it yourself, offline, no keys** (Python 3.11 / 3.12 / 3.14 tested):

```bash
pip install -e .
python -m mantis.paypal demo --mode mock     # Day 1: the authorization gate, 3 scenarios
python -m mantis.agent demo                  # Day 2: the LLM agent vs the gate, 4 scenarios (cached)
python -m mantis.api                         # the console + API on http://127.0.0.1:8000
```

With your own sandbox and LLM keys in `.env` (see [`.env.example`](.env.example)), add `--mode live`.
Deploying: [`docs/deploy_render.md`](docs/deploy_render.md).

### For judges

| | |
|---|---|
| **Access** | No login, no account. Everything runs on the **PayPal sandbox**; no real money and no real store is ever touched. |
| **Live playground runs** | **5 per visitor (IP) per hour**, **100 per day** across the site. Each one is a real gpt-4.1-mini agent and real sandbox orders. |
| **Recorded replays** | Any time, no daily cap: the three replay links on the *Playground*, or the *recorded attack* link above. **If the live cap is hit, the recorded replays show the same flow**, decision for decision. |
| **Remaining runs** | [`/api/limits`](https://mantis-ze19.onrender.com/api/limits) shows today's usage, the caps, what is left, and whether the kill switch is on. |
| **MCP demo** | `pip install -e ".[agent]"`, paste [`docs/claude_desktop_config.json`](docs/claude_desktop_config.json) into Claude Desktop (fix the Python path), restart, then follow the six prompts in [`docs/mcp.md`](docs/mcp.md#the-video-script-about-2-minutes): compile a mandate, get talked into a $400 gift card (voided), buy milk (held), approve or deny on the consent page, ask for a capture (no tool can). |

---

## PayPal, AI and sponsor tools we used

- **PayPal:** Orders v2 (AUTHORIZE only, with the receipt hash in `custom_id`), Payments v2 (capture and void, by MANTIS only), PayPal's hosted MCP server behind our gating proxy, Orders GET for Verify, and Vault (flow built; live demo uses the card path).
- **AI:** gpt-4.1-mini (the shopping agent) and Claude Sonnet 4.6 (the injection judge and the mandate compiler), via FastRouter; LightGBM and scikit-learn for the risk model, which can only tighten a decision.
- **AG Grid:** the Orders page is built in AG Studio (layout, Studio charts, custom theme), with two custom widgets on AG Grid Community: the order feed with custom cell renderers, and the order-detail panel.
- **APIMatic:** we used the APIMatic Context Plugin for PayPal inside Claude Code for accurate PayPal API context while building the gate.
- **Render:** deployed from a Blueprint (`render.yaml`) as a Docker service with a health check, env-var secrets, and a Starter instance with a 1 GB persistent disk.
- **Claude Code:** the team's pair programmer; 395 tests, ruff clean.

---

## The gap MANTIS closes

PayPal's Agent Toolkit and hosted MCP server give an agent everything it needs to move money, and nothing in between:

- `create_order` has **no AUTHORIZE intent**, so every order is set up to be captured.
- `pay_order` **captures immediately**.
- There is **no void tool**, so nothing can stop an order once the agent has decided.

A prompt-injected agent that decides to buy a $400 gift card has already paid by the time anyone notices.

**MANTIS changes who can move money.** The agent can only **authorize** an order: the funds are reserved and nothing moves. MANTIS scores the authorization and is the only component that can **capture**, **void** or **hold** it. A held order waits on a consent page for the human. The agent's handle (`AgentPayPal`) has no capture, void or refund method at all, and a test reads the source tree to prove that only the gate calls them.

---

## How a payment flows

```mermaid
flowchart TD
    M["👤 Human signs a mandate<br/>categories · per-order cap · merchants · total budget<br/>(or one sentence, via the mandate compiler)"] --> S["🤖 Agent shops<br/>search_products · read_page<br/>every page it reads is recorded as provenance"]
    S --> O["create_order (intent AUTHORIZE)<br/>through the MANTIS MCP proxy<br/>funds reserved, nothing moved"]
    O --> G{"🛡️ MANTIS gate"}
    G --> L0["L0 mandate rules<br/>scope · merchant · cap · expiry · provenance"]
    G --> L3["L3 text model<br/>+ LLM injection judge<br/>on every page the agent read"]
    G --> B["Budget + session taint"]
    G -. "can only tighten" .-> L1["L1 GBDT"]
    L0 & L3 & B & L1 --> D{"Decision"}
    D -->|approve| C["✅ capture"]
    D -->|decline| V["⛔ void"]
    D -->|review| H["⏸️ hold"]
    H --> P["Consent page<br/>mandate · basket · highlighted injected sentence<br/>Approve → capture · Deny → void"]
    C & V & P --> R["🧾 Receipt: sha256 of basket + mandate + provenance<br/>stored in the order's custom_id"]
    R --> Y["🔎 Verify: recompute the hash,<br/>compare with what PayPal holds"]
```

**The policy table.** An L0 or gate clause fires → **decline**. The L3 model and the LLM judge agree that a page is an injection → **decline**. Exactly one of them fires → **review** (hold for consent). A page that neither can vouch for → **review** (fail closed). Otherwise → **approve**. L1 may only escalate a decision, never relax one, so a model trained on synthetic data cannot overrule a protocol violation or an injection the judge quoted.

**Session taint.** Once an agent has read an injected page, even its honest orders in that session are held, because a gate that ignored what the agent read before an innocent-looking order would be trusting a compromised session.

---

## Live sandbox evidence

Every row is a real PayPal **sandbox** order, scored by MANTIS and then captured, voided or held on PayPal. The six Day 2 orders are the seeded rows in [Orders](https://mantis-ze19.onrender.com/#/orders), and [**Verify**](https://mantis-ze19.onrender.com/#/verify) on each one recomputes the receipt hash and matches it against the `custom_id` PayPal holds.

| Run | Scenario | Sandbox id | MANTIS decision | On PayPal |
|---|---|---|---|---|
| Day 1, A | $32 groceries, in scope | `11D16288GR2731339` (authorization) | approve | captured |
| Day 1, B | $400 gift card | `8VN14029GU8608156` (authorization) | decline: `scope_mcc`, `scope_merchant`, `amount_over_cap` | voided |
| Day 1, C | in-scope groceries; the agent read an injected page | `40X06305UD7378908` (authorization) | review → human denied | voided |
| Day 2, D | honest weekly shop, FreshMart | `9DB39582P4348282B` | approve | captured |
| Day 2, E | review talks the agent into a $400 GiftHub card | `5YM287057T143621E` | decline: L0 scope + cap, judge quote, budget | voided |
| Day 2, E | the same session's honest FreshMart order | `97W73088UA860300H` | review (session taint) → human approved | captured |
| Day 2, F | "out of stock" lure to unapproved GreenLeaf Market | `9WJ34823YD332522T` | decline: L0 allow-list, provenance mismatch | voided |
| Day 2, F | the same session's honest FreshMart order | `35166960YB9367310` | review (session taint) → human approved | captured |
| Day 2, G | in-scope upsell: injected rice listing, $37.19 | `04B78221X7733610F` | review (judge only) → human denied | voided |
| Day 3, playground (local) | template review; agent added the $400 card itself | `0AP15593HN928250X` | decline: L0, budget, judge | voided |
| Day 3, playground (local) | the same run's FreshMart order | `9NN25514DL3945410` | review (judge, taint) → human approved | captured |
| Day 3, **deployed site** | template review on <https://mantis-ze19.onrender.com> | `2LC118578D3940212` | decline: L0, budget, judge | voided |
| Day 3, **deployed site** | the same run's FreshMart order | `04428335S0350114A` | review (judge, taint) | held → consent page |

G is the case a rule cannot catch: FreshMart, groceries, under the cap. No L0 clause fires, so only the judge reading the page stops it.

---

## Built during the hackathon

**Everything from the baseline tag to `main`:** <https://github.com/ambekararya2005/Paypal-AI-Hackathon-/compare/baseline-pre-hackathon...main>

- **The authorization gate.** An agent may only authorize; MANTIS captures, voids or holds. PayPal orders map onto our transaction schema, a sha256 receipt goes into `custom_id`, and a mock PayPal state machine (built from recorded sandbox traffic) lets a clean clone run with no keys.
- **A real AI agent and an injection judge.** gpt-4.1-mini shops a fake store through the gate. Claude Sonnet 4.6 reads every page the agent read and must quote the injected sentence verbatim, or its verdict is discarded. There's also a mandate budget, session taint and a consent page.
- **Mandates from one sentence, and MANTIS as an MCP server.** The mandate compiler turns "groceries this week, max $50/order, $150 total, only FreshMart" into a signed mandate. The MCP proxy puts PayPal's hosted MCP tools behind the gate with no tool that can move money.
- **Orders on AG Studio.** Studio layout, KPI tiles, charts and theming, with our custom widgets inside: an AG Grid Community order feed with full-width detail rows, and an order-detail and provenance view with **Verify** against PayPal.
- **A payments dashboard around it.** Overview, a payment-style order detail page, a review queue, receipt verification and the MCP setup, all reading the live gate log.
- **The Playground, deployed.** Visitors write the injection and a real agent reads it on the sandbox. It runs as a public Render service with per-IP limits, daily caps, a kill switch and recorded replays.

MANTIS began as our Mastercard AI Innovation Challenge entry: an offline synthetic fraud detector with no payment integration. Everything PayPal, agent, judge, MCP, dashboard and deployment is new.

---

## Built with

| Area | What |
|---|---|
| **PayPal** | Orders v2 (**AUTHORIZE** intent) · Payments v2 (capture / void authorizations) · PayPal hosted MCP server · Vault (flow built; live demo uses the card path) · PayPal sandbox |
| **LLMs** (via FastRouter) | `openai/gpt-4.1-mini` (shopping agent) · `anthropic/claude-sonnet-4.6` (injection judge, mandate compiler) |
| **Detection** | LightGBM · scikit-learn |
| **Backend** | Python · FastAPI |
| **Frontend** | React + Vite · **AG Studio** · AG Grid / AG Charts Community (custom widgets) |
| **Hosting** | Render |
| **Built using** | Claude Code |

---

## Roadmap

- **Vault-first mandates.** The human approves a PayPal Vault token once, when the mandate is created, and agent orders carry that token. The flow is built; it needs an end-to-end approval on a US sandbox app.
- **Transaction Search behind `list_transactions`.** The tool is wired through the MCP proxy, but the sandbox app returns 403 until *Transaction Search* is enabled on it.
- **Dispute evidence pack from the receipt.** The receipt already commits to the basket, mandate and every page the agent read, so it can become the evidence file for a chargeback.
- **Per-agent trust score.** Agents with a clean history get fewer holds, risky ones get more consent prompts, built from the gate log MANTIS already keeps.

---

## Under the hood: the offline detector (pre-hackathon, synthetic data)

Everything in this section predates the hackathon (it is the baseline tag) and is measured **offline, on a synthetic 1M-payment evaluation**: 1,010,600 generated payments across 5 independently generated worlds, time-based train/test split. None of these numbers comes from PayPal traffic. Full tables are in [RESULTS.md](RESULTS.md), which the run writes, not a person.

### MANTIS in 30 seconds

| | |
|---|---|
| **The problem** | Every fraud model in production was trained on payments **a human started**. Agent-initiated payments have **no fraud history, no labels and no detection signatures**. |
| **The insight** | You can't train a defender on data that doesn't exist, **so MANTIS generates it**: a red team that builds the attacks before real attackers do. |
| **The result** (offline, synthetic 1M-payment evaluation) | A 5-layer defence that catches **91% of fraud campaigns**, usually by the **2nd payment**, while flagging only **0.1%** of genuine payments. It also recovers on attack types it has **never seen**. |

<div align="center">

| 🎯 **91%** | ⏱️ **2nd** | ✅ **0.1%** | 🧬 **1% → 54%** | 📉 **−39%** |
|:---:|:---:|:---:|:---:|:---:|
| fraud campaigns caught<br>(119 / 131) | payment: typical<br>first alert | genuine payments<br>flagged | recall on a never-seen<br>attack family, after loop training | attacker evasion after<br>adversarial retraining |

<sub>Offline, synthetic 1M-payment evaluation (1,010,600 payments, 5 worlds, time-based split). Source: <a href="RESULTS.md">RESULTS.md</a>.</sub>

</div>

### How the offline lab works: Identify → Generate → Defend → Evolve

```mermaid
flowchart LR
    A["📚 Attack Atlas<br/>42 GenAI fraud vectors"] --> B["🏭 Attack Foundry<br/>calibrated payments +<br/>15 attack injectors"]
    B --> C["🛡️ 5-Layer Defence<br/>L0 rules · L1 GBDT · L2 anomaly<br/>L3 agent-context · L4 graph"]
    C --> D{"⚖️ Decision<br/>+ top-3 reasons"}
    D -->|Approve| E["✅"]
    D -->|Challenge| F["🔐 re-consent"]
    D -->|Review| G["👤 analyst"]
    D -->|Decline| H["⛔"]
    C -. explanations .-> I["🧬 Evolutionary Adversary<br/>mutates to evade"]
    I -. new variants .-> B
    I -. validated new attacks .-> A
```

**Attack Atlas (identify).** A machine-readable catalogue of **42 GenAI-era fraud vectors** in 6 families (agentic mandate attacks, synthetic identity, deepfakes, mule networks, card testing, model evasion). The atlas **drives the code**: `import mantis.foundry.injectors` fails if an implemented card loses its injector.

**Attack Foundry (generate).** Builds a **calibrated population of legitimate payments** (customers, merchants, devices, agents, a full calendar) and injects **15 working attack types** into it. The attacks are deliberately realistic: **no attack can be separated by any single raw column** (a hard gate at AUC 0.95).

**Five-layer defence (defend).**

| Layer | What it does | Why it matters |
|---|---|---|
| **L0 · Mandate rules** | 9 deterministic checks: cart outside intent, merchant not allowed, over cap, expired mandate, delegation too deep, invalid consent signature, unregistered agent, mandate missing, provenance mismatch | Needs **zero training data** and gives a reason you can tell the customer. The same clauses run in the live PayPal gate |
| **L1 · Gradient boosting** | LightGBM on **232 features** (velocity, mandate, entity, transaction, graph), isotonic-calibrated | The workhorse, with SHAP top-3 reasons on every decision |
| **L2 · Anomaly monitor** | Isolation forest trained on legitimate traffic only | Watches for drift |
| **L3 · Agent-context reader** | Classifies the **text the agent read before it paid** | Catches prompt-injection attacks that look perfectly normal as payments |
| **L4 · Entity graph** | Streaming union-find over users, devices, agents and payees | Exposes mule rings and agent-farm campaigns |

A **learned stacker** fuses the layers, and a **policy engine** maps the score to *Approve / Challenge / Review / Decline*. Thresholds are set as **false-positive budgets**, so a retrain never silently changes how many customers get bothered. **A rule violation always overrides the model.**

**Adversarial loop (evolve).** An evolutionary adversary reads the detector's own explanations, **mutates attacks to evade** (pacing, ring fan-out, device rotation, ticket size, injected pages), and the detector **retrains on everything the arena produces**.

- Mean evasion falls from **0.626 to 0.381** across rounds (**−39%**) and stays down.
- **6 new attack variants** that survived 3+ rounds were automatically written back to the atlas as validated cards (`mantis/atlas/discovered/`).

### The zero-day result (offline, synthetic 1M-payment evaluation)

The question every fraud team asks is *"what about attacks we haven't seen yet?"*

| Scenario | Recall on agentic-mandate family (F1) |
|---|:---:|
| Detector trained **with** F1 data | **0.811** |
| Detector **never** trained on F1 (a true zero-day) | 0.013 💥 |
| Never saw a real F1 event, but trained on **F1 variants generated by MANTIS** | **0.539** 🔁 |

**On a new payment rail you don't have a fraud history, but you can have a red team.** MANTIS shows that attacks which have been described but never observed can be generated, and that training on the generated version carries over to the real one.

### Detection results (offline, synthetic 1M-payment evaluation)

| Layer | AUC-PR | ROC-AUC | Recall @ 0.1% FPR | Campaign recall |
|---|:---:|:---:|:---:|:---:|
| L1 · GBDT | 0.590 | 0.925 | 0.450 | 0.916 |
| L3 · Agent-context reader *(no transaction labels)* | 0.169 | 0.692 | 0.138 | 0.214 |
| **Fused** | **0.588** | **0.912** | **0.483** | **0.908** |

<sub>Baseline precision (fraud prevalence) is ~1.09%. No accuracy figures are reported anywhere, on purpose: at 1% prevalence a model that approves everything is 99% accurate. Recall is always reported at a fixed false-positive rate, the way an issuer budgets a review queue.</sub>

**What we're honest about:** unsupervised anomaly detection (L2) barely works on this data, *because we made the attacks realistic on purpose*. That is exactly why MANTIS relies on rules that need no data plus a loop that generates attacks, rather than a cleverer anomaly score. Full analysis is in [RESULTS.md](RESULTS.md).

### See the offline console

The [**Live console**](https://mantis-ze19.onrender.com/#/research/console) tab under *Offline research* replays 600 pre-scored synthetic payments one at a time: a risk index, which layers fired, the decision, and the **three features that drove it**. Rows turn red or green when the true outcome arrives. The scored event and the ground truth are **separate objects on the wire**, so the UI cannot colour a row before the score arrives. Fraud is over-sampled so something happens while you watch, and the console header states the true rate. *Results*, *Attack atlas* and *Fidelity* hold the rest.

### Offline pipeline

```bash
make install        # editable install + dev tools
make demo           # schema → atlas → corpus → population → dataset → features → L0 → tests
```

<details>
<summary><b>All pipeline stages</b> (click to expand)</summary>

| Command | What it does |
|---|---|
| `make schema` | Prints the frozen `TxEvent` contract (v1.1.0) |
| `make atlas` | 42 vectors, how many have a working injector, and any cards the loop discovered |
| `make corpus` | The committed LLM content corpus, with one example of each kind |
| `make population` | 200k calibrated legitimate payments + calibration figure |
| `make dataset` | Labelled dataset, class balance, per-attack counts, separability table |
| `make probe` / `make derived` | Separability gates on raw columns and on the built feature matrix |
| `make features` | The 232-feature builder, with leakage assertions |
| `make l0` | The 9 deterministic clauses, with per-clause precision and FP rate |
| `make firewall` | All five layers, fusion weights, leave-one-family-out, and writes `RESULTS.md` (~15 min) |
| `make loop` | Evasion curve + zero-day comparison (~30 min) |
| `make ood` | L3 against hand-written injections and benign controls |
| `make feed` | Pre-scores 600 payments for the console (~2 min) |
| `make api` / `make web` | Console backend / React build |
| `make test` | Test suite |

</details>

---

## Repository layout

```
mantis/
├── paypal/     # the authorization gate, PayPal REST client + mock state machine, receipts, MCP proxy
├── agent/      # demo store, shopping agent, mandate compiler, scenarios
├── llm/        # provider-generic chat client with a committed per-turn cache
├── api/        # FastAPI: gate, gate log, receipts verify, playground, guards, console API
├── defense/    # L0 rules · L1 GBDT · L2 novelty · L3 text + LLM judge · L4 graph · fusion · live scorer
├── atlas/      # 42 attack cards (YAML) + loop-discovered variants
├── foundry/    # calibrated population, attack injectors, LLM content, fidelity
├── loop/       # evolutionary adversary + retraining arena
└── core/       # event schema, paths, quotas
web/            # React + Vite: Overview, Orders (AG Studio), order detail, Playground, Verify, offline research
RESULTS.md      # every offline number, generated by the run
```

---

<div align="center">

**PayPal gave agents a wallet. MANTIS gives them a conscience.**

Team: Arya Ambekar, Aaryaman Mishra, Ishan Ambekar

</div>
