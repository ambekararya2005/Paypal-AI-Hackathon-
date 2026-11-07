.PHONY: help install lint fmt test atlas schema population figure dataset probe corpus features l0 firewall render loop arena derived drift slices reference fidelity latency ood feed api static web docx submission deploy gate demo clean

PY ?= python

help:
	@echo "MANTIS targets:"
	@echo "  install  install the package plus dev tools (editable)"
	@echo "  lint     ruff check"
	@echo "  fmt      ruff format + autofix imports"
	@echo "  test     pytest"
	@echo "  atlas    print the attack-atlas family summary"
	@echo "  schema   print the frozen event-schema contract"
	@echo "  population  generate data/generated/population.parquet (200k events)"
	@echo "  dataset  generate data/generated/dataset_v1.parquet (background + 8 attacks)"
	@echo "  probe    best-single-feature AUC per injector, on a small background"
	@echo "  corpus   build/inspect the committed LLM content corpus (no network)"
	@echo "  features build the feature matrix and print the leakage assertion firing"
	@echo "  l0       run the deterministic clauses + the Day 3 bucket-contract verdict"
	@echo "  firewall THE Day 4/5 experiment: five layers, fusion, LOFO -> RESULTS.md"
	@echo "  render   re-write RESULTS.md from the cached run, no refit"
	@echo "  loop     THE Day 5 gate: evasion curve + zero-day demo -> arena.json"
	@echo "  derived  the separability probe run over the BUILT feature matrix"
	@echo "  reference   fetch the Kaggle Sparkov panel the scorecard needs (~210 MB)"
	@echo "  fidelity    THE Day 7 scorecard: KS/JS, TSTR, real-vs-synthetic discriminator"
	@echo "  latency     end-to-end per-event scoring latency, p50/p95/p99"
	@echo "  drift    every marginal vs the reference, against a sampling-noise band"
	@echo "  slices   audit every probe_slice and print its denominator"
	@echo "  figure   redraw the calibration figure from the existing parquet"
	@echo "  ood      L3 against hand-authored text it did not come from"
	@echo "  feed     pre-score a window of authorisations for the live console"
	@echo "  api      serve the console backend on :8000"
	@echo "  static   freeze the API to JSON so the console needs no backend"
	@echo "  web      build the React console (needs npm install in web/ first)"
	@echo "  docx     generate the submission document FROM RESULTS.md"
	@echo "  submission  ood + feed + docx + web: everything a judge is handed"
	@echo "  gate     the daily acceptance gate"
	@echo "  demo     THE acceptance test - must pass from a clean clone"

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	$(PY) -m ruff check .

fmt:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

test:
	$(PY) -m pytest

atlas:
	$(PY) -m mantis.atlas.loader

schema:
	$(PY) -m mantis.core.events

# Day 1: the calibrated legitimate population. No network, no Kaggle token.
population:
	$(PY) -m mantis.foundry.base --n 200000 --seed 7

figure:
	$(PY) -m mantis.foundry.base.calibration

# Day 2: the labelled dataset. Prints class balance, per-attack counts and the
# best-single-feature AUC table, then writes the parquet and a manifest.
dataset:
	$(PY) -m mantis.foundry --attacks all --out data/generated/dataset_v1.parquet --show-content

# The subtlety gate on its own, against a small background. Fast enough to run
# after touching an injector.
probe:
	$(PY) -m mantis.foundry.injectors.probe

# Day 3: the text the agents read. Cache-first, so this runs identically with no
# Ollama, no GPU and no network -- see CLAUDE.md HARD RULE 3. Add --live --refresh
# to re-author it against a local model.
corpus:
	$(PY) -m mantis.foundry.llm --show

# Day 4: the Mandate Firewall.

# The feature layer, with the three-tier leakage assertion fired deliberately so
# a reader can watch it reject a label.
features:
	$(PY) -m mantis.defense.features

# L0's per-clause precision and FP rate, then the Day 3 HARD/CLEAN contract
# checked against a real L0 for the first time.
l0:
	$(PY) -m mantis.defense.l0_rules

# THE Day 4 deliverable. Pools five seeds, fits L1 and L2, runs
# leave-one-family-out, and writes RESULTS.md. ~10 minutes from a cold pool.
firewall:
	$(PY) -m mantis.defense

# Re-render RESULTS.md from the cached experiment, without refitting anything.
# For editing the document's prose without a fifteen-minute round trip.
render:
	$(PY) -m mantis.defense --render-only

# Day 5: the closed loop. Writes data/generated/arena.json -- the evasion curve
# and the zero-day comparison -- and any surviving variant back into
# mantis/atlas/discovered/. Re-run `make firewall` afterwards to fold its numbers
# into RESULTS.md. ~35 minutes; add `--cards all` for the whole atlas and hours.
loop:
	$(PY) -m mantis.loop --generations 5 --population 5 --seeds 2 --family F1

# A short arena with no zero-day run, for when you have touched a gene.
arena:
	$(PY) -m mantis.loop --generations 3 --population 4 --no-zero-day --no-writeback

# Day 5 QA: the single-feature probe cannot see a feature that is two columns and
# a regression, which is how mnd_deliberation_residual_z reached 0.99 on F1-01
# unnoticed. This runs the same gate over the matrix L1 actually trains on.
derived:
	$(PY) scripts/probe_derived.py

# Day 7: the reference panel. Downloads once, ~210 MB, never committed. The
# scorecard degrades cleanly without it, which is the state of a clean clone.
reference:
	$(PY) scripts/fetch_reference.py

# Day 7: THE fidelity scorecard - criterion 2's artefact. Marginal KS/JS against
# sampling-noise bands, correlation-matrix distance, TSTR, and a real-vs-synthetic
# discriminator whose target is 0.5. Writes data/generated/fidelity.json.
# Exits 1 (not 0) when the reference panel is absent, so a missing input is loud.
fidelity:
	$(PY) -m mantis.foundry.fidelity

# Day 7: end-to-end scoring latency through the full fused stack, against the
# 50 ms budget. Reports p50/p95/p99 as measured, not as targeted.
latency:
	$(PY) scripts/latency_bench.py

# Day 6 carry-over: does L3 fire on an injection somebody else wrote? Ships
# hand-authored payloads AND benign controls in the same registers, because the
# recall number is uninterpretable without them. See the script's docstring.
ood:
	$(PY) scripts/l3_ood.py

# Day 6: the live console's data. Fits the firewall once, offline, and writes
# 600 pre-scored authorisations. The API then does NO model work per request.
feed:
	$(PY) scripts/build_console_feed.py

# Day 6: the console backend. Serves RESULTS.md and arena.json straight from
# disk plus the SSE authorisation stream. Nothing is recomputed per request.
api:
	$(PY) -m mantis.api

# Day 6: freeze every API response to JSON so the console can be hosted with no
# backend at all. Produced by calling the real route handlers, never by a second
# implementation of the response shapes.
static:
	$(PY) scripts/build_static_site.py

# Day 6: the React console. `cd web && npm install` once, first.
# Depends on `static` so the built bundle always carries a working offline mode.
web: static
	cd web && npm run build

# Day 6: THE submission document. Every figure is read out of RESULTS.md, so
# this regenerates rather than being retyped. CLAUDE.md section 7: never cut.
docx:
	$(PY) scripts/build_docx.py

# The three artifacts a judge receives, rebuilt in dependency order.
# `fidelity` is deliberately NOT a prerequisite: it needs the 210 MB Kaggle
# panel, it exits 1 without it, and data/generated/fidelity.json is committed -
# so a clean clone assembles the submission without a download. Re-run
# `make reference && make fidelity` when the scorecard itself changes.
submission: ood feed docx web
	@echo ""
	@echo "Submission artifacts:"
	@echo "  1. repo       this working tree"
	@echo "  2. document   docs/MANTIS_submission.docx"
	@echo "  3. prototype  web/dist  - a SELF-CONTAINED static site."
	@echo "                Upload that folder to any static host; it needs no"
	@echo "                backend. Run 'make api' alongside it for the live"
	@echo "                SSE stream instead of the offline replay."

# Day 4 carry-overs: distribution drift, and the probe-slice audit.
drift:
	$(PY) scripts/drift_check.py

slices:
	$(PY) scripts/audit_probe_slices.py

# Gate: the schema imports, the atlas validates and agrees with the injector
# registry, the population generates, the labelled dataset generates.
gate:
	$(PY) -c "from mantis.core.events import SCHEMA_VERSION; print(f'schema v{SCHEMA_VERSION} OK')"
	$(PY) -m mantis.atlas.loader
	$(PY) -c "from mantis.foundry.injectors import REGISTRY; print(f'{len(REGISTRY)} injectors, atlas agrees')"
	$(PY) -m mantis.foundry.llm
	$(PY) -m mantis.foundry.base --n 200000 --seed 7
	$(PY) -m mantis.foundry --attacks all --out data/generated/dataset_v1.parquet --show-content
	$(PY) -m mantis.defense.l0_rules
	$(PY) -m pytest

# The acceptance test. Grows a stage per day; must always run end to end with
# no network, no GPU, no API key, no Kaggle token. See CLAUDE.md HARD RULE 4.
demo: schema atlas corpus population dataset features l0 test
	@echo ""
	@echo "MANTIS demo complete."
	@echo "  Day 0-1 stages: schema contract, 42-card atlas, legitimate population,"
	@echo "                  calibration figure, tests."
	@echo "  Day 2 stages:   injector framework, 8 labelled attacks, class balance,"
	@echo "                  per-attack counts, best-single-feature AUC table."
	@echo "  Day 3 stages:   schema v1.1.0 lifecycle block, LLM content corpus,"
	@echo "                  7 agentic F1 injectors split HARD/CLEAN, 15 cards."
	@echo "  Day 4 stages:   232-feature builder with a three-tier leakage assertion,"
	@echo "                  L0 clauses + bucket-contract verdict."
	@echo "  Day 5 stages:   L4 entity graph (28 streamed features), L3 page classifier"
	@echo "                  with two hold-out protocols, weighted fusion, decision"
	@echo "                  policy, campaign-level recall, the FPR curve."
	@echo "                  Run 'make firewall' for the tables (~15 min) and"
	@echo "                  'make loop' for the evasion curve and the zero-day"
	@echo "                  comparison (~30 min)."
	@echo "  Pending stages: fidelity scorecard, live console."

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find data/generated -type f ! -name ".gitkeep" -delete
