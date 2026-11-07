"""The MANTIS API — the live defence console's backend.

    python -m mantis.api            # http://127.0.0.1:8000, docs at /docs

Endpoints
-----------
========================  =====================================================
``GET  /health``          what is loaded and what is missing
``GET  /atlas``           42 cards, family counts, the honest implemented count
``GET  /atlas/{id}``      one card in full, including its observable signals
``GET  /results``         RESULTS.md, parsed — every table, plus the headline
``GET  /arena``           ``arena.json``: evasion curve, survivors, zero-day
``GET  /fidelity``        the Day 7 scorecard, or ``available: false``
``POST /simulate``        register a replay, returns a ``run_id``
``GET  /stream/{run_id}`` SSE: scored authorisations, one frame at a time
========================  =====================================================

The one rule this service has
-------------------------------
**No model runs inside a request.** Every score, decision and attribution was
computed by ``scripts/build_console_feed.py`` and committed; every metric was
computed by ``python -m mantis.defense`` and written into RESULTS.md. A route
handler here reads a dict and returns it. That is why the console is instant on a
judge's laptop, and it is why ``mantis/api/`` sits at the end of the dependency
order and imports the rest of the project read-only.

Why the ground truth is a separate object on the wire
-------------------------------------------------------
Each SSE frame carries ``event`` (what the firewall knew) and ``truth`` (whether
it was actually fraud) as sibling objects. The console renders the decision, then
reveals the outcome. Shipping ``is_fraud`` inside the event would let the UI
colour a row before the score arrived, and the demo would be a lookup wearing a
detector's clothes. Same discipline as HARD RULE 1, one layer further out.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Final

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from mantis.api.models import (
    ArenaResponse,
    AtlasCardDetail,
    AtlasCardSummary,
    AtlasFamilyCount,
    AtlasResponse,
    FidelityResponse,
    HealthResponse,
    LatencyResponse,
    ObservableSignalOut,
    ResultsResponse,
    ResultsTable,
    SimulateRequest,
    SimulateResponse,
    ZeroDayRow,
)
from mantis.api.store import STORE
from mantis.atlas.loader import ATLAS, DISCOVERED
from mantis.core.events import SCHEMA_VERSION

#: Vite's dev server, plus the two hosts a deployed console is served from. A
#: regex rather than a list for the deployed case because preview deployments get
#: a fresh subdomain per push and would otherwise each need a redeploy of the API.
CORS_ORIGINS: Final[list[str]] = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]
CORS_ORIGIN_REGEX: Final[str] = r"https://.*\.(vercel\.app|netlify\.app|hf\.space)"

#: How long a registered run stays claimable before it is swept. A run is a few
#: hundred bytes, but a public deployment with an unbounded dict is a memory leak
#: with a URL.
RUN_TTL_SECONDS: Final[float] = 3600.0

#: Frames a single stream will emit before closing, whatever the client asked
#: for. The feed is finite; this makes the ceiling explicit rather than letting a
#: request for a million events sit in a loop.
MAX_FRAMES: Final[int] = 600


@dataclass(slots=True)
class Run:
    """A registered replay. Holds indices into the feed, never event copies."""

    run_id: str
    indices: list[int]
    rate: float
    created: float = field(default_factory=time.time)


#: Registered runs, in memory. Deliberately not persisted: a run is a view over a
#: committed file, so losing one on restart costs a judge one button press.
RUNS: Final[dict[str, Run]] = {}


def _sweep() -> None:
    cutoff = time.time() - RUN_TTL_SECONDS
    for run_id in [r for r, run in RUNS.items() if run.created < cutoff]:
        RUNS.pop(run_id, None)


app = FastAPI(
    title="MANTIS",
    version=SCHEMA_VERSION,
    description=(
        "Adversarial fraud-data foundry and Mandate Firewall for agentic commerce. "
        "Every figure this API returns was computed offline and read from disk; no "
        "model runs inside a request."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------- health --
@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        schema_version=SCHEMA_VERSION,
        atlas_cards=len(ATLAS),
        atlas_implemented=int(STORE.headline.get("atlas_implemented", 0)),
        feed_events=len(STORE.events),
        results_available=STORE.results is not None,
        arena_available=STORE.arena is not None,
        fidelity_available=STORE.fidelity is not None,
    )


# -------------------------------------------------------------------- atlas --
def _summary(card: Any) -> AtlasCardSummary:
    return AtlasCardSummary(
        id=card.id,
        name=card.name,
        family=str(card.family),
        status=str(card.status),
        rails=list(card.rails),
        detected_by=[str(layer) for layer in card.detected_by],
        has_injector=card.generator is not None,
        discovered_by=str(card.discovered_by),
    )


@app.get("/atlas", response_model=AtlasResponse, tags=["atlas"])
def atlas(family: str | None = None, implemented_only: bool = False) -> AtlasResponse:
    """The executable atlas.

    ``implemented`` counts cards a registered injector actually generates, which
    is the ratchet CLAUDE.md §8 describes — not the number of YAML files.
    """
    cards = list(ATLAS.values())
    if family:
        cards = [c for c in cards if str(c.family).upper() == family.upper()]
    if implemented_only:
        cards = [c for c in cards if c.generator]

    families: dict[str, list[Any]] = {}
    for card in ATLAS.values():
        families.setdefault(str(card.family), []).append(card)

    return AtlasResponse(
        cards=[_summary(c) for c in sorted(cards, key=lambda c: c.id)],
        families=[
            AtlasFamilyCount(
                family=name,
                total=len(group),
                implemented=sum(1 for c in group if c.generator),
            )
            for name, group in sorted(families.items())
        ],
        total=len(ATLAS),
        implemented=sum(1 for c in ATLAS.values() if c.generator),
        discovered=[_summary(c) for c in sorted(DISCOVERED.values(), key=lambda c: c.id)],
    )


@app.get("/atlas/{card_id}", response_model=AtlasCardDetail, tags=["atlas"])
def atlas_card(card_id: str) -> AtlasCardDetail:
    """One card in full. Looks in the frozen 42 first, then in the loop's finds."""
    key = card_id.upper()
    card = ATLAS.get(key) or DISCOVERED.get(key)
    if card is None:
        raise HTTPException(404, f"no atlas card {card_id!r}")
    return AtlasCardDetail(
        **_summary(card).model_dump(),
        actor=card.actor,
        genai_enabler=card.genai_enabler,
        description=card.description,
        preconditions=list(card.preconditions),
        observable_signals=[
            ObservableSignalOut(signal=s.signal, feature=s.feature, layer=str(s.layer))
            for s in card.observable_signals
        ],
        mitigations=list(card.mitigations),
        generator=card.generator,
        references=list(card.references),
    )


# ------------------------------------------------------------------ results --
@app.get("/results", response_model=ResultsResponse, tags=["results"])
def results() -> ResultsResponse:
    """RESULTS.md, parsed. 503 when it has never been generated."""
    if STORE.results is None:
        raise HTTPException(
            503,
            f"RESULTS.md is not available: {STORE.results_error}. "
            "Run 'make firewall' to generate it.",
        )
    doc = STORE.results
    tables = [
        ResultsTable(title=section.title, header=table.header, rows=table.rows)
        for section in doc.sections
        for table in section.tables
    ]
    zero = doc.table_for("zero-day demonstration")
    zero_rows: list[ZeroDayRow] = []
    if zero:
        for row in zero.rows:
            if len(row) >= 2:
                try:
                    zero_rows.append(ZeroDayRow(detector=row[0], recall=float(row[1])))
                except ValueError:
                    continue

    dateline = next(
        (line for line in doc.sections[0].prose if line.startswith("*Day")),
        "",
    ).strip("*")

    return ResultsResponse(
        generated=dateline,
        tables=tables,
        layer_performance=STORE.table_records("Layer performance"),
        per_family=STORE.table_records("Leave one family out"),
        per_attack=STORE.table_records("Per attack card"),
        decisions=STORE.table_records("The decision layer"),
        zero_day=zero_rows,
        evasion_curve=STORE.evasion_curve(),
        headline=STORE.headline,
    )


@app.get("/arena", response_model=ArenaResponse, tags=["results"])
def arena() -> ArenaResponse:
    """``data/generated/arena.json``, straight from disk."""
    if STORE.arena is None:
        raise HTTPException(503, "arena.json is not available. Run 'make loop' to generate it.")
    payload = STORE.arena
    return ArenaResponse(
        operating_fpr=float(payload.get("operating_fpr", 0.001)),
        cards=list(payload.get("cards", [])),
        seconds=float(payload.get("seconds", 0.0)),
        n_background=int(payload.get("n_background", 0)),
        evasion_curve=[float(v) for v in payload.get("evasion_curve", [])],
        generations=list(payload.get("generations", [])),
        survivors=list(payload.get("survivors", [])),
        zero_day=payload.get("zero_day"),
    )


@app.get("/fidelity", response_model=FidelityResponse, tags=["results"])
def fidelity() -> FidelityResponse:
    """The fidelity scorecard, straight from ``data/generated/fidelity.json``.

    Reports ``available: false`` rather than 404-ing when the scorecard has never
    been generated, so the console renders an honest "not measured yet" panel
    instead of an error. The Day 1 population manifest is returned in that case,
    because it is measured and real — it is simply not the scorecard.

    Note the two states inside ``available: true``. A scorecard produced without
    the reference panel carries ``reference.available == false`` and has no
    marginal, TSTR or discriminator section, and the console must render that as
    "not measured" rather than as zero. See
    :mod:`mantis.foundry.fidelity.scorecard`.
    """
    card = STORE.fidelity
    if card is None:
        return FidelityResponse(
            available=False,
            note=(
                "The fidelity scorecard has not been generated. Run "
                "'make reference && make fidelity'. The population calibration below "
                "comes from the Day 1 manifest; it is measured, but it is not the "
                "scorecard."
            ),
            population=dict(STORE.population or {}),
        )

    has_panel = bool(card.get("reference", {}).get("available"))
    return FidelityResponse(
        available=True,
        note=(
            "Measured against the Sparkov reference panel on dimensionless shape "
            "features."
            if has_panel
            else "Generated without a reference panel: sections 2-4 were skipped."
        ),
        generated=str(card.get("generated", "")),
        schema_version=str(card.get("schema_version", "")),
        calibration=dict(card.get("calibration", {})),
        reference=dict(card.get("reference", {})),
        synthetic=dict(card.get("synthetic", {})),
        headline=dict(card.get("headline", {})),
        marginals=dict(card.get("marginals", {})),
        tstr=dict(card.get("tstr", {})),
        discriminator=dict(card.get("discriminator", {})),
        discriminator_ablated=dict(card.get("discriminator_ablated", {})),
        adjudications=list(card.get("adjudications", [])),
        known_divergences=list(card.get("known_divergences", [])),
        population=dict(STORE.population or {}),
    )


@app.get("/latency", response_model=LatencyResponse, tags=["results"])
def latency() -> LatencyResponse:
    """Per-event scoring latency against the 50 ms authorisation budget.

    The other half of criterion 5, and the one ``RESULTS.md`` refused to quote
    before it was measured. Written by ``scripts/latency_bench.py``.
    """
    payload = STORE.latency
    if payload is None:
        return LatencyResponse(
            available=False,
            note=(
                "Latency has not been measured. Run 'make latency'. No number is "
                "substituted for it."
            ),
        )
    return LatencyResponse(
        available=True,
        note=str(payload.get("note", "")),
        generated=str(payload.get("generated", "")),
        n_events=int(payload.get("n_events", 0)),
        warm_events=int(payload.get("warm_events", 0)),
        budget_ms=float(payload.get("budget_ms", 0.0)),
        within_budget=bool(payload.get("within_budget", False)),
        headroom=float(payload.get("headroom", 0.0)),
        end_to_end_ms=dict(payload.get("end_to_end_ms", {})),
        stages_ms=dict(payload.get("stages_ms", {})),
    )


@app.get("/figure/{name}", tags=["results"])
def figure(name: str) -> FileResponse:
    """A committed figure from ``docs/`` — the calibration plot, mainly."""
    path = STORE.docs_figure(name)
    if path is None:
        raise HTTPException(404, f"no figure {name!r}")
    return FileResponse(path)


# ----------------------------------------------------------------- simulate --
@app.post("/simulate", response_model=SimulateResponse, tags=["stream"])
def simulate(request: SimulateRequest) -> SimulateResponse:
    """Register a replay and hand back a ``run_id`` to stream.

    Nothing is generated here. The run is a list of indices into the committed
    feed, so registering is a slice and the POST returns in microseconds.
    """
    _sweep()
    events = STORE.events
    if not events:
        raise HTTPException(
            503,
            "console_feed.json is not available. Run "
            "'python scripts/build_console_feed.py' to generate it.",
        )

    indices = list(range(len(events)))
    if request.family:
        wanted = request.family.upper()
        indices = [
            i
            for i in indices
            if (events[i]["truth"].get("attack_id") or "").upper().startswith(wanted)
            or not events[i]["truth"]["is_fraud"]
        ]
    if request.offset:
        indices = indices[request.offset :] + indices[: request.offset]
    indices = indices[: min(request.n_events, MAX_FRAMES)]

    run = Run(run_id=uuid.uuid4().hex[:12], indices=indices, rate=request.rate)
    RUNS[run.run_id] = run
    manifest = STORE.feed_manifest
    return SimulateResponse(
        run_id=run.run_id,
        n_events=len(indices),
        rate=run.rate,
        stream_url=f"/stream/{run.run_id}",
        note=manifest.get("sampling_note", ""),
    )


@app.get("/stream/{run_id}", tags=["stream"])
async def stream(run_id: str, request: Request) -> EventSourceResponse:
    """Server-sent events: one scored authorisation per frame.

    Frame types, so the console can drive an animation rather than a table:

    ``meta``   once, first — the feed's provenance and how many frames follow.
    ``auth``   the scored event and its ground truth, ``seq`` counting from 0.
    ``done``   once, last — the tallies, so the console need not accumulate them.

    The generator checks ``request.is_disconnected()`` every frame. Without it a
    judge closing the tab leaves a coroutine sleeping its way through six hundred
    events, and four demos in a row leaves four.
    """
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, f"unknown run_id {run_id!r} (expired or never registered)")

    events = STORE.events
    delay = 1.0 / run.rate

    async def generator():
        manifest = STORE.feed_manifest
        yield {
            "event": "meta",
            "data": json.dumps(
                {
                    "run_id": run.run_id,
                    "n_events": len(run.indices),
                    "rate": run.rate,
                    "operating_fpr": manifest.get("operating_fpr"),
                    "sampling_note": manifest.get("sampling_note", ""),
                    "provenance_note": manifest.get("provenance_note", ""),
                    "thresholds": manifest.get("thresholds", {}),
                }
            ),
        }

        tally: dict[str, int] = {}
        caught = missed = false_positives = 0
        for seq, index in enumerate(run.indices):
            if await request.is_disconnected():
                return
            record = events[index]
            decision = record["decision"]
            tally[decision] = tally.get(decision, 0) + 1
            flagged = decision != "approve"
            if record["truth"]["is_fraud"]:
                caught += flagged
                missed += not flagged
            else:
                false_positives += flagged

            payload = {k: v for k, v in record.items() if k != "truth"}
            yield {
                "event": "auth",
                "data": json.dumps(
                    {"seq": seq, "event": payload, "truth": record["truth"]}
                ),
            }
            await asyncio.sleep(delay)

        yield {
            "event": "done",
            "data": json.dumps(
                {
                    "n_events": len(run.indices),
                    "decisions": tally,
                    "caught": caught,
                    "missed": missed,
                    "false_positives": false_positives,
                }
            ),
        }

    return EventSourceResponse(generator())
