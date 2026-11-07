"""Pydantic response models for the MANTIS API.

Typed rather than raw dicts for the ordinary reason — the console is TypeScript
and an OpenAPI schema is worth more than a README — and for one project-specific
one: **these models are where the label discipline reaches the wire.**

CLAUDE.md HARD RULE 1 keeps ``is_fraud``, ``attack_id`` and ``attack_campaign``
out of any feature matrix. They are equally dangerous on an SSE frame, for a
different reason: an authorisation arriving at the console carrying its own
ground truth would let the console draw a red border before the score came back,
and a judge would be watching a lookup dressed as a detector. So a scored event
splits into two objects — :class:`ScoredEvent`, which is everything the firewall
knew when it decided, and :class:`Truth`, which is the answer. The stream sends
the first, then the second, and the console reveals them in that order.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = [
    "ArenaResponse",
    "AtlasCardDetail",
    "AtlasCardSummary",
    "AtlasResponse",
    "Contribution",
    "FidelityResponse",
    "HealthResponse",
    "LayerScore",
    "ResultsResponse",
    "ResultsTable",
    "ScoredEvent",
    "SimulateRequest",
    "SimulateResponse",
    "StreamFrame",
    "Truth",
]

Decision = Literal["approve", "challenge", "review", "decline"]


# --------------------------------------------------------------------- atlas --
class ObservableSignalOut(BaseModel):
    signal: str
    feature: str
    layer: str


class AtlasCardSummary(BaseModel):
    """One card, as the atlas grid shows it."""

    id: str
    name: str
    family: str
    status: str
    rails: list[str]
    detected_by: list[str]
    has_injector: bool = Field(
        description="True iff a registered injector generates this card. The atlas "
        "ratchet: status 'implemented' and this flag cannot disagree."
    )
    discovered_by: str


class AtlasCardDetail(AtlasCardSummary):
    """The whole card, for the drill-down."""

    actor: str
    genai_enabler: str
    description: str
    preconditions: list[str]
    observable_signals: list[ObservableSignalOut]
    mitigations: list[str]
    generator: str | None
    references: list[str]


class AtlasFamilyCount(BaseModel):
    family: str
    total: int
    implemented: int


class AtlasResponse(BaseModel):
    cards: list[AtlasCardSummary]
    families: list[AtlasFamilyCount]
    total: int
    implemented: int = Field(
        description="Cards with a working injector. The honest count, not the mapped count."
    )
    discovered: list[AtlasCardSummary] = Field(
        default_factory=list,
        description="Variants written back by the adversarial loop. Beside the atlas, "
        "never inside it.",
    )


# ------------------------------------------------------------------- results --
class ResultsTable(BaseModel):
    """One table lifted out of RESULTS.md, unmodified."""

    title: str
    header: list[str]
    rows: list[list[str]]


class ZeroDayRow(BaseModel):
    detector: str
    recall: float


class ResultsResponse(BaseModel):
    """The firewall's published numbers, read off RESULTS.md at startup.

    Nothing here is computed at request time. If a number is not in RESULTS.md it
    is not served, which is the property that keeps the console from ever showing
    a figure the repo cannot reproduce.
    """

    generated: str = Field(description="The dateline RESULTS.md carries.")
    tables: list[ResultsTable]
    layer_performance: list[dict[str, str]]
    per_family: list[dict[str, str]]
    per_attack: list[dict[str, str]]
    decisions: list[dict[str, str]]
    zero_day: list[ZeroDayRow]
    evasion_curve: list[float]
    headline: dict[str, Any] = Field(
        description="The few numbers a slide quotes, pre-extracted so the console "
        "does not have to parse a table to render a big number."
    )


# --------------------------------------------------------------------- arena --
class ArenaResponse(BaseModel):
    """``data/generated/arena.json``, served straight from disk."""

    operating_fpr: float
    cards: list[str]
    seconds: float
    n_background: int
    evasion_curve: list[float]
    generations: list[dict[str, Any]]
    survivors: list[dict[str, Any]]
    zero_day: dict[str, Any] | None


# ------------------------------------------------------------------ fidelity --
class FidelityResponse(BaseModel):
    """The fidelity scorecard - criterion 2's artefact, served whole.

    Written by ``python -m mantis.foundry.fidelity`` and read off disk here.
    ``available: false`` when it has never been generated, because a console that
    renders "not measured yet" is telling the truth and one that 404s looks
    broken.

    The sections are passed through rather than reshaped. Every one of them is
    already the shape a reader needs - the scorecard's CLI and this endpoint
    print the same dict - and adding a second schema here would be one more place
    for the numbers to drift from what the code produced.
    """

    available: bool
    note: str
    generated: str = ""
    schema_version: str = ""

    #: Which calibration path the population used: fitted, or committed priors.
    #: First, because every number below is conditional on it.
    calibration: dict[str, Any] = Field(default_factory=dict)
    #: The external panel, its provenance line, and the level ratios that are
    #: reported *without* a distance attached. Absent when no panel was present.
    reference: dict[str, Any] = Field(default_factory=dict)
    synthetic: dict[str, Any] = Field(default_factory=dict)

    headline: dict[str, Any] = Field(default_factory=dict)
    marginals: dict[str, Any] = Field(default_factory=dict)
    tstr: dict[str, Any] = Field(default_factory=dict)
    discriminator: dict[str, Any] = Field(default_factory=dict)
    #: The same discriminator with the adjudicated axes removed. Served beside
    #: the full one, never instead of it: the ablation is a judgement, and a
    #: reader has to be able to see both numbers to reject it.
    discriminator_ablated: dict[str, Any] = Field(default_factory=dict)
    adjudications: list[dict[str, Any]] = Field(default_factory=list)
    known_divergences: list[dict[str, Any]] = Field(default_factory=list)

    #: Day 1 population calibration, from the manifest. Retained for the
    #: no-scorecard case, where it is the only measured thing available.
    population: dict[str, Any] = Field(default_factory=dict)


class LatencyResponse(BaseModel):
    """Per-event scoring latency, measured one event at a time against warm state.

    Criterion 5's other number. ``available: false`` until
    ``scripts/latency_bench.py`` has been run.
    """

    available: bool
    note: str = ""
    generated: str = ""
    n_events: int = 0
    warm_events: int = 0
    budget_ms: float = 0.0
    within_budget: bool = False
    headroom: float = 0.0
    end_to_end_ms: dict[str, float] = Field(default_factory=dict)
    stages_ms: dict[str, dict[str, float]] = Field(default_factory=dict)


# ------------------------------------------------------------------ stream ----
class LayerScore(BaseModel):
    score: float | None
    percentile: float | None = Field(
        description="Rank against legitimate traffic on this layer, in [0,1]. The "
        "layers are on different scales — an isotonic-calibrated probability and a "
        "raw isolation-forest score — and this is what makes their bars comparable."
    )


class Contribution(BaseModel):
    feature: str
    value: Any = Field(description="Rendered as itself: categoricals stay strings, "
                       "and a missing key is null rather than NaN.")
    contribution: float = Field(description="Log-odds of L1's raw margin.")


class L0Verdict(BaseModel):
    fired: bool
    reason: str | None


class Truth(BaseModel):
    """Ground truth. Sent **after** the decision, never with it."""

    is_fraud: bool
    attack_id: str | None
    attack_campaign: str | None


class ScoredEvent(BaseModel):
    """One authorisation, and what the firewall made of it.

    Deliberately carries no ground-truth field. See the module docstring.
    """

    event_id: str
    ts: str
    amount: float
    currency: str
    mcc: str | None
    channel: str
    entry_mode: str | None
    customer_id: str | None
    merchant_id: str | None
    merchant_country: str | None
    card_bin: str | None
    device_id: str | None
    txn_type: str | None
    ag_agent_id: str | None
    ag_agent_platform: str | None
    ag_mandate_type: str | None
    ag_human_present: bool | None
    ag_delegation_depth: float | None
    layers: dict[str, LayerScore]
    l0: L0Verdict
    risk: float | None = Field(description="0-1. The fused score's rank against legitimate "
                               "traffic, which is what the console's dial shows.")
    decision: Decision
    contributions: list[Contribution]


class StreamFrame(BaseModel):
    """One SSE frame. Documented here; emitted as JSON by the stream endpoint."""

    seq: int
    event: ScoredEvent
    truth: Truth


# ------------------------------------------------------------------ simulate --
class SimulateRequest(BaseModel):
    n_events: int = Field(default=200, ge=1, le=600)
    #: Events per second. The console's pace control; the server honours it so the
    #: stream cannot be sped past what a judge can read.
    rate: float = Field(default=6.0, gt=0, le=60)
    #: Restrict the replay to one attack family, or None for the mixed stream.
    family: str | None = None
    #: Start the replay somewhere other than the beginning, so a second demo in
    #: the same session does not show the same first ten events.
    offset: int = Field(default=0, ge=0)


class SimulateResponse(BaseModel):
    run_id: str
    n_events: int
    rate: float
    stream_url: str
    note: str


# -------------------------------------------------------------------- health --
class HealthResponse(BaseModel):
    status: str
    schema_version: str
    atlas_cards: int
    atlas_implemented: int
    feed_events: int
    results_available: bool
    arena_available: bool
    fidelity_available: bool
