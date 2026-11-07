"""Atlas card schema — the contract every attack card in ``cards/*.yaml`` obeys.

The atlas is **executable**, not documentation. The foundry imports it: an
injector exists because a card names it in ``generator``, and the defense layer
knows a signal is worth building because a card names it in
``observable_signals``. A card with no observable signal is a threat-deck bullet
point, not an entry in this atlas.

The load-bearing field is therefore ``observable_signals``. Each one is a
structured triple — the human-readable signal, the ``feature`` name the defense
layer will actually compute, and the firewall ``layer`` that consumes it. That
triple is the bridge from Pillar 1 (IDENTIFY) to Pillar 3 (DEFEND), and it is
what stops the atlas from drifting away from the detector as the build goes on.

``rails`` is validated against ``mantis.core.events.Channel``, so a card cannot
name a payment rail the schema cannot represent.

Safety posture (CLAUDE.md HARD RULE 5): cards describe attack *classes* at the
level of observable behaviour and mitigation. They are written so a defender can
build a detector, not so an attacker can build a tool. Any vector that cannot be
described that way stays ``status: mapped`` and never receives an injector.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mantis.core.events import Channel

__all__ = [
    "FAMILY_NAMES",
    "AttackCard",
    "DiscoveredBy",
    "Family",
    "Layer",
    "ObservableSignal",
    "Status",
]

_ID_RE: Final[re.Pattern[str]] = re.compile(r"^F[1-6]-\d{2}$")
_GENERATOR_RE: Final[re.Pattern[str]] = re.compile(
    r"^[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)+:[a-z_][a-z0-9_]*$"
)
_FEATURE_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z_][a-z0-9_]*$")


class Family(StrEnum):
    """Attack family. Card ids carry the family prefix; numbering is atlas-global."""

    F1 = "F1"
    F2 = "F2"
    F3 = "F3"
    F4 = "F4"
    F5 = "F5"
    F6 = "F6"


#: Display names for the atlas summary and the console. Keep in sync with the deck.
FAMILY_NAMES: Final[dict[Family, str]] = {
    Family.F1: "Mandate & Delegation Abuse",
    Family.F2: "Synthetic Identity & Onboarding",
    Family.F3: "Social Engineering & Scam Orchestration",
    Family.F4: "Adversarial Evasion & Model Attack",
    Family.F5: "Agent Platform & Tool Integrity",
    Family.F6: "Laundering & Monetisation",
}


class Status(StrEnum):
    """Whether the foundry can actually generate this attack yet.

    ``implemented`` cards contribute to the diversity score honestly; ``mapped``
    cards are the taxonomy we claim coverage of but have not built. Never blur
    the two in a slide.
    """

    IMPLEMENTED = "implemented"
    MAPPED = "mapped"


class Layer(StrEnum):
    """Mandate Firewall layer that consumes a signal."""

    L0 = "L0"  # deterministic mandate/policy rules
    L1 = "L1"  # supervised GBDT on tabular features
    L2 = "L2"  # unsupervised novelty / outlier
    L3 = "L3"  # text & provenance (prompt-injection detection)
    L4 = "L4"  # entity graph


class DiscoveredBy(StrEnum):
    """Provenance of the card itself — part of the novelty story."""

    HUMAN = "human"
    DISCOVERY_AGENT = "discovery_agent"
    ADVERSARIAL_LOOP = "adversarial_loop"


class ObservableSignal(BaseModel):
    """One detectable trace an attack leaves, bound to the feature that catches it.

    ``feature`` is a snake_case name that must exist (eventually) in
    ``mantis/defense/features/``. Keeping the name here means the atlas doubles as
    the feature backlog, and lets the loader report signal coverage per layer.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    signal: str = Field(description="What a defender would observe, in plain language.")
    feature: str = Field(description="snake_case feature name in mantis/defense/features/.")
    layer: Layer = Field(description="Firewall layer that consumes this feature.")

    @field_validator("feature")
    @classmethod
    def _check_feature(cls, v: str) -> str:
        if not _FEATURE_RE.match(v):
            raise ValueError(f"feature must be snake_case, got {v!r}")
        return v


class AttackCard(BaseModel):
    """One vector in the 42-card atlas. One YAML file, stem == ``id``."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = Field(description="Atlas id, e.g. 'F1-01'. Must equal the filename stem.")
    name: str
    family: Family
    status: Status
    rails: list[str] = Field(
        min_length=1, description="Channels this attack rides; validated against Channel."
    )
    actor: str = Field(description="Who runs it, and with what capability.")
    genai_enabler: str = Field(
        description="What GenAI specifically makes newly possible, cheap, or scalable here."
    )
    description: str
    preconditions: list[str] = Field(min_length=1)
    observable_signals: list[ObservableSignal] = Field(
        min_length=1, description="Bridge to Pillar 3. Every card must be detectable."
    )
    mitigations: list[str] = Field(min_length=1)
    generator: str | None = Field(
        default=None, description="Dotted path to the injector, 'pkg.module:function'."
    )
    detected_by: list[Layer] = Field(min_length=1)
    references: list[str] = Field(default_factory=list)
    discovered_by: DiscoveredBy = DiscoveredBy.HUMAN

    @field_validator("id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        if not _ID_RE.match(v):
            raise ValueError(f"id must look like 'F1-01' (F1-F6, two digits), got {v!r}")
        return v

    @field_validator("rails")
    @classmethod
    def _check_rails(cls, v: list[str]) -> list[str]:
        valid = {c.value for c in Channel}
        bad = [r for r in v if r not in valid]
        if bad:
            raise ValueError(f"unknown rails {bad}; valid channels are {sorted(valid)}")
        return v

    @field_validator("generator")
    @classmethod
    def _check_generator(cls, v: str | None) -> str | None:
        if v is not None and not _GENERATOR_RE.match(v):
            raise ValueError(f"generator must be 'pkg.module:function', got {v!r}")
        return v

    @model_validator(mode="after")
    def _check_consistency(self) -> AttackCard:
        if self.id.split("-", 1)[0] != self.family.value:
            raise ValueError(f"id {self.id!r} disagrees with family {self.family.value!r}")
        if self.status is Status.IMPLEMENTED and self.generator is None:
            raise ValueError(f"{self.id}: status='implemented' requires a generator path")
        signal_layers = {s.layer for s in self.observable_signals}
        orphan = signal_layers - set(self.detected_by)
        if orphan:
            raise ValueError(
                f"{self.id}: signals target layer(s) {sorted(o.value for o in orphan)} "
                f"not listed in detected_by"
            )
        return self

    @property
    def filename(self) -> str:
        """Expected card filename. The loader enforces stem == id."""
        return f"{self.id}.yaml"

    @property
    def family_name(self) -> str:
        """Human-readable family label."""
        return FAMILY_NAMES[self.family]
