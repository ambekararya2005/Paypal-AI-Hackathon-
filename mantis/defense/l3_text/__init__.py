"""L3 — the text layer: what the agent read before it decided to pay.

    from mantis.defense.l3_text import L3Model
    l3 = L3Model().fit()             # no y: the label is a property of the text
    scores = l3.score(events)        # NaN on rows carrying no provenance chain

See :mod:`mantis.defense.l3_text.model` for the two hold-out protocols — an
unseen phrasing and an unseen adversarial *kind* — which are what separate
"detects injection" from "remembers our corpus".
"""

from __future__ import annotations

from mantis.defense.l3_text.model import (
    HELD_OUT_KIND,
    HELD_OUT_VARIANTS,
    L3Model,
    chains_for,
    held_out_artifacts,
)

__all__ = [
    "HELD_OUT_KIND",
    "HELD_OUT_VARIANTS",
    "L3Model",
    "chains_for",
    "held_out_artifacts",
]
