"""L2 — unsupervised novelty, fitted on legitimate traffic only.

See :mod:`mantis.defense.l2_novelty.model` for why "only" is load-bearing.
"""

from __future__ import annotations

from mantis.defense.l2_novelty.model import MIN_PRESENT_SHARE, L2Model

__all__ = ["MIN_PRESENT_SHARE", "L2Model"]
