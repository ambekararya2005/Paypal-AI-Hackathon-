"""Per-event attribution, from LightGBM's own contributions rather than SHAP.

    from mantis.defense.explain import top_contributions
    reasons = top_contributions(l1, X_test.iloc[[0]])

See :mod:`mantis.defense.explain.contributions` for why the native call is the
same computation SHAP would have made, and for the one thing to be careful about
(contributions live in the raw margin space, not the calibrated one).
"""

from __future__ import annotations

from mantis.defense.explain.contributions import (
    TOP_K,
    Attribution,
    explain_events,
    top_contributions,
)

__all__ = ["TOP_K", "Attribution", "explain_events", "top_contributions"]
