"""Population foundry: the calibrated legitimate payment stream the attacks hide in.

Public surface, in the order you would use it::

    from mantis.foundry.base import load_reference_stats, build_population, simulate_frame

    stats = load_reference_stats()              # fitted JSON if present, else priors
    pop   = build_population(stats, seed=7)     # customers, cards, devices, merchants
    frame = simulate_frame(SimulationConfig(n_events=200_000, seed=7), stats, pop)

Or from the shell::

    python -m mantis.foundry.base --n 200000 --seed 7

The generated events are ``is_fraud=False`` by definition — this module produces
the background only. Injectors (``mantis.foundry.injectors``) are what stamp
``attack_id`` onto a copy of it.

Re-exports are lazy (PEP 562). Two reasons, both practical: importing this
package no longer drags in ``scipy`` and ``matplotlib`` for a caller that only
wanted ``ReferenceStats``, which matters once the API is reporting a latency
budget; and ``python -m mantis.foundry.base.simulator`` stops emitting the
"found in sys.modules after import of package" warning that eager re-exports
cause for every submodule that carries its own ``main()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from mantis.foundry.base.calibration import calibration_report, plot_calibration
    from mantis.foundry.base.entities import Population, build_population
    from mantis.foundry.base.reference import ReferenceStats, load_reference_stats
    from mantis.foundry.base.simulator import (
        DEFAULT_SEED,
        IST,
        SimulationConfig,
        iter_events,
        simulate_frame,
    )

__all__ = [
    "DEFAULT_SEED",
    "IST",
    "Population",
    "ReferenceStats",
    "SimulationConfig",
    "build_population",
    "calibration_report",
    "iter_events",
    "load_reference_stats",
    "plot_calibration",
    "simulate_frame",
]

#: Public name -> the submodule that defines it.
_EXPORTS: Final[dict[str, str]] = {
    "DEFAULT_SEED": "simulator",
    "IST": "simulator",
    "Population": "entities",
    "ReferenceStats": "reference",
    "SimulationConfig": "simulator",
    "build_population": "entities",
    "calibration_report": "calibration",
    "iter_events": "simulator",
    "load_reference_stats": "reference",
    "plot_calibration": "calibration",
    "simulate_frame": "simulator",
}


# PEP 562 hook. The return type is genuinely Any: it varies by name.
def __getattr__(name: str) -> Any:
    """Resolve a public name by importing only the submodule that defines it."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value  # cache, so the import happens at most once
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
