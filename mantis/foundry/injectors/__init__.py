"""Attack injectors — the atlas, made executable.

Importing this package imports every injector module and then runs
:func:`~mantis.foundry.injectors.base.validate_registry`, which **fails the
import** unless the atlas and the code agree in both directions:

* no card may claim ``status: implemented`` without a registered injector, and
* no injector may exist for a card the atlas does not have, or still calls
  ``mapped``, or whose ``generator`` path does not resolve to that injector's own
  module.

That assertion is the seam between Pillar 1 and Pillar 2. Without it the atlas
is documentation that drifts; with it, the number of implemented cards on the
slide is the number of injectors that ran.

Adding an injector is therefore a two-file change, in this order:

1. Write ``fN_MM_slug.py`` with a ``BaseAttack`` subclass decorated with
   ``@register`` and a module-level ``inject = card_entry_point(TheClass)``.
2. Promote the card: set ``status: implemented`` and
   ``generator: mantis.foundry.injectors.fN_MM_slug:inject``.
3. Add the module to ``_MODULES`` below.

Do them the other way round and the import tells you immediately.
"""

from __future__ import annotations

from importlib import import_module
from typing import Final

from mantis.foundry.injectors.base import (
    REGISTRY,
    BaseAttack,
    InjectorError,
    PopulationView,
    campaign_id,
    get_injector,
    register,
    run_injector,
    validate_attack_frame,
    validate_registry,
)

__all__ = [
    "REGISTRY",
    "BaseAttack",
    "InjectorError",
    "PopulationView",
    "campaign_id",
    "get_injector",
    "register",
    "run_injector",
    "validate_attack_frame",
    "validate_registry",
]

#: Every injector module, imported at package import so the registry is complete
#: before it is validated. Explicit rather than a directory scan: a scan would
#: silently skip a module with an import error, which is precisely the failure
#: this package exists to make loud.
_MODULES: Final[tuple[str, ...]] = (
    "f1_01_cart_tampering",
    "f1_02_scope_inflation",
    "f1_03_refund_hijack",
    "f1_04_category_drift",
    "f1_05_delegation_laundering",
    "f1_09_presence_spoof",
    "f1_10_mandate_replay",
    "f2_13_synthetic_identity",
    "f2_16_bust_out",
    "f3_19_digital_arrest",
    "f4_27_adaptive_bin",
    "f4_28_threshold_probe",
    "f6_38_mule_fanout",
    "f6_39_shell_merchant",
    "f6_40_stored_value",
)

for _module in _MODULES:
    import_module(f"{__name__}.{_module}")

validate_registry()
