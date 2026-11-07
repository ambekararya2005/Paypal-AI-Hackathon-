"""L0 — deterministic protocol-integrity clauses for the Mandate Firewall.

    from mantis.defense.l0_rules import evaluate
    result = evaluate(frame)
    result.fired    # bool per row
    result.reason   # the named clause that fired first

See :mod:`mantis.defense.l0_rules.rules` for what each clause asserts, and for
the argument about the one clause that is declared but never fires.
"""

from __future__ import annotations

from mantis.defense.l0_rules.rules import (
    CLAUSES,
    DECLARED_ONLY,
    MAX_DELEGATION_DEPTH,
    Clause,
    L0Result,
    evaluate,
    make_untrusted_domain_clause,
    trusted_domains,
)

__all__ = [
    "CLAUSES",
    "DECLARED_ONLY",
    "MAX_DELEGATION_DEPTH",
    "Clause",
    "L0Result",
    "evaluate",
    "make_untrusted_domain_clause",
    "trusted_domains",
]
