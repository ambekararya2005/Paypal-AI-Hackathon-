"""Print the probe-slice audit as a table a human can read.

    python scripts/audit_probe_slices.py [--n 200000] [--seed 1337]

Why this is a script and not just the test suite
------------------------------------------------
``tests/test_probe_slices.py`` already *enforces* the slice rule — declared
columns on the allow-list, the mask proved to be a function of those columns
alone, the whole attack inside its own slice. What it does not do is show the
numbers. A green test says "no slice cheats"; it does not say *how thin the
denominator got*, and the denominator is the thing that decides whether a
conditional AUC can be quoted on a slide.

The rule being audited, restated
---------------------------------
**A slice may condition only on facts a detector already knows before it scores,
and which are not consequences of the attack.** Rail, processing code, category.
Not amount, not the issuer's decision on this message, not any ``ag_scope_*`` or
behavioural column — all of those are the attack's own footprint, and grading an
attack inside a slice it created is marking your own homework.

Four checks per injector, matching the tests one-for-one:

``declared``   every column in ``slice_columns`` is on ``SLICE_ALLOWED_COLUMNS``
``function``   the mask is constant within every group of rows agreeing on the
               declared columns — so it cannot be reading anything else
``covers``     every attack row falls inside its own slice
``n``          the slice denominator, flagged when under ``THIN_SLICE_ROWS``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mantis.foundry.base.reference import load_reference_stats  # noqa: E402
from mantis.foundry.base.simulator import SimulationConfig, simulate_frame  # noqa: E402
from mantis.foundry.injectors import REGISTRY  # noqa: E402
from mantis.foundry.injectors.base import PopulationView, run_injector  # noqa: E402
from mantis.foundry.injectors.probe import SLICE_ALLOWED_COLUMNS, THIN_SLICE_ROWS  # noqa: E402


def audit_one(
    card_id: str, background: pd.DataFrame, view: PopulationView, seed: int
) -> dict[str, object]:
    """Audit one injector's slice. Returns a row for the table."""
    cls = REGISTRY[card_id]
    row: dict[str, object] = {
        "attack_id": card_id,
        "columns": ", ".join(cls.slice_columns) or "-",
    }
    if not cls.slice_columns:
        # No slice: graded against the whole population, which is the default and
        # needs no justification.
        row.update(
            {"declared": "n/a", "function": "n/a", "covers": "n/a", "n": len(background),
             "share": 1.0, "thin": False}
        )
        return row

    illegal = set(cls.slice_columns) - SLICE_ALLOWED_COLUMNS
    row["declared"] = "ok" if not illegal else f"ILLEGAL {sorted(illegal)}"

    mask = np.asarray(cls.probe_slice(background), dtype=bool)
    key = background[list(cls.slice_columns)].astype(str).agg("\x1f".join, axis=1)
    varies = pd.Series(mask).groupby(key.to_numpy()).nunique()
    n_bad = int((varies > 1).sum())
    row["function"] = "ok" if n_bad == 0 else f"LEAKS ({n_bad} mixed groups)"

    attack = run_injector(cls, view, seed=seed)
    inside = np.asarray(cls.probe_slice(attack), dtype=bool)
    row["covers"] = "ok" if inside.all() else f"DROPS {int((~inside).sum())}/{len(attack)}"

    n = int(mask.sum())
    row["n"] = n
    row["share"] = n / len(background)
    row["thin"] = n < THIN_SLICE_ROWS
    return row


def format_table(rows: list[dict[str, object]], n_background: int) -> str:
    lines = [
        "probe-slice audit",
        "",
        f"  {'card':<7} {'declared':<9} {'function':<9} {'covers':<9} "
        f"{'slice n':>9} {'share':>7}  conditions on",
        f"  {'-' * 7} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 7}  {'-' * 26}",
    ]
    for r in rows:
        flag = " !" if r["thin"] else "  "
        lines.append(
            f"  {r['attack_id']:<7} {r['declared']!s:<9} {r['function']!s:<9} "
            f"{r['covers']!s:<9} {int(r['n']):>9,}{flag}{float(r['share']):>6.1%}  "
            f"{r['columns']}"
        )
    thin = [r["attack_id"] for r in rows if r["thin"]]
    lines += [
        "",
        f"  background {n_background:,} rows. '!' marks a slice under "
        f"{THIN_SLICE_ROWS:,} rows, the point below which the 95% interval on a",
        "  conditional AUC starts to straddle the 0.95 gate. Not an error: a rare",
        "  slice is still the right denominator. It is a caveat that must travel",
        "  with the number.",
    ]
    if thin:
        lines.append(f"  THIN: {', '.join(thin)} - quote the conditional AUC with its n attached.")
    else:
        lines.append("  no thin slices.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/audit_probe_slices.py")
    parser.add_argument("--n", type=int, default=200_000, help="background events")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    background = simulate_frame(
        SimulationConfig(n_events=args.n, seed=args.seed, n_customers=5_000, n_merchants=12_000),
        load_reference_stats(),
    )
    view = PopulationView.build(background)
    rows = [audit_one(c, background, view, args.seed) for c in sorted(REGISTRY)]
    print(format_table(rows, len(background)))

    broken = [r for r in rows if "ok" not in (str(r["declared"]), str(r["function"]))
              and r["declared"] != "n/a"]
    broken += [r for r in rows if str(r["covers"]).startswith("DROPS")]
    if broken:
        print()
        print(f"  FAIL: {sorted({str(r['attack_id']) for r in broken})}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
