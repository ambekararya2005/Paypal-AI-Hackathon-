"""The one place repo-relative paths are resolved. Never use ``os.getcwd()``.

CLAUDE.md §5 requires a single helper module for this, and the reason is
HARD RULE 4: the repo must run from a clean clone. Anchoring on the location of
this file (not the working directory, not an environment variable) means
``python -m mantis.foundry.base`` writes to the same parquet whether it is run
from the repo root, from ``scripts/``, or from a judge's home directory.

``core`` imports nothing of ours, and this module imports nothing but the
standard library, so it sits at the very bottom of the dependency order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

__all__ = [
    "CACHE_DIR",
    "DATA_DIR",
    "DOCS_DIR",
    "GENERATED_DIR",
    "POPULATION_PARQUET",
    "REFERENCE_DIR",
    "REFERENCE_STATS_JSON",
    "REPO_ROOT",
    "ensure_dir",
]

#: Repo root: ``mantis/core/paths.py`` -> ``mantis/core`` -> ``mantis`` -> root.
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

DATA_DIR: Final[Path] = REPO_ROOT / "data"

#: Committed calibration tables. Small, versioned, safe to ship.
REFERENCE_DIR: Final[Path] = DATA_DIR / "reference"

#: Synthetic output. Gitignored — reproducible from a seed, so never committed.
GENERATED_DIR: Final[Path] = DATA_DIR / "generated"

#: LLM output cache. COMMITTED on purpose; see CLAUDE.md HARD RULE 3.
CACHE_DIR: Final[Path] = DATA_DIR / "cache"

DOCS_DIR: Final[Path] = REPO_ROOT / "docs"

#: Optional fitted calibration produced by ``scripts/fit_reference.py``. Absent
#: by default — the foundry falls back to committed Indian-market priors.
REFERENCE_STATS_JSON: Final[Path] = REFERENCE_DIR / "reference_stats.json"

#: The legitimate population the foundry writes and the injectors read.
POPULATION_PARQUET: Final[Path] = GENERATED_DIR / "population.parquet"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    """Print the resolved paths. Run: ``python -m mantis.core.paths``."""
    print("MANTIS resolved paths")
    for name in (
        "REPO_ROOT",
        "DATA_DIR",
        "REFERENCE_DIR",
        "GENERATED_DIR",
        "CACHE_DIR",
        "DOCS_DIR",
        "REFERENCE_STATS_JSON",
        "POPULATION_PARQUET",
    ):
        path: Path = globals()[name]
        mark = "exists" if path.exists() else "missing"
        print(f"  {name:<22} {mark:<8} {path}")


if __name__ == "__main__":
    main()
