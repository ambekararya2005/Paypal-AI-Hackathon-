"""Fetch the Kaggle Sparkov reference panel the fidelity scorecard measures against.

    python scripts/fetch_reference.py            # ~210 MB, into data/reference/
    python scripts/fetch_reference.py --force    # re-download over an existing copy

This is an **optional upgrade, never a dependency**. HARD RULE 4 says the repo
runs from a clean clone with no Kaggle token, and it still does: without this file
the scorecard prints its provenance and level sections, marks the marginal, TSTR
and discriminator sections skipped, and exits 1. Nothing else in the project reads
it.

Why no Kaggle token is needed
------------------------------
``kartik2112/fraud-detection`` is a public dataset, and Kaggle's
``/api/v1/datasets/download/{owner}/{slug}`` endpoint serves public archives
without authentication. That is why this script is 60 lines of the standard
library rather than a dependency on the ``kaggle`` package plus a credentials
file, and it is why running it needs nothing but a network connection.

Why the CSVs are gitignored
----------------------------
501 MB unzipped. ``data/reference/`` is described in CLAUDE.md as "calibration
tables (committed, small)", and half a gigabyte of somebody else's data is
neither. ``.gitignore`` excludes ``data/reference/*.csv`` explicitly.

What this does **not** do
--------------------------
It does not run ``scripts/fit_reference.py``. Fitting the population's shape
parameters from this panel would re-roll every calibration number the last six
days pinned, and the scorecard's whole job is to *measure* the gap between the
prior-calibrated population and an external panel rather than to close it by
refitting. The two scripts are deliberately separate steps.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Final

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # so the script runs without an editable install
    sys.path.insert(0, str(REPO_ROOT))

from mantis.core.paths import REFERENCE_DIR, ensure_dir  # noqa: E402
from mantis.foundry.fidelity.real import REAL_FILES  # noqa: E402

DATASET: Final[str] = "kartik2112/fraud-detection"
URL: Final[str] = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}"

#: Read size for the streamed download. 1 MB keeps the progress line moving
#: without making a syscall per kilobyte.
CHUNK: Final[int] = 1 << 20


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the Sparkov reference panel.")
    parser.add_argument("--force", action="store_true", help="re-download over an existing copy")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)

    ensure_dir(REFERENCE_DIR)
    present = [name for name in REAL_FILES if (REFERENCE_DIR / name).is_file()]
    if present and not args.force:
        print("Reference panel already present:")
        for name in present:
            size = (REFERENCE_DIR / name).stat().st_size
            print(f"  {name:<20} {size / 1e6:>8.1f} MB")
        print()
        print("  Nothing to do. Pass --force to re-download.")
        print("  Next:  python -m mantis.foundry.fidelity")
        return 0

    archive = REFERENCE_DIR / "fraud-detection.zip"
    print(f"Fetching {DATASET} (~210 MB, no Kaggle token needed)")
    started = time.time()
    try:
        with urllib.request.urlopen(URL, timeout=args.timeout) as response:
            declared = response.headers.get("Content-Length")
            if declared:
                print(f"  {int(declared) / 1e6:.1f} MB")
            with archive.open("wb") as handle:
                shutil.copyfileobj(response, handle, CHUNK)
    except OSError as error:
        # A network failure here is a normal state, not a crash: the scorecard
        # degrades without the panel, so this reports and returns rather than
        # raising into a traceback that looks like a broken repo.
        print(f"  download failed: {error}")
        print("  The fidelity scorecard will run without it and mark sections 2-4 skipped.")
        archive.unlink(missing_ok=True)
        return 1

    print(f"  downloaded in {time.time() - started:.0f}s, extracting")
    with zipfile.ZipFile(archive) as bundle:
        wanted = [name for name in bundle.namelist() if name in REAL_FILES]
        if not wanted:
            print(f"  archive did not contain {REAL_FILES}; it held {bundle.namelist()}")
            return 1
        bundle.extractall(REFERENCE_DIR, members=wanted)
    archive.unlink(missing_ok=True)

    print()
    for name in REAL_FILES:
        path = REFERENCE_DIR / name
        if path.is_file():
            print(f"  {name:<20} {path.stat().st_size / 1e6:>8.1f} MB")
    print()
    print("  Gitignored on purpose - see this script's docstring.")
    print("  Next:  python -m mantis.foundry.fidelity")
    return 0


if __name__ == "__main__":
    sys.exit(main())
