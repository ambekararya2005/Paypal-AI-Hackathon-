"""Settings from the environment, with ``.env`` as a fallback. Standard library only.

Lives in ``core`` because two layers need it that must not import each other:
``mantis.llm`` (provider keys, model ids) sits low in the dependency order and
``mantis.paypal`` (PayPal keys) sits high. ``core`` imports nothing of ours.

Real environment variables always win over the file, so a CI secret or a shell
export overrides a stale ``.env`` rather than being silently shadowed by it.
Values are never logged by anything in this module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from mantis.core.paths import REPO_ROOT

__all__ = ["ENV_FILE", "read_env_file", "setting"]

ENV_FILE: Final[Path] = REPO_ROOT / ".env"


def read_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    """Parse ``KEY=value`` lines. A missing file is not an error.

    Blank lines and ``#`` comments are skipped, surrounding quotes stripped, and a
    stray ``\\r`` from a Windows editor removed — the last one matters, because a
    trailing carriage return in a secret is a 401 that looks like a wrong password.
    """
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip().rstrip("\r")
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key.strip()] = value
    return out


def setting(key: str, default: str = "", *, env_file: Path = ENV_FILE) -> str:
    """One setting: the process environment first, then ``.env``, then ``default``."""
    return os.environ.get(key) or read_env_file(env_file).get(key) or default
