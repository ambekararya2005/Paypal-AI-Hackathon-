"""The public deployment's spending guards: a kill switch and daily caps.

    python -m mantis.core.quota        # today's counters, the caps, and the switch

Everything that can cost money or create a real sandbox order — a live LLM call,
a PayPal sandbox call, a playground agent run — calls :func:`spend` first. It is
standard library only and lives in ``core`` so the LLM client (low in the
dependency order) and the PayPal client (high) can both reach it.

``MANTIS_LIVE_DISABLED=1``
    The kill switch. Every :func:`spend` raises, so no live call is made anywhere;
    the public site keeps working on mock PayPal and recorded runs. Applies
    everywhere, including a developer's laptop.
``MANTIS_PUBLIC=1``
    Turns the **daily caps** on (set on Render). Off locally, so recording a demo
    or running the tests is never throttled by a counter on disk.
``MANTIS_CAP_RUNS`` / ``MANTIS_CAP_LLM`` / ``MANTIS_CAP_PAYPAL``
    Per-UTC-day ceilings; defaults below.

Counters are kept in ``<runtime dir>/quota.json`` — on Render the persistent disk,
so a restart does not hand out a fresh day's budget.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from mantis.core.paths import DATA_DIR

__all__ = [
    "DEFAULT_CAPS",
    "QuotaExceededError",
    "caps",
    "caps_enforced",
    "live_disabled",
    "runtime_dir",
    "snapshot",
    "spend",
]

#: Per UTC day. "run" is a playground agent run (100/day, the agreed cap); "llm"
#: and "paypal" bound the calls inside runs and anything else that goes live.
DEFAULT_CAPS: Final[dict[str, int]] = {"run": 100, "llm": 2500, "paypal": 1500}

_LOCK = threading.Lock()


class QuotaExceededError(RuntimeError):
    """A live call was refused: the kill switch is on or today's cap is spent."""

    def __init__(self, kind: str, reason: str) -> None:
        self.kind = kind
        self.reason = reason
        super().__init__(f"live {kind} call refused: {reason}")


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def runtime_dir() -> Path:
    """Writable state: ``MANTIS_RUNTIME_DIR`` (Render's disk) or ``data/runtime``."""
    raw = os.environ.get("MANTIS_RUNTIME_DIR", "").strip()
    return Path(raw) if raw else DATA_DIR / "runtime"


def live_disabled() -> bool:
    return _truthy("MANTIS_LIVE_DISABLED")


def caps_enforced() -> bool:
    return _truthy("MANTIS_PUBLIC")


#: Spelled out rather than derived from the kind: deriving gave ``MANTIS_CAP_RUN``
#: while render.yaml sets ``MANTIS_CAP_RUNS``, and the cap was silently ignored.
_CAP_ENV: Final[dict[str, str]] = {
    "run": "MANTIS_CAP_RUNS", "llm": "MANTIS_CAP_LLM", "paypal": "MANTIS_CAP_PAYPAL",
}


def caps() -> dict[str, int]:
    out = dict(DEFAULT_CAPS)
    for kind in out:
        raw = os.environ.get(_CAP_ENV[kind], "").strip()
        if raw.isdigit():
            out[kind] = int(raw)
    return out


def _path() -> Path:
    return runtime_dir() / "quota.json"


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _load() -> dict[str, int]:
    try:
        raw = json.loads(_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: int(v) for k, v in raw.get("counts", {}).items()} if raw.get("day") == _today() \
        else {}


def spend(kind: str, n: int = 1) -> None:
    """Record ``n`` live calls of ``kind``, or raise :class:`QuotaExceededError`.

    Check-and-increment happens under one lock, so two threads cannot both take
    the last unit of a cap.
    """
    if live_disabled():
        raise QuotaExceededError(kind, "live calls are disabled on this deployment "
                                       "(MANTIS_LIVE_DISABLED=1)")
    if not caps_enforced():
        return
    limit = caps().get(kind)
    with _LOCK:
        counts = _load()
        used = counts.get(kind, 0)
        if limit is not None and used + n > limit:
            raise QuotaExceededError(kind, f"today's cap of {limit} is spent; "
                                           "it resets at 00:00 UTC")
        counts[kind] = used + n
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"day": _today(), "counts": counts}), encoding="utf-8")
        tmp.replace(path)


def snapshot() -> dict[str, object]:
    """For ``/api/limits`` and the playground banner. No secrets in here."""
    counts = _load()
    limits = caps()
    return {
        "live_disabled": live_disabled(),
        "caps_enforced": caps_enforced(),
        "day": _today(),
        "used": {k: counts.get(k, 0) for k in limits},
        "caps": limits,
        "remaining": {k: max(0, limits[k] - counts.get(k, 0)) for k in limits},
    }


def main() -> None:
    print(json.dumps(snapshot(), indent=2))


if __name__ == "__main__":
    main()
