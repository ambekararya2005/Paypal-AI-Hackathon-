"""FastAPI backend for the live defence console.

Nothing in the project imports this package — see CLAUDE.md §3, the dependency
direction is one-way and ``api`` is at the end of it. It reads committed
artefacts and serves them; it never fits a model inside a request.
"""

from __future__ import annotations

__all__ = ["app"]


def __getattr__(name: str):
    # Imported lazily so that `import mantis.api` stays cheap and does not drag
    # FastAPI and its dependency tree into processes that only wanted the store.
    if name == "app":
        from mantis.api.app import app

        return app
    raise AttributeError(name)
