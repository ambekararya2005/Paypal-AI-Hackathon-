"""One origin, one process: the built console **and** the API together.

    python -m mantis.api            # http://127.0.0.1:8000  -> the console
    python -m mantis.api --api-only # just the JSON API, as Day 6 served it

Why this exists
---------------
Day 6 shipped two deployables — a static bundle on one host and a FastAPI
service on another — and the console had to negotiate between them across an
origin boundary. That is two things to keep awake, a CORS regex that has to
anticipate every preview subdomain, and two URLs to hand a judge. This module
collapses it to one:

``/api/*``   the whole Day 6 API, mounted unchanged. ``web/src/api.ts`` already
             defaults ``API_BASE`` to ``/api``, so a same-origin build needs no
             ``VITE_API_BASE`` and no CORS header ever gets exercised.
``/health``  duplicated at the root, because a container healthcheck should not
             have to know the API's mount point.
``/*``       the built Vite bundle from ``web/dist``, with an SPA fallback.

The API object itself is untouched: ``mantis.api.app:app`` is still the thing
``tests/test_api.py`` exercises and ``scripts/build_static_site.py`` freezes, so
the offline mode keeps working and this composition adds no second
implementation of any response shape.

**The console still works if ``web/dist`` is absent.** The mount is skipped and
``/`` returns a plain-text instruction to run ``make web``, rather than the app
refusing to start — a missing frontend should not take the API down with it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from starlette.responses import Response
from starlette.staticfiles import StaticFiles

from mantis.api.app import app as api_app
from mantis.api.models import HealthResponse
from mantis.core.paths import REPO_ROOT

__all__ = ["WEB_DIST", "site"]

#: The Vite build output. Gitignored and rebuilt by ``make web``; baked into the
#: container by the Dockerfile's node stage.
WEB_DIST: Final[Path] = REPO_ROOT / "web" / "dist"


class SPAStaticFiles(StaticFiles):
    """``StaticFiles`` that serves ``index.html`` for unmatched paths.

    The console is a single page with in-app tabs rather than a router, so this
    is belt and braces — but a judge who bookmarks a deep link and gets a bare
    404 has a broken demo, and the cost of preventing that is six lines.

    A missing **asset** still 404s: only extensionless paths fall through to the
    shell, so a mistyped bundle hash fails loudly instead of returning HTML with
    a JavaScript content type.
    """

    async def get_response(self, path: str, scope) -> Response:  # type: ignore[no-untyped-def]
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404 and "." not in Path(path).name:
                return await super().get_response("index.html", scope)
            raise


site = FastAPI(
    title="MANTIS console",
    version=api_app.version,
    description=(
        "The MANTIS live defence console and its API, served from one process. "
        "The JSON API is under /api; everything else is the built console."
    ),
    docs_url="/api-docs",
    openapi_url="/api-openapi.json",
)

# Mounted before the catch-all static mount: Starlette matches routes in the
# order they were added, so `/` would otherwise swallow every API path.
site.mount("/api", api_app, name="api")


@site.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """The same payload as ``/api/health``, at the path a container probes."""
    from mantis.api.app import health as api_health

    return api_health()


if WEB_DIST.is_dir() and (WEB_DIST / "index.html").is_file():
    site.mount("/", SPAStaticFiles(directory=WEB_DIST, html=True), name="console")
else:

    @site.get("/", response_class=PlainTextResponse, tags=["meta"])
    def no_console(request: Request) -> PlainTextResponse:
        return PlainTextResponse(
            "The MANTIS API is running, but the console bundle is not built.\n"
            f"Expected: {WEB_DIST}\n"
            "Build it with:  cd web && npm install && npm run build   (or: make web)\n"
            "The JSON API is available at /api  (docs at /api-docs).\n",
            status_code=503,
        )
