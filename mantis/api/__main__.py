"""Run MANTIS: the console and its API, from one process.

    python -m mantis.api                    # 127.0.0.1:8000 -> the console
    python -m mantis.api --host 0.0.0.0 --port 8080 --reload
    python -m mantis.api --api-only         # the bare JSON API, as Day 6 served it

By default this serves ``mantis.api.site:site`` — the built console at ``/`` and
the whole API under ``/api``, same origin. That is the Day 7 deployment shape:
one container, one URL, no CORS, nothing cross-service to keep awake. See
``mantis/api/site.py``.

Binds to loopback by default. A deployment passes ``--host 0.0.0.0`` explicitly,
which is the right way round: a demo server that listens on every interface
because that was the default is a demo server exposed on conference wifi.
"""

from __future__ import annotations

import argparse
import os

from mantis.api.store import STORE


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the MANTIS console API.")
    parser.add_argument("--host", default="127.0.0.1")
    # Render, Railway and Hugging Face Spaces all inject $PORT and expect the
    # process to honour it. Reading it here is what makes `python -m mantis.api`
    # a valid start command on all three without a wrapper script.
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="serve only the JSON API at the root, without the console bundle",
    )
    args = parser.parse_args()

    from mantis.api.site import WEB_DIST

    target = "mantis.api.app:app" if args.api_only else "mantis.api.site:site"
    # Printing the bind address back verbatim would tell a developer to open
    # http://0.0.0.0:8000, which is not a URL a browser can follow.
    shown = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host

    print("MANTIS API" if args.api_only else "MANTIS")
    print(f"  atlas    {STORE.headline.get('atlas_cards')} cards, "
          f"{STORE.headline.get('atlas_implemented')} implemented")
    print(f"  feed     {len(STORE.events)} pre-scored events")
    print(f"  results  {'loaded' if STORE.results else 'MISSING (run make firewall)'}")
    print(f"  arena    {'loaded' if STORE.arena else 'MISSING (run make loop)'}")
    print(f"  fidelity {'loaded' if STORE.fidelity else 'MISSING (run make fidelity)'}")
    if args.api_only:
        print(f"  -> http://{shown}:{args.port}/docs")
    else:
        built = (WEB_DIST / "index.html").is_file()
        print(f"  console  {'built' if built else 'NOT BUILT (run make web)'}  {WEB_DIST}")
        print(f"  -> http://{shown}:{args.port}/         the console")
        print(f"  -> http://{shown}:{args.port}/api-docs the API")
    print()

    import uvicorn

    uvicorn.run(
        target,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
