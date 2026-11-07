"""Freeze every API response to JSON, so the console can be hosted anywhere.

Run: ``python scripts/build_static_site.py``  (``make static``)

The problem this solves
-------------------------
The console needs a backend, and a backend needs a host. Every free host that
runs Python sleeps after fifteen minutes and takes half a minute to wake — which
is fine for a judge who is browsing, and fatal for a judge who opens the link
during a four-minute pitch and sees a spinner.

So the console has two modes and picks between them automatically. If an API is
reachable it uses it, because that is the real system and the SSE stream is a
real stream. If one is not, it falls back to these frozen files and replays the
same committed feed on a client-side timer. Same events, same scores, same
decisions, same top-3 contributions — the numbers are identical because they come
from the same artefacts, and the API's own route handlers produced these files.

That last point is the design decision worth stating: this script does **not**
reimplement the endpoints. It calls the FastAPI app through ``TestClient`` and
writes what comes back. A hand-written dumper would be a second implementation
of every response shape, free to drift from the first, and the drift would show
up as a console that works locally and is subtly wrong when deployed.

What it does not fake
-----------------------
The fallback replays a pre-scored feed on a timer. It is **not** a server pushing
events, and the console says so on screen when it is in that mode rather than
letting a judge believe there is a backend where there is not. Streaming is a
property of the deployment, not a claim about the detector; the detector's
latency is measured and reported separately.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # so the script runs without an editable install
    sys.path.insert(0, str(REPO_ROOT))

from mantis.core.paths import ensure_dir  # noqa: E402

OUT_DIR: Final[Path] = REPO_ROOT / "web" / "public" / "data"

#: ``(filename, method, path, body)``. Ordered so the summary reads like the
#: console's own startup: what it is, then what it knows, then what it replays.
ROUTES: Final[tuple[tuple[str, str, str, dict[str, Any] | None], ...]] = (
    ("health.json", "GET", "/health", None),
    ("atlas.json", "GET", "/atlas", None),
    ("results.json", "GET", "/results", None),
    ("arena.json", "GET", "/arena", None),
    ("fidelity.json", "GET", "/fidelity", None),
    ("latency.json", "GET", "/latency", None),
)


def main() -> None:
    from fastapi.testclient import TestClient

    from mantis.api.app import app
    from mantis.api.store import STORE

    client = TestClient(app)
    ensure_dir(OUT_DIR)

    print("FREEZING THE API FOR STATIC HOSTING")
    print("=" * 74)
    print(f"  out  {OUT_DIR}")
    print()

    written = 0
    for name, method, path, body in ROUTES:
        response = (
            client.get(path) if method == "GET" else client.post(path, json=body or {})
        )
        if response.status_code != 200:
            # A missing optional artefact is not a failure. /fidelity is expected
            # to be unavailable until Day 7, and the console renders that state
            # deliberately. Skipping is right; inventing a payload is not.
            print(f"  {name:<16} SKIPPED  ({response.status_code} — artefact not generated)")
            continue
        target = OUT_DIR / name
        target.write_text(json.dumps(response.json(), separators=(",", ":")), encoding="utf-8")
        print(f"  {name:<16} {target.stat().st_size / 1024:>8.0f} KB")
        written += 1

    # Every atlas card in full, as one map keyed by id.
    #
    # /atlas/{id} is a route per card, and a static host has no routes. The
    # offline console therefore needs the details in a single file it can load
    # once - 42 cards is about 60 KB, which is cheaper than 42 requests would be
    # even against a live API. Built by calling the real route handler for each
    # card, like every other frozen response here, so there is no second
    # implementation of the card shape.
    detail = {}
    for card in client.get("/atlas").json()["cards"]:
        one = client.get(f"/atlas/{card['id']}")
        if one.status_code == 200:
            detail[card["id"]] = one.json()
    if detail:
        target = OUT_DIR / "atlas_cards.json"
        target.write_text(json.dumps(detail, separators=(",", ":")), encoding="utf-8")
        print(f"  {'atlas_cards.json':<16} {target.stat().st_size / 1024:>8.0f} KB  "
              f"({len(detail)} cards)")
        written += 1

    # The feed is written from the store rather than through a route, because no
    # endpoint returns the whole feed — /stream doles it out one frame at a time,
    # which is exactly what the fallback has to reproduce on a timer.
    if STORE.events:
        feed = {
            "manifest": STORE.feed_manifest,
            "frames": [
                {
                    "seq": i,
                    "event": {k: v for k, v in event.items() if k != "truth"},
                    "truth": event["truth"],
                }
                for i, event in enumerate(STORE.events)
            ],
        }
        target = OUT_DIR / "feed.json"
        target.write_text(json.dumps(feed, separators=(",", ":")), encoding="utf-8")
        print(f"  {'feed.json':<16} {target.stat().st_size / 1024:>8.0f} KB  "
              f"({len(feed['frames'])} frames)")
        written += 1
    else:
        print(f"  {'feed.json':<16} SKIPPED  (run 'make feed' first)")

    print()
    print(f"  {written} files. The console falls back to these when no API answers,")
    print("  so web/dist is a self-contained static site after 'npm run build'.")
    print()
    print("  Note: the frozen frames carry the SAME split the live stream uses —")
    print("  'event' is what the firewall knew, 'truth' is the answer, and they")
    print("  stay separate objects so the fallback cannot colour a row early either.")


if __name__ == "__main__":
    main()
