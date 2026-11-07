"""The API, and the contract the console reads it through.

Most of this file is not testing FastAPI. It is pinning the **column names**
the React components index into.

Why that is worth a test file
-------------------------------
The results screen reads ``r["L1 (trained WITH it)"]`` out of a table that was
parsed out of a markdown document that was written by ``report.py`` from a
fitted experiment. That is four hops, and every one of them is a string. Rename
a heading in ``report.py`` to something that reads better and the chart silently
renders ``NaN`` — no exception, no failed request, no red anywhere. A judge sees
an empty bar and concludes the recall is zero.

So the strings the console depends on are asserted here, next to a comment
saying which component reads them. If a heading is deliberately reworded, this
file fails and the TypeScript gets updated in the same commit. That is the whole
point: make the coupling loud instead of discovering it during a demo.

The second thing this file protects is the label discipline at the API boundary
(``test_stream_frame_never_carries_ground_truth``). CLAUDE.md HARD RULE 1 keeps
labels out of feature matrices; the same reasoning applies on the wire, because
an event arriving at the console with ``is_fraud`` on it would let the UI colour
a row before the score came back.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from mantis.api.app import app
from mantis.api.store import STORE
from mantis.defense import results_doc


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


# --------------------------------------------------------------------- meta --
def test_health_reports_what_is_loaded(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    # 42 is the frozen atlas size and is pinned in tests/test_atlas.py too. If
    # both move together that is a deliberate act; if only one moves it is a bug.
    assert body["atlas_cards"] == 42
    assert body["atlas_implemented"] <= body["atlas_cards"]


def test_atlas_implemented_count_is_the_ratchet(client: TestClient) -> None:
    """``implemented`` counts injectors, not YAML files."""
    body = client.get("/atlas").json()
    assert body["total"] == 42
    with_generator = [c for c in body["cards"] if c["has_injector"]]
    assert body["implemented"] == len(with_generator)
    # And the flag cannot disagree with the card's own status field.
    for card in body["cards"]:
        assert card["has_injector"] == (card["status"] == "implemented")


def test_unknown_card_is_404(client: TestClient) -> None:
    assert client.get("/atlas/F9-99").status_code == 404


def test_fidelity_reports_absence_rather_than_erroring(client: TestClient) -> None:
    """Day 7's artefact is missing, and the endpoint must say so, not 404.

    A console panel that renders "not measured yet" is telling the truth. One
    that renders an error looks like the service is broken.
    """
    response = client.get("/fidelity")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["available"], bool)
    if not body["available"]:
        assert "not" in body["note"].lower()


# ------------------------------------------------- the console's column names --
def _records(client: TestClient, key: str) -> list[dict[str, str]]:
    return client.get("/results").json()[key]


@pytest.mark.skipif(STORE.results is None, reason="RESULTS.md not generated")
def test_layer_performance_columns_the_console_reads(client: TestClient) -> None:
    """Read by web/src/Results.tsx — FprCurve and the layer-performance table."""
    rows = _records(client, "layer_performance")
    assert rows, "no layer performance rows"
    for column in ("layer", "AUC-PR", "recall@0.1%", "recall@0.5%", "recall@1.0%",
                   "campaign recall"):
        assert column in rows[0], f"Results.tsx indexes {column!r}; report.py no longer emits it"
    assert any(r["layer"].startswith("L1") for r in rows)
    assert any(r["layer"].startswith("fused") for r in rows)


@pytest.mark.skipif(STORE.results is None, reason="RESULTS.md not generated")
def test_per_family_columns_the_console_reads(client: TestClient) -> None:
    """Read by web/src/Results.tsx — FamilyRecall."""
    rows = _records(client, "per_family")
    assert rows
    for column in ("family", "L1 (trained WITH it)", "L1 (family HELD OUT)"):
        assert column in rows[0], f"Results.tsx indexes {column!r}"


@pytest.mark.skipif(STORE.results is None, reason="RESULTS.md not generated")
def test_per_attack_columns_the_console_reads(client: TestClient) -> None:
    """Read by web/src/Results.tsx — PerCard."""
    rows = _records(client, "per_attack")
    assert rows
    assert "card" in rows[0] and "fused" in rows[0]


@pytest.mark.skipif(STORE.results is None, reason="RESULTS.md not generated")
def test_zero_day_is_exactly_three_rows(client: TestClient) -> None:
    """The recovery table is the submission's argument and it has three rows.

    Trained / held out / loop-augmented. The console renders them as three big
    numbers in that order and captions each one specifically, so a fourth row or
    a reordering is a change the UI cannot absorb silently.
    """
    rows = client.get("/results").json()["zero_day"]
    assert len(rows) == 3
    trained, held_out, augmented = (r["recall"] for r in rows)
    assert trained > held_out, "holding the family out must reduce recall"
    assert augmented > held_out, "the loop must recover some of the collapse"
    assert augmented < trained, (
        "the loop recovering MORE than training on the real family would mean the "
        "variants leak the test rows; that is the failure this assertion exists for"
    )


@pytest.mark.skipif(STORE.results is None, reason="RESULTS.md not generated")
def test_headline_numbers_are_present_and_sane(client: TestClient) -> None:
    """Read by web/src/Results.tsx — the tally strip across the top."""
    headline = client.get("/results").json()["headline"]
    for key in ("n_events", "atlas_cards", "atlas_implemented", "l1_recall", "fused_recall"):
        assert headline.get(key) is not None, f"headline is missing {key!r}"
    assert 0.0 <= headline["l1_recall"] <= 1.0
    assert 0.0 <= headline["fused_recall"] <= 1.0


# -------------------------------------------------------------------- arena --
@pytest.mark.skipif(STORE.arena is None, reason="arena.json not generated")
def test_arena_shape_the_console_charts(client: TestClient) -> None:
    body = client.get("/arena").json()
    assert body["evasion_curve"], "no evasion curve"
    for generation in body["generations"]:
        for key in ("generation", "mean_evasion", "max_evasion"):
            assert key in generation
        assert generation["max_evasion"] >= generation["mean_evasion"]
    if body["zero_day"]:
        for key in ("family", "n_test_positive", "gap_closed", "n_variant_events"):
            assert key in body["zero_day"]


# ------------------------------------------------------------------- stream --
@pytest.mark.skipif(not STORE.events, reason="console_feed.json not generated")
def test_simulate_then_stream_delivers_frames(client: TestClient) -> None:
    run = client.post("/simulate", json={"n_events": 5, "rate": 60}).json()
    assert run["n_events"] == 5

    kinds: list[str] = []
    payloads: list[dict] = []
    with client.stream("GET", f"/stream/{run['run_id']}") as stream:
        current: str | None = None
        for line in stream.iter_lines():
            if line.startswith("event:"):
                current = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and current:
                kinds.append(current)
                payloads.append(json.loads(line.split(":", 1)[1].strip()))
                current = None

    assert kinds[0] == "meta"
    assert kinds[-1] == "done"
    assert kinds.count("auth") == 5
    # seq must be dense and ordered; the console keys React rows on it.
    auth_seqs = [p["seq"] for k, p in zip(kinds, payloads, strict=True) if k == "auth"]
    assert auth_seqs == list(range(5))


@pytest.mark.skipif(not STORE.events, reason="console_feed.json not generated")
def test_stream_frame_never_carries_ground_truth(client: TestClient) -> None:
    """HARD RULE 1's reasoning, applied at the API boundary.

    ``event`` is what the firewall knew when it decided. ``truth`` is the answer.
    They are siblings on the frame, and the answer must never be inside the
    event — otherwise the console could colour a row before the score arrived and
    the demo would be a lookup wearing a detector's clothes.
    """
    run = client.post("/simulate", json={"n_events": 3, "rate": 60}).json()
    seen = 0
    with client.stream("GET", f"/stream/{run['run_id']}") as stream:
        current: str | None = None
        for line in stream.iter_lines():
            if line.startswith("event:"):
                current = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and current == "auth":
                frame = json.loads(line.split(":", 1)[1].strip())
                event = frame["event"]
                for banned in ("is_fraud", "attack_id", "attack_campaign", "truth"):
                    assert banned not in event, f"{banned!r} leaked into the scored event"
                assert set(frame["truth"]) == {"is_fraud", "attack_id", "attack_campaign"}
                seen += 1
                current = None
    assert seen == 3


@pytest.mark.skipif(not STORE.events, reason="console_feed.json not generated")
def test_every_feed_event_has_what_the_console_renders() -> None:
    """Read by web/src/Console.tsx — AuthRow and Inspector."""
    for event in STORE.events:
        assert event["decision"] in {"approve", "challenge", "review", "decline"}
        assert isinstance(event["l0"]["fired"], bool)
        # Top-3, per the brief. Fewer would leave the alert panel short.
        assert len(event["contributions"]) == 3
        for contribution in event["contributions"]:
            assert set(contribution) == {"feature", "value", "contribution"}
        for layer in ("L1", "L2", "L3"):
            assert layer in event["layers"]


@pytest.mark.skipif(not STORE.events, reason="console_feed.json not generated")
def test_risk_index_agrees_with_the_decision() -> None:
    """The dial and the badge must not contradict each other.

    A raw percentile does contradict it — every decision this firewall makes
    lives in the top 1% of the legitimate distribution, so an ordinary approved
    authorisation reads 0.95. The feed builder therefore interpolates a 0-100
    index through the policy's own boundaries, and this pins that it worked.

    Declines are exempt in one direction: an L0 protocol violation declines
    outright whatever the score says, so a low-risk decline is correct and is
    exactly what the console's L0 banner exists to explain.
    """
    for event in STORE.events:
        risk = event["risk"]
        if risk is None:
            continue
        assert 0.0 <= risk <= 100.0
        if event["decision"] == "approve":
            assert risk < 50.0, f"an approved event reads {risk:.0f} on the dial"
        if event["decision"] == "decline" and not event["l0"]["fired"]:
            assert risk >= 50.0


@pytest.mark.skipif(not STORE.events, reason="console_feed.json not generated")
def test_feed_declares_that_it_over_samples_fraud(client: TestClient) -> None:
    """The replay is curated and must say so where a judge can read it.

    The console prints this string in its header. A stream that looks like
    production traffic and is not is something a judge should hear from the
    console rather than work out afterwards.
    """
    manifest = STORE.feed_manifest
    assert "over-sampled" in manifest["sampling_note"]
    assert manifest["fraud_share_in_feed"] > manifest["true_prevalence_in_test_window"]
    # And the API repeats it on the run registration and the meta frame.
    assert "over-sampled" in client.post("/simulate", json={"n_events": 1}).json()["note"]


def test_unknown_run_id_is_404(client: TestClient) -> None:
    assert client.get("/stream/deadbeef").status_code == 404


# ------------------------------------------------------- the markdown parser --
def test_parser_ignores_prose_that_merely_contains_pipes() -> None:
    """A table is a header row *followed by a separator*, not any line with pipes."""
    doc = results_doc.parse(
        "## S\n\nthe grid is 0.5 | 1.0 | 2.0 wide\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    )
    section = doc.find("S")
    assert section is not None
    assert len(section.tables) == 1
    assert section.tables[0].header == ["a", "b"]


def test_parser_strips_markup_from_cells() -> None:
    doc = results_doc.parse("## S\n\n| x |\n|---|\n| **0.539** |\n| `code` |\n")
    table = doc.table_for("S")
    assert table is not None
    assert table.rows == [["0.539"], ["code"]]


def test_parser_tolerates_a_ragged_row() -> None:
    """A short row degrades one field rather than blanking a whole endpoint."""
    doc = results_doc.parse("## S\n\n| a | b | c |\n|---|---|---|\n| 1 | 2 |\n")
    table = doc.table_for("S")
    assert table is not None
    assert table.as_records() == [{"a": "1", "b": "2", "c": ""}]


def test_loading_a_document_with_no_tables_raises() -> None:
    """Silence here would mean every downstream number is quietly empty."""
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "RESULTS.md"
        path.write_text("# nothing but prose\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no tables"):
            results_doc.load(path)


# ------------------------------------------------- the static (no-API) bundle --
def test_static_bundle_matches_the_live_api(client: TestClient) -> None:
    """The frozen files and the live routes must not drift apart.

    ``web/public/data/`` is what the console reads when no backend answers. It is
    produced by calling these same route handlers, so the two agree by
    construction — but only until someone regenerates one and not the other. A
    console that shows one set of numbers when deployed and another when run
    locally is the exact failure this asserts against.
    """
    from mantis.core.paths import REPO_ROOT

    data_dir = REPO_ROOT / "web" / "public" / "data"
    if not data_dir.exists():
        pytest.skip("static bundle not built; run 'make static'")

    for name, path in (
        ("health.json", "/health"),
        ("atlas.json", "/atlas"),
        ("arena.json", "/arena"),
    ):
        frozen_path = data_dir / name
        if not frozen_path.exists():
            continue
        live = client.get(path)
        if live.status_code != 200:
            continue
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        assert frozen == live.json(), (
            f"{name} has drifted from {path}; re-run 'make static'"
        )


def test_static_feed_keeps_ground_truth_out_of_the_event() -> None:
    """The offline replay must not leak the answer either.

    The live stream splits ``event`` from ``truth`` so the console cannot colour
    a row before the score arrives. The frozen feed is read by the same
    components down the same code path, so it has to hold the same split — and
    it is a separate file written by a separate function, so it needs its own
    assertion rather than inheriting the stream's.
    """
    from mantis.core.paths import REPO_ROOT

    path = REPO_ROOT / "web" / "public" / "data" / "feed.json"
    if not path.exists():
        pytest.skip("static bundle not built; run 'make static'")

    feed = json.loads(path.read_text(encoding="utf-8"))
    assert feed["frames"], "the frozen feed is empty"
    for frame in feed["frames"]:
        assert set(frame) == {"seq", "event", "truth"}
        for banned in ("is_fraud", "attack_id", "attack_campaign"):
            assert banned not in frame["event"]
    # seq must be dense from zero: the console keys React rows on it.
    assert [f["seq"] for f in feed["frames"]] == list(range(len(feed["frames"])))
