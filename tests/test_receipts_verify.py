"""Receipt verify: PayPal's custom_id against the receipt's recomputed hash."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import mantis.api.receipts as receipts
from mantis.api.app import app

ORDER = "5YM287057T143621E"  # seeded: the $400 GiftHub card, voided


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never reach the sandbox from a test: verify against the recordings."""
    monkeypatch.setenv("MANTIS_LIVE_DISABLED", "1")


def test_seeded_order_matches_its_receipt() -> None:
    body = TestClient(app).get(f"/receipts/{ORDER}/verify").json()
    assert body["match"] is True and body["source"] == "recorded-paypal"
    assert body["custom_id"] == body["receipt_hash"] == body["recorded_hash"]


def test_a_tampered_receipt_is_a_mismatch(tmp_path: Path) -> None:
    row_hash = receipts.verify_order(ORDER)["recorded_hash"]
    shutil.copy(receipts.SEED_RECEIPTS / f"{row_hash}.json", tmp_path / f"{row_hash}.json")
    path = tmp_path / f"{row_hash}.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["order"]["total"] = "4.00"  # someone edits the record after the fact
    path.write_text(json.dumps(receipt), encoding="utf-8")

    body = receipts.verify_order(ORDER, receipt_dirs=[tmp_path])
    assert body["match"] is False
    assert body["receipt_hash"] != body["recorded_hash"]
    assert "edited" in body["detail"]


def test_an_order_that_does_not_commit_to_the_receipt_is_a_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    order = json.loads((receipts.RECORDED_ORDERS / f"{ORDER}.json").read_text(encoding="utf-8"))
    order["purchase_units"][0]["custom_id"] = "0" * 64
    (tmp_path / f"{ORDER}.json").write_text(json.dumps(order), encoding="utf-8")
    monkeypatch.setattr(receipts, "RECORDED_ORDERS", tmp_path)

    body = receipts.verify_order(ORDER)
    assert body["match"] is False and "not the one scored" in body["detail"]


def test_unknown_order_is_404() -> None:
    assert TestClient(app).get("/receipts/NOT-AN-ORDER/verify").status_code == 404
