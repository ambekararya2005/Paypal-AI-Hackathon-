"""Public-deployment guards: kill switch, daily caps, rate limit, body size."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from mantis.api.guards import MAX_BODY_BYTES, SlidingWindow
from mantis.core import quota


@pytest.fixture(autouse=True)
def runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MANTIS_RUNTIME_DIR", str(tmp_path))
    for var in ("MANTIS_LIVE_DISABLED", "MANTIS_PUBLIC", "MANTIS_CAP_RUNS", "MANTIS_CAP_LLM"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def test_kill_switch_refuses_every_live_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANTIS_LIVE_DISABLED", "1")
    for kind in ("llm", "paypal", "run"):
        with pytest.raises(quota.QuotaExceededError, match="disabled"):
            quota.spend(kind)


def test_kill_switch_stops_the_llm_client_before_the_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mantis.llm.client import LlmChat, Settings

    class Boom:
        name = "fastrouter"

        def complete(self, *_: object) -> tuple[dict[str, object], dict[str, int]]:
            raise AssertionError("the provider must never be reached")

    monkeypatch.setenv("MANTIS_LIVE_DISABLED", "1")
    chat = LlmChat("agent", mode="live", cache_dir=tmp_path / "c",
                   settings=Settings("fastrouter", {"agent": "m", "judge": "j", "compiler": "c"},
                                     "k", None), provider=Boom())  # type: ignore[arg-type]
    with pytest.raises(quota.QuotaExceededError):
        chat.chat([{"role": "user", "content": "hi"}])


def test_kill_switch_degrades_env_live_to_mock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mantis.paypal.config import load_config

    env = tmp_path / ".env"
    env.write_text("MANTIS_PAYPAL_MODE=live\nPAYPAL_CLIENT_ID=a\nPAYPAL_CLIENT_SECRET=b\n",
                   encoding="utf-8")
    monkeypatch.delenv("MANTIS_PAYPAL_MODE", raising=False)
    assert load_config(env_file=env).mode == "live"
    monkeypatch.setenv("MANTIS_LIVE_DISABLED", "1")
    assert load_config(env_file=env).mode == "mock"


def test_caps_are_off_locally_and_on_in_public(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANTIS_CAP_RUNS", "2")
    for _ in range(5):
        quota.spend("run")  # MANTIS_PUBLIC unset: never throttled
    monkeypatch.setenv("MANTIS_PUBLIC", "1")
    quota.spend("run")
    quota.spend("run")
    with pytest.raises(quota.QuotaExceededError, match="cap of 2"):
        quota.spend("run")
    assert quota.snapshot()["remaining"]["run"] == 0  # type: ignore[index]


def test_cap_holds_under_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANTIS_PUBLIC", "1")
    monkeypatch.setenv("MANTIS_CAP_LLM", "10")
    granted: list[int] = []

    def take() -> None:
        try:
            quota.spend("llm")
            granted.append(1)
        except quota.QuotaExceededError:
            pass

    threads = [threading.Thread(target=take) for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(granted) == 10, "exactly the cap, never one more"


def test_sliding_window_limits_and_recovers() -> None:
    window = SlidingWindow(3, 60.0)
    assert [window.allow("ip", now=t)[0] for t in (0, 1, 2, 3)] == [True, True, True, False]
    assert window.allow("other", now=3)[0], "limits are per key"
    assert window.allow("ip", now=61)[0], "the oldest hit has left the window"


def test_api_rate_limit_and_body_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from mantis.api.guards import GuardMiddleware

    app = FastAPI()
    app.add_middleware(GuardMiddleware, per_min=3, reads_per_min=5)

    @app.get("/ping")
    def ping() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/echo")
    def echo() -> dict[str, bool]:
        return {"ok": True}

    client = TestClient(app)
    ip = {"x-forwarded-for": "1.2.3.4"}
    writes = [client.post("/echo", headers=ip).status_code for _ in range(4)]
    assert writes == [200, 200, 200, 429], "writes: the tight limit"
    reads = [client.get("/ping", headers=ip).status_code for _ in range(6)]
    assert reads == [200] * 5 + [429], "reads have their own, looser limit"
    assert client.get("/ping", headers={"x-forwarded-for": "5.6.7.8"}).status_code == 200
    big = client.post("/echo", content=b"x" * (MAX_BODY_BYTES + 1),
                      headers={"x-forwarded-for": "9.9.9.9"})
    assert big.status_code == 413
