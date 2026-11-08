"""The LLM chat layer: cache-first, no network in mock, no secret on disk."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from mantis.core.paths import REPO_ROOT
from mantis.llm.cache import chat_key
from mantis.llm.client import CacheMissError, LlmChat, Settings

SECRET = "sk-test-THIS-MUST-NEVER-REACH-DISK"


def _settings() -> Settings:
    return Settings("fastrouter", {"agent": "m-agent", "judge": "m-judge", "compiler": "m-c"},
                    SECRET, "https://example.invalid/v1")


class _FakeProvider:
    """Counts calls and answers deterministically. Never touches the network."""

    name = "fastrouter"

    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self, model: str, request: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, int]]:
        self.calls += 1
        if request.get("tools"):
            return ({"content": None, "tool_calls": [
                {"id": "call_1", "name": "read_page", "arguments": '{"url": "https://x.test"}'}]},
                {"prompt_tokens": 10, "completion_tokens": 5})
        return {"content": f"answer from {model}", "tool_calls": []}, {}


def _chat(tmp_path: Path, mode: str, provider: _FakeProvider | None = None) -> LlmChat:
    return LlmChat("agent", mode=mode, cache_dir=tmp_path, settings=_settings(),  # type: ignore[arg-type]
                   provider=provider)


MSGS = [{"role": "user", "content": "hello"}]


def test_live_writes_through_and_mock_replays_without_a_provider(tmp_path: Path) -> None:
    provider = _FakeProvider()
    live = _chat(tmp_path, "live", provider).chat(MSGS)
    assert live.source == "live" and live.content == "answer from m-agent"
    assert provider.calls == 1

    replay = _chat(tmp_path, "mock").chat(MSGS)  # no provider at all
    assert replay.source == "cache" and replay.content == live.content
    assert replay.key == live.key and replay.latency_ms == pytest.approx(live.latency_ms, abs=0.1)


def test_tool_calls_round_trip_through_the_cache(tmp_path: Path) -> None:
    tools = [{"type": "function", "function": {"name": "read_page", "parameters": {}}}]
    _chat(tmp_path, "live", _FakeProvider()).chat(MSGS, tools=tools)
    replay = _chat(tmp_path, "mock").chat(MSGS, tools=tools)
    assert [(c.name, c.parsed()) for c in replay.tool_calls] == [("read_page", {"url": "https://x.test"})]
    assert replay.assistant_message()["tool_calls"][0]["function"]["name"] == "read_page"


def test_mock_miss_raises_and_never_calls_a_provider(tmp_path: Path) -> None:
    provider = _FakeProvider()
    with pytest.raises(CacheMissError):
        _chat(tmp_path, "mock", provider).chat(MSGS)
    assert provider.calls == 0


def test_key_ignores_dict_order_but_not_content() -> None:
    a = chat_key("p", "m", [{"role": "user", "content": "x"}], params={"temperature": 0, "seed": 1})
    b = chat_key("p", "m", [{"content": "x", "role": "user"}], params={"seed": 1, "temperature": 0})
    assert a == b
    assert a != chat_key("p", "m2", [{"role": "user", "content": "x"}],
                         params={"temperature": 0, "seed": 1})
    assert a != chat_key("p", "m", [{"role": "user", "content": "x"}],
                         params={"temperature": 1, "seed": 1})


def test_no_credential_is_ever_written(tmp_path: Path) -> None:
    _chat(tmp_path, "live", _FakeProvider()).chat(MSGS)
    for path in tmp_path.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert SECRET not in text
        json.loads(text)  # and every entry is valid JSON


def test_mock_replay_never_imports_the_provider_sdk(tmp_path: Path) -> None:
    """A clean clone has no 'openai' installed and must still replay."""
    _chat(tmp_path, "live", _FakeProvider()).chat(MSGS)
    code = (
        "import sys; from pathlib import Path\n"
        "from mantis.llm.client import LlmChat, Settings\n"
        "s = Settings('fastrouter', {'agent': 'm-agent', 'judge': 'j', 'compiler': 'c'}, '', None)\n"
        f"r = LlmChat('agent', mode='mock', cache_dir=Path(r'{tmp_path}'), settings=s)"
        ".chat([{'role': 'user', 'content': 'hello'}])\n"
        "print(r.source, 'openai' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=REPO_ROOT, check=True)
    assert out.stdout.split() == ["cache", "False"]
