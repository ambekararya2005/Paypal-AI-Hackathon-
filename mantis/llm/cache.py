"""The chat cache. Every agent, judge and compiler call goes through here.

Same discipline as :mod:`mantis.foundry.llm.cache` (HARD RULE 3), one level up:
that cache holds single-prompt completions for the content corpus; this one holds
**chat turns** — a message list, optional tools, optional JSON schema — because
the shopping agent is a tool-calling loop and the judge returns structured output.

The key
-------
``sha256(provider | model | messages | tools | response_format | params)`` over
canonical JSON, truncated to 32 hex characters. Every component matters: the same
conversation against a different model, a different tool list or a different
temperature is a different generation and must not collide.

For a multi-turn agent run, each turn is its own entry and its key covers the
**whole conversation so far** — including the tool results the previous turn
produced. Replay therefore reproduces the run exactly as long as the tools are
deterministic, which is why the agent's tools never put a timestamp, a random id
or a live PayPal order id into text the model reads.

The format
----------
One JSON file per turn under ``data/cache/<role>/`` (``agent``, ``judge``,
``compiler``), committed. Each records the request next to the response, the
wall-clock latency of the live call (replayed as the agent's deliberation
latency, so a cached run feeds the detector the same features as the live one),
and token usage. **No credential is ever written**: the provider object holds the
key, and the cache only ever sees the request body.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from mantis.core.paths import CACHE_DIR

__all__ = ["CHAT_CACHE_VERSION", "ChatCache", "ChatEntry", "chat_key", "role_cache_dir"]

#: Bump only when the *meaning* of a hit changes. ``--refresh`` regenerates.
CHAT_CACHE_VERSION: Final[str] = "1"


def role_cache_dir(role: str) -> Path:
    """``data/cache/agent``, ``data/cache/judge``, ``data/cache/compiler``."""
    return CACHE_DIR / role


def chat_key(
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    response_format: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """Deterministic key over the full request. Dict order cannot change it."""
    payload = json.dumps(
        {
            "v": CHAT_CACHE_VERSION,
            "provider": provider,
            "model": model,
            "messages": messages,
            "tools": tools or [],
            "response_format": response_format or {},
            "params": params or {},
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


@dataclass(slots=True)
class ChatEntry:
    """One cached chat turn and enough context to audit it."""

    key: str
    provider: str
    model: str
    request: dict[str, Any]
    #: ``{"content": str | None, "tool_calls": [{"id", "name", "arguments"}]}``
    response: dict[str, Any]
    latency_ms: float
    usage: dict[str, int] = field(default_factory=dict)
    recorded: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "provider": self.provider,
            "model": self.model,
            "recorded": self.recorded,
            "latency_ms": round(self.latency_ms, 1),
            "usage": self.usage,
            "request": self.request,
            "response": self.response,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> ChatEntry:
        return cls(
            key=str(raw["key"]),
            provider=str(raw["provider"]),
            model=str(raw["model"]),
            request=dict(raw.get("request", {})),
            response=dict(raw["response"]),
            latency_ms=float(raw.get("latency_ms", 0.0)),
            usage=dict(raw.get("usage", {})),
            recorded=str(raw.get("recorded", "")),
        )


@dataclass(slots=True)
class ChatCache:
    """A directory of chat turns. A corrupt file is a miss, never a crash."""

    directory: Path

    def path_for(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> ChatEntry | None:
        path = self.path_for(key)
        if not path.exists():
            return None
        try:
            return ChatEntry.from_json(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError):
            return None

    def put(self, entry: ChatEntry) -> None:
        """Atomic write: a crash mid-write leaves the old entry or none, never half."""
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.path_for(entry.key).with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(entry.to_json(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(self.path_for(entry.key))

    def keys(self) -> list[str]:
        if not self.directory.exists():
            return []
        return sorted(p.stem for p in self.directory.glob("*.json"))
