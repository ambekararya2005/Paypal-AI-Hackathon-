"""Provider-generic chat client: committed cache first, a live provider only if allowed.

    python -m mantis.llm              # resolved provider, models, cache counts; no network
    python -m mantis.llm --probe      # one 5-token call per role (live; spends tokens)

The contract
------------
:meth:`LlmChat.chat` returns a :class:`ChatResult` from the **cache** when the
request has been seen, in either mode. In ``mock`` mode a miss raises
:class:`CacheMissError` immediately — no socket is opened, nothing waits, and the
caller decides what a miss means (the judge treats it as "no opinion"; the agent
demo reports the scenario as not recorded). In ``live`` mode a miss goes to the
provider and the answer is written through to the committed cache, which is how
the cache gets built.

The provider SDK is imported **only on a live miss**. A clean clone replays the
whole agent run with neither a key nor the ``openai`` package installed.

Providers
---------
``fastrouter`` and ``openai`` share :class:`OpenAICompatibleProvider` (FastRouter
is OpenAI-compatible; only the base URL and key differ). ``anthropic`` and
``gemini`` are named in the interface and refused with a clear message until a key
for them exists to test against — an untested provider would be a demo-day
surprise, not a feature.

Settings (environment first, then ``.env``)
---------------------------------------------
``LLM_PROVIDER``; ``LLM_MODEL_AGENT`` / ``LLM_MODEL_JUDGE`` / ``LLM_MODEL_COMPILER``;
``FASTROUTER_API_KEY`` + ``FASTROUTER_BASE_URL``, or ``OPENAI_API_KEY``.

Determinism
-----------
``temperature=0`` and ``seed=1337`` are sent on every call. Some models ignore
one or both (the GPT-5 family may ignore temperature). That cannot break
replay, because replay never asks the model: the cache *is* the determinism.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, Protocol

from mantis.core.env import setting
from mantis.llm.cache import ChatCache, ChatEntry, chat_key, role_cache_dir

__all__ = [
    "DEFAULT_PARAMS",
    "CacheMissError",
    "ChatResult",
    "LlmChat",
    "LlmConfigError",
    "ToolCall",
    "resolve_settings",
]

Role = Literal["agent", "judge", "compiler"]
LlmMode = Literal["mock", "live"]

#: Sent on every call unless the caller overrides. See "Determinism" above.
DEFAULT_PARAMS: Final[dict[str, Any]] = {"temperature": 0, "seed": 1337, "max_tokens": 1500}

_DEFAULT_MODELS: Final[dict[str, str]] = {
    # Must match the model the committed agent cache was recorded with: a clean
    # clone has no .env, so this default *is* the replay key. It said gpt-5-mini
    # after the Day 2 switch and replay missed on every turn (caught on 3.12).
    "agent": "openai/gpt-4.1-mini",
    "judge": "anthropic/claude-sonnet-4.6",
    "compiler": "anthropic/claude-sonnet-4.6",
}
_FASTROUTER_URL: Final[str] = "https://api.fastrouter.ai/api/v1"
_TIMEOUT_S: Final[float] = 90.0


class LlmConfigError(RuntimeError):
    """Live mode was asked for and the provider cannot be reached as configured."""


class CacheMissError(LookupError):
    """Mock mode, and this exact request was never recorded."""

    def __init__(self, role: str, key: str) -> None:
        self.role = role
        self.key = key
        super().__init__(
            f"no cached {role} response for request {key} (mock mode never calls a model). "
            f"Record it with --mode live."
        )


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    #: The raw JSON argument string, exactly as the model produced it.
    arguments: str

    def parsed(self) -> dict[str, Any]:
        try:
            value = json.loads(self.arguments or "{}")
        except ValueError:
            return {}
        return value if isinstance(value, dict) else {}


@dataclass(slots=True)
class ChatResult:
    content: str | None
    tool_calls: list[ToolCall]
    #: The live call's wall-clock latency; replayed verbatim from the cache.
    latency_ms: float
    source: Literal["cache", "live"]
    model: str
    key: str
    usage: dict[str, int] = field(default_factory=dict)

    def assistant_message(self) -> dict[str, Any]:
        """This turn as an OpenAI-format assistant message, for the next request."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.name, "arguments": c.arguments}}
                for c in self.tool_calls
            ]
        return message


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #


class Provider(Protocol):
    name: str

    def complete(
        self, model: str, request: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, int]]:
        """``(response, usage)``. ``response`` is ``{"content", "tool_calls"}``."""
        ...


class OpenAICompatibleProvider:
    """OpenAI Chat Completions, at OpenAI or any compatible router (FastRouter)."""

    def __init__(self, name: str, api_key: str, base_url: str | None) -> None:
        if not api_key:
            raise LlmConfigError(f"LLM_PROVIDER={name} needs an API key in .env")
        self.name = name
        self._api_key = api_key
        self._base_url = base_url
        self._client: Any = None  # Any: the openai SDK client, imported lazily

    def _sdk(self) -> Any:  # Any: see above
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise LlmConfigError(
                    "live LLM calls need the 'openai' package: pip install -e \".[agent]\""
                ) from exc
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url,
                                  timeout=_TIMEOUT_S, max_retries=1)
        return self._client

    def complete(
        self, model: str, request: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, int]]:
        kwargs: dict[str, Any] = {"model": model, "messages": request["messages"],
                                  **request["params"]}
        if request.get("tools"):
            kwargs["tools"] = request["tools"]
        if request.get("response_format"):
            kwargs["response_format"] = request["response_format"]
        out = self._sdk().chat.completions.create(**kwargs)
        message = out.choices[0].message
        response = {
            "content": message.content,
            "tool_calls": [
                {"id": c.id, "name": c.function.name, "arguments": c.function.arguments}
                for c in (message.tool_calls or [])
            ],
            "finish_reason": out.choices[0].finish_reason,
        }
        usage = {}
        if out.usage is not None:
            usage = {"prompt_tokens": int(out.usage.prompt_tokens or 0),
                     "completion_tokens": int(out.usage.completion_tokens or 0)}
        return response, usage


@dataclass(frozen=True, slots=True)
class Settings:
    provider: str
    models: dict[str, str]
    api_key: str
    base_url: str | None

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)


def resolve_settings() -> Settings:
    """Provider, per-role models and credentials. Never logs a value."""
    provider = setting("LLM_PROVIDER", "fastrouter").strip().lower()
    models = {role: setting(f"LLM_MODEL_{role.upper()}", default)
              for role, default in _DEFAULT_MODELS.items()}
    if provider == "fastrouter":
        return Settings(provider, models, setting("FASTROUTER_API_KEY"),
                        setting("FASTROUTER_BASE_URL", _FASTROUTER_URL))
    if provider == "openai":
        return Settings(provider, models, setting("OPENAI_API_KEY"),
                        setting("OPENAI_BASE_URL") or None)
    return Settings(provider, models, "", None)


def _make_provider(settings: Settings) -> Provider:
    if settings.provider in ("fastrouter", "openai"):
        return OpenAICompatibleProvider(settings.provider, settings.api_key, settings.base_url)
    raise LlmConfigError(
        f"LLM_PROVIDER={settings.provider!r} is not implemented yet. Implemented: fastrouter, "
        "openai. Adding one means a class with complete(model, request) -> (response, usage)."
    )


# --------------------------------------------------------------------------- #
# The client
# --------------------------------------------------------------------------- #


class LlmChat:
    """One role's chat client. Cache first; the provider only in live mode on a miss."""

    def __init__(self, role: Role, *, mode: LlmMode = "mock", refresh: bool = False,
                 cache_dir: Path | None = None, settings: Settings | None = None,
                 provider: Provider | None = None) -> None:
        self.role = role
        self.mode = mode
        self.refresh = refresh
        self.settings = settings or resolve_settings()
        self.model = self.settings.models[role]
        self.cache = ChatCache(cache_dir or role_cache_dir(role))
        self._provider = provider
        #: Per-instance accounting, so a demo can say how much was replayed.
        self.calls = {"cache": 0, "live": 0}

    @property
    def provider_name(self) -> str:
        return self._provider.name if self._provider is not None else self.settings.provider

    def chat(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None,
             response_format: dict[str, Any] | None = None,
             **params: Any) -> ChatResult:  # Any: provider pass-through parameters
        merged = {**DEFAULT_PARAMS, **params}
        messages = copy.deepcopy(messages)
        key = chat_key(self.provider_name, self.model, messages, tools=tools,
                       response_format=response_format, params=merged)

        if not self.refresh:
            hit = self.cache.get(key)
            if hit is not None:
                self.calls["cache"] += 1
                return self._result(hit, "cache")
        if self.mode != "live":
            raise CacheMissError(self.role, key)

        from mantis.core.quota import spend

        spend("llm")  # kill switch + daily cap on the public deployment; raises if refused
        provider = self._provider or _make_provider(self.settings)
        self._provider = provider
        request = {"messages": messages, "tools": tools or [],
                   "response_format": response_format or {}, "params": merged}
        started = time.perf_counter()
        response, usage = provider.complete(self.model, request)
        entry = ChatEntry(
            key=key, provider=provider.name, model=self.model, request=request,
            response=response, latency_ms=(time.perf_counter() - started) * 1000.0,
            usage=usage, recorded=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        self.cache.put(entry)
        self.calls["live"] += 1
        return self._result(entry, "live")

    @staticmethod
    def _result(entry: ChatEntry, source: Literal["cache", "live"]) -> ChatResult:
        return ChatResult(
            content=entry.response.get("content"),
            tool_calls=[ToolCall(str(c["id"]), str(c["name"]), str(c.get("arguments") or "{}"))
                        for c in entry.response.get("tool_calls") or []],
            latency_ms=entry.latency_ms,
            source=source,
            model=entry.model,
            key=entry.key,
            usage=dict(entry.usage),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mantis.llm", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", action="store_true",
                        help="live: one tiny call per role (spends a few tokens; not cached)")
    args = parser.parse_args(argv)

    s = resolve_settings()
    print("MANTIS LLM layer")
    print(f"  provider : {s.provider}  (base url {s.base_url or 'default'})")
    print(f"  api key  : {'set' if s.has_key else 'not set'}  (never printed)")
    for role, model in s.models.items():
        n = len(ChatCache(role_cache_dir(role)).keys())
        print(f"  {role:<9}: {model:<34} cached turns: {n}")
    if not args.probe:
        return 0
    try:
        provider = _make_provider(s)
    except LlmConfigError as exc:
        print(f"probe: {exc}", file=sys.stderr)
        return 2
    for role, model in s.models.items():
        started = time.perf_counter()
        try:
            response, _ = provider.complete(model, {
                "messages": [{"role": "user", "content": "Reply with the word OK."}],
                "params": {"max_tokens": 5}})
            print(f"  probe {role:<9} OK  {(time.perf_counter() - started) * 1000:6.0f} ms "
                  f"-> {str(response.get('content'))[:20]!r}")
        except Exception as exc:
            print(f"  probe {role:<9} FAIL {type(exc).__name__}: {str(exc)[:120]}")
    return 0
