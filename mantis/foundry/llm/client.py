"""Three-stage LLM client: live Ollama -> committed cache -> bundled fallback.

The contract
------------
:meth:`LlmClient.generate` **always returns text**. It never raises, never hangs
and never waits on a network that is not there. That is not defensive
programming for its own sake — it is CLAUDE.md HARD RULE 3 and HARD RULE 4 stated
as a method signature. A judge running ``make demo`` on a laptop in aeroplane
mode gets the same corpus as the machine that generated it.

The three stages, in the order they are tried
----------------------------------------------
1. **Cache.** Checked first, always. A hit is returned immediately even when a
   live model is available, because determinism beats freshness: the numbers on
   the slides must survive a re-run. ``refresh=True`` skips this stage, and is
   the only way to reach a live model for a prompt that has already been cached.
2. **Live Ollama**, and only if it was explicitly enabled — by ``allow_live`` or
   the ``MANTIS_LLM_LIVE`` environment variable. Disabled by default, so the
   default path opens no socket at all. A success writes through to the cache,
   which is how the committed corpus gets built in the first place.
3. **Fallback corpus.** Deterministic, seeded on the prompt hash, assembled from
   the templates in :mod:`mantis.foundry.llm.fallback`. Never a crash, never a
   placeholder string, never an empty completion.

Every reachability check is behind a short timeout and a blanket ``except
OSError``. The failure mode of "Ollama is installed but the model is not pulled"
is exactly as survivable as "Ollama is not installed".

Only the standard library is used for HTTP. Adding ``requests`` for one POST
would put a dependency in ``pyproject.toml`` that the demo path never executes.

Safety posture (HARD RULE 5)
-----------------------------
The prompts in :mod:`mantis.foundry.llm.prompts` ask for short, obviously
synthetic, clearly-labelled *specimens* of attack content, for the purpose of
training a detector to catch them. They do not ask for working exploits, they
name no real platform or merchant, and every generated payload is bounded in
length. The cache records the prompt next to the completion so that this claim
is auditable rather than asserted.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Final

from mantis.foundry.llm.cache import CacheEntry, LlmCache, cache_key
from mantis.foundry.llm.fallback import fallback_completion

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_PARAMS",
    "GenerationResult",
    "LlmClient",
    "ollama_available",
]

#: The model that authored the committed corpus, and therefore the default.
#:
#: This is not a quality ranking. The cache key is a hash of *(model, prompt,
#: params)*, so the default model has to be the one whose generations are in
#: ``data/cache/llm/`` — otherwise a judge on the default path gets a cache miss
#: on every prompt and silently falls through to the bundled corpus, and HARD
#: RULE 3 is satisfied in letter but not in substance. ``qwen2.5:7b`` and
#: ``llama3.1:8b`` both work; switching to one is a one-line change here plus
#: ``python -m mantis.foundry.llm --live --refresh`` to re-author and re-commit.
DEFAULT_MODEL: Final[str] = "mistral:7b-instruct-q4_K_M"

#: Low temperature: we want a *representative* specimen, not creative writing,
#: and a lower temperature makes a live regeneration closer to the cached text.
DEFAULT_PARAMS: Final[dict[str, Any]] = {"temperature": 0.6, "top_p": 0.9, "num_predict": 320}

#: Where a local Ollama listens. Overridable for a non-default port.
DEFAULT_HOST: Final[str] = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

#: Seconds. Deliberately short: a hung backend must not stall a demo.
PROBE_TIMEOUT: Final[float] = 1.5
GENERATE_TIMEOUT: Final[float] = 90.0

#: Set to a truthy value to permit live generation. Absent on a judge's machine,
#: which is the point — the default path never opens a socket.
LIVE_ENV_VAR: Final[str] = "MANTIS_LLM_LIVE"


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def ollama_available(host: str = DEFAULT_HOST, timeout: float = PROBE_TIMEOUT) -> bool:
    """Is a local Ollama answering? Never raises, never blocks for long."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as response:
            return bool(response.status == 200)
    except (urllib.error.URLError, OSError, ValueError):
        return False


@dataclass(slots=True, frozen=True)
class GenerationResult:
    """One completion plus, crucially, *where it came from*.

    The foundry prints the mix of sources on every run. A corpus that silently
    degraded to fallback would otherwise look identical to one a model wrote,
    and "we generated this with an LLM" is a claim we should only make when it
    is true for the run in front of the judge.
    """

    text: str
    source: str  # "ollama" | "cache" | "fallback"
    key: str
    model: str


@dataclass(slots=True)
class LlmClient:
    """Cache-first LLM client with a deterministic floor."""

    model: str = DEFAULT_MODEL
    params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    host: str = DEFAULT_HOST
    cache: LlmCache = field(default_factory=LlmCache)
    #: ``None`` means "read the environment". Explicit ``True``/``False`` wins.
    allow_live: bool | None = None
    #: Counts by source, for the run summary.
    stats: dict[str, int] = field(default_factory=dict)
    _live_checked: bool | None = field(default=None, repr=False)

    # -- capability ---------------------------------------------------------- #

    @property
    def live_enabled(self) -> bool:
        """Whether live generation is permitted at all. Off unless asked for."""
        if self.allow_live is not None:
            return self.allow_live
        return _truthy(os.environ.get(LIVE_ENV_VAR))

    def live_reachable(self) -> bool:
        """Whether a live model is both permitted and actually answering.

        Probed at most once per client: a judge with no Ollama should pay one
        1.5-second timeout for a whole corpus build, not one per prompt.
        """
        if not self.live_enabled:
            return False
        if self._live_checked is None:
            self._live_checked = ollama_available(self.host)
        return self._live_checked

    # -- the three stages ----------------------------------------------------- #

    def generate(self, prompt: str, *, kind: str = "generic", refresh: bool = False) -> str:
        """Return a completion. Always. See :meth:`generate_detailed`."""
        return self.generate_detailed(prompt, kind=kind, refresh=refresh).text

    def generate_detailed(
        self, prompt: str, *, kind: str = "generic", refresh: bool = False
    ) -> GenerationResult:
        """Run the degradation ladder and report which rung answered.

        Args:
            prompt: The full prompt. Hashed verbatim into the cache key.
            kind: Content kind, used to pick a fallback template family. It is
                deliberately **not** part of the cache key — the prompt already
                determines the output, and folding the kind in would fork the
                cache whenever a caller relabelled the same prompt.
            refresh: Skip the cache read and force a live call if one is
                available. The only route to a fresh generation.
        """
        key = cache_key(prompt, self.model, self.params)

        if not refresh:
            hit = self.cache.get(key)
            if hit is not None:
                self._count("cache")
                return GenerationResult(hit.completion, "cache", key, hit.model)

        if self.live_reachable():
            text = self._call_ollama(prompt)
            if text:
                self.cache.put(
                    CacheEntry(
                        key=key,
                        model=self.model,
                        prompt=prompt,
                        params=dict(self.params),
                        completion=text,
                        source="ollama",
                    )
                )
                self._count("ollama")
                return GenerationResult(text, "ollama", key, self.model)

        # Stage 3. Deterministic on the key, so the same prompt always yields
        # the same specimen whether or not anyone ever runs a model.
        text = fallback_completion(kind, key)
        self._count("fallback")
        return GenerationResult(text, "fallback", key, "fallback")

    # -- stage 2 internals ----------------------------------------------------- #

    def _call_ollama(self, prompt: str) -> str:
        """POST to ``/api/generate``. Returns ``""`` on any failure at all."""
        body = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": dict(self.params),
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=GENERATE_TIMEOUT) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
            # One failure disables live for the rest of the run. A model that is
            # not pulled fails identically on every subsequent prompt, and
            # paying the timeout hundreds of times would turn a fast fallback
            # into a hang.
            self._live_checked = False
            return ""
        return str(payload.get("response", "")).strip()

    def _count(self, source: str) -> None:
        self.stats[source] = self.stats.get(source, 0) + 1

    # -- reporting -------------------------------------------------------------- #

    def summary(self) -> str:
        """One line naming where this run's text actually came from."""
        if not self.stats:
            return "no generations"
        parts = ", ".join(f"{v} {k}" for k, v in sorted(self.stats.items()))
        if self.live_reachable():
            mode = "live"
        elif not self.live_enabled:
            mode = "cache-only"
        else:
            mode = "live requested, unreachable"
        return f"{parts}  [{mode}]"


def main() -> None:
    """Report client capability. Run: ``python -m mantis.foundry.llm.client``."""
    client = LlmClient()
    print("MANTIS LLM client")
    print(f"  model          : {client.model}")
    print(f"  host           : {client.host}")
    print(f"  live permitted : {client.live_enabled}   (set {LIVE_ENV_VAR}=1 to enable)")
    print(f"  live reachable : {client.live_reachable()}")
    print(f"  cache entries  : {len(client.cache)}  ->  {client.cache.directory}")
    print()
    result = client.generate_detailed("MANTIS self-test: reply with the word OK.", kind="generic")
    print(f"  self-test source : {result.source}")
    print(f"  self-test text   : {result.text[:120]}")


if __name__ == "__main__":
    main()
