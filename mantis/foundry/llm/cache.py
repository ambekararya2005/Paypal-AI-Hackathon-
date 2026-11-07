"""The mandatory disk cache. Every LLM call in MANTIS goes through here.

Why this is a hard rule and not an optimisation
------------------------------------------------
CLAUDE.md HARD RULE 3: a judge cloning this repo will not have Ollama running,
will not have a GPU, and may not have a network. The demo must still produce
identical text to the one on the slides. So the cache is not a speed-up bolted
onto a live client — it is the **primary** source of truth, and the live model is
the thing that populates it during development.

The key
-------
``sha256(model | prompt | sorted params)``, truncated to 16 hex characters. All
three components matter: the same prompt against a different model, or the same
model at a different temperature, is a different generation and must not collide.
Truncation to 64 bits is safe here because the keyspace is a few hundred entries
authored by us, not an adversarial input.

The format
----------
One JSON file per entry under ``data/cache/llm/``, committed to git. One file per
entry rather than a single bundle for three reasons that matter at 3am: a git
diff shows exactly which generation changed, two branches adding different
prompts merge without conflict, and a corrupt entry costs one generation instead
of the whole corpus.

Entries record the prompt and parameters alongside the completion. That makes the
cache auditable: anyone can see what was asked as well as what came back, which
matters for HARD RULE 5 — the injected payloads in this corpus have to be
demonstrably short, synthetic and defensive, and you cannot demonstrate that from
the output alone.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from mantis.core.paths import CACHE_DIR, ensure_dir

__all__ = ["CACHE_VERSION", "LLM_CACHE_DIR", "CacheEntry", "LlmCache", "cache_key"]

#: Bumping this invalidates every entry. Only bump when the *meaning* of a cache
#: hit changes — never to force a regeneration, which ``--refresh`` already does.
CACHE_VERSION: Final[str] = "1"

#: Committed. See HARD RULE 3.
LLM_CACHE_DIR: Final[Path] = CACHE_DIR / "llm"


def cache_key(prompt: str, model: str, params: dict[str, Any]) -> str:
    """Deterministic 16-hex key over the full request.

    Parameters are serialised with sorted keys so that dict ordering — which
    Python preserves but authors do not — cannot produce two keys for one
    request and silently double the corpus.
    """
    payload = json.dumps(
        {"v": CACHE_VERSION, "model": model, "prompt": prompt, "params": params},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(slots=True)
class CacheEntry:
    """One cached generation, plus enough context to audit it."""

    key: str
    model: str
    prompt: str
    params: dict[str, Any]
    completion: str
    #: Where the text actually came from. ``ollama`` means a live model produced
    #: it; ``fallback`` means the bundled corpus did. Recorded so the foundry can
    #: report honestly which mode a run was in rather than implying a model ran.
    source: str = "ollama"

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "model": self.model,
            "source": self.source,
            "params": self.params,
            "prompt": self.prompt,
            "completion": self.completion,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> CacheEntry:
        return cls(
            key=str(raw["key"]),
            model=str(raw["model"]),
            prompt=str(raw["prompt"]),
            params=dict(raw.get("params", {})),
            completion=str(raw["completion"]),
            source=str(raw.get("source", "ollama")),
        )


@dataclass(slots=True)
class LlmCache:
    """A directory of JSON generations, loaded lazily and written atomically."""

    directory: Path = LLM_CACHE_DIR
    _memo: dict[str, CacheEntry] = field(default_factory=dict, repr=False)

    def path_for(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> CacheEntry | None:
        """Return the cached entry, or ``None``. Never raises on a bad file.

        A corrupt entry is treated as a miss rather than an error: the fallback
        corpus is right behind it, and a judge's demo must not die because one
        JSON file lost a brace.
        """
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        path = self.path_for(key)
        if not path.exists():
            return None
        try:
            entry = CacheEntry.from_json(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError):
            return None
        self._memo[key] = entry
        return entry

    def put(self, entry: CacheEntry) -> None:
        """Write an entry, atomically, and memoise it."""
        ensure_dir(self.directory)
        self._memo[entry.key] = entry
        tmp = self.path_for(entry.key).with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(entry.to_json(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        tmp.replace(self.path_for(entry.key))

    def __len__(self) -> int:
        return len(list(self.directory.glob("*.json"))) if self.directory.exists() else 0

    def keys(self) -> list[str]:
        if not self.directory.exists():
            return []
        return sorted(p.stem for p in self.directory.glob("*.json"))


def main() -> None:
    """Report the committed cache. Run: ``python -m mantis.foundry.llm.cache``."""
    cache = LlmCache()
    keys = cache.keys()
    print(f"MANTIS LLM cache  ->  {cache.directory}")
    print(f"  entries : {len(keys)}")
    if not keys:
        print("  (empty — run 'python -m mantis.foundry.llm' to build the corpus)")
        return
    by_source: dict[str, int] = {}
    for key in keys:
        entry = cache.get(key)
        if entry is not None:
            by_source[entry.source] = by_source.get(entry.source, 0) + 1
    for source, count in sorted(by_source.items()):
        print(f"  {source:<10} {count:>4}")


if __name__ == "__main__":
    main()
