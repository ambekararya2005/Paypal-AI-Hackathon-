"""The content store — what turns ``ingested_content_ids`` into text L3 can read.

The join
--------
``AgenticContext.provenance_chain`` is the ordered list of URLs an agent read
before it paid; ``ingested_content_ids`` is the parallel list of
``sha256:<12 hex>`` digests of those URLs. Both are in the parquet. Neither
contains a single character of the actual content, because an authorisation
message would not. This module is the side-channel that holds the content, keyed
by exactly those ids, and it is what makes the L3 layer a text classifier rather
than a thought experiment.

Why the store is not one artefact per URL
------------------------------------------
A 200,000-event population produces roughly 700,000 provenance URLs. Storing a
generation for each is absurd, and generating them is impossible. So resolution
is two-tier:

* **Explicit bindings** — written by an injector when it plants a specific
  payload on a specific URL. Small (one per injected page), and persisted.
* **Deterministic assignment** — everything else. A content id that has no
  explicit binding is hashed into the benign pool, so it *always* resolves, to
  the *same* artefact, on every machine, with nothing stored at all.

That second tier is not a convenience, it is a leakage control. If only attacked
content ids resolved to text, then "does this id resolve" would be a perfect
label, and the L3 metrics would be measuring our storage layout. Every id in the
dataset resolves; the classifier has to read the words.

Persistence
-----------
One JSONL file, committed under ``data/cache/content/``. Line-oriented so it
diffs, appends and merges cleanly, and so a corrupt line costs one artefact.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from mantis.core.paths import CACHE_DIR, ensure_dir
from mantis.foundry.llm.fallback import kind_is_injected

__all__ = [
    "CONTENT_STORE",
    "CORPUS_DIR",
    "ContentArtifact",
    "ContentStore",
    "content_id_for_url",
]

#: Committed alongside the LLM cache. See HARD RULE 3.
CORPUS_DIR: Final[Path] = CACHE_DIR / "content"

#: The bundled corpus itself, and the bindings written during a dataset run.
CORPUS_FILE: Final[str] = "corpus.jsonl"
BINDINGS_FILE: Final[str] = "bindings.jsonl"


def content_id_for_url(url: str) -> str:
    """The content id for a URL. **Must** match the simulator's convention.

    ``mantis.foundry.base.simulator._build_provenance`` writes
    ``sha256:<first 12 hex of sha256(url)>``, and so does
    ``injectors.base._repair_agentic``. If this drifts, every id in the parquet
    stops resolving and L3 silently sees an empty corpus — so the convention is
    named once, here, and the population test asserts the round trip.
    """
    return f"sha256:{hashlib.sha256(url.encode('utf-8')).hexdigest()[:12]}"


@dataclass(slots=True, frozen=True)
class ContentArtifact:
    """One piece of retrievable text, with the labels L3 will be scored against.

    ``injected`` is the L3 ground truth and is subject to the same discipline as
    every other label: it lives here, alongside the text, and never travels into
    a feature matrix. ``attack_id`` names the atlas card the specimen was written
    for, so per-card text recall is reportable.
    """

    artifact_id: str
    kind: str
    title: str
    text: str
    #: Which rung of the degradation ladder produced this text.
    source: str = "fallback"
    #: L3 ground truth. Derived from ``kind`` unless overridden.
    injected: bool | None = None
    attack_id: str | None = None

    @property
    def is_injected(self) -> bool:
        return kind_is_injected(self.kind) if self.injected is None else self.injected

    def to_json(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "title": self.title,
            "source": self.source,
            "injected": self.is_injected,
            "attack_id": self.attack_id,
            "text": self.text,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> ContentArtifact:
        return cls(
            artifact_id=str(raw["artifact_id"]),
            kind=str(raw["kind"]),
            title=str(raw.get("title", "")),
            text=str(raw["text"]),
            source=str(raw.get("source", "fallback")),
            injected=bool(raw["injected"]) if raw.get("injected") is not None else None,
            attack_id=raw.get("attack_id"),
        )


@dataclass(slots=True)
class ContentStore:
    """Artefact pool plus content-id bindings, with a deterministic fallback."""

    directory: Path = CORPUS_DIR
    artifacts: dict[str, ContentArtifact] = field(default_factory=dict)
    #: content_id -> artifact_id. Only for content that was *planted*.
    bindings: dict[str, str] = field(default_factory=dict)
    #: Benign artifact ids, ordered, for deterministic assignment.
    _benign: list[str] = field(default_factory=list, repr=False)

    # -- population ---------------------------------------------------------- #

    def add(self, artifact: ContentArtifact) -> ContentArtifact:
        """Add an artefact to the pool. Re-adding the same id is a no-op."""
        self.artifacts[artifact.artifact_id] = artifact
        self._benign = []
        return artifact

    def bind(self, content_id: str, artifact_id: str) -> None:
        """Record that a specific content id carries a specific artefact."""
        if artifact_id not in self.artifacts:
            raise KeyError(f"no artefact {artifact_id!r} in the store")
        self.bindings[content_id] = artifact_id

    def plant(self, url: str, artifact_id: str) -> str:
        """Bind an artefact to a URL and return the resulting content id."""
        content_id = content_id_for_url(url)
        self.bind(content_id, artifact_id)
        return content_id

    # -- resolution ------------------------------------------------------------ #

    @property
    def benign_pool(self) -> list[str]:
        """Ordered benign artefact ids. Sorted, so assignment is machine-stable."""
        if not self._benign:
            self._benign = sorted(
                a.artifact_id for a in self.artifacts.values() if not a.is_injected
            )
        return self._benign

    def resolve(self, content_id: str) -> ContentArtifact | None:
        """The text behind a content id. Always resolves when the pool is loaded.

        Explicit binding first; otherwise the id is hashed into the benign pool.
        See the module docstring: universal resolution is a leakage control, not
        a nicety.
        """
        artifact_id = self.bindings.get(content_id)
        if artifact_id is not None:
            return self.artifacts.get(artifact_id)
        pool = self.benign_pool
        if not pool:
            return None
        digest = content_id.split(":")[-1]
        try:
            index = int(digest[:8], 16)
        except ValueError:
            index = sum(map(ord, digest))
        return self.artifacts[pool[index % len(pool)]]

    def resolve_chain(self, content_ids: list[str]) -> list[ContentArtifact]:
        """Resolve a whole ``ingested_content_ids`` list, dropping misses."""
        out = [self.resolve(cid) for cid in content_ids]
        return [a for a in out if a is not None]

    # -- persistence ------------------------------------------------------------ #

    def write(self, directory: Path | None = None) -> Path:
        """Write the pool and the bindings. Returns the corpus path."""
        directory = ensure_dir(directory or self.directory)
        corpus = directory / CORPUS_FILE
        corpus.write_text(
            "".join(
                json.dumps(a.to_json(), ensure_ascii=False) + "\n"
                for a in sorted(self.artifacts.values(), key=lambda a: a.artifact_id)
            ),
            encoding="utf-8",
        )
        (directory / BINDINGS_FILE).write_text(
            "".join(
                json.dumps({"content_id": c, "artifact_id": a}, ensure_ascii=False) + "\n"
                for c, a in sorted(self.bindings.items())
            ),
            encoding="utf-8",
        )
        return corpus

    def load(self, directory: Path | None = None) -> ContentStore:
        """Load a committed corpus. Missing files are not an error."""
        directory = directory or self.directory
        corpus = directory / CORPUS_FILE
        if corpus.exists():
            for line in corpus.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    self.add(ContentArtifact.from_json(json.loads(line)))
                except (ValueError, KeyError):
                    continue  # one bad line must not cost the whole corpus
        bindings = directory / BINDINGS_FILE
        if bindings.exists():
            for line in bindings.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    self.bindings[str(raw["content_id"])] = str(raw["artifact_id"])
                except (ValueError, KeyError):
                    continue
        return self

    # -- reporting --------------------------------------------------------------- #

    def counts(self) -> dict[str, int]:
        """Artefacts per kind, plus the injected/benign split."""
        out: dict[str, int] = {}
        for artifact in self.artifacts.values():
            out[artifact.kind] = out.get(artifact.kind, 0) + 1
        out["_injected"] = sum(1 for a in self.artifacts.values() if a.is_injected)
        out["_benign"] = len(self.artifacts) - out["_injected"]
        out["_bindings"] = len(self.bindings)
        return out

    def pick(self, kind: str, index: int) -> ContentArtifact:
        """Deterministically pick the ``index``-th artefact of a kind.

        Injectors use this instead of an RNG draw so that the payload planted on
        a given attack event is stable across runs — the same reason every seed
        in this repo is derived rather than drawn.
        """
        pool = sorted(a.artifact_id for a in self.artifacts.values() if a.kind == kind)
        if not pool:
            raise KeyError(f"no artefacts of kind {kind!r}; build the corpus first")
        return self.artifacts[pool[index % len(pool)]]


#: The process-wide store. Injectors read it; the foundry CLI writes it out
#: alongside the parquet. A module-level singleton rather than a parameter
#: because ``BaseAttack.inject``'s signature is part of the injector contract and
#: threading a corpus through it would change every existing injector for the
#: benefit of two.
CONTENT_STORE: Final[ContentStore] = ContentStore()


def load_content_store(directory: Path | None = None) -> ContentStore:
    """Load the committed corpus into the process-wide store, once."""
    if not CONTENT_STORE.artifacts:
        CONTENT_STORE.load(directory)
    return CONTENT_STORE


def main() -> None:
    """Summarise the committed corpus. Run: ``python -m mantis.foundry.llm.corpus``."""
    store = ContentStore().load()
    print(f"MANTIS content corpus  ->  {store.directory}")
    if not store.artifacts:
        print("  (empty — run 'python -m mantis.foundry.llm' to build it)")
        return
    for kind, count in sorted(store.counts().items()):
        print(f"  {kind:<22} {count:>4}")
    print()
    sample = sorted(store.artifacts)[0]
    print(f"  sample artefact: {sample}")
    print(f"    {store.artifacts[sample].text[:200]}")


if __name__ == "__main__":
    main()
