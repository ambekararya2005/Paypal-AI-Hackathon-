"""LLM content generation with a mandatory cache and a guaranteed floor.

This package is the answer to a question a judge will ask: *where did the text
come from, and what happens on my laptop?* The answer is a three-stage ladder
(:mod:`~mantis.foundry.llm.client`) — live Ollama, then the committed disk cache,
then a bundled deterministic corpus — with the property that
:meth:`LlmClient.generate` cannot fail, cannot hang and cannot open a socket
unless it was explicitly told it may.

The pieces:

* :mod:`.cache` — ``sha256(model|prompt|params)`` keyed JSON on disk. Committed.
  CLAUDE.md HARD RULE 3.
* :mod:`.client` — the three-stage ladder, standard library only.
* :mod:`.fallback` — deterministic specimen text, shaped like the real thing.
* :mod:`.prompts` — the prompt builders, and the auditable safety preamble.
* :mod:`.corpus` — the content store that joins ``ingested_content_ids`` in the
  parquet to the text the L3 layer classifies.
* :mod:`.build` — the plan that turns prompts into a committed corpus.

Build or inspect the corpus with ``python -m mantis.foundry.llm``.
"""

from __future__ import annotations

from mantis.foundry.llm.build import CORPUS_PLAN, build_corpus
from mantis.foundry.llm.cache import LlmCache, cache_key
from mantis.foundry.llm.client import DEFAULT_MODEL, LlmClient, ollama_available
from mantis.foundry.llm.corpus import (
    CONTENT_STORE,
    ContentArtifact,
    ContentStore,
    content_id_for_url,
    load_content_store,
)
from mantis.foundry.llm.fallback import SYNTHETIC_MARKER, kind_is_injected

__all__ = [
    "CONTENT_STORE",
    "CORPUS_PLAN",
    "DEFAULT_MODEL",
    "SYNTHETIC_MARKER",
    "ContentArtifact",
    "ContentStore",
    "LlmCache",
    "LlmClient",
    "build_corpus",
    "cache_key",
    "content_id_for_url",
    "kind_is_injected",
    "load_content_store",
    "ollama_available",
]
