"""Build the content corpus: prompts -> client -> artefacts -> committed JSONL.

This is the module that actually produces ``data/cache/content/corpus.jsonl``.
Run it once with a live Ollama to author the corpus, commit the result, and every
subsequent run — on a judge's laptop, in CI, at 3am with no network — reproduces
it exactly from the committed files.

The composition is deliberate
------------------------------
Roughly two-thirds of the corpus is **benign**. That is not padding. The L3 layer
is a classifier, and a corpus that was 90% attack content would let it score well
by answering "yes" — the same failure mode as reporting accuracy on a 0.64%
prevalence dataset. The benign pool also backs the deterministic assignment in
:class:`~mantis.foundry.llm.corpus.ContentStore`, so that *every* content id in
the parquet resolves to text and "does it resolve" cannot become the label.

Artefact ids are ``<kind>-<nnn>``, assigned in generation order. They are stable
because the generation order is a fixed product of ``(kind, category, variant)``
with no randomness anywhere in it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from mantis.foundry.llm import prompts
from mantis.foundry.llm.client import LlmClient
from mantis.foundry.llm.corpus import ContentArtifact, ContentStore

__all__ = ["CORPUS_PLAN", "CorpusSpec", "build_corpus"]

#: The categories every kind is generated across. Keeping one list means the
#: benign and adversarial halves cover the same ground, so a classifier cannot
#: separate them on subject matter instead of on register.
_CATEGORIES: Final[tuple[str, ...]] = (
    "home appliances",
    "personal care",
    "travel gear",
    "kitchenware",
    "consumer electronics",
    "groceries",
)

#: Scam themes. Named as fraud *patterns*, which is how the atlas names them.
_SCAM_THEMES: Final[tuple[str, ...]] = (
    "digital arrest",
    "law-enforcement impersonation",
    "bank-support impersonation",
    "refund reversal",
)


@dataclass(slots=True, frozen=True)
class CorpusSpec:
    """One kind of content: how to prompt for it and how much of it to make."""

    kind: str
    builder: Callable[[str, int], str]
    subjects: tuple[str, ...]
    variants: int


#: The corpus plan. 6 categories x variants per kind. Sized so a live build is a
#: few minutes on a laptop 7B, and the committed JSONL stays well under a
#: megabyte — small enough to belong in git, which HARD RULE 3 requires.
CORPUS_PLAN: Final[tuple[CorpusSpec, ...]] = (
    # -- benign: the negative class, and the deterministic-assignment pool ----- #
    CorpusSpec("benign_page", prompts.benign_page_prompt, _CATEGORIES, 12),
    CorpusSpec("benign_transcript", prompts.benign_transcript_prompt, _CATEGORIES, 6),
    CorpusSpec("merchant_copy", prompts.merchant_copy_prompt, _CATEGORIES, 5),
    # -- adversarial: specimens of published attack classes -------------------- #
    CorpusSpec("injected_page", prompts.injected_page_prompt, _CATEGORIES, 4),
    CorpusSpec("injected_review", prompts.injected_review_prompt, _CATEGORIES, 2),
    CorpusSpec("agent_transcript", prompts.agent_transcript_prompt, _CATEGORIES, 3),
    CorpusSpec("refund_ticket", prompts.refund_ticket_prompt, _CATEGORIES, 3),
    CorpusSpec("scam_script", prompts.scam_script_prompt, _SCAM_THEMES, 3),
    CorpusSpec("shell_merchant_copy", prompts.shell_merchant_copy_prompt, _CATEGORIES, 2),
)


def build_corpus(
    client: LlmClient | None = None,
    store: ContentStore | None = None,
    *,
    refresh: bool = False,
    plan: tuple[CorpusSpec, ...] = CORPUS_PLAN,
) -> ContentStore:
    """Generate every artefact in ``plan`` and return the populated store.

    Args:
        client: The LLM client. Defaults to a cache-first one, which means this
            function is safe to call on any machine at any time.
        store: Store to fill. Defaults to a fresh one.
        refresh: Force live regeneration of every prompt. Only reaches a model
            if one is both permitted and reachable.
        plan: What to generate.
    """
    client = client or LlmClient()
    store = store or ContentStore()

    for spec in plan:
        index = 0
        for subject in spec.subjects:
            for variant in range(spec.variants):
                prompt = spec.builder(subject, variant)
                result = client.generate_detailed(prompt, kind=spec.kind, refresh=refresh)
                store.add(
                    ContentArtifact(
                        artifact_id=f"{spec.kind}-{index:03d}",
                        kind=spec.kind,
                        title=f"{subject} ({spec.kind} v{variant})",
                        text=result.text,
                        source=result.source,
                    )
                )
                index += 1
    return store


def main() -> None:
    """Not the entry point — see ``python -m mantis.foundry.llm``."""
    from mantis.foundry.llm.__main__ import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
