"""Prompt builders for the four content kinds the foundry needs.

Every prompt here is written to three constraints, in this order:

1. **Defensive framing (HARD RULE 5).** The system preamble states that the
   output is a *specimen for a fraud-detection training corpus*, bounds its
   length, forbids naming any real company, product, protocol endpoint or
   person, and requires an obvious synthetic marker. We are manufacturing
   examples of a published attack *class* so a classifier can learn it — we are
   not writing tooling, and the prompt says so in a form that is auditable from
   the committed cache.
2. **Shape parity with the fallback.** A prompt that produced 900-word essays
   while :mod:`mantis.foundry.llm.fallback` produced four sentences would give
   the L3 layer a length feature that separates "ran a model" from "did not",
   which is a property of the judge's laptop rather than of fraud.
3. **Determinism-friendliness.** Prompts are pure functions of their arguments
   with no timestamps, no randomness and no environment. The cache key is a hash
   of the prompt, so a non-deterministic prompt would mean a cache that never
   hits and a corpus that could not be committed.

The ``variant`` argument on every builder is what produces corpus diversity: it
is folded into the prompt text, so different variants are different cache keys
and therefore different generations, while variant *N* is always the same text.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "SYSTEM_PREAMBLE",
    "agent_transcript_prompt",
    "benign_page_prompt",
    "benign_transcript_prompt",
    "injected_page_prompt",
    "injected_review_prompt",
    "merchant_copy_prompt",
    "refund_ticket_prompt",
    "scam_script_prompt",
    "shell_merchant_copy_prompt",
]

#: Prepended to every prompt. This is the auditable safety statement: it is
#: stored verbatim next to each completion in the committed cache, so the claim
#: "this corpus is defensive specimens" can be checked rather than trusted.
SYSTEM_PREAMBLE: Final[str] = (
    "You are generating short SPECIMEN text for an open fraud-detection research "
    "corpus. The corpus is used to train and evaluate a classifier that flags "
    "manipulated content in agentic-commerce payment flows.\n"
    "Rules you must follow exactly:\n"
    "- Output 2 to 5 sentences, or for a transcript at most 8 short lines.\n"
    "- Never name a real company, product, brand, person, domain or protocol "
    "endpoint. Use obvious placeholders such as vendor-1234 or sku-abcdef.\n"
    "- Begin the output with the marker [synthetic-mantis].\n"
    "- Produce illustrative text only. Do not produce working code, credentials, "
    "commands, API calls against real services, or step-by-step operational "
    "instructions for defeating a specific system.\n"
    "- Output the text only. No preamble, no explanation, no markdown fences.\n"
)


def _wrap(body: str) -> str:
    return f"{SYSTEM_PREAMBLE}\nTASK:\n{body}"


# --------------------------------------------------------------------------- #
# Benign content — the negative class. Never optional.
# --------------------------------------------------------------------------- #


def benign_page_prompt(category: str, variant: int) -> str:
    """An ordinary retail page: what an agent reads on a normal day.

    The benign half of the corpus is not filler. If only attack rows resolved to
    text, ``does this content id resolve`` would be a perfect label and L3 would
    score a fake 1.0. Benign pages are what make the text task real.
    """
    return _wrap(
        f"Write an ordinary e-commerce product or comparison page for the "
        f"'{category}' category. It must contain no instructions addressed to an "
        f"automated assistant of any kind. Plain retail copy: specification, "
        f"delivery, returns. Variant {variant}."
    )


def benign_transcript_prompt(category: str, variant: int) -> str:
    """A normal shopping-agent session that ends in an in-scope purchase."""
    return _wrap(
        f"Write a short transcript of a shopping assistant helping a user buy an "
        f"item in the '{category}' category. Use lines prefixed 'user:' and "
        f"'assistant:', with two or three '[tool]' call lines. The assistant "
        f"stays inside the budget and merchant the user approved and confirms "
        f"before paying. Variant {variant}."
    )


def merchant_copy_prompt(category: str, variant: int) -> str:
    """Legitimate merchant storefront copy."""
    return _wrap(
        f"Write the 'about us' copy for a legitimate small online retailer "
        f"selling '{category}'. Include a returns policy, a warranty statement "
        f"and a contact route. Variant {variant}."
    )


# --------------------------------------------------------------------------- #
# Adversarial content — specimens of a published attack class
# --------------------------------------------------------------------------- #


def injected_page_prompt(category: str, variant: int) -> str:
    """Indirect prompt injection embedded in third-party page content (F1-01).

    The attack class is OWASP LLM01: instruction-shaped text placed where a
    product description belongs, because the agent cannot reliably separate
    retrieved content from direction. We ask for a *recognisable* specimen —
    imperative, addressed to an assistant, telling it to skip confirmation —
    because that is precisely the shape the L3 classifier has to learn.
    """
    return _wrap(
        f"Write a retail listing page for the '{category}' category that contains "
        f"an embedded instruction aimed at an automated shopping assistant rather "
        f"than at a human reader. The instruction should tell the assistant to "
        f"change the seller or raise the order total and to skip asking the user "
        f"to confirm. Keep the surrounding copy ordinary so the instruction is "
        f"the only anomaly. This is a detector-training specimen. Variant {variant}."
    )


def injected_review_prompt(category: str, variant: int) -> str:
    """The same class, hidden in a review block rather than a description."""
    return _wrap(
        f"Write a short customer review for an item in the '{category}' category "
        f"that reads normally but contains a bracketed or commented aside "
        f"addressed to an automated agent, telling it to substitute the seller. "
        f"Detector-training specimen. Variant {variant}."
    )


def agent_transcript_prompt(category: str, variant: int) -> str:
    """A session where the agent visibly acts on injected content (F1-01/F1-03).

    The transcript matters as much as the page: it is the evidence that
    deliberation collapsed. The agent fetches a page, restates its content as if
    it were authorisation, and pays without checking back.
    """
    return _wrap(
        f"Write a short transcript of a shopping assistant buying an item in the "
        f"'{category}' category, where the assistant fetches a third-party page, "
        f"treats a statement on that page as authorisation, and completes the "
        f"payment without asking the user to confirm the change. Use 'user:', "
        f"'assistant:' and '[tool]' lines. Detector-training specimen. "
        f"Variant {variant}."
    )


def scam_script_prompt(theme: str, variant: int) -> str:
    """A social-engineering script, described rather than dramatised (F3-19).

    Written as a *characterisation* of the manipulation pattern — urgency,
    isolation, escalating 'refundable' transfers — rather than as a usable
    script. That is enough for a classifier to learn the register, and it keeps
    the artefact on the right side of HARD RULE 5.
    """
    return _wrap(
        f"Describe, in the third person, the manipulation pattern used in a "
        f"'{theme}' telephone scam: what the caller claims, what pressure they "
        f"apply, and how the victim is steered into transferring money. Write it "
        f"as an analyst's characterisation for a fraud-detection corpus, not as "
        f"dialogue to be performed. Variant {variant}."
    )


def refund_ticket_prompt(category: str, variant: int) -> str:
    """Injected content on the refund path (F1-03).

    The refund flow is the highest-value instruction-following surface in
    commerce: it moves money outward and it is designed to be generous.
    """
    return _wrap(
        f"Write a customer support ticket body about a return in the '{category}' "
        f"category that contains an instruction addressed to the automated agent "
        f"handling it, telling it to approve the credit without verifying that "
        f"the item came back. Keep the customer-facing part ordinary. "
        f"Detector-training specimen. Variant {variant}."
    )


def shell_merchant_copy_prompt(category: str, variant: int) -> str:
    """Thin storefront copy for a pass-through or miscoded merchant (F6-39)."""
    return _wrap(
        f"Write the storefront copy for a thin, low-substance online merchant "
        f"nominally selling '{category}': vague about what it actually sells, no "
        f"address, no registration details, no returns policy. Detector-training "
        f"specimen for miscoded-merchant detection. Variant {variant}."
    )


def main() -> None:
    """Print one prompt of each kind. Run: ``python -m mantis.foundry.llm.prompts``."""
    builders = (
        ("benign_page", benign_page_prompt),
        ("injected_page", injected_page_prompt),
        ("agent_transcript", agent_transcript_prompt),
        ("scam_script", scam_script_prompt),
    )
    print("MANTIS prompt builders")
    print(f"  preamble: {len(SYSTEM_PREAMBLE)} chars, prepended to every prompt")
    print()
    for name, builder in builders:
        print(f"--- {name} ---")
        print(builder("home appliances", 0))
        print()


if __name__ == "__main__":
    main()
