"""Plain English -> a signed-ready mandate, or a question back. Never a guess.

    python -m mantis.agent.mandate_compiler "groceries this week, max $50/order, $150 total,
        only FreshMart"                                                # one sentence
    python -m mantis.agent.mandate_compiler --mode live "..."      # record a new sentence

The human says what they want once. The compiler turns it into the
:class:`~mantis.paypal.models.MandateRequest` the gate signs, and echoes it back as
"Agent CAN … / Agent CANNOT …" so the person approves the *meaning*, not JSON.

Two layers, and the second one is the point
-------------------------------------------
1. **The model** (role ``compiler``, Claude Sonnet 4.6, cached under
   ``data/cache/compiler/``) reads the sentence and fills a strict JSON schema.
   Categories and stores are **enums**: the model picks ``groceries`` or
   ``FreshMart``, and the code maps those to MCC ``5411`` and ``m-freshmart``. A
   model never writes an MCC or a merchant id, so it cannot invent one.
2. **Deterministic validation**, after the model, decides whether there is
   enough to sign. No per-order limit, no category, no duration, or a total
   budget smaller than one order: the result is ``needs_clarification`` with a
   question, whatever the model thought. A mandate is an authorisation to spend;
   "the model filled in something plausible" is not consent.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Final, Literal

from mantis.core.events import MandateScope, MandateType
from mantis.llm.client import CacheMissError, LlmChat, LlmConfigError
from mantis.paypal.models import MandateRequest

__all__ = ["CATEGORY_MCC", "MERCHANT_IDS", "CompileResult", "MandateCompiler"]

#: The categories the store and the gate understand, and their MCCs.
CATEGORY_MCC: Final[dict[str, str]] = {"groceries": "5411", "gift_cards": "6540"}
#: The stores a mandate may name, and their merchant ids.
MERCHANT_IDS: Final[dict[str, str]] = {
    "FreshMart": "m-freshmart",
    "GreenLeaf Market": "m-greenleaf",
    "GiftHub": "m-gifthub",
}

_SCHEMA: Final[dict[str, Any]] = {
    "type": "json_schema",
    "json_schema": {
        "name": "mandate",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "status": {"type": "string", "enum": ["ok", "needs_clarification"]},
                "questions": {"type": "array", "items": {"type": "string"}},
                "categories": {"type": "array",
                               "items": {"type": "string", "enum": sorted(CATEGORY_MCC)}},
                "merchants": {"type": "array",
                              "items": {"type": "string", "enum": sorted(MERCHANT_IDS)}},
                "max_per_order": {"type": ["number", "null"]},
                "budget_total": {"type": ["number", "null"]},
                "duration_days": {"type": ["integer", "null"]},
            },
            "required": ["status", "questions", "categories", "merchants", "max_per_order",
                         "budget_total", "duration_days"],
        },
    },
}

_SYSTEM: Final[str] = (
    "You turn a person's instruction to their shopping agent into a payment mandate. "
    f"Categories you may use: {', '.join(sorted(CATEGORY_MCC))}. "
    f"Stores you may use: {', '.join(sorted(MERCHANT_IDS))}. "
    "Fill only what the person actually said. 'this week' is 7 days, 'today' is 1, "
    "'this month' is 30. Amounts are US dollars. 'max $X per order' is max_per_order; "
    "'$Y total' or 'up to $Y overall' is budget_total. merchants is empty if they did not "
    "restrict stores. If the instruction is ambiguous, names a store or category not in "
    "the lists, or leaves out how much one order may cost, set status to "
    "needs_clarification and ask short questions. Never invent a limit the person did "
    "not give. Reply with JSON only."
)

Status = Literal["ok", "needs_clarification", "unavailable"]


@dataclass(slots=True)
class CompileResult:
    status: Status
    text: str
    request: MandateRequest | None = None
    can: list[str] = field(default_factory=list)
    cannot: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    source: str = ""
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        if self.status != "ok":
            return "\n".join(["The agent needs you to clarify:",
                              *(f"  ? {q}" for q in self.questions)])
        return "\n".join(["Agent CAN:", *(f"  + {c}" for c in self.can),
                          "Agent CANNOT:", *(f"  - {c}" for c in self.cannot)])

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "text": self.text, "can": self.can,
                "cannot": self.cannot, "questions": self.questions, "source": self.source,
                "mandate": self.request.model_dump(mode="json") if self.request else None}


def _parse(content: str) -> dict[str, Any]:
    content = content.strip()
    try:
        value = json.loads(content)
    except ValueError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match is None:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("mandate is not a JSON object")
    return value


class MandateCompiler:
    """Sentence in, :class:`CompileResult` out. Cache-first, like every LLM role."""

    def __init__(self, mode: Literal["mock", "live"] = "mock", *,
                 chat: LlmChat | None = None) -> None:
        self.chat = chat or LlmChat("compiler", mode=mode)

    def compile(self, text: str, *, customer_id: str = "cust-demo-002",
                agent_id: str = "agent-shopper-01",
                now: datetime | None = None) -> CompileResult:
        messages = [{"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": text.strip()}]
        try:
            result = self.chat.chat(messages, response_format=_SCHEMA, max_tokens=500)
            raw = _parse(result.content or "{}")
        except CacheMissError:
            return CompileResult("unavailable", text, questions=[
                "This sentence has not been compiled before and mock mode makes no LLM calls. "
                "Run with --mode live to compile it."], source="miss")
        except (LlmConfigError, ValueError, Exception) as exc:
            return CompileResult("unavailable", text,
                                 questions=[f"The compiler is unavailable ({type(exc).__name__})."],
                                 source="error")
        return self._validate(text, raw, result.source, result.latency_ms, customer_id,
                              agent_id, now or datetime.now(UTC))

    def _validate(self, text: str, raw: dict[str, Any], source: str, latency_ms: float,
                  customer_id: str, agent_id: str, now: datetime) -> CompileResult:
        """The deterministic half. Decides whether there is enough to sign."""
        asked = [str(q) for q in raw.get("questions") or []]
        questions: list[str] = []
        categories = [c for c in raw.get("categories") or [] if c in CATEGORY_MCC]
        merchants = [m for m in raw.get("merchants") or [] if m in MERCHANT_IDS]
        cap = raw.get("max_per_order")
        budget = raw.get("budget_total")
        days = raw.get("duration_days")

        if not categories:
            questions.append("What may the agent buy (groceries, gift cards)?")
        if not isinstance(cap, (int, float)) or cap <= 0:
            questions.append("What is the most the agent may spend on a single order?")
        if not isinstance(days, int) or days <= 0:
            questions.append("For how long should the agent be allowed to buy?")
        if isinstance(budget, (int, float)) and isinstance(cap, (int, float)) and budget < cap:
            questions.append(f"The total (${budget:.2f}) is less than one order (${cap:.2f}). "
                             "Which is right?")
        if raw.get("status") != "ok" or questions:
            # The model's own questions read better; ours fill in only when it asked
            # none. Either way nothing is signed: the gate above is deterministic.
            return CompileResult("needs_clarification", text,
                                 questions=_dedupe(asked or questions),
                                 source=source, latency_ms=latency_ms, raw=raw)

        cap_d = Decimal(str(cap)).quantize(Decimal("0.01"))
        budget_d = (Decimal(str(budget)).quantize(Decimal("0.01"))
                    if isinstance(budget, (int, float)) and budget > 0 else None)
        request = MandateRequest(
            customer_id=customer_id, agent_id=agent_id, mandate_type=MandateType.INTENT,
            scope=MandateScope(categories=[CATEGORY_MCC[c] for c in categories],
                               max_amount=float(cap_d), max_items=30,
                               allowed_merchants=[MERCHANT_IDS[m] for m in merchants],
                               ttl_seconds=int(days) * 86_400),
            description=text.strip(), budget_total=budget_d,
        )
        can, cannot = describe(request, now)
        return CompileResult("ok", text, request=request, can=can, cannot=cannot,
                             source=source, latency_ms=latency_ms, raw=raw)


_MCC_NAMES: Final[dict[str, str]] = {mcc: name.replace("_", " ")
                                     for name, mcc in CATEGORY_MCC.items()}
_MERCHANT_NAMES: Final[dict[str, str]] = {mid: name for name, mid in MERCHANT_IDS.items()}


def describe(request: MandateRequest, issued: datetime) -> tuple[list[str], list[str]]:
    """"Agent CAN / CANNOT" for any mandate request, compiled or hand-built.

    Used by the compiler's echo and by the gate log's mandate column, so the two
    can never describe the same mandate differently.
    """
    scope = request.scope
    what = " and ".join(_MCC_NAMES.get(m, f"MCC {m}") for m in scope.categories) or "anything"
    merchants = [_MERCHANT_NAMES.get(m, m) for m in scope.allowed_merchants]
    where = ", ".join(merchants) if merchants else "any store"
    cap = Decimal(str(scope.max_amount)).quantize(Decimal("0.01")) if scope.max_amount else None
    budget = request.budget_total
    until = ((issued + timedelta(seconds=scope.ttl_seconds)).strftime("%a %d %b %Y")
             if scope.ttl_seconds else None)
    can = [f"buy {what}" + (f" (MCC {', '.join(scope.categories)})" if scope.categories else ""),
           f"shop at {where}"]
    if cap is not None:
        total = f", ${budget:.2f} in total" if budget else ""
        can.append(f"spend up to ${cap} per order{total}")
    if until:
        can.append(f"act until {until}")
    cannot = [f"buy anything other than {what}"] if scope.categories else []
    if merchants:
        cannot.append(f"shop anywhere except {where}")
    if cap is not None:
        cannot.append(f"spend more than ${cap} on one order")
    if budget:
        cannot.append(f"spend more than ${budget:.2f} in total")
    if until:
        cannot.append(f"act after {until}")
    cannot.append("capture, refund or void a payment (only the gate and you can)")
    return can, cannot


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    return [x for x in items if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mantis.agent.mandate_compiler")
    parser.add_argument("text", nargs="+")
    parser.add_argument("--mode", choices=("mock", "live"), default="mock")
    args = parser.parse_args(argv)
    result = MandateCompiler(args.mode).compile(" ".join(args.text))
    print(f'"{result.text}"  [{result.status}, {result.source}, {result.latency_ms:.0f} ms]')
    print(result.summary())
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
