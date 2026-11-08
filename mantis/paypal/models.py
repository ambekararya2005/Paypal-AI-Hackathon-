"""What the human signs, what the agent asks for, and what the agent read.

Three objects cross the gate, and each maps onto something the frozen schema
already has a place for — nothing here adds a field to ``TxEvent``:

* :class:`Mandate` is the human's signed envelope. Its scope **is** a
  :class:`~mantis.core.events.MandateScope`, reused rather than re-declared, so the
  L0 clauses that were measured on the foundry read a PayPal mandate with exactly
  the semantics they were measured with.
* :class:`PurchaseIntent` is what the agent wants to buy: a merchant and a basket.
* :class:`AgentContext` is who the agent is and what it read on the way. The
  ``pages`` list becomes ``provenance_chain`` and ``ingested_content_ids``.

The mandate signature
-----------------------
The human approves a mandate once. :meth:`Mandate.sign` stores an HMAC over the
mandate's canonical bytes, and the adapter sets ``consent_sig_valid`` from
:meth:`Mandate.verify` at scoring time. A mandate edited after signing — a widened
ceiling, an extra category — therefore fails verification and trips L0's
``consent_invalid`` clause. ``mandate_hash`` is the SHA-256 of the same bytes,
which is the replay key the schema already names.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mantis.core.events import MandateScope, MandateType

__all__ = [
    "AgentContext",
    "Mandate",
    "MandateRequest",
    "OrderItem",
    "Page",
    "PurchaseIntent",
    "canonical_json",
]


def canonical_json(payload: object) -> bytes:
    """Deterministic JSON: sorted keys, no whitespace, UTF-8.

    The one serialisation every hash in this package goes through. Two dicts with
    the same content always produce the same bytes, whichever order they were
    built in — which is the whole point of hashing them.
    """
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")


class MandateRequest(BaseModel):
    """What ``POST /mandates`` accepts: the human's terms, before signing."""

    model_config = ConfigDict(extra="forbid")

    customer_id: str
    agent_id: str
    mandate_type: MandateType = MandateType.INTENT
    scope: MandateScope
    currency: str = "USD"
    description: str = ""
    #: PayPal Vault payment token the human approved once, if Vault is used.
    vault_id: str | None = None


class Mandate(MandateRequest):
    """A mandate the human signed. Immutable in meaning: editing it breaks the HMAC."""

    mandate_id: str
    issued_ts: datetime
    signature: str | None = None

    def body(self) -> dict[str, object]:
        """Every signed field, i.e. everything except the signature itself."""
        return self.model_dump(mode="json", exclude={"signature"})

    def digest(self) -> str:
        """SHA-256 hex of the canonical body. Becomes ``ag_mandate_hash``."""
        return hashlib.sha256(canonical_json(self.body())).hexdigest()

    def sign(self, key: str) -> Mandate:
        """Return a copy carrying an HMAC-SHA256 signature over the body."""
        mac = hmac.new(key.encode("utf-8"), canonical_json(self.body()), hashlib.sha256)
        return self.model_copy(update={"signature": mac.hexdigest()})

    def verify(self, key: str) -> bool:
        """Constant-time check that the body is what was signed."""
        if not self.signature:
            return False
        expected = hmac.new(key.encode("utf-8"), canonical_json(self.body()), hashlib.sha256)
        return hmac.compare_digest(expected.hexdigest(), self.signature)

    @property
    def ttl_seconds(self) -> int | None:
        return self.scope.ttl_seconds

    @classmethod
    def issue(cls, request: MandateRequest, *, mandate_id: str, key: str,
              now: datetime | None = None) -> Mandate:
        """Issue and sign a mandate from a request."""
        mandate = cls(
            **request.model_dump(),
            mandate_id=mandate_id,
            issued_ts=now or datetime.now(UTC),
        )
        return mandate.sign(key)


class OrderItem(BaseModel):
    """One basket line. Money is :class:`~decimal.Decimal`, because PayPal checks
    that ``item_total`` equals the sum of the lines to the cent and a float sum
    will eventually disagree with it."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=127)
    sku: str = Field(default="", max_length=127)
    quantity: int = Field(default=1, ge=1)
    unit_amount: Decimal = Field(gt=0)
    category: Literal["PHYSICAL_GOODS", "DIGITAL_GOODS", "DONATION"] = "PHYSICAL_GOODS"

    @field_validator("unit_amount")
    @classmethod
    def _two_places(cls, v: Decimal) -> Decimal:
        return v.quantize(Decimal("0.01"))


class PurchaseIntent(BaseModel):
    """What the agent asks to buy, under which mandate."""

    model_config = ConfigDict(extra="forbid")

    mandate_id: str
    merchant_id: str
    items: list[OrderItem] = Field(min_length=1)
    currency: str = "USD"
    description: str = ""

    @property
    def total(self) -> Decimal:
        return sum((i.unit_amount * i.quantity for i in self.items), Decimal("0.00"))

    @property
    def n_items(self) -> int:
        return sum(i.quantity for i in self.items)


class Page(BaseModel):
    """One thing the agent read. ``text`` is optional: when the agent does not
    forward it, L3 resolves the URL's content id through the committed store."""

    model_config = ConfigDict(extra="forbid")

    url: str
    text: str | None = None


class AgentContext(BaseModel):
    """Who the agent is and how it got to this purchase."""

    model_config = ConfigDict(extra="forbid")

    agent_id: str
    agent_platform: str
    kya_token: str | None = None
    delegation_depth: int = Field(default=0, ge=0)
    pages: list[Page] = Field(default_factory=list)
    tool_call_count: int = Field(default=0, ge=0)
    deliberation_latency_ms: int | None = Field(default=None, ge=0)
    #: An agent paying on its own is, by definition, not a human at the device.
    human_present: bool = False

    @property
    def provenance_chain(self) -> list[str]:
        return [p.url for p in self.pages]
