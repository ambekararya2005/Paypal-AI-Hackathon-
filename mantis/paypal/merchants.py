"""The merchant and agent registries the gate resolves against.

A PayPal order does not carry a merchant category code. An issuer's
authorisation does, and every L0 scope clause is written against one, so the gate
needs a merchant profile to supply it — the same thing an acquirer's merchant
master file does on a card rail. Two demo merchants are enough for the three
scenarios; a deployment would back this with the payee's onboarding record.

The KYA registry is the same idea for agents: ``kya_registered`` is True only
when the agent id is known **and** presents the token it was registered with.

Shop hosts follow ``shop.<merchant>.test``, the shape
:func:`~mantis.defense.l0_rules.rules.provenance_merchant_mismatch` reads, so a
trail that ends on the paid merchant's own page passes that clause and a trail
that ends anywhere else fails it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__ = ["AGENTS", "MERCHANTS", "AgentRecord", "Merchant", "kya_registered"]


@dataclass(frozen=True, slots=True)
class Merchant:
    merchant_id: str
    name: str
    #: ISO 18245. 5411 grocery; 6540 stored-value / gift-card load (the F6-40 MCC).
    mcc: str
    country: str
    shop_host: str

    @property
    def shop_url(self) -> str:
        return f"https://{self.shop_host}"


@dataclass(frozen=True, slots=True)
class AgentRecord:
    agent_id: str
    platform: str
    kya_token: str


MERCHANTS: Final[dict[str, Merchant]] = {
    m.merchant_id: m
    for m in (
        Merchant("m-freshmart", "FreshMart Groceries", "5411", "US", "shop.freshmart.test"),
        Merchant("m-giftly", "Giftly Gift Cards", "6540", "US", "shop.giftly.test"),
    )
}

AGENTS: Final[dict[str, AgentRecord]] = {
    a.agent_id: a
    for a in (AgentRecord("agent-shopper-01", "mantis-demo-agent", "kya-demo-shopper-01"),)
}


def kya_registered(agent_id: str, token: str | None) -> bool:
    """Known agent presenting its registered token."""
    record = AGENTS.get(agent_id)
    return record is not None and token is not None and token == record.kya_token
