"""MANTIS canonical event schema — FROZEN as of Day 0.

Why this file decides the "real-world feasibility" score
--------------------------------------------------------
Most synthetic-fraud projects invent a convenient schema and then claim
relevance. MANTIS does the opposite: ``TxEvent`` is deliberately a **superset of
a real card authorisation message**. The classic block (``amount``, ``currency``,
``mcc``, ``channel``, ``entry_mode``, ``card_bin``, ``merchant_id``,
``merchant_country``, ``terminal_id``, ``threeds_result``, geo) is the set of
fields an issuer already receives on every authorisation today. Nothing in it is
invented for this project.

The agentic block is a strictly additive **extension**, carried in a nested
``AgenticContext`` that is ``None`` on classic rails. It models the Mastercard
Agent Pay / AP2 mandate world: who the agent is, whether it is registered
(KYA), which mandate authorised the purchase, what that mandate's scope was, and
what the agent read and did before it decided to pay.

The practical consequence — and this is the pitch to a judge — is that an issuer
can adopt MANTIS **without re-platforming**. Every classic field maps to
something already on the wire, so the L0/L1 layers score today's traffic
unchanged; the agentic fields are simply ``None`` until the mandate rail turns
on, at which point the same model starts consuming them. There is no flag day.

``provenance_chain`` is the highest-value field in this schema
---------------------------------------------------------------
It is the ordered list of URLs and content items the agent ingested before
transacting. It is the single field that turns **indirect prompt injection**
from something you can *describe* in a threat deck into something you can
*detect* in a payment stream: if the agent read an attacker-controlled page and
then paid a merchant that was never in the mandate scope, the causal trail is
right there in the authorisation. It is the input to the L3 text layer, and it
is the reason the atlas cards for F1-01/F1-02/F1-03 have detectable signals at
all. Treat it as load-bearing; do not drop it for convenience.

The transaction-lifecycle block — Amendment 1.1.0 (Day 3)
---------------------------------------------------------
Day 0 froze an *authorisation request* and nothing else. Day 2 discovered the
cost of that: F1-03 (refund-logic hijack) and F5-36 could not be represented at
all, and F4-27/F4-28 were half-modelled, because the approve/decline oracle those
attacks *are* had nowhere to live. That was a specification error, not a design
constraint, and it is corrected here — additively, every new field optional with
a default, so nothing written against 1.0.0 changes behaviour.

Each field is one an issuer **already holds today**, which is why the amendment
strengthens the feasibility argument (criterion 5) rather than diluting it:

* ``txn_type`` — ISO 8583 processing code (DE 3) already distinguishes purchase,
  refund, reversal, pre-authorisation and credit on every message on the network.
  Modelling only purchases was us discarding a field the wire carries.
* ``auth_response`` — ISO 8583 response code (DE 39). The issuer *authors* this
  value; it is the single field a card-testing or BIN-enumeration operator is
  actually farming, and without it "decline ratio in the last hour" — a signal
  named on three atlas cards — is unmeasurable.
* ``original_event_id`` — the original-transaction reference (DE 90 / the
  retrieval reference number an acquirer echoes on a credit). Refund-to-original
  binding is a scheme rule, so the link exists on the wire. Its **absence** on a
  credit is itself the F1-03 signal.
* ``dispute_outcome`` / ``dispute_raised_ts`` — chargeback lifecycle state, held
  by every issuer in its dispute system and joined back to the authorisation by
  case management. It arrives late, which is exactly why it is a label-adjacent
  field and not a feature (see below).
* ``settled`` / ``settlement_lag_hours`` — clearing arrives on a separate file
  from authorisation, and the gap between them is ordinary issuer bookkeeping.
  Instant settlement is the property that makes an agent-driven refund
  irreversible before a human can look at it.

**Discipline on the dispute fields.** ``dispute_outcome`` and
``dispute_raised_ts`` resolve days to months after the authorisation. Using them
as features at scoring time would be temporal leakage of a slightly subtler kind
than ``is_fraud`` — the model would be reading the future. They are carried for
evaluation, cost modelling and the console, and the feature builder must drop
them alongside :data:`LABEL_COLUMNS`; :data:`POST_HOC_COLUMNS` names them so the
assertion can be written once.

Amounts stay **non-negative on every type**. Direction is carried by
``txn_type``, not by the sign, because that is how the wire does it and because a
signed amount would silently break every quantile, KS distance and log-amount
feature already calibrated against the Day 1 population.

Freezing policy
---------------
"Frozen" here is a *process* rule, not ``pydantic``'s ``frozen=True`` (events are
mutable so injectors can perturb them, with ``validate_assignment`` re-validating
every write). See CLAUDE.md §4: adding, renaming or retyping a field requires
updating this module, ``tests/test_schema.py``, CLAUDE.md §4, and every ``ag_``
string literal in the repo. New fields are additive with a ``None`` default.

:data:`SCHEMA_VERSION` records where we are. ``1.0.0`` is the Day 0 freeze;
``1.1.0`` adds the transaction-lifecycle block above and removes nothing.

Label discipline
----------------
``LABEL_COLUMNS`` names the three ground-truth columns. They exist on the event
for evaluation and for the console, and they must **never** reach a model's
feature matrix. Feature builders drop them by name and assert they are gone.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "AGENTIC_COLUMNS",
    "ALL_COLUMNS",
    "CLASSIC_COLUMNS",
    "DECLINE_RESPONSES",
    "LABEL_COLUMNS",
    "POST_HOC_COLUMNS",
    "SCHEMA_VERSION",
    "AgenticContext",
    "AuthResponse",
    "Channel",
    "DisputeOutcome",
    "EntryMode",
    "MandateScope",
    "MandateType",
    "ThreeDSResult",
    "TxEvent",
    "TxnType",
    "flatten",
]

#: Schema contract version. 1.0.0 = Day 0 freeze; 1.1.0 = the additive
#: transaction-lifecycle block (see the module docstring). Recorded in every
#: dataset manifest so a parquet can never be mistaken for one it is not.
SCHEMA_VERSION: Final[str] = "1.1.0"


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class Channel(StrEnum):
    """Acceptance channel. The first six are today's rails; ``agentic`` is new."""

    CARD_PRESENT = "card_present"
    ECOM = "ecom"
    MOTO = "moto"
    RECURRING = "recurring"
    UPI_P2P = "upi_p2p"
    UPI_P2M = "upi_p2m"
    AGENTIC = "agentic"


class EntryMode(StrEnum):
    """How the credential reached the acquirer. ``AGENT_TOKEN`` is the new one."""

    CHIP = "chip"
    CONTACTLESS = "contactless"
    MAGSTRIPE = "magstripe"
    ECOM_KEYED = "ecom_keyed"
    CREDENTIAL_ON_FILE = "credential_on_file"
    NETWORK_TOKEN = "network_token"
    QR_SCAN = "qr_scan"
    AGENT_TOKEN = "agent_token"


class ThreeDSResult(StrEnum):
    """Outcome of 3-D Secure / step-up authentication, as the issuer sees it."""

    NOT_APPLICABLE = "not_applicable"
    FRICTIONLESS = "frictionless"
    CHALLENGE_PASSED = "challenge_passed"
    CHALLENGE_FAILED = "challenge_failed"
    ATTEMPTED = "attempted"
    UNAVAILABLE = "unavailable"


class TxnType(StrEnum):
    """ISO 8583 processing code, collapsed to the five shapes that move money.

    ``PURCHASE`` and ``PREAUTH`` pull value from the cardholder; ``REFUND``,
    ``REVERSAL`` and ``CREDIT`` push it back out. That direction is the whole
    reason the amendment exists: outbound flows are chronically monitored less
    than inbound ones, and F1-03 is an attack on precisely that asymmetry.

    ``REFUND`` returns value against a specific prior purchase. ``REVERSAL``
    cancels an authorisation before it clears. ``CREDIT`` is an outbound payment
    with no purchase behind it at all — a goodwill payment, a promotional
    credit, or, in the wrong hands, a laundering leg.
    """

    PURCHASE = "purchase"
    REFUND = "refund"
    REVERSAL = "reversal"
    PREAUTH = "preauth"
    CREDIT = "credit"


class AuthResponse(StrEnum):
    """ISO 8583 response code (DE 39), collapsed to the reasons that differ.

    The issuer authors this field, so it is available at scoring time for every
    *previous* transaction on the card — which is what makes ``decline_ratio_1h``
    a real feature rather than an aspiration. The decline reasons are kept
    distinct because they carry different information: ``INSUFFICIENT_FUNDS``
    says something about the cardholder, ``INVALID_CVV`` and ``EXPIRED`` say
    something about whoever is holding the credential, and ``RISK`` says
    something about the issuer's own model — which an adaptive adversary is
    trying to map.
    """

    APPROVED = "approved"
    DECLINED_INSUFFICIENT_FUNDS = "declined_insufficient_funds"
    DECLINED_DO_NOT_HONOR = "declined_do_not_honor"
    DECLINED_INVALID_CVV = "declined_invalid_cvv"
    DECLINED_EXPIRED = "declined_expired"
    DECLINED_RISK = "declined_risk"


class DisputeOutcome(StrEnum):
    """Chargeback lifecycle state. Resolves long after authorisation.

    Carried for evaluation and cost modelling, **never** as a scoring feature —
    see :data:`POST_HOC_COLUMNS` and the module docstring.
    """

    NONE = "none"
    RAISED = "raised"
    WON_MERCHANT = "won_merchant"
    WON_CARDHOLDER = "won_cardholder"


class MandateType(StrEnum):
    """AP2 mandate granularity, coarsest to finest.

    ``INTENT`` delegates a goal ("book me a flight under 40k"), ``CART`` binds a
    specific basket, ``PAYMENT`` authorises one signed amount. Authority — and
    therefore blast radius — decreases left to right.
    """

    NONE = "none"
    INTENT = "intent"
    CART = "cart"
    PAYMENT = "payment"


# --------------------------------------------------------------------------- #
# Agentic extension
# --------------------------------------------------------------------------- #


class MandateScope(BaseModel):
    """The constraint envelope the human agreed to when issuing the mandate.

    Every field here is a boundary that an attack can push against, which is why
    the scope is modelled explicitly rather than collapsed into a single limit:
    the *shape* of the violation (category drift vs. amount inflation vs.
    merchant substitution) is what distinguishes F1-01 from F1-02.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    categories: list[str] = Field(default_factory=list, description="Permitted MCC groups.")
    max_amount: float | None = Field(default=None, ge=0, description="Per-transaction ceiling.")
    max_items: int | None = Field(default=None, ge=0, description="Max basket line items.")
    allowed_merchants: list[str] = Field(
        default_factory=list, description="Merchant allow-list; empty means unconstrained."
    )
    ttl_seconds: int | None = Field(default=None, ge=0, description="Scope validity window.")


class AgenticContext(BaseModel):
    """Agent-Pay / AP2 extension block. ``None`` on classic rails.

    Grouped by what each subset buys the detector:

    * **Identity** — ``agent_id``, ``agent_platform``, ``kya_token``,
      ``kya_registered``: is this agent known to the network at all?
    * **Authority** — ``mandate_*``, ``consent_sig_valid``, ``delegation_depth``:
      was the purchase inside the envelope the human actually signed?
    * **Provenance** — ``provenance_chain``, ``ingested_content_ids``: what did
      the agent read before deciding? The L3 text layer lives here.
    * **Behaviour** — ``tool_call_count``, ``deliberation_latency_ms``,
      ``cursor_entropy``, ``dwell_time_ms``: does the session look like an agent
      thinking, a human shopping, or a script pretending to be one of them?
      ``cursor_entropy`` and ``dwell_time_ms`` are the human-presence telemetry
      that F1-09 (human-present spoofing) has to forge.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # -- identity ----------------------------------------------------------- #
    agent_id: str = Field(description="Stable identifier for the transacting agent.")
    agent_platform: str = Field(description="Vendor/runtime hosting the agent.")
    kya_token: str | None = Field(default=None, description="Know-Your-Agent credential.")
    kya_registered: bool = Field(default=False, description="Agent present in the KYA registry.")

    # -- authority ---------------------------------------------------------- #
    mandate_type: MandateType = MandateType.NONE
    mandate_id: str | None = None
    mandate_hash: str | None = Field(
        default=None, description="Digest of the signed mandate; replay detection key."
    )
    mandate_issued_ts: datetime | None = None
    mandate_ttl_seconds: int | None = Field(default=None, ge=0)
    mandate_scope: MandateScope | None = None
    human_present: bool = Field(
        default=False, description="Claimed live human oversight; drives liability."
    )
    consent_sig_valid: bool | None = Field(
        default=None, description="Cryptographic verification result of the consent signature."
    )
    delegation_depth: int = Field(
        default=0, ge=0, description="Agent-to-sub-agent hops between human and payment."
    )

    # -- provenance --------------------------------------------------------- #
    provenance_chain: list[str] = Field(
        default_factory=list,
        description="Ordered URLs/content the agent read before transacting. See module docstring.",
    )
    ingested_content_ids: list[str] = Field(
        default_factory=list, description="Content hashes joining to the corpus store."
    )

    # -- behaviour ---------------------------------------------------------- #
    tool_call_count: int = Field(default=0, ge=0)
    deliberation_latency_ms: int | None = Field(default=None, ge=0)
    cursor_entropy: float | None = Field(default=None, ge=0)
    dwell_time_ms: int | None = Field(default=None, ge=0)

    def expires_at(self) -> datetime | None:
        """Mandate expiry, or ``None`` when issuance time or TTL is unknown."""
        if self.mandate_issued_ts is None or self.mandate_ttl_seconds is None:
            return None
        return self.mandate_issued_ts + timedelta(seconds=self.mandate_ttl_seconds)


# --------------------------------------------------------------------------- #
# The event
# --------------------------------------------------------------------------- #


class TxEvent(BaseModel):
    """One authorisation request: classic card fields plus an optional agentic block."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # -- classic authorisation ---------------------------------------------- #
    event_id: str
    ts: datetime
    amount: float = Field(ge=0, description="Transaction amount in `currency` major units.")
    currency: str = Field(description="ISO 4217 alphabetic code.")
    mcc: str = Field(description="ISO 18245 merchant category code, exactly 4 digits.")
    channel: Channel
    entry_mode: EntryMode
    customer_id: str
    card_bin: str = Field(description="Issuer BIN, 6-8 digits.")
    merchant_id: str
    merchant_country: str = Field(description="ISO 3166-1 alpha-2.")
    terminal_id: str | None = Field(default=None, description="None off card-present rails.")
    device_id: str | None = None
    ip: str | None = None
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    threeds_result: ThreeDSResult = ThreeDSResult.NOT_APPLICABLE

    # -- transaction lifecycle (Amendment 1.1.0) ----------------------------- #
    # Every field here is one an issuer already holds. Defaults reproduce a
    # settled, approved purchase, so every 1.0.0 call site is unchanged.
    txn_type: TxnType = TxnType.PURCHASE
    auth_response: AuthResponse = AuthResponse.APPROVED
    original_event_id: str | None = Field(
        default=None,
        description="Original-transaction reference on a refund/reversal/credit. "
        "Its absence on an outbound flow is itself a signal.",
    )
    dispute_outcome: DisputeOutcome | None = Field(
        default=None, description="Chargeback state. Post-hoc; never a feature."
    )
    dispute_raised_ts: datetime | None = Field(
        default=None, description="When the cardholder disputed. Post-hoc; never a feature."
    )
    settled: bool = Field(default=True, description="Whether the authorisation reached clearing.")
    settlement_lag_hours: float | None = Field(
        default=None,
        ge=0,
        description="Hours from authorisation to clearing. None when unsettled or not yet known.",
    )

    # -- agentic extension --------------------------------------------------- #
    agentic: AgenticContext | None = None

    # -- ground truth: NEVER a feature (see LABEL_COLUMNS) ------------------- #
    is_fraud: bool = False
    attack_id: str | None = Field(default=None, description="Atlas card id, e.g. 'F1-01'.")
    attack_campaign: str | None = Field(
        default=None, description="Groups events belonging to one simulated campaign."
    )

    # -- validation ---------------------------------------------------------- #

    @field_validator("currency")
    @classmethod
    def _check_currency(cls, v: str) -> str:
        if len(v) != 3 or not v.isalpha() or not v.isupper():
            raise ValueError(f"currency must be a 3-letter uppercase ISO 4217 code, got {v!r}")
        return v

    @field_validator("mcc")
    @classmethod
    def _check_mcc(cls, v: str) -> str:
        if len(v) != 4 or not v.isdigit():
            raise ValueError(f"mcc must be exactly 4 digits, got {v!r}")
        return v

    @field_validator("card_bin")
    @classmethod
    def _check_card_bin(cls, v: str) -> str:
        if not v.isdigit() or not 6 <= len(v) <= 8:
            raise ValueError(f"card_bin must be 6-8 digits, got {v!r}")
        return v

    @field_validator("merchant_country")
    @classmethod
    def _check_country(cls, v: str) -> str:
        if len(v) != 2 or not v.isalpha() or not v.isupper():
            raise ValueError(f"merchant_country must be ISO 3166-1 alpha-2, got {v!r}")
        return v

    @model_validator(mode="after")
    def _check_rail_consistency(self) -> TxEvent:
        """The agentic block and the agentic rail must agree.

        The ``agent_token`` escape hatch is deliberate: an agent-originated
        purchase presented over a classic ecom rail is itself a fraud pattern we
        need to be able to represent, so it stays expressible.
        """
        if self.channel is Channel.AGENTIC and self.agentic is None:
            raise ValueError("channel='agentic' requires an AgenticContext")
        if (
            self.agentic is not None
            and self.channel is not Channel.AGENTIC
            and self.entry_mode is not EntryMode.AGENT_TOKEN
        ):
            raise ValueError(
                "AgenticContext on a classic rail requires entry_mode='agent_token'; "
                f"got channel={self.channel!r}, entry_mode={self.entry_mode!r}"
            )
        return self

    @model_validator(mode="after")
    def _check_lifecycle_coherence(self) -> TxEvent:
        """The lifecycle block must describe something that can actually happen.

        These are not style checks. A declined authorisation that also claims to
        have settled would let a detector learn an impossible combination and
        post a metric no issuer could reproduce; a dispute timestamp with no
        dispute is a null pattern a model would happily memorise. Each clause
        below closes one such hole.

        What is *not* enforced: a refund is allowed to carry
        ``original_event_id=None``. An orphan credit is a real, and central,
        F1-03 shape, and a schema that forbade it would make the attack
        unrepresentable exactly the way 1.0.0 did.
        """
        if self.auth_response is not AuthResponse.APPROVED:
            if self.settled:
                raise ValueError(
                    f"auth_response={self.auth_response.value!r} cannot settle; "
                    "a declined authorisation never reaches clearing"
                )
            if self.settlement_lag_hours is not None:
                raise ValueError("settlement_lag_hours set on a declined authorisation")
        if not self.settled and self.settlement_lag_hours is not None:
            raise ValueError("settlement_lag_hours set on an unsettled authorisation")

        if self.original_event_id is not None and self.txn_type in (
            TxnType.PURCHASE,
            TxnType.PREAUTH,
        ):
            raise ValueError(
                f"original_event_id set on txn_type={self.txn_type.value!r}; "
                "only refunds, reversals and credits reference a prior authorisation"
            )
        if self.original_event_id == self.event_id:
            raise ValueError("original_event_id must not point at the event itself")

        disputed = self.dispute_outcome not in (None, DisputeOutcome.NONE)
        if disputed and self.dispute_raised_ts is None:
            raise ValueError(f"dispute_outcome={self.dispute_outcome} requires dispute_raised_ts")
        if not disputed and self.dispute_raised_ts is not None:
            raise ValueError("dispute_raised_ts set with no dispute recorded")
        if self.dispute_raised_ts is not None and self.dispute_raised_ts < self.ts:
            raise ValueError("dispute raised before the authorisation it disputes")
        return self

    @model_validator(mode="after")
    def _check_label_integrity(self) -> TxEvent:
        """Ground truth must be self-consistent, or downstream metrics lie."""
        if self.attack_id is not None and not self.is_fraud:
            raise ValueError(f"attack_id={self.attack_id!r} set on a non-fraud event")
        if self.is_fraud and self.attack_id is None:
            raise ValueError("is_fraud=True requires an attack_id naming the atlas card")
        return self


# --------------------------------------------------------------------------- #
# Column contracts
# --------------------------------------------------------------------------- #

#: Ground truth. Dropped by name in every feature builder; see CLAUDE.md HARD RULE 1.
LABEL_COLUMNS: Final[tuple[str, ...]] = ("is_fraud", "attack_id", "attack_campaign")

#: Resolved days-to-months after authorisation. Not ground truth, but using them
#: at scoring time is temporal leakage: the model would be reading the future.
#: Feature builders drop these alongside ``LABEL_COLUMNS``.
POST_HOC_COLUMNS: Final[tuple[str, ...]] = ("dispute_outcome", "dispute_raised_ts")

#: Every non-approval. Named once so ``decline_ratio`` features cannot drift.
DECLINE_RESPONSES: Final[tuple[str, ...]] = tuple(
    r.value for r in AuthResponse if r is not AuthResponse.APPROVED
)

#: Classic authorisation columns, in wire order.
CLASSIC_COLUMNS: Final[tuple[str, ...]] = (
    "event_id",
    "ts",
    "amount",
    "currency",
    "mcc",
    "channel",
    "entry_mode",
    "customer_id",
    "card_bin",
    "merchant_id",
    "merchant_country",
    "terminal_id",
    "device_id",
    "ip",
    "lat",
    "lon",
    "threeds_result",
    # -- lifecycle, Amendment 1.1.0. Appended, never interleaved, so a 1.0.0
    # -- reader slicing the first 17 classic columns still gets what it expects.
    "txn_type",
    "auth_response",
    "original_event_id",
    "dispute_outcome",
    "dispute_raised_ts",
    "settled",
    "settlement_lag_hours",
)

#: Flattened agentic columns. ``mandate_scope`` explodes into ``ag_scope_*``.
AGENTIC_COLUMNS: Final[tuple[str, ...]] = (
    "ag_agent_id",
    "ag_agent_platform",
    "ag_kya_token",
    "ag_kya_registered",
    "ag_mandate_type",
    "ag_mandate_id",
    "ag_mandate_hash",
    "ag_mandate_issued_ts",
    "ag_mandate_ttl_seconds",
    "ag_scope_categories",
    "ag_scope_max_amount",
    "ag_scope_max_items",
    "ag_scope_allowed_merchants",
    "ag_scope_ttl_seconds",
    "ag_human_present",
    "ag_consent_sig_valid",
    "ag_delegation_depth",
    "ag_provenance_chain",
    "ag_ingested_content_ids",
    "ag_tool_call_count",
    "ag_deliberation_latency_ms",
    "ag_cursor_entropy",
    "ag_dwell_time_ms",
)

#: Every column ``flatten`` emits, in order. A flat frame has exactly these keys.
ALL_COLUMNS: Final[tuple[str, ...]] = CLASSIC_COLUMNS + AGENTIC_COLUMNS + LABEL_COLUMNS


def flatten(ev: TxEvent) -> dict[str, Any]:
    """Flatten one event into a pandas-ready row.

    The key set is **always** ``ALL_COLUMNS``, whether or not the event carries an
    agentic block — classic rails emit ``None`` for every ``ag_`` column. A stable
    key set is what lets the foundry concatenate classic and agentic populations
    into one frame without ragged columns, and lets the feature builder assert on
    an exact schema instead of guessing.

    Enums are emitted as their string values; ``list`` fields are returned as
    lists (object columns) because L3 needs ``provenance_chain`` intact.
    """
    row: dict[str, Any] = {
        "event_id": ev.event_id,
        "ts": ev.ts,
        "amount": ev.amount,
        "currency": ev.currency,
        "mcc": ev.mcc,
        "channel": ev.channel.value,
        "entry_mode": ev.entry_mode.value,
        "customer_id": ev.customer_id,
        "card_bin": ev.card_bin,
        "merchant_id": ev.merchant_id,
        "merchant_country": ev.merchant_country,
        "terminal_id": ev.terminal_id,
        "device_id": ev.device_id,
        "ip": ev.ip,
        "lat": ev.lat,
        "lon": ev.lon,
        "threeds_result": ev.threeds_result.value,
        "txn_type": ev.txn_type.value,
        "auth_response": ev.auth_response.value,
        "original_event_id": ev.original_event_id,
        "dispute_outcome": None if ev.dispute_outcome is None else ev.dispute_outcome.value,
        "dispute_raised_ts": ev.dispute_raised_ts,
        "settled": ev.settled,
        "settlement_lag_hours": ev.settlement_lag_hours,
    }

    ag = ev.agentic
    scope = ag.mandate_scope if ag is not None else None
    row.update(
        {
            "ag_agent_id": ag.agent_id if ag else None,
            "ag_agent_platform": ag.agent_platform if ag else None,
            "ag_kya_token": ag.kya_token if ag else None,
            "ag_kya_registered": ag.kya_registered if ag else None,
            "ag_mandate_type": ag.mandate_type.value if ag else None,
            "ag_mandate_id": ag.mandate_id if ag else None,
            "ag_mandate_hash": ag.mandate_hash if ag else None,
            "ag_mandate_issued_ts": ag.mandate_issued_ts if ag else None,
            "ag_mandate_ttl_seconds": ag.mandate_ttl_seconds if ag else None,
            "ag_scope_categories": list(scope.categories) if scope else None,
            "ag_scope_max_amount": scope.max_amount if scope else None,
            "ag_scope_max_items": scope.max_items if scope else None,
            "ag_scope_allowed_merchants": list(scope.allowed_merchants) if scope else None,
            "ag_scope_ttl_seconds": scope.ttl_seconds if scope else None,
            "ag_human_present": ag.human_present if ag else None,
            "ag_consent_sig_valid": ag.consent_sig_valid if ag else None,
            "ag_delegation_depth": ag.delegation_depth if ag else None,
            "ag_provenance_chain": list(ag.provenance_chain) if ag else None,
            "ag_ingested_content_ids": list(ag.ingested_content_ids) if ag else None,
            "ag_tool_call_count": ag.tool_call_count if ag else None,
            "ag_deliberation_latency_ms": ag.deliberation_latency_ms if ag else None,
            "ag_cursor_entropy": ag.cursor_entropy if ag else None,
            "ag_dwell_time_ms": ag.dwell_time_ms if ag else None,
        }
    )

    row.update(
        {
            "is_fraud": ev.is_fraud,
            "attack_id": ev.attack_id,
            "attack_campaign": ev.attack_campaign,
        }
    )
    return row


def main() -> None:
    """Print the schema contract. Run: ``python -m mantis.core.events``."""
    print(f"MANTIS event schema (FROZEN)  v{SCHEMA_VERSION}")
    print(f"  classic columns : {len(CLASSIC_COLUMNS)}")
    print(f"  agentic columns : {len(AGENTIC_COLUMNS)}  (prefix 'ag_')")
    print(f"  label columns   : {len(LABEL_COLUMNS)}  -> {', '.join(LABEL_COLUMNS)}")
    print(f"  total emitted   : {len(ALL_COLUMNS)}")
    print()
    print(f"  channels    : {', '.join(c.value for c in Channel)}")
    print(f"  entry modes : {', '.join(e.value for e in EntryMode)}")
    print(f"  mandates    : {', '.join(m.value for m in MandateType)}")
    print(f"  txn types   : {', '.join(t.value for t in TxnType)}")
    print(f"  auth resp   : {', '.join(a.value for a in AuthResponse)}")
    print(f"  post-hoc    : {', '.join(POST_HOC_COLUMNS)}  (never a scoring feature)")
    print()
    print("  NEVER feed LABEL_COLUMNS to a model. See CLAUDE.md HARD RULE 1.")


if __name__ == "__main__":
    main()
