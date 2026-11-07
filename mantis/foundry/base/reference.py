"""Calibration tables for the legitimate population — the "don't invent it" layer.

A payments judge spots an invented distribution instantly: uniform amounts, a
flat hour-of-day curve, every merchant equally popular. So every stochastic
choice the simulator makes is driven by a value in :class:`ReferenceStats`, and
:class:`ReferenceStats` has exactly two provenances, both of them declared in
the object itself and printed by the simulator:

1. **Fitted** — ``data/reference/reference_stats.json`` exists, because someone
   ran ``scripts/fit_reference.py`` over the Kaggle ``kartik2112/fraud-detection``
   (Sparkov) CSV. Shape parameters (hour curve, day-of-week curve, per-category
   log-amount sigma, category mix, merchant Zipf exponent, per-customer velocity)
   come straight off that data.
2. **Priors** — the file is absent, which is the default for a clean clone
   (HARD RULE 4: no Kaggle token, no download at runtime). We fall back to the
   Indian-market priors in this module.

The priors are *stated*, not hidden: each block below carries the reasoning and
the public source class it is drawn from, and ``ReferenceStats.provenance``
carries the same thing in machine-readable form so the fidelity scorecard can
print "these numbers are priors, not measurements" without anyone having to
remember to say it.

**Honesty note that belongs in the writeup.** The priors are practitioner
estimates of the Indian card/UPI acceptance mix — RBI *Payment System
Indicators* and NPCI monthly UPI statistics give the channel and ticket-size
order of magnitude; the per-MCC medians are round-number estimates consistent
with them. They are calibrated in the sense of "reproducing the published
aggregates", not in the sense of "fitted to a licensed transaction panel". The
fitted path exists precisely so that claim can be upgraded when real data is
available, and the fidelity scorecard reports which path was used.

Partial overrides work: every field has a default, so a JSON containing only
``hour_weights`` replaces the hour curve and inherits the rest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mantis.core.events import AuthResponse, Channel, EntryMode, MandateType, ThreeDSResult
from mantis.core.paths import REFERENCE_STATS_JSON

__all__ = [
    "BinProfile",
    "CityPoint",
    "MccProfile",
    "ReferenceStats",
    "load_reference_stats",
]

#: Classic (non-agentic) channels. The agentic rail is selected separately, so it
#: never appears in an MCC's ``channel_weights``.
CLASSIC_CHANNELS: Final[frozenset[str]] = frozenset(
    c.value for c in Channel if c is not Channel.AGENTIC
)


def _normalise(weights: dict[str, float]) -> dict[str, float]:
    """Scale a weight mapping to sum to 1.0, rejecting degenerate input."""
    total = sum(weights.values())
    if total <= 0:
        raise ValueError(f"weights must sum to a positive number, got {total}")
    return {k: v / total for k, v in weights.items()}


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #


class MccProfile(BaseModel):
    """Everything the simulator needs to know about one merchant category.

    Amounts are log-normal per MCC rather than one global distribution, because
    the single most obvious tell of a synthetic payment file is a fuel purchase
    and a flight booking drawn from the same curve. ``log_amount_mu`` is the
    natural log of the **median** ticket (log-normal median is ``exp(mu)``), so
    the priors below can be read and argued about in rupees.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    mcc: str = Field(description="ISO 18245 code, 4 digits.")
    label: str
    weight: float = Field(gt=0, description="Share of classic volume; normalised on load.")
    log_amount_mu: float = Field(description="ln(median ticket) in the reference currency.")
    log_amount_sigma: float = Field(gt=0, description="Dispersion of ln(amount).")
    channel_weights: dict[str, float] = Field(
        description="Classic channel mix for this MCC; normalised on load."
    )
    agentic_affinity: float = Field(
        ge=0,
        description=(
            "Multiplier on this category's odds of being routed through an agent. "
            "Flights and subscriptions are high; fuel and transit are near zero, "
            "because an agent cannot fill a tank."
        ),
    )

    @field_validator("mcc")
    @classmethod
    def _check_mcc(cls, v: str) -> str:
        if len(v) != 4 or not v.isdigit():
            raise ValueError(f"mcc must be exactly 4 digits, got {v!r}")
        return v

    @field_validator("channel_weights")
    @classmethod
    def _check_channels(cls, v: dict[str, float]) -> dict[str, float]:
        bad = sorted(set(v) - CLASSIC_CHANNELS)
        if bad:
            raise ValueError(
                f"unknown/ineligible channels {bad}; valid: {sorted(CLASSIC_CHANNELS)}"
            )
        return _normalise(v)


class CityPoint(BaseModel):
    """A metro anchor. Customer home points and merchant sites scatter around these."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    weight: float = Field(gt=0)


class BinProfile(BaseModel):
    """One issuer BIN in the synthetic portfolio.

    These are **structurally valid but synthetic** — six-digit prefixes drawn
    from the payment-sandbox conventions everyone in the industry recognises as
    test values. They are deliberately not intended to resolve to any real
    issuer, which is what lets the population be published.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    card_bin: str
    network: str
    weight: float = Field(gt=0)

    @field_validator("card_bin")
    @classmethod
    def _check_bin(cls, v: str) -> str:
        if not v.isdigit() or not 6 <= len(v) <= 8:
            raise ValueError(f"card_bin must be 6-8 digits, got {v!r}")
        return v


# --------------------------------------------------------------------------- #
# Indian-market priors
# --------------------------------------------------------------------------- #

# Per-MCC ticket-size priors, in INR. Medians are round-number estimates
# consistent with the published aggregates: UPI P2M average ticket sits in the
# high hundreds, card POS in the low thousands, card-not-present higher again.
# Sigmas encode dispersion, and are the parameter Sparkov actually calibrates:
# fuel is tight (people buy round amounts), travel is wide.
# fmt: off
_MCC_PRIORS: Final[tuple[dict[str, Any], ...]] = (
    # -- daily spend: high frequency, small ticket, POS + UPI heavy ---------- #
    {
        "mcc": "5411", "label": "Grocery stores & supermarkets", "weight": 0.115,
        "median": 850.0, "sigma": 0.85, "agentic_affinity": 1.4,
        "ch": {"card_present": 0.46, "ecom": 0.16, "recurring": 0.01, "upi_p2m": 0.37},
    },
    {
        "mcc": "5812", "label": "Eating places & restaurants", "weight": 0.105,
        "median": 690.0, "sigma": 0.90, "agentic_affinity": 0.8,
        "ch": {"card_present": 0.48, "ecom": 0.12, "upi_p2m": 0.40},
    },
    {
        "mcc": "5814", "label": "Fast food restaurants", "weight": 0.080,
        "median": 320.0, "sigma": 0.75, "agentic_affinity": 0.5,
        "ch": {"card_present": 0.34, "ecom": 0.18, "upi_p2m": 0.48},
    },
    {
        "mcc": "5541", "label": "Service stations (fuel)", "weight": 0.085,
        "median": 1000.0, "sigma": 0.55, "agentic_affinity": 0.05,
        "ch": {"card_present": 0.55, "upi_p2m": 0.45},
    },
    {
        "mcc": "4121", "label": "Taxicabs & ride-hailing", "weight": 0.062,
        "median": 235.0, "sigma": 0.80, "agentic_affinity": 0.9,
        "ch": {"card_present": 0.15, "ecom": 0.30, "upi_p2m": 0.55},
    },
    {
        "mcc": "4111", "label": "Local commuter transport", "weight": 0.030,
        "median": 55.0, "sigma": 0.70, "agentic_affinity": 0.2,
        "ch": {"card_present": 0.28, "ecom": 0.14, "upi_p2m": 0.58},
    },
    # -- bills & top-ups: the recurring rail --------------------------------- #
    {
        "mcc": "4900", "label": "Utilities", "weight": 0.055,
        "median": 1450.0, "sigma": 0.90, "agentic_affinity": 1.6,
        "ch": {"ecom": 0.40, "recurring": 0.45, "upi_p2m": 0.15},
    },
    {
        "mcc": "4814", "label": "Telecom & prepaid recharge", "weight": 0.058,
        "median": 299.0, "sigma": 0.55, "agentic_affinity": 1.5,
        "ch": {"ecom": 0.46, "recurring": 0.24, "upi_p2m": 0.30},
    },
    {
        "mcc": "5734", "label": "Computer software & subscriptions", "weight": 0.018,
        "median": 499.0, "sigma": 0.80, "agentic_affinity": 1.8,
        "ch": {"ecom": 0.38, "recurring": 0.58, "upi_p2m": 0.04},
    },
    {
        "mcc": "6300", "label": "Insurance premiums", "weight": 0.014,
        "median": 8200.0, "sigma": 0.90, "agentic_affinity": 1.1,
        "ch": {"ecom": 0.40, "recurring": 0.55, "moto": 0.05},
    },
    {
        "mcc": "7997", "label": "Health clubs & memberships", "weight": 0.012,
        "median": 1900.0, "sigma": 0.85, "agentic_affinity": 0.5,
        "ch": {"card_present": 0.24, "ecom": 0.28, "recurring": 0.48},
    },
    # -- retail -------------------------------------------------------------- #
    {
        "mcc": "5912", "label": "Drug stores & pharmacies", "weight": 0.050,
        "median": 470.0, "sigma": 0.95, "agentic_affinity": 1.1,
        "ch": {"card_present": 0.52, "ecom": 0.20, "upi_p2m": 0.28},
    },
    {
        "mcc": "5999", "label": "Miscellaneous retail", "weight": 0.062,
        "median": 880.0, "sigma": 1.10, "agentic_affinity": 1.3,
        "ch": {"card_present": 0.38, "ecom": 0.34, "moto": 0.01, "upi_p2m": 0.27},
    },
    {
        "mcc": "5651", "label": "Family clothing stores", "weight": 0.048,
        "median": 1580.0, "sigma": 1.00, "agentic_affinity": 1.2,
        "ch": {"card_present": 0.44, "ecom": 0.40, "upi_p2m": 0.16},
    },
    {
        "mcc": "5732", "label": "Consumer electronics", "weight": 0.034,
        "median": 4400.0, "sigma": 1.15, "agentic_affinity": 1.4,
        "ch": {"card_present": 0.34, "ecom": 0.52, "upi_p2m": 0.14},
    },
    {
        "mcc": "5311", "label": "Department stores", "weight": 0.030,
        "median": 2150.0, "sigma": 0.95, "agentic_affinity": 0.7,
        "ch": {"card_present": 0.56, "ecom": 0.28, "upi_p2m": 0.16},
    },
    {
        "mcc": "5977", "label": "Cosmetic stores", "weight": 0.015,
        "median": 950.0, "sigma": 0.90, "agentic_affinity": 1.0,
        "ch": {"card_present": 0.36, "ecom": 0.48, "upi_p2m": 0.16},
    },
    {
        "mcc": "5945", "label": "Hobby, toy & game shops", "weight": 0.011,
        "median": 780.0, "sigma": 0.95, "agentic_affinity": 1.1,
        "ch": {"card_present": 0.30, "ecom": 0.56, "upi_p2m": 0.14},
    },
    {
        "mcc": "5942", "label": "Book stores", "weight": 0.014,
        "median": 640.0, "sigma": 0.85, "agentic_affinity": 1.3,
        "ch": {"card_present": 0.26, "ecom": 0.58, "upi_p2m": 0.16},
    },
    # -- travel & leisure: the canonical agentic use case -------------------- #
    {
        "mcc": "4511", "label": "Airlines", "weight": 0.016,
        "median": 6800.0, "sigma": 0.95, "agentic_affinity": 2.2,
        "ch": {"card_present": 0.06, "ecom": 0.88, "moto": 0.06},
    },
    {
        "mcc": "7011", "label": "Lodging & hotels", "weight": 0.018,
        "median": 5200.0, "sigma": 0.95, "agentic_affinity": 2.0,
        "ch": {"card_present": 0.30, "ecom": 0.62, "moto": 0.08},
    },
    {
        "mcc": "4722", "label": "Travel agencies", "weight": 0.012,
        "median": 9400.0, "sigma": 1.20, "agentic_affinity": 2.0,
        "ch": {"card_present": 0.08, "ecom": 0.80, "moto": 0.12},
    },
    {
        "mcc": "7832", "label": "Motion picture theatres", "weight": 0.026,
        "median": 520.0, "sigma": 0.60, "agentic_affinity": 1.0,
        "ch": {"card_present": 0.22, "ecom": 0.62, "upi_p2m": 0.16},
    },
    # -- services ------------------------------------------------------------ #
    {
        "mcc": "8099", "label": "Health & medical services", "weight": 0.026,
        "median": 1150.0, "sigma": 1.00, "agentic_affinity": 0.4,
        "ch": {"card_present": 0.54, "ecom": 0.26, "upi_p2m": 0.20},
    },
    {
        "mcc": "8220", "label": "Colleges & schools", "weight": 0.014,
        "median": 12500.0, "sigma": 1.05, "agentic_affinity": 0.6,
        "ch": {"card_present": 0.12, "ecom": 0.78, "moto": 0.10},
    },
    # -- person to person ---------------------------------------------------- #
    {
        "mcc": "6012", "label": "Financial institutions (P2P transfer)", "weight": 0.045,
        "median": 1500.0, "sigma": 1.30, "agentic_affinity": 0.1,
        "ch": {"upi_p2p": 1.0},
    },
)
# fmt: on


def _default_mcc_profiles() -> list[MccProfile]:
    """Materialise ``_MCC_PRIORS`` into validated profiles (median -> ln median)."""
    import math

    return [
        MccProfile(
            mcc=p["mcc"],
            label=p["label"],
            weight=float(p["weight"]),
            log_amount_mu=math.log(float(p["median"])),
            log_amount_sigma=float(p["sigma"]),
            channel_weights=dict(p["ch"]),
            agentic_affinity=float(p["agentic_affinity"]),
        )
        for p in _MCC_PRIORS
    ]


# Hour-of-day activity, share of daily volume. Indian retail/UPI shape: a dead
# 02:00-05:00 trough, a late-morning ramp, a lunch shoulder around 12:00-13:00,
# and the true peak in the 19:00-21:00 evening window. Normalised on load.
# fmt: off
_HOUR_PRIOR: Final[tuple[float, ...]] = (
    0.008, 0.005, 0.004, 0.003, 0.004, 0.007,
    0.014, 0.024, 0.036, 0.048, 0.058, 0.064,
    0.068, 0.066, 0.058, 0.054, 0.055, 0.061,
    0.069, 0.075, 0.073, 0.059, 0.036, 0.019,
)
# fmt: on

# Day of week, Monday=0. Mild weekday floor with a Saturday peak; Sunday falls
# back because a chunk of weekday volume is commute, fuel and workplace food.
_DOW_PRIOR: Final[tuple[float, ...]] = (0.132, 0.133, 0.136, 0.140, 0.156, 0.166, 0.137)

# How the credential reached the acquirer, conditional on channel. Contactless
# has overtaken chip at the Indian POS for low-value taps; card-not-present
# splits between one-off keyed entry, stored credentials and network tokens.
_ENTRY_MODE_PRIOR: Final[dict[str, dict[str, float]]] = {
    Channel.CARD_PRESENT.value: {
        EntryMode.CONTACTLESS.value: 0.46,
        EntryMode.CHIP.value: 0.50,
        EntryMode.MAGSTRIPE.value: 0.04,
    },
    Channel.ECOM.value: {
        EntryMode.ECOM_KEYED.value: 0.55,
        EntryMode.CREDENTIAL_ON_FILE.value: 0.25,
        EntryMode.NETWORK_TOKEN.value: 0.20,
    },
    Channel.MOTO.value: {EntryMode.ECOM_KEYED.value: 1.0},
    Channel.RECURRING.value: {
        EntryMode.CREDENTIAL_ON_FILE.value: 0.75,
        EntryMode.NETWORK_TOKEN.value: 0.25,
    },
    Channel.UPI_P2M.value: {
        EntryMode.QR_SCAN.value: 0.88,
        EntryMode.ECOM_KEYED.value: 0.12,
    },
    Channel.UPI_P2P.value: {
        EntryMode.QR_SCAN.value: 0.35,
        EntryMode.ECOM_KEYED.value: 0.65,
    },
    Channel.AGENTIC.value: {EntryMode.AGENT_TOKEN.value: 1.0},
}

# 3-D Secure outcome, conditional on channel. Card-present and UPI never invoke
# it. Ecom is mostly frictionless under exemptions, with a real challenge tail.
# The agentic rail is modelled as frictionless-dominant: the mandate *is* the
# authentication, which is exactly why F1-09 (human-present spoofing) is worth
# money to an attacker.
_THREEDS_PRIOR: Final[dict[str, dict[str, float]]] = {
    Channel.CARD_PRESENT.value: {ThreeDSResult.NOT_APPLICABLE.value: 1.0},
    Channel.UPI_P2M.value: {ThreeDSResult.NOT_APPLICABLE.value: 1.0},
    Channel.UPI_P2P.value: {ThreeDSResult.NOT_APPLICABLE.value: 1.0},
    Channel.MOTO.value: {ThreeDSResult.NOT_APPLICABLE.value: 1.0},
    Channel.ECOM.value: {
        ThreeDSResult.FRICTIONLESS.value: 0.72,
        ThreeDSResult.CHALLENGE_PASSED.value: 0.22,
        ThreeDSResult.ATTEMPTED.value: 0.03,
        ThreeDSResult.CHALLENGE_FAILED.value: 0.02,
        ThreeDSResult.UNAVAILABLE.value: 0.01,
    },
    Channel.RECURRING.value: {
        ThreeDSResult.NOT_APPLICABLE.value: 0.85,
        ThreeDSResult.FRICTIONLESS.value: 0.15,
    },
    # Agentic authorisations present as card-not-present carrying a delegated
    # authentication. Most clear frictionless or under a mandate-based exemption
    # (not_applicable); a minority step up; a thin tail fails or is unavailable
    # exactly as it does on ecom. The first cut of this gave agentic 86%
    # frictionless and *no* failure tail at all, which made three 3DS outcomes
    # 100% non-agentic and turned threeds_result into a 0.86-AUC rail detector.
    Channel.AGENTIC.value: {
        ThreeDSResult.FRICTIONLESS.value: 0.58,
        ThreeDSResult.NOT_APPLICABLE.value: 0.24,
        ThreeDSResult.CHALLENGE_PASSED.value: 0.13,
        ThreeDSResult.ATTEMPTED.value: 0.03,
        ThreeDSResult.CHALLENGE_FAILED.value: 0.015,
        ThreeDSResult.UNAVAILABLE.value: 0.005,
    },
}

# Metro anchors with rough shares of card-accepting volume. Not census
# population: the four biggest metros are over-weighted relative to headcount
# because card and agent penetration follow income, not people.
# fmt: off
_CITY_PRIOR: Final[tuple[dict[str, Any], ...]] = (
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777, "weight": 0.135},
    {"name": "Delhi", "lat": 28.6139, "lon": 77.2090, "weight": 0.130},
    {"name": "Bengaluru", "lat": 12.9716, "lon": 77.5946, "weight": 0.115},
    {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867, "weight": 0.075},
    {"name": "Chennai", "lat": 13.0827, "lon": 80.2707, "weight": 0.070},
    {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639, "weight": 0.062},
    {"name": "Pune", "lat": 18.5204, "lon": 73.8567, "weight": 0.058},
    {"name": "Ahmedabad", "lat": 23.0225, "lon": 72.5714, "weight": 0.048},
    {"name": "Jaipur", "lat": 26.9124, "lon": 75.7873, "weight": 0.032},
    {"name": "Surat", "lat": 21.1702, "lon": 72.8311, "weight": 0.028},
    {"name": "Lucknow", "lat": 26.8467, "lon": 80.9462, "weight": 0.026},
    {"name": "Indore", "lat": 22.7196, "lon": 75.8577, "weight": 0.022},
    {"name": "Kochi", "lat": 9.9312, "lon": 76.2673, "weight": 0.022},
    {"name": "Nagpur", "lat": 21.1458, "lon": 79.0882, "weight": 0.020},
    {"name": "Chandigarh", "lat": 30.7333, "lon": 76.7794, "weight": 0.020},
    {"name": "Coimbatore", "lat": 11.0168, "lon": 76.9558, "weight": 0.020},
    {"name": "Kanpur", "lat": 26.4499, "lon": 80.3319, "weight": 0.018},
    {"name": "Visakhapatnam", "lat": 17.6868, "lon": 83.2185, "weight": 0.017},
    {"name": "Bhopal", "lat": 23.2599, "lon": 77.4126, "weight": 0.016},
    {"name": "Patna", "lat": 25.5941, "lon": 85.1376, "weight": 0.016},
    {"name": "Guwahati", "lat": 26.1445, "lon": 91.7362, "weight": 0.010},
    {"name": "Bhubaneswar", "lat": 20.2961, "lon": 85.8245, "weight": 0.010},
)
# fmt: on

# Synthetic issuer BINs. Six-digit prefixes from the payment-sandbox conventions
# (the 4111.., 5555.., 4242.. family) plus RuPay-shaped 6-series values. The
# network split leans debit-heavy, which is the Indian reality.
# fmt: off
_BIN_PRIOR: Final[tuple[dict[str, Any], ...]] = (
    {"card_bin": "555555", "network": "mastercard", "weight": 0.060},
    {"card_bin": "510510", "network": "mastercard", "weight": 0.058},
    {"card_bin": "520082", "network": "mastercard", "weight": 0.055},
    {"card_bin": "222100", "network": "mastercard", "weight": 0.052},
    {"card_bin": "545454", "network": "mastercard", "weight": 0.055},
    {"card_bin": "530606", "network": "mastercard", "weight": 0.050},
    {"card_bin": "411111", "network": "visa", "weight": 0.064},
    {"card_bin": "424242", "network": "visa", "weight": 0.062},
    {"card_bin": "400000", "network": "visa", "weight": 0.058},
    {"card_bin": "453900", "network": "visa", "weight": 0.058},
    {"card_bin": "465944", "network": "visa", "weight": 0.056},
    {"card_bin": "421765", "network": "visa", "weight": 0.052},
    {"card_bin": "607400", "network": "rupay", "weight": 0.085},
    {"card_bin": "652855", "network": "rupay", "weight": 0.080},
    {"card_bin": "508500", "network": "rupay", "weight": 0.078},
    {"card_bin": "817200", "network": "rupay", "weight": 0.077},
)
# fmt: on

# Acquirer country of the merchant. Domestic-dominant with a thin cross-border
# tail, all of which is card-not-present — a physical terminal abroad would be
# travel, which the simulator models separately as a home-city excursion.
_MERCHANT_COUNTRY_PRIOR: Final[dict[str, float]] = {
    "IN": 0.962,
    "US": 0.009,
    "AE": 0.007,
    "SG": 0.006,
    "GB": 0.005,
    "NL": 0.003,
    "HK": 0.002,
    "AU": 0.002,
    "MY": 0.002,
    "JP": 0.002,
}

# Agent runtimes. Deliberately generic synthetic vendor names: naming real
# products in a file that also drives an attack simulator is exactly the kind of
# thing HARD RULE 5 exists to prevent.
_AGENT_PLATFORM_PRIOR: Final[dict[str, float]] = {
    "agentpay-runtime": 0.30,
    "shopbot-cloud": 0.24,
    "assistant-hosted": 0.20,
    "oem-device-agent": 0.14,
    "merchant-embedded": 0.12,
}

# Where each platform's agent actually executes. An on-device agent transacts
# from the customer's existing device_id; a hosted runtime carries its own.
# This is what stops device_id being a perfect rail separator: without a
# meaningful on-device share, every device in the population is either 100%
# agentic or 0% agentic and a model reads the rail straight off the device.
# Weighted mean lands near 50%: consumer agentic commerce is overwhelmingly
# phone-mediated, so about half of agent activity executes on hardware the
# cardholder already owns and already transacts from.
_AGENT_ON_DEVICE_PRIOR: Final[dict[str, float]] = {
    "agentpay-runtime": 0.40,
    "shopbot-cloud": 0.20,
    "assistant-hosted": 0.70,
    "oem-device-agent": 0.97,
    "merchant-embedded": 0.45,
}

# Mandate granularity for *legitimate* agentic traffic. Cart mandates dominate
# because most agentic commerce today is "buy this specific basket"; intent
# mandates are the growth case and the wider blast radius.
_MANDATE_TYPE_PRIOR: Final[dict[str, float]] = {
    MandateType.CART.value: 0.62,
    MandateType.INTENT.value: 0.30,
    MandateType.PAYMENT.value: 0.08,
}

# --------------------------------------------------------------------------- #
# Transaction lifecycle (schema amendment 1.1.0)
# --------------------------------------------------------------------------- #

#: Per-channel authorisation decline rate. Card-present is the cleanest rail --
#: the credential is physically there and the terminal pre-screens -- while
#: card-not-present carries the well-known double-digit decline rate the whole
#: "false decline" industry exists because of. The agentic rail sits a touch
#: above ecom: issuers have no history on it, so they are cautious. These are the
#: rates an *adaptive* adversary is measuring, so a background that approved
#: everything would make F4-27's oracle a pure generator artefact.
_DECLINE_RATE_PRIOR: Final[dict[str, float]] = {
    Channel.CARD_PRESENT.value: 0.028,
    Channel.ECOM.value: 0.115,
    Channel.MOTO.value: 0.140,
    Channel.RECURRING.value: 0.095,
    Channel.UPI_P2M.value: 0.052,
    Channel.UPI_P2P.value: 0.061,
    Channel.AGENTIC.value: 0.128,
}

#: Why an authorisation was declined. Insufficient funds dominates in a
#: debit-heavy market; ``declined_risk`` is deliberately the smallest slice,
#: because an issuer risk decline is rare next to a plain funding failure -- and
#: because F4-28's operator is hunting for exactly that slice.
_DECLINE_REASON_PRIOR: Final[dict[str, float]] = {
    AuthResponse.DECLINED_INSUFFICIENT_FUNDS.value: 0.46,
    AuthResponse.DECLINED_DO_NOT_HONOR.value: 0.27,
    AuthResponse.DECLINED_INVALID_CVV.value: 0.13,
    AuthResponse.DECLINED_EXPIRED.value: 0.08,
    AuthResponse.DECLINED_RISK.value: 0.06,
}

#: Median hours from authorisation to clearing, per channel. Card rails clear on
#: the next acquirer file; UPI settles in near-real time. The **bimodality** is
#: load-bearing: it is what stops "settled in minutes" from being a free
#: detector for F1-03's instant-settlement refunds.
_SETTLEMENT_LAG_PRIOR: Final[dict[str, float]] = {
    Channel.CARD_PRESENT.value: 26.0,
    Channel.ECOM.value: 31.0,
    Channel.MOTO.value: 34.0,
    Channel.RECURRING.value: 25.0,
    Channel.UPI_P2M.value: 0.09,
    Channel.UPI_P2P.value: 0.03,
    Channel.AGENTIC.value: 22.0,
}

#: Delegation depth for legitimate agentic traffic. Thins out rather than
#: stopping: see ``ReferenceStats.delegation_depth_weights``.
_DELEGATION_DEPTH_PRIOR: Final[dict[str, float]] = {
    "1": 0.780,
    "2": 0.170,
    "3": 0.035,
    "4": 0.012,
    "5": 0.003,
}

_DEFAULT_PROVENANCE: Final[dict[str, str]] = {
    "amounts": "Indian-market priors; per-MCC log-normal medians in INR, practitioner estimates "
    "consistent with published RBI/NPCI ticket-size aggregates.",
    "hour_dow": "Indian retail/UPI diurnal shape; evening-peak prior.",
    "channel_mix": "Derived per MCC from the Indian acceptance mix (POS / CNP / UPI P2M).",
    "merchant_popularity": "Zipf rank-frequency, exponent 1.08, the usual range for "
    "merchant-acceptance long tails.",
    "geography": "Metro anchors weighted by card-accepting volume, not headcount.",
    "agentic": "Modelled, not measured. There is no public agentic-payments panel; "
    "manufacturing this is the entire point of the project.",
    "lifecycle": "Decline rates, settlement lag and dispute rates are practitioner "
    "estimates for the Indian card/UPI mix; the shape that matters is the "
    "card-versus-UPI settlement bimodality, not the exact medians.",
}


# --------------------------------------------------------------------------- #
# The calibration object
# --------------------------------------------------------------------------- #


class ReferenceStats(BaseModel):
    """Every distribution the population simulator draws from, in one object.

    Every field is defaulted, so ``ReferenceStats()`` is the prior-only build and
    a partial JSON override inherits the rest. ``source`` and ``provenance``
    travel with the object into the run manifest and the fidelity scorecard, so
    a reader always knows which numbers were fitted and which were asserted.

    ``validate_default=True`` is not decoration. Pydantic skips field validators
    on defaults unless told otherwise, and every weight block in this model is
    normalised *by* its validator — so without it the priors would reach the
    simulator un-normalised and every ``rng.choice`` would reject them.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True, validate_default=True)

    source: str = Field(
        default="indian-market-priors",
        description="'indian-market-priors' or 'fitted:<dataset>'.",
    )
    currency: str = Field(default="INR", description="ISO 4217 code amounts are expressed in.")
    provenance: dict[str, str] = Field(default_factory=lambda: dict(_DEFAULT_PROVENANCE))

    # -- what is bought, for how much, where ---------------------------------- #
    mcc_profiles: list[MccProfile] = Field(default_factory=_default_mcc_profiles, min_length=1)
    cities: list[CityPoint] = Field(
        default_factory=lambda: [CityPoint(**c) for c in _CITY_PRIOR], min_length=1
    )
    merchant_country_weights: dict[str, float] = Field(
        default_factory=lambda: dict(_MERCHANT_COUNTRY_PRIOR)
    )
    card_bins: list[BinProfile] = Field(
        default_factory=lambda: [BinProfile(**b) for b in _BIN_PRIOR], min_length=1
    )

    # -- when ----------------------------------------------------------------- #
    hour_weights: list[float] = Field(
        default_factory=lambda: list(_HOUR_PRIOR), min_length=24, max_length=24
    )
    dow_weights: list[float] = Field(
        default_factory=lambda: list(_DOW_PRIOR), min_length=7, max_length=7
    )
    agentic_hour_uniform_blend: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description=(
            "Agents do not sleep. Agentic timestamps are drawn from the human hour "
            "curve blended this far toward uniform — a modelled assumption, and one "
            "the fidelity scorecard flags as unvalidated."
        ),
    )

    # -- how ------------------------------------------------------------------ #
    entry_mode_weights: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {k: dict(v) for k, v in _ENTRY_MODE_PRIOR.items()}
    )
    threeds_weights: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {k: dict(v) for k, v in _THREEDS_PRIOR.items()}
    )

    # -- population structure -------------------------------------------------- #
    merchant_zipf_exponent: float = Field(
        default=1.08,
        gt=0,
        description="Merchant rank-frequency exponent: p(rank) proportional to rank**-a.",
    )
    customer_rate_gamma_shape: float = Field(
        default=1.6,
        gt=0,
        description="Gamma shape for per-customer daily transaction rate. <2 keeps the "
        "heavy tail that makes velocity features informative.",
    )
    customer_rate_mean_per_day: float = Field(default=0.45, gt=0)
    burst_probability: float = Field(
        default=0.14,
        ge=0.0,
        le=1.0,
        description="Chance a customer's transaction is pulled into a short session "
        "burst behind its predecessor, rather than sitting where the diurnal "
        "curve put it. Real cardholders shop in sessions.",
    )
    travel_probability: float = Field(
        default=0.055,
        ge=0.0,
        le=1.0,
        description="Chance a transaction happens outside the customer's home metro.",
    )
    remote_geo_missing_p: float = Field(
        default=0.22,
        ge=0.0,
        le=1.0,
        description=(
            "Chance a card-not-present authorisation reaches the issuer with no "
            "usable lat/lon. Real auth streams are not fully populated, and a "
            "synthetic file with geo on every single row is a tell. Card-present "
            "rows always carry terminal geo, so this applies to remote rails only."
        ),
    )

    # -- the agentic rail (modelled, not measured) ---------------------------- #
    agentic_share: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Share of all events on the agentic rail."
    )
    agent_adoption_rate: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Share of customers who use an agent at all. Adoption is "
        "concentrated, not spread thinly: that concentration is what makes "
        "per-agent graph features (L4) meaningful.",
    )
    agent_platform_weights: dict[str, float] = Field(
        default_factory=lambda: dict(_AGENT_PLATFORM_PRIOR)
    )
    agent_on_device_p: dict[str, float] = Field(
        default_factory=lambda: dict(_AGENT_ON_DEVICE_PRIOR),
        description="Platform -> probability the agent runs on the customer's own device.",
    )
    agent_propensity_alpha: float = Field(
        default=0.6,
        gt=0,
        description=(
            "Beta shape a for per-customer agent propensity. Below 1 piles mass "
            "near zero, which is what concentrated adoption looks like."
        ),
    )
    agent_propensity_beta: float = Field(
        default=2.0,
        gt=0,
        description=(
            "Beta shape b for per-customer agent propensity. The pair (0.6, 2.0) "
            "keeps adoption concentrated without giving any customer a hard zero -- "
            "a hard zero made customer_id a 0.90-AUC predictor of the rail."
        ),
    )
    mandate_type_weights: dict[str, float] = Field(
        default_factory=lambda: dict(_MANDATE_TYPE_PRIOR)
    )
    delegation_depth_weights: dict[str, float] = Field(
        default_factory=lambda: dict(_DELEGATION_DEPTH_PRIOR),
        description=(
            "Agent-to-sub-agent hops between the human and the payment. Day 1 "
            "capped legitimate traffic at three, which made depth>=4 a perfect "
            "detector for F1-05 -- a property of the generator, not of "
            "delegation laundering. Multi-agent orchestration genuinely does "
            "produce deep chains legitimately (a shopping agent calling a "
            "booking agent calling a checkout capability is three hops before "
            "anything unusual has happened), so the tail now runs to five and "
            "thins out rather than stopping."
        ),
    )

    # -- transaction lifecycle (schema amendment 1.1.0) ----------------------- #
    decline_rate: dict[str, float] = Field(
        default_factory=lambda: dict(_DECLINE_RATE_PRIOR),
        description="Channel -> share of authorisations declined. NOT normalised: "
        "each entry is an independent probability, not a share of a whole.",
    )
    decline_reason_weights: dict[str, float] = Field(
        default_factory=lambda: dict(_DECLINE_REASON_PRIOR)
    )
    decline_amount_tilt: float = Field(
        default=0.55,
        ge=0.0,
        description=(
            "How much a large ticket raises the odds of a decline, as a "
            "coefficient on the log-amount z-score. Non-zero on purpose: a "
            "decline rate independent of amount would let a detector treat "
            "declines as pure noise, when in reality they carry signal."
        ),
    )
    settlement_lag_median_hours: dict[str, float] = Field(
        default_factory=lambda: dict(_SETTLEMENT_LAG_PRIOR),
        description="Channel -> median hours from authorisation to clearing.",
    )
    settlement_lag_sigma: float = Field(
        default=0.55, gt=0, description="Dispersion of ln(settlement lag)."
    )
    unsettled_share: float = Field(
        default=0.012,
        ge=0.0,
        le=1.0,
        description="Approved authorisations that never clear inside the window: "
        "abandoned pre-auths, expired holds, late acquirer files.",
    )
    refund_share: float = Field(
        default=0.021,
        ge=0.0,
        le=0.2,
        description="Share of events that are refunds against an earlier purchase. "
        "Around 2% of retail card volume comes back.",
    )
    reversal_share: float = Field(
        default=0.006,
        ge=0.0,
        le=0.2,
        description="Share of events that are authorisation reversals, cancelled "
        "before clearing. Never settle, by definition.",
    )
    credit_share: float = Field(
        default=0.0016,
        ge=0.0,
        le=0.2,
        description="Outbound credits with no purchase behind them: goodwill "
        "payments, promotional credits. Small, but it must be non-zero -- a "
        "txn_type level absent from the background would be a free label for "
        "any attack that used it.",
    )
    preauth_share: float = Field(
        default=0.024,
        ge=0.0,
        le=0.2,
        description="Pre-authorisation holds: fuel, hotels, car hire.",
    )
    refund_full_share: float = Field(
        default=0.68, ge=0.0, le=1.0, description="Refunds returning the whole original amount."
    )
    refund_lag_median_hours: float = Field(
        default=62.0, gt=0, description="Median hours from a purchase to its refund."
    )
    refund_instant_share: float = Field(
        default=0.24,
        ge=0.0,
        le=1.0,
        description=(
            "Share of legitimate refunds that clear in minutes rather than on "
            "the next acquirer file. Instant refunds are a real and growing "
            "merchant offering, and this tail is load-bearing: F1-03's whole "
            "operational goal is a credit that settles before a human looks, so "
            "if no legitimate refund ever settled fast, 'settled fast' would be "
            "a one-column detector at 0.99 AUC and the attack would be a "
            "cartoon. Same argument as the card-versus-UPI settlement "
            "bimodality, applied to the refund path."
        ),
    )
    dispute_rate: float = Field(
        default=0.0009,
        ge=0.0,
        le=1.0,
        description="Share of settled purchases the cardholder disputes. Basis "
        "points, as in reality -- and post-hoc, so never a scoring feature.",
    )
    dispute_won_cardholder_share: float = Field(
        default=0.62, ge=0.0, le=1.0, description="Of resolved disputes, share the cardholder wins."
    )
    dispute_unresolved_share: float = Field(
        default=0.18,
        ge=0.0,
        le=1.0,
        description="Disputes still open at the end of the window. A file where "
        "every dispute has already resolved would be a file from the future.",
    )
    human_present_passive_share: float = Field(
        default=0.11,
        ge=0.0,
        le=1.0,
        description=(
            "Share of genuinely human-present agentic sessions where the person "
            "is watching but not touching -- reading on a second screen while the "
            "agent drives -- so the telemetry looks machine-like anyway. Without "
            "this tail, (human_present=True, low cursor_entropy) would be a "
            "perfect deterministic detector for F1-09, which would be a property "
            "of our generator rather than of spoofing."
        ),
    )

    # -- validation & normalisation -------------------------------------------- #

    @field_validator("currency")
    @classmethod
    def _check_currency(cls, v: str) -> str:
        if len(v) != 3 or not v.isalpha() or not v.isupper():
            raise ValueError(f"currency must be a 3-letter uppercase ISO 4217 code, got {v!r}")
        return v

    @field_validator("hour_weights")
    @classmethod
    def _norm_hours(cls, v: list[float]) -> list[float]:
        return cls._norm_list(v, "hour_weights")

    @field_validator("dow_weights")
    @classmethod
    def _norm_dow(cls, v: list[float]) -> list[float]:
        return cls._norm_list(v, "dow_weights")

    @staticmethod
    def _norm_list(v: list[float], name: str) -> list[float]:
        if any(x < 0 for x in v):
            raise ValueError(f"{name} must be non-negative")
        total = sum(v)
        if total <= 0:
            raise ValueError(f"{name} must sum to a positive number")
        return [x / total for x in v]

    @field_validator("merchant_country_weights", "agent_platform_weights")
    @classmethod
    def _norm_simple(cls, v: dict[str, float]) -> dict[str, float]:
        return _normalise(v)

    @field_validator("delegation_depth_weights")
    @classmethod
    def _norm_delegation(cls, v: dict[str, float]) -> dict[str, float]:
        bad = [k for k in v if not k.isdigit() or int(k) < 1]
        if bad:
            raise ValueError(f"delegation depths must be positive integers, got {bad}")
        return _normalise(v)

    @field_validator("decline_reason_weights")
    @classmethod
    def _norm_decline_reasons(cls, v: dict[str, float]) -> dict[str, float]:
        valid = {r.value for r in AuthResponse if r is not AuthResponse.APPROVED}
        unknown = set(v) - valid
        if unknown:
            raise ValueError(f"unknown decline reasons {sorted(unknown)}")
        return _normalise(v)

    @field_validator("decline_rate", "settlement_lag_median_hours")
    @classmethod
    def _check_per_channel(cls, v: dict[str, float]) -> dict[str, float]:
        """Per-channel probabilities and lags. Every channel must be covered."""
        missing = {c.value for c in Channel} - set(v)
        if missing:
            raise ValueError(f"no entry for channel(s) {sorted(missing)}")
        bad = {k: x for k, x in v.items() if x < 0}
        if bad:
            raise ValueError(f"negative values: {bad}")
        return v

    @field_validator("mandate_type_weights")
    @classmethod
    def _norm_mandate(cls, v: dict[str, float]) -> dict[str, float]:
        valid = {m.value for m in MandateType}
        bad = sorted(set(v) - valid)
        if bad:
            raise ValueError(f"unknown mandate types {bad}; valid: {sorted(valid)}")
        if MandateType.NONE.value in v and v[MandateType.NONE.value] > 0:
            raise ValueError(
                "legitimate agentic traffic always carries a mandate; "
                "mandate_type='none' belongs to F1 attacks, not the base population"
            )
        return _normalise(v)

    @field_validator("entry_mode_weights")
    @classmethod
    def _norm_entry(cls, v: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        channels = {c.value for c in Channel}
        modes = {e.value for e in EntryMode}
        missing = sorted(channels - set(v))
        if missing:
            raise ValueError(f"entry_mode_weights missing channel(s) {missing}")
        out: dict[str, dict[str, float]] = {}
        for channel, inner in v.items():
            if channel not in channels:
                raise ValueError(f"unknown channel {channel!r} in entry_mode_weights")
            bad = sorted(set(inner) - modes)
            if bad:
                raise ValueError(f"unknown entry mode(s) {bad} for channel {channel!r}")
            out[channel] = _normalise(inner)
        return out

    @field_validator("threeds_weights")
    @classmethod
    def _norm_threeds(cls, v: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        channels = {c.value for c in Channel}
        results = {t.value for t in ThreeDSResult}
        missing = sorted(channels - set(v))
        if missing:
            raise ValueError(f"threeds_weights missing channel(s) {missing}")
        out: dict[str, dict[str, float]] = {}
        for channel, inner in v.items():
            if channel not in channels:
                raise ValueError(f"unknown channel {channel!r} in threeds_weights")
            bad = sorted(set(inner) - results)
            if bad:
                raise ValueError(f"unknown 3DS result(s) {bad} for channel {channel!r}")
            out[channel] = _normalise(inner)
        return out

    @model_validator(mode="after")
    def _check_agent_hosting(self) -> ReferenceStats:
        """Every platform needs a hosting probability, or its agents vanish on-device."""
        missing = sorted(set(self.agent_platform_weights) - set(self.agent_on_device_p))
        if missing:
            raise ValueError(f"agent_on_device_p missing platform(s) {missing}")
        bad = {k: v for k, v in self.agent_on_device_p.items() if not 0.0 <= v <= 1.0}
        if bad:
            raise ValueError(f"agent_on_device_p values must be probabilities, got {bad}")
        return self

    @model_validator(mode="after")
    def _norm_mcc_and_check_country(self) -> ReferenceStats:
        seen = [p.mcc for p in self.mcc_profiles]
        dupes = sorted({m for m in seen if seen.count(m) > 1})
        if dupes:
            raise ValueError(f"duplicate mcc profile(s) {dupes}")
        total = sum(p.weight for p in self.mcc_profiles)
        for profile in self.mcc_profiles:
            profile.weight = profile.weight / total
        for code in self.merchant_country_weights:
            if len(code) != 2 or not code.isalpha() or not code.isupper():
                raise ValueError(f"merchant country must be ISO 3166-1 alpha-2, got {code!r}")
        return self

    # -- convenience ----------------------------------------------------------- #

    @property
    def is_fitted(self) -> bool:
        """True when these numbers came off a dataset rather than out of this file."""
        return self.source.startswith("fitted:")

    def mcc_index(self) -> dict[str, MccProfile]:
        """Profiles keyed by MCC."""
        return {p.mcc: p for p in self.mcc_profiles}

    def describe(self) -> str:
        """One human-readable block, printed by the simulator and the fitter."""
        lines = [
            f"reference stats: source={self.source}  currency={self.currency}",
            f"  mcc profiles      : {len(self.mcc_profiles)}",
            f"  cities            : {len(self.cities)}",
            f"  card bins         : {len(self.card_bins)}",
            f"  merchant zipf a   : {self.merchant_zipf_exponent:.3f}",
            f"  cust rate/day     : mean {self.customer_rate_mean_per_day:.3f}, "
            f"gamma shape {self.customer_rate_gamma_shape:.2f}",
            f"  agentic share     : {self.agentic_share:.1%} "
            f"(mean per-customer propensity {self.agent_adoption_rate:.0%})",
            "  provenance:",
        ]
        lines.extend(f"    {k:<20} {v}" for k, v in sorted(self.provenance.items()))
        return "\n".join(lines)


def load_reference_stats(path: Path | None = None) -> ReferenceStats:
    """Load calibration, preferring a fitted JSON and falling back to the priors.

    Never raises on a *missing* file — that is the clean-clone path and it is
    supposed to work (HARD RULE 4). A file that exists but is malformed *does*
    raise, because silently reverting to priors after someone deliberately fitted
    the stats would make the fidelity scorecard lie about its own provenance.

    Args:
        path: Override for the stats file. Defaults to
            ``data/reference/reference_stats.json``.

    Returns:
        Validated, weight-normalised calibration.
    """
    target = REFERENCE_STATS_JSON if path is None else path
    if not target.is_file():
        return ReferenceStats()

    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{target}: reference stats present but unreadable: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{target}: expected a JSON object, got {type(raw).__name__}")

    stats = ReferenceStats.model_validate(raw)
    if stats.source == "indian-market-priors":
        # A file that forgot to say where it came from still gets a truthful label.
        stats.source = f"fitted:{target.name}"
    return stats


def main() -> None:
    """Print the active calibration. Run: ``python -m mantis.foundry.base.reference``."""
    stats = load_reference_stats()
    print(stats.describe())
    print()
    print(f"  {'mcc':<6} {'weight':>7} {'median':>10} {'sigma':>6} {'agentic':>8}  label")
    print(f"  {'-' * 6} {'-' * 7} {'-' * 10} {'-' * 6} {'-' * 8}  {'-' * 38}")
    import math

    for profile in sorted(stats.mcc_profiles, key=lambda p: -p.weight):
        median = math.exp(profile.log_amount_mu)
        print(
            f"  {profile.mcc:<6} {profile.weight:>7.4f} {median:>10.0f} "
            f"{profile.log_amount_sigma:>6.2f} {profile.agentic_affinity:>8.2f}  {profile.label}"
        )
    if not stats.is_fitted:
        print()
        print("  Using priors. Run scripts/fit_reference.py to fit from a reference CSV.")


if __name__ == "__main__":
    main()
