"""Entity aggregates: what this customer, merchant and BIN normally look like.

Velocity asks "how much has happened recently on this key". This module asks the
different question "how does *this* event compare to what this key normally
does" — and the difference matters, because most of the atlas is not about
volume. F3-19's coerced transfer is one event; it is not fast, it is *large for
this victim*, and only a per-customer baseline can say so.

Fitted on train only
---------------------
Every profile here is estimated from the training split and then applied
unchanged to the test split. This is the standard discipline and it is not
optional: a per-customer mean computed over the whole file includes the test
period, so a customer's own future spending would be baked into the baseline
their test-period events are scored against. That inflates every metric and is
invisible unless you look for it.

Entities unseen at fit time get NaN rather than a global fallback. A brand-new
customer genuinely *is* a different situation from an established one — that is
the whole of F2-13 (synthetic identity onboarding) — and collapsing the two by
substituting a population average would erase the signal the attack is made of.

The two ratios the amendment made possible
--------------------------------------------
``refund-to-purchase ratio`` per customer and per merchant. A merchant whose
outbound flow is a large share of its inbound flow is either a returns-heavy
retailer or a laundering leg, and F1-03 and F6-39 both live in that gap.
Unbuildable before 1.1.0, for the same reason the decline ratios were.

``settlement-lag deviation from the rail's own mode``. Note the subtlety: the
*current* event's settlement lag is not known when it is scored (see
:data:`mantis.defense.features.spec.FUTURE_COLUMNS`), so this is not that. It is
the deviation of the **merchant's historical mean lag** from the modal lag of the
rail it sits on — a property of the merchant, established before this event, and
one that separates a merchant clearing on the normal acquirer file from one whose
money moves unusually fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

from mantis.core.events import DECLINE_RESPONSES
from mantis.defense.features.transaction import OUTBOUND_TYPES

__all__ = ["EntityProfiles", "entity_features"]

#: Minimum events before a per-entity statistic is trusted. Below this the
#: estimate is noise, and a noisy baseline is worse than none: it manufactures
#: extreme z-scores for entities that simply have not been seen much.
_MIN_EVENTS: Final[int] = 5


@dataclass(slots=True)
class EntityProfiles:
    """Per-entity baselines, fitted on the training split only."""

    customer_log_amount_mean: dict[str, float] = field(default_factory=dict)
    customer_log_amount_std: dict[str, float] = field(default_factory=dict)
    customer_amount_p99: dict[str, float] = field(default_factory=dict)
    customer_refund_ratio: dict[str, float] = field(default_factory=dict)
    customer_n: dict[str, float] = field(default_factory=dict)

    merchant_log_amount_mean: dict[str, float] = field(default_factory=dict)
    merchant_log_amount_std: dict[str, float] = field(default_factory=dict)
    merchant_refund_ratio: dict[str, float] = field(default_factory=dict)
    merchant_decline_ratio: dict[str, float] = field(default_factory=dict)
    merchant_customers: dict[str, float] = field(default_factory=dict)
    merchant_lag_mean: dict[str, float] = field(default_factory=dict)
    merchant_n: dict[str, float] = field(default_factory=dict)

    bin_decline_ratio: dict[str, float] = field(default_factory=dict)
    mcc_log_amount_mean: dict[str, float] = field(default_factory=dict)
    mcc_log_amount_std: dict[str, float] = field(default_factory=dict)
    #: Modal settlement lag per rail. The reference calls this bimodal on
    #: purpose: UPI clears in seconds, card rails on tomorrow's file.
    channel_lag_median: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> EntityProfiles:
        """Estimate every baseline from ``frame``. Call on the TRAIN split only."""
        work = frame.copy()
        work["_log_amount"] = np.log1p(work["amount"].to_numpy(dtype=float))
        work["_outbound"] = np.isin(work["txn_type"].to_numpy(), OUTBOUND_TYPES)
        work["_declined"] = np.isin(work["auth_response"].to_numpy(), DECLINE_RESPONSES)

        def _keep(series: pd.Series, counts: pd.Series) -> dict[str, float]:
            """Drop entities with too little history to estimate anything."""
            kept = series[counts.reindex(series.index).fillna(0) >= _MIN_EVENTS]
            return {str(k): float(v) for k, v in kept.items() if v == v}

        by_customer = work.groupby("customer_id", observed=True)
        customer_n = by_customer.size()
        by_merchant = work.groupby("merchant_id", observed=True)
        merchant_n = by_merchant.size()

        settled = work[work["settlement_lag_hours"].notna()]

        return cls(
            customer_log_amount_mean=_keep(by_customer["_log_amount"].mean(), customer_n),
            customer_log_amount_std=_keep(by_customer["_log_amount"].std(), customer_n),
            customer_amount_p99=_keep(by_customer["amount"].quantile(0.99), customer_n),
            customer_refund_ratio=_keep(by_customer["_outbound"].mean(), customer_n),
            customer_n={str(k): float(v) for k, v in customer_n.items()},
            merchant_log_amount_mean=_keep(by_merchant["_log_amount"].mean(), merchant_n),
            merchant_log_amount_std=_keep(by_merchant["_log_amount"].std(), merchant_n),
            merchant_refund_ratio=_keep(by_merchant["_outbound"].mean(), merchant_n),
            merchant_decline_ratio=_keep(by_merchant["_declined"].mean(), merchant_n),
            merchant_customers=_keep(by_merchant["customer_id"].nunique(), merchant_n),
            merchant_lag_mean=_keep(
                settled.groupby("merchant_id", observed=True)["settlement_lag_hours"].mean(),
                merchant_n,
            ),
            merchant_n={str(k): float(v) for k, v in merchant_n.items()},
            bin_decline_ratio={
                str(k): float(v)
                for k, v in work.groupby("card_bin", observed=True)["_declined"].mean().items()
            },
            mcc_log_amount_mean={
                str(k): float(v)
                for k, v in work.groupby("mcc", observed=True)["_log_amount"].mean().items()
            },
            mcc_log_amount_std={
                str(k): float(v)
                for k, v in work.groupby("mcc", observed=True)["_log_amount"].std().items()
                if v == v
            },
            channel_lag_median={
                str(k): float(v)
                for k, v in settled.groupby("channel", observed=True)[
                    "settlement_lag_hours"
                ]
                .median()
                .items()
            },
        )


def _map(series: pd.Series, table: dict[str, float]) -> np.ndarray:
    """Look up a fitted statistic, NaN for entities not seen at fit time."""
    return series.astype(str).map(table).to_numpy(dtype=float)


def entity_features(frame: pd.DataFrame, profiles: EntityProfiles) -> pd.DataFrame:
    """Compare each event against the baselines fitted for its entities."""
    out = pd.DataFrame(index=frame.index)
    amount = frame["amount"].to_numpy(dtype=float)
    log_amount = np.log1p(amount)

    # -- the customer's own normal ------------------------------------------- #
    cust_mean = _map(frame["customer_id"], profiles.customer_log_amount_mean)
    cust_std = _map(frame["customer_id"], profiles.customer_log_amount_std)
    out["ent_customer_amount_z"] = (log_amount - cust_mean) / np.where(
        cust_std > 1e-6, cust_std, np.nan
    )
    cust_p99 = _map(frame["customer_id"], profiles.customer_amount_p99)
    # F3-19's signature: a coerced transfer sits at the very top of the victim's
    # own range without being remarkable against the population's.
    out["ent_amount_vs_customer_p99"] = amount / np.where(cust_p99 > 0, cust_p99, np.nan)
    out["ent_customer_refund_ratio"] = _map(frame["customer_id"], profiles.customer_refund_ratio)
    out["ent_customer_n_events"] = _map(frame["customer_id"], profiles.customer_n)
    # NaN means "never seen in training" — a new entity, which is F2-13's shape.
    out["ent_customer_unseen"] = np.isnan(out["ent_customer_n_events"].to_numpy()).astype(float)

    # -- the merchant's own normal -------------------------------------------- #
    merch_mean = _map(frame["merchant_id"], profiles.merchant_log_amount_mean)
    merch_std = _map(frame["merchant_id"], profiles.merchant_log_amount_std)
    out["ent_merchant_amount_z"] = (log_amount - merch_mean) / np.where(
        merch_std > 1e-6, merch_std, np.nan
    )
    out["ent_merchant_refund_ratio"] = _map(frame["merchant_id"], profiles.merchant_refund_ratio)
    out["ent_merchant_decline_ratio"] = _map(frame["merchant_id"], profiles.merchant_decline_ratio)
    out["ent_merchant_customers"] = _map(frame["merchant_id"], profiles.merchant_customers)
    out["ent_merchant_n_events"] = _map(frame["merchant_id"], profiles.merchant_n)
    out["ent_merchant_unseen"] = np.isnan(out["ent_merchant_n_events"].to_numpy()).astype(float)

    # -- settlement-lag deviation, computed the only legitimate way ------------ #
    # The merchant's HISTORICAL mean lag against its rail's modal lag. Not this
    # event's lag: that is a future fact. See the module docstring.
    merchant_lag = _map(frame["merchant_id"], profiles.merchant_lag_mean)
    rail_lag = frame["channel"].astype(str).map(profiles.channel_lag_median).to_numpy(dtype=float)
    out["ent_merchant_lag_vs_rail"] = np.where(
        rail_lag > 0, merchant_lag / rail_lag, np.nan
    )

    # -- the category's normal, and the BIN's ---------------------------------- #
    mcc_mean = _map(frame["mcc"], profiles.mcc_log_amount_mean)
    mcc_std = _map(frame["mcc"], profiles.mcc_log_amount_std)
    out["ent_mcc_amount_z"] = (log_amount - mcc_mean) / np.where(mcc_std > 1e-6, mcc_std, np.nan)
    out["ent_bin_decline_ratio"] = _map(frame["card_bin"], profiles.bin_decline_ratio)
    return out
