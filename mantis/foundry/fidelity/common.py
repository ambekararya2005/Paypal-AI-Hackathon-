"""The common space: the only frame in which two payment panels can be compared.

MANTIS generates Indian rupee card and UPI traffic with an agentic rail. The
reference panel is US dollar card traffic with no agentic rail at all. Almost
every column they nominally share means something different, and the two that do
not — amount and category — are the two where the difference is largest.

So nothing is compared raw. Both sides are projected into a **shape space** of
dimensionless features, and every metric in this package is computed there.

The projection, and the argument for each feature
--------------------------------------------------
=============================  =================================================
feature                        why it survives the currency and the border
=============================  =================================================
``hour``                       Diurnal retail rhythm is a human constant, and it
                               is measured in local time on both sides.
``dow``                        Same, weekly.
``log_amount_z``               ln(amount) centred on **its own category's**
                               median and scaled by that category's own IQR. A
                               fuel purchase is compared to fuel purchases. What
                               survives is dispersion, which is scale-free; the
                               location, which is not, is divided out.
``amount_vs_customer``         ln(amount) minus this cardholder's own mean
                               ln(amount). "Is this big for this person" -- the
                               single most transferable fraud signal there is.
``gap_ratio_log``              ln(seconds since this cardholder's previous
                               transaction) minus ln of **that cardholder's own
                               median gap**. Burstiness relative to a person's
                               own rhythm, so a panel of heavy users and a panel
                               of light ones are on the same axis.
``burst_1h``                   1 when this cardholder transacted at all in the
                               previous hour. A probability, not a count.
``merchant_rank_pct``          The merchant's popularity rank as a percentile of
                               the estate. Pure rank-frequency structure, which
                               is what a Zipf exponent summarises, and already
                               normalised by however many merchants exist.
``category_shift``             1 when the category differs from this
                               cardholder's previous transaction. Basket
                               volatility.
=============================  =================================================

**Every feature is backward-looking.** ``burst_1h`` and ``gap_ratio_log`` look
only at transactions strictly before the current one; ``merchant_rank_pct`` is
computed on the panel as a whole, which is the one exception and is stated as
such — a merchant's popularity is a property of the estate rather than of any
event, and both sides get the same treatment.

The rate is divided out, and then reported separately
------------------------------------------------------
The reference panel models **~920 heavy cardholders**: 2.45 transactions each per
day. This population models **~4,950 ordinary ones**: 0.37 a day. That is a 6.6x
difference in rate, and it is a difference in how the two panels were *composed*
rather than in whether either resembles a payment stream. It is reported by
:func:`panel_levels` as a labelled ratio with **no distance attached to it**, and
the shape space is built so that it does not leak into any distance.

Two features were removed rather than standardised, and both removals are
findings about how easy it is to fake a fidelity number
--------------------------------------------------------------------------------
``customer_txn_1h_z`` and ``customer_txn_24h_z`` — the trailing counts, centred
and scaled on each panel's own moments — were the first version of this list, and
they produced a **discriminator AUC of 1.000 on an artefact**. The counts are
discrete and overwhelmingly zero: 87.6% zero here, 83.4% zero in the reference.
Standardising maps that shared modal atom to ``-0.370`` on one panel and
``-0.411`` on the other, and a tree splitting between the two separates the
panels perfectly while the underlying distributions (0.876/0.121/0.003 against
0.834/0.142/0.021) are in fact close. **A z-score of a mostly-constant discrete
variable is a panel fingerprint, not a comparable quantity.** The measurement was
reporting the transform.

``customer_merchant_share`` — the share of a cardholder's history spent at one
merchant — has its *floor* set by history length: a cardholder with 1,300
transactions cannot spend less than 0.08% at one merchant, one with 18 cannot
spend less than 5.6%. No centring recovers a comparable quantity from it.

Both are recorded here rather than quietly deleted, because the general lesson
outlives them: **a normalisation chosen to make two panels comparable can create
the very separation it was meant to remove**, and a discriminator is the only
instrument in this package that would have caught it.

What is deliberately absent
---------------------------
Absolute amount, currency, category identity, MCC, geography, BIN, entry mode,
3-D Secure, channel, and every ``ag_`` column. Four of those would let a
discriminator separate real from synthetic at AUC 1.0 by reading a country off
the row, which would be a true statement about two datasets and a useless one
about fidelity. The rest do not exist on the reference side at all.

The agentic rail is excluded from the synthetic side for the same reason and it
is the sharpest one: **the reference panel has no agentic transactions, because
no panel does.** That absence is the entire premise of this project. Including
our agentic rows would hand the discriminator a free separation and hand TSTR a
fraud class with no analogue to be tested against.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

__all__ = [
    "CATEGORICAL_FEATURES",
    "CONTINUOUS_FEATURES",
    "SHAPE_FEATURES",
    "panel_levels",
    "to_common",
    "to_shape",
]

#: Continuous shape features, compared with a two-sample KS.
CONTINUOUS_FEATURES: Final[tuple[str, ...]] = (
    "log_amount_z",
    "amount_vs_customer",
    "gap_ratio_log",
    "merchant_rank_pct",
)

#: Categorical shape features, compared with a Jensen-Shannon divergence.
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = ("hour", "dow", "category_shift", "burst_1h")

#: Everything a discriminator or a TSTR model is allowed to see.
SHAPE_FEATURES: Final[tuple[str, ...]] = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

#: Classic rails only. The agentic rail is excluded from every comparison in this
#: package -- see the module docstring's last paragraph, which is the important
#: one.
CLASSIC_RAILS: Final[frozenset[str]] = frozenset(
    {"card_present", "ecom", "moto", "recurring", "upi_p2m", "upi_p2p"}
)


def to_common(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Project either side onto ``ts, amount, category, customer_id, merchant_id``.

    ``source`` is ``"synthetic"`` or ``"real"``. The synthetic side is filtered to
    the classic rails and its MCC becomes the category; the real side already has
    the five columns and passes through. Purchases only on both: a refund and a
    reversal are conversions of a purchase in this population and have no
    counterpart in the reference panel, so including them would compare a
    lifecycle the reference does not model.
    """
    if source == "real":
        out = frame[["ts", "amount", "category", "customer_id", "merchant_id"]].copy()
        out["is_fraud"] = frame["is_fraud"].to_numpy()
        out["attack_id"] = None
    elif source == "synthetic":
        keep = frame["channel"].isin(CLASSIC_RAILS)
        if "txn_type" in frame.columns:
            keep &= frame["txn_type"].fillna("purchase").eq("purchase")
        sub = frame.loc[keep]
        out = pd.DataFrame(
            {
                # tz-aware on the synthetic side, naive on the reference side.
                # Both curves are read in local time, which is what makes them
                # comparable at all, so the offset is dropped rather than
                # converted to UTC -- converting would rotate our diurnal curve
                # by five and a half hours and manufacture a difference.
                "ts": pd.to_datetime(sub["ts"]).dt.tz_localize(None),
                "amount": sub["amount"].astype(float),
                "category": sub["mcc"].astype(str),
                "customer_id": sub["customer_id"].astype(str),
                "merchant_id": sub["merchant_id"].astype(str),
                "is_fraud": sub["is_fraud"].astype(bool),
                "attack_id": sub["attack_id"] if "attack_id" in sub.columns else None,
            }
        )
    else:
        raise ValueError(f"source must be 'synthetic' or 'real', got {source!r}")

    return out.sort_values("ts", kind="mergesort").reset_index(drop=True)


def to_shape(common: pd.DataFrame) -> pd.DataFrame:
    """Build the dimensionless feature matrix from a common frame.

    One forward pass per grouping. The velocity counts use a merge-asof against
    the cardholder's own history rather than a rolling window over the whole
    frame, because the two panels have different densities (983 cardholders over
    537 days against ~8,800 over 90) and a window defined in rows rather than in
    time would measure that difference instead of the behaviour.
    """
    frame = common.reset_index(drop=True)
    out = pd.DataFrame(index=frame.index)

    ts = pd.to_datetime(frame["ts"])
    out["hour"] = ts.dt.hour.astype("int16")
    out["dow"] = ts.dt.dayofweek.astype("int16")

    log_amount = np.log1p(frame["amount"].clip(lower=0.0))

    # --- within-category standardisation -----------------------------------
    # Median and IQR rather than mean and standard deviation: spend within a
    # category is long-tailed on both panels, and a single large purchase should
    # not set the scale that every other purchase is measured against.
    grouped = log_amount.groupby(frame["category"])
    centre = grouped.transform("median")
    spread = grouped.transform(lambda s: s.quantile(0.75) - s.quantile(0.25))
    out["log_amount_z"] = ((log_amount - centre) / spread.replace(0.0, np.nan)).fillna(0.0)

    # --- relative to this cardholder ---------------------------------------
    by_customer = log_amount.groupby(frame["customer_id"])
    out["amount_vs_customer"] = (log_amount - by_customer.transform("mean")).fillna(0.0)

    # --- timing, relative to each cardholder's own rhythm --------------------
    epoch = ts.astype("int64") // 10**9
    by_id = epoch.groupby(frame["customer_id"])
    gap = (epoch - by_id.shift(1)).clip(lower=0)
    # The cardholder's own median gap is the denominator, so the panel's rate
    # divides out exactly and what is left is how bursty each person is against
    # their own baseline. A cardholder with one transaction has no median; they
    # get 0.0, which is "exactly typical" and adds no signal either way.
    own_median = gap.groupby(frame["customer_id"]).transform("median")
    out["gap_ratio_log"] = (np.log1p(gap) - np.log1p(own_median)).fillna(0.0)
    out["burst_1h"] = (gap.fillna(np.inf) <= 3_600).astype("int8")

    # --- estate structure ---------------------------------------------------
    counts = frame["merchant_id"].value_counts()
    rank_pct = counts.rank(method="average", ascending=False, pct=True)
    out["merchant_rank_pct"] = frame["merchant_id"].map(rank_pct).astype(float)

    prev_category = frame.groupby("customer_id")["category"].shift(1)
    out["category_shift"] = (
        (prev_category.notna() & (prev_category != frame["category"])).astype("int8")
    )

    return out[list(SHAPE_FEATURES)]


def panel_levels(common: pd.DataFrame) -> dict[str, float]:
    """The absolute rates the shape space standardises away.

    Reported by the scorecard as a labelled table with **no distance attached**.
    A ratio of 5.5 between two panels' cardholder velocity is a fact about how
    each was composed; calling it a KS distance would dress a design decision up
    as a fidelity failure.
    """
    ts = pd.to_datetime(common["ts"])
    days = max((ts.max() - ts.min()).days, 1)
    customers = int(common["customer_id"].nunique())
    merchants = int(common["merchant_id"].nunique())

    epoch = ts.astype("int64") // 10**9
    gap = (epoch - epoch.groupby(common["customer_id"]).shift(1)).dropna()

    shares = np.sort(common["merchant_id"].value_counts(normalize=True).to_numpy())[::-1]
    head = int(max(1, round(0.01 * merchants)))

    return {
        "events": float(len(common)),
        "days": float(days),
        "customers": float(customers),
        "merchants": float(merchants),
        "categories": float(common["category"].nunique()),
        "txn_per_customer_per_day": len(common) / customers / days,
        "median_hours_between": float(gap.median() / 3600.0) if len(gap) else float("nan"),
        "merchants_per_customer": float(
            common.groupby("customer_id")["merchant_id"].nunique().mean()
        ),
        "top_1pct_merchant_share": float(shares[:head].sum()),
    }
