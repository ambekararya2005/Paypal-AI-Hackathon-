"""The standing population: customers, their cards and devices, and merchants.

Entities are built **once** per run and then held fixed while transactions are
drawn against them. That ordering is the whole point of this module, and it is
what separates a payment population from a table of random rows:

* A customer keeps the same one-to-three cards and one-to-three devices for the
  whole window, so ``device_novelty_for_customer`` and
  ``card_bin_customer_mismatch`` mean something. If devices were redrawn per
  event, every event would look novel and the feature would be pure noise.
* A customer has a *home point*, so ``geo_distance_from_home_km`` and
  ``geo_velocity_kmh`` have a reference to be measured against.
* A merchant keeps its MCC, its city, its terminals and its web domain, so
  ``merchant_novelty_for_customer`` and the L4 bipartite features have a stable
  graph to sit on.
* An agent belongs to a customer and a platform, so ``agent_id_fanout`` is a
  real signal rather than an artefact of resampling.

Merchant popularity is Zipf: a global rank is assigned by random permutation and
weight is ``rank ** -a``. Because MCC assignment is independent of rank, the
realised global rank-frequency curve is a power law with the same exponent, and
sampling stays conditional on MCC. That long tail is what makes the difference
between "this merchant is new to this customer" (common, boring) and "this
merchant is new to *everyone*" (rare, interesting).

Everything here is columnar ``numpy``, padded to a fixed width rather than
ragged, because the simulator indexes into it 200,000 times and object attribute
lookup at that scale is the difference between two seconds and two minutes.

Addressing note: source IPs are drawn from ``100.64.0.0/10``, the CGNAT shared
address space. That is not a cop-out placeholder range, it is what a large share
of Indian mobile subscribers actually sit behind, and it is guaranteed never to
collide with a real routable host.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from mantis.foundry.base.reference import ReferenceStats

__all__ = [
    "MAX_AGENTS",
    "MAX_CARDS",
    "MAX_DEVICES",
    "MAX_DOMAINS",
    "Population",
    "build_population",
]

#: Fixed widths for the padded per-customer arrays. Small on purpose: a
#: cardholder with nine devices is a fraud pattern, not a base-population fact.
MAX_CARDS: Final[int] = 3
MAX_DEVICES: Final[int] = 3
MAX_AGENTS: Final[int] = 2
MAX_DOMAINS: Final[int] = 6

#: Benign sites an agent plausibly reads before buying. Reserved TLDs only
#: (RFC 2606 ``.test`` / ``.example``), so nothing here can ever resolve to a
#: real host. These form each customer's habitual browsing set; the L3 feature
#: ``provenance_chain_novel_domain_ratio`` is measured against exactly this.
_BENIGN_DOMAINS: Final[tuple[str, ...]] = (
    "search.example.test",
    "reviews.example.test",
    "price-compare.example.test",
    "deals.example.test",
    "cashback.example.test",
    "coupons.example.test",
    "loyalty.example.test",
    "spec-sheet.example.test",
    "warranty.example.test",
    "support.example.test",
    "forum.example.test",
    "blog.example.test",
    "news.example.test",
    "video.example.test",
    "social.example.test",
    "recipes.example.test",
    "travel-guide.example.test",
    "maps.example.test",
    "transit.example.test",
    "weather.example.test",
    "calendar.example.test",
    "inbox.example.test",
    "wallet.example.test",
    "statements.example.test",
)

#: Short slug per MCC for readable merchant ids and web domains.
_MCC_SLUG: Final[dict[str, str]] = {
    "4111": "transit",
    "4121": "rides",
    "4511": "air",
    "4722": "travel",
    "4814": "telecom",
    "4900": "utility",
    "5311": "dept",
    "5411": "grocer",
    "5541": "fuel",
    "5651": "apparel",
    "5732": "electro",
    "5734": "software",
    "5812": "dining",
    "5814": "quickbite",
    "5912": "pharma",
    "5942": "books",
    "5945": "hobby",
    "5977": "beauty",
    "5999": "retail",
    "6012": "p2p",
    "6300": "insure",
    "7011": "hotel",
    "7832": "cinema",
    "7997": "fitness",
    "8099": "health",
    "8220": "edu",
}

#: Degrees of latitude ~ 111 km. Customers scatter tighter than merchants because
#: a home is a point and a retail estate is a metro-wide footprint.
_HOME_SCATTER_DEG: Final[float] = 0.055
_MERCHANT_SCATTER_DEG: Final[float] = 0.095


def _hex_ids(rng: np.random.Generator, prefix: str, count: int, width: int = 8) -> np.ndarray:
    """``count`` distinct ids of the form ``prefix-<width hex chars>``.

    Walks the id space with a random odd stride from a random start. Because the
    space is ``2 ** (4 * width)`` and an odd stride is coprime to it, the walk
    cannot repeat before it has visited every value, so the ids are distinct by
    construction rather than by rejection sampling. Rejection sampling would have
    been fine too, but this stays O(count) in both time and memory, which
    ``rng.choice(..., replace=False)`` over a four-billion-wide space is not.
    """
    span = 16**width
    start = int(rng.integers(0, span))
    stride = 2 * int(rng.integers(0, span // 2)) + 1
    values = (start + stride * np.arange(count, dtype=np.int64)) % span
    return np.array([f"{prefix}-{int(v):0{width}x}" for v in values], dtype=object)


def _choice_p(weights: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Split a normalised weight mapping into aligned key and probability arrays."""
    keys = np.array(list(weights), dtype=object)
    probs = np.asarray(list(weights.values()), dtype=float)
    return keys, probs / probs.sum()


@dataclass(slots=True)
class Population:
    """Customers and merchants, columnar and fixed for the life of a run.

    Every array is indexed by entity index, not by id. The simulator works in
    indices throughout and only materialises strings at the very end.
    """

    stats: ReferenceStats
    seed: int

    # -- customers ------------------------------------------------------------ #
    customer_ids: np.ndarray
    home_city: np.ndarray
    home_lat: np.ndarray
    home_lon: np.ndarray
    ip_prefix: np.ndarray
    tenure_days: np.ndarray
    #: Gamma-drawn daily transaction rate. Its heavy tail is what makes velocity
    #: features separate a busy cardholder from a compromised one.
    rate: np.ndarray
    card_bins: np.ndarray  # (n_customers, MAX_CARDS) object
    n_cards: np.ndarray
    device_ids: np.ndarray  # (n_customers, MAX_DEVICES) object
    n_devices: np.ndarray
    habitual_domains: np.ndarray  # (n_customers, MAX_DOMAINS) object
    n_domains: np.ndarray

    # -- the agent overlay ----------------------------------------------------- #
    #: Per-customer probability weight for routing spend through an agent.
    #: Continuous and never exactly zero -- see ``_build_agent_overlay``.
    agent_propensity: np.ndarray
    agent_ids: np.ndarray  # (n_customers, MAX_AGENTS) object
    agent_platforms: np.ndarray  # (n_customers, MAX_AGENTS) object
    agent_kya_tokens: np.ndarray  # (n_customers, MAX_AGENTS) object
    agent_devices: np.ndarray  # (n_customers, MAX_AGENTS) object
    #: True where the agent runs on the customer's own device rather than in a
    #: hosted runtime. Those events reuse a personal ``device_id``.
    agent_on_device: np.ndarray  # (n_customers, MAX_AGENTS) bool
    n_agents: np.ndarray

    # -- merchants ------------------------------------------------------------- #
    merchant_ids: np.ndarray
    merchant_mcc: np.ndarray
    merchant_country: np.ndarray
    merchant_city: np.ndarray
    merchant_lat: np.ndarray
    merchant_lon: np.ndarray
    merchant_domain: np.ndarray
    merchant_terminals: np.ndarray
    #: Zipf popularity weight; also the merchant's global rank proxy.
    merchant_weight: np.ndarray

    #: MCC -> merchant indices, and the matching normalised sampling probabilities.
    by_mcc: dict[str, np.ndarray]
    by_mcc_p: dict[str, np.ndarray]

    @property
    def n_customers(self) -> int:
        return int(self.customer_ids.size)

    @property
    def n_merchants(self) -> int:
        return int(self.merchant_ids.size)

    def describe(self) -> str:
        """Human-readable summary, printed by the simulator."""
        heavy = int((self.agent_propensity > 0.5).sum())
        on_device = float(self.agent_on_device[self.n_agents > 0].mean())
        return "\n".join(
            [
                f"population: {self.n_customers:,} customers, {self.n_merchants:,} merchants",
                f"  cards/customer    : mean {self.n_cards.mean():.2f}",
                f"  devices/customer  : mean {self.n_devices.mean():.2f}",
                f"  agent propensity  : mean {self.agent_propensity.mean():.3f}, "
                f"{heavy:,} customers above 0.5",
                f"  agents on-device  : {on_device:.1%} (rest hosted)",
                f"  daily rate        : mean {self.rate.mean():.3f}, "
                f"p95 {np.percentile(self.rate, 95):.3f}, max {self.rate.max():.3f}",
                f"  merchants/mcc     : min {min(len(v) for v in self.by_mcc.values())}, "
                f"max {max(len(v) for v in self.by_mcc.values())}",
            ]
        )


def _build_customers(
    rng: np.random.Generator, stats: ReferenceStats, n_customers: int
) -> dict[str, np.ndarray]:
    """Draw the cardholder base: geography, credentials, devices, tenure, velocity."""
    city_p = np.asarray([c.weight for c in stats.cities], dtype=float)
    city_p /= city_p.sum()
    city_lat = np.asarray([c.lat for c in stats.cities], dtype=float)
    city_lon = np.asarray([c.lon for c in stats.cities], dtype=float)

    home_city = rng.choice(len(stats.cities), size=n_customers, p=city_p)
    home_lat = city_lat[home_city] + rng.normal(0.0, _HOME_SCATTER_DEG, n_customers)
    home_lon = city_lon[home_city] + rng.normal(0.0, _HOME_SCATTER_DEG, n_customers)

    # CGNAT /24s. A customer keeps one, so a shared prefix across customers is a
    # genuine coincidence rather than a modelling artefact.
    ip_prefix = np.array(
        [
            f"100.{64 + b}.{c}"
            for b, c in zip(
                rng.integers(0, 64, n_customers), rng.integers(0, 256, n_customers), strict=True
            )
        ],
        dtype=object,
    )

    # Portfolio tenure: a long-tailed mix of legacy and recently-issued accounts.
    tenure_days = np.clip(rng.exponential(520.0, n_customers) + 21.0, 21.0, 4000.0).astype(int)

    # Per-customer daily rate. Gamma with shape < 2 keeps the heavy tail: most
    # cardholders are quiet, a minority carry a disproportionate share of volume.
    shape = stats.customer_rate_gamma_shape
    scale = stats.customer_rate_mean_per_day / shape
    rate = rng.gamma(shape, scale, n_customers)
    rate = np.clip(rate, 1e-4, None)

    n_cards = rng.choice([1, 2, 3], size=n_customers, p=[0.55, 0.33, 0.12])
    n_devices = rng.choice([1, 2, 3], size=n_customers, p=[0.62, 0.30, 0.08])
    n_domains = rng.integers(3, MAX_DOMAINS + 1, n_customers)

    bin_keys, bin_p = _choice_p({b.card_bin: b.weight for b in stats.card_bins})
    card_bins = np.full((n_customers, MAX_CARDS), None, dtype=object)
    for slot in range(MAX_CARDS):
        drawn = bin_keys[rng.choice(len(bin_keys), size=n_customers, p=bin_p)]
        card_bins[:, slot] = np.where(n_cards > slot, drawn, None)

    device_ids = _hex_ids(rng, "dev", n_customers * MAX_DEVICES).reshape(n_customers, MAX_DEVICES)
    device_ids = np.where(np.arange(MAX_DEVICES)[None, :] < n_devices[:, None], device_ids, None)

    domain_pool = np.array(_BENIGN_DOMAINS, dtype=object)
    habitual = np.full((n_customers, MAX_DOMAINS), None, dtype=object)
    for i in range(n_customers):
        picked = rng.choice(len(domain_pool), size=int(n_domains[i]), replace=False)
        habitual[i, : n_domains[i]] = domain_pool[picked]

    return {
        "customer_ids": np.array([f"cust-{i:07d}" for i in range(n_customers)], dtype=object),
        "home_city": home_city,
        "home_lat": np.clip(home_lat, -90.0, 90.0),
        "home_lon": np.clip(home_lon, -180.0, 180.0),
        "ip_prefix": ip_prefix,
        "tenure_days": tenure_days,
        "rate": rate,
        "card_bins": card_bins,
        "n_cards": n_cards,
        "device_ids": device_ids,
        "n_devices": n_devices,
        "habitual_domains": habitual,
        "n_domains": n_domains,
    }


def _build_agent_overlay(
    rng: np.random.Generator, stats: ReferenceStats, n_customers: int
) -> dict[str, np.ndarray]:
    """Give every customer an agent identity, and a *graded* propensity to use it.

    Two things here are the result of an audit finding, and both matter.

    **Propensity is continuous, not a binary adopter flag.** The first version of
    this drew ``is_adopter ~ Bernoulli(0.30)``, which left 70% of customers with
    *exactly zero* probability of an agentic transaction. That made ``customer_id``
    a 0.90-AUC predictor of the rail and ``ip`` a 0.93-AUC one: a model could read
    the rail straight off the cardholder. A Beta draw keeps adoption concentrated
    -- which is what L4 fan-out features need -- while giving every customer a
    non-zero chance, so a customer with no agentic events has none by *sampling*
    rather than by construction.

    **Agents are hosted somewhere, and sometimes that is the customer's own
    phone.** On-device runtimes transact from the customer's existing device_id;
    cloud runtimes carry their own. Without this every device was either 100%
    agentic or 0% agentic -- 8,796 devices, not one of them mixed -- which made
    ``device_id`` a *perfect* separator. Hosting is a property of the platform,
    so the mapping lives in the reference stats.
    """
    # Beta(0.6, 2.0): mean ~0.23, heavy mass near zero, a real tail near one.
    # Concentrated adoption without a hard zero anywhere.
    propensity = rng.beta(stats.agent_propensity_alpha, stats.agent_propensity_beta, n_customers)
    # Rescale so the mean matches the configured adoption rate, keeping the shape.
    propensity = np.clip(propensity * (stats.agent_adoption_rate / propensity.mean()), 1e-4, 1.0)

    # Every customer gets at least one agent identity. Allocation is cheap and it
    # is the propensity, not the existence of an agent, that decides volume.
    n_agents = rng.choice([1, 2], size=n_customers, p=[0.86, 0.14])

    platform_keys, platform_p = _choice_p(stats.agent_platform_weights)
    on_device_p = np.asarray(
        [stats.agent_on_device_p.get(str(k), 0.0) for k in platform_keys], dtype=float
    )

    agent_ids = np.full((n_customers, MAX_AGENTS), None, dtype=object)
    agent_platforms = np.full((n_customers, MAX_AGENTS), None, dtype=object)
    agent_kya = np.full((n_customers, MAX_AGENTS), None, dtype=object)
    agent_devices = np.full((n_customers, MAX_AGENTS), None, dtype=object)
    agent_on_device = np.zeros((n_customers, MAX_AGENTS), dtype=bool)

    for slot in range(MAX_AGENTS):
        active = n_agents > slot
        ids = _hex_ids(rng, "agt", n_customers)
        kya = _hex_ids(rng, "kya", n_customers)
        devs = _hex_ids(rng, "dev-agt", n_customers, width=6)
        choice = rng.choice(len(platform_keys), size=n_customers, p=platform_p)
        plats = platform_keys[choice]
        hosted_on_device = rng.random(n_customers) < on_device_p[choice]
        agent_ids[:, slot] = np.where(active, ids, None)
        agent_platforms[:, slot] = np.where(active, plats, None)
        agent_kya[:, slot] = np.where(active, kya, None)
        agent_devices[:, slot] = np.where(active, devs, None)
        agent_on_device[:, slot] = active & hosted_on_device

    return {
        "agent_propensity": propensity,
        "agent_ids": agent_ids,
        "agent_platforms": agent_platforms,
        "agent_kya_tokens": agent_kya,
        "agent_devices": agent_devices,
        "agent_on_device": agent_on_device,
        "n_agents": n_agents,
    }


def _build_merchants(
    rng: np.random.Generator, stats: ReferenceStats, n_merchants: int
) -> dict[str, object]:
    """Draw the merchant estate, sized per MCC and ranked Zipf by popularity."""
    profiles = stats.mcc_profiles
    weights = np.asarray([p.weight for p in profiles], dtype=float)

    # At least eight merchants per category, so even a thin MCC has a tail.
    counts = np.maximum(8, np.round(weights * n_merchants).astype(int))
    mcc_of = np.concatenate(
        [np.full(int(c), i, dtype=int) for c, i in zip(counts, range(len(profiles)), strict=True)]
    )
    total = int(mcc_of.size)

    city_p = np.asarray([c.weight for c in stats.cities], dtype=float)
    city_p /= city_p.sum()
    city_lat = np.asarray([c.lat for c in stats.cities], dtype=float)
    city_lon = np.asarray([c.lon for c in stats.cities], dtype=float)

    country_keys, country_p = _choice_p(stats.merchant_country_weights)
    country = country_keys[rng.choice(len(country_keys), size=total, p=country_p)]
    is_domestic = country == "IN"

    city = rng.choice(len(stats.cities), size=total, p=city_p)
    lat = city_lat[city] + rng.normal(0.0, _MERCHANT_SCATTER_DEG, total)
    lon = city_lon[city] + rng.normal(0.0, _MERCHANT_SCATTER_DEG, total)
    # A cross-border acquirer has no Indian metro site; its geo is left to the
    # cardholder's own location at authorisation time.
    lat = np.where(is_domestic, lat, np.nan)
    lon = np.where(is_domestic, lon, np.nan)

    # Zipf popularity over a random global rank, so rank is independent of MCC.
    rank = rng.permutation(total) + 1
    weight = rank.astype(float) ** (-stats.merchant_zipf_exponent)

    mcc_codes = np.array([profiles[i].mcc for i in mcc_of], dtype=object)
    slugs = np.array(
        [f"{_MCC_SLUG.get(profiles[i].mcc, 'shop')}{j:05d}" for j, i in enumerate(mcc_of)],
        dtype=object,
    )
    merchant_ids = np.array([f"mer-{s}" for s in slugs], dtype=object)
    domains = np.array([f"shop.{s}.test" for s in slugs], dtype=object)

    # Terminal estate scales with popularity, softened by a fractional exponent so
    # the head does not run away: the busiest merchant runs a few dozen lanes, the
    # long tail runs one. Capped, because a 500-terminal merchant in a 12k estate
    # would dominate every card-present feature on its own.
    popularity = (weight / weight.max()) ** 0.38
    terminals = 1 + np.round(popularity * 45.0 * rng.gamma(2.0, 0.5, total)).astype(int)
    terminals = np.minimum(terminals, 60)

    by_mcc: dict[str, np.ndarray] = {}
    by_mcc_p: dict[str, np.ndarray] = {}
    for i, profile in enumerate(profiles):
        idx = np.flatnonzero(mcc_of == i)
        by_mcc[profile.mcc] = idx
        p = weight[idx]
        by_mcc_p[profile.mcc] = p / p.sum()

    return {
        "merchant_ids": merchant_ids,
        "merchant_mcc": mcc_codes,
        "merchant_country": country,
        "merchant_city": city,
        "merchant_lat": lat,
        "merchant_lon": lon,
        "merchant_domain": domains,
        "merchant_terminals": terminals,
        "merchant_weight": weight,
        "by_mcc": by_mcc,
        "by_mcc_p": by_mcc_p,
    }


def build_population(
    stats: ReferenceStats,
    *,
    seed: int = 1337,
    n_customers: int = 5_000,
    n_merchants: int = 12_000,
) -> Population:
    """Build the standing population. Deterministic for a given ``seed``.

    Args:
        stats: Calibration driving geography, BIN mix, velocity and agent adoption.
        seed: Master seed. The same seed always yields the same entities.
        n_customers: Cardholder count.
        n_merchants: Approximate merchant count; the realised number is slightly
            higher because every MCC gets a floor of eight merchants.

    Returns:
        A fully populated :class:`Population`.
    """
    if n_customers < 1:
        raise ValueError(f"n_customers must be positive, got {n_customers}")
    if n_merchants < 1:
        raise ValueError(f"n_merchants must be positive, got {n_merchants}")

    # A dedicated child stream so that changing the transaction count never
    # perturbs the entities: the population is a function of the seed alone.
    rng = np.random.default_rng([seed, 0xE471])

    fields: dict[str, object] = {"stats": stats, "seed": seed}
    fields.update(_build_customers(rng, stats, n_customers))
    fields.update(_build_agent_overlay(rng, stats, n_customers))
    fields.update(_build_merchants(rng, stats, n_merchants))
    return Population(**fields)  # type: ignore[arg-type]


def main() -> None:
    """Print a population summary. Run: ``python -m mantis.foundry.base.entities``."""
    from mantis.foundry.base.reference import load_reference_stats

    stats = load_reference_stats()
    pop = build_population(stats, seed=1337)
    print(pop.describe())
    print()
    order = np.argsort(-pop.merchant_weight)[:10]
    print("  ten most popular merchants:")
    for i in order:
        print(
            f"    {pop.merchant_ids[i]:<20} mcc={pop.merchant_mcc[i]} "
            f"{pop.merchant_country[i]}  w={pop.merchant_weight[i]:.5f}"
        )


if __name__ == "__main__":
    main()
