"""The arena: an evolutionary adversary against a retraining detector.

What this is for, after the Day 5 reframing
---------------------------------------------
L2 was supposed to be the answer to "what about the attacks you did not think
of", and it is not (CLAUDE.md, "The zero-day answer, reframed"). The
architecture's answer is now two things, and this module is the second of them:
rather than hoping an unsupervised layer generalises to an attack nobody wrote,
**manufacture the attack before an attacker does**, label it by construction, and
retrain on it.

That makes this module load-bearing rather than a flourish, and it is why it
ships with two experiments instead of one.

Experiment 1 — the arena (:func:`run_arena`)
-----------------------------------------------
A population of genomes per card. Each generation:

1. every genome is expressed — its injector runs, the genome is applied to the
   rows, and the result is validated as a schema-conformant instance of its card;
2. the current detector scores them at its fixed 0.1%-FPR operating point;
3. **fitness = evasion x payoff** — the share of the variant's events that slipped
   through, times the money they moved, normalised against the unmutated attack.
   Both terms are necessary. Evasion alone is maximised by a variant that moves
   ₹0 through one event; payoff alone is maximised by one that moves so much it
   is caught on the first authorisation. The product is the operator's actual
   objective;
4. the fittest are crossed and mutated into the next generation;
5. **the detector retrains on everything the arena has produced so far** — which
   is the blue team's move, and the reason the curve is expected to fall rather
   than rise.

The output is ``data/generated/arena.json``: evasion by generation, the surviving
genomes, and which genes moved.

Experiment 2 — the zero-day demonstration (:func:`run_zero_day`)
-------------------------------------------------------------------
This is the one that carries the submission's argument, and it is worth being
exact about what it does and does not claim.

Day 4 measured what happens to a supervised detector when a whole family is
absent from training: **F1's recall collapses from 0.569 to 0.007**. The claim
under test is that the loop closes that gap without anyone ever seeing the real
attack.

    * **Baseline** — L1 trained with family F1 entirely removed. Recall on the
      real F1 test rows.
    * **Loop-augmented** — L1 trained with family F1 still removed, plus the
      **evolved variants** the arena produced for F1's cards. Recall on the same
      real F1 test rows.

**The detector and the loop had different things, and the claim turns on that.**

The *detector* never trained on a single real F1 event. Not one. That is exactly
what the baseline row measures, and 0.013 is what it earns for it.

The *loop* had F1's **atlas cards and their executable injectors** — a written
description of a class of attack, plus code that manufactures instances of it.
That is **a red team, not a fraud history**, and the distinction is the entire
contribution. The augmented detector did not generalise to F1 on its own; it was
handed manufactured F1 training data produced from a specification a human wrote
*before any F1 attack had been observed in the wild*.

What the loop still did not have: a single one of the F1 rows it is then
evaluated on. The variants are not those rows — every gene moved them, and they
were **selected for evading the detector**, so they are systematically
off-distribution from the canonical attack in exactly the direction that makes
the transfer hard. That is why the recovery is partial rather than total.

So the honest statement of the result is: *"an attack family that has been
described but never observed can be manufactured, and training on the
manufactured version transfers to the real one."* Not *"the detector caught
something nobody had thought of"* — nothing does that. Somebody thought of it.
The contribution is that **thinking of it was enough**, which is the only
position available on a rail that has no loss history yet.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from mantis.core.paths import GENERATED_DIR, ensure_dir
from mantis.defense.features import FeatureBuilder
from mantis.defense.l1_gbdt import L1Model
from mantis.defense.metrics import OPERATING_FPR, threshold_at_fpr
from mantis.foundry.injectors import REGISTRY, get_injector
from mantis.foundry.injectors.base import (
    PopulationView,
    run_injector,
    stable_seed,
    validate_attack_frame,
)
from mantis.foundry.llm.corpus import load_content_store
from mantis.loop.genome import AttackGenome, crossover, mutate, random_genome
from mantis.loop.genome import identity_genome as _identity
from mantis.loop.mutate import mutate_rows

__all__ = [
    "ARENA_JSON",
    "SURVIVAL_ROUNDS",
    "ArenaResult",
    "Individual",
    "ZeroDayResult",
    "run_arena",
    "run_zero_day",
    "write_arena",
]

#: Where the evasion curve is written. The gate artefact for Day 5.
ARENA_JSON: Final[Path] = GENERATED_DIR / "arena.json"

#: Share of the chronology used to fit the detector, matching the Day 4 split so
#: that the arena's baseline and RESULTS.md's are the same number.
TRAIN_SHARE: Final[float] = 0.70

#: Generations a genome must survive before it earns an atlas card. Three is the
#: brief's threshold and it is a real bar: with a population of eight and elitism
#: of three, surviving three consecutive rounds against a retraining detector is
#: not something a lucky draw does.
SURVIVAL_ROUNDS: Final[int] = 3


@dataclass(slots=True)
class Individual:
    """One genome and what the arena measured about it."""

    genome: AttackGenome
    generation: int
    n_events: int = 0
    evasion: float = 0.0
    payoff: float = 0.0
    fitness: float = 0.0
    #: Consecutive generations this genome's lineage has survived selection.
    survived: int = 0

    def to_json(self) -> dict[str, object]:
        return {
            "label": self.genome.label(),
            "card_id": self.genome.card_id,
            "generation": self.generation,
            "n_events": self.n_events,
            "evasion": round(self.evasion, 4),
            "payoff": round(self.payoff, 4),
            "fitness": round(self.fitness, 4),
            "survived": self.survived,
            "genes": {k: round(v, 4) for k, v in self.genome.genes.items()},
        }


@dataclass(slots=True)
class ArenaResult:
    """Everything the CLI, ``arena.json`` and RESULTS.md need."""

    cards: list[str] = field(default_factory=list)
    generations: list[dict[str, object]] = field(default_factory=list)
    survivors: list[Individual] = field(default_factory=list)
    #: The last generation, fitness-ranked. Used when nothing cleared
    #: :data:`SURVIVAL_ROUNDS` — which is itself a reportable outcome.
    finalists: list[Individual] = field(default_factory=list)
    seconds: float = 0.0
    n_background: int = 0
    operating_fpr: float = OPERATING_FPR

    def evasion_curve(self) -> list[float]:
        return [float(g["mean_evasion"]) for g in self.generations]

    def to_json(self) -> dict[str, object]:
        return {
            "operating_fpr": self.operating_fpr,
            "n_background": self.n_background,
            "cards": self.cards,
            "seconds": round(self.seconds, 1),
            "evasion_curve": [round(v, 4) for v in self.evasion_curve()],
            "generations": self.generations,
            "survivors": [i.to_json() for i in self.survivors],
        }


@dataclass(slots=True)
class ZeroDayResult:
    """The comparison that carries the argument. See the module docstring."""

    family: str
    n_test_positive: int
    recall_trained: float
    recall_heldout: float
    recall_loop: float
    n_variant_events: int

    @property
    def gap_closed(self) -> float:
        """Share of the held-out collapse the loop recovers."""
        span = self.recall_trained - self.recall_heldout
        if span <= 1e-9:
            return float("nan")
        return (self.recall_loop - self.recall_heldout) / span

    def to_json(self) -> dict[str, object]:
        return {
            "family": self.family,
            "n_test_positive": self.n_test_positive,
            "recall_trained_on_family": round(self.recall_trained, 4),
            "recall_family_held_out": round(self.recall_heldout, 4),
            "recall_loop_augmented": round(self.recall_loop, 4),
            "gap_closed": round(self.gap_closed, 4),
            "n_variant_events": self.n_variant_events,
        }


# --------------------------------------------------------------------------- #
# Expressing a genome
# --------------------------------------------------------------------------- #


def express(
    genome: AttackGenome,
    view: PopulationView,
    *,
    seed: int,
    intensity: float = 1.0,
) -> pd.DataFrame:
    """Run the card's injector, then apply the genome. Validated before it returns.

    The validation is not ceremony. A mutator that produced rows the injector
    framework would have rejected would be feeding the retrain harness data no
    attack could generate, and every number downstream would be about the
    mutator.
    """
    rows = run_injector(get_injector(genome.card_id), view, intensity=intensity, seed=seed)
    rng = np.random.default_rng([seed, stable_seed(genome.label())])
    out = mutate_rows(rows, genome, view, rng)
    validate_attack_frame(out, genome.card_id, view.frame)
    return out


# --------------------------------------------------------------------------- #
# The arena
# --------------------------------------------------------------------------- #


def _fit_detector(
    X: pd.DataFrame, y: np.ndarray, ts: pd.Series, seed: int
) -> L1Model:
    return L1Model(seed=seed).fit(X, y, timestamps=ts)


#: Timestamp columns that carry a timezone and must agree before a concat.
_TZ_COLUMNS: Final[tuple[str, ...]] = ("ts", "ag_mandate_issued_ts", "dispute_raised_ts")


def _align_timezones(extra: pd.DataFrame, background: pd.DataFrame) -> pd.DataFrame:
    """Put freshly-generated rows on the background's own tzinfo object.

    Not cosmetic. A background read back from parquet carries
    ``pytz.FixedOffset(330)``; rows straight out of an injector carry the
    simulator's ``IST``. They are the same offset and a different *object*, and
    ``pd.concat`` of two tz-aware columns whose tzinfo objects differ silently
    produces an **object** column — after which ``.dt`` raises inside the feature
    builder, several frames later, with a message about datetimelike values that
    says nothing about where the mismatch came from.
    """
    if extra.empty:
        return extra
    out = extra.copy()
    for column in _TZ_COLUMNS:
        if column not in out.columns or column not in background.columns:
            continue
        target = getattr(background[column].dtype, "tz", None)
        source = getattr(out[column].dtype, "tz", None)
        if target is not None and source is not None:
            out[column] = out[column].dt.tz_convert(target)
    return out


def _build_matrix(
    background: pd.DataFrame, extra: pd.DataFrame, train_mask_background: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Feature matrix for ``background + extra``, in one chronological pass.

    Profiles and baselines are fitted on the **background's training rows only**,
    never on the variants — an entity baseline fitted on the attack it is meant
    to catch is the same mistake as fitting it on the test period.
    """
    combined = pd.concat([background, _align_timezones(extra, background)], ignore_index=True)
    combined = combined.sort_values("ts", kind="stable").reset_index(drop=True)
    is_extra = combined["_variant"].notna().to_numpy()

    cut = background.loc[train_mask_background, "ts"].max()
    fit_rows = (~is_extra) & (combined["ts"] <= cut).to_numpy()

    builder = FeatureBuilder()
    matrix = builder.fit_transform_stream(combined, pd.Series(fit_rows, index=combined.index))
    return combined, matrix


def run_arena(
    background: pd.DataFrame,
    *,
    cards: tuple[str, ...] | None = None,
    generations: int = 6,
    population: int = 8,
    elite: int = 3,
    seed: int = 1337,
    verbose: bool = True,
) -> ArenaResult:
    """Evolve variants against a detector that retrains between rounds.

    Args:
        background: A clean, labelled pool. Its own attack rows are the training
            signal the generation-zero detector starts from.
        cards: Which atlas cards to evolve. Every implemented card by default.
        generations: Rounds. Six is enough for a slope; the brief allows 5-10.
        population: Genomes per card per generation.
        elite: Genomes carried into the next generation unchanged.
        seed: Determinism, as everywhere else in this repo.
    """

    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    started = time.perf_counter()
    load_content_store()
    cards = tuple(cards) if cards is not None else tuple(sorted(REGISTRY))
    rng = np.random.default_rng(seed)

    background = background.sort_values("ts", kind="stable").reset_index(drop=True)
    if "_variant" not in background.columns:
        background = background.assign(_variant=pd.Series([None] * len(background), dtype=object))
    cut = background["ts"].quantile(TRAIN_SHARE)
    train_mask = (background["ts"] <= cut).to_numpy()

    clean = background[~background["is_fraud"].to_numpy()].drop(columns=["_variant"])
    view = PopulationView.build(clean.reset_index(drop=True))

    # The generation-zero detector: the Day 4 model, on this background.
    log("  fitting the generation-0 detector...")
    _, base_matrix = _build_matrix(background, background.head(0), train_mask)
    y_bg = background["is_fraud"].to_numpy(dtype=bool)
    detector = _fit_detector(
        base_matrix[train_mask], y_bg[train_mask], background.loc[train_mask, "ts"], seed
    )

    populations: dict[str, list[Individual]] = {
        card: [
            Individual(genome=random_genome(card, rng), generation=0)
            for _ in range(population)
        ]
        for card in cards
    }
    # One unmutated individual per card, so the curve has a "what the attack does
    # without evolution" reference inside it rather than only in prose.
    for card in cards:
        populations[card][0] = Individual(genome=_identity(card), generation=0)

    archive: list[pd.DataFrame] = []
    result = ArenaResult(cards=list(cards), n_background=len(background))
    lineage: dict[str, int] = {}
    #: Total amount the **unmutated** attack attempts, per card. Payoff is
    #: normalised against this so that a variant's number means "this much of
    #: what the parent attack was worth", and so that cards of different sizes
    #: are comparable inside one fitness ranking. Measured once, at generation
    #: zero, from each card's identity individual.
    reference_payoff: dict[str, float] = {}

    for generation in range(generations):
        log(f"  generation {generation}: expressing {sum(len(p) for p in populations.values())} "
            "variants...")
        blocks: list[pd.DataFrame] = []
        for individuals in populations.values():
            for index, individual in enumerate(individuals):
                rows = express(
                    individual.genome, view, seed=seed + 1000 * generation + index
                )
                rows = rows.assign(_variant=individual.genome.label())
                individual.n_events = len(rows)
                blocks.append(rows)
        variants = pd.concat(blocks, ignore_index=True)

        combined, matrix = _build_matrix(background, variants, train_mask)
        scores = detector.score(matrix)
        y_combined = combined["is_fraud"].to_numpy(dtype=bool)
        is_variant = combined["_variant"].notna().to_numpy()

        # The operating point: 0.1% of legitimate traffic, as everywhere else.
        legit = ~y_combined & ~is_variant
        cutoff = threshold_at_fpr(scores[legit | (~y_combined)], y_combined[legit | (~y_combined)])
        if not np.isfinite(cutoff):
            cutoff = float(np.quantile(scores[~y_combined], 1.0 - OPERATING_FPR))

        labels = combined["_variant"].to_numpy()
        if generation == 0:
            for card, individuals in populations.items():
                identity = labels == individuals[0].genome.label()
                reference_payoff[card] = max(
                    float(combined.loc[is_variant & identity, "amount"].sum()), 1.0
                )

        for card, individuals in populations.items():
            for individual in individuals:
                mask = labels == individual.genome.label()
                if not mask.any():
                    individual.evasion = 0.0
                    individual.payoff = 0.0
                    individual.fitness = 0.0
                    continue
                caught = scores[mask] >= cutoff
                individual.evasion = float(1.0 - caught.mean())
                # Money that got through, not money attempted. An operator does
                # not book revenue on a declined authorisation.
                individual.payoff = float(
                    combined.loc[mask, "amount"].to_numpy()[~caught].sum()
                    / reference_payoff.get(card, 1.0)
                )
                individual.fitness = individual.evasion * individual.payoff

        flat = [i for individuals in populations.values() for i in individuals]
        record = {
            "generation": generation,
            "n_variants": len(flat),
            "n_events": int(is_variant.sum()),
            "mean_evasion": float(np.mean([i.evasion for i in flat])),
            "max_evasion": float(np.max([i.evasion for i in flat])),
            "mean_fitness": float(np.mean([i.fitness for i in flat])),
            "detector_threshold": float(cutoff),
            "per_card": {
                card: round(float(np.mean([i.evasion for i in individuals])), 4)
                for card, individuals in populations.items()
            },
        }
        result.generations.append(record)
        log(f"    mean evasion {record['mean_evasion']:.3f}  "
            f"max {record['max_evasion']:.3f}  mean fitness {record['mean_fitness']:.3f}")

        archive.append(variants)

        # -- the blue team's move: retrain on everything produced so far -------- #
        if generation < generations - 1:
            log("    retraining the detector on the arena's output...")
            history = pd.concat(archive, ignore_index=True)
            combined_r, matrix_r = _build_matrix(background, history, train_mask)
            y_r = combined_r["is_fraud"].to_numpy(dtype=bool)
            is_variant_r = combined_r["_variant"].notna().to_numpy()
            # Variants are training data wherever they landed on the calendar:
            # they are manufactured, so there is no "future" to leak from them.
            # The background keeps its time split.
            train_r = ((combined_r["ts"] <= cut).to_numpy() & ~is_variant_r) | is_variant_r
            detector = _fit_detector(
                matrix_r[train_r], y_r[train_r], combined_r.loc[train_r, "ts"], seed
            )

            # -- selection ------------------------------------------------------ #
            for card, individuals in populations.items():
                ranked = sorted(individuals, key=lambda i: i.fitness, reverse=True)
                keepers = ranked[:elite]
                for keeper in keepers:
                    lineage[keeper.genome.label()] = lineage.get(keeper.genome.label(), 0) + 1
                    keeper.survived = lineage[keeper.genome.label()]
                children: list[Individual] = [
                    Individual(genome=k.genome, generation=generation + 1, survived=k.survived)
                    for k in keepers
                ]
                while len(children) < population:
                    a, b = (
                        keepers[int(rng.integers(0, len(keepers)))],
                        keepers[int(rng.integers(0, len(keepers)))],
                    )
                    child = mutate(crossover(a.genome, b.genome, rng), rng)
                    children.append(Individual(genome=child, generation=generation + 1))
                populations[card] = children

    flat_final = [i for individuals in populations.values() for i in individuals]
    result.finalists = sorted(flat_final, key=lambda i: i.fitness, reverse=True)
    result.survivors = [i for i in result.finalists if i.survived >= SURVIVAL_ROUNDS]
    result.seconds = time.perf_counter() - started
    return result


def write_arena(
    result: ArenaResult, zero_day: ZeroDayResult | None, path: Path = ARENA_JSON
) -> Path:
    """Write ``arena.json`` — the Day 5 gate artefact."""
    ensure_dir(path.parent)
    payload = result.to_json()
    payload["zero_day"] = zero_day.to_json() if zero_day is not None else None
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# The zero-day demonstration
# --------------------------------------------------------------------------- #


def run_zero_day(
    background: pd.DataFrame,
    *,
    family: str = "F1",
    generations: int = 4,
    population: int = 6,
    seed: int = 1337,
    verbose: bool = True,
) -> ZeroDayResult:
    """Held-out family, then the same family manufactured by the loop.

    Three detectors, one test set, one operating point each — see the module
    docstring for exactly what is and is not being claimed.
    """

    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    load_content_store()
    background = background.sort_values("ts", kind="stable").reset_index(drop=True)
    if "_variant" not in background.columns:
        background = background.assign(_variant=pd.Series([None] * len(background), dtype=object))
    cut = background["ts"].quantile(TRAIN_SHARE)
    train_mask = (background["ts"] <= cut).to_numpy()
    test_mask = ~train_mask

    fam = background["attack_id"].fillna("").str.slice(0, 2).to_numpy()
    y = background["is_fraud"].to_numpy(dtype=bool)
    in_family = (fam == family) & y

    log(f"  building the base matrix ({len(background):,} rows)...")
    _, matrix = _build_matrix(background, background.head(0), train_mask)

    def recall_on_family(scores: np.ndarray) -> float:
        cutoff = threshold_at_fpr(scores[test_mask], y[test_mask])
        target = in_family & test_mask
        if not np.isfinite(cutoff) or not target.any():
            return float("nan")
        return float((scores[target] >= cutoff).mean())

    log("  (1) detector trained WITH the family...")
    full = _fit_detector(
        matrix[train_mask], y[train_mask], background.loc[train_mask, "ts"], seed
    )
    recall_trained = recall_on_family(full.score(matrix))

    log(f"  (2) detector with {family} held out of training...")
    keep = train_mask & ~in_family
    held = _fit_detector(matrix[keep], y[keep], background.loc[keep, "ts"], seed)
    recall_heldout = recall_on_family(held.score(matrix))

    log(f"  (3) evolving {family} variants against that held-out detector...")
    cards = tuple(c for c in sorted(REGISTRY) if c.startswith(family))
    # The arena runs against a background with the family already removed, so the
    # adversary is evolving against exactly the detector that has never seen it.
    stripped = background[~in_family].reset_index(drop=True)
    arena = run_arena(
        stripped,
        cards=cards,
        generations=generations,
        population=population,
        seed=seed,
        verbose=verbose,
    )
    log(f"    arena evasion curve: {[round(v, 3) for v in arena.evasion_curve()]}")

    chosen = arena.survivors or arena.finalists
    provenance = (
        f"survivors of {SURVIVAL_ROUNDS}+ rounds"
        if arena.survivors
        else "the final generation; none cleared the survival bar"
    )
    log(f"    manufacturing from {len(chosen)} genomes ({provenance})")
    clean = background[~y].drop(columns=["_variant"]).reset_index(drop=True)
    view = PopulationView.build(clean)
    manufactured = pd.concat(
        [
            express(individual.genome, view, seed=seed + index).assign(
                _variant=individual.genome.label()
            )
            for index, individual in enumerate(chosen)
        ],
        ignore_index=True,
    )
    # Manufactured rows are confined to the TRAINING window. This matters more
    # than it looks: a variant landing in the test period would sit inside the
    # velocity and graph state of the real test-period attack rows it is being
    # evaluated against, inflating their counts and making them easier to catch
    # for a reason that has nothing to do with what the detector learned. The
    # comparison would then be measuring the injection, not the transfer.
    # Truncating a campaign at the cut is fine — a partial ring is still a valid
    # labelled instance of its card, which is what the retrain harness consumes.
    before = len(manufactured)
    manufactured = manufactured[manufactured["ts"] <= cut].reset_index(drop=True)
    log(f"    confined to the training window: {len(manufactured):,} of {before:,} rows kept")
    log(f"  (4) retraining on {len(manufactured):,} manufactured {family} events...")

    combined, matrix_aug = _build_matrix(background, manufactured, train_mask)
    y_aug = combined["is_fraud"].to_numpy(dtype=bool)
    is_variant = combined["_variant"].notna().to_numpy()
    fam_aug = combined["attack_id"].fillna("").str.slice(0, 2).to_numpy()
    in_family_aug = (fam_aug == family) & y_aug & ~is_variant
    train_aug = ((combined["ts"] <= cut).to_numpy() & ~in_family_aug) | is_variant
    loop_model = _fit_detector(
        matrix_aug[train_aug], y_aug[train_aug], combined.loc[train_aug, "ts"], seed
    )
    scores_aug = loop_model.score(matrix_aug)
    test_aug = (combined["ts"] > cut).to_numpy() & ~is_variant
    cutoff = threshold_at_fpr(scores_aug[test_aug], y_aug[test_aug])
    target = in_family_aug & test_aug
    recall_loop = (
        float((scores_aug[target] >= cutoff).mean())
        if np.isfinite(cutoff) and target.any()
        else float("nan")
    )

    return ZeroDayResult(
        family=family,
        n_test_positive=int((in_family & test_mask).sum()),
        recall_trained=recall_trained,
        recall_heldout=recall_heldout,
        recall_loop=recall_loop,
        n_variant_events=len(manufactured),
    )
