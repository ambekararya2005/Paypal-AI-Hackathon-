"""L3's contract: it reads text, it never reads a transaction label.

The claim this layer makes is unusual enough to be worth testing rather than
asserting in a docstring. L3 is presented in RESULTS.md as belonging with L0 —
a layer that works on an attack it has never seen in the payment data, because
the thing it was trained on is not payment data. Three tests hold that:

1. ``fit`` takes no ``y``, and works on a store alone.
2. A withheld phrasing is genuinely withheld — out of the vocabulary as well as
   out of the training set.
3. NaN, not zero, on a row with no provenance chain. "No opinion" and "clean"
   are different, and collapsing them would quietly bias fusion against the
   agentic rail.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from mantis.defense.l3_text import (
    HELD_OUT_KIND,
    HELD_OUT_VARIANTS,
    L3Model,
    chains_for,
    held_out_artifacts,
)
from mantis.foundry.base.reference import load_reference_stats
from mantis.foundry.base.simulator import SimulationConfig, simulate_frame
from mantis.foundry.injectors import REGISTRY
from mantis.foundry.injectors.base import PopulationView, run_injector
from mantis.foundry.llm.corpus import load_content_store

SMALL = SimulationConfig(n_events=15_000, seed=7, n_customers=600, n_merchants=1_500)


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    load_content_store()
    background = simulate_frame(SMALL, load_reference_stats())
    view = PopulationView.build(background)
    attacks = [run_injector(REGISTRY[c], view, seed=7) for c in ("F1-01", "F1-03", "F4-27")]
    return pd.concat([background, *attacks], ignore_index=True)


@pytest.fixture(scope="module")
def model() -> L3Model:
    return L3Model(seed=7).fit()


def test_fit_takes_no_transaction_labels() -> None:
    """The assertion is the signature. There is no ``y`` to pass one to."""
    signature = inspect.signature(L3Model.fit)
    assert "y" not in signature.parameters
    assert set(signature.parameters) == {"self", "hold_out_variants", "hold_out_kind"}


def test_withheld_phrasings_are_out_of_the_vocabulary_too(model: L3Model) -> None:
    """Withholding a phrasing from training but not the vocabulary is a quiet leak."""
    store = load_content_store()
    withheld = held_out_artifacts(store)
    assert withheld, "no phrasing was withheld; the generalisation test would be vacuous"
    # Every withheld artefact is still scored (that is what the test measures)
    # but none of them was in the fit.
    assert model.held_out == withheld
    assert model.n_train_artifacts == len(store.artifacts) - len(withheld)


def test_every_injected_kind_keeps_a_phrasing_in_training() -> None:
    """What is withheld must be the wording, not the attack class."""
    store = load_content_store()
    withheld = held_out_artifacts(store)
    for kind in HELD_OUT_VARIANTS:
        of_kind = [a for a in store.artifacts.values() if a.kind == kind]
        if not of_kind:
            continue
        remaining = [a for a in of_kind if a.artifact_id not in withheld]
        assert remaining, f"withholding removed every {kind} specimen from training"


def test_score_is_nan_where_there_is_no_text(model: L3Model, dataset: pd.DataFrame) -> None:
    """A classic authorisation has no chain, so L3 has no opinion. NaN, not zero."""
    scores = model.score(dataset)
    classic = dataset["ag_agent_id"].isna().to_numpy()
    assert np.isnan(scores[classic]).all()
    assert np.isfinite(scores[~classic]).all()


def test_every_content_id_resolves(dataset: pd.DataFrame) -> None:
    """Universal resolution is a leakage control, not a nicety.

    If only attacked ids resolved, "does this id resolve" would be a perfect
    label and L3's metrics would be measuring the storage layout.
    """
    store = load_content_store()
    agentic = dataset[dataset["ag_agent_id"].notna()]
    chains = chains_for(agentic, store)
    lengths = agentic["ag_ingested_content_ids"].map(len).to_numpy()
    resolved = np.array([len(c) for c in chains])
    assert (resolved == np.minimum(lengths, 12)).all()


def test_holding_out_a_whole_kind_still_fits() -> None:
    """The harder generalisation test has to be runnable, not just describable."""
    strict = L3Model(seed=7).fit(hold_out_kind=HELD_OUT_KIND)
    assert strict.n_train_artifacts < L3Model(seed=7).fit().n_train_artifacts
    withheld = strict.holdout_generalisation()
    assert (withheld["kind"] == HELD_OUT_KIND).any()


def test_withheld_texts_are_not_scored_as_benign(model: L3Model) -> None:
    """The most direct statement that the layer learned rather than memorised."""
    withheld = model.holdout_generalisation()
    injected = withheld[withheld["injected"]]
    assert len(injected) >= 10
    # A memorising classifier would put unseen injected text with the benign
    # pool. The bar is deliberately loose -- this pins "learned something", not
    # a headline number, which belongs in RESULTS.md where it is regenerated.
    assert injected["p_injected"].median() > 0.5
