"""Pillar 2's scorecard: is the synthetic data actually like real payment data?

    python -m mantis.foundry.fidelity

Criterion 2 of the judging table is "fidelity of simulation", and the artefact
that scores it is this package. It measures the synthetic population against an
external reference panel on three axes, and it is written to fail loudly rather
than to reassure:

* :mod:`~mantis.foundry.fidelity.marginals` - per-feature KS and JS distances,
  each against the band that pure sampling noise would produce at these sample
  sizes, plus the correlation-matrix distance that catches a generator drawing
  every column independently.
* :mod:`~mantis.foundry.fidelity.tstr` - train a detector on synthetic, test it
  on real, and report the gap against a train-real ceiling.
* :mod:`~mantis.foundry.fidelity.discriminator` - one model trying to tell the
  two panels apart. The target is **0.5**: here, higher is worse.

Two design decisions carry the whole package and are argued where they are made:

1. **Nothing is compared raw.** A rupee population with an agentic rail cannot be
   compared to a dollar panel without one on absolute amount, category, geography
   or channel. Both sides are projected into a dimensionless *shape space* first -
   see :mod:`~mantis.foundry.fidelity.common`.
2. **The reference panel is itself synthetic** (Kaggle's Sparkov). Calling it
   "real" is a claim about its *role* - data this project did not author - and
   :mod:`~mantis.foundry.fidelity.real` says so before quoting a number.

Everything degrades when the panel is absent, which is the state of a clean clone:
the scorecard prints what it can, marks sections 2 to 4 skipped, and refuses to
substitute the self-consistency check in ``scripts/drift_check.py`` for a fidelity
measurement. They answer different questions and conflating them is the easiest
way for this section of a submission to be untrue.
"""

from __future__ import annotations

__all__ = ["build_scorecard", "write_scorecard"]


def __getattr__(name: str):
    # Lazy, like mantis.api: importing the package should not drag LightGBM and
    # scikit-learn into a process that only wanted to ask whether a panel exists.
    if name in __all__:
        from mantis.foundry.fidelity import scorecard

        return getattr(scorecard, name)
    raise AttributeError(name)
