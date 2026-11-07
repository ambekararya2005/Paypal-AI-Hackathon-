---
title: MANTIS
emoji: 🦟
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Adversarial fraud foundry for agentic commerce
---

# MANTIS

**Agentic commerce created a live payment rail with zero labelled fraud data.**
You cannot train a detector on data that does not exist — so MANTIS manufactures
it, adversarially, and then builds the detector against it.

This Space is the **live defence console**: the React front end and its FastAPI
backend in one container, one origin. Press **Start authorisation stream** and
watch pre-scored agent-initiated payment authorisations arrive one at a time, get a
decision from a five-layer firewall, and resolve against ground truth.

### What you are looking at

| screen | what it shows |
|---|---|
| **Console** | Authorisations scored live. Click a declined row for the layer scores and the three features that drove the decision. |
| **Results** | Every measured number: recall at a fixed 0.1% false-positive rate, the leave-one-family-out collapse, and the zero-day recovery. |
| **Atlas** | 42 attack cards across six families — the executable taxonomy the generator imports. |
| **Fidelity** | Whether the synthetic data is realistic, measured against the real Sparkov panel: KS, JS, TSTR, and a real-vs-synthetic discriminator. |

### The one number

**Recall 0.450 at a 0.1% false-positive rate**, and the zero-day row underneath
it: a detector that has never seen attack family F1 catches **1.3%** of it — and
**53.9%** once the evolutionary loop has manufactured that family from its atlas
card alone.

No accuracy figure appears anywhere in this project. At ~1% prevalence, a model
that approves everything is 99% accurate.

### Nothing here runs a model

Every score, decision, attribution and metric was computed offline and committed.
A request to this Space reads a file. That is why it is instant on a phone, and
it is why the numbers on this page are the same numbers `python -m mantis.defense`
prints from a clean clone.

Source, and the full 36-hour build log, in the repository.
