# Deploying MANTIS

**Live: <https://mantis-ze19.onrender.com>**, one Render web service (Starter plan, 1 GB
disk), defined in [`render.yaml`](render.yaml) and built from the [`Dockerfile`](Dockerfile).

The runbook (Blueprint setup, secrets, environment variables, the kill switch and
the abuse guards) is [`docs/deploy_render.md`](docs/deploy_render.md).

Before the PayPal hackathon, the offline console was published as a static Hugging
Face Space (<https://aryaambekar-mantis.static.hf.space>) with `scripts/deploy_hf.py
--static`. That page still serves the frozen offline console. It has no gate, no
PayPal and no playground, and it is no longer the demo.
