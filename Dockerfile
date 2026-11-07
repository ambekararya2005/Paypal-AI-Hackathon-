# MANTIS, as one container: the built console and its API in a single process.
#
#   docker build -t mantis .
#   docker run -p 7860:7860 mantis     # -> http://localhost:7860
#
# The target is a Hugging Face Docker Space, which expects the app on 7860.
# Render and Railway inject $PORT instead; both work unchanged because
# mantis/api/__main__.py reads $PORT with 7860 as the default here.
#
# Why one container and not two services
# --------------------------------------
# The console fetches its data from /api on its own origin, so there is no CORS
# preflight, no second host to keep awake, no cross-service latency, and exactly
# one URL to hand a judge. See mantis/api/site.py.

# --------------------------------------------------------------- stage 1: web
# Node exists only to produce web/dist. It is not in the final image.
FROM node:20-alpine AS web

WORKDIR /web
# The lockfile first, so a change to a .tsx does not reinstall the toolchain.
COPY web/package.json web/package-lock.json ./
RUN npm ci

COPY web/tsconfig.json web/vite.config.ts web/index.html ./
COPY web/src ./src
# public/data holds the frozen API responses that back the console's offline
# mode. They are committed, and they ship even here: if the API ever fails to
# answer /health, the deployed page falls back to the replay instead of blanking.
COPY web/public ./public

# No VITE_API_BASE: the default is the relative "/api", which is exactly right
# for a same-origin deployment. Baking an absolute host in here is what created
# the CORS problem this Dockerfile exists to remove.
RUN npm run build


# ------------------------------------------------------------ stage 2: runtime
FROM python:3.11-slim

# HF Spaces runs the container as uid 1000 and mounts nothing writable at /app,
# so everything the process needs is baked in and read-only at runtime.
WORKDIR /app

# Install the serve extra only. `--no-deps` on the project itself keeps pip from
# resolving the full scientific stack, which this container never imports; see
# the comment on [project.optional-dependencies].serve in pyproject.toml.
COPY pyproject.toml README.md ./
COPY mantis ./mantis
RUN pip install --no-cache-dir \
      "pydantic>=2.7" "pyyaml>=6.0" "fastapi>=0.110" "uvicorn>=0.29" "sse-starlette>=2.0" \
 && pip install --no-cache-dir --no-deps -e .

# The committed artefacts the API serves. data/generated is gitignored except
# for these, which are the demo's whole payload: pre-scored authorisations, the
# arena, the fidelity scorecard, and the population manifest. No parquet, no
# model file, nothing that needs refitting.
COPY RESULTS.md ./RESULTS.md
COPY data/cache ./data/cache
COPY data/reference/l3_ood_payloads.json ./data/reference/
# A glob rather than a list: every artefact under data/generated that is small
# enough to commit is JSON, the .dockerignore drops the parquets and the pickle,
# and a named file that has not been generated yet would fail the build outright.
COPY data/generated/*.json ./data/generated/
COPY docs ./docs

# The bundle from stage 1, at the path mantis/api/site.py resolves.
COPY --from=web /web/dist ./web/dist

ENV PORT=7860 \
    PYTHONUNBUFFERED=1
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",7860)}/health')"

CMD ["python", "-m", "mantis.api", "--host", "0.0.0.0"]
