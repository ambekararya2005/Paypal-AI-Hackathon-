# MANTIS, as one container: the built console and its API in a single process.
#
#   docker build -t mantis .
#   docker run -p 7860:7860 mantis                      # mock mode, no keys
#   docker run -p 7860:7860 --env-file .env mantis      # live sandbox, your keys
#
# Render (render.yaml) and Railway inject $PORT; mantis/api/__main__.py reads it,
# with 7860 as the default here.
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
COPY web/public ./public

# No VITE_API_BASE: the default is the relative "/api", right for same origin.
# No licence key either: the AG Studio key is read at runtime from
# /api/client-config, so it lives in Render's env vars and never in the image.
RUN npm run build


# ------------------------------------------------------------ stage 2: runtime
# Python 3.12: what the deployment runs and what the Day 3 test pass verified.
FROM python:3.12-slim

WORKDIR /app

# The `server` extra: the console API plus the live gate, judge, agent and
# playground. Dependencies first so a code change does not reinstall numpy.
COPY pyproject.toml README.md ./
RUN mkdir -p mantis && touch mantis/__init__.py \
 && pip install --no-cache-dir ".[server]" \
 && pip uninstall -y mantis
COPY mantis ./mantis
RUN pip install --no-cache-dir --no-deps -e .

# Committed artefacts only: results, the LLM/agent/judge/compiler caches (the
# replay path), recorded PayPal fixtures, and the small JSON artefacts. No
# parquet, no model file, no .env (the .dockerignore keeps it out of the context).
COPY RESULTS.md ./RESULTS.md
COPY data/cache ./data/cache
COPY data/fixtures ./data/fixtures
COPY data/reference/l3_ood_payloads.json ./data/reference/
COPY data/generated/*.json ./data/generated/
COPY docs ./docs

COPY --from=web /web/dist ./web/dist

# Writable state (gate log, receipts, quota counters, playground candidates).
# On Render a persistent disk mounts here; locally it is the container's own.
ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    MANTIS_RUNTIME_DIR=/var/data
RUN mkdir -p /var/data
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",7860)}/health')"

CMD ["python", "-m", "mantis.api", "--host", "0.0.0.0"]
