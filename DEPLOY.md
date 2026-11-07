# Deploying MANTIS

Three artifacts go to the judges: **the repo**, **the document**, **the
prototype URL**. This file is the runbook for the third.

---

## It is deployed

**<https://aryaambekar-mantis.static.hf.space>**

Verified from here: `index.html`, both bundles and all six frozen payloads return
200, and the numbers coming off the wire are the ones in `RESULTS.md`
(discriminator 0.9994 / 0.8399, latency p99 171.4 ms, zero-day 0.811 / 0.013 /
0.539, 42 atlas cards, 600 feed frames). What is **not** verified from here is how
it looks; open it on a phone.

To redeploy after any change:

```bash
make web                                                  # refreeze + rebuild
python scripts/deploy_hf.py --space <you>/mantis --static
```

Note the subdomain: a **static** Space is served from
`<user>-<space>.static.hf.space`, not `<user>-<space>.hf.space`. The second 404s
while the Space itself reports RUNNING, which is exactly as confusing as it
sounds, and the deploy script now prints the right one.

## The two modes, and why the free one is not a compromise

`--static` uploads `web/dist` to a **static** Space. That bundle is
self-contained by construction: Day 6 froze every API response into
`web/public/data/` **by calling the real route handlers**, and the console
replays the committed authorisation feed on a client-side timer when nothing
answers `/api/health`. So a static Space is not a degraded build - it is the
offline mode the console already shipped. No cold start, nothing to keep awake,
no cost, and a CDN cannot be asleep when a judge clicks.

What it gives up is exactly one thing: the stream is a client-side replay rather
than real server-sent events. The console says which mode it is in, in the
control row, rather than letting a viewer assume a backend exists.

Without `--static`, the script pushes the whole repo to a **Docker** Space
running `mantis/api/site.py` - the console at `/` and the live API under `/api`,
one process, one origin. **Hugging Face now bills Docker Spaces**: a free account
gets a `402 Payment Required` on `create_repo`, and the script catches that and
tells you to re-run with `--static`. Static Spaces stay free for everyone.

First Docker build is ~3-5 minutes (npm, then pip). A static Space goes live in
seconds, with no build step at all.

---

## Why one container

Day 6 shipped two deployables — a static bundle on one host, an API on another —
and the console had to negotiate across an origin boundary between them. That is
two things to keep alive, a CORS regex that has to anticipate every preview
subdomain, and two URLs to hand over.

Day 7 collapsed it. `mantis/api/site.py` mounts the API at `/api` and serves
`web/dist` at `/`. `web/src/api.ts` already defaulted `API_BASE` to the relative
`/api`, so a same-origin build needs no `VITE_API_BASE` and never exercises a
CORS header at all.

**Hugging Face rather than Render or Railway**, and the reason is availability
rather than elegance: a free dyno on either **sleeps after ~15 minutes idle and
takes ~30 s to wake**, which is exactly the shape of a demo that looks broken at
the moment a judge clicks it.

That argument survived the discovery that Docker Spaces are no longer free - it
just resolved the other way. The static Space is a CDN, and a CDN has no cold
start at all, which is a stronger version of the property we were buying. The
composed container below is still the better *demo* (a real SSE stream instead of
a client-side replay) and it is still what runs locally and what the Dockerfile
builds; it is simply not what is deployed today.

The container is small on purpose. `mantis/api/` runs no model inside a request —
every score, metric and attribution is read off a committed artefact — so the
image installs the `serve` extra only. Verified rather than assumed: importing
`mantis.api.site` loads none of pandas, numpy, scikit-learn, scipy, LightGBM,
SHAP, matplotlib, networkx or pyarrow. That takes the image from ~1.5 GB to
~200 MB and the cold build from minutes to seconds.

---

## The procedure, in full

### 1. A Hugging Face write token

<https://huggingface.co/settings/tokens>, **write** scope. Then either:

```bash
hf auth login          # stores it; the deploy script picks it up
# or
export HF_TOKEN=hf_...
# or
python scripts/deploy_hf.py --space <you>/mantis --static --token hf_...
```

The token is never written into the repo. HARD RULE 4 is about a clean clone
needing no credentials, and deployment is the one place one is used.

### 2. Preflight

```bash
python scripts/deploy_hf.py --check --static     # 11 files, 1.5 MB
python scripts/deploy_hf.py --check              # 430 files, 4.5 MB (docker)
```

Lists what would be uploaded and names anything missing that
would produce a Space that builds and then serves an empty console — a missing
`console_feed.json`, a `fidelity.json` that has never been generated, a frozen
`web/public/data/fidelity.json` that is stale against it.

### 3. Push

```bash
python scripts/deploy_hf.py --space <you>/mantis --static
```

Creates the Space if it does not exist, uploads the matching Space card
(`deploy/hf/README-static.md` or `README.md` - the YAML frontmatter is what tells
Spaces which SDK it is, and for Docker which port), then uploads the payload.

Static lands immediately at `https://<you>-mantis.static.hf.space`. Docker builds
`web/dist` itself from `web/src` in the Dockerfile's node stage - so the
gitignored `dist` never has to be committed - and lands at
`https://<you>-mantis.hf.space` after ~3-5 minutes. Watch the **Logs** tab.

### What is uploaded, and what is not

Only what the `Dockerfile` copies: the package, the web sources, the committed
LLM cache, the small JSON artefacts, `RESULTS.md` and `docs/`. Explicitly **not**
the parquets (hundreds of megabytes), the pickled experiment, or the Kaggle
reference CSVs. `ALLOW` and `IGNORE` in `scripts/deploy_hf.py` are kept in one
list beside the `Dockerfile` so drift between them is a visible edit rather than
a build failure ten minutes into a push.

---

## Running it locally first

```bash
make web                       # make static, then npm run build
python -m mantis.api           # -> http://127.0.0.1:8000
```

`/` is the console, `/api-docs` is the API. `--api-only` serves the bare JSON API
at the root, which is how Day 6 ran it.

To build the container locally (needs Docker, which the dev machine does not
have):

```bash
docker build -t mantis .
docker run -p 7860:7860 mantis
```

---

## Other static hosts, if the Space fails on the night

`web/dist` is a **self-contained static site** wherever it is put. It carries
frozen copies of every API response in `web/public/data/` — produced by calling
the real route handlers, never by a second implementation — and replays the
committed authorisation feed on a client-side timer. The console probes
`/api/health` once with a two-second timeout and displays which mode it is in.

So any static host will do:

```bash
make web
npx vercel deploy --prod ./web/dist       # or drag web/dist to netlify.com/drop
```

`web/vercel.json` sets the framework and the SPA rewrite. One caveat on the
dashboard route: Vercel builds from the git checkout, so `web/public/data/` must
be **committed** for the offline mode to survive the build. It is. If you
regenerate it with `make static`, commit the result.

`render.yaml` and `railway.json` are still committed and still work, with the
sleep caveat above.

---

## Before you believe it works

Verified from here already: a clean clone serves every endpoint, the composed app
serves the console and the API on one origin, the SPA fallback resolves deep
links while a mistyped asset still 404s, the SSE stream delivers well-formed
frames, the docx regenerates from `RESULTS.md`, and 287 tests pass. On the
**deployed** URL, every asset and every frozen payload returns 200 and carries the
numbers `RESULTS.md` quotes.

What no check from this machine can tell you is whether the thing *looks* right —
whether a chart overflows on a 390px screen, whether the drawer scrolls, whether
the stream is watchable at six rows a second. Do that yourself:

- [ ] Open the URL **on a phone**, on mobile data rather than the venue wifi.
- [ ] Press **Start authorisation stream** and watch rows arrive and resolve
      red/green.
- [ ] Click a declined row — the alert panel must show the layer bars and the
      top-3 contributions.
- [ ] **Results** — the three-row recovery table (0.811 / 0.013 / 0.539) must
      render large.
- [ ] **Attack atlas** — the six family bars, and a card drawer that opens with
      its observable signals wired to feature names.
- [ ] **Fidelity** — the discriminator chart with its 0.5 target line, and the
      latency panel underneath it.
- [ ] Check the badge in the console's control row: `API connected` or
      `offline replay`. Either is fine; know which one you are demoing.

---

## Regenerating everything

```bash
make firewall      # refits the layers, rewrites RESULTS.md      (~15 min)
make loop          # the evasion curve and the zero-day rows     (~35 min)
make reference     # the Kaggle Sparkov panel, ~210 MB           (~1 min)
make fidelity      # the scorecard -> data/generated/fidelity.json
make latency       # p50/p95/p99 -> data/generated/latency.json  (~3 min)
make submission    # ood, feed, docx, static, web                (~4 min)
```

The order matters. `RESULTS.md` is written by `make firewall`, and it renders the
fidelity and latency sections **from their JSON artefacts** — so generate those
first, then `make render` to fold them in without a refit. The document and the
results screen are both generated from `RESULTS.md`, so they cannot drift from
what the code produces. Nothing downstream is ever retyped.

`make fidelity` is deliberately **not** a prerequisite of `make submission`: it
needs the 210 MB Kaggle panel and exits 1 without it, and
`data/generated/fidelity.json` is committed — so a clean clone assembles the
submission with no download.
