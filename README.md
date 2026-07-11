# WizCodes Blog Agent

An autonomous **LangGraph** agent that writes and publishes genuinely human-like,
SEO-optimized blog posts to the WizCodes site — **1–2 posts/day at randomized
times**, each one **unique** (checked against a knowledge base of every existing
post), grounded in the studio's **real facts** (no hallucinated numbers/clients),
and shipped straight into the site's private GitHub repo.

Only recurring cost: the **ClaudeStore proxy API key**. Everything else (Render
cron, GitHub Actions, Firebase Hosting, local embeddings) is free.

---

## How it works

```
GitHub Actions cron (hourly, free on the public agent repo)
        │  "is a post due now?" (date-seeded plan vs. posts already published today)
        ▼
LangGraph pipeline (Sonnet 4.6 via ClaudeStore proxy)
  load facts → pick topic → [uniqueness gate] → outline → write
    → fact-check → [MDX validate] → humanize → registry
    → [final uniqueness] → publish
        │  commit .mdx + registry entry, push to main
        ▼
GitHub Action → npm run build → firebase deploy  (site goes live)
```

- **Self-correcting**: fact-check and MDX-validation loop back to the writer; a run
  that can't produce a clean, unique, truthful post **aborts and publishes nothing**.
- **Uniqueness**: local MiniLM embeddings (zero API cost). The KB is rebuilt from the
  site repo every run, so it's correct even on Render's stateless free tier.
- **Grounded**: every prompt carries a snapshot of real WizCodes facts (services,
  projects, open-source, existing posts) pulled from the site repo.

See the architecture detail and node-by-node design in the chat plan / `graph/`.

---

## Project layout

```
agent/
  main.py               # cron entrypoint: publish only if a slot is due (--now / --plan)
  run_once.py           # generate one post (dry-run writes to output/)
  config.py             # all env-driven config
  llm/                  # proxy client (CLI UA + retries) + output sanitizer
  graph/                # LangGraph state, nodes, and wiring
  knowledge/            # local-embedding KB (store + ingest) — uniqueness engine
  facts/                # real-WizCodes facts snapshot (anti-hallucination grounding)
  seo/                  # deterministic MDX/BLOG_FORMAT validator + reading time
  publish/              # git publisher (commit + push to the private site repo)
  prompts/              # all LLM prompts (phrased to avoid the proxy's injection guard)
  scheduler/            # human-like randomized daily plan (free-cron friendly)
  render.yaml           # Render Blueprint (free cron job)
```

---

## Local setup & testing

```bash
cd agent
python -m venv .venv && .venv/Scripts/activate      # (venv already present)
pip install -r requirements.txt
cp .env.example .env                                 # fill in the proxy key
python -m knowledge.ingest                           # embed the 12 existing posts
python run_once.py                                   # DRY_RUN=1 → writes to output/
```

Inspect `output/<slug>.mdx`, `output/<slug>.summary.json` (similarity scores,
revisions, humanize score, validation warnings). Nothing is pushed in dry-run.

Useful commands:
```bash
python main.py --plan     # show today's randomized publish plan
python main.py --now      # force one generation now (ignores the schedule)
python main.py            # cron mode: publish only if a slot is due right now
```

---

## Going live — 3 one-time setups

### 1. GitHub token (lets the agent commit posts)
Create a **fine-grained Personal Access Token** scoped to **only** the
`dkcodes121617/wizcodes_main_website` repo, permission **Contents: Read and write**.
This is `GITHUB_TOKEN`.

### 2. GitHub Action → Firebase (builds + deploys on push)
The workflow lives at `wizcodes_next/.github/workflows/deploy.yml`. In that repo's
**Settings → Secrets and variables → Actions**, add:
- `FIREBASE_SERVICE_ACCOUNT` — a Firebase service-account JSON with Hosting deploy
  rights (Firebase console → Project settings → Service accounts → generate key).

Get it once with the CLI if you prefer: `firebase init hosting:github` wires this
up automatically. After this, any push that touches `src/**` builds and deploys.

### 3. Run the agent on GitHub Actions (free)
The agent runs as a scheduled workflow in THIS (public) repo — public repos get
unlimited free Actions minutes, so there's no server to pay for. Render was dropped
because Render cron jobs are not free.

- The workflow is `.github/workflows/publish.yml` (hourly + a manual "Run workflow"
  button). It's stateless — no server, no saved files.
- In this repo → **Settings → Secrets and variables → Actions**, add two secrets:
  - `ANTHROPIC_API_KEY` — your ClaudeStore key.
  - `PUBLISH_TOKEN` — the fine-grained PAT from step 1 (Actions reserves the name
    `GITHUB_TOKEN`, so the secret is `PUBLISH_TOKEN`; the workflow maps it to the
    `GITHUB_TOKEN` env var the app reads).
- Test safely: **Actions → Publish blog post → Run workflow** with *force = true*
  to generate one post immediately.

The cron then fires hourly; each run publishes only if today's plan says a post is
due and fewer than that many are already published today (counted from the site
repo, so it can't double-post).

> **Keep-alive note:** GitHub disables scheduled workflows after ~60 days with no
> commits to the repo. If you don't touch this repo for two months, GitHub emails
> you a one-click re-enable — or just push any small change occasionally.

---

## One-time SEO launch step (optional but recommended)
Verify the site in **Google Search Console** and submit `https://wizcodes.site/sitemap.xml`
**once**. After that, discovery is automatic — the sitemap regenerates on every
build and the `/blog` index links each new post, so **no manual pinging is ever
needed**.

---

## Guardrails baked in
- **Never fabricates**: a fact-check node removes any claim not supported by the
  real facts snapshot; unrecoverable → abort.
- **Never duplicates**: topic-level and body-level cosine gates vs. the KB; a
  near-duplicate → abort (publishes nothing).
- **Never ships broken MDX**: a deterministic validator mirrors the blog contract
  (no H1/frontmatter, required components, no raw `<`/`{` in prose, valid internal
  links); failures loop back to the writer.
- **Anti-flag cadence**: ≤2 posts/day, randomized times, occasional zero-days.
- **Proxy quirks handled**: CLI User-Agent (or Cloudflare 403s) and
  injection-guard-safe prompt phrasing.

## Tuning
Everything is env-driven (`.env` locally / workflow `env:` in Actions):
`MAX_POSTS_PER_DAY`, `AVG_POSTS_PER_DAY`, publish window, `MIN_GAP_HOURS`,
`TOPIC_SIM_THRESHOLD`, `BODY_SIM_THRESHOLD`, `SCHEDULE_TZ`.
