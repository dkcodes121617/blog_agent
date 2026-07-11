"""Cron entrypoint for the WizCodes blog agent.

Runs on a schedule (hourly) from GitHub Actions (free for the public agent repo)
or any other cron. STATELESS by design — nothing is persisted between runs:

  1. Decide if a post is due now: compare today's deterministic plan (seeded by the
     date, so every run agrees) against how many posts are ALREADY published today
     (counted from the site repo's posts.ts — the real source of truth).
  2. If due: health-check the proxy, rebuild the KB from the site repo, generate +
     publish one post.
  3. Otherwise exit immediately (near-zero cost).

Because "already published today" is read from the repo, it's impossible to
double-post even though runners are ephemeral and keep no local state.

Run modes:
  python main.py            # cron mode: publish only if a post is due now
  python main.py --now      # force one generation now (ignores the schedule)
  python main.py --plan     # print today's deterministic plan and exit
"""
from __future__ import annotations

import logging
import sys

from config import CONFIG


def _setup_logging():
    logging.basicConfig(
        level=getattr(logging, CONFIG.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "sentence_transformers", "urllib3", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


def _ensure_kb(site_dir=None):
    """Rebuild the KB from the (freshly pulled) site repo.

    The KB is DERIVED state — every published post lives in the site repo's
    posts.ts + <slug>.mdx. Re-ingesting each run makes the agent safe on stateless
    runners: uniqueness memory is rebuilt from the repo, which already contains
    everything ever published (including prior auto-posts).
    """
    from knowledge.ingest import ingest_existing
    return ingest_existing(site_dir=site_dir)


def main(argv: list[str]) -> int:
    _setup_logging()
    log = logging.getLogger("agent.main")

    if "--plan" in argv:
        from scheduler.planner import plan_times, slots_due
        times = plan_times()
        print(f"Today's plan: {len(times)} post(s) at " +
              (", ".join(t.strftime('%H:%M') for t in times) or "(none)"))
        print(f"Slots due by now: {slots_due()}")
        return 0

    force = "--now" in argv

    # In live mode we need the site repo to (a) count today's posts and (b) rebuild
    # the KB. In dry-run we use the local sibling site folder. Clone/pull up front
    # only when we might actually publish.
    site_dir = None

    if not force:
        # Cheap decision first. In live mode this needs the repo (to count today's
        # posts); in dry-run, use the local folder.
        from publish.github_publisher import count_posts_today
        if not CONFIG.dry_run:
            from publish.github_publisher import ensure_repo
            repo = ensure_repo()
            site_dir = repo.working_tree_dir
        published_today = count_posts_today(site_dir)

        from scheduler.planner import is_post_due
        if not is_post_due(published_today):
            log.info("no post due this run (published today=%d) — exiting", published_today)
            return 0
        log.info("a post is due now (published today=%d) — generating", published_today)

    # Pre-flight health check (circuit breaker): if the proxy is 502-ing/slow, exit
    # cheaply now. Nothing is marked done, so the next hourly run simply retries.
    from llm.client import LLMClient
    ok, detail = LLMClient().ping()
    if not ok:
        log.warning("proxy health check failed (%s) — skipping this run", detail[:120])
        return 0
    log.info("proxy healthy — proceeding")

    # Ensure the repo is present (for --now we may not have cloned above).
    if not CONFIG.dry_run and site_dir is None:
        from publish.github_publisher import ensure_repo
        repo = ensure_repo()
        site_dir = repo.working_tree_dir
    _ensure_kb(site_dir=site_dir)

    from run_once import run_once
    result = run_once()
    status = result.get("status")

    if status in ("ready", "published"):
        log.info("run succeeded: %s (%s)", result.get("slug"), status)
        return 0

    # Generation aborted (unique-topic exhaustion or a proxy error). Nothing is
    # persisted, so the next hourly run retries cleanly.
    log.warning("run did not publish: %s / %s", status, result.get("abort_reason"))
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
