"""Cron entrypoint for the WizCodes blog agent (free Render Cron Job).

Render invokes this on a schedule (hourly). Each run:
  1. Ensures the KB is bootstrapped.
  2. Asks the planner whether a post is due right now.
  3. If due, generates + publishes one post, then marks the slot done.
  4. Otherwise exits immediately (near-zero cost).

Run modes:
  python main.py            # cron mode: publish only if a slot is due
  python main.py --now      # force one generation now (ignores the schedule)
  python main.py --plan     # print today's plan and exit
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
    """Refresh the KB from the (freshly pulled) site repo.

    The KB is DERIVED state — every published post lives in the site repo's
    posts.ts + <slug>.mdx. So on each run we re-ingest from there, which makes the
    agent safe on Render's free tier (no persistent disk): even if the local KB
    files are wiped between cron runs, uniqueness memory is rebuilt from the repo,
    which already contains everything ever published (including prior auto-posts).
    """
    from knowledge.ingest import ingest_existing
    return ingest_existing(site_dir=site_dir)


def main(argv: list[str]) -> int:
    _setup_logging()
    log = logging.getLogger("agent.main")

    if "--plan" in argv:
        from scheduler.planner import load_or_make_plan
        plan = load_or_make_plan()
        print(f"Plan for {plan.day}:")
        for s in plan.slots:
            print(f"  {s.time_iso[11:16]}  {'done' if s.done else 'pending'}")
        if not plan.slots:
            print("  (no posts scheduled today)")
        return 0

    force = "--now" in argv

    # Cheap check FIRST — most hourly cron runs have nothing due and should exit
    # in a second without cloning the repo or loading the embedding model.
    slot = None
    if not force:
        from scheduler.planner import due_slot, mark_slot_done
        slot = due_slot()
        if slot is None:
            log.info("no post due this run — exiting")
            return 0
        log.info("slot %s is due — generating", slot.time_iso[11:16])

    # Pre-flight health check (circuit breaker): if the proxy is having a 502/slow
    # spell, exit cheaply now instead of burning topic/outline calls into a wall.
    # The slot stays due, so the next hourly cron retries when the proxy recovers.
    from llm.client import LLMClient
    ok, detail = LLMClient().ping()
    if not ok:
        log.warning("proxy health check failed (%s) — skipping this run; slot stays due", detail[:120])
        return 0
    log.info("proxy healthy — proceeding")

    # A post is due (or --now). In live mode, pull the site repo up front so the
    # KB is rebuilt against everything already published (prior auto-posts too),
    # which keeps uniqueness correct even on a stateless free-tier box.
    site_dir = None
    if not CONFIG.dry_run:
        from publish.github_publisher import ensure_repo
        repo = ensure_repo()
        site_dir = repo.working_tree_dir
    _ensure_kb(site_dir=site_dir)

    from run_once import run_once
    result = run_once()
    status = result.get("status")

    if status in ("ready", "published"):
        if not force:
            mark_slot_done(slot)
        log.info("run succeeded: %s (%s)", result.get("slug"), status)
        return 0

    # Generation aborted (e.g. couldn't find a unique topic). Do NOT mark the slot
    # done — a later cron run will retry the slot.
    log.warning("run did not publish: %s / %s", status, result.get("abort_reason"))
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
