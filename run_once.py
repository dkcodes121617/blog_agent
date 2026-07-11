"""Run the generation pipeline once and (in DRY_RUN) write the result to output/.

This is the core "generate one post" entrypoint the scheduler calls. In DRY_RUN
it never touches git — it writes the .mdx + a registry snippet + a run summary to
output/ so you can inspect quality. With DRY_RUN=0 it hands off to the publisher.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

from config import CONFIG
from graph.build import make_pipeline
from knowledge.store import KnowledgeBase


def _setup_logging():
    logging.basicConfig(
        level=getattr(logging, CONFIG.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def run_once() -> dict:
    _setup_logging()
    log = logging.getLogger("agent.run")

    problems = CONFIG.validate_for_publish()
    if problems and not CONFIG.dry_run:
        log.error("cannot publish: %s", "; ".join(problems))
        return {"status": "aborted", "abort_reason": "; ".join(problems)}

    kb = KnowledgeBase()
    if len(kb) == 0:
        log.info("KB empty — running first-time ingest")
        from knowledge.ingest import ingest_existing
        kb = ingest_existing()

    graph, nodes, kb = make_pipeline(kb=kb)
    log.info("invoking pipeline (KB has %d posts, dry_run=%s)", len(kb), CONFIG.dry_run)

    # A proxy outage (502/timeout) mid-run raises out of a node. Catch it so a
    # failed run exits cleanly (status=aborted) instead of crashing — on Render the
    # next hourly cron simply retries the still-due slot.
    from llm.client import LLMError, LLMTransient
    try:
        final = graph.invoke({}, config={"recursion_limit": 50})
    except (LLMError, LLMTransient) as e:
        log.error("pipeline aborted on LLM/proxy error: %s", e)
        return {"status": "aborted", "abort_reason": f"llm error: {str(e)[:200]}"}

    status = final.get("status")
    if status != "ready":
        log.warning("pipeline did not produce a post: %s / %s",
                    status, final.get("abort_reason"))
        return dict(final)

    log.info("READY: slug=%s title=%r", final.get("slug"), final.get("title"))

    if CONFIG.dry_run:
        _write_dry_run(final)
    else:
        from publish.github_publisher import publish_post
        path = publish_post(final, kb)
        final["published_path"] = path
        final["status"] = "published"

    return dict(final)


def _write_dry_run(state: dict):
    out = CONFIG.output_dir
    slug = state["slug"]
    (out / f"{slug}.mdx").write_text(state["body_mdx"], encoding="utf-8")
    registry_entry = {
        "slug": slug,
        "title": state["title"],
        "description": state["description"],
        "date": date.today().isoformat(),
        "tags": state["tags"],
        "readingMinutes": state["reading_minutes"],
    }
    (out / f"{slug}.registry.json").write_text(
        json.dumps(registry_entry, indent=2), encoding="utf-8"
    )
    summary = {
        "slug": slug,
        "title": state["title"],
        "description": state["description"],
        "tags": state["tags"],
        "reading_minutes": state["reading_minutes"],
        "primary_keyword": state.get("primary_keyword"),
        "angle": state.get("angle"),
        "topic_similarity": round(state.get("topic_similarity", 0), 3),
        "body_similarity": round(state.get("body_similarity", 0), 3),
        "revisions": state.get("revision"),
        "humanize_score": state.get("humanize_score"),
        "validation_warnings": state.get("validation_warnings", []),
        "related_slugs": state.get("related_slugs", []),
    }
    (out / f"{slug}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + "=" * 70)
    print("DRY RUN — wrote to output/:")
    print(f"  {slug}.mdx  ({state['reading_minutes']} min, {len(state['body_mdx'])} chars)")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    result = run_once()
    sys.exit(0 if result.get("status") in ("ready", "published") else 1)
