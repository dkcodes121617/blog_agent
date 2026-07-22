"""One-time (idempotent) bootstrap of the knowledge base from existing posts.

Reads the 12 real posts from the site repo — registry metadata (posts.ts) plus
each `<slug>.mdx` body — and embeds them into the KB so the very first generated
post is already checked against everything that exists.

AUTHORITATIVE, not additive. posts.ts is the source of truth: anything in the
ledger that is not in posts.ts is removed. It used to only ever append, so a
renamed post lingered forever and `known_slugs` reported URLs that no longer
existed — the MDX validator then happily passed a /blog/ link straight to a 404.
"""
from __future__ import annotations

import logging
from pathlib import Path

from config import CONFIG
from facts.snapshot import _resolve_site_dir, build_snapshot
from knowledge.store import KnowledgeBase

log = logging.getLogger("agent.kb.ingest")


def _read_body(site_dir: Path, slug: str) -> str:
    path = site_dir / CONFIG.blog_content_rel / f"{slug}.mdx"
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def ingest_existing(site_dir: Path | str | None = None) -> KnowledgeBase:
    # Coerce to Path — callers may pass a str (e.g. GitPython's working_tree_dir).
    site_dir = Path(site_dir) if site_dir else _resolve_site_dir()
    snap = build_snapshot(site_dir)
    kb = KnowledgeBase()

    # Prune BEFORE adding, so a rename lands as one clean replacement rather than
    # leaving the old slug alive alongside the new one.
    live = {p["slug"] for p in snap.existing_posts}
    kb.prune_to(live)

    added = 0
    for post in snap.existing_posts:
        slug = post["slug"]
        if kb.has_slug(slug):
            continue
        body = _read_body(site_dir, slug)
        kb.add(
            slug=slug,
            title=post["title"],
            description=post["description"],
            tags=post["tags"],
            body_text=body,
        )
        added += 1

    log.info("ingest complete: +%d post(s), KB now mirrors posts.ts with %d entries",
             added, len(kb))
    return kb


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    kb = ingest_existing()
    print(f"KB entries: {len(kb)}")
    print("slugs:", kb.all_slugs())
