"""Publish a finished post into the private wizcodes repo via git, then push main.

The agent only touches git — it never builds Next.js or deploys. Pushing to the
content path triggers the repo's GitHub Action, which builds and runs
`firebase deploy`. Flow:

  1. Ensure a local clone of the repo exists (clone once, pull thereafter) using
     the GitHub token embedded in the remote URL.
  2. Write src/content/blog/<slug>.mdx.
  3. Insert the typed registry entry at the top of the posts[] array in posts.ts.
  4. Commit (as the blog bot) and push to the configured branch (main).
  5. Update the local KB with the new post so tomorrow's run sees it.

All string insertion is done with anchored, idempotent edits and verified before
commit; if anything looks off we raise instead of pushing a broken file.
"""
from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path

from git import Repo

from config import CONFIG
from knowledge.store import KnowledgeBase

log = logging.getLogger("agent.publish")

_REGISTRY_ANCHOR = "export const posts: BlogPost[] = ["


def _authed_remote() -> str:
    """https remote with the token embedded (never logged)."""
    token = CONFIG.github_token
    return f"https://x-access-token:{token}@github.com/{CONFIG.github_repo}.git"


def ensure_repo() -> Repo:
    d = CONFIG.site_repo_dir
    if (d / ".git").exists():
        repo = Repo(d)
        log.info("pulling latest %s", CONFIG.github_branch)
        repo.remotes.origin.set_url(_authed_remote())
        repo.git.checkout(CONFIG.github_branch)
        repo.remotes.origin.pull(CONFIG.github_branch)
        return repo
    log.info("cloning %s into %s", CONFIG.github_repo, d)
    return Repo.clone_from(_authed_remote(), d, branch=CONFIG.github_branch, depth=1)


def _render_registry_entry(state: dict, iso_date: str) -> str:
    """Build the TypeScript object literal for posts.ts (matches the file's style)."""
    tags = ", ".join(f"'{_esc(t)}'" for t in state["tags"])
    return (
        "  {\n"
        f"    slug: '{_esc(state['slug'])}',\n"
        f"    title: '{_esc(state['title'])}',\n"
        "    description:\n"
        f"      '{_esc(state['description'])}',\n"
        f"    date: '{iso_date}',\n"
        f"    tags: [{tags}],\n"
        f"    readingMinutes: {int(state['reading_minutes'])},\n"
        "  },\n"
    )


def _esc(s: str) -> str:
    """Escape for a single-quoted TS string. Also strip chars MDX/TS dislike."""
    return s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ").strip()


def _insert_registry(posts_ts: str, entry: str) -> str:
    idx = posts_ts.find(_REGISTRY_ANCHOR)
    if idx == -1:
        raise RuntimeError("could not find posts[] array anchor in posts.ts")
    insert_at = idx + len(_REGISTRY_ANCHOR)
    # Insert the new entry right after the opening '[' (newest-first ordering).
    # The anchor ends with '['; the file then has a newline before the first entry.
    # We emit "\n<entry>" and keep everything that followed the bracket intact.
    head = posts_ts[:insert_at]
    tail = posts_ts[insert_at:]
    return f"{head}\n{entry.rstrip()}{tail}"


def publish_post(state: dict, kb: KnowledgeBase | None = None) -> str:
    """Write files, commit, push. Returns the committed .mdx path (repo-relative)."""
    if not CONFIG.github_token:
        raise RuntimeError("GITHUB_TOKEN not set — cannot publish")

    repo = ensure_repo()
    root = Path(repo.working_tree_dir)
    slug = state["slug"]
    iso_date = state.get("date") or date.today().isoformat()

    mdx_rel = f"{CONFIG.blog_content_rel}/{slug}.mdx"
    mdx_path = root / mdx_rel
    if mdx_path.exists():
        raise RuntimeError(f"refusing to overwrite existing post: {mdx_rel}")
    mdx_path.write_text(state["body_mdx"], encoding="utf-8")

    registry_path = root / CONFIG.posts_registry_rel
    posts_ts = registry_path.read_text(encoding="utf-8")
    if f"slug: '{slug}'" in posts_ts:
        raise RuntimeError(f"slug already in registry: {slug}")
    entry = _render_registry_entry(state, iso_date)
    new_ts = _insert_registry(posts_ts, entry)
    # Sanity: the entry must now be present exactly once.
    if new_ts.count(f"slug: '{slug}'") != 1:
        raise RuntimeError("registry insertion sanity check failed")
    registry_path.write_text(new_ts, encoding="utf-8")

    # Commit + push.
    repo.git.add(mdx_rel, CONFIG.posts_registry_rel)
    author = f"{CONFIG.git_author_name} <{CONFIG.git_author_email}>"
    repo.git.commit(
        "-m", f"blog: {state['title']}",
        "-m", f"Automated post. keyword={state.get('primary_keyword','')}",
        "--author", author,
    )
    log.info("pushing to %s", CONFIG.github_branch)
    repo.remotes.origin.push(CONFIG.github_branch)

    # Update KB so the next run sees this post.
    if kb is not None:
        kb.add(
            slug=slug, title=state["title"], description=state["description"],
            tags=state["tags"], body_text=state["body_mdx"],
        )

    log.info("published %s", mdx_rel)
    return mdx_rel
