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


def count_posts_by_date(site_dir=None) -> dict[str, int]:
    """{ISO date -> posts published that day}, read from the site repo's posts.ts.

    This is the stateless anti-double-post guard: instead of persisting how many
    we posted, we ask the real source of truth. Works on ephemeral runners.

    The whole map rather than just today's count, because the scheduler also needs
    to know whether a PREVIOUS day's slot went unfilled — a run that fails leaves
    no trace anywhere else, and without this the post it should have published is
    simply lost at midnight.
    """
    site_dir = Path(site_dir) if site_dir else CONFIG.site_repo_dir
    registry = site_dir / CONFIG.posts_registry_rel
    if not registry.exists():
        return {}
    text = registry.read_text(encoding="utf-8", errors="replace")
    counts: dict[str, int] = {}
    for iso in re.findall(r"date:\s*'(\d{4}-\d{2}-\d{2})'", text):
        counts[iso] = counts.get(iso, 0) + 1
    return counts




def ensure_repo() -> Repo:
    d = CONFIG.site_repo_dir
    if (d / ".git").exists():
        repo = Repo(d)
        log.info("pulling latest %s", CONFIG.github_branch)
        repo.remotes.origin.set_url(_authed_remote())
        repo.git.checkout(CONFIG.github_branch)
        repo.remotes.origin.pull(CONFIG.github_branch)
    else:
        log.info("cloning %s into %s", CONFIG.github_repo, d)
        repo = Repo.clone_from(_authed_remote(), d, branch=CONFIG.github_branch, depth=1)
    # CI runners (GitHub Actions) have NO global git identity, so a commit would
    # fail with "empty ident name". Set a repo-local committer identity = the bot.
    repo.git.config("user.email", CONFIG.git_author_email)
    repo.git.config("user.name", CONFIG.git_author_name)
    return repo


# ─── SERP length budgets ───
#
# The site's layout appends " | WizCodes" (11 chars) to every title via the root
# metadata template, so the BARE title has 49 characters before the brand suffix
# starts getting cut. Google truncates around 575-600px, which is roughly 55-60
# characters at average letter widths; past that it drops the brand and then real
# words, and past ~65 it often rewrites the title from page content instead —
# which throws away the keyword placement the outliner worked for.
#
# The prompts already ask for "~50-60 chars" (library.py) and "140-160 chars",
# but nothing enforced it, so 11 of 17 published titles ran 62-84 characters and
# 24 pages shipped descriptions over 165. Asking is not enforcing.
#
# Trimming here rather than rejecting: the post is written, validated and about to
# ship, and a hard failure at this point costs the whole run. A tidy trim at a word
# boundary is strictly better than a mid-word ellipsis added by Google.
TITLE_BUDGET = 52
DESC_BUDGET = 158
# Below this a shortened title has lost so much that the original, truncated by
# Google, is the better outcome. A headline is not salvageable by machine.
TITLE_FLOOR = 24
# Separators a writer uses to append a secondary clause. Cutting here keeps the
# leading clause, which is where the outliner is told to put the primary keyword.
_TITLE_CUTS = (" (", ": ", " — ", " – ", " - ", " | ")
_SENTENCE_END = (".", "!", "?")


def _fit_title(title: str, slug: str) -> str:
    """Shorten a title ONLY at a natural clause boundary — never mid-thought.

    Word-boundary trimming was tried and rejected: it turned "What custom software
    actually costs in 2026 (and why AI changed the math)" into "What custom software
    actually costs in 2026 (and", which is worse than anything Google would do to it.
    A fragment in the SERP costs more than a truncation, so when no clean cut exists
    this returns the original and logs instead.
    """
    title = (title or "").strip()
    if len(title) <= TITLE_BUDGET:
        return title

    # Longest clean cut that fits — keeps as much of the headline as possible.
    best = ""
    for sep in _TITLE_CUTS:
        idx = title.find(sep)
        while idx != -1:
            head = title[:idx].strip().rstrip(",;:-–—(")
            if TITLE_FLOOR <= len(head) <= TITLE_BUDGET and len(head) > len(best):
                best = head
            idx = title.find(sep, idx + 1)

    if best:
        log.warning(
            "%s: title %d chars -> %d, cut at a clause boundary. Was: %r",
            slug, len(title), len(best), title,
        )
        return best

    log.error(
        "%s: title is %d chars (budget %d) and has no clean cut point, so it ships "
        "as written and Google will truncate it. Fix the length instruction in "
        "prompts/library.py — it does not account for the ' | WizCodes' suffix the "
        "site's metadata template appends. Title: %r",
        slug, len(title), TITLE_BUDGET, title,
    )
    return title


def _fit_description(desc: str, slug: str) -> str:
    """Trim a description to the budget, preferring a sentence boundary."""
    desc = (desc or "").strip()
    if len(desc) <= DESC_BUDGET:
        return desc

    # Prose survives a clean cut, unlike a headline: prefer the last full sentence.
    window = desc[:DESC_BUDGET]
    cut = max(window.rfind(c) for c in _SENTENCE_END)
    if cut >= 80:
        out = window[: cut + 1].strip()
    else:
        words, out = window.split(), ""
        for w in words:
            candidate = f"{out} {w}".strip()
            if len(candidate) > DESC_BUDGET - 1:
                break
            out = candidate
        out = out.rstrip(" ,;:-–—(")
        for tail in (" and", " or", " with", " for", " the", " a", " to", " in", " of"):
            if out.endswith(tail):
                out = out[: -len(tail)].rstrip(" ,;:-")
        if out and not out.endswith(_SENTENCE_END):
            out += "."

    log.warning("%s: description %d chars -> %d. Was: %r", slug, len(desc), len(out), desc)
    return out


def _fit_metadata(state: dict) -> None:
    """Clamp title/description to their SERP budgets, in place.

    The prompts already ASK for these lengths (prompts/library.py) and nothing
    enforced it, so 11 of 17 published titles ran 62-84 characters including the
    brand suffix and 24 pages shipped descriptions over 165. Asking is not
    enforcing, and a truncated SERP entry costs every impression for the life of
    the post.
    """
    slug = state.get("slug", "?")
    state["title"] = _fit_title(state.get("title", ""), slug)
    state["description"] = _fit_description(state.get("description", ""), slug)


def _render_registry_entry(state: dict, iso_date: str) -> str:
    """Build the TypeScript object literal for posts.ts (matches the file's style)."""
    _fit_metadata(state)
    tags = ", ".join(f"'{_esc(t)}'" for t in state["tags"])
    # Record the shape this post was written as, so the NEXT run's rotation reads a
    # fact instead of re-deriving it from the title with a regex. Omitted entirely
    # when unset rather than written as '' — an empty string is a value the reader
    # would have to special-case, whereas an absent optional field already means
    # "fall back to inference".
    archetype = _esc(state.get("archetype", ""))
    archetype_line = f"    archetype: '{archetype}',\n" if archetype else ""
    # Which cover artwork the site renders for this post. Same optional-field contract
    # as archetype: absent means the site derives one from the slug hash, which is how
    # every post published before covers existed still has one.
    motif = _esc(state.get("cover_motif", ""))
    motif_line = f"    coverMotif: '{motif}',\n" if motif else ""
    return (
        "  {\n"
        f"    slug: '{_esc(state['slug'])}',\n"
        f"    title: '{_esc(state['title'])}',\n"
        "    description:\n"
        f"      '{_esc(state['description'])}',\n"
        f"    date: '{iso_date}',\n"
        f"    tags: [{tags}],\n"
        f"    readingMinutes: {int(state['reading_minutes'])},\n"
        f"{archetype_line}"
        f"{motif_line}"
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
    # The scheduler's timezone, not the runner's. The runner is UTC, and stamping a
    # post with a date the scheduler doesn't recognise as "today" is what would make
    # the cadence guard miscount its own work either side of midnight IST.
    from scheduler.planner import today_local
    iso_date = state.get("date") or today_local().isoformat()

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

    # Commit + push. Committer identity comes from the repo-local config set in
    # ensure_repo(); author defaults to the same, so no --author flag is needed.
    repo.git.add(mdx_rel, CONFIG.posts_registry_rel)
    repo.git.commit(
        "-m", f"blog: {state['title']}",
        "-m", f"Automated post. keyword={state.get('primary_keyword','')}",
    )
    log.info("pushing to %s", CONFIG.github_branch)
    repo.remotes.origin.push(CONFIG.github_branch)

    # Update KB so the next run sees this post.
    if kb is not None:
        kb.add(
            slug=slug, title=state["title"], description=state["description"],
            tags=state["tags"], body_text=state["body_mdx"],
        )

    _ping_indexnow(slug)

    log.info("published %s", mdx_rel)
    return mdx_rel


def _ping_indexnow(slug: str) -> None:
    """Tell Bing/IndexNow the new URL exists.

    Bing visibility is a documented input to AI answer engines (ChatGPT's retrieval
    leans on it), and IndexNow is the only push-notification channel available — there
    is no Google equivalent since the ping endpoints were retired. Google discovers the
    post through the sitemap on the next crawl.

    Best-effort by design: a failed ping must never fail a publish that already
    succeeded, so every error is swallowed with a warning.
    """
    key = CONFIG.indexnow_key
    if not key:
        log.info("indexnow: no INDEXNOW_KEY set — skipping ping")
        return
    try:
        import requests

        resp = requests.post(
            "https://api.indexnow.org/IndexNow",
            json={
                "host": "wizcodes.site",
                "key": key,
                "keyLocation": f"https://wizcodes.site/{key}.txt",
                "urlList": [f"https://wizcodes.site/blog/{slug}"],
            },
            timeout=15,
        )
        log.info("indexnow: HTTP %s for /blog/%s", resp.status_code, slug)
    except Exception as e:  # noqa: BLE001
        log.warning("indexnow ping failed (ignored): %s", e)
