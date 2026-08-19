"""Validate site MDX files against the agent's own contract and editorial standard.

The agent checks every post it writes. Posts written or edited BY HAND — the back
catalogue being cleaned of prices and durations — went through nothing, which is how
the site ended up with three components the renderer cannot read and 509 price/duration
claims. This points the same validator at files on disk so a hand edit is held to the
rule the agent is held to.

    python check_posts.py                     # every post in the site repo
    python check_posts.py <slug> [<slug> ...]  # just these
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

from config import CONFIG
from seo import content_policy as cp
from seo.mdx_validator import validate_mdx, word_count

BLOG_DIR = CONFIG.local_site_dir / CONFIG.blog_content_rel


def known_slugs() -> set[str]:
    """Slugs the registry actually declares — what /blog/ links are checked against."""
    registry = (BLOG_DIR / "posts.ts").read_text(encoding="utf-8")
    return set(re.findall(r"slug:\s*'([^']+)'", registry))


def main(argv: list[str]) -> int:
    slugs = known_slugs()
    targets = (
        [BLOG_DIR / f"{s}.mdx" for s in argv]
        if argv else sorted(BLOG_DIR.glob("*.mdx"))
    )

    total_errors = 0
    for path in targets:
        if not path.exists():
            print(f"  MISSING  {path.name}")
            total_errors += 1
            continue
        text = io.open(path, encoding="utf-8").read()
        report = validate_mdx(text, known_slugs=slugs)
        policy = cp.body_violations(text)
        words = word_count(text)
        registered = path.stem in slugs

        flags = []
        if not registered:
            flags.append("NOT IN REGISTRY")
        if report.errors:
            flags.append(f"{len(report.errors)} error(s)")
        if policy:
            flags.append(f"{len(policy)} policy")
        status = "  ".join(flags) or "clean"

        print(f"\n{path.stem}  —  {words} words  —  {status}")
        for e in report.errors:
            print(f"      error: {e}")
        for v in policy[:6]:
            print(f"      policy: {v}")
        if len(policy) > 6:
            print(f"      policy: … {len(policy) - 6} more")
        total_errors += len(report.errors) + len(policy)

    print(f"\n{len(targets)} post(s) checked — {total_errors} problem(s)")
    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
