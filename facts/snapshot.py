"""Builds the 'real WizCodes facts' grounding context for the writer.

The #1 hallucination risk (confirmed in proxy testing) is the model inventing
numbers, client names, and sectors. The defence is to feed every writing prompt a
compact, TRUE snapshot drawn straight from the site repo, and instruct the model
to ground claims in it. This module reads:

  - src/config/site.ts        → brand, services, contact, real socials
  - src/data/projects.ts      → real project names / clients / countries / tech / slugs
  - src/data/openSource.ts    → real OSS projects (ClarivueXAI 1000+ PyPI, etc.)
  - src/content/blog/posts.ts  → existing post titles/descriptions (topic coverage)
  - details.md (if present)    → the brand/SEO playbook (voice, USPs, guardrails)

It never executes TS — it extracts durable fields with tolerant regex, so a minor
formatting change in the source won't crash the agent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from config import CONFIG


@dataclass
class ProjectFact:
    id: str = ""
    name: str = ""
    category: str = ""
    client: str = ""
    client_country: str = ""
    description: str = ""
    tech: list[str] = field(default_factory=list)
    slug: str = ""
    hide_status: bool = False


@dataclass
class FactsSnapshot:
    brand_name: str = "WizCodes"
    tagline: str = ""
    url: str = "https://wizcodes.site"
    services: list[dict] = field(default_factory=list)      # {name, slug, oneLineDescription}
    projects: list[ProjectFact] = field(default_factory=list)
    open_source: list[dict] = field(default_factory=list)   # {name, description, downloads}
    existing_posts: list[dict] = field(default_factory=list)  # {slug, title, description, tags}
    playbook_excerpt: str = ""

    # ── prompt-ready rendering ──
    def to_prompt_block(self, max_playbook_chars: int = 6000) -> str:
        """A compact, human-readable block to inject as grounding facts."""
        lines: list[str] = []
        lines.append(f"BRAND: {self.brand_name} — {self.tagline}")
        lines.append(f"SITE: {self.url}")
        lines.append("")
        lines.append("SERVICES (canonical names + slugs — only these labels/paths exist):")
        for s in self.services:
            lines.append(f"  - {s['name']} (/services/{s['slug']}): {s.get('oneLineDescription','')}")
        lines.append("")
        lines.append("REAL PROJECTS you may reference by name (do NOT invent others; do NOT")
        lines.append("claim live/store status for entries marked [no-status]):")
        for p in self.projects:
            tag = " [no-status]" if p.hide_status else ""
            loc = f" — {p.client}, {p.client_country}" if p.client else ""
            path = f" (/work/{p.slug})" if p.slug else ""
            lines.append(f"  - {p.name} [{p.category}]{loc}{path}{tag}: {p.description}")
            if p.tech:
                lines.append(f"      tech: {', '.join(p.tech)}")
        lines.append("")
        if self.open_source:
            lines.append("OPEN SOURCE (real, on /open-source):")
            for o in self.open_source:
                dl = f" — {o['downloads']} downloads" if o.get("downloads") else ""
                lines.append(f"  - {o['name']}{dl}: {o.get('description','')}")
            lines.append("")
        lines.append("EXISTING BLOG POSTS (do NOT duplicate these topics or slugs):")
        for post in self.existing_posts:
            lines.append(f"  - /blog/{post['slug']}: {post['title']}")
        if self.playbook_excerpt:
            lines.append("")
            lines.append("BRAND/SEO PLAYBOOK (voice, USPs, guardrails — obey these):")
            lines.append(self.playbook_excerpt[:max_playbook_chars].strip())
        return "\n".join(lines)


# ── extraction helpers ──
def _read(site_dir: Path, rel: str) -> str:
    p = site_dir / rel
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def _extract_services(site_ts: str) -> list[dict]:
    services = []
    # Match { name: '...', slug: '...', oneLineDescription: '...' } blocks.
    for m in re.finditer(
        r"\{\s*name:\s*'([^']+)',\s*slug:\s*'([^']+)',\s*oneLineDescription:\s*\n?\s*'([^']+)'",
        site_ts,
    ):
        services.append({"name": m.group(1), "slug": m.group(2), "oneLineDescription": m.group(3)})
    return services


def _extract_scalar(site_ts: str, key: str) -> str:
    m = re.search(rf"{key}:\s*'([^']*)'", site_ts)
    return m.group(1) if m else ""


def _extract_projects(projects_ts: str) -> list[ProjectFact]:
    """Split the file into object literals and pull durable fields from each."""
    projects: list[ProjectFact] = []
    # Grab each { ... } that contains an id: '...' and a name: '...'.
    for block in re.finditer(r"\{[^{}]*?id:\s*'[^']+'[^{}]*?\}", projects_ts, re.DOTALL):
        text = block.group(0)
        pf = ProjectFact()
        pf.id = _f(text, "id")
        pf.name = _f(text, "name")
        pf.category = _f(text, "category")
        pf.client = _f(text, "client")
        pf.client_country = _f(text, "clientCountry")
        pf.description = _f(text, "description")
        pf.slug = _f(text, "slug")
        pf.hide_status = "hideStatus: true" in text
        tech_m = re.search(r"tech:\s*\[([^\]]*)\]", text)
        if tech_m:
            pf.tech = [t.strip().strip("'\"") for t in tech_m.group(1).split(",") if t.strip()]
        if pf.name:
            projects.append(pf)
    return projects


def _extract_open_source(oss_ts: str) -> list[dict]:
    out = []
    for block in re.finditer(r"\{[^{}]*?name:\s*'[^']+'[^{}]*?\}", oss_ts, re.DOTALL):
        text = block.group(0)
        name = _f(text, "name")
        if not name:
            continue
        out.append(
            {"name": name, "description": _f(text, "description"), "downloads": _f(text, "downloads")}
        )
    return out


def _f(text: str, key: str) -> str:
    m = re.search(rf"{key}:\s*'([^']*)'", text)
    return m.group(1) if m else ""


def _extract_posts(posts_ts: str) -> list[dict]:
    out = []
    for block in re.finditer(r"\{[^{}]*?slug:\s*'[^']+'[^{}]*?\}", posts_ts, re.DOTALL):
        text = block.group(0)
        slug = _f(text, "slug")
        if not slug:
            continue
        tags_m = re.search(r"tags:\s*\[([^\]]*)\]", text)
        tags = (
            [t.strip().strip("'\"") for t in tags_m.group(1).split(",") if t.strip()]
            if tags_m
            else []
        )
        out.append(
            {
                "slug": slug,
                "title": _f(text, "title"),
                "description": _multiline_field(text, "description"),
                "tags": tags,
            }
        )
    return out


def _multiline_field(text: str, key: str) -> str:
    """description often spans lines: description:\n  '...'."""
    m = re.search(rf"{key}:\s*\n?\s*'([^']*)'", text)
    return m.group(1) if m else _f(text, key)


def build_snapshot(site_dir: Path | None = None) -> FactsSnapshot:
    site_dir = site_dir or _resolve_site_dir()
    site_ts = _read(site_dir, CONFIG.site_config_rel)
    projects_ts = _read(site_dir, f"{CONFIG.data_dir_rel}/projects.ts")
    oss_ts = _read(site_dir, f"{CONFIG.data_dir_rel}/openSource.ts")
    posts_ts = _read(site_dir, CONFIG.posts_registry_rel)
    playbook = _read(site_dir, "details.md")

    snap = FactsSnapshot(
        brand_name=_extract_scalar(site_ts, "brandName") or "WizCodes",
        tagline=_extract_scalar(site_ts, "tagline"),
        url=_extract_scalar(site_ts, "url") or "https://wizcodes.site",
        services=_extract_services(site_ts),
        projects=_extract_projects(projects_ts),
        open_source=_extract_open_source(oss_ts),
        existing_posts=_extract_posts(posts_ts),
        playbook_excerpt=playbook,
    )
    return snap


def _resolve_site_dir() -> Path:
    """Prefer the cloned publish repo; fall back to the local sibling folder."""
    if (CONFIG.site_repo_dir / CONFIG.posts_registry_rel).exists():
        return CONFIG.site_repo_dir
    return CONFIG.local_site_dir


if __name__ == "__main__":
    snap = build_snapshot()
    print(f"services={len(snap.services)} projects={len(snap.projects)} "
          f"oss={len(snap.open_source)} posts={len(snap.existing_posts)} "
          f"playbook_chars={len(snap.playbook_excerpt)}")
    print("=" * 70)
    print(snap.to_prompt_block()[:2500])
