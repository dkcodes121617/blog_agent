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
from datetime import date
from pathlib import Path

from config import CONFIG


@dataclass
class ProjectFact:
    id: str = ""
    name: str = ""
    category: str = ""
    industry: str = ""
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
    # Visual component types used by the most recent posts, so the writer can rotate
    # away from them. Not stored in the registry, so it is read back out of the MDX.
    recent_visuals: list[str] = field(default_factory=list)
    playbook_excerpt: str = ""

    # ── prompt-ready rendering ──
    def to_prompt_block(self, max_playbook_chars: int = 12000) -> str:
        """A compact, human-readable block to inject as grounding facts."""
        lines: list[str] = []
        # The model has no reliable sense of "now" and will otherwise default to its
        # training-era year — which is exactly how two published posts ended up titled
        # "...in 2025" while being published in July 2026. State the date explicitly and
        # tell the writer which year any year-reference must use.
        today = date.today()
        lines.append(f"TODAY'S DATE: {today.isoformat()}")
        lines.append(
            f"CURRENT YEAR: {today.year}. Any year mentioned in a title, heading, slug, "
            f"or statement about current prices/trends must be {today.year}. Never write "
            f"{today.year - 1} as though it were the current year."
        )
        lines.append("")
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
            ind = f" · {p.industry}" if p.industry else ""
            lines.append(f"  - {p.name} [{p.category}{ind}]{loc}{path}{tag}: {p.description}")
            if p.tech:
                lines.append(f"      tech: {', '.join(p.tech)}")
        lines.append("")
        if self.open_source:
            lines.append("OPEN SOURCE (real, on /open-source):")
            for o in self.open_source:
                dl = f" — {o['downloads']} downloads" if o.get("downloads") else ""
                lines.append(f"  - {o['name']}{dl}: {o.get('description','')}")
            lines.append("")
        # Conversion + reference pages the writer can link to. Without this the model
        # only knows about /services, /work and /blog, so posts never link to the pages
        # that actually convert.
        lines.append("OTHER PAGES YOU MAY LINK TO (these exist; do not invent others):")
        lines.append("  - /get-started: the free-prototype offer + request form (best CTA target)")
        lines.append("  - /pricing: how fixed-scope pricing works vs hourly/retainer/freelancer")
        lines.append("  - /faq: NDAs, ownership, invoicing currency, support after handover")
        lines.append("  - /testimonials: all real client messages")
        lines.append("  - /contact, /about, /work, /blog, /open-source")
        lines.append("")
        lines.append("EXISTING BLOG POSTS (do NOT duplicate these topics or slugs):")
        for post in self.existing_posts:
            lines.append(f"  - /blog/{post['slug']}: {post['title']}")
        if self.playbook_excerpt:
            lines.append("")
            lines.append("BRAND/SEO PLAYBOOK (voice, USPs, guardrails — obey these):")
            lines.append(_playbook_sections(self.playbook_excerpt, max_playbook_chars))
        return "\n".join(lines)


# ── playbook extraction ──
# details.md is ~57k chars of full project reference. Naively truncating it to the
# first N characters delivered tone-of-voice and the USPs but cut off everything from
# roughly section 6 onward — including the integrity rules and the per-project
# confidentiality boundaries, which sit near char 47,000. The writer was therefore
# never shown the rules it most needs to obey.
#
# So pull the sections a writer actually needs, by heading, in priority order, and
# spend the character budget on those instead of on whatever happens to come first.
#
# Order is priority order, NOT document order: the budget is spent top-down and the
# rules that must never be broken (confidentiality, integrity, naming conventions)
# come first. Voice and positioning matter, but a post in slightly-off tone is a
# quality problem, whereas a post that leaks a client's internals is a trust problem.
_PLAYBOOK_SECTIONS = [
    "### 20.1 Work / portfolio (`src/data/projects.ts`)",   # integrity + confidentiality
    "## 21. Conventions & guardrails",
    "### 2.4 Tone of voice",
    "### 2.3 Brand positioning",
    "### 2.2 Vision, mission, values",
    "## 3. Unique Selling Propositions (USPs)",
    "### 2.6 What differentiates WizCodes from traditional software companies",
    "## 4. The Free Prototype offering",
    "### 2.5 Target audience",
]


def _playbook_sections(playbook: str, budget: int) -> str:
    """Return the writer-relevant sections of details.md, within `budget` chars.

    Falls back to plain truncation if the headings can't be found, so a reformatted
    details.md degrades rather than silently emitting nothing.
    """
    out: list[str] = []
    used = 0
    for heading in _PLAYBOOK_SECTIONS:
        start = playbook.find(heading)
        if start == -1:
            continue
        # Run to the next heading of the same or higher level.
        after = playbook[start + len(heading):]
        ends = [after.find(m) for m in ("\n## ", "\n### ") if after.find(m) != -1]
        end = min(ends) if ends else len(after)
        block = (heading + after[:end]).strip()
        if used + len(block) > budget:
            break
        out.append(block)
        used += len(block)
    return "\n\n".join(out) if out else playbook[:budget].strip()


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
        pf.industry = _f(text, "industry")
        pf.client = _f(text, "client")
        pf.client_country = _f(text, "clientCountry")
        pf.description = _f(text, "description")
        # The site derives `slug: p.slug ?? p.id` at runtime, so most entries carry no
        # literal slug in the source text. Mirror that fallback here — otherwise this
        # regex only finds the six hardcoded slugs and the writer believes the other
        # twenty project pages do not exist, so it can never link to them.
        pf.slug = _f(text, "slug") or pf.id
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


_VISUAL_COMPONENTS = (
    "FlowDiagram", "CompareDiagram", "BarChart", "Figure",
    "StatGrid", "Timeline", "DecisionTree", "ConceptDiagram", "QuadrantMap",
)


def _recent_visuals(site_dir: Path, posts: list[dict], lookback: int = 4) -> list[str]:
    """Visual component types used by the newest `lookback` posts.

    The archetype map alone put a BarChart in 4 of 5 published posts and never once
    used FlowDiagram or Figure. Feeding recent usage back lets the writer rotate.
    """
    used: list[str] = []
    for post in posts[:lookback]:
        path = site_dir / CONFIG.blog_content_rel / f"{post['slug']}.mdx"
        if not path.exists():
            continue
        body = path.read_text(encoding="utf-8", errors="replace")
        for comp in _VISUAL_COMPONENTS:
            if f"<{comp}" in body and comp not in used:
                used.append(comp)
    return used


def build_snapshot(site_dir: Path | str | None = None) -> FactsSnapshot:
    # Coerce to Path — callers may pass a str (e.g. GitPython's working_tree_dir).
    site_dir = Path(site_dir) if site_dir else _resolve_site_dir()
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
        recent_visuals=_recent_visuals(site_dir, _extract_posts(posts_ts)),
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
