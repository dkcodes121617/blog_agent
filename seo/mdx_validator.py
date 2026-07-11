"""Deterministic MDX + SEO contract validator.

Mirrors wizcodes_next/BLOG_FORMAT.md. This is the quality backbone: no post
reaches git unless it passes here. Two severities:
  - ERROR   → would break the Next build or violates a hard contract rule. Blocks.
  - WARNING → quality/SEO issue worth a rewrite, but wouldn't break the build.

The graph treats any ERROR as "loop back to the writer"; too many WARNINGs can
also trigger a rewrite. Returns a structured report the nodes can act on.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# The 8 components registered in src/mdx-components.tsx (the only allowed tags).
ALLOWED_COMPONENTS = {
    "KeyTakeaways", "Callout", "FlowDiagram", "CompareDiagram",
    "BarChart", "Figure", "FAQ", "BlogCTA",
}
# Internal route prefixes that actually exist on the site.
VALID_ROUTE_PREFIXES = ("/services/", "/work/", "/blog/", "/about", "/contact", "/open-source", "/")
VALID_SERVICE_SLUGS = {"web", "mobile", "ai"}


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_feedback(self) -> str:
        """Compact text the writer node can use to fix the draft."""
        lines = []
        if self.errors:
            lines.append("ERRORS (must fix):")
            lines += [f"  - {e}" for e in self.errors]
        if self.warnings:
            lines.append("WARNINGS (should fix):")
            lines += [f"  - {w}" for w in self.warnings]
        return "\n".join(lines) or "No issues."


def _prose_lines(mdx: str) -> list[tuple[int, str]]:
    """Lines that are prose (not inside a JSX component block or a code fence)."""
    out = []
    in_fence = False
    in_component = 0  # depth of an open multi-line component tag
    for i, line in enumerate(mdx.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        # Track multiline component props: a line opening "<Comp" without closing.
        if re.match(r"^<[A-Z]", stripped):
            if not (stripped.endswith("/>") or re.search(r"</[A-Z]\w+>\s*$", stripped)):
                in_component += 1
            continue
        if in_component:
            if "/>" in stripped or re.match(r"^\}?\s*/>", stripped) or stripped.endswith("/>"):
                in_component = max(0, in_component - 1)
            continue
        out.append((i, line))
    return out


def validate_mdx(mdx: str, known_slugs: set[str] | None = None) -> ValidationReport:
    r = ValidationReport()
    known_slugs = known_slugs or set()
    text = mdx.strip()

    if not text:
        r.errors.append("body is empty")
        return r

    # ── Hard contract: no H1, no frontmatter ──
    if text.startswith("---"):
        r.errors.append("body must not start with YAML frontmatter")
    if re.search(r"^#\s+\S", text, re.MULTILINE):
        r.errors.append("body must not contain an H1 ('# ...') — the route renders the title")

    # ── Required blocks ──
    if "<KeyTakeaways" not in text:
        r.errors.append("missing required <KeyTakeaways ... />")
    if "<FAQ" not in text:
        r.errors.append("missing required <FAQ ... />")
    if "<BlogCTA" not in text:
        r.errors.append("missing required <BlogCTA ... />")

    h2s = re.findall(r"^##\s+(.+)$", text, re.MULTILINE)
    if len(h2s) < 3:
        r.errors.append(f"needs 3-5 H2 sections, found {len(h2s)}")
    elif len(h2s) > 6:
        r.warnings.append(f"{len(h2s)} H2 sections — consider tightening to 3-5")

    # At least one illustration component (Flow/Compare/Bar/Figure).
    illustration = any(c in text for c in ("<FlowDiagram", "<CompareDiagram", "<BarChart", "<Figure"))
    if not illustration:
        r.errors.append("missing at least one illustration (FlowDiagram/CompareDiagram/BarChart/Figure)")

    # ── Unknown components ──
    for tag in set(re.findall(r"<([A-Z]\w+)", text)):
        if tag not in ALLOWED_COMPONENTS:
            r.errors.append(f"unknown component <{tag}> — not registered in mdx-components.tsx")

    # ── MDX gotchas: no markdown tables, no raw < or { in prose ──
    for lineno, line in _prose_lines(text):
        if re.match(r"^\s*\|.+\|\s*$", line):
            r.errors.append(f"line {lineno}: markdown table not allowed (use <CompareDiagram>)")
        # bare '<' not starting a component tag or closing tag
        for m in re.finditer(r"<(?![A-Za-z/])", line):
            r.errors.append(f"line {lineno}: raw '<' in prose (MDX treats it as JSX) — reword")
            break
        # bare '{' in prose (JSX expression) — allow it only inside component props (filtered out)
        if re.search(r"(?<![\w`])\{", line) and not line.strip().startswith("{"):
            r.warnings.append(f"line {lineno}: raw '{{' in prose can break MDX — reword")

    # ── Internal linking ──
    links = re.findall(r"\]\((/[^)]*)\)", text)
    internal = [l for l in links if l.startswith("/")]
    if len(internal) < 2:
        r.warnings.append(f"only {len(internal)} internal links — aim for 2-3+ (services/work/blog/contact)")
    for l in internal:
        # service path must use a real slug
        sm = re.match(r"^/services/([a-z]+)", l)
        if sm and sm.group(1) not in VALID_SERVICE_SLUGS:
            r.errors.append(f"invalid service link {l} (valid: /services/web|mobile|ai)")
        bm = re.match(r"^/blog/([a-z0-9-]+)$", l)
        if bm and known_slugs and bm.group(1) not in known_slugs:
            r.warnings.append(f"blog link {l} points to a slug not in the registry")

    # ── Lead paragraph (first non-empty block is prose, not a component/heading) ──
    first_block = text.split("\n\n", 1)[0].strip()
    if first_block.startswith("<") or first_block.startswith("#"):
        r.errors.append("first block must be a lead paragraph (prose), not a component or heading")
    elif len(first_block) < 120:
        r.warnings.append("lead paragraph looks short (aim for 2-3 sentences)")

    return r


def word_count(mdx: str) -> int:
    """Approx prose word count (strip components + code fences) → readingMinutes.

    Components carry short labels, not article prose, but their text still counts
    as reader-facing words, so we count the whole body minus code fences and JSX
    syntax tokens rather than deleting entire multi-line component blocks (which
    a greedy DOTALL strip would over-remove, collapsing the whole article).
    """
    text = re.sub(r"```.*?```", " ", mdx, flags=re.DOTALL)      # drop code fences
    text = re.sub(r"</?[A-Z]\w+", " ", text)                    # strip <Comp and </Comp
    text = re.sub(r"[<>{}\[\]/=]", " ", text)                   # strip JSX punctuation
    text = re.sub(r"\b\w+=", " ", text)                          # strip prop names (label=)
    return len(re.findall(r"[A-Za-z0-9']+", text))


def reading_minutes(mdx: str) -> int:
    return max(1, round(word_count(mdx) / 200))
