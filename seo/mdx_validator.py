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

# The 11 components registered in src/mdx-components.tsx (the only allowed tags).
ALLOWED_COMPONENTS = {
    "KeyTakeaways", "Callout", "FlowDiagram", "CompareDiagram",
    "BarChart", "Figure", "FAQ", "BlogCTA",
    # New visual components (Phase 2)
    "StatGrid", "Timeline", "DecisionTree",
    # Illustrations: "how it works" / "where it sits" visuals
    "ConceptDiagram", "QuadrantMap",
}
# ── Outbound citations ──
# The proxy has NO web search (probed and confirmed), so the writer cannot look
# anything up. Left free, it would invent plausible-looking URLs, and a hallucinated
# citation is far worse than none. External links are therefore restricted to an
# allowlist of stable, authoritative sources whose URLs do not churn.
CITATION_ALLOWLIST = (
    "developer.apple.com", "developer.android.com", "developers.google.com",
    "support.google.com", "web.dev", "schema.org", "www.w3.org",
    "flutter.dev", "docs.flutter.dev", "reactnative.dev", "react.dev",
    "nextjs.org", "nodejs.org", "docs.python.org", "fastapi.tiangolo.com",
    "postgresql.org", "www.postgresql.org", "owasp.org", "cheatsheetseries.owasp.org",
    "gdpr.eu", "www.iso.org", "pypi.org", "github.com",
    "docs.expo.dev", "stripe.com", "docs.stripe.com", "www.revenuecat.com",
)

MIN_VISUALS = 3          # was effectively 1; every published post shipped exactly 2
MIN_VISUAL_TYPES = 2     # stop 4-of-5 posts all being a BarChart
MIN_WORDS = 1400         # published posts ran 1,172-1,674; thin for cluster content

# Internal route prefixes that actually exist on the site.
VALID_ROUTE_PREFIXES = (
    "/services/", "/work/", "/blog/", "/open-source", "/about", "/contact",
    "/pricing", "/faq", "/testimonials", "/get-started", "/",
)
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

    # Section count scales with length. The old 3-5 range was set when posts ran ~700
    # words; at the current 1,400+ minimum that means ~300 words per section, which
    # reads as a wall. More sections also means more question-style headings, which is
    # what answer engines extract.
    h2s = re.findall(r"^##\s+(.+)$", text, re.MULTILINE)
    if len(h2s) < 4:
        r.errors.append(f"needs 4-8 H2 sections, found {len(h2s)}")
    elif len(h2s) > 8:
        r.warnings.append(f"{len(h2s)} H2 sections — consider merging a few")

    # At least one illustration component (Flow/Compare/Bar/Figure/StatGrid/Timeline/DecisionTree).
    visual_types = [c for c in (
        "FlowDiagram", "CompareDiagram", "BarChart", "Figure",
        "StatGrid", "Timeline", "DecisionTree", "ConceptDiagram", "QuadrantMap",
    ) if f"<{c}" in text]
    visual_count = sum(text.count(f"<{c}") for c in visual_types)
    if visual_count == 0:
        r.errors.append(
            "missing at least one illustration (FlowDiagram/CompareDiagram/BarChart/"
            "Figure/StatGrid/Timeline/DecisionTree/ConceptDiagram/QuadrantMap)")
    elif visual_count < MIN_VISUALS:
        r.warnings.append(
            f"only {visual_count} visual(s) — aim for {MIN_VISUALS}+ spread through the body")
    if len(visual_types) == 1 and visual_count > 1:
        r.warnings.append(
            f"every visual is a <{visual_types[0]}> — vary the type so posts don't all look alike")

    # ── Every visual must carry a caption ──
    # This is the highest-leverage image rule and it is not really about images.
    # Answer engines and LLM retrieval read TEXT: they never see the rendered chart.
    # A <BarChart> with a caption is a retrievable fact ("annual maintenance cost
    # breakdown"); the identical chart without one is invisible to every LLM, and to
    # Google Images, which ranks on surrounding context far more than on pixels.
    # The caption also becomes the <figcaption> and the exported filename.
    for comp in visual_types:
        for m in re.finditer(re.escape("<" + comp) + r"[\s/>]", text):
            tail = text[m.start(): m.start() + 600]
            if 'caption=' not in tail.split("/>")[0]:
                r.errors.append(
                    f"<{comp}> has no caption — captions are what search and answer "
                    f"engines actually read, so every visual needs one")
                break

    # ── Length ──
    words = word_count(text)
    if words < MIN_WORDS:
        r.warnings.append(f"{words} words — thin for a cluster post, aim for {MIN_WORDS}+")

    # ── Outbound citations must be on the allowlist (no web search = no invented URLs) ──
    for url in re.findall(r"\]\((https?://[^)]+)\)", text):
        host = re.sub(r"^https?://", "", url).split("/")[0].lower()
        if not any(host == a or host.endswith("." + a) for a in CITATION_ALLOWLIST):
            r.errors.append(
                f"external link to {host} is not on the citation allowlist — the writer "
                f"cannot verify sources, so only well-known stable references are allowed")

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
            # ERROR, not a warning. A /blog/ link to a slug that isn't in the registry
            # is a guaranteed 404 the moment the post goes live, and warnings don't
            # loop back to the writer — so this used to ship silently. Verified safe
            # to tighten: all 16 currently published posts pass this check.
            #
            # The `known_slugs` guard matters: it is empty on code paths that don't
            # load the KB, and without it every link would be flagged.
            r.errors.append(
                f"blog link {l} points to a slug not in the registry — link an existing "
                f"post or drop the link"
            )

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
