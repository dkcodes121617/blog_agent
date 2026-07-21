"""Prompt builders for every LLM node.

Design rules learned from testing the ClaudeStore proxy:
  - Phrase everything as a normal professional content task. NEVER use
    override/compliance language ("reply with exactly", "never break character",
    "obey this contract") — the proxy's injection guard refuses those.
  - Put the real WizCodes facts in the prompt and tell the model to ground claims
    in them, which is what stops hallucinated numbers/clients.
  - Ask for JSON where we need structure; ask for MDX where we need the body.

Each function returns (system, user).
"""
from __future__ import annotations

# ── Archetypes ──────────────────────────────────────────────────────────────
# Four content archetypes, each with its own H2 scaffold, required illustration
# type, and intro hook style. The archetype is picked at topic-selection time and
# threads through every subsequent node.

ARCHETYPES = ("cost_breakdown", "vs_comparison", "decision_framework", "mistake_guide")

# ── Archetype rotation ──────────────────────────────────────────────────────
# The archetype is chosen at generation time but is NOT stored in the site's posts.ts
# registry (its schema is slug/title/description/date/tags/readingMinutes). On a
# stateless runner there is therefore no record of what shape recent posts took —
# which is how four consecutive `cost_breakdown` posts shipped, each following the
# same five-heading scaffold below.
#
# Rather than change the site's registry schema, the archetype is inferred back out
# of the title. Titles are written to fit the archetype, so the signal is strong.
# The result is used to remove over-used archetypes from the menu offered to the
# topic strategist, so the blog cannot drift into being all one shape.

# Ordered most- to least-specific: first match wins, so "Should I build X or Y?"
# classifies as a decision framework rather than a comparison.
_ARCHETYPE_SIGNATURES: list[tuple[str, "re.Pattern[str]"]] = []


def _build_signatures():
    import re
    return [
        ("mistake_guide", re.compile(
            r"\bmistakes?\b|\bpitfalls?\b|\bavoid\b|\bwrong\b|\bdon'?t\b|\bstop\b", re.I)),
        ("decision_framework", re.compile(
            r"\bshould (i|you|we)\b|\bhow to (choose|decide|pick)\b|\bwhich\b"
            r"|\bwhen to\b|\bframework\b|\bchecklist\b|\bdecisions?\b|\bcriteria\b", re.I)),
        ("vs_comparison", re.compile(
            r"\bvs\.?\b|\bversus\b|\bcompared? (to|with)\b|\balternatives?\b"
            r"|\btrade-?offs?\b", re.I)),
        ("cost_breakdown", re.compile(
            r"\bcosts?\b|\bpric(e|es|ing)\b|\bbudget\b|\bhow much\b|\broi\b"
            r"|\bfees?\b|\$\d", re.I)),
    ]


def infer_archetype(title: str, description: str = "") -> str:
    """Best-effort archetype for an already-published post, from its title.

    The title is checked first because it is the most deliberate signal; the
    description is only a fallback. Defaults to decision_framework, the most common
    shape for a buyer-facing blog, when nothing matches.
    """
    global _ARCHETYPE_SIGNATURES
    if not _ARCHETYPE_SIGNATURES:
        _ARCHETYPE_SIGNATURES = _build_signatures()
    for text in (title or "", description or ""):
        for name, pattern in _ARCHETYPE_SIGNATURES:
            if pattern.search(text):
                return name
    return "decision_framework"


def blocked_archetypes(recent_posts: list[dict], *, lookback: int = 4) -> list[str]:
    """Archetypes the next post may NOT use.

    `recent_posts` is the registry list, newest first — the publisher inserts at the
    head of the array, so registry order is publication order.

    Two rules, both needed:
      - never repeat the immediately previous post's archetype (no back-to-back);
      - never use an archetype already accounting for 2+ of the last `lookback` posts
        (stops a 3-in-4 run even when it alternates).

    Guard: never block everything. If the rules would leave no valid choice, only the
    back-to-back rule is kept, so at least three options always remain.
    """
    if not recent_posts:
        return []

    window = recent_posts[:lookback]
    inferred = [
        infer_archetype(p.get("title", ""), p.get("description", "")) for p in window
    ]

    blocked = {inferred[0]}
    for name in ARCHETYPES:
        if inferred.count(name) >= 2:
            blocked.add(name)

    if len(blocked) >= len(ARCHETYPES):
        blocked = {inferred[0]}

    return sorted(blocked)

# Archetype → natural H2 scaffold (template strings, not final titles).
_H2_PATTERNS: dict[str, list[str]] = {
    "cost_breakdown": [
        "What actually drives the cost of [topic]",
        "Where most budgets get wasted",
        "A realistic cost breakdown",
        "How to get more for less without cutting corners",
        "When to invest more — and when not to",
    ],
    "vs_comparison": [
        "What each option actually offers",
        "Where [option A] wins",
        "Where [option B] wins",
        "The real decision criteria most guides skip",
        "How to pick the right one for your situation",
    ],
    "decision_framework": [
        "Why this decision is harder than it looks",
        "The key variables that change the answer",
        "A practical decision framework",
        "Common mistakes that lead to the wrong choice",
        "How to validate your decision before committing",
    ],
    "mistake_guide": [
        "The most expensive mistake founders make with [topic]",
        "Why conventional advice on [topic] backfires",
        "What to do instead",
        "How to spot these mistakes before they cost you",
        "A smarter approach from the ground up",
    ],
}

# Archetype → preferred primary illustration type.
_PRIMARY_ILLUSTRATION: dict[str, str] = {
    "cost_breakdown": "BarChart",
    "vs_comparison": "CompareDiagram",
    "decision_framework": "DecisionTree",
    "mistake_guide": "CompareDiagram",
}

# Archetype → preferred secondary illustration type.
_SECONDARY_ILLUSTRATION: dict[str, str] = {
    "cost_breakdown": "StatGrid",
    "vs_comparison": "BarChart",
    "decision_framework": "Timeline",
    "mistake_guide": "FlowDiagram",
}

# Archetype → intro hook guidance (how the lead paragraph should open).
_INTRO_HOOK: dict[str, str] = {
    # AEO: retrieval weights the opening heavily and wants the ANSWER there, not a
    # story. Each hook is capped at one sentence, followed immediately by a direct
    # answer. The narrative still happens, further down, where it costs nothing.
    # These previously all asked for a scene-setting anecdote, which spent the whole
    # extractable window on throat-clearing.
    "cost_breakdown": (
        "ONE short sentence naming the situation (a surprise bill, a wildly varying "
        "quote). Then immediately answer the question the title asks, in plain terms, "
        "with the real range or the real determining factor."
    ),
    "vs_comparison": (
        "ONE short sentence acknowledging the choice. Then immediately state which "
        "option suits which situation - the actual answer - before any elaboration."
    ),
    "decision_framework": (
        "ONE short sentence on what goes wrong when this decision is rushed. Then "
        "immediately give the short version of the framework: the one or two "
        "variables that actually decide it."
    ),
    "mistake_guide": (
        "ONE short sentence naming the mistake plainly. Then immediately say what to "
        "do instead, before explaining why."
    ),
}

# The studio persona reused across writing nodes. Warm, senior, honest — matches
# BLOG_FORMAT.md voice rules.
STUDIO_PERSONA = (
    "You are the founding engineer at WizCodes, a small remote-first software "
    "studio (web, mobile, and AI). You write the studio's blog. Your voice is "
    "first-person plural ('we'), plain-spoken, senior, and honest — never hyped, "
    "never generic. You ground everything in the studio's real work and never "
    "invent numbers, clients, or statistics."
)


# ── Topic focus ─────────────────────────────────────────────────────────────
# The convergence problem was never the uniqueness gate: four "cost of X" posts
# scored at most 0.693 pairwise against an 0.82 threshold, so they passed easily.
# Cosine similarity finds near-duplicates, not thematic sameness.
#
# The actual cause is that the strategist was asked to "propose ONE topic" from a
# fixed prior (a software studio's services), so it re-derived the same highest-
# probability answer every run: how much does X cost.
#
# The fix is to stop asking an open question. Each run is assigned a FOCUS drawn
# from the real project corpus — an industry, a delivery category, or a technology —
# picked as whichever axis the blog has covered least. The model then chooses a topic
# *within* that focus. Convergence becomes structurally impossible rather than
# discouraged, and every topic stays anchored to work actually delivered.

# Axis values worth writing about. Countries are deliberately excluded: they are
# evidence, not subject matter.
_FOCUS_AXES = ("industry", "category", "tech")

_CATEGORY_LABEL = {
    "mobile": "mobile app development",
    "web": "web and SaaS development",
    "ai": "AI automation and agents",
    "game": "game development",
}

# Technology is only a valid focus when a BUYER would actually weigh it up. A founder
# choosing between Flutter and React Native is a real commercial decision; "building
# with Dart" or "Flame Engine" is an implementation detail and would produce exactly
# the developer tutorial the strategist is told not to write. So the tech axis runs
# off an allowlist rather than off whatever appears in the tech arrays.
_BUYER_FACING_TECH = {
    "Flutter": "choosing Flutter for a cross-platform app",
    "React Native": "choosing React Native for a cross-platform app",
    "Expo": "shipping and updating apps with Expo",
    "Next.js": "choosing Next.js for a web product",
    "React": "choosing React for a web product",
    "FastAPI": "choosing a Python backend for a product",
    "Firebase": "using Firebase as a product backend",
    "Supabase": "using Supabase as a product backend",
    "PostgreSQL": "choosing a database for a growing product",
    "LangGraph": "orchestrating multi-step AI agents",
    "LLM": "putting an LLM into a production product",
    "OpenAI": "choosing between LLM providers for a product",
    "RevenueCat": "handling subscriptions and in-app purchases",
    "Stripe": "taking payments in a product",
    "WhatsApp Business API": "automating customer conversations on WhatsApp",
}

# Anything appearing on only one project is too thin to anchor an article.
_MIN_TECH_PROJECTS = 2

# How many of the least-covered options to rotate between. Small enough that the
# strongest opportunities keep coming up, large enough that consecutive runs differ.
_ROTATION_POOL = 8


def _coverage(value: str, posts: list[dict]) -> int:
    """How many existing posts already touch this axis value."""
    needle = value.lower()
    n = 0
    for p in posts:
        hay = f"{p.get('title','')} {p.get('description','')} {' '.join(p.get('tags',[]))}".lower()
        if needle in hay:
            n += 1
    return n


def build_focus_options(projects: list, posts: list[dict]) -> list[dict]:
    """Every candidate focus, with its anchor projects and current coverage.

    `projects` are ProjectFact-like objects (name / category / industry / tech /
    hide_status / slug). Returned newest-opportunity-first: least covered wins.
    """
    from collections import defaultdict

    by_industry: dict[str, list] = defaultdict(list)
    by_category: dict[str, list] = defaultdict(list)
    by_tech: dict[str, list] = defaultdict(list)

    for p in projects:
        if getattr(p, "industry", ""):
            by_industry[p.industry].append(p)
        if getattr(p, "category", ""):
            by_category[p.category].append(p)
        for tech in getattr(p, "tech", []) or []:
            by_tech[tech].append(p)

    options: list[dict] = []
    for industry, anchors in by_industry.items():
        options.append({
            "axis": "industry", "value": industry,
            "brief": f"software for the {industry} sector",
            "anchors": anchors,
        })
    for category, anchors in by_category.items():
        if category in _CATEGORY_LABEL:
            options.append({
                "axis": "category", "value": category,
                "brief": _CATEGORY_LABEL[category],
                "anchors": anchors,
            })
    for tech, anchors in by_tech.items():
        if len(anchors) >= _MIN_TECH_PROJECTS and tech in _BUYER_FACING_TECH:
            options.append({
                "axis": "tech", "value": tech,
                "brief": _BUYER_FACING_TECH[tech],
                "anchors": anchors,
            })

    for o in options:
        o["coverage"] = _coverage(o["value"], posts)
        o["anchor_count"] = len(o["anchors"])

    # Least-covered first; break ties toward the axis with more real projects behind
    # it, then alphabetically so the choice is deterministic on a stateless runner.
    options.sort(key=lambda o: (o["coverage"], -o["anchor_count"], o["value"]))
    return options


def pick_focus(projects: list, posts: list[dict], *, rotation_seed: int = 0) -> dict | None:
    """Choose this run's topic focus: the least-covered axis with real work behind it.

    `rotation_seed` (a date ordinal) rotates between the few least-covered options so
    consecutive runs don't all land on the same one before any of them is published.
    """
    options = build_focus_options(projects, posts)
    if not options:
        return None
    # Rotate within the least-covered tier, but only across the strongest few — the
    # tier is already sorted by coverage then by anchor count, so slicing keeps the
    # best-grounded opportunities in play instead of drifting to thin ones.
    floor = options[0]["coverage"]
    tier = [o for o in options if o["coverage"] == floor] or options
    pool = tier[:_ROTATION_POOL]
    return pool[rotation_seed % len(pool)]


def describe_focus(focus: dict) -> str:
    """The focus, rendered for the prompt, with its real anchor projects."""
    if not focus:
        return ""
    lines = [f"ASSIGNED FOCUS for this post: {focus['brief']}."]
    lines.append("Real WizCodes work you can draw on for this focus (reference by name,")
    lines.append("and never claim live/shipped status for entries marked [no-status]):")
    for a in focus["anchors"][:6]:
        tag = " [no-status]" if getattr(a, "hide_status", False) else ""
        path = f" (/work/{a.slug})" if getattr(a, "slug", "") else ""
        lines.append(f"  - {a.name}{path}{tag}: {a.description}")
    return "\n".join(lines)


# ── Node: topic strategist ──────────────────────────────────────────────────
def topic_prompt(
    facts_block: str,
    avoid_recent: list[str],
    blocked: list[str] | None = None,
    focus: dict | None = None,
) -> tuple[str, str]:
    system = (
        "You are a lead generation strategist for a B2B software studio. Your job "
        "is to pick blog topics that attract business owners, startup founders, and "
        "non-technical decision-makers who are evaluating software development options. "
        "You write for buyers, not builders."
    )
    avoid = "\n".join(f"  - {s}" for s in avoid_recent) or "  (none yet)"

    # Over-used archetypes are removed from the menu entirely rather than merely
    # discouraged — the model reliably picks whatever is listed, and a "please vary"
    # instruction did not stop four cost breakdowns in a row.
    focus_block = (describe_focus(focus) + "\n\n") if focus else ""
    blocked = blocked or []
    available = [a for a in ARCHETYPES if a not in blocked] or list(ARCHETYPES)
    archetype_list = "\n".join(f"  - {a}" for a in available)
    if blocked:
        archetype_list += (
            "\n\nRecent posts have already used "
            + ", ".join(blocked)
            + ", so those are not available this time. Pick from the list above."
        )
    user = f"""Here are the studio's real facts and its existing blog coverage:

{facts_block}

Recently covered topics to avoid repeating:
{avoid}

{focus_block}Propose ONE blog topic for a BUSINESS OWNER or STARTUP FOUNDER who is evaluating or
buying software services — someone who controls a budget but is not a developer.

The topic must sit inside the assigned focus above. Ground it in the real projects
listed for that focus — that is what makes the post worth reading rather than generic.

Target a keyword with commercial or navigational search intent. Cost and pricing are
only ONE option among many, and recent posts have leaned on them heavily: prefer
build-vs-buy decisions, vendor and platform comparisons, industry-specific guides,
common business mistakes, evaluation criteria, or decision frameworks.

Do NOT propose:
  - Developer tutorials ("how to build X", "implementing Y", "a guide to Z library")
  - Technical deep-dives aimed at engineers or developers
  - "How to code", "how to set up", "step-by-step technical implementation" posts
  - Posts where the primary audience would be a software developer, not a business buyer

Required archetype — pick exactly one:
{archetype_list}

  cost_breakdown   → cost/ROI/pricing breakdowns, budget guides, "what does X really cost"
  vs_comparison    → X vs Y comparisons, vendor/tool evaluations, tradeoff analyses
  decision_framework → decision guides, "how to choose", evaluation criteria, checklists
  mistake_guide    → common mistakes, pitfalls, "don't make this error", cautionary guides

Reply as JSON:
{{"primary_keyword": string (the exact phrase a buyer would search),
  "angle": string (the specific thesis/take, one sentence, written for a business buyer),
  "audience": string (be specific: e.g. "early-stage startup founder considering building an app"),
  "archetype": string (one of the four above),
  "intent_type": "commercial" | "informational" | "navigational",
  "rationale": string (why this keyword has buyer intent + why WizCodes can write it credibly)}}"""
    return system, user


# ── Node: SEO outliner ──────────────────────────────────────────────────────
def outline_prompt(
    facts_block: str,
    primary_keyword: str,
    angle: str,
    audience: str,
    archetype: str,
    related_slugs: list[str],
) -> tuple[str, str]:
    system = STUDIO_PERSONA + (
        " Right now you are outlining a post before writing it, thinking about "
        "search intent, business-buyer framing, and internal linking."
    )
    related = ", ".join(f"/blog/{s}" for s in related_slugs) or "(none especially close)"

    # Build archetype-specific guidance block.
    h2_pattern = _H2_PATTERNS.get(archetype, _H2_PATTERNS["decision_framework"])
    h2_guidance = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(h2_pattern))
    primary_ill = _PRIMARY_ILLUSTRATION.get(archetype, "CompareDiagram")
    secondary_ill = _SECONDARY_ILLUSTRATION.get(archetype, "BarChart")
    hook_guidance = _INTRO_HOOK.get(archetype, "Open with a concrete business problem.")

    user = f"""Studio facts (ground truth — reference real projects, never invent):

{facts_block}

Plan a blog post for a BUSINESS AUDIENCE (founders, operators, decision-makers — not developers).
Primary keyword: "{primary_keyword}"
Angle: {angle}
Audience: {audience}
Archetype: {archetype}

H2 pattern to adapt for this archetype (adapt the placeholders to the specific topic):
{h2_guidance}

Intro hook style: {hook_guidance}

Most topically-related existing posts (link to 1-2 for topic clustering):
{related}

Plan THREE to FOUR illustrations for this post, each a DIFFERENT component type. Every
one is exported as a standalone SVG that Google Images and AI systems index on its own,
so each is a separate entry point to the article - a post with one chart has one visual
entry point, a post with four varied visuals has four.

Vary the type deliberately: a post that is all BarCharts looks like every other post
and adds no new visual surface. Match the type to the content instead - BarChart for
magnitudes, StatGrid for headline numbers, CompareDiagram for this-vs-that, FlowDiagram
for a process, Timeline for something that unfolds over time, DecisionTree for a
branching choice.

Give every illustration a specific, descriptive caption. The caption becomes the
image's filename, its <title>, and the text answer engines actually read - "Annual
maintenance cost breakdown" earns search traffic, "Chart 1" earns none.


Produce a JSON plan:
{{
  "working_title": string (natural, ~50-60 chars, includes the primary keyword, sounds like a business article — no "how to build X"),
  "h2s": [3-5 strings adapted from the archetype pattern above, keyword-rich but natural and buyer-facing],
  "lsi_keywords": [5-8 semantic variants to weave in naturally — business terms, not technical jargon],
  "internal_links": [3-4 objects {{"path": "/services/... or /work/... or /blog/... or /contact", "anchor": string}}],
  "primary_illustration": {{"type": "{primary_ill}", "purpose": string, "data_hint": string (what data/content to put in it)}},
  "secondary_illustration": {{"type": "{secondary_ill}", "purpose": string, "data_hint": string}},
  "extra_illustrations": [
    {{"type": one of BarChart|CompareDiagram|StatGrid|FlowDiagram|Timeline|DecisionTree,
      "purpose": string, "data_hint": string}}
  ],
  "real_projects_to_cite": [names from the facts that genuinely fit this topic]
}}
Only use internal link paths that exist in the facts above."""
    return system, user


# ── Node: draft writer (single monolithic call — kept for reference) ────────
def write_prompt(facts_block: str, state: dict) -> tuple[str, str]:
    outline = state["outline"]
    archetype = state.get("archetype", "decision_framework")
    hook = _INTRO_HOOK.get(archetype, "Open with a concrete business problem.")
    system = STUDIO_PERSONA + (
        " You write in MDX for the studio's blog, which has a fixed component kit."
    )
    user = f"""Write a complete blog post body in MDX, for a BUSINESS AUDIENCE (founders,
operators, decision-makers — not developers).

STUDIO FACTS (ground every specific claim in these — do not invent numbers,
clients, sectors, or statistics; if you don't have a real number, speak
qualitatively):

{facts_block}

POST PLAN:
  primary keyword: {state['primary_keyword']}
  angle: {state['angle']}
  archetype: {archetype}
  working title: {outline.get('working_title')}
  H2 sections: {outline.get('h2s')}
  semantic keywords to weave in: {outline.get('lsi_keywords')}
  internal links to include: {outline.get('internal_links')}
  primary illustration: {outline.get('primary_illustration')}
  secondary illustration: {outline.get('secondary_illustration')}
  real projects you may cite by name: {outline.get('real_projects_to_cite')}

INTRO HOOK for archetype "{archetype}": {hook}

FORMAT RULES (this blog's contract):
  - Start with a 2-3 sentence lead paragraph (no heading) using the hook guidance above.
    Name the problem in terms a business owner would recognise. Use the primary keyword naturally.
  - Then a <KeyTakeaways points={{["...", "...", "..."]}} /> with 3-5 short business-outcome points
    (what the reader will be able to decide or do after reading this).
  - Then the H2 sections (## ...). Write for a non-technical business reader: explain the "why"
    and "so what" first, then any mechanics. Use **bold**, bullet lists, and the occasional
    > blockquote for punch. Vary sentence length so it reads human.
  - Include the primary illustration component:
      <FlowDiagram caption="..." steps={{[{{ label: "...", sub: "..." }}, ...]}} />
      <CompareDiagram caption="..." columns={{[{{ title, tone: "good"|"bad"|"neutral", points: [...] }}]}} />
      <BarChart caption="..." unit="..." data={{[{{ label, value }}, ...]}} />
      <StatGrid caption="..." stats={{[{{ label, value, unit?, context? }}, ...]}} />
      <Timeline caption="..." events={{[{{ date, label, description? }}, ...]}} />
      <DecisionTree caption="..." question="..." yes={{{{ label, outcome }}}} no={{{{ label, outcome }}}} />
  - Include the secondary illustration component in a later section.
  - Add 2-3+ internal markdown links from the plan, e.g. [text](/services/web).
  - End with <FAQ items={{[{{ q: "...", a: "..." }}, ...]}} /> (3-5 real Q&As written for a
    business buyer — questions about cost, time, risk, ownership — not technical questions)
    and then <BlogCTA />.

HARD MDX RULES:
  - No H1 (#) and no YAML frontmatter — the page adds the title itself.
  - No markdown tables. Use <CompareDiagram> instead.
  - Never write a raw '<' or '{{' in ordinary prose. Write "under 200 ms", not the
    symbol version; write "the data", not "the {{data}}".
  - Only use these components: KeyTakeaways, Callout, FlowDiagram, CompareDiagram,
    BarChart, StatGrid, Timeline, DecisionTree, Figure, FAQ, BlogCTA. No imports.

Write only the MDX body, starting with the lead paragraph."""
    return system, user


# ── Sectioned writing (robust: many short calls instead of one long one) ────
# Each of these produces a SMALL chunk (~10-15s call) so a proxy 502/timeout on
# any one chunk only costs that chunk, not the whole article. The write node
# assembles the chunks into the final MDX deterministically.

_MDX_RULES = """HARD MDX RULES:
  - No H1 (#) and no YAML frontmatter.
  - No markdown tables. Never write a raw '<' or '{' in ordinary prose (write
    "under 200 ms", not the symbol; "the data", not "the {data}").
  - Only these components exist: KeyTakeaways, Callout, FlowDiagram, CompareDiagram,
    BarChart, StatGrid, Timeline, DecisionTree, Figure, FAQ, BlogCTA. No imports."""


def section_intro_prompt(facts_block: str, state: dict) -> tuple[str, str]:
    """Lead paragraph + KeyTakeaways only."""
    outline = state["outline"]
    archetype = state.get("archetype", "decision_framework")
    hook = _INTRO_HOOK.get(archetype, "Open with a concrete business problem.")
    system = STUDIO_PERSONA + " You write in MDX. Right now you write only the opening."
    user = f"""Write ONLY the opening of a studio blog post in MDX.
The post is for a BUSINESS AUDIENCE (founders, operators, decision-makers — not developers).

STUDIO FACTS (ground claims in these; never invent numbers/clients):
{facts_block}

Post: title "{outline.get('working_title')}", primary keyword
"{state['primary_keyword']}", angle: {state['angle']}.
Archetype: {archetype}
It will cover these sections (do not write them now): {outline.get('h2s')}

INTRO HOOK for this archetype: {hook}

Write, in order:
  1. A 2-3 sentence lead paragraph (no heading) using the hook guidance above.
     Write for a business owner, not a developer — name a business problem (cost,
     risk, wasted time, missed opportunity), not a technical challenge. Use the
     primary keyword naturally.
  2. A <KeyTakeaways points={{["...", "...", "..."]}} /> with exactly 3-4 short
     skimmable points (each under 16 words) framed as business outcomes —
     what the reader will be able to decide, save, or avoid after reading this.

Keep it tight: about 60-90 words total for the lead, then the component. Do not
write any section headings or body sections — only the lead and the KeyTakeaways.

{_MDX_RULES}

Output only those two things (lead paragraph, then the KeyTakeaways component)."""
    return system, user


def section_body_prompt(
    facts_block: str, state: dict, h2: str, assignments: dict,
) -> tuple[str, str]:
    """One H2 section. `assignments` may include an illustration and/or a link."""
    outline = state["outline"]
    archetype = state.get("archetype", "decision_framework")
    # Tell this section what the OTHER sections cover, so it stays in its lane.
    others = [h for h in (outline.get("h2s") or []) if h != h2]
    others_line = "; ".join(others) if others else "(none)"
    extra = []
    if assignments.get("illustration"):
        ill = assignments["illustration"]
        ill_type = ill.get("type", "CompareDiagram")
        purpose = ill.get("purpose", "illustrate the point")
        data_hint = ill.get("data_hint", "use relevant data from the facts or qualitative estimates")
        syntax_examples = {
            "FlowDiagram": '<FlowDiagram caption="..." steps={[{ label: "...", sub: "..." }, ...]} />',
            "CompareDiagram": '<CompareDiagram caption="..." columns={[{ title, tone: "good"|"bad"|"neutral", points: [...] }]} />',
            "BarChart": '<BarChart caption="..." unit="..." data={[{ label, value }, ...]} />',
            "StatGrid": '<StatGrid caption="..." stats={[{ label: "...", value: "...", unit: "...", context: "..." }, ...]} />',
            "Timeline": '<Timeline caption="..." events={[{ date: "...", label: "...", description: "..." }, ...]} />',
            "DecisionTree": '<DecisionTree caption="..." question="..." yes={{ label: "...", outcome: "..." }} no={{ label: "...", outcome: "..." }} />',
        }
        syntax = syntax_examples.get(ill_type, syntax_examples["CompareDiagram"])
        extra.append(
            f"Include one {ill_type} component here (purpose: {purpose}; data hint: {data_hint}). "
            f"Use this exact syntax:\n  {syntax}"
        )
    if assignments.get("link"):
        lk = assignments["link"]
        extra.append(f'Include exactly one internal markdown link: [{lk.get("anchor","see this")}]({lk.get("path","/contact")}).')
    if assignments.get("callout"):
        extra.append('You may add one <Callout variant="tip">...</Callout> if it genuinely helps a business reader.')
    extra_block = "\n".join(f"  - {e}" for e in extra) if extra else "  - (prose only for this section)"

    system = STUDIO_PERSONA + " You write in MDX. Right now you write only ONE section."
    user = f"""Write ONE section of a studio blog post in MDX.
The post is for a BUSINESS AUDIENCE — write for the non-technical decision-maker,
not for a developer. Explain business implications, costs, risks, and outcomes first.

STUDIO FACTS (ground claims in these; never invent numbers/clients; you may cite
these real projects if relevant: {outline.get('real_projects_to_cite')}):
{facts_block}

Archetype: {archetype}
The post's primary keyword is "{state['primary_keyword']}". Weave in these semantic
terms only where natural: {outline.get('lsi_keywords')}.

Write the section under this exact H2 heading:
## {h2}

OTHER sections of this post (already being written separately) cover: {others_line}.
Stay strictly within YOUR heading's scope — do not restate their points or re-explain
the overall thesis; assume the reader has read them.

Then 130-200 words of body copy (aim for that length — concise, not padded; vary
sentence length; use **bold**, a bullet list, or a > blockquote where it helps —
human, specific, written for a business buyer). Requirements:
{extra_block}

{_MDX_RULES}

Start with the "## {h2}" line and output only this one section."""
    return system, user


def section_closing_prompt(facts_block: str, state: dict) -> tuple[str, str]:
    """FAQ + BlogCTA."""
    outline = state["outline"]
    archetype = state.get("archetype", "decision_framework")
    system = STUDIO_PERSONA + " You write in MDX. Right now you write only the closing."
    user = f"""Write ONLY the closing of a studio blog post in MDX.

The post is about "{state['primary_keyword']}" ({state['angle']}).
Archetype: {archetype}. It already has an intro and these sections: {outline.get('h2s')}.

STUDIO FACTS (ground answers in these; never invent):
{facts_block}

Write, in order:
  1. A <FAQ items={{[{{ q: "...", a: "..." }}, ...]}} /> with exactly 3-4 real questions
     a BUSINESS BUYER would search about this topic — questions about cost, time,
     risk, ownership, or decision-making — NOT technical implementation questions.
     Each answer should be self-contained in 1-2 sentences.
  2. A <BlogCTA /> on its own line (optionally with a short text="..." that invites
     the reader to describe their project / get a free prototype).

Keep it tight — the FAQ answers should be brief.

{_MDX_RULES}

Output only the FAQ component then the BlogCTA."""
    return system, user


# ── Node: fact-check guard ──────────────────────────────────────────────────
# Deliberately NARROW: only flag fabrications ABOUT WIZCODES itself.
def factcheck_prompt(facts_block: str, body_mdx: str) -> tuple[str, str]:
    system = (
        "You verify that a blog draft doesn't fabricate specific claims about the "
        "studio WizCodes. You are narrow and precise: you only flag invented facts "
        "ATTRIBUTED TO WIZCODES, never general industry statements or advice."
    )
    user = f"""SOURCE FACTS (everything known to be true about WizCodes):

{facts_block}

DRAFT:
'''
{body_mdx}
'''

Flag a claim ONLY if it invents something specific about WIZCODES that the source
facts don't support, such as:
  - a WizCodes client, project, or product name not in the facts;
  - a specific statistic/number/metric/date attributed to WizCodes (e.g. "we cut
    costs by 40%", "we've built 200 apps") that isn't in the facts;
  - a claim WizCodes did something it didn't (a service, a technology used on a
    named project) that contradicts the facts.

Do NOT flag (these are all fine):
  - general industry statements, best practices, opinions, or business advice;
  - qualitative statements ("fast", "affordable", "production-ready");
  - common knowledge about tools/frameworks (React, Firebase, Stripe, etc.);
  - the real WizCodes projects/services that ARE in the facts.

Be conservative — when unsure, do NOT flag it. Most drafts should return zero issues.

Reply as JSON:
{{"issues": [ {{"quote": "the exact phrase from the draft", "problem": "why it invents a WizCodes fact", "fix": "how to reword truthfully"}} ]}}
If nothing genuinely invents a WizCodes fact, return {{"issues": []}}."""
    return system, user


# ── Node: surgical claim fixer ──────────────────────────────────────────────
def fix_claims_prompt(body_mdx: str, issues: list[str]) -> tuple[str, str]:
    issue_lines = "\n".join(f"  - {i}" for i in issues) or "  (none)"
    system = (
        "You make minimal surgical edits to an MDX blog draft to remove or reword a "
        "few specific claims, changing nothing else. You preserve all components, "
        "links, headings, and the FAQ exactly."
    )
    user = f"""Here is an MDX blog draft:
'''
{body_mdx}
'''

Reword or remove ONLY the following flagged claims so they no longer state the
unsupported fact (make them qualitative/general, or drop the sentence). Change
NOTHING else — keep every heading, component, link, and FAQ item identical:
{issue_lines}

Output the full corrected MDX body (same format and components), and nothing else."""
    return system, user


# ── Node: humanizer / critic ─────────────────────────────────────────────────
def humanize_prompt(body_mdx: str) -> tuple[str, str]:
    system = STUDIO_PERSONA + (
        " You are reviewing a draft for how human and specific it reads, then "
        "improving it. You keep all components and links intact."
    )
    user = f"""Review this MDX draft for the specific patterns that make writing read as
machine-generated. These were measured against the studio's hand-written posts, so
they are the real tells, not generic advice:

  - The "not X. It's Y" rhetorical flip ( "It isn't about cost. It's about control." )
    appeared 4 times across the automated posts and ZERO times in the hand-written
    ones. Use it at most once, ideally never.
  - Sentences that all start the same way. The automated posts opened 3-5 sentences
    per article with "The", "Here's", "That's" or "It's"; the hand-written ones did
    it once. Vary the openings.
  - Uniform sentence length. Real writing alternates long and short. Put a
    three-word sentence next to a twenty-five-word one on purpose.
  - Triplets used as filler ("fast, reliable, and scalable"). Keep them only where
    all three words carry weight.
  - Throat-clearing intros, filler transitions, and empty adjectives.

Rewrite to fix those — WITHOUT changing the components, the internal links, the FAQ
questions/answers, or the core facts.

DRAFT:
'''
{body_mdx}
'''

Output the improved full MDX body (starting with the lead paragraph, same format
and components). Then, on a final separate line, add a marker exactly like this:
<!--HUMANSCORE: N--> where N is 0-100 for how human the ORIGINAL draft read."""
    return system, user


# ── Node: registry builder ───────────────────────────────────────────────────
def registry_prompt(body_mdx: str, primary_keyword: str, existing_slugs: list[str]) -> tuple[str, str]:
    system = (
        "You write SEO metadata for a blog post: the slug, title, meta description, "
        "and tags. You follow length limits precisely."
    )
    user = f"""Here is a finished blog post body:

'''
{body_mdx[:4000]}
'''

Primary keyword: "{primary_keyword}"
Slugs already taken (must not reuse): {existing_slugs}

Write the TITLE for click-through, not just for accuracy. Titles like "Cost to
Maintain a Mobile App Per Year: Real Numbers" are findable but flat. What works:

  - A curiosity gap that WITHHOLDS THE MECHANISM, never the value. "The One Thing
    That Doubles App Maintenance Bills" invites a click; "Everything About App
    Maintenance" does not.
  - Specific numbers beat vague ones: "7 reasons", not "some reasons".
  - Brackets or parentheses at the end lift click-through noticeably, e.g.
    "(And What To Do Instead)", "(With Real Numbers)".
  - Openers that work: "What Most Founders Get Wrong About...", "What Nobody Tells
    You About...", "The Real Reason...".

Two hard rules, because a headline that overpromises loses more than it gains:
  1. The title's core promise MUST be answered in the post's first paragraph. If the
     reader has to hunt for what you promised, the title is wrong.
  2. Never imply a number, claim, or outcome the body does not actually contain.

Produce the registry metadata as JSON:
{{
  "slug": string (kebab-case, contains the primary keyword, unique vs the taken list),
  "title": string (55-65 chars, includes the primary keyword, written for a business reader),
  "description": string (140-160 chars, includes the keyword and a concrete business benefit),
  "tags": [2-4 Title Case tags]
}}"""
    return system, user
