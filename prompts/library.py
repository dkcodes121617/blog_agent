"""Prompt builders for every LLM node.

Design rules learned from testing the proxy (ClaudeStore, now LLMsRelay):
  - Phrase everything as a normal professional content task. NEVER use
    override/compliance language ("reply with exactly", "never break character",
    "obey this contract") — the proxy's injection guard refuses those.
  - Put the real WizCodes facts in the prompt and tell the model to ground claims
    in them, which is what stops hallucinated numbers/clients.
  - Ask for JSON where we need structure; ask for MDX where we need the body.

Each function returns (system, user).
"""
from __future__ import annotations

from seo import content_policy

# ── Archetypes ──────────────────────────────────────────────────────────────
# Four content archetypes, each with its own H2 scaffold, required illustration
# type, and intro hook style. The archetype is picked at topic-selection time and
# threads through every subsequent node.

ARCHETYPES = (
    "explainer",        # what it is and how it actually works
    "playbook",         # how to approach it well — criteria and sequence, no durations
    "tips_list",        # N specific, checkable tips / signals / checks
    "mistake_guide",    # what goes wrong, and what to do instead
    "vs_comparison",    # A vs B on capability, fit and ownership — never on price
    "trend_brief",      # what changed recently in this area and what it means now
)

# The shape to fall back on when a caller has no archetype, or names one that no longer
# exists. `explainer` because every subject can be written as one — a fallback that only
# suits some topics is a fallback that produces a bad post on the rest.
DEFAULT_ARCHETYPE = "explainer"

# `cost_breakdown` is GONE, and its absence is the point.
#
# It was one of four shapes, which sounds like a quarter of the output; in practice the
# strategist reached for it whenever it was available, four times consecutively at one
# point, and half the published back catalogue is built on price or duration figures.
# That content argues against the studio's own position — custom software, affordable
# and fast because it is built with AI — and it recruits readers who are collecting
# quotes rather than readers with a problem. See seo/content_policy.py, which now makes
# the topic unpickable rather than merely unfashionable.
#
# The six that replaced it are all KNOWLEDGE shapes, chosen for what gets quoted back by
# an answer engine: a definition with a mechanism under it, a checkable list, a named
# failure and its fix, a comparison decided on capability, and what changed lately.
# Those are the questions people actually put to Google and to an assistant, and none of
# them has a number in the headline.

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
            r"\bmistakes?\b|\bpitfalls?\b|\bavoid\b|\bwrong\b|\bdon'?t\b|\bstop\b"
            r"|\bgets? wrong\b|\bmyths?\b", re.I)),
        # Spelled-out counts as well as digits: the registry prompt asks for "7 checks"
        # and the writer regularly returns "Seven Checks", which matched nothing and
        # classified as the default shape.
        ("tips_list", re.compile(
            r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
            r"(?:tips|ways|things|signs|checks|rules|habits|questions|steps|lessons)\b"
            r"|\btips\b|\bchecklist\b|\bsigns\b", re.I)),
        ("trend_brief", re.compile(
            r"\bin 20\d\d\b|\bwhat'?s (new|changed)\b|\bnow\b|\blatest\b|\btoday\b"
            r"|\bstate of\b|\bchanged\b", re.I)),
        ("vs_comparison", re.compile(
            r"\bvs\.?\b|\bversus\b|\bcompared? (to|with)\b|\balternatives?\b"
            r"|\btrade-?offs?\b", re.I)),
        ("playbook", re.compile(
            r"\bhow to (choose|decide|pick|approach|plan|scope|run)\b|\bshould (i|you|we)\b"
            r"|\bwhich\b|\bwhen to\b|\bframework\b|\bcriteria\b|\bplaybook\b", re.I)),
        ("explainer", re.compile(
            r"\bwhat is\b|\bwhat are\b|\bhow (does|do)\b|\bexplained\b|\bactually works?\b"
            r"|\bunder the hood\b|\bguide to\b", re.I)),
    ]


# Retired archetypes, mapped onto the shape that replaced them.
#
# The registry records the archetype the publisher chose, and twenty published posts
# carry names that no longer exist. Without this, blocked_archetypes() reads a window of
# posts whose recorded shapes match nothing in ARCHETYPES, concludes nothing is
# over-used, and the rotation quietly stops working on exactly the run after the rename.
_LEGACY_ARCHETYPES = {
    "cost_breakdown": "explainer",
    "decision_framework": "playbook",
}


def infer_archetype(title: str, description: str = "") -> str:
    """Best-effort archetype for an already-published post, from its title.

    The title is checked first because it is the most deliberate signal; the
    description is only a fallback. Defaults to the explainer shape, which is the
    one every subject can be written as, when nothing matches.
    """
    global _ARCHETYPE_SIGNATURES
    if not _ARCHETYPE_SIGNATURES:
        _ARCHETYPE_SIGNATURES = _build_signatures()
    for text in (title or "", description or ""):
        for name, pattern in _ARCHETYPE_SIGNATURES:
            if pattern.search(text):
                return name
    return "explainer"


def normalise_archetype(name: str) -> str:
    """A current archetype name, translating the retired ones."""
    n = (name or "").strip().lower()
    return _LEGACY_ARCHETYPES.get(n, n)


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
    # Prefer the archetype the publisher RECORDED over one reverse-engineered from
    # the title. Rotation exists to stop four cost-breakdowns in a row, and it was
    # deciding that from a regex over headlines — so a decision_framework post
    # titled "What custom software actually costs" was counted as a cost_breakdown
    # and blocked the wrong thing. Inference stays as the fallback for posts
    # published before the field existed.
    inferred = [
        normalise_archetype(
            p.get("archetype")
            or infer_archetype(p.get("title", ""), p.get("description", ""))
        )
        for p in window
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
    # Every scaffold below is answerable in a self-contained paragraph, because that is
    # the unit an answer engine lifts. A heading like "A realistic cost breakdown" also
    # met that bar, which is why the shapes changed rather than the phrasing.
    "explainer": [
        "What [topic] actually means",
        "How it works, step by step",
        "What it looks like in a real build",
        "Where teams get the model wrong",
        "How to tell whether you need it",
    ],
    "playbook": [
        "What actually decides the answer",
        "The questions to settle first",
        "How to approach [topic] in practice",
        "What to check before you commit",
        "How to tell it is working",
    ],
    "tips_list": [
        "Start with the data model, not the screens",
        "The checks most teams skip",
        "What separates a good build from a fragile one",
        "Small decisions with outsized consequences",
        "How to keep it maintainable",
    ],
    "mistake_guide": [
        "The mistake almost everyone makes with [topic]",
        "Why the usual advice backfires",
        "What to do instead",
        "How to spot it early",
        "A cleaner approach from the start",
    ],
    "vs_comparison": [
        "What each option actually does",
        "Where [option A] wins",
        "Where [option B] wins",
        "The criteria most comparisons skip",
        "How to pick for your situation",
    ],
    "trend_brief": [
        "What changed in [topic]",
        "Why it matters for a custom build",
        "What it makes possible that was not before",
        "What to be sceptical about",
        "What to do about it now",
    ],
}

def h2_scaffold(archetype: str, topic: str = "") -> list[str]:
    """Usable H2 headings for an archetype, as a floor under a thin outline.

    Patterns whose placeholders can't be resolved (the option A/B slots in the
    comparison scaffold) are dropped rather than shipped with brackets in them,
    and generic sections are appended so there is always something to top up with.
    """
    patterns = _H2_PATTERNS.get(archetype) or _H2_PATTERNS[DEFAULT_ARCHETYPE]
    out = []
    for pattern in patterns:
        heading = pattern.replace("[topic]", topic).strip() if topic else pattern
        if "[" in heading:
            continue
        out.append(heading)
    out += [
        "What this means in practice",
        "What to watch for before you commit",
        "How to get started",
    ]
    return out


# Archetype → preferred primary illustration type.
#
# Timeline is absent from BOTH maps and from the menu offered in the outline and section
# prompts. It is a component for showing something unfolding over time, so the writer
# reliably filled it with "Month 0 / Month 6 / Month 18" — the exact claim the editorial
# standard forbids, rendered into a chart, then baked into the post's cover image and its
# alt text. The component still exists for the posts that already use it; it is simply
# never commissioned again.
_PRIMARY_ILLUSTRATION: dict[str, str] = {
    "explainer": "ConceptDiagram",
    "playbook": "DecisionTree",
    "tips_list": "CompareDiagram",
    "mistake_guide": "CompareDiagram",
    "vs_comparison": "CompareDiagram",
    "trend_brief": "ConceptDiagram",
}

# Archetype → preferred secondary illustration type.
_SECONDARY_ILLUSTRATION: dict[str, str] = {
    "explainer": "FlowDiagram",
    "playbook": "QuadrantMap",
    "tips_list": "FlowDiagram",
    "mistake_guide": "FlowDiagram",
    "vs_comparison": "QuadrantMap",
    "trend_brief": "CompareDiagram",
}

# Archetype → intro hook guidance (how the lead paragraph should open).
_INTRO_HOOK: dict[str, str] = {
    # AEO: retrieval weights the opening heavily and wants the ANSWER there, not a
    # story. Each hook is capped at one sentence, followed immediately by a direct
    # answer. The narrative still happens, further down, where it costs nothing.
    # These previously all asked for a scene-setting anecdote, which spent the whole
    # extractable window on throat-clearing.
    "explainer": (
        "ONE short sentence naming the thing in plain words — a definition a reader "
        "could quote on its own. Then immediately say how it works in one or two "
        "sentences, before any context or history."
    ),
    "playbook": (
        "ONE short sentence on what goes wrong when this is approached backwards. Then "
        "immediately give the short version: the one or two things that actually decide "
        "the answer."
    ),
    "tips_list": (
        "ONE short sentence on why this is where builds go sideways. Then immediately "
        "state the single most useful of the tips, in full, before listing the rest."
    ),
    "mistake_guide": (
        "ONE short sentence naming the mistake plainly. Then immediately say what to "
        "do instead, before explaining why."
    ),
    "vs_comparison": (
        "ONE short sentence acknowledging the choice. Then immediately state which "
        "option suits which situation - the actual answer - before any elaboration."
    ),
    "trend_brief": (
        "ONE short sentence naming exactly what changed. Then immediately say what it "
        "changes in practice for someone building custom software, before the detail."
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
    "\n\n"
    # ── Readability ──
    # Earlier posts were accurate but hard work to read: long multi-clause sentences
    # and abstract vocabulary. The audience is founders and business owners, not
    # engineers, and many read English as a second language. Plain writing also
    # extracts better — answer engines quote short, self-contained sentences far
    # more readily than nested ones. This lives on the persona so every writing node
    # inherits it.
    "Write so a busy, non-technical founder understands it on one read:\n"
    "  - Short sentences, one idea each. If a sentence has three commas, split it.\n"
    "  - Mostly 8-18 words, with an occasional very short one for rhythm.\n"
    "  - Plain words where they are exact. 'We agree the scope up front' beats\n"
    "    'scope is determined a priori'. Never use a longer word just to sound senior.\n"
    "  - Explain any technical term the first time it appears, in the same sentence.\n"
    "  - Short paragraphs: two to four sentences, then a break.\n"
    "  - Active voice and concrete nouns. Say who does what.\n"
    "  - No filler openers, no throat-clearing, and no empty adjectives such as\n"
    "    robust, seamless, cutting-edge or powerful.\n"
    "  - Simple is not casual. Keep the professional register and never talk down."
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
        "You are the content strategist for a custom software studio. Your job is to "
        "pick topics that answer the questions people actually type into Google and ask "
        "an AI assistant when they are trying to understand or plan a piece of custom "
        "software — and to pick the ones this studio can answer better than a generic "
        "listicle, because it builds these systems. You are aiming to be the source "
        "that gets quoted, not the one that gets scrolled past."
    )
    avoid = "\n".join(f"  - {s}" for s in avoid_recent) or "  (none yet)"

    # Over-used archetypes are removed from the menu entirely rather than merely
    # discouraged — the model reliably picks whatever is listed, and a "please vary"
    # instruction did not stop four cost breakdowns in a row.
    focus_block = (describe_focus(focus) + "\n\n") if focus else ""
    policy_block = content_policy.POLICY_PROMPT_BLOCK
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

{focus_block}Propose ONE blog topic that helps someone understand or plan a piece of CUSTOM-BUILT
software. The reader owns the problem — a founder, an operations lead, a product owner,
a technical decision-maker — and wants to understand the thing well enough to make a
good call. They are not looking for a quote.

The topic must sit inside the assigned focus above. Ground it in the real projects
listed for that focus — that is what makes the post worth reading rather than generic.

{policy_block}

WHAT MAKES A GOOD TOPIC HERE

Pick a question people genuinely ask — the phrasing they use with a search engine or an
assistant. The best ones are specific enough to answer completely:

  - "how X actually works" / "what X means in practice"
  - "how to decide between X and Y" / "what to look for in X"
  - "why X breaks" / "what most teams get wrong about X"
  - "what changed in X" / "what X makes possible now"
  - "N things to get right in X" / "the checks that matter in X"

Aim to be CITABLE. A post gets quoted by an answer engine when it states a claim
plainly, in a self-contained sentence, and backs it with something specific — a
mechanism, a named trade-off, a concrete example from real work. Vague, hedged and
comprehensive-but-shallow all lose to specific.

Anchor it to CUSTOM SOFTWARE. This studio builds bespoke systems, so the useful angle is
almost always about what a purpose-built system does differently: owning the data model,
fitting an actual workflow, extending without asking a vendor, keeping the source. Not
"the 10 best CRM tools".

Do NOT propose:
  - Anything about price, budget, ROI, or how long something takes (see the standard above)
  - Developer tutorials — "how to build X", "implementing Y", "a guide to the Z library"
  - Code walkthroughs, setup instructions, or step-by-step technical implementation
  - Generic listicles of third-party products with no build insight
  - Topics already covered by the recent posts listed above

Required archetype — pick exactly one:
{archetype_list}

  explainer      → what it is and how it actually works, with the mechanism made concrete
  playbook       → how to approach it well: the criteria and the order to settle them in
  tips_list      → a specific, checkable set of tips, signals or checks
  mistake_guide  → the common failure, why it happens, and what to do instead
  vs_comparison  → A vs B decided on capability, fit, data ownership and maintenance
  trend_brief    → what has genuinely changed here lately and what it means for a build

Reply as JSON:
{{"primary_keyword": string (the exact phrase someone would type or ask),
  "angle": string (the specific thesis, one sentence — what this post claims that a generic article would not),
  "audience": string (be specific: e.g. "ops lead whose team has outgrown a shared spreadsheet"),
  "archetype": string (one of the archetypes listed above),
  "intent_type": "informational" | "commercial",
  "citable_claim": string (the one sentence you would want an AI assistant to quote from this post),
  "rationale": string (why people search this + why WizCodes can answer it more concretely than a generic article)}}"""
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
    h2_pattern = _H2_PATTERNS.get(archetype, _H2_PATTERNS[DEFAULT_ARCHETYPE])
    h2_guidance = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(h2_pattern))
    primary_ill = _PRIMARY_ILLUSTRATION.get(archetype, "CompareDiagram")
    secondary_ill = _SECONDARY_ILLUSTRATION.get(archetype, "BarChart")
    hook_guidance = _INTRO_HOOK.get(archetype, "Open with a concrete business problem.")
    policy_block = content_policy.POLICY_PROMPT_BLOCK

    user = f"""Studio facts (ground truth — reference real projects, never invent):

{facts_block}

Plan a blog post for someone deciding on or planning a piece of CUSTOM-BUILT software —
a founder, an operations lead, a product owner, a technical decision-maker. They want to
understand the subject properly, not read a sales page and not read a code tutorial.
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
and adds no new visual surface. Match the type to the content instead - ConceptDiagram
for how the pieces fit together, FlowDiagram for a process, CompareDiagram for
this-vs-that, DecisionTree for a branching choice, QuadrantMap for positioning,
BarChart for magnitudes, StatGrid for headline figures.

Give every illustration a specific, descriptive caption. The caption becomes the
image's filename, its <title>, and the text answer engines actually read - "How a
custom CRM routes an inbound lead" earns search traffic, "Chart 1" earns none.

{policy_block}

This applies to the illustrations above all. A "data_hint" asking for a cost comparison
or a delivery schedule is what actually produces the figures - the section writer builds
the chart the plan asks for. Plan charts that carry capability, fit, ownership, effort
in relative terms, failure modes, or before-and-after behaviour instead.


Produce a JSON plan:
{{
  "working_title": string (AT MOST 52 characters — the site appends " | WizCodes" (11 more) so anything longer is truncated in Google. Natural, leads with the primary keyword, sounds like a business article — no "how to build X"),
  "h2s": [4-5 strings adapted from the archetype pattern above, keyword-rich but natural and buyer-facing — four is the minimum a post can ship with],
  "lsi_keywords": [5-8 semantic variants to weave in naturally — business terms, not technical jargon],
  "internal_links": [3-4 objects {{"path": "/services/... or /work/... or /blog/... or /contact", "anchor": string}}],
  "primary_illustration": {{"type": "{primary_ill}", "purpose": string, "data_hint": string (what data/content to put in it)}},
  "secondary_illustration": {{"type": "{secondary_ill}", "purpose": string, "data_hint": string}},
  "extra_illustrations": [
    {{"type": one of BarChart|CompareDiagram|StatGrid|FlowDiagram|DecisionTree|ConceptDiagram|QuadrantMap,
      "purpose": string, "data_hint": string}}
  ],
  "real_projects_to_cite": [names from the facts that genuinely fit this topic]
}}
Only use internal link paths that exist in the facts above."""
    return system, user


# The single monolithic write_prompt used to live here. It was dead code — defined,
# never called, and superseded by the sectioned prompts below — but it still carried a
# full copy of the MDX contract, the component menu and the audience framing. A second
# copy of the rules that nothing executes is worse than no copy: this one still listed
# Timeline and still asked the FAQ for questions about cost and time, so any edit made
# by reading it would have been an edit to the wrong prompt.

# ── Sectioned writing (robust: many short calls instead of one long one) ────
# Each of these produces a SMALL chunk (~10-15s call) so a proxy 502/timeout on
# any one chunk only costs that chunk, not the whole article. The write node
# assembles the chunks into the final MDX deterministically.

_MDX_RULES = f"""HARD MDX RULES:
  - No H1 (#) and no YAML frontmatter.
  - Diagram text must be SHORT or it gets clipped inside its box: labels 2-3 words
    (under ~18 characters), sub-labels and bullets under ~40 characters. Put the
    detail in the prose around the diagram, not inside it.
  - Keep diagrams SPARSE - a diagram is a glance, not a second article. 3-5 flow
    steps, 3-4 bullets per compare column, at most 4 bars, one idea per row.
  - PROP NAMES ARE FIXED. Each component reads one specific prop and renders nothing
    at all if it is called something else:
      BarChart data=   StatGrid stats=   CompareDiagram columns=   FlowDiagram steps=
      ConceptDiagram nodes=   QuadrantMap quadrants=   KeyTakeaways points=   FAQ items=
    Entry keys are fixed too: FlowDiagram steps use `label`, ConceptDiagram nodes and
    CompareDiagram columns use `title`, StatGrid stats use `value`.
  - Your illustration data also becomes the post's COVER IMAGE and its alt text, so use
    real, specific values and keep data labels short and concrete. Never put a price or
    a duration in a diagram — it ends up enlarged on the blog index.
  - No markdown tables. Never write a raw '<' or '{{' in ordinary prose (write
    "under 200 ms", not the symbol; "the data", not "the {{data}}").
  - Only these components exist: KeyTakeaways, Callout, FlowDiagram, CompareDiagram,
    BarChart, StatGrid, DecisionTree, ConceptDiagram, QuadrantMap, Figure, FAQ,
    BlogCTA. No imports.

{content_policy.POLICY_PROMPT_BLOCK}"""


def section_intro_prompt(facts_block: str, state: dict) -> tuple[str, str]:
    """Lead paragraph + KeyTakeaways only."""
    outline = state["outline"]
    archetype = state.get("archetype", DEFAULT_ARCHETYPE)
    hook = _INTRO_HOOK.get(archetype, "Open with a concrete business problem.")
    system = STUDIO_PERSONA + " You write in MDX. Right now you write only the opening."
    user = f"""Write ONLY the opening of a studio blog post in MDX.
The post is for someone planning or deciding on custom-built software — not a developer
looking for code, and not a buyer looking for a quote.

STUDIO FACTS (ground claims in these; never invent numbers/clients):
{facts_block}

Post: title "{outline.get('working_title')}", primary keyword
"{state['primary_keyword']}", angle: {state['angle']}.
Archetype: {archetype}
It will cover these sections (do not write them now): {outline.get('h2s')}

INTRO HOOK for this archetype: {hook}

Write, in order:
  1. A 2-3 sentence lead paragraph (no heading) using the hook guidance above.
     Name the problem the way the reader would describe it — the thing that keeps
     breaking, the decision they cannot settle, the part nobody explains properly.
     Not a price and not a schedule. Use the primary keyword naturally, and make the
     second sentence a claim that stands on its own if it were quoted alone.
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
    archetype = state.get("archetype", DEFAULT_ARCHETYPE)
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
            "DecisionTree": '<DecisionTree caption="..." question="..." yes={{ label: "...", outcome: "..." }} no={{ label: "...", outcome: "..." }} />',
            "ConceptDiagram": '<ConceptDiagram caption="..." nodes={[{ title: "...", sub: "..." }, ...]} />',
            "QuadrantMap": '<QuadrantMap caption="..." xAxis={{ low: "...", high: "..." }} yAxis={{ low: "...", high: "..." }} quadrants={{ topLeft: "...", topRight: "...", bottomLeft: "...", bottomRight: "..." }} />',
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
The post is for someone planning custom-built software. Explain the mechanism and the
trade-offs in plain words — what it does, why it behaves that way, what that means for
their build. Never price and never duration.

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
    archetype = state.get("archetype", DEFAULT_ARCHETYPE)
    system = STUDIO_PERSONA + " You write in MDX. Right now you write only the closing."
    user = f"""Write ONLY the closing of a studio blog post in MDX.

The post is about "{state['primary_keyword']}" ({state['angle']}).
Archetype: {archetype}. It already has an intro and these sections: {outline.get('h2s')}.

STUDIO FACTS (ground answers in these; never invent):
{facts_block}

Write, in order:
  1. A <FAQ items={{[{{ q: "...", a: "..." }}, ...]}} /> with 4-6 real questions
     someone would actually type or ask an assistant about this topic — what it
     means, how it works, what to check, what goes wrong, how to tell if it fits,
     what happens to the data. NOT questions about price or how long it takes, and
     NOT technical implementation questions.
     Write each answer to be QUOTED: a direct, self-contained 1-2 sentences that
     makes sense on its own, with no "as mentioned above" and no hedging.
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


# ── Node: surgical MDX repair ────────────────────────────────────────────────
def shorten_labels_prompt(items: list[dict]) -> tuple[str, str]:
    """Ask for replacement STRINGS, not a replacement post.

    The draft is never sent back through the model here. Rewriting the whole body
    to fix a 29-character label is what made the old repair loop diverge — every
    rewrite regenerated every diagram and clipped a different one. Asking for the
    strings alone means the edit is applied in Python, so nothing outside the
    flagged values can change and the pass always converges.
    """
    lines = []
    for it in items:
        if it["kind"] == "caption":
            lines.append(
                f'  {it["id"]}. A caption for the <{it["component"]}> in the section '
                f'"{it["where"]}" — up to {it["max_chars"]} characters, describing what '
                f"the reader learns from it.")
        else:
            lines.append(
                f'  {it["id"]}. Shorten this {it["what"]} to {it["max_chars"]} characters '
                f'or fewer: "{it["current"]}"')
    system = (
        "You write the short text that goes inside diagram boxes for a software "
        "studio's blog. Diagram labels are cramped, so you write them the way a "
        "designer would: concrete, specific, and as short as the box allows."
    )
    user = f"""These diagram strings need to fit their boxes. Keep the meaning and the
specifics (numbers, product names, the actual distinction being drawn) — drop the
filler words, articles and repeated context instead. Plain sentence case, no
trailing punctuation, no ellipsis.

{chr(10).join(lines)}

Return a JSON array of objects with "id" and "text", one per item above."""
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

Write the TITLE for click-through, not just for accuracy. What works:

  - A curiosity gap that WITHHOLDS THE MECHANISM, never the value. "The One Decision
    That Sinks Most Custom CRMs" invites a click; "Everything About CRM Data Models"
    does not.
  - Specific counts beat vague ones: "7 checks", not "some checks".
  - Brackets or parentheses at the end lift click-through noticeably, e.g.
    "(And What To Do Instead)", "(With Real Examples)".
  - Openers that work: "What Most Teams Get Wrong About...", "What Nobody Tells
    You About...", "The Real Reason...", "How X Actually Works".
  - NEVER a price or a duration in the title, and never a title ABOUT price or
    duration - no "what it costs", "how much", "how long", "in six weeks". This
    blog does not publish those.

Two hard rules, because a headline that overpromises loses more than it gains:
  1. The title's core promise MUST be answered in the post's first paragraph. If the
     reader has to hunt for what you promised, the title is wrong.
  2. Never imply a number, claim, or outcome the body does not actually contain.

Produce the registry metadata as JSON:
{{
  "slug": string (kebab-case, contains the primary keyword, unique vs the taken list),
  "title": string (HARD LIMIT 52 characters. The site's metadata template appends " | WizCodes" — 11 more — and Google truncates the result around 60. Lead with the primary keyword; put any secondary clause after a colon so it can be dropped without breaking the headline),
  "description": string (AT MOST 155 characters, includes the keyword and a concrete business benefit. Google cuts at ~158 on desktop and ~120 on mobile, so front-load the value),
  "tags": [2-4 Title Case tags]
}}"""
    return system, user
