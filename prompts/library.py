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
    "cost_breakdown": (
        "Open with the moment a founder or business owner got a surprise bill or a "
        "wildly varying quote. Name the real cost range, then promise to explain what "
        "actually drives it."
    ),
    "vs_comparison": (
        "Open with the painful situation of having to choose between two options with "
        "conflicting advice online. Acknowledge the tension, then promise a clear, "
        "opinionated framework."
    ),
    "decision_framework": (
        "Open with a costly mistake that happens when people skip this decision or rush "
        "it. Set up why a clear framework matters more than gut feeling here."
    ),
    "mistake_guide": (
        "Open by naming the mistake bluntly — the one most founders make at exactly this "
        "stage. Make it feel recognisable, not preachy. Then promise the fix."
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


# ── Node: topic strategist ──────────────────────────────────────────────────
def topic_prompt(facts_block: str, avoid_recent: list[str]) -> tuple[str, str]:
    system = (
        "You are a lead generation strategist for a B2B software studio. Your job "
        "is to pick blog topics that attract business owners, startup founders, and "
        "non-technical decision-makers who are evaluating software development options. "
        "You write for buyers, not builders."
    )
    avoid = "\n".join(f"  - {s}" for s in avoid_recent) or "  (none yet)"
    archetype_list = "\n".join(f"  - {a}" for a in ARCHETYPES)
    user = f"""Here are the studio's real facts and its existing blog coverage:

{facts_block}

Recently covered topics to avoid repeating:
{avoid}

Propose ONE blog topic for a BUSINESS OWNER or STARTUP FOUNDER who is evaluating or
buying software services — someone who controls a budget but is not a developer.

The topic should target a keyword with commercial or navigational search intent:
cost analyses, vendor comparisons, ROI breakdowns, build-vs-buy decisions, industry
guides, common business mistakes, or decision frameworks.

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

Produce a JSON plan:
{{
  "working_title": string (natural, ~50-60 chars, includes the primary keyword, sounds like a business article — no "how to build X"),
  "h2s": [3-5 strings adapted from the archetype pattern above, keyword-rich but natural and buyer-facing],
  "lsi_keywords": [5-8 semantic variants to weave in naturally — business terms, not technical jargon],
  "internal_links": [3-4 objects {{"path": "/services/... or /work/... or /blog/... or /contact", "anchor": string}}],
  "primary_illustration": {{"type": "{primary_ill}", "purpose": string, "data_hint": string (what data/content to put in it)}},
  "secondary_illustration": {{"type": "{secondary_ill}", "purpose": string, "data_hint": string}},
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
    user = f"""Review this MDX draft for anything that reads as generic AI writing:
uniform sentence rhythm, filler transitions ("in today's fast-paced world"),
throat-clearing intros, empty adjectives, or vague claims. Then rewrite it to be
more human and specific — vary sentence length, cut filler, and make the concrete
points sharper — WITHOUT changing the components, the internal links, the FAQ
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

Produce the registry metadata as JSON:
{{
  "slug": string (kebab-case, contains the primary keyword, unique vs the taken list),
  "title": string (50-60 chars, includes the primary keyword, compelling, written for a business reader),
  "description": string (140-160 chars, includes the keyword and a concrete business benefit),
  "tags": [2-4 Title Case tags]
}}"""
    return system, user
