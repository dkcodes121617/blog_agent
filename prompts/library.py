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

# The studio persona reused across writing nodes. Warm, senior, honest — matches
# BLOG_FORMAT.md voice rules.
STUDIO_PERSONA = (
    "You are the founding engineer at WizCodes, a small remote-first software "
    "studio (web, mobile, and AI). You write the studio's blog. Your voice is "
    "first-person plural ('we'), plain-spoken, senior, and honest — never hyped, "
    "never generic. You ground everything in the studio's real work and never "
    "invent numbers, clients, or statistics."
)


# ── Node: topic strategist ──
def topic_prompt(facts_block: str, avoid_recent: list[str]) -> tuple[str, str]:
    system = (
        "You are an SEO content strategist for a software development studio. You "
        "pick blog topics that can realistically rank and attract qualified buyers "
        "(startup founders, SMBs in the US/UK/Canada/Europe looking for web, mobile, "
        "or AI development)."
    )
    avoid = "\n".join(f"  - {s}" for s in avoid_recent) or "  (none yet)"
    user = f"""Here are the studio's real facts and its existing blog coverage:

{facts_block}

Recently generated topics to avoid repeating:
{avoid}

Propose ONE fresh, high-value blog topic that:
  - targets a specific primary keyword a real buyer would search (commercial or
    informational intent), not covered by the existing posts above;
  - fits the studio's services (web / mobile / AI) and can be written from real
    experience with the projects listed;
  - is specific and ownable, not a broad generic listicle.

Reply as JSON:
{{"primary_keyword": string, "angle": string (the specific thesis/take, one sentence),
 "audience": string, "rationale": string (why this can rank + convert)}}"""
    return system, user


# ── Node: SEO outliner ──
def outline_prompt(
    facts_block: str, primary_keyword: str, angle: str, audience: str,
    related_slugs: list[str],
) -> tuple[str, str]:
    system = STUDIO_PERSONA + (
        " Right now you are outlining a post before writing it, thinking about "
        "search intent and internal linking."
    )
    related = ", ".join(f"/blog/{s}" for s in related_slugs) or "(none especially close)"
    user = f"""Studio facts (ground truth — reference real projects, never invent):

{facts_block}

Plan a blog post.
Primary keyword: "{primary_keyword}"
Angle: {angle}
Audience: {audience}

The most topically-related existing posts (link to 1-2 of these for a topic cluster):
{related}

Produce a JSON plan:
{{
  "working_title": string (natural, ~50-60 chars, includes the primary keyword),
  "h2s": [3-5 strings, each a real question or claim, keyword-rich but natural],
  "lsi_keywords": [5-8 semantic variants to weave in naturally],
  "internal_links": [3-4 objects {{"path": "/services/... or /work/... or /blog/... or /contact", "anchor": string}}],
  "illustration": {{"type": "FlowDiagram" | "CompareDiagram" | "BarChart", "purpose": string}},
  "real_projects_to_cite": [names from the facts that genuinely fit this topic]
}}
Only use internal link paths that exist in the facts above."""
    return system, user


# ── Node: draft writer ──
def write_prompt(facts_block: str, state: dict) -> tuple[str, str]:
    outline = state["outline"]
    system = STUDIO_PERSONA + (
        " You write in MDX for the studio's blog, which has a fixed component kit."
    )
    user = f"""Write a complete blog post body in MDX, following the studio's blog format.

STUDIO FACTS (ground every specific claim in these — do not invent numbers,
clients, sectors, or statistics; if you don't have a real number, speak
qualitatively):

{facts_block}

POST PLAN:
  primary keyword: {state['primary_keyword']}
  angle: {state['angle']}
  working title: {outline.get('working_title')}
  H2 sections: {outline.get('h2s')}
  semantic keywords to weave in: {outline.get('lsi_keywords')}
  internal links to include: {outline.get('internal_links')}
  suggested illustration: {outline.get('illustration')}
  real projects you may cite by name: {outline.get('real_projects_to_cite')}

FORMAT RULES (this blog's contract):
  - Start with a 2-3 sentence lead paragraph (no heading) that hooks the reader,
    names the problem, and uses the primary keyword naturally.
  - Then a <KeyTakeaways points={{["...", "...", "..."]}} /> with 3-5 short points.
  - Then the H2 sections (## ...). Use **bold**, bullet lists, and the occasional
    > blockquote for punch. Vary sentence length so it reads human.
  - Include at least one illustration component with real data:
      <FlowDiagram caption="..." steps={{[{{ label: "...", sub: "..." }}, ...]}} />
      or <CompareDiagram caption="..." columns={{[{{ title, tone: "good"|"bad", points: [...] }}]}} />
      or <BarChart caption="..." unit="..." data={{[{{ label, value }}, ...]}} />
  - Add 2-3+ internal markdown links from the plan, e.g. [text](/services/web).
  - End with <FAQ items={{[{{ q: "...", a: "..." }}, ...]}} /> (3-5 real Q&As) and then <BlogCTA />.

HARD MDX RULES:
  - No H1 (#) and no YAML frontmatter — the page adds the title itself.
  - No markdown tables. Use <CompareDiagram> instead.
  - Never write a raw '<' or '{{' in ordinary prose. Write "under 200 ms", not the
    symbol version; write "the data", not "the {{data}}".
  - Only use these components: KeyTakeaways, Callout, FlowDiagram, CompareDiagram,
    BarChart, Figure, FAQ, BlogCTA. No imports.

Write only the MDX body, starting with the lead paragraph."""
    return system, user


# ── Sectioned writing (robust: many short calls instead of one long one) ──
# Each of these produces a SMALL chunk (~10-15s call) so a proxy 502/timeout on
# any one chunk only costs that chunk, not the whole article. The write node
# assembles the chunks into the final MDX deterministically.

_MDX_RULES = """HARD MDX RULES:
  - No H1 (#) and no YAML frontmatter.
  - No markdown tables. Never write a raw '<' or '{' in ordinary prose (write
    "under 200 ms", not the symbol; "the data", not "the {data}").
  - Only these components exist: KeyTakeaways, Callout, FlowDiagram, CompareDiagram,
    BarChart, Figure, FAQ, BlogCTA. No imports."""


def section_intro_prompt(facts_block: str, state: dict) -> tuple[str, str]:
    """Lead paragraph + KeyTakeaways only."""
    outline = state["outline"]
    system = STUDIO_PERSONA + " You write in MDX. Right now you write only the opening."
    user = f"""Write ONLY the opening of a studio blog post in MDX.

STUDIO FACTS (ground claims in these; never invent numbers/clients):
{facts_block}

Post: title "{outline.get('working_title')}", primary keyword
"{state['primary_keyword']}", angle: {state['angle']}.
It will cover these sections (do not write them now): {outline.get('h2s')}

Write, in order:
  1. A 2-3 sentence lead paragraph (no heading) that hooks the reader, names the
     problem, and uses the primary keyword naturally in a human first-person voice.
  2. A <KeyTakeaways points={{["...", "...", "..."]}} /> with exactly 3-4 short
     skimmable points (each under 16 words).

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
    # Tell this section what the OTHER sections cover, so it stays in its lane and
    # doesn't repeat their thesis (sections are generated independently/blind).
    others = [h for h in (outline.get("h2s") or []) if h != h2]
    others_line = "; ".join(others) if others else "(none)"
    extra = []
    if assignments.get("illustration"):
        ill = assignments["illustration"]
        extra.append(
            f"Include one {ill.get('type','FlowDiagram')} component here with REAL data "
            f"(purpose: {ill.get('purpose','illustrate the point')}). Use the exact syntax, e.g.\n"
            "  <FlowDiagram caption=\"...\" steps={[{ label: \"...\", sub: \"...\" }, ...]} />\n"
            "  <CompareDiagram caption=\"...\" columns={[{ title, tone: \"good\"|\"bad\", points: [...] }]} />\n"
            "  <BarChart caption=\"...\" unit=\"...\" data={[{ label, value }, ...]} />"
        )
    if assignments.get("link"):
        lk = assignments["link"]
        extra.append(f'Include exactly one internal markdown link: [{lk.get("anchor","see this")}]({lk.get("path","/contact")}).')
    if assignments.get("callout"):
        extra.append('You may add one <Callout variant="tip">...</Callout> if it genuinely helps.')
    extra_block = "\n".join(f"  - {e}" for e in extra) if extra else "  - (prose only for this section)"

    system = STUDIO_PERSONA + " You write in MDX. Right now you write only ONE section."
    user = f"""Write ONE section of a studio blog post in MDX.

STUDIO FACTS (ground claims in these; never invent numbers/clients; you may cite
these real projects if relevant: {outline.get('real_projects_to_cite')}):
{facts_block}

The post's primary keyword is "{state['primary_keyword']}". Weave in these semantic
terms only where natural: {outline.get('lsi_keywords')}.

Write the section under this exact H2 heading:
## {h2}

OTHER sections of this post (already being written separately) cover: {others_line}.
Stay strictly within YOUR heading's scope — do not restate their points or re-explain
the overall thesis; assume the reader has read them.

Then 130-200 words of body copy (aim for that length — concise, not padded; vary
sentence length; use **bold**, a bullet list, or a > blockquote where it helps —
human, specific, not generic). Requirements:
{extra_block}

{_MDX_RULES}

Start with the "## {h2}" line and output only this one section."""
    return system, user


def section_closing_prompt(facts_block: str, state: dict) -> tuple[str, str]:
    """FAQ + BlogCTA."""
    outline = state["outline"]
    system = STUDIO_PERSONA + " You write in MDX. Right now you write only the closing."
    user = f"""Write ONLY the closing of a studio blog post in MDX.

The post is about "{state['primary_keyword']}" ({state['angle']}). It already has an
intro and these sections: {outline.get('h2s')}.

STUDIO FACTS (ground answers in these; never invent):
{facts_block}

Write, in order:
  1. A <FAQ items={{[{{ q: "...", a: "..." }}, ...]}} /> with exactly 3-4 real questions
     people search about this topic, each with a self-contained 1-2 sentence answer.
  2. A <BlogCTA /> on its own line (optionally with a short text="..." that invites
     the reader to describe their project / get a free prototype).

Keep it tight — the FAQ answers should be brief.

{_MDX_RULES}

Output only the FAQ component then the BlogCTA."""
    return system, user


# ── Node: fact-check guard ──
def factcheck_prompt(facts_block: str, body_mdx: str) -> tuple[str, str]:
    system = (
        "You are a meticulous editor checking a draft for factual claims that are "
        "not supported by the provided source facts. You care about truthfulness "
        "and brand safety."
    )
    user = f"""SOURCE FACTS (the only things known to be true about the studio):

{facts_block}

DRAFT:
'''
{body_mdx}
'''

Find every specific factual claim in the draft that is NOT supported by the source
facts — invented numbers, statistics, client names, sectors, dates, project claims,
or any "we did X" that the facts don't back up. General industry statements and
clearly qualitative language are fine; only flag concrete unsupported claims.

Reply as JSON:
{{"issues": [ {{"quote": "the exact phrase from the draft", "problem": "why it's unsupported", "fix": "how to reword truthfully"}} ]}}
If nothing is unsupported, return {{"issues": []}}."""
    return system, user


# ── Node: humanizer / critic ──
# Returns the revised MDX as plain text (NOT wrapped in JSON) — round-tripping a
# 1500-word body through a JSON string field is fragile (escaping/truncation). We
# read the score from a trailing marker line instead.
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


# ── Node: registry builder ──
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
  "title": string (50-60 chars, includes the primary keyword, compelling),
  "description": string (140-160 chars, includes the keyword and a concrete benefit),
  "tags": [2-4 Title Case tags]
}}"""
    return system, user
