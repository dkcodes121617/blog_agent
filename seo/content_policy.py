"""Editorial policy: no prices, no delivery durations, ever.

WizCodes sells custom software on the promise that it is affordable and fast *because*
the studio builds with AI. A blog that answers "what does custom software cost" with a
$45,000 range and "how long does it take" with 12-24 weeks argues the opposite case, in
writing, on the studio's own domain — and it attracts exactly the visitor who is price
shopping rather than the one with a problem to solve. Over twenty published posts that
shape had become most of what the agent wrote: nine of them are built end to end on
figures, and 545 separate price/duration claims shipped, including inside chart labels
where nobody thinks to look.

Prompting alone does not hold this line. The strategist was already told "cost and
pricing are only ONE option among many" and produced four cost breakdowns in a row
anyway, which is what the archetype-rotation machinery was built to work around. So the
rule lives in code, on three levels:

  1. TOPIC   — a proposed keyword or angle about money or duration is rejected outright
               and the strategist re-picks. Cheapest possible place to catch it.
  2. BODY    — concrete figures found in a finished draft become fix_claims issues, so
               they are edited out surgically by the loop that already exists for
               unsupported claims.
  3. SHIP    — wizcodes_next/scripts/check-content-policy.mjs re-checks registry, MDX
               and rendered SVG text at build time. That one holds even if the agent is
               bypassed entirely, which is why it exists separately rather than being
               imported from here.

The pattern lists below are kept in the same order as their JavaScript counterparts so
the two can be diffed by eye.

Deliberately narrow on prose. "The cost of getting this wrong is rework" stays legal: it
carries no figure and makes no price claim. What is banned is the NUMBER — the thing a
reader anchors to and an answer engine quotes back.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ── BLOCK: concrete figures ─────────────────────────────────────────────────

_MONEY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # The trailing group is not needed to DETECT the amount — `[$]\d` would do — but it
    # is needed to report it. Matching only the first digit made every error read
    # "remove the currency amount '$4'", which is not a string anyone can find in the
    # draft, and fix_claims is asked to edit by quotation.
    # `[ ]?` rather than `\s?`: `\s` matches a NEWLINE, so a BarChart with unit="$" on
    # one line and `value: 24000` on the next reported as the currency amount "$\n24000".
    # It is a real violation, but the quoted string cannot be found in the draft, and
    # fix_claims edits by quotation. The split case is caught properly by
    # _joined_measures() below, which reassembles the pair as a reader would see it.
    (re.compile(r"[$£€₹¥][ ]?[\d,.]+[ ]?[kKmMbB]?"), "currency amount"),
    (re.compile(r"\b\d[\d,.]*\s*[kKmM]?\s*(?:USD|INR|EUR|GBP|dollars?|rupees?|lakhs?|crores?)\b"),
     "currency amount"),
    (re.compile(r"\b(?:USD|INR|EUR|GBP)\s*\d"), "currency amount"),
    (re.compile(r"\b\d[\d,.]*\s*(?:per|/)\s*(?:seat|user|month|year|hour|head)\b", re.I), "unit price"),
]

# `\b\d+\s*(?:weeks?|months?)` deliberately excludes "years": "we have been shipping
# this for 3 years" is credibility, not an estimate. Years are caught only inside an
# explicit range, where they are nearly always a payback projection — a price claim
# wearing a hat.
_TIME_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b\d+\s*[-–—]\s*\d+\s*(?:hours?|days?|weeks?|months?|years?)\b", re.I), "duration range"),
    (re.compile(r"\b\d+\s*(?:hours?|days?|weeks?|months?)\b", re.I), "duration"),
    # "one day you want to hire your own team" is an idiom, not an estimate, and it
    # appears in the hand-written posts that are the voice this blog is copying. The
    # lookahead keeps the idiom legal while still catching "one day of work" and
    # "takes one day". Only `one` needs it — nobody writes "two days you will regret".
    (re.compile(
        r"\bone[\s-](?:hours?|days?|weeks?|months?)\b"
        # The punctuation is optional because the idiom usually arrives mid-sentence:
        # "bring the app in-house one day, the wider pool matters more".
        r"(?!\s*[,;:]?\s*(?:you|we|they|he|she|it|i|the|somebody|someone|everyone|that|this))",
        re.I), "duration"),
    (re.compile(
        r"\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|eighteen)"
        r"[\s-](?:hours?|days?|weeks?|months?)\b", re.I), "duration"),
    (re.compile(r"\b(?:week|month|day|quarter|sprint)\s*#?\d+\b", re.I), "timeline marker"),
    (re.compile(r"\bQ[1-4]\s*20\d\d\b"), "timeline marker"),
]

# ── BLOCK: framing ──────────────────────────────────────────────────────────
# Checked on the things that DECLARE a subject — a proposed keyword, a title, a
# description, a heading. A post whose headline is about cost is a cost post however
# carefully the figures are kept out of it, so at topic level this is fatal too.
_FRAMING_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:cost|costs|costly|pricing|price|prices|priced|budget|budgets|roi|"
                r"fees?|spend|spending|cheap|cheaper|cheapest|expensive|affordab\w+|"
                r"invoice|quote|quotes|rates?|salary|salaries)\b", re.I), "money framing"),
    (re.compile(r"\bhow much\b", re.I), "money framing"),
    (re.compile(r"\b(?:how long|timeline|timelines|turnaround|deadlines?|"
                r"time[-\s]to[-\s]\w+|delivery time|lead time|time frame|timeframe|"
                r"how fast|how quickly)\b", re.I), "time framing"),
]

# Things that legitimately carry digits and are not reader-facing claims.
_NOISE = [
    re.compile(r"\]\([^)]*\)"),        # markdown link targets
    re.compile(r"https?://\S+"),       # bare URLs
    re.compile(r"`[^`]*`"),            # inline code
    re.compile(r"\b20\d\d\b"),         # bare years ("in 2026")
]


@dataclass(frozen=True)
class Violation:
    level: str   # "block" | "framing"
    why: str     # "currency amount", "duration range", …
    text: str    # the matched string

    def __str__(self) -> str:
        return f"{self.why}: {self.text!r}"


def _strip_noise(text: str) -> str:
    out = text or ""
    for pattern in _NOISE:
        out = pattern.sub(" ", out)
    return out


def _scan(text: str, patterns, level: str) -> list[Violation]:
    """Match every pattern, keeping only the widest hit over any stretch of text.

    The patterns overlap by design — the duration-range pattern and the bare-duration
    pattern both match "12-24 weeks" — and reporting the same claim twice, once as
    "12-24 weeks" and once as "24 weeks", makes fix_claims chase an edit it has already
    made. Patterns are ordered widest-first within each list, so a later match that
    falls inside an earlier one is redundant.
    """
    found: list[Violation] = []
    spans: list[tuple[int, int]] = []
    for pattern, why in patterns:
        for m in pattern.finditer(text):
            start, end = m.span()
            if any(s <= start and end <= e for s, e in spans):
                continue
            spans.append((start, end))
            found.append(Violation(level, why, m.group(0).strip()))
    return found


def figure_violations(text: str) -> list[Violation]:
    """Concrete prices and durations. These are never publishable."""
    clean = _strip_noise(text)
    return _scan(clean, _MONEY_PATTERNS, "block") + _scan(clean, _TIME_PATTERNS, "block")


def framing_violations(text: str) -> list[Violation]:
    """Money/time SUBJECT words. Fatal in a title or a proposed keyword."""
    return _scan(_strip_noise(text), _FRAMING_PATTERNS, "framing")


def topic_violations(*parts: str) -> list[Violation]:
    """Everything that disqualifies a proposed topic, across keyword/angle/title/tags."""
    joined = " ".join(p for p in parts if p)
    return figure_violations(joined) + framing_violations(joined)


# A stat's magnitude and its unit are SEPARATE props, and the reader sees them joined.
#
#   { label: "Implementation", value: "12-24", unit: "weeks" }
#
# renders as "12-24 weeks" on the page and, enlarged, on the post's cover image — but
# the two halves never appear next to each other in the source, so a regex over the raw
# MDX finds neither a duration nor a price. This is not a hypothetical: that exact stat
# is live on the blog right now, and it is the string the owner pointed at.
_VALUE_UNIT = re.compile(
    r'value:\s*"([^"]*)"\s*,\s*unit:\s*"([^"]*)"'
    r'|unit:\s*"([^"]*)"\s*,\s*value:\s*"([^"]*)"')
# A BarChart declares one unit for the whole series, far from the numbers it applies to.
_CHART_UNIT = re.compile(r'unit=\{?"([^"]*)"\}?')
_CHART_VALUE = re.compile(r"value:\s*([\d.]+)")


def _render_measure(value: str, unit: str) -> str:
    """`value` and `unit` as the page prints them — currency leads, everything trails.

    Order matters to the scan, not just to the reader: a chart with unit="$" and
    value 24000 shows "$24,000", and the money patterns all expect the symbol first.
    Joining these in source order would produce "24000 $", which matches nothing.
    """
    u = (unit or "").strip()
    if not u:
        return value
    return f"{u}{value}" if _CURRENCY_LEADS.match(u) else f"{value} {u}"


_CURRENCY_LEADS = re.compile(r"^[$£€₹¥]")


def _joined_measures(mdx: str) -> str:
    """Every value/unit pair in the draft, rendered the way a reader will see it."""
    parts: list[str] = []
    for m in _VALUE_UNIT.finditer(mdx or ""):
        value = m.group(1) if m.group(1) is not None else m.group(4)
        unit = m.group(2) if m.group(2) is not None else m.group(3)
        if value and unit:
            parts.append(_render_measure(value, unit))
    for unit in _CHART_UNIT.findall(mdx or ""):
        if not unit.strip():
            continue
        for value in _CHART_VALUE.findall(mdx or ""):
            parts.append(_render_measure(value, unit))
    return "\n".join(parts)


def body_violations(mdx: str) -> list[Violation]:
    """Figures anywhere in a finished draft, including inside component props.

    Component props are scanned too, and that is the point: the worst offenders were
    never in prose. "$1,450K" and "12-24 weeks" were StatGrid values and BarChart data,
    which prose-only checking reads straight past and which then get baked into the
    post's cover image and its alt text.
    """
    return figure_violations((mdx or "") + "\n" + _joined_measures(mdx))


def heading_violations(mdx: str) -> list[Violation]:
    """Money/time framing in H2s — a section titled 'What it costs' is a cost section."""
    out: list[Violation] = []
    for line in re.findall(r"^##\s+(.+)$", mdx or "", re.M):
        out.extend(framing_violations(line))
    return out


def describe(violations: list[Violation], limit: int = 8) -> str:
    """A short, de-duplicated summary for logs and for the fix-claims prompt."""
    seen: list[str] = []
    for v in violations:
        s = str(v)
        if s not in seen:
            seen.append(s)
    extra = len(seen) - limit
    body = "; ".join(seen[:limit])
    return f"{body} (+{extra} more)" if extra > 0 else body


# ── The rule, phrased for a prompt ──────────────────────────────────────────
# Kept here rather than in prompts/library.py so the enforcement and the instruction
# can never describe different rules. Phrased as an editorial standard, not as a
# compliance directive, because the proxy's injection guard refuses the latter.
POLICY_PROMPT_BLOCK = """EDITORIAL STANDARD — this blog does not publish prices or delivery durations.

The studio's whole position is that custom software is affordable and fast because we
build with AI. Naming a figure argues against that, and it attracts readers who are
comparing quotes rather than readers with a problem to solve. So:

  - No money figures at all. No currency amounts, no per-seat or per-month prices, no
    budget ranges, no ROI or payback calculations — in prose, in headings, in FAQ
    answers, and above all in diagram data, where they end up on the post's cover image.
  - No delivery durations. No "6-8 weeks", no "three months", no "Week 1 / Month 6"
    timeline markers, no turnaround estimates.
  - Do not write ABOUT cost or duration either. "What X costs", "How long X takes",
    "Is X worth the money" are not topics for this blog.

Write about the thing itself instead: how it works, what decides whether it fits, what
goes wrong, what to check, what changed recently. Compare options on capability, data
ownership, maintenance burden and fit — never on price or delivery time.

Qualitative statements are fine and often better: "cheaper to maintain than a rewrite"
carries the meaning without the number."""
