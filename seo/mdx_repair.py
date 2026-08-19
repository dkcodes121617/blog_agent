"""Surgical, convergent repair of MDX contract violations.

The writer produces a post in one pass per section, and until now the only answer
to a validation error was to write the whole post AGAIN. That never converged:
a rewrite regenerates every diagram, so it fixes the flagged label and clips a
different one. Real runs went 2 errors -> 4, 3 -> 5, 8 -> 2, and every one of them
burned the revision budget and aborted without publishing.

The fix is to stop regenerating and start EDITING. Two classes of violation cover
almost every real failure, and both are edits to a single string:

  - a diagram label/title/sub/outcome longer than its box
  - a visual with no caption=

This module does those edits deterministically. The node in graph/nodes wraps it
with one cheap LLM call that writes better replacements than a truncator can, and
falls back to here for anything the model leaves over-long — so the pass has a
guaranteed fixed point instead of a budget it can exhaust.

Anything structural (too few H2s, an unknown component, a malformed BarChart) is
NOT repaired here; those genuinely need the writer, and the graph still routes
them to a rewrite.
"""
from __future__ import annotations

import re

from seo.mdx_validator import (
    COMPONENT_ITEM_KEYS,
    COMPONENT_PROPS,
    DIAGRAM_TEXT_LIMITS,
    diagram_text_pattern,
    overlong_diagram_text,
    uncaptioned_visuals,
)

# Compactions applied before truncation, longest-first so "without" beats "with".
# Every one of these is a form a human editor would actually write inside a
# diagram box — the point is to lose characters, not meaning.
_COMPACTIONS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pat, re.IGNORECASE), rep) for pat, rep in (
        (r"\bapproximately\b", "~"),
        (r"\bwithout\b", "w/o"),
        (r"\bversus\b", "vs"),
        (r"\bpercent\b", "%"),
        (r"\bwith\b", "w/"),
        (r"\bunder\b", "<"),
        (r"\bover\b", ">"),
        (r"\band\b", "&"),
        (r"\bseconds?\b", "s"),
        (r"\bminutes?\b", "min"),
        (r"\bhours?\b", "hr"),
        (r"\bweeks?\b", "wk"),
        (r"\bmonths?\b", "mo"),
        (r"\bdevelopers?\b", "devs"),
        (r"\bapplications?\b", "apps"),
        (r"\benvironments?\b", "envs"),
    )
)

_LEADING_FILLER = re.compile(r"^(the|a|an|your|our|this|that)\s+", re.IGNORECASE)
_TRAILING_JUNK = re.compile(r"[\s,;:.\-–—]+$")

# A sentence end: terminator + space, not preceded by a digit. The digit guard keeps
# "$1.5M and up" and "under 1. 5x" from being read as two sentences.
_SENTENCE_END = re.compile(r"(?<=[^\d])([.!?])\s+")


def _first_sentences(text: str, limit: int) -> str | None:
    """The longest whole-sentence prefix of `text` that fits `limit`, if any.

    Only <Timeline> descriptions are long enough to contain a sentence break, and for
    them this is the difference between "Django ships an admin panel." and "Django ships
    an admin panel. FastAPI needs a custom build or a third-party" — a complete thought
    versus a severed one. Labels have no terminators, so this never fires on them.
    """
    best = None
    for m in _SENTENCE_END.finditer(text):
        end = m.end(1)  # keep the terminator, drop the space
        if end > limit:
            break
        best = text[:end]
    return best


def shorten(value: str, limit: int) -> str:
    """Fit `value` into `limit` characters, preferring compaction over truncation.

    Deterministic and total: the return value is always <= limit, which is what
    makes this usable as the last line of defence. No ellipsis — a diagram label
    ending in "…" reads as a rendering bug, not as an abbreviation.
    """
    text = re.sub(r"\s+", " ", value).strip()
    if len(text) <= limit:
        return text

    # Dropping a whole trailing sentence beats compacting every word in the ones that
    # remain, so this is tried before the abbreviations rather than after them.
    sentences = _first_sentences(text, limit)
    if sentences:
        return sentences

    for pattern, replacement in _COMPACTIONS:
        if len(text) <= limit:
            break
        text = re.sub(r"\s+", " ", pattern.sub(replacement, text)).strip()
        # "< 2 s" reads worse than "<2s", and those spaces are characters we need.
        text = re.sub(r"([<>~])\s+", r"\1", text)
        text = re.sub(r"(\d)\s+(s|min|hr|wk|mo)\b", r"\1\2", text)
    if len(text) <= limit:
        return text

    while len(text) > limit and _LEADING_FILLER.search(text):
        text = _LEADING_FILLER.sub("", text, count=1)
    if len(text) <= limit:
        return text

    # Truncate on a word boundary; only slice mid-word if the first word alone
    # already overflows.
    cut = text[:limit]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    cut = _TRAILING_JUNK.sub("", cut)
    return cut or text[:limit]


def apply_shortenings(text: str, mapping: dict[str, str]) -> str:
    """Replace diagram prop values in place, `prop: "old"` -> `prop: "new"`.

    Substitution is anchored on the same pattern the validator matches, so this
    can only touch strings that were actually flagged — never prose, never a
    caption, never a value that happens to be identical somewhere harmless.
    """
    if not mapping:
        return text

    def repl(m: re.Match[str]) -> str:
        new = mapping.get(m.group(1))
        if new is None:
            return m.group(0)
        return m.group(0).replace(f'"{m.group(1)}"', '"' + new.replace('"', "'") + '"')

    for prop, _limit, _what in DIAGRAM_TEXT_LIMITS:
        text = diagram_text_pattern(prop).sub(repl, text)
    return text


def section_heading_before(text: str, offset: int) -> str:
    """The nearest `## …` heading above `offset` — context for a caption."""
    heads = [m.group(1).strip() for m in re.finditer(r"^##\s+(.+)$", text[:offset], re.MULTILINE)]
    return heads[-1] if heads else ""


_COMPONENT_CAPTION_NOUN = {
    "FlowDiagram": "the process step by step",
    "CompareDiagram": "the options side by side",
    "BarChart": "the numbers compared",
    "Figure": "the detail referenced above",
    "StatGrid": "the figures at a glance",
    "Timeline": "how it unfolds over time",
    "DecisionTree": "which path to take",
    "ConceptDiagram": "how the pieces fit together",
    "QuadrantMap": "where each option sits",
}


def caption_for(component: str, heading: str) -> str:
    """A plain, honest caption derived from the section it sits in.

    Deliberately descriptive rather than clever: this is the fallback for when the
    model didn't supply one, and a caption that describes the wrong thing is worse
    for answer engines than a plain one that describes the right thing.
    """
    noun = _COMPONENT_CAPTION_NOUN.get(component, "the detail above")
    heading = re.sub(r"[\"“”]", "", heading).strip().rstrip("?:.")
    text = f"{heading}: {noun}" if heading else noun.capitalize()
    return text[:120]


def insert_captions(text: str, targets: list[tuple[str, int]],
                    texts: list[str | None] | None = None) -> str:
    """Add `caption="…"` to each visual in `targets` (component, offset).

    `texts` supplies a written caption per target where one is available; any gap
    falls back to one derived from the surrounding section. Applied right-to-left
    so each insertion leaves the earlier offsets valid.
    """
    supplied = list(texts or [])
    supplied += [None] * (len(targets) - len(supplied))
    for (component, offset), written in sorted(
        zip(targets, supplied), key=lambda pair: -pair[0][1]
    ):
        anchor = offset + len(component) + 1  # just past "<Component"
        caption = clean_caption(written) or caption_for(
            component, section_heading_before(text, offset))
        text = f'{text[:anchor]} caption="{caption}"{text[anchor:]}'
    return text


def clean_caption(value: str | None) -> str:
    """A caption safe to drop into a JSX double-quoted prop."""
    if not value:
        return ""
    text = re.sub(r"\s+", " ", value).replace('"', "'").strip()
    return text[:120].strip()


def _resolve_collisions(overlong: list[tuple[str, str, int, str]],
                        mapping: dict[str, str]) -> dict[str, str]:
    """Keep labels that were distinct before shortening distinct after it.

    Truncation keeps the head of a string, and comparison labels put the thing
    being compared at the END — "Doc processing 2M tok/mo - OpenAI" and the same
    line for Anthropic and Google all truncate to one identical label, which turns
    a three-bar chart into three bars with the same name. Where that happens the
    distinguishing last word is preserved and the head is compressed to fit around it.
    """
    limits = {value: limit for _prop, value, limit, _what in overlong}
    grouped: dict[str, set[str]] = {}
    for value in limits:
        grouped.setdefault(mapping[value], set()).add(value)

    for shortened, originals in grouped.items():
        if len(originals) < 2:
            continue
        for original in originals:
            limit = limits[original]
            words = original.split()
            if len(words) < 2:
                continue
            tail = words[-1]
            budget = limit - len(tail) - 1
            if budget < 3:
                continue
            candidate = f"{shorten(' '.join(words[:-1]), budget)} {tail}".strip()
            if len(candidate) <= limit:
                mapping[original] = candidate
    return mapping


def repair_deterministic(text: str) -> tuple[str, list[str]]:
    """Fix every mechanically-fixable violation. Returns (text, notes).

    Idempotent and guaranteed to clear the two error classes it covers, which is
    the property the pipeline needs: after this runs, no run can still be blocked
    by an over-long label or a missing caption.
    """
    notes: list[str] = []

    overlong = overlong_diagram_text(text)
    if overlong:
        mapping = {value: shorten(value, limit) for _prop, value, limit, _what in overlong}
        mapping = _resolve_collisions(overlong, mapping)
        text = apply_shortenings(text, mapping)
        notes += [f'shortened {what}: "{old}" -> "{mapping[old]}"'
                  for _prop, old, _limit, what in overlong]

    missing = uncaptioned_visuals(text)
    if missing:
        text = insert_captions(text, missing)
        notes += [f"added a caption to <{comp}>" for comp, _ in missing]

    text, renamed = rename_component_props(text)
    notes += renamed

    return text, notes


def rename_component_props(text: str) -> tuple[str, list[str]]:
    """Rename invented component props onto the ones the site actually reads.

    A wrong prop name is the single cheapest error in the pipeline to fix and was the
    most expensive to ignore: `<StatGrid data={…}>` renders nothing, and before the site
    grew guards it crashed the static export and blocked every deploy for three days.

    It is fixed HERE, mechanically, rather than by asking the model to rewrite the post.
    The edit is a rename — `data=` to `stats=`, `title:` to `label:` — with no judgement
    in it, so spending a model call on it would be slower, cost a retry budget, and
    risk the rewrite changing something else in passing.

    Only renames a prop the component does NOT already have. A component carrying both
    `stats=` and `data=` is a different problem, and blindly renaming would produce two
    `stats=` props and invalid MDX.
    """
    notes: list[str] = []

    for name, spec in COMPONENT_PROPS.items():
        required = spec["required"]
        for alias in spec["aliases"]:
            def _sub(m: "re.Match[str]") -> str:
                attrs = m.group(2)
                if re.search(rf"\b{required}\s*=", attrs):
                    return m.group(0)
                if not re.search(rf"\b{alias}\s*=", attrs):
                    return m.group(0)
                notes.append(f"<{name}>: renamed `{alias}=` to `{required}=`")
                return m.group(1) + re.sub(rf"\b{alias}(\s*=)", rf"{required}\1", attrs, count=1) + m.group(3)

            text = re.sub(rf"(<{name}\b)((?:[^>\"']|\"[^\"]*\"|'[^']*')*?)(/?>)", _sub, text, flags=re.S)

    for name, (key, wrong_keys) in COMPONENT_ITEM_KEYS.items():
        for wrong in wrong_keys:
            def _sub_item(m: "re.Match[str]") -> str:
                block = m.group(2)
                if re.search(rf"\b{key}\s*:", block) or not re.search(rf"\b{wrong}\s*:", block):
                    return m.group(0)
                notes.append(f"<{name}> entries: renamed `{wrong}:` to `{key}:`")
                return m.group(1) + re.sub(rf"\b{wrong}(\s*:)", rf"{key}\1", block) + m.group(3)

            text = re.sub(rf"(<{name}\b)(.*?)(/>)", _sub_item, text, flags=re.S)

    return text, notes


def repairable(errors: list[str]) -> bool:
    """Whether the surgical pass can plausibly clear these errors.

    Used by the router: routing a "needs 4-8 H2 sections" error to a string
    editor would spend a budget on something it structurally cannot fix.

    Prop renames and editorial-policy figures are both listed because both are
    string-level edits, and the alternative route is a full rewrite that this module's
    header documents as non-convergent. A rewrite to remove one price regenerates every
    section and every diagram, and the new draft reliably arrives with a different price
    in a different place.
    """
    marks = (
        "too long for its box",
        "has no caption",
        "rename the prop",
        "rename the key",
        "editorial policy",
    )
    return any(any(m in e for m in marks) for e in errors)
