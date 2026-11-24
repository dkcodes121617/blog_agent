"""Run the website's own graphics checks on a post before it is pushed.

## Why this exists

From 14 Oct to 24 Nov 2026 every deploy of the site failed — eighteen in a row —
while this agent reported success on each post it pushed. The site's prebuild
runs `export-graphics.mjs` (fitcheck) and `test-covers.mjs`, which MEASURE every
diagram label as rendered; this agent's validator only counts characters, and
knows nothing about QuadrantMap axes or BarChart units at all. "Personalized
feed" is 17 characters, inside the 18 the validator allows, and still ran off
the right edge of the fifth box. One such label fails the build, the build
stops at the first failure, and every later post sat committed but not live.

The only check that agrees with the site is the site's. Its scripts import
nothing but node builtins, so they run in the clone this agent already has,
with no `npm ci`.

## What blocks a post

Only failures in graphics that belong to THIS post (`<slug>-*.svg`,
`<slug>-cover.svg`). A failure elsewhere means the site is already broken by
something else; refusing to publish would stall the blog without fixing it, so
that is logged loudly and the post goes ahead.

If node or the scripts are missing, the gate is skipped with a warning rather
than blocking: it is a safety net added on top of a pipeline that already
publishes, not a new way for it to stop.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("agent.site_gate")

SITE_CHECKS = ("scripts/export-graphics.mjs", "scripts/test-covers.mjs")
_TIMEOUT_SECONDS = 300
# Props whose text is drawn inside a graphic. Captions and links are rendered as
# HTML around the figure, so shortening them could never clear a fit failure.
_SKIP_PROPS = {"caption", "href", "src", "alt", "archetype", "accent", "tone"}
_PROP_LITERAL = re.compile(r'\b(\w+)\s*[:=]\s*"([^"\n]{2,300})"')
_SVG_NAME = re.compile(r"([a-z0-9][a-z0-9-]*\.svg)")
_QUOTED = re.compile(r'"([^"]{1,200})"')


class SiteGateError(RuntimeError):
    """The site's checks rejected this post's graphics."""

    def __init__(self, report: GateReport):
        self.report = report
        super().__init__("; ".join(report.summary_lines()[:4]))


@dataclass
class GateReport:
    ran: bool = False
    # svg filename -> the quoted strings the checks complained about
    own: dict[str, list[str]] = field(default_factory=dict)
    others: dict[str, list[str]] = field(default_factory=dict)
    raw: str = ""

    @property
    def blocking(self) -> bool:
        return bool(self.own)

    def fragments(self) -> list[str]:
        out: list[str] = []
        for strings in self.own.values():
            for s in strings:
                if s not in out:
                    out.append(s)
        return out

    def summary_lines(self) -> list[str]:
        return [f"{name}: {', '.join(repr(s) for s in strings)}"
                for name, strings in self.own.items()]


def run_checks(site_dir: Path | str, slug: str) -> GateReport:
    """Run the site's graphics checks in `site_dir` and attribute the failures."""
    site_dir = Path(site_dir)
    node = shutil.which("node")
    scripts = [s for s in SITE_CHECKS if (site_dir / s).exists()]
    if not node or not scripts:
        log.warning("site gate skipped: %s", "node not on PATH" if not node
                    else f"none of {SITE_CHECKS} exist in the site repo")
        return GateReport(ran=False)

    outputs: list[str] = []
    for script in scripts:
        try:
            proc = subprocess.run(
                [node, script], cwd=site_dir, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            log.warning("site gate: %s could not run (%s) - skipping it", script, e)
            continue
        if proc.returncode != 0:
            outputs.append(f"{script}:\n{proc.stdout}\n{proc.stderr}")

    report = GateReport(ran=True, raw="\n".join(outputs))
    for name, strings in parse_failures(report.raw).items():
        mine = name == f"{slug}-cover.svg" or name.startswith(f"{slug}-")
        (report.own if mine else report.others)[name] = strings
    if report.others:
        log.error("site gate: the site is ALREADY failing its graphics checks on "
                  "graphics this post did not add, so its deploy will fail anyway: %s",
                  "; ".join(sorted(report.others))[:400])
    return report


def parse_failures(output: str) -> dict[str, list[str]]:
    """{svg filename: [quoted strings under it]} from fitcheck / test-covers output.

    Both scripts print the failing file on one line and the offending text, in
    double quotes, on the indented lines after it.
    """
    found: dict[str, list[str]] = {}
    current = ""
    for line in output.splitlines():
        m = _SVG_NAME.search(line)
        if m:
            current = m.group(1)
            found.setdefault(current, [])
            continue
        if not current:
            continue
        for q in _QUOTED.findall(line):
            q = q.replace("…", "").strip()
            if q and q not in found[current]:
                found[current].append(q)
    return {k: v for k, v in found.items() if v}


def offending_literals(body: str, fragments: list[str]) -> list[str]:
    """The diagram prop values in `body` that a reported fragment points at.

    A fragment is what the renderer printed, which is not always the literal:
    a clipped label ("contacts" from "New column in contacts table"), or a value
    joined with its unit ("100 writes per 100 messages" from unit="writes per
    100 messages"). So match containment both ways.
    """
    out: list[str] = []
    for m in _PROP_LITERAL.finditer(body):
        prop, value = m.group(1), m.group(2)
        if prop in _SKIP_PROPS or value in out:
            continue
        for frag in fragments:
            if (len(frag) >= 4 and frag in value) or (len(value) >= 4 and value in frag):
                out.append(value)
                break
    return out


def replace_literals(body: str, mapping: dict[str, str]) -> str:
    """Swap prop values `prop: "old"` / `prop="old"` -> new. Never touches prose."""
    if not mapping:
        return body

    def repl(m: re.Match[str]) -> str:
        new = mapping.get(m.group(2))
        if new is None or m.group(1) in _SKIP_PROPS:
            return m.group(0)
        return m.group(0).replace(f'"{m.group(2)}"', '"' + new.replace('"', "'") + '"')

    return _PROP_LITERAL.sub(repl, body)


def repair(body: str, report: GateReport, llm=None) -> str | None:
    """Shorten the strings the site flagged. None when nothing could be mapped.

    Same contract as the validator's repair node: the model is asked for
    replacement STRINGS only and they are spliced in here, so nothing outside the
    flagged values can change. The target is a third shorter than what failed,
    because the renderer measured it as too wide, not as too many characters.
    """
    values = offending_literals(body, report.fragments())
    if not values:
        log.warning("site gate: could not map %s back to any diagram text",
                    report.fragments())
        return None

    items = [
        {"id": i + 1, "kind": "label", "current": v,
         "max_chars": max(6, int(len(v) * 0.65)),
         "what": "diagram label that overflowed its box when the site rendered it"}
        for i, v in enumerate(values)
    ]
    written: dict[int, str] = {}
    if llm is not None:
        try:
            from prompts.library import shorten_labels_prompt

            system, user = shorten_labels_prompt(items)
            data = llm.complete_json(system=system, user=user, max_tokens=800, attempts=2)
            rows = data if isinstance(data, list) else (data or {}).get("items", [])
            for row in rows:
                if isinstance(row, dict) and row.get("text"):
                    try:
                        written[int(row["id"])] = str(row["text"]).strip().rstrip(".…")
                    except (KeyError, TypeError, ValueError):
                        continue
        except Exception as e:  # noqa: BLE001 - any model failure falls back below
            log.warning("site gate: model shortening failed (%s) - using word cuts",
                        str(e)[:120])

    mapping: dict[str, str] = {}
    for it in items:
        new = written.get(it["id"], "")
        if not new or len(new) > it["max_chars"]:
            new = blunt_shorten(it["current"], it["max_chars"])
        if new and new != it["current"]:
            mapping[it["current"]] = new
    for old, new in mapping.items():
        log.info("  site gate repair: %r -> %r", old, new)
    return replace_literals(body, mapping) if mapping else None


def blunt_shorten(value: str, limit: int) -> str:
    """Word-boundary cut with no ellipsis - the fallback when the model is down."""
    if len(value) <= limit:
        return value
    words, out = value.split(), ""
    for w in words:
        candidate = f"{out} {w}".strip()
        if len(candidate) > limit:
            break
        out = candidate
    return (out or value[:limit]).rstrip(" ,;:-")
