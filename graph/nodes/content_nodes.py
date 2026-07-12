"""The LangGraph nodes. Each is a pure-ish function: BlogState in, partial dict out.

Flow (assembled in graph/build.py):

  load_context -> pick_topic -> [uniqueness gate] -> outline -> write
    -> factcheck -> [validate_mdx] -> humanize -> build_registry
    -> [final uniqueness] -> finalize

Conditional edges (in build.py) implement the self-correcting loops:
  - topic too similar  -> back to pick_topic (up to N)
  - validation errors  -> back to write (up to N)
  - factcheck issues   -> back to write (up to N)
  - final near-dup     -> abort (publish nothing)
"""
from __future__ import annotations

import logging

from config import CONFIG
from graph.state import BlogState
from knowledge.store import KnowledgeBase
from llm.client import LLMClient, LLMError, LLMTransient
from llm.sanitize import sanitize_prose
from prompts import library as P
from seo.mdx_validator import reading_minutes, validate_mdx

log = logging.getLogger("agent.nodes")

MAX_TOPIC_ATTEMPTS = 4
MAX_REVISIONS = 2          # full rewrites for MDX-contract errors (rare)
FIXCLAIMS_BUDGET = 2      # surgical fact-check fixes before we ship a valid draft
HUMANIZE_MIN_SCORE = 70


class Nodes:
    """Bundles the shared clients so nodes can be plain methods."""

    def __init__(self, facts_block: str, kb: KnowledgeBase, llm: LLMClient | None = None):
        self.facts_block = facts_block
        self.kb = kb
        self.llm = llm or LLMClient()

    # ── load ──
    def load_context(self, state: BlogState) -> dict:
        log.info("node: load_context")
        return {
            "facts_block": self.facts_block,
            "known_slugs": self.kb.all_slugs(),
            "revision": 0,
            "topic_attempts": 0,
            "status": "running",
        }

    # ── topic ──
    def pick_topic(self, state: BlogState) -> dict:
        attempts = state.get("topic_attempts", 0) + 1
        log.info("node: pick_topic (attempt %d)", attempts)
        recent = state.get("known_slugs", [])
        # Include the just-rejected topic so the model doesn't re-propose it.
        if state.get("primary_keyword"):
            recent = recent + [state["primary_keyword"]]
        system, user = P.topic_prompt(self.facts_block, recent)
        data = self.llm.complete_json(system=system, user=user, max_tokens=1000)
        return {
            "primary_keyword": data.get("primary_keyword", "").strip(),
            "angle": data.get("angle", "").strip(),
            "audience": data.get("audience", "").strip(),
            "rationale": data.get("rationale", "").strip(),
            "topic_attempts": attempts,
        }

    def check_topic_uniqueness(self, state: BlogState) -> dict:
        probe = f"{state['primary_keyword']}. {state['angle']}"
        sim, slug = self.kb.max_similarity(probe)
        related = [s for s, _ in self.kb.top_related(probe, k=3)]
        log.info("node: check_topic_uniqueness sim=%.3f vs %s", sim, slug)
        return {"topic_similarity": sim, "similar_slug": slug, "related_slugs": related}

    # ── outline ──
    def outline(self, state: BlogState) -> dict:
        log.info("node: outline for %r", state["primary_keyword"])
        system, user = P.outline_prompt(
            self.facts_block, state["primary_keyword"], state["angle"],
            state.get("audience", ""), state.get("related_slugs", []),
        )
        data = self.llm.complete_json(system=system, user=user, max_tokens=900)
        return {"outline": data}

    # ── write (SECTIONED — many short calls instead of one long one) ──
    def write(self, state: BlogState) -> dict:
        rev = state.get("revision", 0) + 1
        outline = state["outline"]
        h2s = [h for h in (outline.get("h2s") or []) if isinstance(h, str) and h.strip()][:5]
        if not h2s:
            h2s = ["How it works", "What to watch for", "Getting started"]
        log.info("node: write (revision %d) — sectioned, %d sections", rev, len(h2s))

        # Rewrite feedback (if we looped back) — appended to every chunk prompt.
        feedback = ""
        if state.get("validation_errors"):
            feedback += "\n\nAvoid these problems from the last draft:\n" + "\n".join(state["validation_errors"])
        if state.get("factcheck_issues"):
            feedback += "\n\nDo NOT make these unsupported claims:\n" + "\n".join(state["factcheck_issues"])

        assignments = self._assign_sections(outline, h2s)
        parts: list[str] = []

        # 1) intro (lead + KeyTakeaways)
        parts.append(self._chunk(*P.section_intro_prompt(self.facts_block, state),
                                 feedback, max_tokens=800, label="intro"))
        # 2) one call per H2 section
        for i, h2 in enumerate(h2s):
            sysp, usrp = P.section_body_prompt(self.facts_block, state, h2, assignments[i])
            parts.append(self._chunk(sysp, usrp, feedback, max_tokens=1400, label=f"h2[{i}]"))
        # 3) closing (FAQ + BlogCTA)
        parts.append(self._chunk(*P.section_closing_prompt(self.facts_block, state),
                                 feedback, max_tokens=1200, label="closing"))

        body = "\n\n".join(p.strip() for p in parts if p.strip())
        return {"body_mdx": sanitize_prose(body), "revision": rev,
                "validation_errors": [], "factcheck_issues": []}

    def _chunk(self, system: str, user: str, feedback: str, *, max_tokens: int, label: str) -> str:
        """Generate one short section; sanitize its fences/encoding. Short calls are
        far less likely to hit the proxy's 502/timeout window, and each retries
        independently, so a bad moment costs one section, not the whole article."""
        if feedback:
            user = user + feedback
        raw = self.llm.complete(system=system, user=user, max_tokens=max_tokens, temperature=0.8)
        return sanitize_prose(raw)

    @staticmethod
    def _assign_sections(outline: dict, h2s: list[str]) -> list[dict]:
        """Distribute the illustration + internal links across the H2 sections so
        each section gets at most one of each (keeps every call small and focused)."""
        n = len(h2s)
        assignments = [dict() for _ in range(n)]
        # Illustration → the second section if it exists, else the first.
        if outline.get("illustration"):
            assignments[min(1, n - 1)]["illustration"] = outline["illustration"]
        # Internal links → one per section, in order.
        links = [l for l in (outline.get("internal_links") or []) if isinstance(l, dict) and l.get("path")]
        for i, link in enumerate(links):
            assignments[i % n].setdefault("link", link)
        # A single tip callout on the middle section for texture.
        if n >= 2:
            assignments[n // 2]["callout"] = True
        return assignments

    # ── factcheck ──
    def factcheck(self, state: BlogState) -> dict:
        log.info("node: factcheck")
        system, user = P.factcheck_prompt(self.facts_block, state["body_mdx"])
        data = self.llm.complete_json(system=system, user=user, max_tokens=1200)
        issues = data.get("issues", []) if isinstance(data, dict) else []
        formatted = [f"{i.get('quote','')} -> {i.get('fix','')}" for i in issues]
        log.info("  factcheck found %d issue(s)", len(formatted))
        return {"factcheck_issues": formatted}

    # ── surgical fact-check fix ──
    # Remove ONLY the flagged claims, in one fast call. Converges (unlike a full
    # rewrite, which introduces fresh claims). Best-effort: on a proxy error or a
    # revision that breaks validation, keep the prior (valid) body.
    def fix_claims(self, state: BlogState) -> dict:
        attempts = state.get("fix_attempts", 0) + 1
        issues = state.get("factcheck_issues", [])
        log.info("node: fix_claims (attempt %d, %d issue(s))", attempts, len(issues))
        try:
            system, user = P.fix_claims_prompt(state["body_mdx"], issues)
            raw = self.llm.complete(system=system, user=user, max_tokens=4000, temperature=0.3)
        except (LLMError, LLMTransient) as e:
            log.warning("  fix_claims proxy error (%s); keeping prior body", e)
            return {"fix_attempts": attempts}
        out: dict = {"fix_attempts": attempts, "factcheck_issues": []}
        candidate = sanitize_prose(raw)
        report = validate_mdx(candidate, known_slugs=set(state.get("known_slugs", [])))
        if report.ok and len(candidate) > 400:
            out["body_mdx"] = candidate
        else:
            log.info("  fix_claims result failed validation; keeping prior body")
        return out

    # ── validate (deterministic) ──
    def validate(self, state: BlogState) -> dict:
        log.info("node: validate")
        report = validate_mdx(state["body_mdx"], known_slugs=set(state.get("known_slugs", [])))
        log.info("  validation ok=%s errors=%d warnings=%d",
                 report.ok, len(report.errors), len(report.warnings))
        return {"validation_errors": report.errors, "validation_warnings": report.warnings}

    # ── humanize ──
    # Best-effort polish. This is a longer (whole-body) call, so it's the most
    # 502-prone step — but the sectioned body is ALREADY valid and human, so a
    # humanize failure must NOT sink the run. On any error we keep the prior body.
    def humanize(self, state: BlogState) -> dict:
        log.info("node: humanize (best-effort)")
        try:
            system, user = P.humanize_prompt(state["body_mdx"])
            raw = self.llm.complete(system=system, user=user, max_tokens=4000, temperature=0.7)
        except (LLMError, LLMTransient) as e:
            log.warning("  humanize skipped (proxy error: %s); keeping sectioned body", e)
            return {"humanize_score": 0, "critic_notes": ["humanize skipped"]}
        # Extract the trailing score marker, then strip it out of the body.
        score = 0
        m = re.search(r"<!--\s*HUMANSCORE:\s*(\d+)\s*-->", raw)
        if m:
            score = int(m.group(1))
            raw = raw[: m.start()] + raw[m.end():]
        out: dict = {"humanize_score": score, "critic_notes": []}
        candidate = sanitize_prose(raw)
        # Only accept the revision if it still passes validation (never regress).
        report = validate_mdx(candidate, known_slugs=set(state.get("known_slugs", [])))
        if report.ok and len(candidate) > 400:
            out["body_mdx"] = candidate
        else:
            log.info("  humanize revision failed validation; keeping prior body")
        return out

    # ── registry ──
    def build_registry(self, state: BlogState) -> dict:
        log.info("node: build_registry")
        system, user = P.registry_prompt(
            state["body_mdx"], state["primary_keyword"], state.get("known_slugs", [])
        )
        data = self.llm.complete_json(system=system, user=user, max_tokens=700)
        slug = _slugify(data.get("slug", state["primary_keyword"]))
        # Guarantee uniqueness against the registry.
        if slug in state.get("known_slugs", []):
            slug = _dedupe_slug(slug, state.get("known_slugs", []))
        return {
            "slug": slug,
            "title": data.get("title", "").strip(),
            "description": data.get("description", "").strip(),
            "tags": data.get("tags", [])[:4],
            "reading_minutes": reading_minutes(state["body_mdx"]),
        }

    # ── final uniqueness ──
    def final_uniqueness(self, state: BlogState) -> dict:
        combined = f"{state['title']}\n{state['description']}\n{state['body_mdx']}"
        sim, slug = self.kb.max_similarity(combined)
        log.info("node: final_uniqueness sim=%.3f vs %s", sim, slug)
        return {"body_similarity": sim, "similar_slug": slug}

    def finalize(self, state: BlogState) -> dict:
        log.info("node: finalize slug=%s", state.get("slug"))
        return {"status": "ready"}


# ── helpers ──
def _slugify(text: str) -> str:
    import re
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:70].strip("-")


def _dedupe_slug(slug: str, taken: list[str]) -> str:
    i = 2
    base = slug
    while f"{base}-{i}" in taken:
        i += 1
    return f"{base}-{i}"
