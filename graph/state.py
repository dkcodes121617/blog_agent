"""Shared state for the blog-generation graph.

One TypedDict threaded through every node. LangGraph merges each node's returned
dict into this. Keeping it flat + explicit makes the pipeline easy to reason about
and to checkpoint/resume.
"""
from __future__ import annotations

from typing import Any, TypedDict


class BlogState(TypedDict, total=False):
    # ── grounding (set once at load) ──
    facts_block: str            # the real-WizCodes facts prompt block
    known_slugs: list[str]      # existing post slugs (uniqueness + link validation)

    # ── topic strategy (node: topic) ──
    primary_keyword: str
    angle: str                  # the specific take/thesis
    audience: str
    rationale: str              # why this topic ranks / is valuable

    # ── uniqueness ──
    topic_similarity: float     # max cosine of topic vs KB
    body_similarity: float      # max cosine of finished body vs KB
    similar_slug: str | None
    related_slugs: list[str]    # top related existing posts (for internal linking)

    # ── outline (node: outline) ──
    outline: dict[str, Any]     # {h2s: [...], lsi_keywords: [...], internal_links: [...]}

    # ── draft (nodes: write / factcheck / humanize) ──
    body_mdx: str               # the MDX body
    factcheck_issues: list[str]
    humanize_score: int         # 0-100 self-assessed human-ness
    critic_notes: list[str]

    # ── registry (node: registry) ──
    slug: str
    title: str
    description: str
    tags: list[str]
    reading_minutes: int
    date: str                   # ISO YYYY-MM-DD assigned at publish

    # ── validation ──
    validation_errors: list[str]
    validation_warnings: list[str]

    # ── control / bookkeeping ──
    revision: int               # how many write→validate loops so far
    fix_attempts: int           # how many surgical fact-check fixes so far
    topic_attempts: int         # how many topic re-picks so far
    status: str                 # running | aborted | ready | published
    abort_reason: str
    published_path: str
