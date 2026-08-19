"""Topic strategy: archetypes, rotation, and the prompts built from them.

The blog's failure mode was never a broken build — it was converging on one shape and
one subject until half the catalogue was the same post about money. These tests pin the
machinery that stops that, including the parts that quietly stopped working when the
archetype names changed.
"""
from __future__ import annotations

import pytest

import prompts.library as P
from seo import content_policy as cp


# ── The set itself ───────────────────────────────────────────────────────────

def test_no_retired_archetype_survives_in_the_live_set():
    assert "cost_breakdown" not in P.ARCHETYPES
    assert "decision_framework" not in P.ARCHETYPES
    assert P.DEFAULT_ARCHETYPE in P.ARCHETYPES


def test_every_archetype_is_fully_wired():
    """A shape with no scaffold, illustration or hook silently falls back to another.

    Adding an archetype to the tuple and forgetting one of the four tables is the
    obvious way to break this, and the symptom — posts of the new shape reading like
    the default one — is invisible without reading the output closely.
    """
    for name in P.ARCHETYPES:
        assert name in P._H2_PATTERNS, f"{name} has no H2 scaffold"
        assert name in P._PRIMARY_ILLUSTRATION, f"{name} has no primary illustration"
        assert name in P._SECONDARY_ILLUSTRATION, f"{name} has no secondary illustration"
        assert name in P._INTRO_HOOK, f"{name} has no intro hook"


def test_no_archetype_scaffold_asks_for_a_price_or_a_duration():
    for name, headings in P._H2_PATTERNS.items():
        for h in headings:
            assert not cp.framing_violations(h), f"{name} scaffold heading: {h!r}"


def test_timeline_is_never_commissioned():
    """Timeline is where "Month 0 / Month 6" kept coming from.

    The component still exists for the posts that already use it; it is simply no
    longer offered to the writer, so a duration cannot be requested by construction.
    """
    assert "Timeline" not in P._PRIMARY_ILLUSTRATION.values()
    assert "Timeline" not in P._SECONDARY_ILLUSTRATION.values()


# ── Rotation ─────────────────────────────────────────────────────────────────

def test_rotation_reads_legacy_archetypes_from_the_registry():
    """Twenty published posts record names that no longer exist.

    Without the legacy mapping, blocked_archetypes() matches none of them, concludes
    nothing is over-used, and the rotation stops working on the first run after the
    rename — the exact failure it was built to prevent, reintroduced by a rename.
    """
    recent = [
        {"archetype": "cost_breakdown", "title": "What X costs"},
        {"archetype": "cost_breakdown", "title": "What Y costs"},
        {"archetype": "vs_comparison", "title": "A vs B"},
        {"archetype": "decision_framework", "title": "How to choose"},
    ]
    blocked = P.blocked_archetypes(recent)
    assert "explainer" in blocked, "two mapped cost_breakdowns should block explainer"


def test_rotation_never_blocks_everything():
    recent = [{"archetype": a, "title": "t"} for a in P.ARCHETYPES for _ in (0, 1)]
    blocked = P.blocked_archetypes(recent, lookback=len(recent))
    assert len(blocked) < len(P.ARCHETYPES)


def test_rotation_blocks_the_immediately_previous_shape():
    recent = [{"archetype": "tips_list", "title": "7 checks"}]
    assert P.blocked_archetypes(recent) == ["tips_list"]


def test_no_recent_posts_blocks_nothing():
    assert P.blocked_archetypes([]) == []


# ── Inference ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("title,expected", [
    ("What Most Teams Get Wrong About Data Models", "mistake_guide"),
    ("Seven Checks Before You Ship An AI Agent", "tips_list"),
    ("Postgres vs MongoDB For A Changing Schema", "vs_comparison"),
    ("How To Choose A Backend For A Product That Grows", "playbook"),
    ("How A Custom CRM Actually Routes A Lead", "explainer"),
])
def test_archetype_inference(title, expected):
    assert P.infer_archetype(title) == expected


def test_inference_always_returns_a_live_archetype():
    for title in ["", "A completely unclassifiable headline", "Zzz"]:
        assert P.infer_archetype(title) in P.ARCHETYPES


def test_normalise_maps_retired_names_onto_live_ones():
    assert P.normalise_archetype("cost_breakdown") in P.ARCHETYPES
    assert P.normalise_archetype("decision_framework") in P.ARCHETYPES
    assert P.normalise_archetype("explainer") == "explainer"


# ── The prompts ──────────────────────────────────────────────────────────────

def test_every_prompt_builder_runs_for_every_archetype():
    """A KeyError in a prompt builder is a run that publishes nothing.

    This caught three `_H2_PATTERNS["decision_framework"]` fallbacks left behind by the
    rename — each one an exception on the first run of the new set.
    """
    state = {
        "outline": {"working_title": "T", "h2s": ["A", "B"], "lsi_keywords": [],
                    "internal_links": [], "real_projects_to_cite": []},
        "primary_keyword": "k", "angle": "a",
    }
    for name in P.ARCHETYPES:
        s = dict(state, archetype=name)
        assert P.topic_prompt("F", ["x"], [name], None)
        assert P.outline_prompt("F", "k", "a", "aud", name, ["slug"])
        assert P.section_intro_prompt("F", s)
        assert P.section_body_prompt("F", s, "A", {})
        assert P.section_closing_prompt("F", s)
        assert P.h2_scaffold(name, "custom CRM")


def test_h2_scaffold_never_leaks_an_unresolved_placeholder():
    for name in P.ARCHETYPES:
        for heading in P.h2_scaffold(name, "custom CRM"):
            assert "[" not in heading, f"{name}: {heading!r}"


def test_the_policy_reaches_every_writing_prompt():
    """Rejecting a bad draft is the fallback; not writing one is the fix."""
    state = {
        "outline": {"working_title": "T", "h2s": ["A"], "lsi_keywords": [],
                    "real_projects_to_cite": []},
        "primary_keyword": "k", "angle": "a", "archetype": "explainer",
    }
    for _, user in (
        P.section_intro_prompt("F", state),
        P.section_body_prompt("F", state, "A", {}),
        P.section_closing_prompt("F", state),
    ):
        assert "does not publish prices" in user


def test_topic_prompt_offers_only_unblocked_archetypes():
    _, user = P.topic_prompt("F", [], ["explainer", "tips_list"], None)
    menu = user.split("Required archetype")[1].split("Reply as JSON")[0]
    assert "  - explainer\n" not in menu
    assert "  - playbook\n" in menu


def test_topic_prompt_asks_for_a_citable_claim():
    """Citability is the whole point of the new direction, so it is a required field."""
    _, user = P.topic_prompt("F", [], [], None)
    assert "citable_claim" in user
