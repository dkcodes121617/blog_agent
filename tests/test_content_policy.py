"""The editorial standard: no prices, no delivery durations.

These are the tests that matter most in this repo. Every other rule the agent enforces
is about whether a post is good; this one is about whether the studio's own blog argues
against the studio. It shipped 509 violations before anything checked, so the cases
below are drawn from what actually published rather than from what seemed likely.
"""
from __future__ import annotations

import pytest

from seo import content_policy as cp


def whys(violations):
    return {v.why for v in violations}


def texts(violations):
    return {v.text for v in violations}


# ── Money ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "The rebuild came in at $45,000 all told.",
    "Entry tier is $425K and enterprise is $1,450K.",
    "Budget about ₹5,00,000 for the first phase.",
    "They were paying 3,500 USD/year for seats nobody used.",
    "That is 6,000 dollars a year in unused licences.",
    "It works out at $12 per seat per month.",
    "Expect 40 USD per user before support.",
])
def test_money_figures_are_blocked(text):
    assert cp.figure_violations(text), f"missed a price in: {text}"


def test_money_amount_is_reported_in_full():
    """The reported string has to be findable in the draft.

    The first version matched only `[$]\\d`, so every finding read "remove the currency
    amount '$4'" for a draft containing $45,000 — and fix_claims is asked to edit by
    quoting the offending text, so it had nothing to match on.
    """
    assert "$45,000" in texts(cp.figure_violations("about $45,000 all in"))


# ── Duration ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "Implementation runs 12-24 weeks with consultants billing throughout.",
    "Give it six weeks before judging the result.",
    "The migration took 3 months of evenings.",
    "Week 1 is discovery, Month 6 is the first review.",
    "Plan for 2-3 years to break even.",
    "Expect 48 hours for the first response.",
    "We ship the prototype in ten days.",
])
def test_durations_are_blocked(text):
    assert cp.figure_violations(text), f"missed a duration in: {text}"


def test_overlapping_matches_report_once():
    """"12-24 weeks" must not also report as "24 weeks".

    Both patterns match it by design — the range pattern is there to give the better
    message. Reporting both sends fix_claims after an edit it has already made, and the
    fix loop has a budget of two.
    """
    found = cp.figure_violations("Implementation runs 12-24 weeks.")
    assert texts(found) == {"12-24 weeks"}


# ── What must stay legal ─────────────────────────────────────────────────────
# A gate that fails honest sentences gets deleted by whoever it blocks at 2am, so the
# false-positive cases are as load-bearing as the true ones.

@pytest.mark.parametrize("text", [
    "The cost of getting the data model wrong is a rewrite.",
    "We have been shipping Flutter apps for 3 years.",
    "Released in 2026, it changed how the routing layer works.",
    "It is cheaper to maintain than a rewrite, and easier to hand over.",
    "Latency stays under 200 ms on the voice path.",
    "Roughly 40% of support tickets never needed a human.",
    "See our [pricing](/pricing) page for the current packages.",
    "The team is 12 people across two offices.",
])
def test_legitimate_prose_is_not_blocked(text):
    assert cp.figure_violations(text) == [], f"false positive on: {text}"


def test_qualitative_money_language_survives():
    """Qualitative claims are the replacement for figures, so they must pass."""
    assert cp.figure_violations(
        "A purpose-built system costs less to run than the licences it replaces.") == []


# ── Framing vs figures ───────────────────────────────────────────────────────

def test_framing_is_separate_from_figures():
    """A cost headline with no number in it is still a cost post."""
    text = "What custom software actually costs in 2026"
    assert cp.figure_violations(text) == []
    assert cp.framing_violations(text)


@pytest.mark.parametrize("headline", [
    "What custom software actually costs in 2026",
    "How much does a mobile app cost",
    "How long does an MVP take to build",
    "Enterprise Browser Isolation: What It Costs",
    "Task Management Software Pricing Guide",
    "Is a custom CRM worth the budget",
])
def test_real_published_headlines_are_rejected(headline):
    """Every one of these is live on the blog today."""
    assert cp.topic_violations(headline), f"would still commission: {headline}"


@pytest.mark.parametrize("headline", [
    "How a custom CRM decides which lead to route first",
    "What most teams get wrong about their data model",
    "Postgres or MongoDB for a product that has to change shape",
    "Seven checks before you let an AI agent touch production",
    "What changed in on-device inference, and what it makes possible",
])
def test_good_headlines_pass(headline):
    assert cp.topic_violations(headline) == [], f"would reject: {headline}"


# ── Where the worst offenders actually hid ───────────────────────────────────

def test_component_props_are_scanned():
    """The published violations were mostly in chart data, not prose.

    A prose-only check reads straight past this, and the figure then gets enlarged onto
    the post's cover image and written into its alt text.
    """
    mdx = """Some ordinary prose about data models.

<StatGrid caption="x" stats={[
  { label: "Implementation", value: "12-24", unit: "weeks" },
  { label: "Entry tier", value: "$425K", unit: "" }
]} />
"""
    found = cp.body_violations(mdx)
    assert whys(found) >= {"duration range", "currency amount"}


def test_headings_are_checked_for_framing():
    mdx = "Lead paragraph.\n\n## What it costs to run\n\nSome prose.\n"
    assert cp.heading_violations(mdx)


def test_describe_is_deduplicated_and_bounded():
    many = cp.figure_violations(" ".join(f"${n},000" for n in range(1, 20)))
    summary = cp.describe(many, limit=3)
    assert summary.count(";") == 2
    assert "more)" in summary


def test_policy_block_is_stated_as_a_standard_not_a_directive():
    """The proxy refuses override/compliance phrasing (see prompts/library.py header).

    A prompt block that trips the injection guard returns "I can't discuss that." from
    every model, which would take out topic selection and every section call at once.
    """
    block = cp.POLICY_PROMPT_BLOCK.lower()
    for banned in ("you must", "never break", "reply with exactly", "obey"):
        assert banned not in block
    assert "price" in block and "duration" in block


# ── "one day" is usually an idiom ────────────────────────────────────────────
# Both hand-written posts that use it mean "eventually", not "in 24 hours". A gate
# that rewrites honest prose to satisfy itself is a gate that gets switched off.

@pytest.mark.parametrize("text", [
    "One day you want to hire your own team and find out what you own.",
    "Then one day the vendor stops answering.",
    "One day we will look back at this differently.",
])
def test_idiomatic_one_day_is_allowed(text):
    assert cp.figure_violations(text) == [], f"false positive on: {text}"


@pytest.mark.parametrize("text", [
    "The migration takes one day of focused work.",
    "We can turn that around in one week.",
    "Budget one month for the data clean-up.",
])
def test_one_as_a_real_duration_is_still_blocked(text):
    assert cp.figure_violations(text), f"missed a duration in: {text}"


def test_idiomatic_one_day_survives_punctuation():
    """"...in-house one day, the wider pool matters more" — live in the Flutter post."""
    assert cp.figure_violations(
        "If you plan to bring the app in-house one day, the wider pool matters more.") == []
