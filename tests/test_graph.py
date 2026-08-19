"""Graph wiring: the routers, and the guarantee that every branch has somewhere to go.

A router that returns a key the graph has no edge for raises at RUNTIME, mid-post, after
every expensive call has already been paid for. LangGraph validates the mapping at
compile time, so compiling the real graph is itself the test.
"""
from __future__ import annotations

from graph.build import (
    build_graph,
    route_after_factcheck,
    route_after_final_uniqueness,
    route_after_topic_uniqueness,
    route_after_validate,
)


class _StubNodes:
    """Every node the builder wires, as a no-op. Nothing here calls the proxy."""

    def __getattr__(self, _name):
        return lambda state: {}


def test_the_real_graph_compiles():
    """Catches an unmapped router key — including a new abort branch with no node."""
    assert build_graph(_StubNodes()) is not None


# ── Topic gate ───────────────────────────────────────────────────────────────

def test_a_policy_rejected_topic_is_re_picked():
    state = {"topic_similarity": 0.0, "topic_rejected": True, "topic_attempts": 1}
    assert route_after_topic_uniqueness(state) == "retry_topic"


def test_topic_attempts_are_bounded():
    """Otherwise a strategist stuck on cost topics loops until the runner is killed."""
    state = {"topic_similarity": 0.0, "topic_rejected": True, "topic_attempts": 99}
    assert route_after_topic_uniqueness(state) == "abort"


def test_a_clean_unique_topic_proceeds():
    state = {"topic_similarity": 0.1, "topic_rejected": False, "topic_attempts": 1}
    assert route_after_topic_uniqueness(state) == "ok"


# ── Final gate ───────────────────────────────────────────────────────────────

def test_policy_violating_metadata_aborts_the_run():
    """The metadata call runs last and sees only the body, so it can still hand back a
    cost headline over a perfectly clean article. The title and slug decide what the
    post ranks for, so publishing it anyway would defeat the whole standard."""
    state = {"body_similarity": 0.1, "metadata_policy_hits": ["money framing: 'cost'"]}
    assert route_after_final_uniqueness(state) == "abort_policy"


def test_clean_metadata_publishes():
    state = {"body_similarity": 0.1, "metadata_policy_hits": []}
    assert route_after_final_uniqueness(state) == "ok"


def test_missing_policy_key_does_not_crash_the_router():
    """BlogState is a plain TypedDict — a node that never ran leaves the key absent."""
    assert route_after_final_uniqueness({"body_similarity": 0.1}) == "ok"


def test_a_near_duplicate_still_aborts():
    state = {"body_similarity": 0.99, "metadata_policy_hits": []}
    assert route_after_final_uniqueness(state) == "abort"


# ── Repair and fact-check budgets ────────────────────────────────────────────

def test_validation_errors_prefer_surgical_repair_over_a_rewrite():
    """A rewrite regenerates every diagram, so it swaps one clipped label for another —
    three consecutive publish days aborted that way."""
    state = {"validation_errors": ["label 'x' is too long for its box"], "repair_attempts": 0}
    assert route_after_validate(state) == "repair"


def test_factcheck_ships_a_valid_draft_once_the_fix_budget_is_spent():
    state = {"factcheck_issues": ["something"], "fix_attempts": 99}
    assert route_after_factcheck(state) == "ok"


def test_factcheck_fixes_within_budget():
    state = {"factcheck_issues": ["something"], "fix_attempts": 0}
    assert route_after_factcheck(state) == "fix"


# ── What counts as surgically repairable ─────────────────────────────────────

def test_prop_renames_and_policy_figures_route_to_repair_not_a_rewrite():
    """Both are string-level edits. Routing them to a rewrite regenerates every section
    and every diagram, and the new draft arrives with a different violation in a
    different place — the non-convergence this module's header documents."""
    from seo.mdx_repair import repairable
    assert repairable(["<StatGrid> uses `data=` but the component reads `stats=` — rename the prop"])
    assert repairable(["editorial policy — remove the currency amount '$45,000'"])


def test_structural_errors_still_need_the_writer():
    from seo.mdx_repair import repairable
    assert not repairable(["body has 2 H2 sections, needs 4-8"])
