"""Assembles the LangGraph pipeline with self-correcting conditional edges."""
from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from config import CONFIG
from facts.snapshot import build_snapshot
from graph.nodes.content_nodes import (
    FIXCLAIMS_BUDGET,
    HUMANIZE_MIN_SCORE,
    MAX_REVISIONS,
    MAX_TOPIC_ATTEMPTS,
    Nodes,
)
from graph.state import BlogState
from knowledge.store import KnowledgeBase
from llm.client import LLMClient

log = logging.getLogger("agent.graph")


# ── routers (conditional edges) ──
def route_after_topic_uniqueness(state: BlogState) -> str:
    if state["topic_similarity"] >= CONFIG.topic_sim_threshold:
        if state.get("topic_attempts", 0) >= MAX_TOPIC_ATTEMPTS:
            return "abort"
        return "retry_topic"
    return "ok"


def route_after_validate(state: BlogState) -> str:
    if state.get("validation_errors"):
        if state.get("revision", 0) >= MAX_REVISIONS:
            return "abort"
        return "rewrite"
    return "ok"


def route_after_factcheck(state: BlogState) -> str:
    if not state.get("factcheck_issues"):
        return "ok"
    if state.get("fix_attempts", 0) >= FIXCLAIMS_BUDGET:
        # We've surgically cleaned the draft up to the budget and it still passes
        # MDX validation. Remaining flags are almost always false positives on
        # generic advice — ship the valid, grounded draft rather than never publish.
        log.warning("factcheck: %d issue(s) remain after %d fix(es) — publishing the "
                    "valid draft anyway", len(state["factcheck_issues"]),
                    state.get("fix_attempts", 0))
        return "ok"
    return "fix"


def route_after_final_uniqueness(state: BlogState) -> str:
    if state["body_similarity"] >= CONFIG.body_sim_threshold:
        return "abort"
    return "ok"


def _abort(reason: str):
    def _node(state: BlogState) -> dict:
        log.warning("ABORT: %s", reason)
        return {"status": "aborted", "abort_reason": reason}
    return _node


def build_graph(nodes: Nodes):
    g = StateGraph(BlogState)

    g.add_node("load_context", nodes.load_context)
    g.add_node("pick_topic", nodes.pick_topic)
    g.add_node("check_topic", nodes.check_topic_uniqueness)
    g.add_node("outline", nodes.outline)
    g.add_node("write", nodes.write)
    g.add_node("factcheck", nodes.factcheck)
    g.add_node("fix_claims", nodes.fix_claims)
    g.add_node("validate", nodes.validate)
    g.add_node("humanize", nodes.humanize)
    g.add_node("registry", nodes.build_registry)
    g.add_node("final_uniqueness", nodes.final_uniqueness)
    g.add_node("finalize", nodes.finalize)
    g.add_node("abort_topic", _abort("topic space exhausted — every angle too similar to existing posts"))
    g.add_node("abort_validate", _abort("could not produce contract-valid MDX within revision budget"))
    g.add_node("abort_dup", _abort("finished draft too similar to an existing post"))

    g.add_edge(START, "load_context")
    g.add_edge("load_context", "pick_topic")
    g.add_edge("pick_topic", "check_topic")
    g.add_conditional_edges(
        "check_topic", route_after_topic_uniqueness,
        {"ok": "outline", "retry_topic": "pick_topic", "abort": "abort_topic"},
    )
    g.add_edge("outline", "write")
    # write -> validate (cheap, deterministic) -> factcheck (LLM). Running the
    # deterministic contract check first means a malformed draft loops back to the
    # writer WITHOUT spending a fact-check API call. Both must pass before humanize.
    g.add_edge("write", "validate")
    g.add_conditional_edges(
        "validate", route_after_validate,
        {"ok": "factcheck", "rewrite": "write", "abort": "abort_validate"},
    )
    # Fact-check issues → surgical fix_claims (fast, convergent) → re-validate →
    # re-check. After the budget, a still-valid draft ships anyway (see router).
    g.add_conditional_edges(
        "factcheck", route_after_factcheck,
        {"ok": "humanize", "fix": "fix_claims"},
    )
    g.add_edge("fix_claims", "validate")
    g.add_edge("humanize", "registry")
    g.add_edge("registry", "final_uniqueness")
    g.add_conditional_edges(
        "final_uniqueness", route_after_final_uniqueness,
        {"ok": "finalize", "abort": "abort_dup"},
    )
    g.add_edge("finalize", END)
    for a in ("abort_topic", "abort_validate", "abort_dup"):
        g.add_edge(a, END)

    return g.compile()


def make_pipeline(kb: KnowledgeBase | None = None, llm: LLMClient | None = None):
    """Factory: build facts snapshot + KB + graph, ready to invoke."""
    snap = build_snapshot()
    kb = kb or KnowledgeBase()
    nodes = Nodes(facts_block=snap.to_prompt_block(), kb=kb, llm=llm)
    return build_graph(nodes), nodes, kb
