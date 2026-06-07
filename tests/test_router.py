# tests/test_router.py

"""
Tests for should_continue router logic.
Each test builds a minimal StateDict and asserts the correct routing decision.
"""

import pytest
from app.graph.router import should_continue
from app.graph.state import StateDict


# ── HELPERS ──────────────────────────────────────────────────────────────────

def base_state(**overrides) -> StateDict:
    """Minimal valid state. Override only what each test needs."""
    state = {
        "goal": "test goal",
        "iteration_count": 0,
        "max_iterations": 5,
        "status": "gaps_found",
        "strategy": {"search_queries": ["q1"], "focus_areas": []},
        "papers": [],
        "gaps": [],
        "search_queries_tried": [],
        "failed_searches": [],
        "conversation_history": [],
        "last_error": None,
    }
    state.update(overrides)
    return state


# ── TEST CASES ────────────────────────────────────────────────────────────────

def test_hitl_resume_routes_to_human_input():
    """awaiting_user_input must ALWAYS go to human_input — even at max iterations."""
    state = base_state(
        status="awaiting_user_input",
        iteration_count=5,   # at the limit — but HITL must win
        max_iterations=5,
    )
    assert should_continue(state) == "human_input"


def test_max_iterations_routes_to_synthesis():
    state = base_state(iteration_count=5, max_iterations=5)
    assert should_continue(state) == "synthesis"


def test_error_status_routes_to_synthesis():
    state = base_state(status="error", last_error="LLM timeout")
    assert should_continue(state) == "synthesis"


def test_missing_strategy_routes_to_synthesis():
    state = base_state(strategy=None)
    assert should_continue(state) == "synthesis"


def test_no_results_routes_to_clarification():
    state = base_state(status="no_results")
    assert should_continue(state) == "clarification"


def test_no_papers_to_rank_routes_to_synthesis():
    state = base_state(status="no_papers_to_rank")
    assert should_continue(state) == "synthesis"


def test_no_papers_to_analyze_routes_to_synthesis():
    state = base_state(status="no_papers_to_analyze")
    assert should_continue(state) == "synthesis"


def test_gaps_resolved_routes_to_synthesis():
    state = base_state(status="gaps_resolved")
    assert should_continue(state) == "synthesis"


def test_loop_detection_routes_to_clarification():
    """All tried queries have failed — agent is spinning. Escalate."""
    state = base_state(
        status="gaps_found",
        search_queries_tried=["q1", "q2"],
        failed_searches=["q1", "q2"],   # exact match → loop detected
    )
    assert should_continue(state) == "clarification"


def test_partial_failures_still_routes_to_paper_collection():
    """Some queries failed but not all — still worth searching."""
    state = base_state(
        status="gaps_found",
        search_queries_tried=["q1", "q2"],
        failed_searches=["q1"],   # only q1 failed, q2 worked
    )
    assert should_continue(state) == "paper_collection"


def test_default_routes_to_paper_collection():
    """Normal mid-loop state — keep collecting papers."""
    state = base_state(status="gaps_found", iteration_count=2)
    assert should_continue(state) == "paper_collection"