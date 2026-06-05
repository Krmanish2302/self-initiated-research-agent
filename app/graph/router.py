# app/graph/router.py

"""
Conditional routing: decides which node runs next based on state.

The should_continue function is called after gap_analysis_node,
and it returns the name of the next node to execute.
"""

from typing import Literal
from app.graph.state import StateDict
import logging

logger = logging.getLogger(__name__)

# No magic strings — explicit Literal type for all valid routing targets
RoutingDecision = Literal[
    "paper_collection",
    "clarification",
    "human_input",
    "synthesis"
]


def should_continue(state: StateDict) -> RoutingDecision:
    """
    The brainstem of the agent — turns continuous signals in state
    into a discrete choice of next node.

    Priority order (top = highest priority):
    1. HITL resume path   — awaiting_user_input → human_input
    2. Hard stop          — max_iterations reached → synthesis
    3. Error / bad state  — status=error or no strategy → synthesis
    4. No papers at all   — status=no_results → clarification
    5. Empty corpus       — no_papers_to_rank/analyze → synthesis
    6. Gaps resolved      — gaps_resolved → synthesis
    7. Loop detection     — repeated/failed queries → clarification
    8. Default            — keep collecting papers
    """

    # ── 1. HITL RESUME PATH (must be first — never skip human answers) ──
    if state["status"] == "awaiting_user_input":
        logger.info("should_continue: HITL resume → human_input")
        return "human_input"

    # ── 2. HARD STOP ──
    if state["iteration_count"] >= state["max_iterations"]:
        logger.info(
            f"should_continue: max_iterations ({state['max_iterations']}) reached"
        )
        return "synthesis"

    # ── 3. ERROR / STRUCTURALLY INVALID STATE ──
    if state["status"] == "error" or not state.get("strategy"):
        logger.warning(
            f"should_continue: error or missing strategy → synthesis | "
            f"last_error={state.get('last_error')}"
        )
        return "synthesis"

    # ── 4. NO PAPERS FOUND AT ALL ──
    if state["status"] == "no_results":
        logger.info("should_continue: no papers found → clarification")
        return "clarification"

    # ── 5. EMPTY CORPUS (ranking/analysis had nothing to work with) ──
    if state["status"] in ("no_papers_to_analyze", "no_papers_to_rank"):
        logger.info(f"should_continue: empty corpus ({state['status']}) → synthesis")
        return "synthesis"

    # ── 6. GAPS RESOLVED ──
    if state["status"] == "gaps_resolved":
        logger.info("should_continue: gaps resolved → synthesis")
        return "synthesis"

    # ── 7. LOOP DETECTION ──
    # If the planner keeps generating queries we've already tried or
    # that have already failed — escalate to clarification instead of
    # sending the agent back into a dead-end search loop
    failed = set(state.get("failed_searches", []))
    tried = set(state.get("search_queries_tried", []))
    if failed and failed == tried:
        logger.warning(
            "should_continue: all tried queries have failed → clarification"
        )
        return "clarification"

    # ── 8. DEFAULT: keep researching ──
    logger.info("should_continue: continuing research loop → paper_collection")
    return "paper_collection"