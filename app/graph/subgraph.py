"""
Exposes our compiled research agent as a subgraph node.
The orchestrator calls invoke_research_subgraph() as a single node —
it has no visibility into what happens inside.
"""

from typing import Any
from app.graph.builder import build_graph
from app.schemas.models import ResearchBrief
import logging

logger = logging.getLogger(__name__)

# Compile once at import time — not on every call
_research_app = build_graph()


async def invoke_research_subgraph(state: dict[str, Any]) -> dict[str, Any]:
    """
    Wraps our full research agent as a single callable node.

    The orchestrator passes in a goal.
    This node runs the entire agent and returns only the ResearchBrief.

    Args:
        state: Must contain 'goal' (str)

    Returns:
        dict with 'research_brief' (ResearchBrief) set
    """
    goal = state.get("goal", "")

    if not goal:
        logger.error("invoke_research_subgraph: no goal provided")
        return {"research_brief": None, "status": "error"}

    config = {"configurable": {"thread_id": f"subgraph-{id(goal)}"}}

    try:
        # Run the full agent to completion
        final_state = await _research_app.ainvoke(
            {"goal": goal},
            config=config,
        )

        # Extract only the brief — orchestrator gets a summary, not full transcript
        brief: ResearchBrief = final_state.get("research_brief")

        logger.info(f"invoke_research_subgraph: completed for goal='{goal[:50]}...'")
        return {"research_brief": brief, "status": "completed"}

    except Exception as e:
        logger.error(f"invoke_research_subgraph: failed — {e}")
        return {"research_brief": None, "status": "error", "last_error": str(e)}