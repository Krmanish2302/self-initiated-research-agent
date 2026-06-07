"""
Orchestrator graph — routes between research and writing subgraphs.
The research agent is a single node from the orchestrator's perspective.
"""

from typing import Any
from langgraph.graph import StateGraph, START, END
from app.orchestrator.state import OrchestratorState
from app.graph.subgraph import invoke_research_subgraph
import logging

logger = logging.getLogger(__name__)


# ── NODES ────────────────────────────────────────────────────────────────────

async def routing_node(state: OrchestratorState) -> dict[str, Any]:
    """
    Reads the goal and decides which subgraph to call first.
    Simple keyword routing for now — can be replaced with LLM routing later.
    """
    goal = state.get("goal", "").lower()

    # Keyword heuristic — good enough for M12
    if any(kw in goal for kw in ["research", "papers", "study", "survey", "find"]):
        task_type = "research"
    elif any(kw in goal for kw in ["write", "draft", "summarize", "report"]):
        task_type = "writing"
    else:
        task_type = "research"  # default: research first

    logger.info(f"routing_node: task_type={task_type} for goal='{goal[:50]}'")
    return {"task_type": task_type, "status": "routing"}


async def writing_subgraph_placeholder(state: OrchestratorState) -> dict[str, Any]:
    """
    Placeholder for the writing subgraph (M13+).
    For now, just formats the ResearchBrief as a string.
    """
    brief = state.get("research_brief")

    if not brief:
        return {"final_output": "No research brief available.", "status": "error"}

    # Format brief as readable output — writing agent will replace this
    output = f"Research Summary: {brief.summary}\n\nKey Findings:\n"
    for finding in brief.key_findings:
        output += f"- {finding}\n"

    return {"final_output": output, "status": "completed"}


# ── ROUTER ───────────────────────────────────────────────────────────────────

def should_research_or_write(state: OrchestratorState) -> str:
    """Routes after routing_node based on task_type."""
    task_type = state.get("task_type", "research")

    if task_type == "writing" and state.get("research_brief"):
        # Brief already exists — skip research, go straight to writing
        return "writing"
    return "research"


def after_research(state: OrchestratorState) -> str:
    """After research subgraph completes, always go to writing."""
    if state.get("status") == "error":
        return END
    return "writing"


# ── GRAPH ASSEMBLY ────────────────────────────────────────────────────────────

def build_orchestrator() -> Any:
    graph = StateGraph(OrchestratorState)

    # Register nodes
    graph.add_node("routing", routing_node)
    graph.add_node("research", invoke_research_subgraph)   # ← our full agent = 1 node
    graph.add_node("writing", writing_subgraph_placeholder)

    # Edges
    graph.add_edge(START, "routing")
    graph.add_conditional_edges("routing", should_research_or_write, {
        "research": "research",
        "writing": "writing",
    })
    graph.add_conditional_edges("research", after_research, {
        "writing": "writing",
        END: END,
    })
    graph.add_edge("writing", END)

    return graph.compile()


# Singleton — compiled once
orchestrator_app = build_orchestrator()