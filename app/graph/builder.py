"""
Build and compile the LangGraph StateGraph.

Assembles all nodes, edges, and routing logic into a runnable agent.
Configures checkpointing for HITL (Human-in-the-Loop) pauses.

M10 changes:
  - Swapped MemorySaver → SqliteSaver (persists across restarts)
  - Added interrupt_after=["gap_analysis"] for HITL pause
  - Added human_input_node for injecting user answers
"""

import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from app.schemas.models import AgentState
from app.graph.state import StateDict
from app.graph.nodes import (
    planning_node,
    paper_collection_node,
    ranking_node,
    context_budgeting_node,
    gap_analysis_node,
    clarification_node,
    human_input_node,
    synthesis_node,
)
from app.graph.router import should_continue
from app.config import settings

logger = logging.getLogger(__name__)

# SQLite DB path — override via DB_PATH env var
DB_PATH = getattr(settings, "db_path", "research_agent.db")


def build_graph():
    """
    Build the research agent graph.

    Graph flow:
        START
          → planning
          → paper_collection
          → ranking
          → context_budgeting
          → gap_analysis          ← HITL PAUSE HERE (interrupt_after)
              ↓ should_continue()
              ├─ paper_collection  (more research needed)
              ├─ clarification     (need user input)
              └─ synthesis         (goal met)
          → human_input_node      (inject user answers → conversation_history)
          → clarification         (generate typed ClarifyingQuestions)
          → synthesis
          → END

    Returns:
        Compiled StateGraph with SqliteSaver checkpointer and HITL interrupt.
    """
    graph = StateGraph(StateDict)

    # ================================================================
    # ADD NODES
    # ================================================================
    graph.add_node("planning", planning_node)
    graph.add_node("paper_collection", paper_collection_node)
    graph.add_node("ranking", ranking_node)
    graph.add_node("context_budgeting", context_budgeting_node)
    graph.add_node("gap_analysis", gap_analysis_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("human_input", human_input_node)
    graph.add_node("synthesis", synthesis_node)

    # ================================================================
    # ADD EDGES (fixed paths)
    # ================================================================
    graph.add_edge(START, "planning")
    graph.add_edge("planning", "paper_collection")
    graph.add_edge("paper_collection", "ranking")
    graph.add_edge("ranking", "context_budgeting")
    graph.add_edge("context_budgeting", "gap_analysis")

    # After human answers questions, run clarification then synthesize
    graph.add_edge("human_input", "clarification")
    graph.add_edge("clarification", "synthesis")
    graph.add_edge("synthesis", END)

    # ================================================================
    # CONDITIONAL EDGES — router decides next step after gap_analysis
    # ================================================================
    graph.add_conditional_edges(
        "gap_analysis",
        should_continue,
        {
            "paper_collection": "paper_collection",
            "clarification": "clarification",   # direct path (no HITL needed)
            "human_input": "human_input",       # HITL path (user answers needed)
            "synthesis": "synthesis",
        }
    )

    # ================================================================
    # COMPILE — SqliteSaver + interrupt_after for HITL
    # ================================================================
    # SqliteSaver: persists state to disk — survives server restarts.
    # interrupt_after=["gap_analysis"]: graph PAUSES after gap_analysis_node
    # completes and before the next node executes. State is fully saved
    # to SQLite at the pause point.
    checkpointer = SqliteSaver.from_conn_string(DB_PATH)

    app = graph.compile(
        checkpointer=checkpointer,
        interrupt_after=["gap_analysis"],
    )

    logger.info(
        f"Graph compiled with SqliteSaver ({DB_PATH}) "
        f"and interrupt_after=['gap_analysis']"
    )
    return app


# Singleton instance — imported by API routes and HITL flow
agent_graph = build_graph()
