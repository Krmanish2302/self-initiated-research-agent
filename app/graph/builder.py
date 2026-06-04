"""
Build and compile the LangGraph StateGraph.

Assembles all nodes, edges, and routing logic into a runnable agent.
Configures checkpointing for HITL (Human-in-the-Loop) pauses.
"""

import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from app.schemas.models import AgentState
from app.graph.state import StateDict
from app.graph.nodes import (
    planning_node,
    paper_collection_node,
    ranking_node,
    gap_analysis_node,
    clarification_node,
    synthesis_node,
)
from app.graph.router import should_continue
from app.config import settings

logger = logging.getLogger(__name__)


def build_graph():
    """
    Build the research agent graph.
    
    Returns:
        Compiled StateGraph with all nodes, edges, and checkpointing configured.
    """
    
    # Create StateGraph with our StateDict definition
    graph = StateGraph(StateDict)
    
    # ================================================================
    # ADD NODES
    # ================================================================
    graph.add_node("planning", planning_node)
    graph.add_node("paper_collection", paper_collection_node)
    graph.add_node("ranking", ranking_node)
    graph.add_node("gap_analysis", gap_analysis_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("synthesis", synthesis_node)
    
    # ================================================================
    # ADD EDGES (fixed paths)
    # ================================================================
    graph.add_edge(START, "planning")  # Entry point
    graph.add_edge("planning", "paper_collection")
    graph.add_edge("paper_collection", "ranking")
    graph.add_edge("ranking", "gap_analysis")
    graph.add_edge("clarification", "synthesis")
    graph.add_edge("synthesis", END)  # Exit point
    
    # ================================================================
    # ADD CONDITIONAL EDGE (router decides)
    # ================================================================
    graph.add_conditional_edges(
        "gap_analysis",
        should_continue,
        {
            "paper_collection": "paper_collection",
            "clarification": "clarification",
            "synthesis": "synthesis",
        }
    )
    
    # ================================================================
    # COMPILE WITH CHECKPOINTING
    # ================================================================
    checkpointer = MemorySaver()
    app = graph.compile(
        checkpointer=checkpointer,
    )

    logger.info("Graph compiled successfully")
    return app


# Singleton instance
agent_graph = build_graph()