"""
State schema for the orchestrator graph.
Separate from ResearchAgentState — the orchestrator
only sees high-level inputs and outputs, not internals.
"""

from typing import Optional, Annotated
import operator
from typing_extensions import TypedDict
from app.schemas.models import ResearchBrief


class OrchestratorState(TypedDict):
    """
    Minimal state for the orchestrator.
    The orchestrator never sees papers, gaps, or conversation_history —
    those are internal to the research subgraph.
    """

    # Input
    goal: str                          # user's original research goal

    # Routing
    task_type: Optional[str]           # "research" | "writing" | "unknown"

    # Outputs from subgraphs
    research_brief: Optional[ResearchBrief]   # populated by research_subgraph
    final_output: Optional[str]               # populated by writing_subgraph

    # Control
    status: Optional[str]              # "routing" | "completed" | "error"
    last_error: Optional[str]