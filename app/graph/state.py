"""
LangGraph state definition with TypedDict and reducers.

Converts Pydantic AgentState into a TypedDict that LangGraph understands,
with reducer annotations that define how state merges when multiple nodes
update the same fields.
"""

from typing import Annotated, TypedDict, Optional, List, Any
from typing_extensions import TypedDict as TypedDictExt
import operator
from app.schemas.models import (
    AgentState,
    RankedPaper,
    KnowledgeGap,
    ResearchStrategy,
)


class StateDict(TypedDictExt):
    """
    LangGraph state definition.
    
    Reducers define how state merges when multiple nodes update the same field:
    - List fields: use operator.add to APPEND (don't overwrite)
    - Other fields: default behavior is OVERWRITE (last node wins)
    """
    
    # --- GOAL & CONTEXT ---
    goal: str
    user_preferences: dict
    
    # --- PLANNING ---
    strategy: Optional[ResearchStrategy]
    
    # --- PAPERS & RANKING ---
    # Reducer: operator.add appends new papers, doesn't replace
    papers: Annotated[List[RankedPaper], operator.add]
    
    # --- KNOWLEDGE GAPS ---
    # Reducer: operator.add appends gaps
    gaps: Annotated[List[KnowledgeGap], operator.add]
    
    # --- CONVERSATION HISTORY ---
    # Reducer: operator.add appends messages
    conversation_history: Annotated[List[dict], operator.add]
    
    # --- ITERATION TRACKING ---
    iteration_count: int
    max_iterations: int
    
    # --- STATUS & CONTROL ---
    status: str
    last_error: Optional[str]
    
    # --- SEARCH HISTORY (for loop detection) ---
    # Reducer: operator.add appends queries
    search_queries_tried: Annotated[List[str], operator.add]
    failed_searches: Annotated[List[str], operator.add]
    
    # --- CONTEXT MANAGEMENT ---
    summarized_history: Optional[str]
