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


def merge_papers(existing: List[RankedPaper], new: List[RankedPaper]) -> List[RankedPaper]:
    """
    Custom reducer: accumulate papers across iterations, deduplicate by arxiv_id.
    operator.add would blindly append — this prevents duplicates on re-runs.
    """
    seen = {p.arxiv_id for p in existing}
    deduped = [p for p in new if p.arxiv_id not in seen]
    return existing + deduped


class StateDict(TypedDictExt):
    """
    LangGraph state definition.
    
    Reducers define how state merges when multiple nodes update the same field:
    - List fields: use custom reducers or operator.add to APPEND (don't overwrite)
    - Other fields: default behavior is OVERWRITE (last node wins)
    """
    
    # --- GOAL & CONTEXT ---
    goal: str
    user_preferences: dict
    
    # --- PLANNING ---
    strategy: Optional[ResearchStrategy]
    
    # --- PAPERS & RANKING ---
    # Custom reducer: accumulates across iterations, deduplicates by arxiv_id
    papers: Annotated[List[RankedPaper], merge_papers]
    
    # Separate field for ranked/sorted view — ranking_node writes here, not to papers
    ranked_papers: Optional[List[RankedPaper]]
    
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
