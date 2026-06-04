# self_initiated_research_agent/app/schemas/models.py

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

# ============================================================
# PAPER MODELS
# ============================================================
class RankingWeights(BaseModel):
    """Weights for composite ranking score components"""
    citation_weight: float = Field(default=0.33, ge=0, le=1)
    relevance_weight: float = Field(default=0.33, ge=0, le=1)
    recency_weight: float = Field(default=0.34, ge=0, le=1)


class PaperMetadata(BaseModel):
    arxiv_id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    abstract: str
    published_date: str
    url: str          
    pdf_url: str = Field(default="")   
    source: str = Field(default="arxiv")  

class CitationData(BaseModel):
    """Citation metrics from Semantic Scholar API"""
    citation_count: int = Field(default=0, description="Total citation count")
    influential_citation_count: int = Field(default=0, description="Highly influential citations")
    h_index: Optional[int] = Field(default=None, description="H-index of paper (if available)")

    class Config:
        json_schema_extra = {
            "example": {
                "citation_count": 245,
                "influential_citation_count": 18,
                "h_index": None
            }
        }


class RankedPaper(BaseModel):
    """Paper with ranking scores and final position"""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    published_date: str
    url: str
    
    # Ranking components (from our paper_ranker_tool)
    citation_count: int
    relevance_score: float = Field(..., ge=0, le=1, description="0-1 relevance to goal")
    recency_score: float = Field(..., ge=0, le=1, description="0-1 recency (newer=higher)")
    composite_rank_score: float = Field(..., ge=0, le=3, description="Sum of weighted scores")
    
    # Final position in sorted list
    rank_position: int = Field(..., ge=1, description="Position in ranked papers (1=best)")

    class Config:
        json_schema_extra = {
            "example": {
                "arxiv_id": "2401.12345",
                "title": "Efficient Transformers",
                "authors": ["Alice"],
                "abstract": "We propose...",
                "published_date": "2024-01-15",
                "url": "https://arxiv.org/abs/2401.12345",
                "citation_count": 245,
                "relevance_score": 0.95,
                "recency_score": 0.88,
                "composite_rank_score": 2.78,
                "rank_position": 1
            }
        }


# ============================================================
# KNOWLEDGE GAP MODELS
# ============================================================

class KnowledgeGap(BaseModel):
    """A gap in understanding identified by the agent"""
    gap_id: str = Field(..., description="Unique ID for this gap (e.g., 'gap_001')")
    description: str = Field(..., description="What don't we know yet?")
    relevance_to_goal: float = Field(..., ge=0, le=1, description="How important is this gap?")
    is_resolved: bool = Field(default=False, description="Did the user answer our question?")
    user_answer: Optional[str] = Field(default=None, description="User's response if resolved")
    clarifying_question: Optional[str] = Field(default=None, description="Question we asked to fill this gap")

    class Config:
        json_schema_extra = {
            "example": {
                "gap_id": "gap_001",
                "description": "No papers on vision transformer efficiency",
                "relevance_to_goal": 0.9,
                "is_resolved": False,
                "user_answer": None,
                "clarifying_question": "Are you interested in vision transformers specifically?"
            }
        }


# ============================================================
# STRATEGY & PLANNING MODELS
# ============================================================

class ResearchStrategy(BaseModel):
    """Decomposed research plan created by planning_node"""
    topics: List[str] = Field(..., description="Search topics (e.g., ['perception', 'planning'])")
    date_range_start: Optional[str] = Field(default=None, description="Start date (YYYY-MM-DD)")
    date_range_end: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD)")
    ranking_criteria: List[str] = Field(
        default=["citation_count", "relevance", "recency"],
        description="What to weight in ranking"
    )
    max_papers_target: int = Field(default=20, description="Ideal number of papers to collect")
    search_depth: str = Field(default="balanced", description="'broad', 'balanced', or 'deep'")

    class Config:
        json_schema_extra = {
            "example": {
                "topics": ["transformer efficiency", "attention mechanisms"],
                "date_range_start": "2022-01-01",
                "date_range_end": "2024-12-31",
                "ranking_criteria": ["citation_count", "relevance", "recency"],
                "max_papers_target": 20,
                "search_depth": "balanced"
            }
        }


# ============================================================
# AGENT STATE MODEL (The Big One)
# ============================================================

class AgentState(BaseModel):
    """
    The full state of the research agent.
    This flows between nodes and is persisted by the checkpointer.
    """

    # --- GOAL & CONTEXT ---
    goal: str = Field(..., description="User's research goal")
    user_preferences: dict = Field(
        default_factory=dict,
        description="User preferences (date range, specific subtopics, etc.)"
    )

    # --- PLANNING ---
    strategy: Optional[ResearchStrategy] = Field(
        default=None,
        description="Current research strategy (set by planning_node)"
    )

    # --- PAPERS & RANKING ---
    papers: List[RankedPaper] = Field(
        default_factory=list,
        description="Papers collected and ranked so far"
    )

    # --- KNOWLEDGE GAPS ---
    gaps: List[KnowledgeGap] = Field(
        default_factory=list,
        description="Knowledge gaps identified by gap_analyzer_node"
    )

    # --- CONVERSATION HISTORY ---
    conversation_history: List[dict] = Field(
        default_factory=list,
        description="Messages between agent and user (for context + HITL)"
    )

    # --- ITERATION TRACKING ---
    iteration_count: int = Field(default=0, description="How many loops have we done?")
    max_iterations: int = Field(default=5, description="Hard limit on iterations")
    
    # --- STATUS & CONTROL ---
    status: str = Field(
        default="initialized",
        description="Current state: initialized, planning, searching, analyzing, paused, complete, error"
    )
    last_error: Optional[str] = Field(default=None, description="Last error message if any")

    # --- SEARCH HISTORY (for loop detection) ---
    search_queries_tried: List[str] = Field(
        default_factory=list,
        description="All search queries attempted (to avoid repeating failed searches)"
    )
    failed_searches: List[str] = Field(
        default_factory=list,
        description="Searches that returned 0 results"
    )

    # --- CONTEXT MANAGEMENT ---
    summarized_history: Optional[str] = Field(
        default=None,
        description="Compressed version of old conversation for context budgeting"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "goal": "Understand transformer efficiency improvements",
                "user_preferences": {"date_range": "2023-2024", "focus": "vision transformers"},
                "strategy": None,
                "papers": [],
                "gaps": [],
                "conversation_history": [],
                "iteration_count": 0,
                "max_iterations": 5,
                "status": "initialized",
                "last_error": None,
                "search_queries_tried": [],
                "failed_searches": []
            }
        }


# ============================================================
# API REQUEST/RESPONSE MODELS (for FastAPI, MODULE 13)
# ============================================================

class ResearchGoalRequest(BaseModel):
    """User's request to start research"""
    goal: str = Field(..., description="What do you want to research?")
    user_preferences: dict = Field(
        default_factory=dict,
        description="Optional preferences (date_range, subtopics, etc.)"
    )


class UserAnswerRequest(BaseModel):
    """User's response to a clarifying question during HITL pause"""
    answer: str = Field(..., description="User's answer")


class ResearchBrief(BaseModel):
    """Final output from synthesis_node"""
    executive_summary: str = Field(..., description="High-level overview")
    top_papers: List[RankedPaper] = Field(..., description="Best papers found")
    key_findings: List[str] = Field(..., description="Main insights from the research")
    remaining_gaps: List[str] = Field(..., description="What we still don't know")
    next_steps: List[str] = Field(..., description="Recommendations for deeper research")
    iterations_taken: int = Field(..., description="How many loops did we do?")
    total_papers_found: int = Field(..., description="How many papers were discovered?")

    class Config:
        json_schema_extra = {
            "example": {
                "executive_summary": "Transformer efficiency has improved 3x in 2 years...",
                "top_papers": [],  # List of RankedPaper objects
                "key_findings": ["Attention sparsity is key", "Vision models lag text models"],
                "remaining_gaps": ["Mobile deployment efficiency", "Real-time inference"],
                "next_steps": ["Investigate edge deployment", "Study knowledge distillation"],
                "iterations_taken": 3,
                "total_papers_found": 18
            }
        }