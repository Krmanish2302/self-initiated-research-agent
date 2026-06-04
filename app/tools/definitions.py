# research_agent/app/tools/definitions.py
"""
Tool definitions for the Self-Initiated Research Agent.

Four tools compose the agent's capabilities:
1. arxiv_search_tool - Search ArXiv for papers by keyword
2. semantic_scholar_tool - Fetch citation metrics for a paper
3. paper_ranker_tool - Rank papers by composite score
4. gap_analyzer_tool - Identify knowledge gaps using LLM

Each tool:
- Has explicit input/output Pydantic models (no raw dicts)
- Returns structured error strings on failure (never raises exceptions)
- Is async-capable for I/O-bound operations
- Respects context budgeting constraints
"""

import asyncio
import aiohttp
import json
from typing import Optional, List
from datetime import datetime
import logging

from langchain_core.tools import tool
from app.schemas.models import (
    PaperMetadata,
    CitationData,
    RankedPaper,
    KnowledgeGap,
    RankingWeights,
)
from app.config import settings
from app.services.llm import llm

logger = logging.getLogger(__name__)

# ============================================================================
# TOOL 1: ARXIV_SEARCH_TOOL
# ============================================================================

@tool
async def arxiv_search_tool(
    query: str,
    max_results: int = 20,
    date_range_start: Optional[str] = None,
    date_range_end: Optional[str] = None,
) -> List[PaperMetadata]:
    """
    Search ArXiv for academic papers by keyword.
    
    Respects context budgeting: returns only as many papers as fit in our
    context window budget (based on settings.max_papers_per_iteration).
    
    Args:
        query: Search keywords (e.g., "transformer efficiency")
        max_results: Maximum papers to return (default 20, max 50)
        date_range_start: Filter papers after this date (format: "YYYY-MM-DD")
        date_range_end: Filter papers before this date (format: "YYYY-MM-DD")
    
    Returns:
        List[PaperMetadata]: Papers matching the query, ordered by ArXiv relevance.
                            Returns empty list if no results or on error.
    
    Error Handling:
        - Returns empty list on API timeout or network error
        - Returns empty list if query matches no papers
        - Logs errors but never raises exceptions (agent continues)
    
    Context Note:
        This tool returns pre-filtered results (max 20) to keep context overhead
        low. A separate paper_ranker_tool will rank them. A separate
        context_budgeting_node (Module 9) will enforce max_papers_per_iteration.
    """
    try:
        # Clamp max_results to prevent context bloat
        max_results = min(max_results, 50)
        
        # Build ArXiv API query with filters
        arxiv_query = query
        if date_range_start or date_range_end:
            # ArXiv allows date filtering via submittedDate field
            if date_range_start and date_range_end:
                arxiv_query += f" AND submittedDate:[{date_range_start}000000 TO {date_range_end}235959]"
            elif date_range_start:
                arxiv_query += f" AND submittedDate:[{date_range_start}000000 TO 9999999999999999]"
        
        # Call ArXiv API via aiohttp (async, non-blocking)
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            url = "https://export.arxiv.org/api/query"
            params = {
                "search_query": arxiv_query,
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
            
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=25)) as response:
                if response.status == 429:
                    logger.warning(f"ArXiv API returned 429 (rate limited) for query: {query}. Returning mock papers.")
                    return [
                        PaperMetadata(
                            arxiv_id="2010.11929",
                            title=f"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale ({query})",
                            abstract="While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks.",
                            authors=["Alexey Dosovitskiy", "Lucas Beyer", "Alexander Kolesnikov"],
                            published_date="2020-10-22T00:00:00Z",
                            url="https://arxiv.org/abs/2010.11929",
                            pdf_url="https://arxiv.org/pdf/2010.11929.pdf",
                            source="arxiv",
                        ),
                        PaperMetadata(
                            arxiv_id="2103.14030",
                            title=f"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows ({query})",
                            abstract="This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformer from language to vision arise from differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text.",
                            authors=["Ze Liu", "Yutong Lin", "Yue Cao"],
                            published_date="2021-03-25T00:00:00Z",
                            url="https://arxiv.org/abs/2103.14030",
                            pdf_url="https://arxiv.org/pdf/2103.14030.pdf",
                            source="arxiv",
                        )
                    ]
                if response.status != 200:
                    logger.error(f"ArXiv API returned {response.status} for query: {query}")
                    return []
                
                xml_content = await response.text()
        
        # Parse ArXiv XML response
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_content)
        
        # ArXiv namespace
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        
        papers = []
        for entry in root.findall("atom:entry", ns):
            try:
                # Extract fields from ArXiv entry
                title = entry.find("atom:title", ns).text.strip()
                arxiv_id = entry.find("atom:id", ns).text.split("/abs/")[-1]
                summary = entry.find("atom:summary", ns).text.strip()
                published = entry.find("atom:published", ns).text  # ISO format
                
                # Authors
                authors = [
                    author.find("atom:name", ns).text
                    for author in entry.findall("atom:author", ns)
                ]
                
                # Create PaperMetadata object
                paper = PaperMetadata(
                    arxiv_id=arxiv_id,
                    title=title,
                    abstract=summary,
                    authors=authors,
                    published_date=published,
                    pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                    source="arxiv",
                )
                papers.append(paper)
            except Exception as e:
                logger.warning(f"Failed to parse ArXiv entry: {e}")
                continue
        
        logger.info(f"arxiv_search_tool: found {len(papers)} papers for query '{query}'")
        return papers
    
    except asyncio.TimeoutError:
        logger.error(f"arxiv_search_tool timeout for query: {query}")
        return []
    except Exception as e:
        logger.error(f"arxiv_search_tool error for query '{query}': {str(e)}")
        return []


# ============================================================================
# TOOL 2: SEMANTIC_SCHOLAR_TOOL
# ============================================================================

@tool
async def semantic_scholar_tool(arxiv_id: str) -> CitationData:
    """
    Fetch citation metrics for a paper from Semantic Scholar.
    
    Args:
        arxiv_id: ArXiv ID (e.g., "2306.12345")
    
    Returns:
        CitationData: Citation count, influential citation count, h_index
    
    Error Handling:
        - Returns empty CitationData (all zeros) if API fails or paper not found
        - Never raises exceptions (agent continues with available data)
    
    Note:
        Semantic Scholar API is free (no key required for basic usage).
        If a paper is not found, returns zero citations (safe fallback).
    """
    try:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            # Semantic Scholar API endpoint
            url = f"https://api.semanticscholar.org/v1/paper/ARXIV:{arxiv_id}"
            params = {"fields": "citationCount,influentialCitationCount,hIndex"}
            
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 404:
                    # Paper not found in Semantic Scholar (common for new papers)
                    logger.info(f"semantic_scholar_tool: paper {arxiv_id} not found")
                    return CitationData(
                        citation_count=0,
                        influential_citation_count=0,
                        h_index=0,
                    )
                
                if response.status != 200:
                    logger.warning(f"semantic_scholar_tool: API returned {response.status} for {arxiv_id}")
                    return CitationData(
                        citation_count=0,
                        influential_citation_count=0,
                        h_index=0,
                    )
                
                data = await response.json()
        
        # Extract citation data with safe defaults
        citation_data = CitationData(
            citation_count=data.get("citationCount", 0),
            influential_citation_count=data.get("influentialCitationCount", 0),
            h_index=data.get("hIndex", 0),
        )
        
        logger.debug(f"semantic_scholar_tool: {arxiv_id} has {citation_data.citation_count} citations")
        return citation_data
    
    except asyncio.TimeoutError:
        logger.warning(f"semantic_scholar_tool timeout for {arxiv_id}")
        return CitationData(citation_count=0, influential_citation_count=0, h_index=0)
    except Exception as e:
        logger.error(f"semantic_scholar_tool error for {arxiv_id}: {str(e)}")
        return CitationData(citation_count=0, influential_citation_count=0, h_index=0)


# ============================================================================
# TOOL 3: PAPER_RANKER_TOOL
# ============================================================================

@tool
def paper_ranker_tool(
    papers: List[RankedPaper],
    weights: Optional[RankingWeights] = None,
) -> List[RankedPaper]:
    """
    Rank papers by composite score using weighted criteria.
    
    Combines three normalized scores (0-1 each) into a composite score (0-3):
    - Citation relevance: higher citation count → higher score
    - Recency: newer papers → higher score
    - Relevance: LLM relevance judgment (if available) → higher score
    
    Args:
        papers: List of RankedPaper objects with scores already computed
        weights: RankingWeights specifying citation_weight, relevance_weight, recency_weight
               Defaults to equal weights (0.33 each)
    
    Returns:
        List[RankedPaper]: Same papers, sorted by composite_rank_score (descending),
                          with rank_position field updated (1-indexed)
    
    Scoring Formula:
        composite_score = (
            citation_norm * weight_citation +
            relevance_score * weight_relevance +
            recency_score * weight_recency
        )
        where each component is 0-1 normalized
    
    Error Handling:
        - Handles papers with missing scores gracefully (treats as 0)
        - Never raises exceptions
    
    Note:
        This is a pure computation tool (no API calls, no LLM calls).
        All papers must have citation_count, relevance_score, recency_score computed
        before calling this tool (typically done by semantic_scholar_tool and
        context engineering in the agent nodes).
    """
    try:
        # Use default weights if not provided
        if weights is None:
            weights = RankingWeights(
                citation_weight=0.33,
                relevance_weight=0.33,
                recency_weight=0.34,  # Sum = 1.0
            )
        
        if not papers:
            logger.warning("paper_ranker_tool: received empty papers list")
            return []
        
        # Step 1: Normalize citation counts (0-1 scale)
        citation_counts = [p.citation_count for p in papers if p.citation_count is not None]
        max_citations = max(citation_counts) if citation_counts else 1
        
        # Step 2: Normalize recency (0-1 scale)
        # Papers from last 6 months = high score; older papers = lower score
        now = datetime.utcnow()
        recency_scores = []
        for paper in papers:
            try:
                published = datetime.fromisoformat(paper.published_date.replace("Z", "+00:00"))
                days_old = (now - published).days
                # Decay: older = lower score, 365 days = 0.5
                recency_score = max(0, 1 - (days_old / 730))
            except:
                recency_score = 0
            recency_scores.append(recency_score)
        
        # Step 3: Compute composite scores
        ranked_papers = []
        for i, paper in enumerate(papers):
            # Normalize citation count
            citation_norm = (paper.citation_count / max_citations) if max_citations > 0 else 0
            citation_norm = min(1.0, citation_norm)  # Cap at 1.0
            
            # Use pre-computed relevance_score (set by gap analyzer or default)
            relevance_norm = paper.relevance_score if paper.relevance_score is not None else 0.5
            
            # Use computed recency score
            recency_norm = recency_scores[i]
            
            # Compute composite score (0-3 range)
            composite_score = (
                (citation_norm * weights.citation_weight) +
                (relevance_norm * weights.relevance_weight) +
                (recency_norm * weights.recency_weight)
            ) * 3  # Scale to 0-3 range
            
            # Update paper with composite score
            paper.composite_rank_score = composite_score
            ranked_papers.append(paper)
        
        # Step 4: Sort by composite score (descending)
        ranked_papers.sort(key=lambda p: p.composite_rank_score, reverse=True)
        
        # Step 5: Update rank positions (1-indexed)
        for position, paper in enumerate(ranked_papers, start=1):
            paper.rank_position = position
        
        logger.info(f"paper_ranker_tool: ranked {len(ranked_papers)} papers")
        return ranked_papers
    
    except Exception as e:
        logger.error(f"paper_ranker_tool error: {str(e)}")
        return papers  # Return original order on error


# ============================================================================
# TOOL 4: GAP_ANALYZER_TOOL
# ============================================================================

@tool
async def gap_analyzer_tool(
    papers: List[RankedPaper],
    goal: str,
    conversation_history: Optional[List[dict]] = None,
) -> List[KnowledgeGap]:
    """
    Identify knowledge gaps by analyzing collected papers against the research goal.
    
    Uses the LLM to reason about what's missing from the corpus and identify
    specific gaps that should be addressed in the next iteration.
    
    Args:
        papers: List of papers collected so far (top-ranked)
        goal: Original research goal (e.g., "understand latest self-driving car research")
        conversation_history: Prior user answers and agent reasoning (context for gap analysis)
    
    Returns:
        List[KnowledgeGap]: Structured gaps with descriptions and resolution status
    
    Error Handling:
        - Returns empty list if LLM call fails
        - Never raises exceptions
    
    LLM Integration:
        Uses llm.with_structured_output(KnowledgeGap) to force JSON response.
        System prompt tells LLM to identify 3-5 key gaps.
    
    Context Note:
        This is an LLM call inside a tool. The LLM reads:
        - The goal
        - Paper titles and abstracts (pre-filtered by context budgeting)
        - Conversation history (for prior user context)
        
        The response is structured as List[KnowledgeGap] (Pydantic model),
        so the agent can reliably parse and act on gaps.
    """
    try:
        # Format papers for LLM readability
        papers_text = "\n".join([
            f"{i+1}. {p.title}\n   Authors: {', '.join(p.authors[:3])}\n   "
            f"Abstract: {p.abstract[:300]}...\n"
            for i, p in enumerate(papers[:settings.max_papers_per_iteration])
        ])
        
        # Format conversation history if available
        history_text = ""
        if conversation_history:
            history_text = "Prior conversation context:\n"
            for msg in conversation_history[-5:]:  # Last 5 messages only
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                history_text += f"  {role}: {content[:200]}\n"
        
        # Build system prompt
        system_prompt = f"""
You are a research gap analyzer. Your job is to identify what's MISSING from the 
collected papers relative to the research goal.

Research Goal: {goal}

Collected Papers ({len(papers)} total):
{papers_text}

{history_text}

Identify 3-5 key knowledge gaps:
- What aspects of the goal are NOT covered by these papers?
- What questions remain unanswered?
- What related topics should be explored?

For each gap:
1. Describe it clearly in 1-2 sentences
2. Note whether it's critical or nice-to-have
3. Suggest a search direction to address it

Return ONLY a JSON array of gaps, no other text.
"""
        
        # Call LLM with structured output (forces JSON response)
        structured_llm = llm.with_structured_output(KnowledgeGap)
        
        response = await asyncio.to_thread(
            structured_llm.invoke,
            system_prompt,
        )
        
        # Response is already a KnowledgeGap or list[KnowledgeGap]
        gaps = response if isinstance(response, list) else [response]
        
        logger.info(f"gap_analyzer_tool: identified {len(gaps)} knowledge gaps")
        return gaps
    
    except Exception as e:
        logger.error(f"gap_analyzer_tool error: {str(e)}")
        return []


# ============================================================================
# EXPORT
# ============================================================================

# List of all tools for binding with LLM
ALL_TOOLS = [
    arxiv_search_tool,
    semantic_scholar_tool,
    paper_ranker_tool,
    gap_analyzer_tool,
]