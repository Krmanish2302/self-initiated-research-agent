"""
Node functions for the research agent graph.

Each node:
- Takes StateDict as input
- Returns dict[str, Any] with only the fields it updates
- LangGraph reducers merge updates (append for lists, overwrite for others)
- Has try-except to set status="error" on failure
- Is async for I/O-bound work (LLM calls, tool calls)
"""

import asyncio
import logging
from typing import Any, Optional, List
from datetime import datetime

from app.schemas.models import (
    ResearchStrategy,
    RankedPaper,
    KnowledgeGap,
    ResearchBrief,
    PaperMetadata,
    CitationData,
)
from app.config import settings
from app.services.llm import llm
from app.tools.definitions import (
    arxiv_search_tool,
    semantic_scholar_tool,
    paper_ranker_tool,
    gap_analyzer_tool,
)
from app.graph.state import StateDict

logger = logging.getLogger(__name__)


# ============================================================================
# NODE 1: PLANNING_NODE
# ============================================================================

async def planning_node(state: StateDict) -> dict[str, Any]:
    """
    Create a research strategy by decomposing the goal into topics.
    
    Input: goal, user_preferences
    Output: ResearchStrategy with topics, date_range, ranking_criteria, search_depth
    
    Uses LLM with structured output to force JSON response.
    """
    try:
        logger.info(f"planning_node: decomposing goal '{state['goal']}'")
        
        system_prompt = f"""
You are a research strategist. Your job is to decompose a research goal into 
specific topics and a search strategy.

Research Goal: {state['goal']}

User Preferences: {state.get('user_preferences', {})}

Create a strategy with:
1. Topics (3-5 specific topics to search for)
2. Date range (start/end for paper filtering)
3. Ranking criteria (what matters: citations, recency, relevance)
4. Search depth (broad=shallow/wide, balanced=normal, deep=thorough/narrow)

Return ONLY valid JSON, no other text.
"""
        
        # Force structured output to ResearchStrategy Pydantic model
        structured_llm = llm.with_structured_output(ResearchStrategy)
        
        strategy = await asyncio.to_thread(
            structured_llm.invoke,
            system_prompt,
        )
        
        logger.info(f"planning_node: created strategy with topics {strategy.topics}")
        
        return {
            "strategy": strategy,
            "status": "strategy_created",
            "iteration_count": state["iteration_count"] + 1,
        }
    
    except Exception as e:
        logger.error(f"planning_node failed: {str(e)}")
        return {
            "status": "error",
            "last_error": f"planning_node: {str(e)}",
            "iteration_count": state["iteration_count"] + 1,
        }


# ============================================================================
# NODE 2: PAPER_COLLECTION_NODE
# ============================================================================

async def paper_collection_node(state: StateDict) -> dict[str, Any]:
    """
    Collect papers by searching ArXiv with multiple queries in parallel.
    
    Strategy: OPTION A (batch queries + parallel execution)
    1. Use LLM to generate search queries from topics
    2. Execute all queries in parallel via asyncio.gather()
    3. Return collected papers (reducer will append to state.papers)
    
    Input: strategy (with topics)
    Output: papers (List[PaperMetadata] + citation data)
    """
    try:
        if not state.get("strategy"):
            logger.error("paper_collection_node: no strategy found")
            return {
                "status": "no_results",
                "last_error": "No research strategy",
            }
        
        strategy = state["strategy"]
        logger.info(f"paper_collection_node: collecting papers for topics {strategy.topics}")
        
        # STEP 1: Use LLM to generate search queries from topics
        topics_text = "\n".join([f"- {t}" for t in strategy.topics])
        
        query_generation_prompt = f"""
Given these research topics:
{topics_text}

Generate 3-5 specific search queries for ArXiv that would find relevant papers.
Each query should be 2-5 words, focused, and distinct from others.

Return ONLY a JSON array of strings like: ["query1", "query2", "query3"]
No other text.
"""
        
        # Call LLM to get search queries
        query_response = await asyncio.to_thread(
            llm.invoke,
            query_generation_prompt,
        )
        
        # Parse JSON response (llm returns string or content)
        import json
        try:
            if hasattr(query_response, "content"):
                response_text = query_response.content
            else:
                response_text = str(query_response)
            
            # Extract JSON array from response
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                search_queries = json.loads(json_match.group())
            else:
                search_queries = strategy.topics  # Fallback to topics
        except Exception as e:
            logger.warning(f"Failed to parse LLM query response: {e}, using topics as fallback")
            search_queries = strategy.topics
        
        logger.info(f"paper_collection_node: generated queries {search_queries}")
        
        # STEP 2: Execute all queries sequentially to respect ArXiv rate limits
        search_results = []
        for q in search_queries:
            try:
                res = await arxiv_search_tool.coroutine(
                    query=q,
                    max_results=20,
                    date_range_start=strategy.date_range_start,
                    date_range_end=strategy.date_range_end,
                )
                search_results.append(res)
                await asyncio.sleep(1.0)  # Short pause to prevent API blocking
            except Exception as e:
                search_results.append(e)
        
        # Flatten results and handle errors
        all_papers = []
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                logger.error(f"Query {search_queries[i]} failed: {result}")
            else:
                all_papers.extend(result)
        
        logger.info(f"paper_collection_node: found {len(all_papers)} papers")
        
        if not all_papers:
            return {
                "status": "no_results",
                "last_error": f"No papers found for queries: {search_queries}",
                "search_queries_tried": search_queries,
            }
        
        # STEP 3: Fetch citation data for each paper (parallel)
        citation_tasks = [
            semantic_scholar_tool.coroutine(paper.arxiv_id)
            for paper in all_papers
        ]
        
        citation_results = await asyncio.gather(*citation_tasks, return_exceptions=True)
        
        # STEP 4: Convert to RankedPaper objects (preliminary ranking)
        ranked_papers = []
        for paper, citation_result in zip(all_papers, citation_results):
            if isinstance(citation_result, Exception):
                citation_data = CitationData(citation_count=0, influential_citation_count=0, h_index=0)
            else:
                citation_data = citation_result
            
            ranked_paper = RankedPaper(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                authors=paper.authors,
                abstract=paper.abstract,
                published_date=paper.published_date,
                url=paper.pdf_url,
                citation_count=citation_data.citation_count,
                relevance_score=0.5,  # Will be refined by gap analyzer
                recency_score=0.5,    # Will be refined by ranking node
                composite_rank_score=0.0,  # Will be set by ranking node
                rank_position=1,
            )
            ranked_papers.append(ranked_paper)
        
        return {
            "papers": ranked_papers,  # merge_papers reducer deduplicates by arxiv_id
            "status": "papers_collected",
            "search_queries_tried": search_queries,
            "iteration_count": state["iteration_count"] + 1,
        }
    
    except Exception as e:
        logger.error(f"paper_collection_node failed: {str(e)}")
        return {
            "status": "error",
            "last_error": f"paper_collection_node: {str(e)}",
            "iteration_count": state["iteration_count"] + 1,
        }


# ============================================================================
# NODE 3: RANKING_NODE
# ============================================================================

def ranking_node(state: StateDict) -> dict[str, Any]:
    """
    Rank papers by composite score (citation + recency + relevance).
    
    Input: papers (collected so far)
    Output: ranked_papers (sorted view) — does NOT write back to papers
    
    Key fix: ranking is a VIEW over papers, not a mutation.
    Writing back to papers field caused duplication via operator.add reducer.
    Now writes to ranked_papers (Optional field, plain overwrite).
    
    Note: Synchronous (pure math, no I/O)
    """
    try:
        if not state.get("papers"):
            logger.warning("ranking_node: no papers to rank")
            return {
                "status": "no_papers_to_rank",
            }
        
        papers = state["papers"]
        logger.info(f"ranking_node: ranking {len(papers)} papers")
        
        # Call the paper_ranker_tool
        ranked = paper_ranker_tool.func(papers)
        
        # Write to ranked_papers (separate field) — NOT back to papers
        # This prevents the duplication bug where operator.add would append
        # the same ranked list on top of already-collected papers
        return {
            "ranked_papers": ranked,
            "status": "papers_ranked",
            "iteration_count": state["iteration_count"] + 1,
        }
    
    except Exception as e:
        logger.error(f"ranking_node failed: {str(e)}")
        return {
            "status": "error",
            "last_error": f"ranking_node: {str(e)}",
            "iteration_count": state["iteration_count"] + 1,
        }


# ============================================================================
# NODE 4: GAP_ANALYSIS_NODE
# ============================================================================

async def gap_analysis_node(state: StateDict) -> dict[str, Any]:
    """
    Identify knowledge gaps by analyzing papers against the goal.
    
    Input: papers (top-ranked), goal, conversation_history
    Output: gaps (List[KnowledgeGap])
    
    Uses LLM with structured output.
    """
    try:
        if not state.get("papers"):
            logger.warning("gap_analysis_node: no papers to analyze")
            return {
                "gaps": [],
                "status": "no_papers_to_analyze",
            }
        
        # Use ranked_papers if available, fall back to papers
        papers_to_analyze = state.get("ranked_papers") or state["papers"]
        logger.info(f"gap_analysis_node: analyzing {len(papers_to_analyze)} papers")
        
        # Call gap analyzer tool (which calls LLM internally)
        gaps = await gap_analyzer_tool.coroutine(
            papers=papers_to_analyze[:settings.max_papers_per_iteration],
            goal=state["goal"],
            conversation_history=state.get("conversation_history", []),
        )
        
        logger.info(f"gap_analysis_node: identified {len(gaps)} gaps")
        
        return {
            "gaps": gaps,
            "status": "gaps_identified",
            "iteration_count": state["iteration_count"] + 1,
        }
    
    except Exception as e:
        logger.error(f"gap_analysis_node failed: {str(e)}")
        return {
            "gaps": [],
            "status": "error",
            "last_error": f"gap_analysis_node: {str(e)}",
            "iteration_count": state["iteration_count"] + 1,
        }


# ============================================================================
# NODE 5: CLARIFICATION_NODE
# ============================================================================

async def clarification_node(state: StateDict) -> dict[str, Any]:
    """
    Generate clarifying questions for the user based on identified gaps.
    
    Input: gaps, goal
    Output: questions (added to conversation_history)
    
    This node runs when the agent needs user input to continue research.
    It PAUSES here (via interrupt_after in builder.py).
    """
    try:
        gaps = state.get("gaps", [])
        if not gaps:
            logger.info("clarification_node: no gaps, skipping")
            return {
                "status": "no_gaps_to_clarify",
            }
        
        logger.info(f"clarification_node: generating questions for {len(gaps)} gaps")
        
        gaps_text = "\n".join([f"- {g.description}" for g in gaps])
        
        question_prompt = f"""
The user is researching: {state['goal']}

We've identified these knowledge gaps:
{gaps_text}

Generate 2-3 specific, actionable clarifying questions to ask the user.
Each question should help us understand:
1. Which gaps are most important to them
2. What specific aspect they care about
3. Any constraints or preferences

Return ONLY a JSON array of question strings like: ["question1?", "question2?"]
No other text.
"""
        
        # Call LLM for questions
        q_response = await asyncio.to_thread(
            llm.invoke,
            question_prompt,
        )
        
        # Parse questions
        import json
        import re
        try:
            if hasattr(q_response, "content"):
                response_text = q_response.content
            else:
                response_text = str(q_response)
            
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group())
            else:
                questions = ["Can you clarify your research goals?"]
        except Exception as e:
            logger.warning(f"Failed to parse questions: {e}")
            questions = ["Can you clarify your research goals?"]
        
        return {
            "conversation_history": [{"role": "agent", "content": q} for q in questions],
            "status": "awaiting_user_input",
            "iteration_count": state["iteration_count"] + 1,
        }
    
    except Exception as e:
        logger.error(f"clarification_node failed: {str(e)}")
        return {
            "status": "error",
            "last_error": f"clarification_node: {str(e)}",
            "iteration_count": state["iteration_count"] + 1,
        }


# ============================================================================
# NODE 6: SYNTHESIS_NODE
# ============================================================================

async def synthesis_node(state: StateDict) -> dict[str, Any]:
    """
    Generate the final research brief from collected papers and insights.
    
    Input: ranked_papers (preferred) or papers, gaps, goal, conversation_history
    Output: ResearchBrief (executive_summary, key_findings, remaining_gaps, etc.)
    """
    try:
        # Use ranked_papers if available (they have composite scores set)
        papers = state.get("ranked_papers") or state.get("papers", [])
        gaps = state.get("gaps", [])
        
        logger.info(f"synthesis_node: synthesizing brief from {len(papers)} papers")
        
        # Format papers for LLM
        papers_text = "\n".join([
            f"{i+1}. {p.title}\n   Citations: {p.citation_count}, Score: {p.composite_rank_score:.2f}"
            for i, p in enumerate(papers[:10])  # Top 10 papers
        ])
        
        # Format gaps
        gaps_text = "\n".join([
            f"- {g.description}"
            for g in gaps[:5]
        ])
        
        synthesis_prompt = f"""
You are a research synthesizer. Create a brief that summarizes the research.

Goal: {state['goal']}

Top Papers ({len(papers)} collected):
{papers_text}

Knowledge Gaps Identified:
{gaps_text}

Create a ResearchBrief with:
1. Executive summary (2-3 sentences)
2. 3-5 key findings
3. Remaining gaps (what we still don't know)
4. Next steps for deeper research

Return ONLY valid JSON with these fields:
- executive_summary: str
- key_findings: list[str]
- remaining_gaps: list[str]
- next_steps: list[str]

No other text.
"""
        
        # Call LLM for synthesis
        structured_llm = llm.with_structured_output(ResearchBrief)
        
        brief = await asyncio.to_thread(
            structured_llm.invoke,
            synthesis_prompt,
        )
        
        # Add metadata
        brief.iterations_taken = state["iteration_count"]
        brief.total_papers_found = len(papers)
        brief.top_papers = papers[:20]
        
        logger.info(f"synthesis_node: created brief with {len(brief.key_findings)} findings")
        
        return {
            "status": "complete",
        }
    
    except Exception as e:
        logger.error(f"synthesis_node failed: {str(e)}")
        return {
            "status": "error",
            "last_error": f"synthesis_node: {str(e)}",
            "iteration_count": state["iteration_count"] + 1,
        }
