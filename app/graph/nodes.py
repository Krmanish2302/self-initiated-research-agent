"""
Node functions for the research agent graph.

Each node:
- Takes StateDict as input
- Returns dict[str, Any] with only the fields it updates
- LangGraph reducers merge updates (append for lists, overwrite for others)
- Has try-except to set status="error" on failure
- Is async for I/O-bound work (LLM calls, tool calls)

M10 additions:
  - clarification_node: now returns typed ClarifyingQuestion Pydantic objects
  - human_input_node: merges user answers into conversation_history
"""

import asyncio
import logging
import tiktoken
from typing import Any, Optional, List
from datetime import datetime

from app.schemas.models import (
    ResearchStrategy,
    RankedPaper,
    KnowledgeGap,
    ResearchBrief,
    PaperMetadata,
    CitationData,
    ClarifyingQuestion,
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

# Token threshold: if conversation_history exceeds this, compress old turns
HISTORY_TOKEN_THRESHOLD = 2000


# ============================================================================
# NODE 1: PLANNING_NODE
# ============================================================================

async def planning_node(state: StateDict) -> dict[str, Any]:
    """
    Create a research strategy by decomposing the goal into topics.

    CoALA fix: assembles working memory from all 4 memory types before calling LLM:
    - Semantic:    goal, user_preferences
    - Procedural:  failed_searches (what NOT to retry)
    - Episodic:    conversation_history (user clarifications)
    - Semantic:    already-collected paper titles (avoid topic overlap)

    Input: goal, user_preferences, failed_searches, conversation_history, papers
    Output: ResearchStrategy with topics, date_range, ranking_criteria, search_depth
    """
    try:
        logger.info(f"planning_node: decomposing goal '{state['goal']}'")

        iteration = state["iteration_count"]

        failed = state.get("failed_searches", [])
        failed_context = (
            f"\nFailed searches (DO NOT retry these): {failed}"
            if failed else ""
        )

        history = state.get("conversation_history", [])
        user_messages = [m["content"] for m in history if m.get("role") == "user"]
        history_context = (
            f"\nUser clarifications so far: {user_messages}"
            if user_messages else ""
        )

        collected_topics = list({p.title[:40] for p in state.get("papers", [])})
        collected_context = (
            f"\nTopics already covered (avoid duplicating): {collected_topics[:10]}"
            if collected_topics else ""
        )

        system_prompt = f"""
You are a research strategist. Iteration {iteration + 1} of research.

Research Goal: {state['goal']}

User Preferences: {state.get('user_preferences', {})}
{failed_context}
{history_context}
{collected_context}

Create a strategy with NEW topics not already covered above:
1. Topics (3-5 specific topics to search for)
2. Date range (start/end for paper filtering)
3. Ranking criteria (what matters: citations, recency, relevance)
4. Search depth (broad=shallow/wide, balanced=normal, deep=thorough/narrow)

Return ONLY valid JSON, no other text.
"""

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

    Input: strategy (with topics)
    Output: papers (List[RankedPaper] appended via operator.add reducer)
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

        topics_text = "\n".join([f"- {t}" for t in strategy.topics])

        query_generation_prompt = f"""
Given these research topics:
{topics_text}

Generate 3-5 specific search queries for ArXiv that would find relevant papers.
Each query should be 2-5 words, focused, and distinct from others.

Return ONLY a JSON array of strings like: ["query1", "query2", "query3"]
No other text.
"""

        query_response = await asyncio.to_thread(
            llm.invoke,
            query_generation_prompt,
        )

        import json
        import re
        try:
            if hasattr(query_response, "content"):
                response_text = query_response.content
            else:
                response_text = str(query_response)

            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                search_queries = json.loads(json_match.group())
            else:
                search_queries = strategy.topics
        except Exception as e:
            logger.warning(f"Failed to parse LLM query response: {e}, using topics as fallback")
            search_queries = strategy.topics

        logger.info(f"paper_collection_node: generated queries {search_queries}")

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
                await asyncio.sleep(1.0)
            except Exception as e:
                search_results.append(e)

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

        citation_tasks = [
            semantic_scholar_tool.coroutine(paper.arxiv_id)
            for paper in all_papers
        ]
        citation_results = await asyncio.gather(*citation_tasks, return_exceptions=True)

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
                relevance_score=0.5,
                recency_score=0.5,
                composite_rank_score=0.0,
                rank_position=1,
            )
            ranked_papers.append(ranked_paper)

        return {
            "papers": ranked_papers,
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
    Writes to ranked_papers (separate field) — NOT back to papers.
    """
    try:
        if not state.get("papers"):
            logger.warning("ranking_node: no papers to rank")
            return {"status": "no_papers_to_rank"}

        papers = state["papers"]
        logger.info(f"ranking_node: ranking {len(papers)} papers")

        ranked = paper_ranker_tool.func(papers)

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
# NODE 3.5: CONTEXT_BUDGETING_NODE
# ============================================================================

async def context_budgeting_node(state: StateDict) -> dict[str, Any]:
    """
    Prune agent state BEFORE gap_analysis_node sees it.

    1. ranked_papers > 20      → keep top-20 by composite_rank_score
    2. conversation_history    → token-aware trigger (> HISTORY_TOKEN_THRESHOLD)
                                  summarize old turns → summarized_history
    3. gaps > 5                → keep top-5 unresolved gaps
    """
    try:
        enc = tiktoken.encoding_for_model("gpt-4")
        updates: dict[str, Any] = {}

        # 1. PRUNE PAPERS
        ranked_papers = state.get("ranked_papers") or []
        if len(ranked_papers) > 20:
            before_tokens = len(enc.encode(
                " ".join(p.title + " " + p.abstract for p in ranked_papers)
            ))
            ranked_papers = sorted(
                ranked_papers,
                key=lambda p: p.composite_rank_score,
                reverse=True
            )[:20]
            after_tokens = len(enc.encode(
                " ".join(p.title + " " + p.abstract for p in ranked_papers)
            ))
            logger.info(
                f"context_budgeting_node: papers {len(state.get('ranked_papers', []))} → 20 | "
                f"tokens {before_tokens} → {after_tokens}"
            )
            updates["ranked_papers"] = ranked_papers

        # 2. SUMMARIZE CONVERSATION HISTORY (token-aware trigger)
        history = state.get("conversation_history", [])
        history_text = " ".join(m.get("content", "") for m in history)
        history_tokens = len(enc.encode(history_text))

        if history_tokens > HISTORY_TOKEN_THRESHOLD:
            old_turns = history[:-4]
            recent_turns = history[-4:]

            old_text = "\n".join(
                f"{m.get('role', 'unknown')}: {m.get('content', '')}"
                for m in old_turns
            )

            before_history_tokens = len(enc.encode(old_text))

            summary_prompt = f"""Summarize this research conversation history concisely (3-5 sentences).
Preserve: key user preferences, important clarifications, what topics were ruled out.
Discard: filler, repetition, back-and-forth confirmations.

History:
{old_text}

Return ONLY the summary, no other text."""

            summary_response = await asyncio.to_thread(llm.invoke, summary_prompt)
            summary_text = (
                summary_response.content
                if hasattr(summary_response, "content")
                else str(summary_response)
            )

            after_history_tokens = len(enc.encode(summary_text))
            logger.info(
                f"context_budgeting_node: history tokens {before_history_tokens} → "
                f"{after_history_tokens} (trigger threshold={HISTORY_TOKEN_THRESHOLD})"
            )

            existing_summary = state.get("summarized_history", "")
            new_summary = (
                f"{existing_summary}\n\n[[Summary of turns 1–{len(history)-4}]]:\n{summary_text}"
                if existing_summary
                else f"[Summary of earlier turns]:\n{summary_text}"
            )

            updates["conversation_history"] = recent_turns
            updates["summarized_history"] = new_summary

        # 3. PRUNE GAPS
        gaps = state.get("gaps", [])
        if len(gaps) > 5:
            unresolved = [g for g in gaps if not g.resolved]
            resolved = [g for g in gaps if g.resolved]
            pruned_gaps = (unresolved + resolved)[:5]
            logger.info(f"context_budgeting_node: gaps {len(gaps)} → 5")
            updates["gaps"] = pruned_gaps

        updates["status"] = "context_budgeted"
        return updates

    except Exception as e:
        logger.error(f"context_budgeting_node failed: {str(e)}")
        return {
            "status": "error",
            "last_error": f"context_budgeting_node: {str(e)}",
        }


# ============================================================================
# NODE 4: GAP_ANALYSIS_NODE
# ============================================================================

async def gap_analysis_node(state: StateDict) -> dict[str, Any]:
    """
    Identify knowledge gaps by analyzing papers against the goal.

    HITL NOTE: Graph pauses AFTER this node (interrupt_after=["gap_analysis"]).
    The full state (including gaps) is persisted to SQLite at this point.
    The API reads state.gaps to surface questions to the user.
    """
    try:
        if not state.get("papers"):
            logger.warning("gap_analysis_node: no papers to analyze")
            return {
                "gaps": [],
                "status": "no_papers_to_analyze",
            }

        papers_to_analyze = state.get("ranked_papers") or state["papers"]
        logger.info(f"gap_analysis_node: analyzing {len(papers_to_analyze)} papers")

        gaps = await gap_analyzer_tool.coroutine(
            papers=papers_to_analyze[:settings.max_papers_per_iteration],
            goal=state["goal"],
            conversation_history=state.get("conversation_history", []),
        )

        logger.info(f"gap_analysis_node: identified {len(gaps)} gaps → graph will pause here (HITL)")

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
# NODE 5: CLARIFICATION_NODE  (M10 — typed ClarifyingQuestion objects)
# ============================================================================

async def clarification_node(state: StateDict) -> dict[str, Any]:
    """
    Generate typed ClarifyingQuestion objects from identified gaps.

    M10 change: Returns List[ClarifyingQuestion] Pydantic objects instead of
    freeform strings. Each question has:
      - question: str          (the question text)
      - gap_id: str            (which KnowledgeGap it addresses)
      - question_type: str     ("preference" | "constraint" | "clarification")
      - required: bool         (must user answer this before agent continues?)

    These typed objects are stored in state.clarifying_questions and surfaced
    to the user via the API. The human_input_node then merges answers back.

    Graph pauses here (interrupt_after=["gap_analysis"]) — this node runs
    AFTER the human answers, as part of the resume flow.
    """
    try:
        gaps = state.get("gaps", [])
        if not gaps:
            logger.info("clarification_node: no gaps, skipping")
            return {"status": "no_gaps_to_clarify"}

        logger.info(f"clarification_node: generating typed questions for {len(gaps)} gaps")

        gaps_text = "\n".join([
            f"- [gap_id={g.gap_id}] {g.description}" for g in gaps
        ])

        question_prompt = f"""
The user is researching: {state['goal']}

Identified knowledge gaps:
{gaps_text}

Generate 2-3 targeted clarifying questions. For each question, specify:
- question: the question text
- gap_id: which gap_id this question addresses
- question_type: one of "preference" | "constraint" | "clarification"
  preference   = user's research priorities (e.g., "Do you care more about speed or accuracy?")
  constraint   = hard limits (e.g., "Only papers from 2023 onwards?")
  clarification = disambiguate vague goal (e.g., "Do you mean on-device or cloud inference?")
- required: true if agent cannot continue without this answer, false otherwise

Return ONLY valid JSON array:
[
  {{"question": "...", "gap_id": "...", "question_type": "preference", "required": true}},
  ...
]
No other text.
"""

        structured_llm = llm.with_structured_output(List[ClarifyingQuestion])

        questions: List[ClarifyingQuestion] = await asyncio.to_thread(
            structured_llm.invoke,
            question_prompt,
        )

        logger.info(f"clarification_node: generated {len(questions)} typed questions")

        return {
            "clarifying_questions": questions,
            "conversation_history": [
                {
                    "role": "agent",
                    "content": q.question,
                    "metadata": {
                        "gap_id": q.gap_id,
                        "question_type": q.question_type,
                        "required": q.required,
                    }
                }
                for q in questions
            ],
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
# NODE 5.5: HUMAN_INPUT_NODE  (M10 — new)
# ============================================================================

async def human_input_node(state: StateDict) -> dict[str, Any]:
    """
    Merge human answers into conversation_history and user_preferences.

    This node runs AFTER the human resumes the graph via:
        app.update_state(config, {"user_answers": answers})
        app.invoke(None, config)

    It reads state.user_answers (a list of dicts with question + answer),
    and writes them into conversation_history so all subsequent nodes
    (planning_node, gap_analysis_node) see the enriched context.

    State fields read:   user_answers (injected by update_state)
    State fields written: conversation_history (appended), user_preferences (merged)

    The None in invoke(None, config) tells LangGraph: "no new user message,
    just load the checkpoint and continue from where we paused."
    """
    try:
        user_answers = state.get("user_answers", [])

        if not user_answers:
            logger.info("human_input_node: no user_answers in state, skipping")
            return {"status": "no_user_input"}

        logger.info(f"human_input_node: merging {len(user_answers)} answers into state")

        # Build conversation_history entries from user answers
        new_history_entries = []
        preference_updates = {}

        for answer_obj in user_answers:
            question = answer_obj.get("question", "")
            answer = answer_obj.get("answer", "")
            question_type = answer_obj.get("question_type", "clarification")

            # All answers go into conversation_history
            new_history_entries.append({
                "role": "user",
                "content": answer,
                "metadata": {
                    "in_response_to": question,
                    "question_type": question_type,
                }
            })

            # Constraint/preference answers ALSO update user_preferences
            # (semantic memory — persists across future iterations)
            if question_type in ("preference", "constraint"):
                preference_updates[question] = answer

        # Merge into existing user_preferences (don't overwrite, update)
        existing_prefs = state.get("user_preferences", {}) or {}
        merged_prefs = {**existing_prefs, **preference_updates}

        logger.info(
            f"human_input_node: added {len(new_history_entries)} history entries, "
            f"updated {len(preference_updates)} preferences"
        )

        return {
            "conversation_history": new_history_entries,  # reducer appends
            "user_preferences": merged_prefs,
            "user_answers": [],   # clear after consuming
            "status": "user_input_merged",
        }

    except Exception as e:
        logger.error(f"human_input_node failed: {str(e)}")
        return {
            "status": "error",
            "last_error": f"human_input_node: {str(e)}",
        }


# ============================================================================
# NODE 6: SYNTHESIS_NODE
# ============================================================================

async def synthesis_node(state: StateDict) -> dict[str, Any]:
    """
    Generate the final research brief from collected papers and insights.

    Fix (M9): brief is explicitly returned in state update.
    Previously brief was created but never written back → silent data loss.
    """
    try:
        papers = state.get("ranked_papers") or state.get("papers", [])
        gaps = state.get("gaps", [])

        logger.info(f"synthesis_node: synthesizing brief from {len(papers)} papers")

        papers_text = "\n".join([
            f"{i+1}. {p.title}\n   Citations: {p.citation_count}, Score: {p.composite_rank_score:.2f}"
            for i, p in enumerate(papers[:10])
        ])

        gaps_text = "\n".join([f"- {g.description}" for g in gaps[:5]])

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

        structured_llm = llm.with_structured_output(ResearchBrief)

        brief = await asyncio.to_thread(
            structured_llm.invoke,
            synthesis_prompt,
        )

        brief.iterations_taken = state["iteration_count"]
        brief.total_papers_found = len(papers)
        brief.top_papers = papers[:20]

        logger.info(f"synthesis_node: created brief with {len(brief.key_findings)} findings")

        return {
            "research_brief": brief,
            "status": "complete",
        }

    except Exception as e:
        logger.error(f"synthesis_node failed: {str(e)}")
        return {
            "status": "error",
            "last_error": f"synthesis_node: {str(e)}",
            "iteration_count": state["iteration_count"] + 1,
        }
