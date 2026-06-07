"""
Human-in-the-Loop (HITL) resume and time-travel flows.

This module contains:
  1. run_until_pause()       — run the graph until it hits the HITL interrupt
  2. get_pending_questions() — read the paused state to surface questions to user
  3. resume_with_answers()   — inject user answers and resume the graph
  4. rewind_to_checkpoint()  — time travel: rewind to any prior state
  5. fork_from_checkpoint()  — time travel: fork a new thread from a prior state

Architecture note:
  The graph pauses AFTER gap_analysis_node via interrupt_after=["gap_analysis"].
  At that point:
    - Full AgentState is saved to research_agent.db (keyed by thread_id)
    - state.gaps contains the identified gaps
    - state.clarifying_questions may be populated if clarification_node has run
  The graph is NOT dead — it is a paused state machine.
  Calling invoke(None, config) resumes it exactly where it stopped.

  API call-sites pass (app, config) to get_pending_questions and
  resume_with_answers — these helpers accept that signature and delegate
  to the compiled app object so callers never hold a raw agent_graph ref.
"""

import logging
from typing import Any, Optional
from app.graph.state import StateDict

logger = logging.getLogger(__name__)


# ============================================================================
# STEP 1 — Run graph until HITL pause
# ============================================================================

def run_until_pause(goal: str, thread_id: str, user_preferences: dict = None) -> dict:
    """
    Start the graph and run until it hits interrupt_after=["gap_analysis"].

    The graph will:
      planning -> paper_collection -> ranking -> context_budgeting -> gap_analysis
      -> PAUSE (state saved to SQLite, keyed by thread_id)

    Args:
        goal:             The research goal string
        thread_id:        Unique session identifier (e.g., UUID or user ID)
        user_preferences: Optional dict of user preferences

    Returns:
        The state snapshot at the pause point (includes gaps, papers, etc.)
    """
    # Import here to avoid circular imports — builder imports nodes which import tools
    from app.graph.builder import agent_graph

    config = {"configurable": {"thread_id": thread_id}}

    initial_input = {
        "goal": goal,
        "user_preferences": user_preferences or {},
        "iteration_count": 0,
        "max_iterations": 5,
        "papers": [],
        "gaps": [],
        "conversation_history": [],
        "failed_searches": [],
        "search_queries_tried": [],
    }

    logger.info(f"run_until_pause: starting graph for thread_id={thread_id}")

    # invoke() runs until the interrupt. Returns the state at the pause point.
    result = agent_graph.invoke(initial_input, config)

    logger.info(
        f"run_until_pause: graph paused. "
        f"gaps={len(result.get('gaps', []))}, "
        f"status={result.get('status')}"
    )

    return result


# ============================================================================
# STEP 2 — Read paused state to surface questions to the user
# ============================================================================

def get_pending_questions(app, config: dict) -> list[str]:
    """
    Read the current paused state for a thread and return the clarifying
    question texts the agent wants to ask.

    Accepts the compiled app object and a LangGraph config dict so that
    callers (main.py) can pass (_research_app, config) without holding a
    separate reference to agent_graph.

    Args:
        app:    The compiled LangGraph app (sync or async)
        config: LangGraph config dict, e.g. {"configurable": {"thread_id": ...}}

    Returns:
        List of question strings (empty if none pending)
    """
    # Support both sync get_state and async aget_state callers
    # main.py already awaits aget_state separately; here we just read values.
    try:
        snapshot = app.get_state(config)
    except Exception:
        # Async app — caller must have already called aget_state; fall back
        return []

    state = snapshot.values

    # Prefer typed ClarifyingQuestion objects; fall back to gap descriptions
    cq = state.get("clarifying_questions") or []
    if cq:
        return [q.question if hasattr(q, "question") else str(q) for q in cq]

    gaps = state.get("gaps") or []
    return [g.description if hasattr(g, "description") else str(g) for g in gaps[:3]]


# ============================================================================
# STEP 3 — Resume with user answers
# ============================================================================

def resume_with_answers(app, config: dict, answers: list) -> dict:
    """
    Inject the human's answers into state and resume the graph.

    Accepts (app, config, answers) so callers pass the compiled app object
    rather than a raw thread_id string — consistent with get_pending_questions.

    Flow:
      1. update_state() injects {"user_answers": answers} into the checkpoint
      2. invoke(None, config) resumes from checkpoint — None means "no new input,
         just continue"
      3. Graph continues: human_input_node -> clarification_node -> synthesis_node -> END

    Args:
        app:     The compiled LangGraph app
        config:  LangGraph config dict
        answers: List of answer dicts, each with:
                   - question: str       (the question text)
                   - answer: str         (the human's answer)
                   - question_type: str  ("preference" | "constraint" | "clarification")

    Returns:
        Final state after graph completes (includes research_brief)
    """
    logger.info(
        f"resume_with_answers: thread={config.get('configurable', {}).get('thread_id')}, "
        f"injecting {len(answers)} answers"
    )

    # STEP A: inject user answers into the paused state snapshot
    # update_state() patches the checkpoint WITHOUT running any nodes.
    # as_node="human_input" tells LangGraph to apply this update
    # as if it came from human_input_node (sets the "next" pointer correctly).
    app.update_state(
        config,
        {"user_answers": answers},
        as_node="human_input",
    )

    # STEP B: resume from checkpoint
    # invoke(None, config) = "don't add new input, just continue from the checkpoint"
    result = app.invoke(None, config)

    logger.info(
        f"resume_with_answers: graph completed. "
        f"status={result.get('status')}, "
        f"brief={'present' if result.get('research_brief') else 'missing'}"
    )

    return result


# ============================================================================
# STEP 4 — Time Travel: Rewind to a prior checkpoint
# ============================================================================

def rewind_to_checkpoint(thread_id: str, checkpoint_id: str) -> dict:
    """
    Rewind the agent to a specific prior state and resume from there.
    """
    from app.graph.builder import agent_graph

    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
        }
    }

    logger.info(
        f"rewind_to_checkpoint: thread_id={thread_id}, "
        f"rewinding to checkpoint_id={checkpoint_id}"
    )

    result = agent_graph.invoke(None, config)

    logger.info(f"rewind_to_checkpoint: completed. status={result.get('status')}")
    return result


# ============================================================================
# STEP 5 — Time Travel: Fork a new thread from a prior checkpoint
# ============================================================================

def fork_from_checkpoint(
    source_thread_id: str,
    checkpoint_id: str,
    new_thread_id: str,
    state_override: dict = None,
) -> dict:
    """
    Fork a new independent session from a prior checkpoint.
    """
    from app.graph.builder import agent_graph

    source_config = {
        "configurable": {
            "thread_id": source_thread_id,
            "checkpoint_id": checkpoint_id,
        }
    }
    fork_config = {"configurable": {"thread_id": new_thread_id}}

    source_snapshot = agent_graph.get_state(source_config)
    forked_state = dict(source_snapshot.values)

    if state_override:
        forked_state.update(state_override)

    logger.info(
        f"fork_from_checkpoint: forking {source_thread_id}@{checkpoint_id} "
        f"-> new thread {new_thread_id}"
    )

    agent_graph.update_state(fork_config, forked_state)
    result = agent_graph.invoke(None, fork_config)

    logger.info(f"fork_from_checkpoint: fork completed. status={result.get('status')}")
    return result


# ============================================================================
# UTILITY — List all checkpoints for a session (for debugging / time travel)
# ============================================================================

def list_checkpoints(thread_id: str) -> list[dict]:
    """
    Return a list of all checkpoints for a thread, newest first.
    """
    from app.graph.builder import agent_graph

    config = {"configurable": {"thread_id": thread_id}}

    history = agent_graph.get_state_history(config)

    result = []
    for snapshot in history:
        result.append({
            "checkpoint_id": snapshot.config["configurable"].get("checkpoint_id"),
            "step": snapshot.next,
            "status": snapshot.values.get("status"),
            "iteration_count": snapshot.values.get("iteration_count"),
            "papers_found": len(snapshot.values.get("papers", [])),
            "gaps_found": len(snapshot.values.get("gaps", [])),
        })

    return result
