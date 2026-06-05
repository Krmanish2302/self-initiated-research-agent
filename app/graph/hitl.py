"""
Human-in-the-Loop (HITL) resume and time-travel flows.

This module contains:
  1. run_until_pause()   — run the graph until it hits the HITL interrupt
  2. get_pending_questions() — read the paused state to surface questions to user
  3. resume_with_answers()  — inject user answers and resume the graph
  4. rewind_to_checkpoint() — time travel: rewind to any prior state
  5. fork_from_checkpoint() — time travel: fork a new thread from a prior state

Architecture note:
  The graph pauses AFTER gap_analysis_node via interrupt_after=["gap_analysis"].
  At that point:
    - Full AgentState is saved to research_agent.db (keyed by thread_id)
    - state.gaps contains the identified gaps
    - state.clarifying_questions may be populated if clarification_node has run
  The graph is NOT dead — it is a paused state machine.
  Calling invoke(None, config) resumes it exactly where it stopped.
"""

import logging
from typing import Any, Optional
from app.graph.builder import agent_graph
from app.graph.state import StateDict

logger = logging.getLogger(__name__)


# ============================================================================
# STEP 1 — Run graph until HITL pause
# ============================================================================

def run_until_pause(goal: str, thread_id: str, user_preferences: dict = None) -> dict:
    """
    Start the graph and run until it hits interrupt_after=["gap_analysis"].

    The graph will:
      planning → paper_collection → ranking → context_budgeting → gap_analysis
      → PAUSE (state saved to SQLite, keyed by thread_id)

    Args:
        goal:             The research goal string
        thread_id:        Unique session identifier (e.g., UUID or user ID)
        user_preferences: Optional dict of user preferences

    Returns:
        The state snapshot at the pause point (includes gaps, papers, etc.)
    """
    config = {"configurable": {"thread_id": thread_id}}

    initial_input = {
        "goal": goal,
        "user_preferences": user_preferences or {},
        "iteration_count": 0,
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

def get_pending_questions(thread_id: str) -> dict:
    """
    Read the current paused state for a thread and return the gaps
    and clarifying questions the agent wants to ask.

    Call this after run_until_pause() to surface questions to the user
    (e.g., via API response, UI, or CLI prompt).

    Args:
        thread_id: The session identifier used in run_until_pause()

    Returns:
        dict with:
          - gaps: List of KnowledgeGap objects
          - clarifying_questions: List of ClarifyingQuestion objects (may be empty)
          - status: current graph status
    """
    config = {"configurable": {"thread_id": thread_id}}

    # get_state() loads the latest checkpoint for this thread_id WITHOUT resuming
    snapshot = agent_graph.get_state(config)

    state = snapshot.values

    logger.info(
        f"get_pending_questions: thread_id={thread_id}, "
        f"gaps={len(state.get('gaps', []))}"
    )

    return {
        "gaps": state.get("gaps", []),
        "clarifying_questions": state.get("clarifying_questions", []),
        "status": state.get("status"),
        "iteration_count": state.get("iteration_count"),
        "papers_found": len(state.get("papers", [])),
    }


# ============================================================================
# STEP 3 — Resume with user answers
# ============================================================================

def resume_with_answers(thread_id: str, answers: list[dict]) -> dict:
    """
    Inject the human's answers into state and resume the graph.

    Flow:
      1. update_state() injects {"user_answers": answers} into the checkpoint
      2. invoke(None, config) resumes from checkpoint — None means "no new input,
         just continue"
      3. Graph continues: human_input_node → clarification_node → synthesis_node → END

    Args:
        thread_id: Session identifier
        answers:   List of answer dicts, each with:
                     - question: str       (the question text)
                     - answer: str         (the human's answer)
                     - question_type: str  ("preference" | "constraint" | "clarification")

    Returns:
        Final state after graph completes (includes research_brief)

    Example:
        answers = [
            {
                "question": "Are you interested in on-device or cloud inference?",
                "answer": "On-device, specifically for mobile",
                "question_type": "clarification",
            },
            {
                "question": "Do you want papers from 2023 onwards only?",
                "answer": "Yes, 2023 onwards",
                "question_type": "constraint",
            },
        ]
        result = resume_with_answers("session-abc", answers)
    """
    config = {"configurable": {"thread_id": thread_id}}

    logger.info(
        f"resume_with_answers: thread_id={thread_id}, "
        f"injecting {len(answers)} answers"
    )

    # STEP A: inject user answers into the paused state snapshot
    # update_state() patches the checkpoint WITHOUT running any nodes.
    # as_node="human_input" tells LangGraph to apply this update
    # as if it came from human_input_node (sets the "next" pointer correctly).
    agent_graph.update_state(
        config,
        {"user_answers": answers},
        as_node="human_input",
    )

    # STEP B: resume from checkpoint
    # invoke(None, config) = "don't add new input, just continue from the checkpoint"
    # Graph continues: human_input_node → clarification_node → synthesis_node → END
    result = agent_graph.invoke(None, config)

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

    Use cases:
      - Debugging: step backward to find where the agent went wrong
      - User correction: "go back and use a different search query"
      - Experimentation: try a different strategy from the same starting point

    How it works:
      LangGraph saves a snapshot after EVERY node execution (not just at pauses).
      Each snapshot has a unique checkpoint_id.
      Passing checkpoint_id in the config tells LangGraph to load THAT snapshot
      instead of the latest one.

    Args:
        thread_id:     Session identifier
        checkpoint_id: The checkpoint to rewind to (get from list_checkpoints())

    Returns:
        State after resuming from the rewound checkpoint

    Example:
        # Get all checkpoints for a session
        checkpoints = list(agent_graph.get_state_history(config))
        # Rewind to step 3 (planning complete, before paper collection)
        result = rewind_to_checkpoint("session-abc", checkpoints[3].config["checkpoint_id"])
    """
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,   # ← this is the rewind key
        }
    }

    logger.info(
        f"rewind_to_checkpoint: thread_id={thread_id}, "
        f"rewinding to checkpoint_id={checkpoint_id}"
    )

    # invoke(None, config) with checkpoint_id → loads THAT checkpoint, resumes from there
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

    Unlike rewind (which continues on the SAME thread), fork creates a
    NEW thread — so both the original and the fork can continue independently.

    Use cases:
      - A/B testing: try two different strategies from the same starting point
      - Safe experimentation: don't corrupt the original session
      - User branches: "what if I had said X instead of Y?"

    How it works:
      1. Load the source checkpoint from source_thread_id
      2. Write its state to new_thread_id (new independent session)
      3. Optionally override state fields (e.g., change the goal)
      4. Resume the new thread from there

    Args:
        source_thread_id: The original session to fork from
        checkpoint_id:    Which checkpoint to fork from
        new_thread_id:    The new session identifier for the fork
        state_override:   Optional dict to patch into the forked state
                          (e.g., {"goal": "different research goal"})

    Returns:
        State of the forked session after completion

    Example:
        # Fork at step 3, try a different strategy
        result = fork_from_checkpoint(
            source_thread_id="session-abc",
            checkpoint_id="ckpt_step3",
            new_thread_id="session-abc-fork-1",
            state_override={"strategy": different_strategy},
        )
    """
    source_config = {
        "configurable": {
            "thread_id": source_thread_id,
            "checkpoint_id": checkpoint_id,
        }
    }
    fork_config = {"configurable": {"thread_id": new_thread_id}}

    # Load the source checkpoint state
    source_snapshot = agent_graph.get_state(source_config)
    forked_state = dict(source_snapshot.values)

    # Apply any overrides
    if state_override:
        forked_state.update(state_override)

    logger.info(
        f"fork_from_checkpoint: forking {source_thread_id}@{checkpoint_id} "
        f"→ new thread {new_thread_id}"
    )

    # Write the forked state as the starting point for the new thread
    agent_graph.update_state(fork_config, forked_state)

    # Resume the new thread
    result = agent_graph.invoke(None, fork_config)

    logger.info(f"fork_from_checkpoint: fork completed. status={result.get('status')}")
    return result


# ============================================================================
# UTILITY — List all checkpoints for a session (for debugging / time travel)
# ============================================================================

def list_checkpoints(thread_id: str) -> list[dict]:
    """
    Return a list of all checkpoints for a thread, newest first.

    Each entry includes:
      - checkpoint_id: use this in rewind_to_checkpoint()
      - step:          which node just completed
      - timestamp:     when the snapshot was saved
      - status:        agent status at that point

    Example:
        checkpoints = list_checkpoints("session-abc")
        for ckpt in checkpoints:
            print(f"{ckpt['step']} @ {ckpt['timestamp']} → {ckpt['status']}")
    """
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
