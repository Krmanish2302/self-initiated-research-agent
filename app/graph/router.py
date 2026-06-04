"""
Conditional routing: decides which node runs next based on state.

The should_continue function is called after gap_analysis_node,
and it returns the name of the next node to execute.
"""

from app.graph.state import StateDict
import logging

logger = logging.getLogger(__name__)


def should_continue(state: StateDict) -> str:
    """
    Decide which node to run next based on current state.
    
    Logic:
    1. If max_iterations reached → synthesize and end
    2. If error occurred → handle error
    3. If no results → ask user for clarification
    4. If gaps resolved → synthesize and end
    5. Otherwise → continue collecting papers
    
    Returns:
        str: Next node name ("synthesis", "clarification", "paper_collection", etc.)
    """
    
    # Check termination conditions
    if state["iteration_count"] >= state["max_iterations"]:
        logger.info(f"should_continue: max_iterations ({state['max_iterations']}) reached")
        return "synthesis"
    
    if state["status"] == "error" or not state.get("strategy"):
        logger.warning(f"should_continue: error state or missing strategy: {state.get('last_error')}")
        return "synthesis"
        
    if state["status"] in ["no_papers_to_analyze", "no_papers_to_rank"]:
        logger.info(f"should_continue: status is {state['status']}, transitioning to synthesis")
        return "synthesis"
    
    if state["status"] == "no_results":
        logger.info("should_continue: no papers found, asking user for clarification")
        return "clarification"
    
    if state["status"] == "gaps_resolved":
        logger.info("should_continue: gaps resolved, moving to synthesis")
        return "synthesis"
    
    if state["status"] == "awaiting_user_input":
        logger.info("should_continue: waiting for user input (will pause here)")
        # This node will pause via interrupt_after
        return "synthesis"  # When resumed with user input, go to synthesis
    
    # Default: continue research loop
    logger.info("should_continue: continuing to collect more papers")
    return "paper_collection"
