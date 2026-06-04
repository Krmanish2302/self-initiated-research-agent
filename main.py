# main.py

import asyncio
import uuid
import logging
from app.graph.builder import agent_graph

# Basic logging so we can see what each node is doing
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


async def run_research(goal: str, max_iterations: int = 3):
    """
    Run the full research agent end-to-end.
    """

    # Minimum required initial state
    initial_state = {
    "goal": goal,
    "max_iterations": max_iterations,
    "status": "initialized",
    "iteration_count": 0,
    "papers": [],
    "gaps": [],
    "conversation_history": [],
    "search_queries_tried": [],
    "failed_searches": [],
    "user_preferences": {},
    "strategy": None,
    "last_error": None,
    "summarized_history": None,
}

    # Unique thread_id = unique session (checkpointer uses this as the key)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    logger.info(f"Starting research: '{goal}'")
    logger.info(f"Thread ID: {config['configurable']['thread_id']}")

    # Stream events so we see each node firing in real time
    async for event in agent_graph.astream(initial_state, config):
        node_name = list(event.keys())[0]
        node_output = event[node_name]

        print(f"\n{'='*50}")
        print(f"NODE: {node_name}")
        print(f"STATUS: {node_output.get('status', 'N/A')}")

        # Print papers count if papers were collected
        if "papers" in node_output:
            print(f"PAPERS THIS STEP: {len(node_output['papers'])}")

        # Print gaps if found
        if "gaps" in node_output:
            print(f"GAPS FOUND: {len(node_output['gaps'])}")
            for g in node_output["gaps"]:
                print(f"  - {g.description}")

        # Print any error
        if node_output.get("last_error"):
            print(f"ERROR: {node_output['last_error']}")

    # After the stream ends, fetch final state from checkpointer
    final_state = agent_graph.get_state(config)
    print(f"\n{'='*50}")
    print("FINAL STATE SUMMARY")
    print(f"Total papers collected: {len(final_state.values.get('papers', []))}")
    print(f"Total gaps identified: {len(final_state.values.get('gaps', []))}")
    print(f"Iterations run: {final_state.values.get('iteration_count', 0)}")
    print(f"Final status: {final_state.values.get('status', 'unknown')}")


if __name__ == "__main__":
    asyncio.run(run_research(
        goal="Understand recent advances in vision transformers",
        max_iterations=2,   # Keep it short for first test run
    ))