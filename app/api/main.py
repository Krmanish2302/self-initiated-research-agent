# app/api/main.py - PART 1: Pydantic models + app setup

"""
FastAPI layer for the Self-Initiated Research Agent.
Exposes 3 endpoints: start, respond (HITL), and status.
Uses Server-Sent Events (SSE) for streaming agent progress.
"""

import uuid
import json
import asyncio
from typing import Any, AsyncIterator, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from app.graph.builder import build_graph
from app.graph.hitl import get_pending_questions, resume_with_answers
from app.schemas.models import ResearchBrief
import logging

logger = logging.getLogger(__name__)

# ── APP SETUP ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Self-Initiated Research Agent",
    description="Streams agent progress via SSE. HITL-aware.",
    version="1.0.0",
)

# Singleton — compiled once
_research_app = build_graph()


# ── PYDANTIC REQUEST / RESPONSE MODELS ───────────────────────────────────────

class ResearchGoalRequest(BaseModel):
    """Input to POST /research"""
    goal: str = Field(..., min_length=10, description="Research goal — at least 10 chars")
    max_iterations: int = Field(default=5, ge=1, le=10)


class UserAnswerRequest(BaseModel):
    """Input to POST /research/{thread_id}/respond"""
    answers: list[str] = Field(..., min_length=1, description="Answers to agent's questions")


class AgentStatusResponse(BaseModel):
    """Output of GET /research/{thread_id}/status"""
    thread_id: str
    status: str                          # "running" | "paused" | "completed" | "error"
    iteration_count: int
    papers_found: int
    pending_questions: list[str]         # non-empty only when status="paused"
    research_brief: Optional[ResearchBrief] = None


class SSEEvent(BaseModel):
    """Single SSE event pushed to client"""
    event: str       # "node_started" | "papers_found" | "questions_ready" | "complete" | "error"
    data: dict[str, Any]

# app/api/main.py - PART 2: SSE helper + three endpoints

# ── SSE HELPER ────────────────────────────────────────────────────────────────

async def event_stream(
    goal: str,
    thread_id: str,
    input_state: dict,
) -> AsyncIterator[str]:
    """
    Runs the agent and yields SSE-formatted strings.
    Closes the stream when agent pauses (HITL) or completes.
    """
    config = {"configurable": {"thread_id": thread_id}}

    def format_sse(event: str, data: dict) -> str:
        """SSE wire format: 'event: X\ndata: {json}\n\n'"""
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    try:
        async for event in _research_app.astream_events(
            input_state, config=config, version="v2"
        ):
            kind = event["event"]
            name = event.get("name", "")

            # Node started
            if kind == "on_chain_start" and name in (
                "planning", "paper_collection", "ranking",
                "context_budgeting", "gap_analysis", "synthesis"
            ):
                yield format_sse("node_started", {"node": name})

            # Papers collected
            elif kind == "on_chain_end" and name == "paper_collection":
                papers = event["data"]["output"].get("papers", [])
                yield format_sse("papers_found", {"count": len(papers)})

            # HITL pause — agent has questions
            elif kind == "on_chain_end" and name == "gap_analysis":
                state = await _research_app.aget_state(config)
                if state.next == ():  # graph paused
                    questions = get_pending_questions(
                        _research_app, config
                    )
                    yield format_sse("questions_ready", {
                        "questions": questions,
                        "thread_id": thread_id,
                    })
                    return  # close SSE stream — HITL takes over

            # Synthesis complete
            elif kind == "on_chain_end" and name == "synthesis":
                output = event["data"]["output"]
                brief = output.get("research_brief")
                yield format_sse("complete", {
                    "research_brief": brief.model_dump() if brief else None
                })
                return  # close SSE stream — done

    except Exception as e:
        logger.error(f"event_stream error: {e}")
        yield format_sse("error", {"message": str(e)})


# ── ENDPOINT 1: START ─────────────────────────────────────────────────────────

@app.post("/research")
async def start_research(request: ResearchGoalRequest):
    """
    Starts a new research session.
    Streams SSE events until agent pauses (HITL) or completes.
    Returns thread_id in the questions_ready or complete event.
    """
    thread_id = str(uuid.uuid4())

    return StreamingResponse(
        event_stream(
            goal=request.goal,
            thread_id=thread_id,
            input_state={
                "goal": request.goal,
                "max_iterations": request.max_iterations,
            },
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Thread-ID": thread_id},
    )


# ── ENDPOINT 2: RESPOND (HITL RESUME) ────────────────────────────────────────

@app.post("/research/{thread_id}/respond")
async def respond_to_questions(thread_id: str, request: UserAnswerRequest):
    """
    Submits user answers to a paused agent.
    Resumes the agent and streams continuation via SSE.
    """
    config = {"configurable": {"thread_id": thread_id}}

    # Verify the thread exists and is actually paused
    try:
        state = await _research_app.aget_state(config)
    except Exception:
        raise HTTPException(status_code=404, detail="Thread not found")

    if state.next != ():  # not paused
        raise HTTPException(
            status_code=400,
            detail="Agent is not paused. Cannot submit answers."
        )

    # Inject answers into checkpoint via update_state
    resume_with_answers(_research_app, config, request.answers)

    # Resume and stream continuation
    return StreamingResponse(
        event_stream(
            goal="",           # goal already in state
            thread_id=thread_id,
            input_state=None,  # None = resume from checkpoint
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


# ── ENDPOINT 3: STATUS ────────────────────────────────────────────────────────

@app.get("/research/{thread_id}/status")
async def get_status(thread_id: str) -> AgentStatusResponse:
    """
    Polls current agent state.
    Use this to check if agent is paused, running, or complete.
    """
    config = {"configurable": {"thread_id": thread_id}}

    try:
        snapshot = await _research_app.aget_state(config)
    except Exception:
        raise HTTPException(status_code=404, detail="Thread not found")

    state = snapshot.values
    is_paused = snapshot.next == ()

    pending_questions = []
    if is_paused:
        pending_questions = get_pending_questions(_research_app, config)

    status = (
        "paused" if is_paused and pending_questions
        else "completed" if state.get("research_brief")
        else "error" if state.get("status") == "error"
        else "running"
    )

    return AgentStatusResponse(
        thread_id=thread_id,
        status=status,
        iteration_count=state.get("iteration_count", 0),
        papers_found=len(state.get("papers", [])),
        pending_questions=pending_questions,
        research_brief=state.get("research_brief"),
    )


# ── GLOBAL EXCEPTION HANDLER ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return HTTPException(status_code=500, detail="Internal agent error")