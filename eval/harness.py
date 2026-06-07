# eval/harness.py

"""
Evaluation harness for the Self-Initiated Research Agent.

Runs all 3 evaluators against a completed AgentState and prints
a structured EvalReport.

Usage:
    python -m eval.harness --goal "transformers in drug discovery"

Or programmatically:
    from eval.harness import run_eval
    report = await run_eval(state, goal)
"""

import asyncio
import argparse
import json
from dataclasses import dataclass, field, asdict
from app.schemas.models import AgentState
from eval.evaluators import (
    research_quality_evaluator,
    question_quality_evaluator,
    efficiency_evaluator,
)


# ── REPORT SCHEMA ─────────────────────────────────────────────────────────────

@dataclass
class EvalReport:
    goal: str

    # Layer 1 — LLM-as-judge
    research_quality_score: float = 0.0
    research_quality_breakdown: dict = field(default_factory=dict)

    question_quality_score: float = 0.0
    question_quality_breakdown: dict = field(default_factory=dict)

    # Layer 2 — Pure heuristic
    efficiency_score: float = 0.0
    iterations_taken: int = 0
    repeated_failures: list = field(default_factory=list)
    loop_detected: bool = False

    # Composite
    overall_score: float = 0.0


# ── MAIN HARNESS ──────────────────────────────────────────────────────────────

async def run_eval(state: AgentState, goal: str) -> EvalReport:
    """
    Run all 3 evaluators against a completed AgentState.

    Layer 1 evaluators run concurrently (both need LLM calls).
    Layer 2 runs instantly — no LLM, just arithmetic.

    Returns a fully populated EvalReport.
    """
    report = EvalReport(goal=goal)

    # ── Layer 2 first (free, instant) ─────────────────────────────────────────
    eff = efficiency_evaluator(
        iteration_count=state.iteration_count,
        search_queries_tried=state.search_queries_tried,
        failed_searches=state.failed_searches,
    )
    report.efficiency_score = eff["score"]
    report.iterations_taken = eff["iterations"]
    report.repeated_failures = eff["repeated_failures"]
    report.loop_detected = eff["penalty_applied"]

    # ── Layer 1 — run both LLM evaluators concurrently ────────────────────────
    tasks = []

    if state.research_brief:
        tasks.append(
            research_quality_evaluator(
                brief=state.research_brief,
                goal=goal,
            )
        )
    else:
        tasks.append(asyncio.coroutine(lambda: {"score": 0.0, "raw_score": 0, "breakdown": {}})())

    if state.clarifying_questions and state.gaps:
        tasks.append(
            question_quality_evaluator(
                questions=state.clarifying_questions,
                gaps=state.gaps,
            )
        )
    else:
        tasks.append(asyncio.coroutine(lambda: {"score": 0.0, "raw_score": 0, "breakdown": {}})())

    rq_result, qq_result = await asyncio.gather(*tasks)

    report.research_quality_score = rq_result["score"]
    report.research_quality_breakdown = rq_result.get("breakdown", {})

    report.question_quality_score = qq_result["score"]
    report.question_quality_breakdown = qq_result.get("breakdown", {})

    # ── Composite score ───────────────────────────────────────────────────────
    # Weights: research quality 50%, efficiency 30%, question quality 20%
    report.overall_score = round(
        (report.research_quality_score * 0.50)
        + (report.efficiency_score * 0.30)
        + (report.question_quality_score * 0.20),
        3,
    )

    return report


# ── PRINT REPORT ──────────────────────────────────────────────────────────────

def print_report(report: EvalReport) -> None:
    divider = "─" * 52
    print(f"\n{divider}")
    print(f"  EVAL REPORT")
    print(f"  Goal: {report.goal[:48]}...")
    print(divider)

    print(f"\n  LAYER 1 — Output Quality")
    print(f"    Research Quality : {report.research_quality_score:.2f} / 1.0")
    for k, v in report.research_quality_breakdown.items():
        print(f"      {k:<22} {v} / 3")
    print(f"    Question Quality : {report.question_quality_score:.2f} / 1.0")
    for k, v in report.question_quality_breakdown.items():
        print(f"      {k:<22} {v} / 3")

    print(f"\n  LAYER 2 — Trajectory Efficiency")
    print(f"    Efficiency Score : {report.efficiency_score:.2f} / 1.0")
    print(f"    Iterations taken : {report.iterations_taken}")
    print(f"    Loop detected    : {'YES ⚠️' if report.loop_detected else 'No ✅'}")
    if report.repeated_failures:
        print(f"    Repeated queries : {report.repeated_failures}")

    print(divider)
    print(f"  OVERALL SCORE      {report.overall_score:.2f} / 1.0")
    print(f"{divider}\n")


# ── CLI ENTRYPOINT ────────────────────────────────────────────────────────────

async def _cli_main(goal: str) -> None:
    """
    CLI mode: builds a mock state for smoke testing.
    Replace with a real persisted AgentState for production runs.
    """
    from app.schemas.models import (
        ResearchBrief, RankedPaper, KnowledgeGap, ClarifyingQuestion
    )

    mock_state = AgentState(
        goal=goal,
        iteration_count=3,
        search_queries_tried=["transformers drug discovery", "protein folding LLM"],
        failed_searches=["quantum biology 2024"],  # not retried → no loop
        gaps=[
            KnowledgeGap(gap_id="gap_1", description="Limited work on multi-modal drug-protein interaction models", priority=1),
            KnowledgeGap(gap_id="gap_2", description="No benchmarks comparing LLM vs. traditional QSAR", priority=2),
        ],
        clarifying_questions=[
            ClarifyingQuestion(question="Are you focused on small molecule or biologics?", gap_id="gap_1", question_type="constraint"),
            ClarifyingQuestion(question="Which disease area should be prioritized?", gap_id="gap_2", question_type="preference"),
        ],
        research_brief=ResearchBrief(
            executive_summary="Transformers are increasingly applied in drug discovery for molecular generation and property prediction.",
            key_findings=[
                "BERT-based models outperform LSTM in SMILES property prediction",
                "AlphaFold 2 adoption has accelerated structure-based drug design",
                "Multi-modal models combining sequence + structure show promise",
            ],
            remaining_gaps=[
                "No large-scale benchmarks for LLM vs. traditional QSAR",
                "Limited multi-modal work on drug-protein interactions",
            ],
            next_steps=["Search for QSAR benchmark datasets", "Look for multi-modal drug-protein papers post-2023"],
            iterations_taken=3,
            total_papers_found=14,
        ),
    )

    report = await run_eval(mock_state, goal)
    print_report(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run agent evaluation harness")
    parser.add_argument("--goal", type=str, default="transformers in drug discovery")
    args = parser.parse_args()
    asyncio.run(_cli_main(args.goal))