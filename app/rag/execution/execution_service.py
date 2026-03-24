"""
Execution Service — runs each Planner step through retrieval and collects results.

Pure orchestrator: no LLM, no business logic, no mutation of input data.
Designed to sit between PlannerService and AnswerService / LLMService.

Usage:
    from app.rag.execution.execution_service import ExecutionService

    execution_service = ExecutionService(retrieval_service)
    result = execution_service.execute(plan)

    for step in result.steps_results:
        print(step["query"], len(step["results"]), "chunks")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger
from app.core.tracing import trace_event
from app.rag.planner.planner_service import Plan
from app.rag.retrieval.models import RetrievedChunk
from app.rag.retrieval.retrieval_service import RetrievalService

logger = get_logger(__name__)

StepResult = dict[str, Any]  # {"query": str, "reason": str, "results": list[RetrievedChunk]}


@dataclass
class ExecutionResult:
    steps_results: list[StepResult]

    def all_chunks(self) -> list[RetrievedChunk]:
        """Flatten all step results into a single deduplicated chunk list."""
        seen: dict[str, RetrievedChunk] = {}
        for step in self.steps_results:
            for chunk in step["results"]:
                if chunk.id not in seen or chunk.score > seen[chunk.id].score:
                    seen[chunk.id] = chunk
        return sorted(seen.values(), key=lambda c: c.score, reverse=True)


class ExecutionService:
    """Executes every step of a Plan and returns per-step retrieval results."""

    def __init__(self, retrieval_service: RetrievalService, top_k: int = 5) -> None:
        self._retrieval = retrieval_service
        self._top_k = top_k

    def execute(self, plan: Plan) -> ExecutionResult:
        logger.info("[execution] steps=%d", len(plan.steps))

        results: list[StepResult] = []
        for i, step in enumerate(plan.steps):
            trace_event(
                logger,
                "execution.step",
                step_index=i,
                query=step.query,
                reason=step.reason,
            )
            step_results = self._retrieval.search(step.query, top_k=self._top_k)
            trace_event(
                logger,
                "execution.step_done",
                step_index=i,
                num_results=len(step_results),
            )
            results.append(
                {
                    "query": step.query,
                    "reason": step.reason,
                    "results": step_results,
                }
            )

        trace_event(logger, "execution.done", total_steps=len(results))
        return ExecutionResult(steps_results=results)
