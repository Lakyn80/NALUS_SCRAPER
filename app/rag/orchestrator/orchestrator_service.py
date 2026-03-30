"""
Orchestrator Service — end-to-end RAG pipeline.

Connects all existing modules in the correct order:
    1. QueryRewriteService  (optional)
    2. PlannerService
    3. ExecutionService
    4. SynthesisService

No module is modified; the orchestrator only wires them together.

Usage:
    from app.rag.orchestrator.orchestrator_service import OrchestratorService

    service = OrchestratorService(
        planner=PlannerService(MockPlannerLLM()),
        execution=ExecutionService(retrieval),
        synthesis=SynthesisService(MockSynthesisLLM()),
        rewrite=QueryRewriteService(MockTextLLM()),   # optional
    )

    result = service.run("matka unesla dítě do Ruska")
    print(result.answer)
    print(result.sources)
    print(result.plan_steps)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.core.logging import get_logger
from app.core.tracing import trace_event
from app.rag.execution.execution_service import ExecutionResult, ExecutionService
from app.rag.planner.planner_service import Plan, PlanStep, PlannerService
from app.rag.retrieval.models import RetrievedChunk
from app.rag.rewrite.query_rewrite_service import QueryRewriteService
from app.rag.synthesis.synthesis_service import SynthesisService

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorResult:
    answer: str
    sources: list[str]
    plan_steps: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# OrchestratorService
# ---------------------------------------------------------------------------


class OrchestratorService:
    """Wires QueryRewrite → Planner → Execution → Synthesis into one call."""

    def __init__(
        self,
        planner: PlannerService,
        execution: ExecutionService,
        synthesis: SynthesisService,
        rewrite: Optional[QueryRewriteService] = None,
    ) -> None:
        self._planner = planner
        self._execution = execution
        self._synthesis = synthesis
        self._rewrite = rewrite

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        """Return raw dense retrieval hits without planner or synthesis."""
        trace_event(logger, "orchestrator.retrieve.start", query=query, top_k=top_k)

        effective_query = query
        if self._rewrite is not None:
            try:
                effective_query = self._rewrite.rewrite(query)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[orchestrator] rewrite failed (%s); using original query", exc)
                effective_query = query

        results = self._execution.retrieve_dense(effective_query, top_k=top_k)
        trace_event(
            logger,
            "orchestrator.retrieve.done",
            query=effective_query,
            total_chunks=len(results),
        )
        return results

    def run(self, query: str) -> OrchestratorResult:
        trace_event(logger, "orchestrator.start", query=query)

        # ------------------------------------------------------------------
        # 1. Rewrite (optional)
        # ------------------------------------------------------------------
        effective_query = query
        if self._rewrite is not None:
            try:
                effective_query = self._rewrite.rewrite(query)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[orchestrator] rewrite failed (%s); using original query", exc)
                effective_query = query
        trace_event(logger, "orchestrator.rewrite", original=query, effective=effective_query)

        # ------------------------------------------------------------------
        # 2. Plan
        # ------------------------------------------------------------------
        plan = _safe_plan(self._planner, effective_query)
        trace_event(logger, "orchestrator.plan", steps=len(plan.steps))

        # ------------------------------------------------------------------
        # 3. Execute
        # ------------------------------------------------------------------
        execution_result = _safe_execute(self._execution, plan)
        total_chunks = sum(
            len(s["results"]) for s in execution_result.steps_results
        )
        trace_event(
            logger, "orchestrator.execute", steps=len(execution_result.steps_results),
            total_chunks=total_chunks,
        )

        # ------------------------------------------------------------------
        # 4. Synthesize
        # ------------------------------------------------------------------
        synthesis_output = _safe_synthesize(self._synthesis, effective_query, execution_result)
        trace_event(
            logger, "orchestrator.synthesis",
            answer_length=len(synthesis_output.answer),
            sources=len(synthesis_output.sources),
        )

        result = OrchestratorResult(
            answer=synthesis_output.answer,
            sources=synthesis_output.sources,
            plan_steps=[step.query for step in plan.steps],
        )

        logger.info(
            "[orchestrator] done query=%s steps=%d sources=%d answer_length=%d",
            effective_query,
            len(plan.steps),
            len(result.sources),
            len(result.answer),
        )
        trace_event(
            logger, "orchestrator.done",
            steps=len(plan.steps),
            sources=len(result.sources),
            answer_length=len(result.answer),
        )

        return result


# ---------------------------------------------------------------------------
# Safe wrappers — fallback without propagating exceptions
# ---------------------------------------------------------------------------


def _safe_plan(planner: PlannerService, query: str) -> Plan:
    try:
        return planner.plan(query)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[orchestrator] planner failed (%s); using single-step fallback", exc)
        return Plan(steps=[PlanStep(query=query, reason="orchestrator fallback")])


def _safe_execute(execution: ExecutionService, plan: Plan) -> ExecutionResult:
    try:
        return execution.execute(plan)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[orchestrator] execution failed (%s); returning empty result", exc)
        return ExecutionResult(steps_results=[])


def _safe_synthesize(
    synthesis: SynthesisService, query: str, execution_result: ExecutionResult
) -> object:
    """Return SynthesisOutput, falling back to a zero-value object on failure."""
    try:
        return synthesis.synthesize(query, execution_result)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[orchestrator] synthesis failed (%s); returning empty answer", exc)
        return _EmptySynthesisOutput()


# Minimal stand-in when synthesis itself raises — not a real SynthesisOutput
# but duck-compatible (has .answer and .sources).
class _EmptySynthesisOutput:
    answer: str = ""
    sources: list = []
