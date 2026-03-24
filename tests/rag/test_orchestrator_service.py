"""
Unit tests for app.rag.orchestrator.orchestrator_service.

Run:
    pytest tests/rag/test_orchestrator_service.py -v
"""

from __future__ import annotations

import logging

import pytest

from app.rag.execution.execution_service import ExecutionResult, ExecutionService
from app.rag.orchestrator.orchestrator_service import OrchestratorResult, OrchestratorService
from app.rag.planner.planner_service import Plan, PlanStep, PlannerService
from app.rag.rewrite.query_rewrite_service import QueryRewriteService
from app.rag.retrieval.models import RetrievedChunk
from app.rag.synthesis.synthesis_service import SynthesisOutput, SynthesisService


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _chunk(id: str = "1", text: str = "text") -> RetrievedChunk:
    return RetrievedChunk(id=id, text=text, score=0.9, source="keyword")


class _FakePlanner:
    """Always returns a two-step plan."""

    def __init__(self, query_override: str | None = None) -> None:
        self._override = query_override
        self.calls: list[str] = []

    def plan(self, query: str) -> Plan:
        self.calls.append(query)
        q = self._override or query
        return Plan([PlanStep(q + " krok1", "r1"), PlanStep(q + " krok2", "r2")])


class _FailingPlanner:
    def plan(self, query: str) -> Plan:
        raise RuntimeError("planner exploded")


class _FakeExecution:
    """Returns fixed chunks for any query."""

    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self._chunks = chunks or [_chunk("A"), _chunk("B")]
        self.calls: list[Plan] = []

    def execute(self, plan: Plan) -> ExecutionResult:
        self.calls.append(plan)
        steps = [
            {"query": s.query, "reason": s.reason, "results": self._chunks}
            for s in plan.steps
        ]
        return ExecutionResult(steps_results=steps)


class _FailingExecution:
    def execute(self, plan: Plan) -> ExecutionResult:
        raise RuntimeError("execution exploded")


class _FakeSynthesis:
    """Returns a fixed answer."""

    def __init__(self, answer: str = "Syntetická odpověď") -> None:
        self._answer = answer
        self.calls: list[tuple[str, ExecutionResult]] = []

    def synthesize(self, query: str, execution_result: ExecutionResult) -> SynthesisOutput:
        self.calls.append((query, execution_result))
        sources = list({c.id for s in execution_result.steps_results for c in s["results"]})
        return SynthesisOutput(answer=self._answer, sources=sources)


class _FailingSynthesis:
    def synthesize(self, query: str, execution_result: ExecutionResult) -> SynthesisOutput:
        raise RuntimeError("synthesis exploded")


class _FakeRewrite:
    """Appends a suffix to mark the query was rewritten."""

    def __init__(self, suffix: str = " [přepsáno]") -> None:
        self._suffix = suffix
        self.calls: list[str] = []

    def rewrite(self, query: str) -> str:
        self.calls.append(query)
        return query + self._suffix


class _FailingRewrite:
    def rewrite(self, query: str) -> str:
        raise RuntimeError("rewrite exploded")


# ---------------------------------------------------------------------------
# Helpers to build a default working service
# ---------------------------------------------------------------------------


def _make_service(
    planner=None,
    execution=None,
    synthesis=None,
    rewrite=None,
) -> tuple[OrchestratorService, _FakePlanner, _FakeExecution, _FakeSynthesis]:
    p = planner or _FakePlanner()
    e = execution or _FakeExecution()
    s = synthesis or _FakeSynthesis()
    svc = OrchestratorService(
        planner=p,
        execution=e,
        synthesis=s,
        rewrite=rewrite,
    )
    return svc, p, e, s


# ---------------------------------------------------------------------------
# OrchestratorResult
# ---------------------------------------------------------------------------


class TestOrchestratorResult:
    def test_stores_answer(self) -> None:
        r = OrchestratorResult(answer="A", sources=["1"])
        assert r.answer == "A"

    def test_stores_sources(self) -> None:
        r = OrchestratorResult(answer="A", sources=["X", "Y"])
        assert r.sources == ["X", "Y"]

    def test_stores_plan_steps(self) -> None:
        r = OrchestratorResult(answer="A", sources=[], plan_steps=["q1", "q2"])
        assert r.plan_steps == ["q1", "q2"]

    def test_plan_steps_defaults_to_empty(self) -> None:
        r = OrchestratorResult(answer="A", sources=[])
        assert r.plan_steps == []


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_returns_orchestrator_result(self) -> None:
        svc, _, _, _ = _make_service()
        result = svc.run("únos dítěte")
        assert isinstance(result, OrchestratorResult)

    def test_answer_from_synthesis(self) -> None:
        svc, _, _, _ = _make_service(synthesis=_FakeSynthesis("Konkrétní odpověď"))
        result = svc.run("dotaz")
        assert result.answer == "Konkrétní odpověď"

    def test_sources_populated(self) -> None:
        svc, _, _, _ = _make_service()
        result = svc.run("dotaz")
        assert len(result.sources) > 0

    def test_plan_steps_populated(self) -> None:
        svc, _, _, _ = _make_service()
        result = svc.run("dotaz")
        assert len(result.plan_steps) >= 1

    def test_plan_steps_are_strings(self) -> None:
        svc, _, _, _ = _make_service()
        result = svc.run("dotaz")
        assert all(isinstance(s, str) for s in result.plan_steps)

    def test_planner_receives_query(self) -> None:
        svc, planner, _, _ = _make_service()
        svc.run("haagská úmluva")
        assert "haagská úmluva" in planner.calls

    def test_execution_receives_plan(self) -> None:
        svc, planner, execution, _ = _make_service()
        svc.run("dotaz")
        assert len(execution.calls) == 1

    def test_synthesis_receives_query(self) -> None:
        svc, _, _, synthesis = _make_service()
        svc.run("rodičovská odpovědnost")
        assert synthesis.calls[0][0] == "rodičovská odpovědnost"

    def test_synthesis_receives_execution_result(self) -> None:
        svc, _, _, synthesis = _make_service()
        svc.run("dotaz")
        assert isinstance(synthesis.calls[0][1], ExecutionResult)


# ---------------------------------------------------------------------------
# Rewrite — optional
# ---------------------------------------------------------------------------


class TestRewriteOptional:
    def test_no_rewrite_when_not_provided(self) -> None:
        svc, planner, _, _ = _make_service(rewrite=None)
        svc.run("původní dotaz")
        assert planner.calls[0] == "původní dotaz"

    def test_rewrite_applied_when_provided(self) -> None:
        rewrite = _FakeRewrite(suffix=" [přepsáno]")
        svc, planner, _, _ = _make_service(rewrite=rewrite)
        svc.run("původní dotaz")
        assert planner.calls[0] == "původní dotaz [přepsáno]"

    def test_rewrite_called_with_original_query(self) -> None:
        rewrite = _FakeRewrite()
        svc, _, _, _ = _make_service(rewrite=rewrite)
        svc.run("testovací dotaz")
        assert rewrite.calls == ["testovací dotaz"]

    def test_synthesis_gets_rewritten_query(self) -> None:
        rewrite = _FakeRewrite(suffix=" XXX")
        svc, _, _, synthesis = _make_service(rewrite=rewrite)
        svc.run("původní")
        assert synthesis.calls[0][0] == "původní XXX"

    def test_rewrite_failure_falls_back_to_original(self) -> None:
        svc, planner, _, _ = _make_service(rewrite=_FailingRewrite())
        svc.run("původní dotaz")
        assert planner.calls[0] == "původní dotaz"

    def test_rewrite_failure_does_not_raise(self) -> None:
        svc, _, _, _ = _make_service(rewrite=_FailingRewrite())
        result = svc.run("dotaz")
        assert isinstance(result, OrchestratorResult)


# ---------------------------------------------------------------------------
# Planner fallback
# ---------------------------------------------------------------------------


class TestPlannerFallback:
    def test_planner_failure_does_not_raise(self) -> None:
        svc, _, _, _ = _make_service(planner=_FailingPlanner())
        result = svc.run("dotaz")
        assert isinstance(result, OrchestratorResult)

    def test_planner_failure_uses_single_step(self) -> None:
        svc, _, execution, _ = _make_service(planner=_FailingPlanner())
        svc.run("původní dotaz")
        plan_used = execution.calls[0]
        assert len(plan_used.steps) == 1
        assert plan_used.steps[0].query == "původní dotaz"

    def test_planner_failure_still_returns_answer(self) -> None:
        svc, _, _, _ = _make_service(planner=_FailingPlanner())
        result = svc.run("dotaz")
        assert isinstance(result.answer, str)


# ---------------------------------------------------------------------------
# Execution fallback
# ---------------------------------------------------------------------------


class TestExecutionFallback:
    def test_execution_failure_does_not_raise(self) -> None:
        svc, _, _, _ = _make_service(execution=_FailingExecution())
        result = svc.run("dotaz")
        assert isinstance(result, OrchestratorResult)

    def test_execution_failure_gives_empty_sources(self) -> None:
        svc, _, _, _ = _make_service(
            execution=_FailingExecution(),
            synthesis=_FakeSynthesis("odpověď"),
        )
        result = svc.run("dotaz")
        # Empty ExecutionResult → synthesis gets no chunks → sources empty
        assert result.sources == []

    def test_execution_failure_answer_still_returned(self) -> None:
        svc, _, _, _ = _make_service(execution=_FailingExecution())
        result = svc.run("dotaz")
        assert isinstance(result.answer, str)


# ---------------------------------------------------------------------------
# Synthesis fallback
# ---------------------------------------------------------------------------


class TestSynthesisFallback:
    def test_synthesis_failure_does_not_raise(self) -> None:
        svc, _, _, _ = _make_service(synthesis=_FailingSynthesis())
        result = svc.run("dotaz")
        assert isinstance(result, OrchestratorResult)

    def test_synthesis_failure_returns_empty_answer(self) -> None:
        svc, _, _, _ = _make_service(synthesis=_FailingSynthesis())
        result = svc.run("dotaz")
        assert result.answer == ""

    def test_synthesis_failure_returns_empty_sources(self) -> None:
        svc, _, _, _ = _make_service(synthesis=_FailingSynthesis())
        result = svc.run("dotaz")
        assert result.sources == []

    def test_synthesis_failure_plan_steps_still_present(self) -> None:
        svc, _, _, _ = _make_service(synthesis=_FailingSynthesis())
        result = svc.run("dotaz")
        assert len(result.plan_steps) >= 1


# ---------------------------------------------------------------------------
# Logging / tracing
# ---------------------------------------------------------------------------


class TestLogging:
    def test_info_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        svc, _, _, _ = _make_service()
        with caplog.at_level(
            logging.INFO,
            logger="app.rag.orchestrator.orchestrator_service",
        ):
            svc.run("dotaz")
        msgs = [r.getMessage() for r in caplog.records]
        assert any("[orchestrator]" in m for m in msgs)

    def test_trace_start_emitted(self, caplog: pytest.LogCaptureFixture) -> None:
        svc, _, _, _ = _make_service()
        with caplog.at_level(
            logging.DEBUG,
            logger="app.rag.orchestrator.orchestrator_service",
        ):
            svc.run("dotaz")
        msgs = [r.getMessage() for r in caplog.records]
        assert any("orchestrator.start" in m for m in msgs)

    def test_trace_done_emitted(self, caplog: pytest.LogCaptureFixture) -> None:
        svc, _, _, _ = _make_service()
        with caplog.at_level(
            logging.DEBUG,
            logger="app.rag.orchestrator.orchestrator_service",
        ):
            svc.run("dotaz")
        msgs = [r.getMessage() for r in caplog.records]
        assert any("orchestrator.done" in m for m in msgs)

    def test_trace_plan_emitted(self, caplog: pytest.LogCaptureFixture) -> None:
        svc, _, _, _ = _make_service()
        with caplog.at_level(
            logging.DEBUG,
            logger="app.rag.orchestrator.orchestrator_service",
        ):
            svc.run("dotaz")
        msgs = [r.getMessage() for r in caplog.records]
        assert any("orchestrator.plan" in m for m in msgs)

    def test_trace_execute_emitted(self, caplog: pytest.LogCaptureFixture) -> None:
        svc, _, _, _ = _make_service()
        with caplog.at_level(
            logging.DEBUG,
            logger="app.rag.orchestrator.orchestrator_service",
        ):
            svc.run("dotaz")
        msgs = [r.getMessage() for r in caplog.records]
        assert any("orchestrator.execute" in m for m in msgs)

    def test_trace_synthesis_emitted(self, caplog: pytest.LogCaptureFixture) -> None:
        svc, _, _, _ = _make_service()
        with caplog.at_level(
            logging.DEBUG,
            logger="app.rag.orchestrator.orchestrator_service",
        ):
            svc.run("dotaz")
        msgs = [r.getMessage() for r in caplog.records]
        assert any("orchestrator.synthesis" in m for m in msgs)

    def test_trace_rewrite_emitted(self, caplog: pytest.LogCaptureFixture) -> None:
        svc, _, _, _ = _make_service(rewrite=_FakeRewrite())
        with caplog.at_level(
            logging.DEBUG,
            logger="app.rag.orchestrator.orchestrator_service",
        ):
            svc.run("dotaz")
        msgs = [r.getMessage() for r in caplog.records]
        assert any("orchestrator.rewrite" in m for m in msgs)

    def test_warning_on_planner_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        svc, _, _, _ = _make_service(planner=_FailingPlanner())
        with caplog.at_level(
            logging.WARNING,
            logger="app.rag.orchestrator.orchestrator_service",
        ):
            svc.run("dotaz")
        msgs = [r.getMessage() for r in caplog.records]
        assert any("planner" in m for m in msgs)

    def test_warning_on_execution_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        svc, _, _, _ = _make_service(execution=_FailingExecution())
        with caplog.at_level(
            logging.WARNING,
            logger="app.rag.orchestrator.orchestrator_service",
        ):
            svc.run("dotaz")
        msgs = [r.getMessage() for r in caplog.records]
        assert any("execution" in m for m in msgs)

    def test_warning_on_synthesis_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        svc, _, _, _ = _make_service(synthesis=_FailingSynthesis())
        with caplog.at_level(
            logging.WARNING,
            logger="app.rag.orchestrator.orchestrator_service",
        ):
            svc.run("dotaz")
        msgs = [r.getMessage() for r in caplog.records]
        assert any("synthesis" in m for m in msgs)
