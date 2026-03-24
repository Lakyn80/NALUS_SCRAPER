"""
Unit tests for app.rag.execution.execution_service.

Run:
    pytest tests/rag/test_execution_service.py -v
"""

import logging

import pytest

from app.rag.execution.execution_service import ExecutionResult, ExecutionService
from app.rag.planner.planner_service import Plan, PlanStep
from app.rag.retrieval.models import RetrievedChunk
from app.rag.retrieval.retrieval_service import RetrievalService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(id: str, score: float = 0.8, text: str = "text") -> RetrievedChunk:
    return RetrievedChunk(id=id, text=text, score=score, source="keyword")


def _plan(*queries: str) -> Plan:
    return Plan([PlanStep(query=q, reason=f"reason for {q}") for q in queries])


class _FakeRetrieval:
    """Returns a fixed list of chunks per query."""

    def __init__(self, corpus: dict[str, list[RetrievedChunk]] | None = None) -> None:
        self._corpus = corpus or {}
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        self.calls.append((query, top_k))
        return self._corpus.get(query, [])


def _make_service(corpus: dict | None = None) -> tuple[ExecutionService, _FakeRetrieval]:
    retrieval = _FakeRetrieval(corpus)
    # ExecutionService accepts anything with .search() — structural typing
    service = ExecutionService(retrieval)  # type: ignore[arg-type]
    return service, retrieval


# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------


class TestExecutionResult:
    def test_stores_steps_results(self) -> None:
        step = {"query": "q", "reason": "r", "results": []}
        result = ExecutionResult(steps_results=[step])
        assert result.steps_results == [step]

    def test_empty_steps_results(self) -> None:
        result = ExecutionResult(steps_results=[])
        assert result.steps_results == []

    def test_all_chunks_flattens_results(self) -> None:
        chunks_a = [_chunk("1", 0.9), _chunk("2", 0.7)]
        chunks_b = [_chunk("3", 0.8)]
        result = ExecutionResult(steps_results=[
            {"query": "q1", "reason": "r1", "results": chunks_a},
            {"query": "q2", "reason": "r2", "results": chunks_b},
        ])
        all_chunks = result.all_chunks()
        assert {c.id for c in all_chunks} == {"1", "2", "3"}

    def test_all_chunks_deduplicates_by_id(self) -> None:
        chunk_low = _chunk("X", score=0.5)
        chunk_high = _chunk("X", score=0.9)
        result = ExecutionResult(steps_results=[
            {"query": "q1", "reason": "r", "results": [chunk_low]},
            {"query": "q2", "reason": "r", "results": [chunk_high]},
        ])
        all_chunks = result.all_chunks()
        assert len(all_chunks) == 1
        assert all_chunks[0].score == pytest.approx(0.9)

    def test_all_chunks_sorted_descending(self) -> None:
        result = ExecutionResult(steps_results=[
            {"query": "q", "reason": "r", "results": [
                _chunk("a", 0.4), _chunk("b", 0.9), _chunk("c", 0.6)
            ]},
        ])
        scores = [c.score for c in result.all_chunks()]
        assert scores == sorted(scores, reverse=True)

    def test_all_chunks_empty_when_no_results(self) -> None:
        result = ExecutionResult(steps_results=[
            {"query": "q", "reason": "r", "results": []},
        ])
        assert result.all_chunks() == []


# ---------------------------------------------------------------------------
# ExecutionService — executes all steps
# ---------------------------------------------------------------------------


class TestExecutionServiceSteps:
    def test_executes_all_steps(self) -> None:
        service, retrieval = _make_service()
        service.execute(_plan("q1", "q2", "q3"))
        assert len(retrieval.calls) == 3

    def test_each_step_query_passed_to_retrieval(self) -> None:
        service, retrieval = _make_service()
        service.execute(_plan("únos dítěte", "Haagská úmluva"))
        queried = [q for q, _ in retrieval.calls]
        assert "únos dítěte" in queried
        assert "Haagská úmluva" in queried

    def test_order_preserved(self) -> None:
        service, retrieval = _make_service()
        service.execute(_plan("first", "second", "third"))
        queried = [q for q, _ in retrieval.calls]
        assert queried == ["first", "second", "third"]

    def test_empty_plan_returns_empty_result(self) -> None:
        service, _ = _make_service()
        result = service.execute(Plan([]))
        assert result.steps_results == []

    def test_single_step_plan(self) -> None:
        service, retrieval = _make_service()
        service.execute(_plan("only step"))
        assert len(retrieval.calls) == 1


# ---------------------------------------------------------------------------
# ExecutionService — return structure
# ---------------------------------------------------------------------------


class TestExecutionServiceReturnStructure:
    def test_returns_execution_result(self) -> None:
        service, _ = _make_service()
        result = service.execute(_plan("q"))
        assert isinstance(result, ExecutionResult)

    def test_steps_results_count_matches_plan(self) -> None:
        service, _ = _make_service()
        result = service.execute(_plan("q1", "q2", "q3"))
        assert len(result.steps_results) == 3

    def test_each_step_result_has_query_key(self) -> None:
        service, _ = _make_service()
        result = service.execute(_plan("q"))
        assert "query" in result.steps_results[0]

    def test_each_step_result_has_reason_key(self) -> None:
        service, _ = _make_service()
        result = service.execute(_plan("q"))
        assert "reason" in result.steps_results[0]

    def test_each_step_result_has_results_key(self) -> None:
        service, _ = _make_service()
        result = service.execute(_plan("q"))
        assert "results" in result.steps_results[0]

    def test_query_field_matches_plan_step(self) -> None:
        service, _ = _make_service()
        result = service.execute(_plan("mezinárodní únos"))
        assert result.steps_results[0]["query"] == "mezinárodní únos"

    def test_reason_field_matches_plan_step(self) -> None:
        service, _ = _make_service()
        plan = Plan([PlanStep("q", "specifický důvod")])
        result = service.execute(plan)
        assert result.steps_results[0]["reason"] == "specifický důvod"

    def test_results_field_is_list(self) -> None:
        service, _ = _make_service()
        result = service.execute(_plan("q"))
        assert isinstance(result.steps_results[0]["results"], list)


# ---------------------------------------------------------------------------
# ExecutionService — retrieval results passed through
# ---------------------------------------------------------------------------


class TestExecutionServiceResultPassthrough:
    def test_retrieval_chunks_appear_in_step_result(self) -> None:
        chunks = [_chunk("A"), _chunk("B")]
        service, _ = _make_service({"únos dítěte": chunks})
        result = service.execute(_plan("únos dítěte"))
        assert result.steps_results[0]["results"] == chunks

    def test_each_step_gets_its_own_chunks(self) -> None:
        corpus = {
            "q1": [_chunk("X")],
            "q2": [_chunk("Y"), _chunk("Z")],
        }
        service, _ = _make_service(corpus)
        result = service.execute(_plan("q1", "q2"))
        assert len(result.steps_results[0]["results"]) == 1
        assert len(result.steps_results[1]["results"]) == 2

    def test_empty_retrieval_gives_empty_results_list(self) -> None:
        service, _ = _make_service()
        result = service.execute(_plan("no match"))
        assert result.steps_results[0]["results"] == []


# ---------------------------------------------------------------------------
# ExecutionService — top_k forwarded
# ---------------------------------------------------------------------------


class TestExecutionServiceTopK:
    def test_default_top_k_forwarded(self) -> None:
        service, retrieval = _make_service()
        service.execute(_plan("q"))
        _, top_k = retrieval.calls[0]
        assert top_k == 5

    def test_custom_top_k_forwarded(self) -> None:
        retrieval = _FakeRetrieval()
        service = ExecutionService(retrieval, top_k=3)  # type: ignore[arg-type]
        service.execute(_plan("q"))
        _, top_k = retrieval.calls[0]
        assert top_k == 3


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestExecutionServiceLogging:
    def test_execution_info_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        service, _ = _make_service()
        with caplog.at_level(logging.INFO, logger="app.rag.execution.execution_service"):
            service.execute(_plan("q1", "q2"))
        msgs = [r.getMessage() for r in caplog.records]
        assert any("[execution]" in m for m in msgs)

    def test_log_contains_step_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        service, _ = _make_service()
        with caplog.at_level(logging.INFO, logger="app.rag.execution.execution_service"):
            service.execute(_plan("q1", "q2", "q3"))
        log = next(r.getMessage() for r in caplog.records if "[execution]" in r.getMessage())
        assert "steps=3" in log

    def test_trace_events_emitted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        service, _ = _make_service()
        with caplog.at_level(logging.DEBUG, logger="app.rag.execution.execution_service"):
            service.execute(_plan("q"))
        msgs = [r.getMessage() for r in caplog.records]
        assert any("execution.step" in m for m in msgs)
        assert any("execution.done" in m for m in msgs)

    def test_empty_plan_still_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        service, _ = _make_service()
        with caplog.at_level(logging.INFO, logger="app.rag.execution.execution_service"):
            service.execute(Plan([]))
        msgs = [r.getMessage() for r in caplog.records]
        assert any("[execution]" in m for m in msgs)
