"""
Unit tests for app.rag.orchestration.pipeline.

Run:
    pytest tests/rag/test_pipeline.py -v
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from app.rag.orchestration.pipeline import RetrievalPipeline, RetrievalPipelineResult
from app.rag.query.query_processor import ProcessedQuery
from app.rag.retrieval.models import RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(id=id, text=f"text-{id}", score=score, source="dense")


def _mock_service(
    results: list[RetrievedChunk] | None = None,
) -> MagicMock:
    service = MagicMock()
    service.search.return_value = results or []
    return service


def _make_pipeline(
    results: list[RetrievedChunk] | None = None,
) -> tuple[RetrievalPipeline, MagicMock]:
    service = _mock_service(results)
    return RetrievalPipeline(service), service


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestPipelineReturnType:
    def test_returns_retrieval_pipeline_result(self) -> None:
        pipeline, _ = _make_pipeline()
        result = pipeline.run("test query")
        assert isinstance(result, RetrievalPipelineResult)

    def test_processed_query_is_processed_query_instance(self) -> None:
        pipeline, _ = _make_pipeline()
        result = pipeline.run("test query")
        assert isinstance(result.processed_query, ProcessedQuery)

    def test_results_is_list(self) -> None:
        pipeline, _ = _make_pipeline()
        assert isinstance(pipeline.run("test").results, list)

    def test_results_contains_retrieved_chunk_instances(self) -> None:
        pipeline, _ = _make_pipeline(results=[_chunk("a", 0.9)])
        for item in pipeline.run("test").results:
            assert isinstance(item, RetrievedChunk)


# ---------------------------------------------------------------------------
# processed_query content
# ---------------------------------------------------------------------------


class TestPipelineProcessedQuery:
    def test_original_query_preserved(self) -> None:
        pipeline, _ = _make_pipeline()
        result = pipeline.run("Únos Dítěte")
        assert result.processed_query.original_query == "Únos Dítěte"

    def test_normalized_query_is_lowercase(self) -> None:
        pipeline, _ = _make_pipeline()
        result = pipeline.run("ÚNOS DÍTĚTE")
        assert result.processed_query.normalized_query == "únos dítěte"

    def test_keywords_extracted(self) -> None:
        pipeline, _ = _make_pipeline()
        result = pipeline.run("únos dítěte ruska")
        assert "únos" in result.processed_query.keywords
        assert "dítěte" in result.processed_query.keywords

    def test_legal_concepts_populated_for_known_terms(self) -> None:
        pipeline, _ = _make_pipeline()
        result = pipeline.run("únos dítěte")
        assert len(result.processed_query.legal_concepts) > 0

    def test_empty_query_gives_empty_processed_fields(self) -> None:
        pipeline, _ = _make_pipeline()
        result = pipeline.run("")
        assert result.processed_query.normalized_query == ""
        assert result.processed_query.keywords == []
        assert result.processed_query.legal_concepts == []


# ---------------------------------------------------------------------------
# retrieval_service interaction
# ---------------------------------------------------------------------------


class TestPipelineServiceCalls:
    def test_retrieval_service_search_called_once(self) -> None:
        pipeline, service = _make_pipeline()
        pipeline.run("test")
        service.search.assert_called_once()

    def test_retrieval_service_receives_original_query(self) -> None:
        pipeline, service = _make_pipeline()
        pipeline.run("Únos Dítěte")
        call_args = service.search.call_args
        query_arg = call_args[0][0] if call_args[0] else call_args[1]["query"]
        assert query_arg == "Únos Dítěte"

    def test_top_k_forwarded_to_service(self) -> None:
        pipeline, service = _make_pipeline()
        pipeline.run("test", top_k=7)
        call_args = service.search.call_args
        top_k_arg = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("top_k")
        assert top_k_arg == 7

    def test_results_come_from_service(self) -> None:
        chunks = [_chunk("x", 0.9), _chunk("y", 0.8)]
        pipeline, _ = _make_pipeline(results=chunks)
        result = pipeline.run("test")
        assert result.results == chunks

    def test_empty_service_results_propagated(self) -> None:
        pipeline, _ = _make_pipeline(results=[])
        assert pipeline.run("test").results == []


# ---------------------------------------------------------------------------
# top_k
# ---------------------------------------------------------------------------


class TestPipelineTopK:
    def test_default_top_k_is_five(self) -> None:
        pipeline, service = _make_pipeline()
        pipeline.run("test")
        call_args = service.search.call_args
        top_k_arg = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("top_k")
        assert top_k_arg == 5

    def test_custom_top_k_forwarded(self) -> None:
        pipeline, service = _make_pipeline()
        pipeline.run("test", top_k=10)
        call_args = service.search.call_args
        top_k_arg = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("top_k")
        assert top_k_arg == 10


# ---------------------------------------------------------------------------
# Trace events
# ---------------------------------------------------------------------------


class TestPipelineTrace:
    def test_all_trace_events_emitted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        pipeline, _ = _make_pipeline(results=[_chunk("a", 0.9)])
        with caplog.at_level(
            logging.DEBUG, logger="app.rag.orchestration.pipeline"
        ):
            pipeline.run("únos dítěte")

        messages = [r.getMessage() for r in caplog.records]
        assert any("TRACE pipeline.start" in m for m in messages)
        assert any("TRACE pipeline.processed" in m for m in messages)
        assert any("TRACE pipeline.done" in m for m in messages)

    def test_trace_start_includes_query_and_top_k(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        pipeline, _ = _make_pipeline()
        with caplog.at_level(
            logging.DEBUG, logger="app.rag.orchestration.pipeline"
        ):
            pipeline.run("únos dítěte", top_k=3)

        start = next(
            r.getMessage()
            for r in caplog.records
            if "pipeline.start" in r.getMessage()
        )
        assert "únos dítěte" in start
        assert "top_k=3" in start

    def test_trace_processed_includes_keywords_and_concepts(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        pipeline, _ = _make_pipeline()
        with caplog.at_level(
            logging.DEBUG, logger="app.rag.orchestration.pipeline"
        ):
            pipeline.run("únos dítěte")

        processed = next(
            r.getMessage()
            for r in caplog.records
            if "pipeline.processed" in r.getMessage()
        )
        assert "keywords=" in processed
        assert "concepts=" in processed

    def test_trace_done_includes_result_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        pipeline, _ = _make_pipeline(
            results=[_chunk("a", 0.9), _chunk("b", 0.8)]
        )
        with caplog.at_level(
            logging.DEBUG, logger="app.rag.orchestration.pipeline"
        ):
            pipeline.run("test")

        done = next(
            r.getMessage()
            for r in caplog.records
            if "pipeline.done" in r.getMessage()
        )
        assert "result_count=2" in done

    def test_no_business_logic_in_pipeline_trace(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Pipeline trace events must come only from its own logger."""
        pipeline, _ = _make_pipeline(results=[_chunk("a", 0.9)])
        with caplog.at_level(
            logging.DEBUG, logger="app.rag.orchestration.pipeline"
        ):
            pipeline.run("test")

        pipeline_records = [
            r for r in caplog.records
            if r.name == "app.rag.orchestration.pipeline"
        ]
        assert len(pipeline_records) == 3  # start, processed, done
