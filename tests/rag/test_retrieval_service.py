"""
Unit tests for app.rag.retrieval.retrieval_service.

Run:
    pytest tests/rag/test_retrieval_service.py -v
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from app.rag.retrieval.models import RetrievedChunk
from app.rag.retrieval.retrieval_service import RetrievalService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(id: str, score: float, source: str = "dense") -> RetrievedChunk:
    return RetrievedChunk(id=id, text=f"text-{id}", score=score, source=source)


def _make_service(
    dense_results: list[RetrievedChunk] | None = None,
    keyword_results: list[RetrievedChunk] | None = None,
) -> tuple[RetrievalService, MagicMock, MagicMock]:
    dense = MagicMock()
    dense.retrieve.return_value = dense_results or []
    keyword = MagicMock()
    keyword.retrieve.return_value = keyword_results or []
    return RetrievalService(dense=dense, keyword=keyword), dense, keyword


# ---------------------------------------------------------------------------
# Basic flow
# ---------------------------------------------------------------------------


class TestRetrievalServiceFlow:
    def test_returns_list_of_retrieved_chunks(self) -> None:
        service, _, _ = _make_service(dense_results=[_chunk("a", 0.9)])
        result = service.search("únos dítěte")
        assert isinstance(result, list)

    def test_returns_retrieved_chunk_instances(self) -> None:
        service, _, _ = _make_service(dense_results=[_chunk("a", 0.9)])
        for item in service.search("únos dítěte"):
            assert isinstance(item, RetrievedChunk)

    def test_dense_retriever_called_once(self) -> None:
        service, dense, _ = _make_service()
        service.search("test query")
        dense.retrieve.assert_called_once()

    def test_keyword_retriever_called_once(self) -> None:
        service, _, keyword = _make_service()
        service.search("test query")
        keyword.retrieve.assert_called_once()

    def test_dense_receives_normalized_query(self) -> None:
        service, dense, _ = _make_service()
        service.search("ÚNOS DÍTĚTE")
        call_args = dense.retrieve.call_args
        query_arg = call_args[0][0] if call_args[0] else call_args[1]["query"]
        assert query_arg == "únos dítěte"

    def test_keyword_receives_joined_keywords(self) -> None:
        service, _, keyword = _make_service()
        # "do" (2 chars) gets filtered by process_query; "únos" and "dítěte" remain
        service.search("únos dítěte do ruska")
        call_args = keyword.retrieve.call_args
        query_arg = call_args[0][0] if call_args[0] else call_args[1]["query"]
        assert "únos" in query_arg
        assert "dítěte" in query_arg

    def test_empty_query_returns_empty_list(self) -> None:
        service, _, _ = _make_service()
        result = service.search("")
        assert result == []

    def test_both_retrievers_empty_returns_empty(self) -> None:
        service, _, _ = _make_service(dense_results=[], keyword_results=[])
        assert service.search("test") == []


# ---------------------------------------------------------------------------
# top_k propagation
# ---------------------------------------------------------------------------


class TestRetrievalServiceTopK:
    def test_top_k_passed_to_dense(self) -> None:
        service, dense, _ = _make_service()
        service.search("test", top_k=3)
        call_args = dense.retrieve.call_args
        top_k_arg = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("top_k")
        assert top_k_arg == 3

    def test_top_k_passed_to_keyword(self) -> None:
        service, _, keyword = _make_service()
        service.search("test", top_k=3)
        call_args = keyword.retrieve.call_args
        top_k_arg = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("top_k")
        assert top_k_arg == 3

    def test_result_does_not_exceed_top_k(self) -> None:
        dense = [_chunk(f"d{i}", 0.9 - i * 0.05) for i in range(8)]
        keyword = [_chunk(f"k{i}", 0.85 - i * 0.05) for i in range(8)]
        service, _, _ = _make_service(dense_results=dense, keyword_results=keyword)
        result = service.search("test", top_k=5)
        assert len(result) <= 5

    def test_default_top_k_is_five(self) -> None:
        dense = [_chunk(f"d{i}", 0.9 - i * 0.05) for i in range(10)]
        service, _, _ = _make_service(dense_results=dense)
        result = service.search("test")
        assert len(result) <= 5


class TestRetrievalServiceDenseOnly:
    def test_dense_only_calls_dense_retriever(self) -> None:
        service, dense, keyword = _make_service(dense_results=[_chunk("a", 0.9)])

        result = service.search_dense("ÚNOS DÍTĚTE", top_k=3)

        assert len(result) == 1
        dense.retrieve.assert_called_once()
        keyword.retrieve.assert_not_called()

    def test_dense_only_uses_normalized_query(self) -> None:
        service, dense, _ = _make_service(dense_results=[_chunk("a", 0.9)])

        service.search_dense("ÚNOS DÍTĚTE", top_k=3)

        call_args = dense.retrieve.call_args
        query_arg = call_args[0][0] if call_args[0] else call_args[1]["query"]
        assert query_arg == "únos dítěte"


# ---------------------------------------------------------------------------
# Fusion behaviour
# ---------------------------------------------------------------------------


class TestRetrievalServiceFusion:
    def test_combines_dense_and_keyword_results(self) -> None:
        dense = [_chunk("dense-1", 0.9)]
        keyword = [_chunk("kw-1", 0.7, source="keyword")]
        service, _, _ = _make_service(dense_results=dense, keyword_results=keyword)
        result = service.search("test", top_k=10)
        ids = {r.id for r in result}
        assert "dense-1" in ids
        assert "kw-1" in ids

    def test_no_duplicate_ids_in_result(self) -> None:
        shared = _chunk("shared", 0.9, source="dense")
        shared_kw = _chunk("shared", 0.6, source="keyword")
        service, _, _ = _make_service(
            dense_results=[shared], keyword_results=[shared_kw]
        )
        result = service.search("test", top_k=10)
        ids = [r.id for r in result]
        assert len(ids) == len(set(ids))

    def test_higher_score_wins_on_duplicate(self) -> None:
        service, _, _ = _make_service(
            dense_results=[_chunk("shared", 0.9)],
            keyword_results=[_chunk("shared", 0.5, source="keyword")],
        )
        result = service.search("test", top_k=10)
        shared = next(r for r in result if r.id == "shared")
        assert shared.score == 0.9

    def test_results_sorted_by_score_desc(self) -> None:
        dense = [_chunk("a", 0.7), _chunk("b", 0.9)]
        keyword = [_chunk("c", 0.8, source="keyword")]
        service, _, _ = _make_service(dense_results=dense, keyword_results=keyword)
        result = service.search("test", top_k=10)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Query processing integration
# ---------------------------------------------------------------------------


class TestRetrievalServiceQueryProcessing:
    def test_process_query_called_with_original_query(self) -> None:
        service, _, _ = _make_service()
        with patch(
            "app.rag.retrieval.retrieval_service.process_query",
            wraps=__import__(
                "app.rag.query.query_processor", fromlist=["process_query"]
            ).process_query,
        ) as mock_pq:
            service.search("Únos Dítěte Do Ruska")
            mock_pq.assert_called_once_with("Únos Dítěte Do Ruska")

    def test_legal_concepts_appear_in_trace(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        service, _, _ = _make_service()
        with caplog.at_level(
            logging.DEBUG,
            logger="app.rag.retrieval.retrieval_service",
        ):
            service.search("únos dítěte")

        processed_msg = next(
            r.getMessage()
            for r in caplog.records
            if "retrieval.processed" in r.getMessage()
        )
        assert "concepts=" in processed_msg


# ---------------------------------------------------------------------------
# Trace events
# ---------------------------------------------------------------------------


class TestRetrievalServiceTrace:
    def test_all_trace_events_emitted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        service, _, _ = _make_service(
            dense_results=[_chunk("a", 0.9)],
            keyword_results=[_chunk("b", 0.7, source="keyword")],
        )
        with caplog.at_level(
            logging.DEBUG,
            logger="app.rag.retrieval.retrieval_service",
        ):
            service.search("únos dítěte")

        messages = [r.getMessage() for r in caplog.records]
        assert any("TRACE retrieval.start" in m for m in messages)
        assert any("TRACE retrieval.processed" in m for m in messages)
        assert any("TRACE retrieval.dense" in m for m in messages)
        assert any("TRACE retrieval.keyword" in m for m in messages)
        assert any("TRACE retrieval.fused" in m for m in messages)

    def test_trace_start_includes_query(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        service, _, _ = _make_service()
        with caplog.at_level(
            logging.DEBUG,
            logger="app.rag.retrieval.retrieval_service",
        ):
            service.search("únos dítěte")

        start = next(
            r.getMessage()
            for r in caplog.records
            if "retrieval.start" in r.getMessage()
        )
        assert "únos dítěte" in start

    def test_trace_dense_includes_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        service, _, _ = _make_service(
            dense_results=[_chunk("a", 0.9), _chunk("b", 0.8)]
        )
        with caplog.at_level(
            logging.DEBUG,
            logger="app.rag.retrieval.retrieval_service",
        ):
            service.search("test")

        dense_msg = next(
            r.getMessage()
            for r in caplog.records
            if "retrieval.dense" in r.getMessage()
        )
        assert "count=2" in dense_msg

    def test_trace_fused_includes_final_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        service, _, _ = _make_service(
            dense_results=[_chunk("a", 0.9)],
            keyword_results=[_chunk("b", 0.7, source="keyword")],
        )
        with caplog.at_level(
            logging.DEBUG,
            logger="app.rag.retrieval.retrieval_service",
        ):
            service.search("test", top_k=10)

        fused_msg = next(
            r.getMessage()
            for r in caplog.records
            if "retrieval.fused" in r.getMessage()
        )
        assert "count=2" in fused_msg
