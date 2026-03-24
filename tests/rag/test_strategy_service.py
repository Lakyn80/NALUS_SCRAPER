"""
Unit tests for app.rag.strategy.strategy_service.

Run:
    pytest tests/rag/test_strategy_service.py -v
"""

import pytest

from app.rag.strategy.strategy_service import StrategyDecision, StrategyService
from app.rag.retrieval.models import RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(id: str, score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(id=id, text="Ústavní soud rozhodl.", score=score, source="dense")


def _chunks(n: int) -> list[RetrievedChunk]:
    return [_chunk(str(i)) for i in range(n)]


# ---------------------------------------------------------------------------
# StrategyDecision
# ---------------------------------------------------------------------------


class TestStrategyDecision:
    def test_mode_stored(self) -> None:
        d = StrategyDecision("llm_summary", "multi-source synthesis")
        assert d.mode == "llm_summary"

    def test_reason_stored(self) -> None:
        d = StrategyDecision("no_results", "empty retrieval")
        assert d.reason == "empty retrieval"

    def test_equality(self) -> None:
        a = StrategyDecision("direct_answer", "single chunk")
        b = StrategyDecision("direct_answer", "single chunk")
        assert a == b

    def test_inequality_mode(self) -> None:
        a = StrategyDecision("no_results", "x")
        b = StrategyDecision("llm_summary", "x")
        assert a != b

    def test_repr_contains_mode(self) -> None:
        d = StrategyDecision("no_results", "empty retrieval")
        assert "no_results" in repr(d)


# ---------------------------------------------------------------------------
# no_results mode
# ---------------------------------------------------------------------------


class TestStrategyNoResults:
    def test_empty_list_returns_no_results(self) -> None:
        d = StrategyService().decide("q", [])
        assert d.mode == "no_results"

    def test_reason_is_empty_retrieval(self) -> None:
        d = StrategyService().decide("q", [])
        assert d.reason == "empty retrieval"

    def test_no_results_regardless_of_query(self) -> None:
        for query in ["únos", "rodičovská odpovědnost", ""]:
            d = StrategyService().decide(query, [])
            assert d.mode == "no_results"


# ---------------------------------------------------------------------------
# direct_answer mode
# ---------------------------------------------------------------------------


class TestStrategyDirectAnswer:
    def test_single_chunk_returns_direct_answer(self) -> None:
        d = StrategyService().decide("q", _chunks(1))
        assert d.mode == "direct_answer"

    def test_single_chunk_reason(self) -> None:
        d = StrategyService().decide("q", _chunks(1))
        assert d.reason == "single chunk"

    def test_two_chunks_returns_direct_answer(self) -> None:
        d = StrategyService().decide("q", _chunks(2))
        assert d.mode == "direct_answer"

    def test_two_chunks_reason_is_fallback(self) -> None:
        d = StrategyService().decide("q", _chunks(2))
        assert d.reason == "default fallback"

    def test_direct_answer_not_no_results(self) -> None:
        d = StrategyService().decide("q", _chunks(1))
        assert d.mode != "no_results"

    def test_direct_answer_not_llm_summary(self) -> None:
        d = StrategyService().decide("q", _chunks(2))
        assert d.mode != "llm_summary"


# ---------------------------------------------------------------------------
# llm_summary mode
# ---------------------------------------------------------------------------


class TestStrategyLLMSummary:
    def test_three_chunks_returns_llm_summary(self) -> None:
        d = StrategyService().decide("q", _chunks(3))
        assert d.mode == "llm_summary"

    def test_three_chunks_reason(self) -> None:
        d = StrategyService().decide("q", _chunks(3))
        assert d.reason == "multi-source synthesis"

    def test_five_chunks_returns_llm_summary(self) -> None:
        d = StrategyService().decide("q", _chunks(5))
        assert d.mode == "llm_summary"

    def test_ten_chunks_returns_llm_summary(self) -> None:
        d = StrategyService().decide("q", _chunks(10))
        assert d.mode == "llm_summary"

    def test_llm_summary_reason_unchanged_for_large_n(self) -> None:
        d = StrategyService().decide("q", _chunks(100))
        assert d.reason == "multi-source synthesis"


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


class TestStrategyBoundary:
    def test_boundary_0_is_no_results(self) -> None:
        assert StrategyService().decide("q", _chunks(0)).mode == "no_results"

    def test_boundary_1_is_direct_answer(self) -> None:
        assert StrategyService().decide("q", _chunks(1)).mode == "direct_answer"

    def test_boundary_2_is_direct_answer(self) -> None:
        assert StrategyService().decide("q", _chunks(2)).mode == "direct_answer"

    def test_boundary_3_is_llm_summary(self) -> None:
        assert StrategyService().decide("q", _chunks(3)).mode == "llm_summary"


# ---------------------------------------------------------------------------
# Pipeline integration — decision attached to result
# ---------------------------------------------------------------------------


class TestStrategyPipelineIntegration:
    """Verify pipeline.run() attaches the decision when strategy is injected."""

    def test_pipeline_without_strategy_has_none_decision(self) -> None:
        from unittest.mock import MagicMock
        from app.rag.orchestration.pipeline import RetrievalPipeline
        from app.rag.retrieval.retrieval_service import RetrievalService
        from app.rag.retrieval.dense_retriever import DenseRetriever
        from app.rag.retrieval.keyword_retriever import KeywordRetriever
        from app.rag.retrieval.embedder import MockEmbedder

        mock_client = MagicMock()
        mock_client.search.return_value = []
        dense = DenseRetriever(client=mock_client, collection_name="t", embedder=MockEmbedder())
        keyword = KeywordRetriever(corpus=[])
        service = RetrievalService(dense=dense, keyword=keyword)
        pipeline = RetrievalPipeline(service)  # no strategy

        result = pipeline.run("únos dítěte", top_k=3)
        assert result.decision is None

    def test_pipeline_with_strategy_has_decision(self) -> None:
        from unittest.mock import MagicMock
        from app.rag.orchestration.pipeline import RetrievalPipeline
        from app.rag.retrieval.retrieval_service import RetrievalService
        from app.rag.retrieval.dense_retriever import DenseRetriever
        from app.rag.retrieval.keyword_retriever import KeywordRetriever
        from app.rag.retrieval.embedder import MockEmbedder

        mock_client = MagicMock()
        mock_client.search.return_value = []
        dense = DenseRetriever(client=mock_client, collection_name="t", embedder=MockEmbedder())
        keyword = KeywordRetriever(corpus=[])
        service = RetrievalService(dense=dense, keyword=keyword)
        pipeline = RetrievalPipeline(service, strategy=StrategyService())

        result = pipeline.run("únos dítěte", top_k=3)
        assert isinstance(result.decision, StrategyDecision)

    def test_empty_retrieval_gives_no_results_decision(self) -> None:
        from unittest.mock import MagicMock
        from app.rag.orchestration.pipeline import RetrievalPipeline
        from app.rag.retrieval.retrieval_service import RetrievalService
        from app.rag.retrieval.dense_retriever import DenseRetriever
        from app.rag.retrieval.keyword_retriever import KeywordRetriever
        from app.rag.retrieval.embedder import MockEmbedder

        mock_client = MagicMock()
        mock_client.search.return_value = []
        dense = DenseRetriever(client=mock_client, collection_name="t", embedder=MockEmbedder())
        keyword = KeywordRetriever(corpus=[])
        service = RetrievalService(dense=dense, keyword=keyword)
        pipeline = RetrievalPipeline(service, strategy=StrategyService())

        result = pipeline.run("q", top_k=5)
        assert result.decision is not None
        assert result.decision.mode == "no_results"

    def test_strategy_decision_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging
        from unittest.mock import MagicMock
        from app.rag.orchestration.pipeline import RetrievalPipeline
        from app.rag.retrieval.retrieval_service import RetrievalService
        from app.rag.retrieval.dense_retriever import DenseRetriever
        from app.rag.retrieval.keyword_retriever import KeywordRetriever
        from app.rag.retrieval.embedder import MockEmbedder

        mock_client = MagicMock()
        mock_client.search.return_value = []
        dense = DenseRetriever(client=mock_client, collection_name="t", embedder=MockEmbedder())
        keyword = KeywordRetriever(corpus=[])
        service = RetrievalService(dense=dense, keyword=keyword)
        pipeline = RetrievalPipeline(service, strategy=StrategyService())

        with caplog.at_level(logging.INFO, logger="app.rag.orchestration.pipeline"):
            pipeline.run("únos dítěte", top_k=5)

        msgs = [r.getMessage() for r in caplog.records]
        assert any("[strategy]" in m for m in msgs)
        assert any("no_results" in m for m in msgs)
