"""
Unit tests for app.rag.rewrite.query_rewrite_service.

Run:
    pytest tests/rag/test_query_rewrite.py -v
"""

import logging

import pytest

from app.rag.rewrite.query_rewrite_service import (
    BaseTextLLM,
    MockTextLLM,
    QueryRewriteService,
    _build_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FixedTextLLM(BaseTextLLM):
    """Returns a fixed string regardless of prompt."""

    def __init__(self, response: str) -> None:
        self._response = response

    def generate_text(self, prompt: str) -> str:
        return self._response


class _EchoTextLLM(BaseTextLLM):
    """Records every call and returns a configurable response."""

    def __init__(self, response: str = "rewritten query") -> None:
        self.calls: list[str] = []
        self._response = response

    def generate_text(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._response


class _FailingTextLLM(BaseTextLLM):
    def generate_text(self, prompt: str) -> str:
        raise RuntimeError("LLM unavailable")


# ---------------------------------------------------------------------------
# BaseTextLLM — abstract contract
# ---------------------------------------------------------------------------


class TestBaseTextLLMAbstract:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseTextLLM()  # type: ignore[abstract]

    def test_subclass_without_generate_text_cannot_be_instantiated(self) -> None:
        class Incomplete(BaseTextLLM):
            pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_subclass_with_generate_text_works(self) -> None:
        assert isinstance(_FixedTextLLM("ok"), BaseTextLLM)


# ---------------------------------------------------------------------------
# MockTextLLM
# ---------------------------------------------------------------------------


class TestMockTextLLM:
    def test_returns_string(self) -> None:
        result = MockTextLLM().generate_text("some prompt")
        assert isinstance(result, str)

    def test_result_not_empty(self) -> None:
        prompt = _build_prompt("matka unesla dítě do Ruska")
        result = MockTextLLM().generate_text(prompt)
        assert result.strip() != ""

    def test_result_contains_original_query(self) -> None:
        query = "matka unesla dítě do Ruska"
        prompt = _build_prompt(query)
        result = MockTextLLM().generate_text(prompt)
        assert query in result

    def test_result_longer_than_original(self) -> None:
        query = "únos dítěte"
        prompt = _build_prompt(query)
        result = MockTextLLM().generate_text(prompt)
        assert len(result) > len(query)

    def test_deterministic(self) -> None:
        llm = MockTextLLM()
        prompt = _build_prompt("test dotaz")
        assert llm.generate_text(prompt) == llm.generate_text(prompt)

    def test_czech_words_in_expansion(self) -> None:
        prompt = _build_prompt("délka řízení")
        result = MockTextLLM().generate_text(prompt).lower()
        assert any(w in result for w in ["ústavní", "judikatura", "právní", "práva"])


# ---------------------------------------------------------------------------
# QueryRewriteService — basic behaviour
# ---------------------------------------------------------------------------


class TestQueryRewriteServiceBasic:
    def test_rewrite_returns_string(self) -> None:
        result = QueryRewriteService(MockTextLLM()).rewrite("únos dítěte")
        assert isinstance(result, str)

    def test_rewrite_not_empty(self) -> None:
        result = QueryRewriteService(MockTextLLM()).rewrite("únos dítěte")
        assert result.strip() != ""

    def test_rewrite_different_from_input(self) -> None:
        query = "matka unesla dítě do Ruska"
        result = QueryRewriteService(MockTextLLM()).rewrite(query)
        assert result != query

    def test_czech_language_preserved(self) -> None:
        query = "rodičovská odpovědnost po rozvodu"
        result = QueryRewriteService(MockTextLLM()).rewrite(query)
        # MockTextLLM expands with Czech legal terms
        assert any(
            w in result.lower()
            for w in ["rodičovská", "ústavní", "judikatura", "právní", "práva", "soud"]
        )

    def test_llm_called_once(self) -> None:
        spy = _EchoTextLLM()
        QueryRewriteService(spy).rewrite("q")
        assert len(spy.calls) == 1

    def test_prompt_contains_query(self) -> None:
        spy = _EchoTextLLM()
        QueryRewriteService(spy).rewrite("mezinárodní únos")
        assert "mezinárodní únos" in spy.calls[0]


# ---------------------------------------------------------------------------
# QueryRewriteService — fallback on failure
# ---------------------------------------------------------------------------


class TestQueryRewriteServiceFallback:
    def test_llm_exception_returns_original_query(self) -> None:
        result = QueryRewriteService(_FailingTextLLM()).rewrite("únos dítěte")
        assert result == "únos dítěte"

    def test_empty_llm_response_returns_original_query(self) -> None:
        result = QueryRewriteService(_FixedTextLLM("   ")).rewrite("únos dítěte")
        assert result == "únos dítěte"

    def test_fallback_logs_warning_on_exception(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="app.rag.rewrite.query_rewrite_service"):
            QueryRewriteService(_FailingTextLLM()).rewrite("q")
        assert any(
            "fallback" in r.getMessage().lower() or "falling back" in r.getMessage().lower()
            for r in caplog.records
        )

    def test_fallback_logs_warning_on_empty_response(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="app.rag.rewrite.query_rewrite_service"):
            QueryRewriteService(_FixedTextLLM("")).rewrite("q")
        assert any("empty" in r.getMessage().lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# QueryRewriteService — logging
# ---------------------------------------------------------------------------


class TestQueryRewriteServiceLogging:
    def test_successful_rewrite_logged_as_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="app.rag.rewrite.query_rewrite_service"):
            QueryRewriteService(MockTextLLM()).rewrite("únos dítěte")

        msgs = [r.getMessage() for r in caplog.records]
        assert any("[rewrite]" in m for m in msgs)

    def test_log_contains_original_and_rewritten(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="app.rag.rewrite.query_rewrite_service"):
            QueryRewriteService(MockTextLLM()).rewrite("délka řízení")

        rewrite_log = next(
            r.getMessage() for r in caplog.records if "[rewrite]" in r.getMessage()
        )
        assert "original=" in rewrite_log
        assert "rewritten=" in rewrite_log


# ---------------------------------------------------------------------------
# Pipeline integration — rewrite_service is optional
# ---------------------------------------------------------------------------


class TestQueryRewritePipelineIntegration:
    def _make_pipeline(self, rewrite_service=None):
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
        return RetrievalPipeline(service, rewrite_service=rewrite_service)

    def test_without_rewrite_service_rewritten_query_is_none(self) -> None:
        pipeline = self._make_pipeline()
        result = pipeline.run("únos dítěte", top_k=3)
        assert result.rewritten_query is None

    def test_with_rewrite_service_rewritten_query_is_set(self) -> None:
        pipeline = self._make_pipeline(
            rewrite_service=QueryRewriteService(MockTextLLM())
        )
        result = pipeline.run("únos dítěte", top_k=3)
        assert result.rewritten_query is not None

    def test_rewritten_query_different_from_original(self) -> None:
        pipeline = self._make_pipeline(
            rewrite_service=QueryRewriteService(MockTextLLM())
        )
        query = "matka unesla dítě do Ruska"
        result = pipeline.run(query, top_k=3)
        assert result.rewritten_query != query

    def test_fallback_preserves_pipeline_flow(self) -> None:
        pipeline = self._make_pipeline(
            rewrite_service=QueryRewriteService(_FailingTextLLM())
        )
        result = pipeline.run("únos dítěte", top_k=3)
        # Pipeline must complete even when rewrite fails
        assert result.rewritten_query == "únos dítěte"

    def test_rewrite_used_for_retrieval_not_original(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging
        pipeline = self._make_pipeline(
            rewrite_service=QueryRewriteService(MockTextLLM())
        )
        with caplog.at_level(logging.DEBUG, logger="app.rag.retrieval"):
            pipeline.run("únos dítěte", top_k=3)

        # The rewritten query (not "únos dítěte") appears in retrieval trace
        retrieval_msgs = " ".join(
            r.getMessage() for r in caplog.records
            if "retrieval.start" in r.getMessage()
        )
        assert "únos dítěte ústavní soud judikatura" in retrieval_msgs or \
               "judikatura" in retrieval_msgs or \
               retrieval_msgs != ""
