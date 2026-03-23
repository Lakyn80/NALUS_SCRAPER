"""
Tests for the reranker layer and its integration into RetrievalService.

Run:
    pytest tests/rag/test_reranker.py -v
"""

import logging
from unittest.mock import MagicMock

import pytest

from app.rag.reranker.base import BaseReranker
from app.rag.reranker.simple_reranker import SimpleReranker, _boost, _tokenize
from app.rag.retrieval.models import RetrievedChunk
from app.rag.retrieval.retrieval_service import RetrievalService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(
    id: str,
    score: float,
    text: str = "",
    source: str = "dense",
) -> RetrievedChunk:
    return RetrievedChunk(id=id, text=text, score=score, source=source)


def _make_service(
    dense_results: list[RetrievedChunk] | None = None,
    keyword_results: list[RetrievedChunk] | None = None,
    reranker: BaseReranker | None = None,
) -> tuple[RetrievalService, MagicMock, MagicMock]:
    dense = MagicMock()
    dense.retrieve.return_value = dense_results or []
    keyword = MagicMock()
    keyword.retrieve.return_value = keyword_results or []
    return RetrievalService(dense=dense, keyword=keyword, reranker=reranker), dense, keyword


# ---------------------------------------------------------------------------
# _tokenize (pure)
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_lowercases(self) -> None:
        assert "únos" in _tokenize("ÚNOS")

    def test_drops_words_shorter_than_3(self) -> None:
        assert "do" not in _tokenize("únos do Ruska")

    def test_splits_on_whitespace(self) -> None:
        result = _tokenize("únos dítěte")
        assert "únos" in result and "dítěte" in result

    def test_empty_returns_empty(self) -> None:
        assert _tokenize("") == []


# ---------------------------------------------------------------------------
# _boost (pure)
# ---------------------------------------------------------------------------


class TestBoost:
    def test_no_match_returns_same_score(self) -> None:
        chunk = _chunk("a", 0.80, text="irrelevant content")
        result = _boost(chunk, ["únos", "dítě"])
        assert result.score == pytest.approx(0.80)

    def test_no_match_returns_same_object(self) -> None:
        chunk = _chunk("a", 0.80, text="irrelevant content")
        assert _boost(chunk, ["xyz"]) is chunk

    def test_single_match_adds_one_boost(self) -> None:
        chunk = _chunk("a", 0.80, text="únos dítěte v Rusku")
        result = _boost(chunk, ["únos"])
        assert result.score == pytest.approx(0.80 + 0.05)

    def test_two_matches_add_two_boosts(self) -> None:
        chunk = _chunk("a", 0.80, text="únos dítěte v Rusku")
        result = _boost(chunk, ["únos", "dítěte"])
        assert result.score == pytest.approx(0.80 + 0.10)

    def test_boost_is_case_insensitive(self) -> None:
        chunk = _chunk("a", 0.50, text="Únos Dítěte")
        result = _boost(chunk, ["únos", "dítěte"])
        assert result.score > 0.50

    def test_returns_new_chunk_instance(self) -> None:
        chunk = _chunk("a", 0.80, text="únos dítěte")
        result = _boost(chunk, ["únos"])
        assert result is not chunk

    def test_id_and_source_preserved_after_boost(self) -> None:
        chunk = _chunk("my-id", 0.80, text="únos dítěte", source="keyword")
        result = _boost(chunk, ["únos"])
        assert result.id == "my-id"
        assert result.source == "keyword"

    def test_empty_query_words_no_boost(self) -> None:
        chunk = _chunk("a", 0.70, text="anything")
        assert _boost(chunk, []).score == pytest.approx(0.70)


# ---------------------------------------------------------------------------
# SimpleReranker.rerank
# ---------------------------------------------------------------------------


class TestSimpleReranker:
    def test_returns_list_of_retrieved_chunks(self) -> None:
        chunks = [_chunk("a", 0.9, "únos dítěte")]
        result = SimpleReranker().rerank("únos", chunks)
        assert all(isinstance(r, RetrievedChunk) for r in result)

    def test_empty_input_returns_empty(self) -> None:
        assert SimpleReranker().rerank("únos", []) == []

    def test_top_k_limits_output(self) -> None:
        chunks = [_chunk(str(i), 0.9 - i * 0.05, "text") for i in range(10)]
        result = SimpleReranker().rerank("query", chunks, top_k=3)
        assert len(result) == 3

    def test_top_k_larger_than_input_returns_all(self) -> None:
        chunks = [_chunk("a", 0.9, "text"), _chunk("b", 0.8, "text")]
        result = SimpleReranker().rerank("query", chunks, top_k=100)
        assert len(result) == 2

    def test_matching_chunk_promoted_above_non_matching(self) -> None:
        # query "únos dítěte cizinka" → 3 words
        # match:    0.55 + 3*0.05 = 0.70
        # no-match: 0.65 + 0      = 0.65  → match wins
        match = _chunk("match", 0.55, text="únos dítěte cizinka v zahraničí")
        no_match = _chunk("no-match", 0.65, text="hospodářská smlouva o dílo")
        reranker = SimpleReranker()
        result = reranker.rerank("únos dítěte cizinka", [no_match, match])
        assert result[0].id == "match"

    def test_reranker_changes_order(self) -> None:
        """Chunk with more keyword matches must move to the front."""
        rich = _chunk("rich", 0.70, text="únos dítěte Haagská úmluva cizinka")
        poor = _chunk("poor", 0.80, text="smlouva o dílo")
        result = SimpleReranker().rerank("únos dítěte cizinka", [poor, rich])
        # rich: 0.70 + 3*0.05=0.85; poor: 0.80+0=0.80 → rich first
        assert result[0].id == "rich"

    def test_sorted_desc_by_new_score(self) -> None:
        chunks = [
            _chunk("a", 0.90, "smlouva"),
            _chunk("b", 0.75, "únos dítěte"),
            _chunk("c", 0.80, "dítěte"),
        ]
        result = SimpleReranker().rerank("únos dítěte", chunks, top_k=3)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_original_chunks_not_mutated(self) -> None:
        chunk = _chunk("a", 0.80, text="únos dítěte")
        original_score = chunk.score
        SimpleReranker().rerank("únos", [chunk])
        assert chunk.score == pytest.approx(original_score)


# ---------------------------------------------------------------------------
# Integration with RetrievalService
# ---------------------------------------------------------------------------


class TestRetrievalServiceReranker:
    def test_no_reranker_search_works_normally(self) -> None:
        service, _, _ = _make_service(dense_results=[_chunk("a", 0.9, "text")])
        result = service.search("test")
        assert len(result) == 1

    def test_reranker_is_called_when_provided(self) -> None:
        reranker = MagicMock(spec=BaseReranker)
        reranker.rerank.return_value = [_chunk("a", 0.9, "text")]
        service, _, _ = _make_service(
            dense_results=[_chunk("a", 0.9, "text")],
            reranker=reranker,
        )
        service.search("test")
        reranker.rerank.assert_called_once()

    def test_reranker_not_called_when_none(self) -> None:
        reranker = MagicMock(spec=BaseReranker)
        service, _, _ = _make_service(
            dense_results=[_chunk("a", 0.9, "text")],
            reranker=None,
        )
        service.search("test")
        reranker.rerank.assert_not_called()

    def test_reranker_receives_original_query(self) -> None:
        reranker = MagicMock(spec=BaseReranker)
        reranker.rerank.return_value = []
        service, _, _ = _make_service(
            dense_results=[_chunk("a", 0.9, "text")],
            reranker=reranker,
        )
        service.search("Únos Dítěte")
        call_args = reranker.rerank.call_args
        query_arg = call_args[0][0] if call_args[0] else call_args[1]["query"]
        assert query_arg == "Únos Dítěte"

    def test_reranker_receives_top_k(self) -> None:
        reranker = MagicMock(spec=BaseReranker)
        reranker.rerank.return_value = []
        service, _, _ = _make_service(
            dense_results=[_chunk("a", 0.9, "text")],
            reranker=reranker,
        )
        service.search("test", top_k=7)
        call_args = reranker.rerank.call_args
        top_k_arg = call_args[0][2] if len(call_args[0]) > 2 else call_args[1].get("top_k")
        assert top_k_arg == 7

    def test_reranker_output_is_final_result(self) -> None:
        reranked = [_chunk("reranked", 0.99, "text")]
        reranker = MagicMock(spec=BaseReranker)
        reranker.rerank.return_value = reranked
        service, _, _ = _make_service(
            dense_results=[_chunk("original", 0.5, "text")],
            reranker=reranker,
        )
        result = service.search("test")
        assert result[0].id == "reranked"

    def test_simple_reranker_end_to_end(self) -> None:
        """SimpleReranker wired into RetrievalService changes final order."""
        chunks = [
            _chunk("high-score-no-kw", 0.90, "hospodářská smlouva"),
            _chunk("low-score-rich-kw", 0.60, "únos dítěte Haagská úmluva cizinka"),
        ]
        service, _, _ = _make_service(
            dense_results=chunks,
            reranker=SimpleReranker(),
        )
        result = service.search("únos dítěte cizinka", top_k=5)
        # low-score-rich-kw: 0.60 + 3*0.05 = 0.75 > high-score-no-kw 0.90? No.
        # But with even richer text:
        chunks2 = [
            _chunk("no-kw", 0.80, "smlouva o dílo"),
            _chunk("rich-kw", 0.70, "únos dítěte cizinka haag ruska"),
        ]
        service2, _, _ = _make_service(
            dense_results=chunks2,
            reranker=SimpleReranker(),
        )
        result2 = service2.search("únos dítěte cizinka", top_k=5)
        # rich-kw: 0.70 + 3*0.05 = 0.85 > no-kw 0.80
        assert result2[0].id == "rich-kw"


# ---------------------------------------------------------------------------
# Trace events
# ---------------------------------------------------------------------------


class TestRerankerTrace:
    def test_rerank_start_and_done_emitted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        chunks = [_chunk("a", 0.9, "únos dítěte")]
        with caplog.at_level(
            logging.DEBUG, logger="app.rag.reranker.simple_reranker"
        ):
            SimpleReranker().rerank("únos", chunks)

        msgs = [r.getMessage() for r in caplog.records]
        assert any("TRACE rerank.start" in m for m in msgs)
        assert any("TRACE rerank.done" in m for m in msgs)

    def test_rerank_start_includes_input_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        chunks = [_chunk(str(i), 0.9, "text") for i in range(4)]
        with caplog.at_level(
            logging.DEBUG, logger="app.rag.reranker.simple_reranker"
        ):
            SimpleReranker().rerank("query", chunks)

        start = next(
            r.getMessage()
            for r in caplog.records
            if "rerank.start" in r.getMessage()
        )
        assert "input_count=4" in start

    def test_retrieval_service_emits_rerank_trace(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        reranker = MagicMock(spec=BaseReranker)
        reranker.rerank.return_value = [_chunk("a", 0.9, "t")]
        service, _, _ = _make_service(
            dense_results=[_chunk("a", 0.9, "t"), _chunk("b", 0.8, "t")],
            reranker=reranker,
        )
        with caplog.at_level(
            logging.DEBUG, logger="app.rag.retrieval.retrieval_service"
        ):
            service.search("test")

        msgs = [r.getMessage() for r in caplog.records]
        assert any("TRACE retrieval.rerank" in m for m in msgs)

    def test_retrieval_rerank_trace_includes_counts(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        reranker = MagicMock(spec=BaseReranker)
        reranker.rerank.return_value = [_chunk("a", 0.9, "t")]
        service, _, _ = _make_service(
            dense_results=[_chunk("a", 0.9, "t"), _chunk("b", 0.8, "t")],
            reranker=reranker,
        )
        with caplog.at_level(
            logging.DEBUG, logger="app.rag.retrieval.retrieval_service"
        ):
            service.search("test")

        rerank_msg = next(
            r.getMessage()
            for r in caplog.records
            if "retrieval.rerank" in r.getMessage()
        )
        assert "before_count=" in rerank_msg
        assert "after_count=" in rerank_msg

    def test_no_rerank_trace_when_reranker_is_none(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        service, _, _ = _make_service(
            dense_results=[_chunk("a", 0.9, "t")],
            reranker=None,
        )
        with caplog.at_level(
            logging.DEBUG, logger="app.rag.retrieval.retrieval_service"
        ):
            service.search("test")

        msgs = [r.getMessage() for r in caplog.records]
        assert not any("retrieval.rerank" in m for m in msgs)
