"""
Unit tests for the retrieval layer:
  - DenseRetriever
  - KeywordRetriever
  - fuse_results

Run:
    pytest tests/rag/test_retrieval.py -v
"""

import logging
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.rag.retrieval.dense_retriever import DenseRetriever
from app.rag.retrieval.embedder import MockEmbedder
from app.rag.retrieval.fusion import fuse_results
from app.rag.retrieval.keyword_retriever import KeywordRetriever, _score_corpus, _tokenize
from app.rag.retrieval.models import RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(id: str, score: float, source: str = "dense", text: str = "text") -> RetrievedChunk:
    return RetrievedChunk(id=id, text=text, score=score, source=source)


@dataclass
class _FakePoint:
    """Minimal stand-in for a Qdrant ScoredPoint."""
    id: str
    score: float
    payload: dict[str, Any]


_FAKE_POINTS = [
    _FakePoint("III.US_255_26_0", 0.95, {"text": "Haagská úmluva a únos dítěte."}),
    _FakePoint("II.US_100_24_2",  0.91, {"text": "Rodičovská odpovědnost."}),
    _FakePoint("I.US_88_23_1",    0.85, {"text": "Styk s dítětem."}),
    _FakePoint("IV.US_301_22_0",  0.79, {"text": "Vyživovací povinnost."}),
    _FakePoint("I.US_44_21_3",    0.72, {"text": "Rozvod manželů."}),
]


def _make_dense(points: list[_FakePoint] | None = None) -> DenseRetriever:
    client = MagicMock()
    client.search.return_value = points if points is not None else _FAKE_POINTS
    return DenseRetriever(client=client, collection_name="test", embedder=MockEmbedder())


_SAMPLE_CORPUS = [
    ("doc-1", "Haagská úmluva a mezinárodní únos dítěte"),
    ("doc-2", "Vyživovací povinnost rodiče žijícího v zahraničí"),
    ("doc-3", "Styk s dítětem po rozvodu manželů"),
    ("doc-4", "Neoprávněné přemístění dítěte cizinkou"),
    ("doc-5", "Rodičovská odpovědnost a svěření do péče"),
]


# ---------------------------------------------------------------------------
# DenseRetriever
# ---------------------------------------------------------------------------


class TestDenseRetriever:
    def test_returns_list_of_retrieved_chunks(self) -> None:
        results = _make_dense().retrieve("únos dítěte")
        assert isinstance(results, list)
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_returns_non_empty_results(self) -> None:
        assert len(_make_dense().retrieve("únos dítěte")) > 0

    def test_source_is_dense(self) -> None:
        for chunk in _make_dense().retrieve("test"):
            assert chunk.source == "dense"

    def test_scores_within_valid_range(self) -> None:
        for chunk in _make_dense().retrieve("test"):
            assert 0.0 <= chunk.score <= 1.0

    def test_top_k_passed_to_client(self) -> None:
        client = MagicMock()
        client.search.return_value = _FAKE_POINTS[:2]
        retriever = DenseRetriever(client=client, collection_name="c", embedder=MockEmbedder())
        retriever.retrieve("test", top_k=2)
        assert client.search.call_args[1]["limit"] == 2

    def test_top_k_larger_than_results_returns_all(self) -> None:
        results = _make_dense(_FAKE_POINTS[:3]).retrieve("test", top_k=100)
        assert len(results) == 3

    def test_each_result_has_id_and_text(self) -> None:
        for chunk in _make_dense().retrieve("test"):
            assert chunk.id
            assert chunk.text


# ---------------------------------------------------------------------------
# KeywordRetriever — _tokenize helper
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_lowercases(self) -> None:
        assert "únos" in _tokenize("ÚNOS")

    def test_filters_short_words(self) -> None:
        assert "do" not in _tokenize("únos do Ruska")

    def test_splits_on_spaces(self) -> None:
        tokens = _tokenize("únos dítěte ruska")
        assert "únos" in tokens
        assert "dítěte" in tokens
        assert "ruska" in tokens

    def test_empty_returns_empty(self) -> None:
        assert _tokenize("") == []


# ---------------------------------------------------------------------------
# KeywordRetriever — _score_corpus helper
# ---------------------------------------------------------------------------


class TestScoreCorpus:
    def test_no_match_returns_empty(self) -> None:
        results = _score_corpus([("id-1", "some text")], ["xyz"])
        assert results == []

    def test_single_match_returns_one_result(self) -> None:
        results = _score_corpus([("id-1", "únos dítěte")], ["únos"])
        assert len(results) == 1

    def test_score_increases_with_more_matches(self) -> None:
        results_one = _score_corpus([("id-1", "únos dítěte ruska")], ["únos"])
        results_two = _score_corpus([("id-1", "únos dítěte ruska")], ["únos", "dítěte"])
        assert results_two[0].score > results_one[0].score

    def test_score_capped_at_max(self) -> None:
        many_words = ["a", "b", "c", "d", "e", "f", "g"]
        text = " ".join(many_words)
        results = _score_corpus([("id-1", text)], many_words)
        assert results[0].score <= 0.9

    def test_empty_query_words_returns_empty(self) -> None:
        results = _score_corpus([("id-1", "any text")], [])
        assert results == []


# ---------------------------------------------------------------------------
# KeywordRetriever — full interface
# ---------------------------------------------------------------------------


class TestKeywordRetriever:
    def test_empty_corpus_returns_empty(self) -> None:
        results = KeywordRetriever([]).retrieve("únos dítěte")
        assert results == []

    def test_matching_query_returns_results(self) -> None:
        retriever = KeywordRetriever(_SAMPLE_CORPUS)
        results = retriever.retrieve("únos dítěte")
        assert len(results) > 0

    def test_unrelated_query_returns_empty(self) -> None:
        retriever = KeywordRetriever(_SAMPLE_CORPUS)
        results = retriever.retrieve("xyz zzz qqq")
        assert results == []

    def test_source_is_keyword(self) -> None:
        retriever = KeywordRetriever(_SAMPLE_CORPUS)
        for chunk in retriever.retrieve("únos dítěte"):
            assert chunk.source == "keyword"

    def test_top_k_limits_results(self) -> None:
        retriever = KeywordRetriever(_SAMPLE_CORPUS)
        results = retriever.retrieve("dítěte", top_k=2)
        assert len(results) <= 2

    def test_results_sorted_by_score_desc(self) -> None:
        retriever = KeywordRetriever(_SAMPLE_CORPUS)
        results = retriever.retrieve("dítěte únos rozvod")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_more_keyword_matches_yields_higher_score(self) -> None:
        corpus = [
            ("high", "únos dítěte ruska cizinka"),
            ("low", "únos soudu"),
        ]
        retriever = KeywordRetriever(corpus)
        results = retriever.retrieve("únos dítěte ruska cizinka")
        high = next(r for r in results if r.id == "high")
        low = next(r for r in results if r.id == "low")
        assert high.score > low.score

    def test_no_corpus_default_is_empty(self) -> None:
        retriever = KeywordRetriever()
        assert retriever.retrieve("test") == []


# ---------------------------------------------------------------------------
# fuse_results
# ---------------------------------------------------------------------------


class TestFuseResults:
    def test_returns_list_of_retrieved_chunks(self) -> None:
        result = fuse_results([_chunk("a", 0.9)], [_chunk("b", 0.8)])
        assert all(isinstance(r, RetrievedChunk) for r in result)

    def test_combines_disjoint_results(self) -> None:
        dense = [_chunk("a", 0.9), _chunk("b", 0.85)]
        keyword = [_chunk("c", 0.7), _chunk("d", 0.6)]
        result = fuse_results(dense, keyword)
        ids = {r.id for r in result}
        assert ids == {"a", "b", "c", "d"}

    def test_deduplicates_same_id_keeps_higher_score(self) -> None:
        dense = [_chunk("shared", 0.9, source="dense")]
        keyword = [_chunk("shared", 0.6, source="keyword")]
        result = fuse_results(dense, keyword)
        shared = next(r for r in result if r.id == "shared")
        assert shared.score == 0.9

    def test_deduplicates_same_id_keyword_wins_when_higher(self) -> None:
        dense = [_chunk("shared", 0.5, source="dense")]
        keyword = [_chunk("shared", 0.8, source="keyword")]
        result = fuse_results(dense, keyword)
        shared = next(r for r in result if r.id == "shared")
        assert shared.score == 0.8

    def test_no_duplicate_ids_in_output(self) -> None:
        dense = [_chunk("x", 0.9), _chunk("y", 0.8)]
        keyword = [_chunk("x", 0.7), _chunk("z", 0.6)]
        result = fuse_results(dense, keyword)
        ids = [r.id for r in result]
        assert len(ids) == len(set(ids))

    def test_sorted_by_score_descending(self) -> None:
        dense = [_chunk("a", 0.7), _chunk("b", 0.9)]
        keyword = [_chunk("c", 0.8)]
        result = fuse_results(dense, keyword)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_output(self) -> None:
        dense = [_chunk(f"d{i}", 0.9 - i * 0.05) for i in range(5)]
        keyword = [_chunk(f"k{i}", 0.85 - i * 0.05) for i in range(5)]
        result = fuse_results(dense, keyword, top_k=3)
        assert len(result) == 3

    def test_top_k_returns_highest_scores(self) -> None:
        dense = [_chunk("a", 0.95), _chunk("b", 0.50)]
        keyword = [_chunk("c", 0.80), _chunk("d", 0.30)]
        result = fuse_results(dense, keyword, top_k=2)
        ids = {r.id for r in result}
        assert "a" in ids
        assert "c" in ids

    def test_empty_dense_returns_keyword_results(self) -> None:
        keyword = [_chunk("k1", 0.7), _chunk("k2", 0.6)]
        result = fuse_results([], keyword)
        assert len(result) == 2

    def test_empty_keyword_returns_dense_results(self) -> None:
        dense = [_chunk("d1", 0.9)]
        result = fuse_results(dense, [])
        assert len(result) == 1

    def test_both_empty_returns_empty(self) -> None:
        assert fuse_results([], []) == []


# ---------------------------------------------------------------------------
# Trace events
# ---------------------------------------------------------------------------


class TestRetrievalTrace:
    def test_dense_emits_start_and_done(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(
            logging.DEBUG, logger="app.rag.retrieval.dense_retriever"
        ):
            _make_dense().retrieve("test")

        messages = [r.getMessage() for r in caplog.records]
        assert any("TRACE dense.start" in m for m in messages)
        assert any("TRACE dense.done" in m for m in messages)

    def test_keyword_emits_start_and_done(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(
            logging.DEBUG, logger="app.rag.retrieval.keyword_retriever"
        ):
            KeywordRetriever(_SAMPLE_CORPUS).retrieve("únos")

        messages = [r.getMessage() for r in caplog.records]
        assert any("TRACE keyword.start" in m for m in messages)
        assert any("TRACE keyword.done" in m for m in messages)

    def test_fusion_emits_start_and_done(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="app.rag.retrieval.fusion"):
            fuse_results([_chunk("a", 0.9)], [_chunk("b", 0.8)])

        messages = [r.getMessage() for r in caplog.records]
        assert any("TRACE fusion.start" in m for m in messages)
        assert any("TRACE fusion.done" in m for m in messages)

    def test_fusion_trace_includes_counts(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="app.rag.retrieval.fusion"):
            fuse_results([_chunk("a", 0.9), _chunk("b", 0.8)], [_chunk("c", 0.7)])

        start = next(
            r.getMessage()
            for r in caplog.records
            if "fusion.start" in r.getMessage()
        )
        assert "dense_count=2" in start
        assert "keyword_count=1" in start
