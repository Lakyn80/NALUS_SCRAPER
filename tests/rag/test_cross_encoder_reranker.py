"""
Unit tests for CrossEncoderReranker.

The CrossEncoder model is always injected as a mock — no sentence-transformers
weights are loaded during these tests.

Run:
    pytest tests/rag/test_cross_encoder_reranker.py -v
"""

import logging
from unittest.mock import MagicMock

import pytest

from app.rag.reranker.cross_encoder_reranker import CrossEncoderReranker
from app.rag.retrieval.models import RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(
    id: str,
    score: float = 0.80,
    text: str = "some legal text",
    source: str = "dense",
) -> RetrievedChunk:
    return RetrievedChunk(id=id, text=text, score=score, source=source)


def _make_reranker(predict_return: list[float]) -> tuple[CrossEncoderReranker, MagicMock]:
    mock_model = MagicMock()
    mock_model.predict.return_value = predict_return
    return CrossEncoderReranker(model=mock_model), mock_model


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestCrossEncoderRerankerConstruction:
    def test_injectable_model_used_directly(self) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5]
        reranker = CrossEncoderReranker(model=mock_model)
        reranker.rerank("q", [_chunk("a")])
        mock_model.predict.assert_called_once()

    def test_no_import_error_with_injected_model(self) -> None:
        # Module must be importable without sentence-transformers installed.
        reranker = CrossEncoderReranker(model=MagicMock())
        assert isinstance(reranker, CrossEncoderReranker)


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


class TestCrossEncoderRerankerEmpty:
    def test_empty_chunks_returns_empty_list(self) -> None:
        reranker, mock_model = _make_reranker([])
        result = reranker.rerank("query", [])
        assert result == []

    def test_empty_chunks_does_not_call_predict(self) -> None:
        reranker, mock_model = _make_reranker([])
        reranker.rerank("query", [])
        mock_model.predict.assert_not_called()


# ---------------------------------------------------------------------------
# Correct number of results
# ---------------------------------------------------------------------------


class TestCrossEncoderRerankerCount:
    def test_returns_up_to_top_k(self) -> None:
        chunks = [_chunk(str(i)) for i in range(6)]
        scores = [float(i) for i in range(6)]
        reranker, _ = _make_reranker(scores)
        result = reranker.rerank("query", chunks, top_k=3)
        assert len(result) == 3

    def test_top_k_larger_than_input_returns_all(self) -> None:
        chunks = [_chunk("a"), _chunk("b")]
        reranker, _ = _make_reranker([0.9, 0.8])
        result = reranker.rerank("query", chunks, top_k=100)
        assert len(result) == 2

    def test_single_chunk_returns_single_result(self) -> None:
        reranker, _ = _make_reranker([0.7])
        result = reranker.rerank("query", [_chunk("only")])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# predict called correctly
# ---------------------------------------------------------------------------


class TestCrossEncoderRerankerPredictCall:
    def test_predict_called_once(self) -> None:
        chunks = [_chunk("a"), _chunk("b")]
        reranker, mock_model = _make_reranker([0.9, 0.8])
        reranker.rerank("query", chunks)
        mock_model.predict.assert_called_once()

    def test_predict_receives_query_text_pairs(self) -> None:
        chunks = [
            _chunk("a", text="únos dítěte"),
            _chunk("b", text="rodinné právo"),
        ]
        reranker, mock_model = _make_reranker([0.9, 0.8])
        reranker.rerank("test query", chunks)
        pairs = mock_model.predict.call_args[0][0]
        assert pairs == [("test query", "únos dítěte"), ("test query", "rodinné právo")]

    def test_pairs_preserve_chunk_order(self) -> None:
        texts = [f"text-{i}" for i in range(5)]
        chunks = [_chunk(str(i), text=t) for i, t in enumerate(texts)]
        reranker, mock_model = _make_reranker([0.5] * 5)
        reranker.rerank("q", chunks)
        pairs = mock_model.predict.call_args[0][0]
        for i, (q, t) in enumerate(pairs):
            assert t == texts[i]


# ---------------------------------------------------------------------------
# Ordering by predicted score
# ---------------------------------------------------------------------------


class TestCrossEncoderRerankerOrdering:
    def test_highest_predicted_score_first(self) -> None:
        chunks = [_chunk("low", 0.9), _chunk("high", 0.5)]
        reranker, _ = _make_reranker([0.2, 0.9])  # second chunk gets higher score
        result = reranker.rerank("q", chunks)
        assert result[0].id == "high"

    def test_sorted_descending_by_predicted_score(self) -> None:
        chunks = [_chunk(str(i)) for i in range(4)]
        reranker, _ = _make_reranker([0.3, 0.8, 0.1, 0.6])
        result = reranker.rerank("q", chunks, top_k=4)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_score_replaced_with_predicted_value(self) -> None:
        chunk = _chunk("a", score=0.50)
        reranker, _ = _make_reranker([0.92])
        result = reranker.rerank("q", [chunk])
        assert result[0].score == pytest.approx(0.92)

    def test_original_score_not_preserved(self) -> None:
        chunk = _chunk("a", score=0.99)
        reranker, _ = _make_reranker([0.11])
        result = reranker.rerank("q", [chunk])
        assert result[0].score == pytest.approx(0.11)


# ---------------------------------------------------------------------------
# id / text / source preservation
# ---------------------------------------------------------------------------


class TestCrossEncoderRerankerPreservation:
    def test_id_preserved(self) -> None:
        chunk = _chunk("my-unique-id")
        reranker, _ = _make_reranker([0.7])
        assert reranker.rerank("q", [chunk])[0].id == "my-unique-id"

    def test_text_preserved(self) -> None:
        chunk = _chunk("a", text="Haagská úmluva a únos dítěte.")
        reranker, _ = _make_reranker([0.7])
        assert reranker.rerank("q", [chunk])[0].text == "Haagská úmluva a únos dítěte."

    def test_source_preserved(self) -> None:
        chunk = _chunk("a", source="keyword")
        reranker, _ = _make_reranker([0.7])
        assert reranker.rerank("q", [chunk])[0].source == "keyword"

    def test_original_chunk_not_mutated(self) -> None:
        chunk = _chunk("a", score=0.50)
        reranker, _ = _make_reranker([0.99])
        reranker.rerank("q", [chunk])
        assert chunk.score == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# Fallback on model.predict exception
# ---------------------------------------------------------------------------


class TestCrossEncoderRerankerFallback:
    def test_fallback_returns_list_on_exception(self) -> None:
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("model error")
        reranker = CrossEncoderReranker(model=mock_model)
        chunks = [_chunk("a", 0.9), _chunk("b", 0.8)]
        result = reranker.rerank("q", chunks)
        assert isinstance(result, list)

    def test_fallback_preserves_original_order(self) -> None:
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("fail")
        reranker = CrossEncoderReranker(model=mock_model)
        chunks = [_chunk("first"), _chunk("second"), _chunk("third")]
        result = reranker.rerank("q", chunks, top_k=3)
        assert [r.id for r in result] == ["first", "second", "third"]

    def test_fallback_respects_top_k(self) -> None:
        mock_model = MagicMock()
        mock_model.predict.side_effect = ValueError("fail")
        reranker = CrossEncoderReranker(model=mock_model)
        chunks = [_chunk(str(i)) for i in range(6)]
        result = reranker.rerank("q", chunks, top_k=2)
        assert len(result) == 2

    def test_fallback_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("boom")
        reranker = CrossEncoderReranker(model=mock_model)
        with caplog.at_level(logging.WARNING, logger="app.rag.reranker.cross_encoder_reranker"):
            reranker.rerank("q", [_chunk("a")])
        assert any("fallback" in r.getMessage().lower() or "failed" in r.getMessage().lower()
                   for r in caplog.records)


# ---------------------------------------------------------------------------
# Trace events
# ---------------------------------------------------------------------------


class TestCrossEncoderRerankerTrace:
    def test_start_and_done_emitted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        reranker, _ = _make_reranker([0.9])
        with caplog.at_level(
            logging.DEBUG, logger="app.rag.reranker.cross_encoder_reranker"
        ):
            reranker.rerank("únos dítěte", [_chunk("a")])

        msgs = [r.getMessage() for r in caplog.records]
        assert any("TRACE rerank.start" in m for m in msgs)
        assert any("TRACE rerank.done" in m for m in msgs)

    def test_trace_start_includes_input_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        chunks = [_chunk(str(i)) for i in range(4)]
        reranker, _ = _make_reranker([float(i) for i in range(4)])
        with caplog.at_level(
            logging.DEBUG, logger="app.rag.reranker.cross_encoder_reranker"
        ):
            reranker.rerank("q", chunks)

        start = next(
            r.getMessage() for r in caplog.records
            if "rerank.start" in r.getMessage()
        )
        assert "input_count=4" in start

    def test_trace_done_includes_output_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        reranker, _ = _make_reranker([0.9, 0.8, 0.7])
        with caplog.at_level(
            logging.DEBUG, logger="app.rag.reranker.cross_encoder_reranker"
        ):
            reranker.rerank("q", [_chunk(str(i)) for i in range(3)], top_k=2)

        done = next(
            r.getMessage() for r in caplog.records
            if "rerank.done" in r.getMessage()
        )
        assert "output_count=2" in done

    def test_trace_done_includes_fallback_flag_on_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("oops")
        reranker = CrossEncoderReranker(model=mock_model)
        with caplog.at_level(
            logging.DEBUG, logger="app.rag.reranker.cross_encoder_reranker"
        ):
            reranker.rerank("q", [_chunk("a")])

        done = next(
            r.getMessage() for r in caplog.records
            if "rerank.done" in r.getMessage()
        )
        assert "fallback" in done

    def test_empty_input_still_emits_done(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        reranker, _ = _make_reranker([])
        with caplog.at_level(
            logging.DEBUG, logger="app.rag.reranker.cross_encoder_reranker"
        ):
            reranker.rerank("q", [])

        msgs = [r.getMessage() for r in caplog.records]
        assert any("rerank.done" in m for m in msgs)
