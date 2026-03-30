"""
Comprehensive tests for the Qdrant-backed DenseRetriever.

Verifies:
- embedder.embed_query is called with the right query
- client.search is called with the right vector, collection and limit
- results are correctly mapped to RetrievedChunk
- edge cases (empty results, missing payload fields)
- trace events

No real Qdrant server needed — all I/O is mocked.

Run:
    pytest tests/rag/test_dense_qdrant_retriever.py -v
"""

import logging
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, call

import pytest

from app.rag.retrieval.dense_retriever import DenseRetriever, _to_chunk
from app.rag.retrieval.embedder import BaseEmbedder, MockEmbedder
from app.rag.retrieval.models import RetrievedChunk


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _FakePoint:
    """Duck-typed stand-in for qdrant_client.models.ScoredPoint."""
    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


class _FixedEmbedder(BaseEmbedder):
    """Returns a known, inspectable vector."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    def embed_query(self, query: str) -> list[float]:
        return list(self._vector)


def _make_retriever(
    points: list[_FakePoint] | None = None,
    embedder: BaseEmbedder | None = None,
    collection: str = "nalus",
) -> tuple[DenseRetriever, MagicMock]:
    client = MagicMock()
    client.search.return_value = points or []
    retriever = DenseRetriever(
        client=client,
        collection_name=collection,
        embedder=embedder or MockEmbedder(),
    )
    return retriever, client


def _full_payload(
    text: str = "Rozhodnutí soudu.",
    case_reference: str = "III.ÚS 255/26",
    ecli: str = "ECLI:CZ:US:2026:3.US.255.26.1",
    decision_date: str = "2026-01-15",
    judge: str = "Jan Novák",
    text_url: str = "https://nalus.usoud.cz/text/255",
    chunk_index: int = 0,
    document_id: int = 136186,
) -> dict[str, Any]:
    return {
        "text": text,
        "case_reference": case_reference,
        "ecli": ecli,
        "decision_date": decision_date,
        "judge": judge,
        "text_url": text_url,
        "chunk_index": chunk_index,
        "document_id": document_id,
        "source": "nalus",
    }


# ---------------------------------------------------------------------------
# _to_chunk helper (pure function)
# ---------------------------------------------------------------------------


class TestToChunk:
    def test_maps_id(self) -> None:
        point = _FakePoint(id="abc-1", score=0.9, payload={"text": "hello"})
        assert _to_chunk(point).id == "abc-1"

    def test_maps_score_as_float(self) -> None:
        point = _FakePoint(id="x", score=0.87, payload={})
        assert _to_chunk(point).score == pytest.approx(0.87)

    def test_source_is_always_dense(self) -> None:
        point = _FakePoint(id="x", score=0.5, payload={"source": "nalus"})
        assert _to_chunk(point).source == "dense"

    def test_text_taken_from_payload(self) -> None:
        point = _FakePoint(id="x", score=0.5, payload={"text": "Haagská úmluva."})
        assert _to_chunk(point).text == "Haagská úmluva."

    def test_missing_text_in_payload_returns_empty_string(self) -> None:
        point = _FakePoint(id="x", score=0.5, payload={})
        assert _to_chunk(point).text == ""

    def test_none_payload_treated_as_empty(self) -> None:
        point = _FakePoint(id="x", score=0.5, payload=None)  # type: ignore[arg-type]
        chunk = _to_chunk(point)
        assert chunk.text == ""

    def test_integer_id_converted_to_str(self) -> None:
        point = _FakePoint(id=42, score=0.5, payload={})  # type: ignore[arg-type]
        assert isinstance(_to_chunk(point).id, str)
        assert _to_chunk(point).id == "42"

    def test_original_id_in_payload_takes_precedence(self) -> None:
        point = _FakePoint(
            id=42,
            score=0.5,
            payload={"text": "Haagská úmluva.", "original_id": "III.ÚS_255_22_0"},
        )
        assert _to_chunk(point).id == "III.ÚS_255_22_0"

    def test_payload_metadata_preserved(self) -> None:
        payload = _full_payload()
        point = _FakePoint(id="doc-1", score=0.5, payload=payload)
        assert _to_chunk(point).metadata == payload


# ---------------------------------------------------------------------------
# Embedder interaction
# ---------------------------------------------------------------------------


class TestDenseRetrieverEmbedder:
    def test_embed_query_called_with_exact_query(self) -> None:
        embedder = MagicMock(spec=BaseEmbedder)
        embedder.embed_query.return_value = [0.1] * 10
        client = MagicMock()
        client.search.return_value = []
        retriever = DenseRetriever(client=client, collection_name="c", embedder=embedder)

        retriever.retrieve("únos dítěte")

        embedder.embed_query.assert_called_once_with("únos dítěte")

    def test_vector_from_embedder_passed_to_client(self) -> None:
        fixed_vector = [0.3, 0.1, 0.7, 0.2, 0.5, 0.9, 0.4, 0.6, 0.8, 0.0]
        embedder = _FixedEmbedder(fixed_vector)
        retriever, client = _make_retriever(embedder=embedder)

        retriever.retrieve("test")

        call_kwargs = client.search.call_args[1]
        assert call_kwargs["query_vector"] == fixed_vector

    def test_mock_embedder_returns_correct_dim(self) -> None:
        embedder = MockEmbedder(dim=384)
        vector = embedder.embed_query("query")
        assert len(vector) == 384

    def test_mock_embedder_returns_floats(self) -> None:
        for v in MockEmbedder().embed_query("q"):
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# Client interaction
# ---------------------------------------------------------------------------


class TestDenseRetrieverClientCalls:
    def test_client_search_called_once(self) -> None:
        retriever, client = _make_retriever()
        retriever.retrieve("test")
        client.search.assert_called_once()

    def test_correct_collection_name_passed(self) -> None:
        retriever, client = _make_retriever(collection="my-collection")
        retriever.retrieve("test")
        assert client.search.call_args[1]["collection_name"] == "my-collection"

    def test_limit_equals_top_k(self) -> None:
        retriever, client = _make_retriever()
        retriever.retrieve("test", top_k=7)
        assert client.search.call_args[1]["limit"] == 7

    def test_default_top_k_is_ten(self) -> None:
        retriever, client = _make_retriever()
        retriever.retrieve("test")
        assert client.search.call_args[1]["limit"] == 10

    def test_each_query_triggers_one_search_call(self) -> None:
        retriever, client = _make_retriever()
        retriever.retrieve("first")
        retriever.retrieve("second")
        assert client.search.call_count == 2


# ---------------------------------------------------------------------------
# Result mapping
# ---------------------------------------------------------------------------


class TestDenseRetrieverResults:
    def test_empty_client_results_return_empty_list(self) -> None:
        retriever, _ = _make_retriever(points=[])
        assert retriever.retrieve("test") == []

    def test_returns_list_of_retrieved_chunks(self) -> None:
        points = [_FakePoint("a", 0.9, {"text": "t"})]
        retriever, _ = _make_retriever(points=points)
        results = retriever.retrieve("test")
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_result_count_matches_client_output(self) -> None:
        points = [_FakePoint(f"id-{i}", 0.9 - i * 0.1, {"text": "t"}) for i in range(4)]
        retriever, _ = _make_retriever(points=points)
        assert len(retriever.retrieve("test")) == 4

    def test_scores_mapped_correctly(self) -> None:
        points = [
            _FakePoint("a", 0.95, {"text": "x"}),
            _FakePoint("b", 0.72, {"text": "y"}),
        ]
        retriever, _ = _make_retriever(points=points)
        results = retriever.retrieve("test")
        assert results[0].score == pytest.approx(0.95)
        assert results[1].score == pytest.approx(0.72)

    def test_text_from_payload_mapped(self) -> None:
        points = [_FakePoint("id-1", 0.9, {"text": "Haagská úmluva."})]
        retriever, _ = _make_retriever(points=points)
        assert retriever.retrieve("test")[0].text == "Haagská úmluva."

    def test_source_is_dense_for_all_results(self) -> None:
        points = [_FakePoint(f"id-{i}", 0.9, {"text": "t"}) for i in range(3)]
        retriever, _ = _make_retriever(points=points)
        for chunk in retriever.retrieve("test"):
            assert chunk.source == "dense"

    def test_id_preserved(self) -> None:
        points = [_FakePoint("III.US_255_26_0", 0.9, {"text": "t"})]
        retriever, _ = _make_retriever(points=points)
        assert retriever.retrieve("test")[0].id == "III.US_255_26_0"

    def test_full_payload_mapped(self) -> None:
        payload = _full_payload(text="Rozhodnutí Ústavního soudu.")
        points = [_FakePoint("doc-1", 0.88, payload)]
        retriever, _ = _make_retriever(points=points)
        chunk = retriever.retrieve("test")[0]
        assert chunk.text == "Rozhodnutí Ústavního soudu."
        assert chunk.score == pytest.approx(0.88)
        assert chunk.id == "doc-1"
        assert chunk.metadata["case_reference"] == "III.ÚS 255/26"
        assert chunk.metadata["document_id"] == 136186

    def test_missing_text_field_gives_empty_string(self) -> None:
        points = [_FakePoint("id-1", 0.8, {"case_reference": "I.ÚS 1/24"})]
        retriever, _ = _make_retriever(points=points)
        assert retriever.retrieve("test")[0].text == ""


# ---------------------------------------------------------------------------
# Trace events
# ---------------------------------------------------------------------------


class TestDenseRetrieverTrace:
    def test_start_and_done_emitted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        retriever, _ = _make_retriever(
            points=[_FakePoint("a", 0.9, {"text": "t"})]
        )
        with caplog.at_level(
            logging.DEBUG, logger="app.rag.retrieval.dense_retriever"
        ):
            retriever.retrieve("únos dítěte")

        msgs = [r.getMessage() for r in caplog.records]
        assert any("TRACE dense.start" in m for m in msgs)
        assert any("TRACE dense.done" in m for m in msgs)

    def test_start_includes_query_top_k_collection(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        retriever, _ = _make_retriever(collection="nalus-prod")
        with caplog.at_level(
            logging.DEBUG, logger="app.rag.retrieval.dense_retriever"
        ):
            retriever.retrieve("únos dítěte", top_k=3)

        start = next(
            r.getMessage()
            for r in caplog.records
            if "dense.start" in r.getMessage()
        )
        assert "únos dítěte" in start
        assert "top_k=3" in start
        assert "nalus-prod" in start

    def test_done_includes_num_results(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        points = [_FakePoint(f"id-{i}", 0.9, {"text": "t"}) for i in range(3)]
        retriever, _ = _make_retriever(points=points)
        with caplog.at_level(
            logging.DEBUG, logger="app.rag.retrieval.dense_retriever"
        ):
            retriever.retrieve("test")

        done = next(
            r.getMessage()
            for r in caplog.records
            if "dense.done" in r.getMessage()
        )
        assert "num_results=3" in done

    def test_empty_results_still_emits_done(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        retriever, _ = _make_retriever(points=[])
        with caplog.at_level(
            logging.DEBUG, logger="app.rag.retrieval.dense_retriever"
        ):
            retriever.retrieve("test")

        msgs = [r.getMessage() for r in caplog.records]
        assert any("dense.done" in m for m in msgs)
        done = next(m for m in msgs if "dense.done" in m)
        assert "num_results=0" in done
