"""Instrumentation tests for QdrantDenseStore latency diagnostics.

Does not call a real embedder or Qdrant. Retrieval mapping and call counts
must stay identical to the pre-instrumentation path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.rag.retrieval.errors import RetrievalConfigurationError
from app.rag.retrieval.production_profile import BGE_M3_DENSE_BM25_RRF, ProductionRetrievalConfig
from app.rag.retrieval.provenance import build_embedding_provenance
from app.rag.retrieval.qdrant_dense_store import QdrantDenseStore


@dataclass
class _FakePoint:
    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


class _CountingEmbedder:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.calls: list[str] = []

    def embed_query(self, query: str) -> list[float]:
        self.calls.append(query)
        return list(self.vector)


def _config(tmp_path: Path) -> ProductionRetrievalConfig:
    return ProductionRetrievalConfig(
        profile=BGE_M3_DENSE_BM25_RRF,
        qdrant_collection="nalus_dense_diag_test",
        bm25_sidecar_path=tmp_path / "bm25.sqlite",
        bm25_index_id=BGE_M3_DENSE_BM25_RRF.name,
        model_path="/app/models/BAAI/bge-m3",
        local_files_only=True,
        trust_remote_code=False,
        device="cpu",
        candidate_multiplier=1,
        min_candidate_count=1,
        max_candidate_count=80,
        lexical_filter_enabled=False,
    )


def _payload(text: str = "Haagská úmluva.") -> dict[str, Any]:
    return build_embedding_provenance(
        payload={
            "text": text,
            "source": "nalus",
            "document_id": "doc-1",
            "chunk_index": 0,
            "chunk_id": "chunk-1",
            "original_id": "orig-1",
        },
        profile=BGE_M3_DENSE_BM25_RRF,
        ingest_run_id="diag-test",
        qdrant_collection="nalus_dense_diag_test",
        bm25_index_id=BGE_M3_DENSE_BM25_RRF.name,
    )


def test_search_maps_points_without_changing_retrieval_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NALUS_QDRANT_QUANTIZATION_ENABLED", raising=False)
    monkeypatch.delenv("NALUS_QDRANT_QUANTIZATION_RESCORE", raising=False)
    monkeypatch.delenv("NALUS_QDRANT_QUANTIZATION_OVERSAMPLING", raising=False)
    embedder = _CountingEmbedder([0.1] * 1024)
    client = MagicMock()
    client.query_points.return_value = SimpleNamespace(
        points=[_FakePoint(id="p1", score=0.91, payload=_payload())]
    )
    store = QdrantDenseStore(client=client, embedder=embedder, config=_config(tmp_path))

    chunks = store.search("únos dítěte", top_k=80)

    assert embedder.calls == ["únos dítěte"]
    client.query_points.assert_called_once()
    kwargs = client.query_points.call_args.kwargs
    assert kwargs["collection_name"] == "nalus_dense_diag_test"
    assert kwargs["query"] == [0.1] * 1024
    assert kwargs["limit"] == 80
    assert kwargs["with_payload"] is True
    assert kwargs["search_params"].quantization.ignore is True
    assert len(chunks) == 1
    assert chunks[0].id == "orig-1"
    assert chunks[0].text == "Haagská úmluva."
    assert chunks[0].score == pytest.approx(0.91)
    assert chunks[0].source == "dense"


def test_v2_legacy_dense_search_omits_search_params(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classic Legal v2 FAST dense (pre-INT8): plain query_points only."""
    monkeypatch.setenv("NALUS_QDRANT_QUANTIZATION_ENABLED", "1")
    embedder = _CountingEmbedder([0.1] * 1024)
    client = MagicMock()
    client.query_points.return_value = SimpleNamespace(
        points=[_FakePoint(id="p1", score=0.5, payload=_payload())]
    )
    store = QdrantDenseStore(
        client=client,
        embedder=embedder,
        config=_config(tmp_path),
        use_quantization_search_params=False,
    )

    store.search("únos", top_k=10)

    kwargs = client.query_points.call_args.kwargs
    assert "search_params" not in kwargs
    assert kwargs["limit"] == 10


def test_search_logs_latency_breakdown_without_query_text(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NALUS_QDRANT_QUANTIZATION_ENABLED", raising=False)
    query = "SECRET_QUERY_TEXT_MUST_NOT_APPEAR_IN_LOGS"
    embedder = _CountingEmbedder([0.2] * 1024)
    client = MagicMock()
    client.query_points.return_value = SimpleNamespace(
        points=[
            _FakePoint(id="p1", score=0.8, payload=_payload("a")),
            _FakePoint(id="p2", score=0.7, payload=_payload("b")),
        ]
    )
    store = QdrantDenseStore(client=client, embedder=embedder, config=_config(tmp_path))

    with caplog.at_level(logging.INFO, logger="app.rag.retrieval.qdrant_dense_store"):
        store.search(query, top_k=80)

    matching = [record for record in caplog.records if "[dense_store] search completed" in record.getMessage()]
    assert len(matching) == 1
    record = matching[0]
    message = record.getMessage()
    assert query not in message
    assert query not in caplog.text
    assert "embedding_latency_ms=" in message
    assert "qdrant_latency_ms=" in message
    assert "conversion_latency_ms=" in message
    assert "total_latency_ms=" in message
    assert "top_k=80" in message
    assert f"query_length={len(query)}" in message
    assert record.embedding_latency_ms >= 0
    assert record.qdrant_latency_ms >= 0
    assert record.conversion_latency_ms >= 0
    assert record.dense_conversion_latency_ms == record.conversion_latency_ms
    assert record.total_latency_ms >= 0
    assert record.dense_store_total_latency_ms == record.total_latency_ms
    assert record.top_k == 80
    assert record.query_length == len(query)
    assert "quantization_enabled=False" in message
    assert "quantization_ignore=True" in message
    assert record.quantization_enabled is False
    assert record.quantization_ignore is True


def test_dimension_mismatch_still_raises_before_qdrant(tmp_path: Path) -> None:
    embedder = _CountingEmbedder([0.1] * 8)
    client = MagicMock()
    store = QdrantDenseStore(client=client, embedder=embedder, config=_config(tmp_path))

    with pytest.raises(RetrievalConfigurationError, match="dimension mismatch"):
        store.search("únos dítěte", top_k=80)

    assert len(embedder.calls) == 1
    client.query_points.assert_not_called()


def test_search_sends_ignore_false_when_quantization_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NALUS_QDRANT_QUANTIZATION_ENABLED", "1")
    monkeypatch.setenv("NALUS_QDRANT_QUANTIZATION_RESCORE", "1")
    monkeypatch.setenv("NALUS_QDRANT_QUANTIZATION_OVERSAMPLING", "2.0")
    embedder = _CountingEmbedder([0.1] * 1024)
    client = MagicMock()
    client.query_points.return_value = SimpleNamespace(
        points=[_FakePoint(id="p1", score=0.91, payload=_payload())]
    )
    store = QdrantDenseStore(client=client, embedder=embedder, config=_config(tmp_path))

    store.search("únos dítěte", top_k=80)

    assert embedder.calls == ["únos dítěte"]
    client.query_points.assert_called_once()
    params = client.query_points.call_args.kwargs["search_params"]
    assert params.quantization.ignore is False
    assert params.quantization.rescore is True
    assert params.quantization.oversampling == pytest.approx(2.0)
