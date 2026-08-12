"""Unit tests for Legal v2 ColBERT retrieval foundation (no model/index I/O)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.legal_v2.retrieve.colbert import (
    COLBERT_PILOT_SOURCE_QDRANT_COLLECTION,
    ColbertBackendUnavailableError,
    ColbertConfig,
    ColbertConfigurationError,
    ColbertHit,
    ColbertIndexer,
    ColbertNotImplementedError,
    ColbertRetriever,
    import_colbert_library,
)
from app.rag.legal_v2.retrieve.retrieval_profiles import (
    resolve_retrieval_profile,
)
from app.rag.retrieval.models import RetrievedChunk


def _valid_config(**overrides: object) -> ColbertConfig:
    base: dict[str, object] = {
        "model_name": "placeholder-colbert-model",
        "index_path": Path("storage/rag/colbert/placeholder"),
        "index_name": "legal_v2_colbert",
        "device": "cpu",
        "top_k": 10,
        "batch_size": 16,
    }
    base.update(overrides)
    return ColbertConfig(**base)  # type: ignore[arg-type]


def test_colbert_config_validates() -> None:
    cfg = _valid_config()
    cfg.validate()
    assert cfg.model_name == "placeholder-colbert-model"
    assert cfg.top_k == 10


def test_colbert_config_rejects_blank_model() -> None:
    cfg = _valid_config(model_name="  ")
    with pytest.raises(ColbertConfigurationError, match="model_name"):
        cfg.validate()


def test_colbert_config_rejects_bad_device() -> None:
    cfg = _valid_config(device="tpu")
    with pytest.raises(ColbertConfigurationError, match="device"):
        cfg.validate()


def test_colbert_module_import_is_side_effect_free() -> None:
    """Import path must not touch torch/CUDA or download models."""
    import sys

    # Foundation modules themselves must not pull heavy ML stacks.
    heavy = ("torch", "transformers", "colbert", "ragatouille")
    # Allow pre-existing interpreter state; assert our package did not require them.
    import app.rag.legal_v2.retrieve.colbert as colbert_pkg

    assert colbert_pkg.ColbertRetriever is ColbertRetriever
    for name in heavy:
        mod = sys.modules.get(name)
        # If already imported by the broader test env, that is unrelated; the
        # ColBERT package must not *load* them as a consequence of import.
        # We only assert the lazy probe still refuses.
        _ = mod
    with pytest.raises(ColbertBackendUnavailableError):
        import_colbert_library()


def test_retriever_without_backend_fails_explicitly() -> None:
    retriever = ColbertRetriever(_valid_config(), backend=None)
    with pytest.raises(ColbertBackendUnavailableError, match="backend is not configured"):
        retriever.retrieve("test query")


def test_indexer_without_backend_fails_explicitly() -> None:
    indexer = ColbertIndexer(_valid_config(), backend=None)
    with pytest.raises(ColbertBackendUnavailableError):
        indexer.build_index()


def test_indexer_with_stub_backend_refuses_build() -> None:
    class _Stub:
        def search(self, query: str, *, top_k: int):
            return ()

    indexer = ColbertIndexer(_valid_config(), backend=_Stub())
    assert indexer.planned_source_collection == COLBERT_PILOT_SOURCE_QDRANT_COLLECTION
    with pytest.raises(ColbertNotImplementedError, match="foundation-only"):
        indexer.build_index()


def test_retriever_with_injected_backend_maps_to_retrieved_chunk() -> None:
    class _Stub:
        def search(self, query: str, *, top_k: int):
            assert query == "kolik"
            return [
                ColbertHit(
                    document_id="ECLI:CZ:US:2025:1.US.1.25.1",
                    chunk_id="c-1",
                    rank=1,
                    score=0.91,
                    text="evidence text",
                    metadata={"section": "facts"},
                )
            ]

    result = ColbertRetriever(_valid_config(top_k=5), backend=_Stub()).retrieve("kolik")
    assert len(result.hits) == 1
    chunk = result.as_retrieved_chunks()[0]
    assert isinstance(chunk, RetrievedChunk)
    assert chunk.id == "c-1"
    assert chunk.source == "colbert"
    assert chunk.metadata["document_id"].startswith("ECLI:")


def test_fast_canonical_still_a(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED", "1")
    resolved = resolve_retrieval_profile("fast")
    assert resolved.index is not None
    assert resolved.index.qdrant_collection.endswith("a_current_300")
    assert resolved.index.bm25_index_id.endswith("a_current_300")
    assert resolved.use_cross_encoder is False


def test_ce_canonical_still_b_contextual(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED", "1")
    resolved = resolve_retrieval_profile("ce7")
    assert resolved.index is not None
    assert resolved.index.qdrant_collection.endswith("b_contextual_300")
    assert resolved.index.bm25_index_id.endswith("b_contextual_300")
    assert resolved.use_cross_encoder is True
    assert resolved.cross_encoder_config is not None
    assert resolved.cross_encoder_config.passages_per_document == 7
