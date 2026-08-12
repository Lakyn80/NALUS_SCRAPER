"""Unit/async tests for Legal v2 ColBERT foundation + backend wiring."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from app.rag.legal_v2.retrieve.colbert import (
    COLBERT_PILOT_SOURCE_QDRANT_COLLECTION,
    ColbertBackendUnavailableError,
    ColbertConfig,
    ColbertConfigurationError,
    ColbertHit,
    ColbertIndexBuildResult,
    ColbertIndexer,
    ColbertRetriever,
    import_colbert_library,
)
from app.rag.legal_v2.retrieve.colbert.mapping import (
    ColbertChunkMapping,
    ColbertMappingRow,
    load_mapping_jsonl,
    write_mapping_jsonl,
)
from app.rag.legal_v2.retrieve.retrieval_profiles import resolve_retrieval_profile
from app.rag.retrieval.models import RetrievedChunk


def _valid_config(**overrides: object) -> ColbertConfig:
    base: dict[str, object] = {
        "model_name": "placeholder-colbert-model",
        "index_path": Path("storage/rag/colbert/placeholder"),
        "index_name": "legal_v2_colbert",
        "device": "cpu",
        "top_k": 10,
        "batch_size": 16,
        "concurrency_limit": 1,
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
    import app.rag.legal_v2.retrieve.colbert as colbert_pkg

    assert colbert_pkg.ColbertRetriever is ColbertRetriever
    # Lazy probe still refuses unless pylate is installed; either outcome is fine
    # as long as import itself did not initialize CUDA/index.
    try:
        import_colbert_library()
    except ColbertBackendUnavailableError:
        pass


@pytest.mark.asyncio
async def test_retriever_without_backend_fails_explicitly() -> None:
    retriever = ColbertRetriever(_valid_config(), backend=None)
    with pytest.raises(ColbertBackendUnavailableError, match="backend is not configured"):
        await retriever.retrieve("test query")


@pytest.mark.asyncio
async def test_indexer_without_backend_fails_explicitly() -> None:
    indexer = ColbertIndexer(_valid_config(), backend=None)
    with pytest.raises(ColbertBackendUnavailableError):
        await indexer.build([{"chunk_id": "c1", "document_id": "d1", "text": "t"}])


@pytest.mark.asyncio
async def test_indexer_requires_documents() -> None:
    class _Stub:
        async def initialize(self) -> None:
            return None

        async def close(self) -> None:
            return None

        async def search(self, query: str, *, top_k: int):
            return ()

        async def build_index(self, documents, *, source_collection=None):
            return ColbertIndexBuildResult(
                status="ok",
                source_collection=source_collection or "x",
                expected_chunk_count=1,
                indexed_chunk_count=1,
                mapping_row_count=1,
                duplicate_chunk_ids=0,
                missing_chunk_ids=0,
                empty_texts=0,
                index_path="idx",
                mapping_path="map",
                model_name="m",
                library="stub",
                library_version="0",
                device="cpu",
            )

    indexer = ColbertIndexer(_valid_config(), backend=_Stub())
    assert indexer.planned_source_collection == COLBERT_PILOT_SOURCE_QDRANT_COLLECTION
    with pytest.raises(ColbertConfigurationError, match="documents are required"):
        await indexer.build(None)


@pytest.mark.asyncio
async def test_retriever_with_injected_backend_maps_to_retrieved_chunk() -> None:
    class _Stub:
        async def initialize(self) -> None:
            return None

        async def close(self) -> None:
            return None

        async def search(self, query: str, *, top_k: int):
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

        async def build_index(self, documents, *, source_collection=None):
            raise AssertionError("not used")

    result = await ColbertRetriever(_valid_config(top_k=5), backend=_Stub()).retrieve(
        "kolik"
    )
    assert len(result.hits) == 1
    chunk = result.as_retrieved_chunks()[0]
    assert isinstance(chunk, RetrievedChunk)
    assert chunk.id == "c-1"
    assert chunk.source == "colbert"
    assert chunk.metadata["document_id"].startswith("ECLI:")


@pytest.mark.asyncio
async def test_backend_search_uses_worker_thread() -> None:
    main_thread = threading.get_ident()
    seen: dict[str, int] = {}

    class _Stub:
        async def initialize(self) -> None:
            return None

        async def close(self) -> None:
            return None

        async def search(self, query: str, *, top_k: int):
            # Mimic production backend: offload blocking work.
            def _blocking():
                seen["thread"] = threading.get_ident()
                return [
                    ColbertHit(
                        document_id="ECLI:CZ:US:2025:1.US.1.25.1",
                        chunk_id="c-1",
                        rank=1,
                        score=1.0,
                        text="t",
                    )
                ]

            return await asyncio.to_thread(_blocking)

        async def build_index(self, documents, *, source_collection=None):
            raise AssertionError("not used")

    await ColbertRetriever(_valid_config(), backend=_Stub()).retrieve("q")
    assert "thread" in seen
    assert seen["thread"] != main_thread


@pytest.mark.asyncio
async def test_backend_exception_propagates_through_async_api() -> None:
    class _Boom:
        async def initialize(self) -> None:
            return None

        async def close(self) -> None:
            return None

        async def search(self, query: str, *, top_k: int):
            raise RuntimeError("backend exploded")

        async def build_index(self, documents, *, source_collection=None):
            raise AssertionError("not used")

    with pytest.raises(RuntimeError, match="backend exploded"):
        await ColbertRetriever(_valid_config(), backend=_Boom()).retrieve("q")


def test_mapping_roundtrip_and_integrity(tmp_path: Path) -> None:
    path = tmp_path / "mapping.jsonl"
    rows = [
        ColbertMappingRow(
            colbert_id="c1",
            chunk_id="c1",
            document_id="ECLI:CZ:US:2025:1.US.1.25.1",
            text="alpha",
            metadata={"section_type": "facts"},
        ),
        ColbertMappingRow(
            colbert_id="c2",
            chunk_id="c2",
            document_id="ECLI:CZ:US:2025:1.US.2.25.1",
            text="beta",
        ),
    ]
    assert write_mapping_jsonl(path, rows) == 2
    loaded = load_mapping_jsonl(path)
    assert len(loaded) == 2
    stats = loaded.integrity(expected_chunk_ids={"c1", "c2"})
    assert stats["duplicate_chunk_ids"] == 0
    assert stats["missing_chunk_ids"] == 0
    assert stats["empty_texts"] == 0


def test_mapping_missing_file(tmp_path: Path) -> None:
    from app.rag.legal_v2.retrieve.colbert.errors import ColbertMappingError

    with pytest.raises(ColbertMappingError, match="missing"):
        load_mapping_jsonl(tmp_path / "nope.jsonl")


@pytest.mark.asyncio
async def test_pylate_backend_missing_index_and_mapping(tmp_path: Path) -> None:
    from app.rag.legal_v2.retrieve.colbert.errors import (
        ColbertIndexError,
        ColbertMappingError,
    )
    from app.rag.legal_v2.retrieve.colbert.pylate_backend import PyLateColbertBackend

    missing_index = PyLateColbertBackend(
        _valid_config(index_path=tmp_path / "no-index", mapping_path=tmp_path / "m.jsonl")
    )
    with pytest.raises(ColbertIndexError, match="index path missing"):
        await missing_index.initialize()

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    missing_map = PyLateColbertBackend(
        _valid_config(index_path=index_dir, mapping_path=tmp_path / "missing.jsonl")
    )
    with pytest.raises(ColbertMappingError, match="mapping missing"):
        await missing_map.initialize()



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
