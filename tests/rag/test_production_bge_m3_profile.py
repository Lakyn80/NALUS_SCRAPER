from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.api import startup
from app.rag.retrieval.bge_m3_embedder import BgeM3Embedder
from app.rag.retrieval.bm25_sidecar import Bm25Record, Bm25Sidecar
from app.rag.retrieval.errors import RetrievalConfigurationError, RetrievalDependencyError
from app.rag.retrieval.hybrid_bge_m3_retriever import HybridBgeM3Retriever
from app.rag.retrieval.models import RetrievedChunk
from app.rag.retrieval.production_profile import (
    BGE_M3_DENSE_BM25_RRF,
    production_retrieval_config_from_env,
)
from app.rag.retrieval.provenance import build_embedding_provenance, validate_embedding_provenance
from app.rag.retrieval.rrf import rrf_fuse


def test_production_profile_is_bge_m3_dense_bm25_rrf() -> None:
    profile = BGE_M3_DENSE_BM25_RRF

    assert profile.name == "nalus_bge_m3_dense_bm25_rrf_v1"
    assert profile.embedding_model == "BAAI/bge-m3"
    assert profile.embedding_dimension == 1024
    assert profile.retrieval_mode == "dense_plus_bm25"
    assert profile.fusion == "rrf"
    assert profile.rrf_k == 60
    assert profile.bm25_k1 == 1.5
    assert profile.bm25_b == 0.75


def test_env_profile_rejects_hash_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL_PROFILE", "hash_smoke_only")

    with pytest.raises(RetrievalConfigurationError):
        production_retrieval_config_from_env()


def test_env_profile_rejects_mpnet_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_DIMENSION", "768")

    with pytest.raises(RetrievalConfigurationError):
        production_retrieval_config_from_env()


def test_env_profile_rejects_mpnet_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )

    with pytest.raises(RetrievalConfigurationError, match="MPNet"):
        production_retrieval_config_from_env()


def test_bge_m3_embedder_does_not_load_model_on_init() -> None:
    config = production_retrieval_config_from_env()
    embedder = BgeM3Embedder(config)

    assert embedder.loaded is False


def test_bge_m3_embedder_loads_only_when_embedding() -> None:
    class _FakeModel:
        def __init__(self) -> None:
            self.calls = 0

        def encode(self, texts, batch_size, normalize_embeddings, show_progress_bar):
            del batch_size, normalize_embeddings, show_progress_bar
            self.calls += 1
            return [[0.0] * 1024 for _ in texts]

    model = _FakeModel()
    config = production_retrieval_config_from_env()
    embedder = BgeM3Embedder(config, model=model)

    assert embedder.loaded is True
    assert embedder.embed_query("test") == [0.0] * 1024
    assert model.calls == 1


def test_missing_bm25_sidecar_is_refused(tmp_path: Path) -> None:
    sidecar = Bm25Sidecar(
        tmp_path / "missing.sqlite",
        k1=1.5,
        b=0.75,
        index_id=BGE_M3_DENSE_BM25_RRF.name,
    )

    with pytest.raises(RetrievalConfigurationError):
        sidecar.assert_ready()


def test_bm25_ranking_is_deterministic() -> None:
    sidecar = Bm25Sidecar.from_records(
        [
            Bm25Record("a", "opomenuté důkazy a spravedlivý proces", {"document_id": "a", "chunk_index": 0}),
            Bm25Record("b", "náklady řízení", {"document_id": "b", "chunk_index": 0}),
        ],
        k1=1.5,
        b=0.75,
        index_id=BGE_M3_DENSE_BM25_RRF.name,
    )

    first = sidecar.search("opomenuté důkazy", top_k=2)
    second = sidecar.search("opomenuté důkazy", top_k=2)

    assert [item.id for item in first] == ["a"]
    assert [item.id for item in second] == ["a"]


def test_rrf_fusion_is_deterministic() -> None:
    dense = [RetrievedChunk("a", "A", 0.9, "dense"), RetrievedChunk("b", "B", 0.8, "dense")]
    bm25 = [RetrievedChunk("b", "B", 3.0, "bm25"), RetrievedChunk("a", "A", 2.0, "bm25")]

    fused = rrf_fuse([dense, bm25], top_k=2, rrf_k=60)

    assert [item.id for item in fused] == ["a", "b"]
    assert fused[0].source == "hybrid"
    assert fused[0].metadata["score_components"]["dense"] == 0.9


def test_hybrid_retriever_requires_dense_and_bm25_results() -> None:
    class _Dense:
        def search(self, query: str, top_k: int):
            del query, top_k
            return []

    sidecar = Bm25Sidecar.from_records(
        [Bm25Record("a", "spravedlivý proces", {"document_id": "a", "chunk_index": 0})],
        k1=1.5,
        b=0.75,
        index_id=BGE_M3_DENSE_BM25_RRF.name,
    )
    retriever = HybridBgeM3Retriever(
        dense_store=_Dense(),
        bm25_sidecar=sidecar,
        config=production_retrieval_config_from_env(),
    )

    with pytest.raises(RetrievalDependencyError):
        retriever.search("spravedlivý proces")


def test_provenance_required_and_validated() -> None:
    payload = build_embedding_provenance(
        payload={"source": "nalus", "document_id": "doc-1", "chunk_index": 0, "text": "text"},
        profile=BGE_M3_DENSE_BM25_RRF,
        ingest_run_id="run-1",
        qdrant_collection="nalus_bge_m3_chunks_v1",
        bm25_index_id=BGE_M3_DENSE_BM25_RRF.name,
    )

    validate_embedding_provenance(
        payload,
        profile=BGE_M3_DENSE_BM25_RRF,
        qdrant_collection="nalus_bge_m3_chunks_v1",
        bm25_index_id=BGE_M3_DENSE_BM25_RRF.name,
    )


def test_missing_bge_m3_model_path_fails_at_startup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    missing = tmp_path / "missing-model"
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", str(missing))

    config = production_retrieval_config_from_env()
    with pytest.raises(RuntimeError, match="BGE-M3 model path is missing"):
        startup._assert_bge_m3_model_ready(config)


def test_api_runtime_modules_do_not_import_nalus_legal_rag() -> None:
    project_root = Path(__file__).resolve().parents[2]
    for module_path in (
        Path(startup.__file__),
        project_root / "app" / "api" / "rag_router.py",
    ):
        source = module_path.read_text(encoding="utf-8")
        assert "nalus_legal_rag" not in source


def test_startup_factory_does_not_import_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    imported: list[str] = []
    real_import = importlib.import_module

    def spy_import(name: str, package: str | None = None):
        imported.append(name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", spy_import)
    config = production_retrieval_config_from_env()
    startup._build_production_retrieval(SimpleNamespace(), config)

    assert not any(name.startswith("sentence_transformers") for name in imported)


def test_api_startup_module_does_not_instantiate_keyword_retriever() -> None:
    source = Path(startup.__file__).read_text(encoding="utf-8")

    assert "KeywordRetriever(" not in source
    assert "paraphrase-multilingual-mpnet-base-v2" not in source
    assert "SentenceTransformersEmbedder" not in source
