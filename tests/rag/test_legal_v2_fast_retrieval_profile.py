"""FAST retrieval profile selector + canonical decision collapse tests."""

from __future__ import annotations

import pytest

from app.rag.legal_v2.query_spec import build_query_spec_v2
from app.rag.legal_v2.retrieve.fast_retrieval_profile import (
    normalize_fast_retrieval_profile,
    resolve_fast_retrieval_profile,
)
from app.rag.legal_v2.retrieve.retriever import (
    LegalV2HybridRetriever,
    LegalV2RetrieverConfig,
)
from app.rag.retrieval.bm25_sidecar import Bm25Record, Bm25Sidecar
from app.rag.retrieval.models import RetrievedChunk


def test_resolve_fast_profile_default_dense(monkeypatch) -> None:
    monkeypatch.delenv("NALUS_FAST_RETRIEVAL_PROFILE", raising=False)
    monkeypatch.delenv("NALUS_LEGAL_V2_STAGE1_FAST_DENSE_ONLY", raising=False)
    profile, source = resolve_fast_retrieval_profile()
    assert profile == "dense"
    assert source == "default"


def test_resolve_fast_profile_explicit_hybrid(monkeypatch) -> None:
    monkeypatch.setenv("NALUS_FAST_RETRIEVAL_PROFILE", "hybrid")
    monkeypatch.setenv("NALUS_LEGAL_V2_STAGE1_FAST_DENSE_ONLY", "1")
    profile, source = resolve_fast_retrieval_profile()
    assert profile == "hybrid"
    assert source == "NALUS_FAST_RETRIEVAL_PROFILE"


def test_resolve_fast_profile_legacy_dense_only(monkeypatch) -> None:
    monkeypatch.delenv("NALUS_FAST_RETRIEVAL_PROFILE", raising=False)
    monkeypatch.setenv("NALUS_LEGAL_V2_STAGE1_FAST_DENSE_ONLY", "1")
    profile, source = resolve_fast_retrieval_profile()
    assert profile == "dense"
    assert source == "NALUS_LEGAL_V2_STAGE1_FAST_DENSE_ONLY"


def test_invalid_fast_profile_fails_safe() -> None:
    with pytest.raises(ValueError, match="unknown NALUS_FAST_RETRIEVAL_PROFILE"):
        normalize_fast_retrieval_profile("colbert")


def test_dense_mode_does_not_call_bm25() -> None:
    dense_chunks = [
        RetrievedChunk(
            "c1",
            "škoda",
            0.9,
            "dense",
            {
                "document_id": "ECLI:CZ:US:2024:1.US.1.24.1",
                "ecli": "ECLI:CZ:US:2024:1.US.1.24.1",
                "chunk_index": 0,
            },
        ),
        RetrievedChunk(
            "c2",
            "škoda pokračování",
            0.85,
            "dense",
            {
                "document_id": "ECLI:CZ:US:2024:1.US.1.24.1",
                "ecli": "ECLI:CZ:US:2024:1.US.1.24.1",
                "chunk_index": 1,
            },
        ),
        RetrievedChunk(
            "c3",
            "jiné rozhodnutí",
            0.8,
            "dense",
            {
                "document_id": "ECLI:CZ:US:2024:2.US.2.24.1",
                "ecli": "ECLI:CZ:US:2024:2.US.2.24.1",
                "chunk_index": 0,
            },
        ),
    ]

    class Dense:
        def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
            del query
            return list(dense_chunks[:top_k])

    class ForbiddenBm25:
        def search(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("BM25 must not initialize/search in dense mode")

    retriever = LegalV2HybridRetriever(
        dense_store=Dense(),
        bm25_sidecar=ForbiddenBm25(),  # type: ignore[arg-type]
        config=LegalV2RetrieverConfig(
            dense_enabled=True,
            bm25_enabled=False,
            dense_candidate_chunks=10,
            candidate_documents=20,
        ),
    )
    result = retriever.retrieve(build_query_spec_v2("škoda zaměstnance"))
    assert result.diagnostics["dense_only"] is True
    assert result.diagnostics["bm25_only"] is False
    assert result.bm25_results == []
    # Multiple chunks from one decision → one rank.
    assert [doc.document_id for doc in result.documents] == [
        "ECLI:CZ:US:2024:1.US.1.24.1",
        "ECLI:CZ:US:2024:2.US.2.24.1",
    ]


def test_bm25_only_does_not_call_dense() -> None:
    payload = {
        "document_id": "ECLI:CZ:US:2024:3.US.3.24.1",
        "ecli": "ECLI:CZ:US:2024:3.US.3.24.1",
        "chunk_index": 0,
    }
    bm25 = Bm25Sidecar.from_records(
        [Bm25Record("chunk-1", "vyživovací povinnost zánik", dict(payload))],
        k1=1.5,
        b=0.75,
        index_id="test-bm25",
    )

    class ForbiddenDense:
        def search(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("Dense must not run in bm25-only mode")

    retriever = LegalV2HybridRetriever(
        dense_store=ForbiddenDense(),
        bm25_sidecar=bm25,
        config=LegalV2RetrieverConfig(
            dense_enabled=False,
            bm25_enabled=True,
            bm25_candidate_chunks=5,
            candidate_documents=10,
        ),
    )
    result = retriever.retrieve(build_query_spec_v2("vyživovací povinnost"))
    assert result.diagnostics["bm25_only"] is True
    assert result.dense_results == []
    assert result.documents[0].document_id == "ECLI:CZ:US:2024:3.US.3.24.1"


def test_hybrid_uses_dense_and_bm25() -> None:
    payload = {
        "document_id": "ECLI:CZ:US:2024:4.US.4.24.1",
        "ecli": "ECLI:CZ:US:2024:4.US.4.24.1",
        "chunk_index": 0,
    }
    dense = [
        RetrievedChunk("chunk-1", "škoda a zavinění zaměstnance", 0.4, "dense", dict(payload))
    ]
    bm25 = Bm25Sidecar.from_records(
        [Bm25Record("chunk-1", "škoda a zavinění zaměstnance", dict(payload))],
        k1=1.5,
        b=0.75,
        index_id="test-bm25",
    )

    class Dense:
        def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
            del query, top_k
            return list(dense)

    retriever = LegalV2HybridRetriever(
        dense_store=Dense(),
        bm25_sidecar=bm25,
        config=LegalV2RetrieverConfig(
            dense_enabled=True,
            bm25_enabled=True,
            dense_candidate_chunks=5,
            bm25_candidate_chunks=5,
        ),
    )
    result = retriever.retrieve(build_query_spec_v2("škoda zavinění zaměstnance"))
    assert result.diagnostics["dense_enabled"] is True
    assert result.diagnostics["bm25_enabled"] is True
    assert result.diagnostics["dense_only"] is False
    assert result.fused_results[0].id == "chunk-1"


def test_top_n_means_unique_decisions() -> None:
    chunks = []
    for decision_i in range(25):
        ecli = f"ECLI:CZ:US:2024:1.US.{decision_i}.24.1"
        for chunk_i in range(3):
            chunks.append(
                RetrievedChunk(
                    f"d{decision_i}-c{chunk_i}",
                    f"text {decision_i} {chunk_i}",
                    1.0 - decision_i * 0.01 - chunk_i * 0.001,
                    "dense",
                    {
                        "document_id": ecli,
                        "ecli": ecli,
                        "chunk_index": chunk_i,
                    },
                )
            )

    class Dense:
        def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
            del query
            return list(chunks[:top_k])

    class ForbiddenBm25:
        def search(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("BM25 must not run")

    retriever = LegalV2HybridRetriever(
        dense_store=Dense(),
        bm25_sidecar=ForbiddenBm25(),  # type: ignore[arg-type]
        config=LegalV2RetrieverConfig(
            dense_enabled=True,
            bm25_enabled=False,
            dense_candidate_chunks=80,
            fused_candidate_chunks=120,
            candidate_documents=20,
        ),
    )
    result = retriever.retrieve(build_query_spec_v2("test query unique decisions"))
    assert len(result.documents) == 20
    ids = [doc.document_id for doc in result.documents]
    assert len(ids) == len(set(ids))


def test_compatible_result_schema_across_modes() -> None:
    """dense/bm25/hybrid expose the same CandidateEvidenceDocument fields."""
    payload = {
        "document_id": "ECLI:CZ:US:2024:5.US.5.24.1",
        "ecli": "ECLI:CZ:US:2024:5.US.5.24.1",
        "chunk_index": 0,
    }
    dense_chunk = RetrievedChunk("chunk-1", "nájem bytu", 0.9, "dense", dict(payload))
    bm25 = Bm25Sidecar.from_records(
        [Bm25Record("chunk-1", "nájem bytu", dict(payload))],
        k1=1.5,
        b=0.75,
        index_id="test-bm25",
    )

    class Dense:
        def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
            del query, top_k
            return [dense_chunk]

    configs = [
        LegalV2RetrieverConfig(dense_enabled=True, bm25_enabled=False),
        LegalV2RetrieverConfig(dense_enabled=False, bm25_enabled=True),
        LegalV2RetrieverConfig(dense_enabled=True, bm25_enabled=True),
    ]
    for config in configs:
        retriever = LegalV2HybridRetriever(
            dense_store=Dense(),
            bm25_sidecar=bm25,
            config=config,
        )
        docs = retriever.retrieve(build_query_spec_v2("nájem")).documents
        assert docs
        doc = docs[0]
        assert doc.document_id == "ECLI:CZ:US:2024:5.US.5.24.1"
        assert hasattr(doc, "score")
        assert hasattr(doc, "chunk_evidence")
