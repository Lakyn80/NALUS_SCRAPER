"""Unit tests for Stage 1 request-level retrieval profiles."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.rag.legal_v2.retrieve.retrieval_profiles import (
    RetrievalStage,
    build_retrieval_stage,
    resolve_retrieval_profile,
)
from app.rag.legal_v2.rerank.selectors.names import DIVERSIFIED_STAGE1_EVIDENCE_V1


def test_fast_profile_disables_ce(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED", "1")
    monkeypatch.setenv("NALUS_LEGAL_V2_COLBERT_ENABLED", "1")
    resolved = resolve_retrieval_profile("fast")
    assert resolved.profile_id == "fast"
    assert resolved.label == "FAST"
    assert resolved.use_cross_encoder is False
    assert resolved.use_colbert is False
    assert resolved.cross_encoder_config is None
    assert resolved.index is not None
    assert resolved.index.qdrant_collection.endswith("a_current_300")
    assert resolved.index.bm25_index_id.endswith("a_current_300")


def test_precise_requires_master_allow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED", "0")
    with pytest.raises(ValueError, match="CROSS_ENCODER_ENABLED"):
        resolve_retrieval_profile("precise")


def test_ce7_alias_maps_to_precise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED", "1")
    resolved = resolve_retrieval_profile("ce7")
    assert resolved.profile_id == "precise"
    assert resolved.label == "PRECISE"
    assert resolved.use_cross_encoder is True
    assert resolved.use_colbert is False


def test_precise_profile_forces_diversified_seven(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED", "1")
    monkeypatch.setenv("NALUS_LEGAL_V2_CE_PASSAGES_PER_DOCUMENT", "3")
    monkeypatch.setenv(
        "NALUS_LEGAL_V2_CE_PASSAGE_SELECTOR", "first_n_stage1_order_v1"
    )
    resolved = resolve_retrieval_profile("precise")
    assert resolved.use_cross_encoder is True
    assert resolved.cross_encoder_config is not None
    assert resolved.cross_encoder_config.passages_per_document == 7
    assert (
        resolved.cross_encoder_config.passage_selector
        == DIVERSIFIED_STAGE1_EVIDENCE_V1
    )
    assert resolved.cross_encoder_config.evidence_pool_limit >= 40
    assert resolved.cross_encoder_config.model_id == "BAAI/bge-reranker-v2-m3"
    assert resolved.index is not None
    assert resolved.index.qdrant_collection.endswith("b_contextual_300")
    assert resolved.index.bm25_index_id.endswith("b_contextual_300")


def test_balanced_requires_colbert_master_allow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NALUS_LEGAL_V2_COLBERT_ENABLED", "0")
    with pytest.raises(ValueError, match="COLBERT_ENABLED"):
        resolve_retrieval_profile("balanced")


def test_balanced_profile_uses_b_indexes_and_colbert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NALUS_LEGAL_V2_COLBERT_ENABLED", "1")
    resolved = resolve_retrieval_profile("balanced")
    assert resolved.profile_id == "balanced"
    assert resolved.label == "BALANCED"
    assert resolved.use_colbert is True
    assert resolved.use_cross_encoder is False
    assert resolved.colbert_candidate_chunks == 80
    assert resolved.index is not None
    assert resolved.index.qdrant_collection.endswith("b_contextual_300")


def test_fast_and_ce_index_bindings_are_distinct() -> None:
    from app.rag.legal_v2.retrieve.retrieval_profiles import (
        balanced_index_binding,
        ce_index_binding,
        fast_index_binding,
    )

    fast = fast_index_binding()
    ce = ce_index_binding()
    balanced = balanced_index_binding()
    assert fast.qdrant_collection != ce.qdrant_collection
    assert fast.bm25_index_id != ce.bm25_index_id
    assert "a_current_300" in fast.qdrant_collection
    assert "b_contextual_300" in ce.qdrant_collection
    assert balanced.qdrant_collection == ce.qdrant_collection
    assert fast.bm25_sidecar_path.name.endswith("a_current_300.sqlite")
    assert ce.bm25_sidecar_path.name.endswith("b_contextual_300.sqlite")


def test_unknown_profile_rejected() -> None:
    with pytest.raises(ValueError, match="unknown retrieval_profile"):
        resolve_retrieval_profile("turbo")


def test_build_retrieval_stage_fast_when_not_applied() -> None:
    assert (
        build_retrieval_stage(rerank_applied=False, passages_per_document=7)
        == RetrievalStage.HYBRID_RRF_STAGE_1.value
    )


def test_build_retrieval_stage_colbert_when_applied() -> None:
    assert (
        build_retrieval_stage(rerank_applied=False, colbert_applied=True)
        == RetrievalStage.HYBRID_RRF_COLBERT.value
    )


def test_build_retrieval_stage_ce7_wins_over_colbert() -> None:
    assert (
        build_retrieval_stage(
            rerank_applied=True,
            passages_per_document=7,
            colbert_applied=True,
        )
        == RetrievalStage.HYBRID_RRF_CE7.value
    )


def test_build_retrieval_stage_ce7_when_applied_with_seven_passages() -> None:
    assert (
        build_retrieval_stage(rerank_applied=True, passages_per_document=7)
        == RetrievalStage.HYBRID_RRF_CE7.value
    )


def test_build_retrieval_stage_generic_ce_when_applied_other_passages() -> None:
    assert (
        build_retrieval_stage(rerank_applied=True, passages_per_document=3)
        == RetrievalStage.HYBRID_RRF_CE.value
    )


def test_build_retrieval_stage_ignores_config_intent_without_applied() -> None:
    """CE may be configured/requested, but stage stays Stage 1 unless applied."""
    assert (
        build_retrieval_stage(rerank_applied=False, passages_per_document=7)
        == "hybrid_rrf_stage_1"
    )


def _fake_docs():
    class _Para:
        def __init__(self, text: str, pid: str) -> None:
            self.normalized_text = text
            self.original_text = text
            self.paragraph_id = pid
            self.section_type = "facts"

    class _Doc:
        def __init__(self, ecli: str, score: float) -> None:
            self.document_id = ecli
            self.score = score
            self.paragraphs = [_Para(f"text {ecli}", f"c-{ecli}")]
            self.metadata = {"ecli": ecli}
            self.dense_rank = 1
            self.bm25_rank = 1
            self.rrf_score = score
            self.chunk_evidence = []

    return [
        _Doc("ECLI:CZ:US:2025:1.US.1111.25.1", 0.9),
        _Doc("ECLI:CZ:US:2025:1.US.2222.25.1", 0.8),
    ]


def test_search_default_profile_is_fast_even_when_ce_env_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.rag.legal_v2.retrieve import case_similarity_search as module
    from app.rag.legal_v2.retrieve.case_similarity_search import (
        CaseSimilarityStage1Runtime,
    )
    from app.rag.legal_v2.retrieve.retriever import LegalV2RetrievalResult

    monkeypatch.setenv("NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED", "1")
    ordered = _fake_docs()

    class _FakeRetriever:
        def retrieve(self, query_spec):
            return LegalV2RetrievalResult(
                documents=ordered,  # type: ignore[arg-type]
                dense_results=[],
                bm25_results=[],
                fused_results=[],
                diagnostics={
                    "dense_latency_ms": 1.0,
                    "bm25_latency_ms": 1.0,
                    "rrf_latency_ms": 1.0,
                    "total_retrieval_latency_ms": 3.0,
                    "dense_candidate_chunks": 2,
                    "bm25_candidate_chunks": 2,
                    "fused_candidate_chunks": 2,
                    "candidate_documents": 2,
                },
            )

    runtime = CaseSimilarityStage1Runtime(
        retriever=_FakeRetriever(),  # type: ignore[arg-type]
        config=MagicMock(
            qdrant_collection="nalus_legal_paragraph_chunks_v2_chunk_ab_v8_a_current_300",
            bm25_index_id="nalus_legal_paragraph_bm25_v2_chunk_ab_v8_a_current_300",
            fused_candidate_chunks=120,
            candidate_documents=40,
        ),
        ready=True,
        ce_retriever=_FakeRetriever(),  # type: ignore[arg-type]
        ce_config=MagicMock(
            qdrant_collection="nalus_legal_paragraph_chunks_v2_chunk_ab_v8_b_contextual_300",
            bm25_index_id="nalus_legal_paragraph_bm25_v2_chunk_ab_v8_b_contextual_300",
            fused_candidate_chunks=120,
            candidate_documents=40,
        ),
    )
    result = asyncio.run(
        module.search_case_similarity_stage1(
            query="krátký testovací dotaz",
            limit=2,
            runtime=runtime,
            retrieval_profile="fast",
        )
    )
    assert result.diagnostics["retrieval_profile"] == "fast"
    assert result.diagnostics["rerank"]["rerank_applied"] is False
    assert result.retrieval_stage == "hybrid_rrf_stage_1"
    assert (
        result.diagnostics["collection"]
        == "nalus_legal_paragraph_chunks_v2_chunk_ab_v8_a_current_300"
    )
    assert [row.ecli for row in result.results] == [
        "ECLI:CZ:US:2025:1.US.1111.25.1",
        "ECLI:CZ:US:2025:1.US.2222.25.1",
    ]


def test_search_balanced_reports_hybrid_rrf_colbert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.rag.legal_v2.retrieve import case_similarity_search as module
    from app.rag.legal_v2.retrieve.case_similarity_search import (
        CaseSimilarityStage1Runtime,
    )
    from app.rag.legal_v2.retrieve.colbert_hybrid import ColbertHybridRetrievalResult
    from app.rag.legal_v2.retrieve.retriever import LegalV2RetrievalResult

    monkeypatch.setenv("NALUS_LEGAL_V2_COLBERT_ENABLED", "1")
    ordered = _fake_docs()

    class _FakeRetriever:
        def retrieve(self, query_spec):
            return LegalV2RetrievalResult(
                documents=ordered,  # type: ignore[arg-type]
                dense_results=[],
                bm25_results=[],
                fused_results=[],
                diagnostics={},
            )

    async def _fake_hybrid(**kwargs):
        return ColbertHybridRetrievalResult(
            documents=ordered,  # type: ignore[arg-type]
            dense_results=[],
            bm25_results=[],
            colbert_results=[],
            fused_results=[],
            diagnostics={
                "dense_latency_ms": 1.0,
                "bm25_latency_ms": 1.0,
                "colbert_latency_ms": 2.0,
                "rrf_latency_ms": 0.5,
                "total_retrieval_latency_ms": 4.5,
                "dense_candidate_chunks": 2,
                "bm25_candidate_chunks": 2,
                "colbert_candidate_chunks": 2,
                "fused_candidate_chunks": 2,
                "candidate_documents": 2,
            },
        )

    monkeypatch.setattr(
        module,
        "_ensure_colbert_retriever",
        AsyncMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        "app.rag.legal_v2.retrieve.colbert_hybrid.retrieve_hybrid_plus_colbert",
        _fake_hybrid,
    )

    runtime = CaseSimilarityStage1Runtime(
        retriever=_FakeRetriever(),  # type: ignore[arg-type]
        config=MagicMock(
            qdrant_collection="nalus_legal_paragraph_chunks_v2_chunk_ab_v8_a_current_300",
            bm25_index_id="nalus_legal_paragraph_bm25_v2_chunk_ab_v8_a_current_300",
            fused_candidate_chunks=120,
            candidate_documents=40,
        ),
        ready=True,
        ce_retriever=_FakeRetriever(),  # type: ignore[arg-type]
        ce_config=MagicMock(
            qdrant_collection="nalus_legal_paragraph_chunks_v2_chunk_ab_v8_b_contextual_300",
            bm25_index_id="nalus_legal_paragraph_bm25_v2_chunk_ab_v8_b_contextual_300",
            fused_candidate_chunks=120,
            candidate_documents=40,
        ),
    )
    result = asyncio.run(
        module.search_case_similarity_stage1(
            query="krátký testovací dotaz",
            limit=2,
            runtime=runtime,
            retrieval_profile="balanced",
        )
    )
    assert result.retrieval_stage == "hybrid_rrf_colbert"
    assert result.diagnostics["retrieval_profile"] == "balanced"
    assert result.diagnostics["colbert_applied"] is True
    assert (
        result.diagnostics["collection"]
        == "nalus_legal_paragraph_chunks_v2_chunk_ab_v8_b_contextual_300"
    )


def test_search_precise_applied_reports_hybrid_rrf_ce7(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.rag.legal_v2.rerank.models import RerankDiagnostics, RerankedDocument, RerankResult
    from app.rag.legal_v2.retrieve import case_similarity_search as module
    from app.rag.legal_v2.retrieve.case_similarity_search import (
        CaseSimilarityStage1Runtime,
    )
    from app.rag.legal_v2.retrieve.retriever import LegalV2RetrievalResult

    monkeypatch.setenv("NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED", "1")
    ordered = _fake_docs()

    class _FakeRetriever:
        def retrieve(self, query_spec):
            return LegalV2RetrievalResult(
                documents=ordered,  # type: ignore[arg-type]
                dense_results=[],
                bm25_results=[],
                fused_results=[],
                diagnostics={
                    "dense_latency_ms": 1.0,
                    "bm25_latency_ms": 1.0,
                    "rrf_latency_ms": 1.0,
                    "total_retrieval_latency_ms": 3.0,
                    "dense_candidate_chunks": 2,
                    "bm25_candidate_chunks": 2,
                    "fused_candidate_chunks": 2,
                    "candidate_documents": 2,
                },
            )

    class _FakeService:
        def rerank(self, query, candidates, require_success=True):
            reversed_docs = list(reversed(candidates))
            docs = tuple(
                RerankedDocument(
                    ecli=doc.ecli,
                    stage1_rank=doc.rank,
                    stage1_score=doc.score,
                    ce_rank=index,
                    ce_score=1.0 - (index * 0.1),
                    passage_scores=(),
                    dense_rank=doc.dense_rank,
                    bm25_rank=doc.bm25_rank,
                    rrf_score=doc.rrf_score,
                    metadata=dict(doc.metadata),
                )
                for index, doc in enumerate(reversed_docs, start=1)
            )
            return RerankResult(
                documents=docs,
                diagnostics=RerankDiagnostics(
                    rerank_enabled=True,
                    rerank_applied=True,
                    reranker_model="BAAI/bge-reranker-v2-m3",
                    reranker_device="cpu",
                    candidate_document_count=len(docs),
                    passage_count=14,
                    pair_count=14,
                    batch_count=1,
                    truncated_pair_count=0,
                    aggregation="max",
                    rerank_latency_ms=1.0,
                    requested_passages_per_document=7,
                    passage_selector=DIVERSIFIED_STAGE1_EVIDENCE_V1,
                ),
            )

    monkeypatch.setattr(
        "app.rag.legal_v2.rerank.service.get_cross_encoder_reranking_service",
        lambda config: _FakeService(),
    )

    runtime = CaseSimilarityStage1Runtime(
        retriever=_FakeRetriever(),  # type: ignore[arg-type]
        config=MagicMock(
            qdrant_collection="nalus_legal_paragraph_chunks_v2_chunk_ab_v8_a_current_300",
            bm25_index_id="nalus_legal_paragraph_bm25_v2_chunk_ab_v8_a_current_300",
            fused_candidate_chunks=120,
            candidate_documents=40,
        ),
        ready=True,
        ce_retriever=_FakeRetriever(),  # type: ignore[arg-type]
        ce_config=MagicMock(
            qdrant_collection="nalus_legal_paragraph_chunks_v2_chunk_ab_v8_b_contextual_300",
            bm25_index_id="nalus_legal_paragraph_bm25_v2_chunk_ab_v8_b_contextual_300",
            fused_candidate_chunks=120,
            candidate_documents=40,
        ),
    )
    result = asyncio.run(
        module.search_case_similarity_stage1(
            query="krátký testovací dotaz",
            limit=2,
            runtime=runtime,
            retrieval_profile="ce7",
        )
    )
    assert result.retrieval_stage == "hybrid_rrf_ce7"
    assert result.diagnostics["retrieval_profile"] == "precise"
    assert result.diagnostics["rerank"]["rerank_applied"] is True
    assert result.diagnostics["rerank"]["requested_passages_per_document"] == 7
    assert (
        result.diagnostics["collection"]
        == "nalus_legal_paragraph_chunks_v2_chunk_ab_v8_b_contextual_300"
    )
    assert [row.ecli for row in result.results] == [
        "ECLI:CZ:US:2025:1.US.2222.25.1",
        "ECLI:CZ:US:2025:1.US.1111.25.1",
    ]
    assert result.results[0].stage1_rank == 2
    assert result.results[0].ce_rank == 1
    assert result.results[0].ce_score == pytest.approx(0.9)
    assert result.results[1].stage1_rank == 1
    assert result.results[1].ce_rank == 2
    assert result.results[1].ce_score == pytest.approx(0.8)


def test_search_ce_failure_does_not_claim_ce7_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CE requested but failing must not return a CE stage label (raises today)."""
    from app.rag.legal_v2.rerank.errors import RerankerInferenceError
    from app.rag.legal_v2.retrieve import case_similarity_search as module
    from app.rag.legal_v2.retrieve.case_similarity_search import (
        CaseSimilarityStage1Runtime,
    )
    from app.rag.legal_v2.retrieve.retriever import LegalV2RetrievalResult

    monkeypatch.setenv("NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED", "1")
    ordered = _fake_docs()

    class _FakeRetriever:
        def retrieve(self, query_spec):
            return LegalV2RetrievalResult(
                documents=ordered,  # type: ignore[arg-type]
                dense_results=[],
                bm25_results=[],
                fused_results=[],
                diagnostics={
                    "dense_latency_ms": 1.0,
                    "bm25_latency_ms": 1.0,
                    "rrf_latency_ms": 1.0,
                    "total_retrieval_latency_ms": 3.0,
                    "dense_candidate_chunks": 2,
                    "bm25_candidate_chunks": 2,
                    "fused_candidate_chunks": 2,
                    "candidate_documents": 2,
                },
            )

    class _FailingService:
        def rerank(self, query, candidates, require_success=True):
            raise RerankerInferenceError("boom")

    monkeypatch.setattr(
        "app.rag.legal_v2.rerank.service.get_cross_encoder_reranking_service",
        lambda config: _FailingService(),
    )

    runtime = CaseSimilarityStage1Runtime(
        retriever=_FakeRetriever(),  # type: ignore[arg-type]
        config=MagicMock(
            qdrant_collection="nalus_legal_paragraph_chunks_v2_chunk_ab_v8_a_current_300",
            bm25_index_id="nalus_legal_paragraph_bm25_v2_chunk_ab_v8_a_current_300",
            fused_candidate_chunks=120,
            candidate_documents=40,
        ),
        ready=True,
        ce_retriever=_FakeRetriever(),  # type: ignore[arg-type]
        ce_config=MagicMock(
            qdrant_collection="nalus_legal_paragraph_chunks_v2_chunk_ab_v8_b_contextual_300",
            bm25_index_id="nalus_legal_paragraph_bm25_v2_chunk_ab_v8_b_contextual_300",
            fused_candidate_chunks=120,
            candidate_documents=40,
        ),
    )
    with pytest.raises(ValueError, match="cross-encoder reranking failed"):
        asyncio.run(
            module.search_case_similarity_stage1(
                query="krátký testovací dotaz",
                limit=2,
                runtime=runtime,
                retrieval_profile="precise",
            )
        )
