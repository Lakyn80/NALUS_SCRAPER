"""Unit tests for Stage 1 request-level retrieval profiles."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.rag.legal_v2.retrieve.retrieval_profiles import (
    resolve_retrieval_profile,
)
from app.rag.legal_v2.rerank.selectors.names import DIVERSIFIED_STAGE1_EVIDENCE_V1


def test_fast_profile_disables_ce(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED", "1")
    resolved = resolve_retrieval_profile("fast")
    assert resolved.profile_id == "fast"
    assert resolved.use_cross_encoder is False
    assert resolved.cross_encoder_config is None


def test_ce7_requires_master_allow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED", "0")
    with pytest.raises(ValueError, match="CROSS_ENCODER_ENABLED"):
        resolve_retrieval_profile("ce7")


def test_ce7_profile_forces_diversified_seven(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED", "1")
    monkeypatch.setenv("NALUS_LEGAL_V2_CE_PASSAGES_PER_DOCUMENT", "3")
    monkeypatch.setenv(
        "NALUS_LEGAL_V2_CE_PASSAGE_SELECTOR", "first_n_stage1_order_v1"
    )
    resolved = resolve_retrieval_profile("ce7")
    assert resolved.use_cross_encoder is True
    assert resolved.cross_encoder_config is not None
    assert resolved.cross_encoder_config.passages_per_document == 7
    assert (
        resolved.cross_encoder_config.passage_selector
        == DIVERSIFIED_STAGE1_EVIDENCE_V1
    )
    assert resolved.cross_encoder_config.evidence_pool_limit >= 40


def test_precise_reserved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED", "1")
    with pytest.raises(ValueError, match="not available yet"):
        resolve_retrieval_profile("precise")


def test_unknown_profile_rejected() -> None:
    with pytest.raises(ValueError, match="unknown retrieval_profile"):
        resolve_retrieval_profile("balanced")


def test_search_default_profile_is_fast_even_when_ce_env_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.rag.legal_v2.retrieve import case_similarity_search as module
    from app.rag.legal_v2.retrieve.case_similarity_search import (
        CaseSimilarityStage1Runtime,
    )
    from app.rag.legal_v2.retrieve.retriever import LegalV2RetrievalResult

    monkeypatch.setenv("NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED", "1")

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

    ordered = [
        _Doc("ECLI:CZ:US:2025:1.US.1111.25.1", 0.9),
        _Doc("ECLI:CZ:US:2025:1.US.2222.25.1", 0.8),
    ]

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
            qdrant_collection="nalus_legal_paragraph_chunks_v2_pilot_600",
            bm25_index_id="nalus_legal_paragraph_bm25_v2_pilot_600",
        ),
        ready=True,
    )
    result = module.search_case_similarity_stage1(
        query="krátký testovací dotaz",
        limit=2,
        runtime=runtime,
        # default / explicit fast
        retrieval_profile="fast",
    )
    assert result.diagnostics["retrieval_profile"] == "fast"
    assert result.diagnostics["rerank"]["rerank_applied"] is False
    assert [row.ecli for row in result.results] == [
        "ECLI:CZ:US:2025:1.US.1111.25.1",
        "ECLI:CZ:US:2025:1.US.2222.25.1",
    ]
