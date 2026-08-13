"""CE service singleton must not allocate a new GPU model per request."""

from __future__ import annotations

from app.rag.legal_v2.rerank.config import CrossEncoderConfig
from app.rag.legal_v2.rerank.service import (
    get_cross_encoder_reranking_service,
    reset_cross_encoder_reranking_service_for_tests,
)
from app.rag.legal_v2.rerank.selectors.names import DIVERSIFIED_STAGE1_EVIDENCE_V1


def _cfg(**overrides) -> CrossEncoderConfig:
    base = dict(
        enabled=True,
        model_id="BAAI/bge-reranker-v2-m3",
        candidate_documents=30,
        passages_per_document=7,
        batch_size=8,
        device="cpu",
        max_length=512,
        allow_download=False,
        local_files_only=True,
        aggregation="max",
        passage_selector=DIVERSIFIED_STAGE1_EVIDENCE_V1,
        evidence_pool_limit=40,
    )
    base.update(overrides)
    return CrossEncoderConfig(**base)


def test_ce_service_singleton_reused_for_equivalent_config() -> None:
    reset_cross_encoder_reranking_service_for_tests()
    a = get_cross_encoder_reranking_service(_cfg())
    b = get_cross_encoder_reranking_service(_cfg())
    assert a is b
    reset_cross_encoder_reranking_service_for_tests()


def test_ce_service_replaced_when_knobs_change() -> None:
    reset_cross_encoder_reranking_service_for_tests()
    a = get_cross_encoder_reranking_service(_cfg(passages_per_document=7))
    b = get_cross_encoder_reranking_service(_cfg(passages_per_document=3))
    assert a is not b
    reset_cross_encoder_reranking_service_for_tests()
