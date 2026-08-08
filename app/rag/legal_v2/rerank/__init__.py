"""Legal v2 Cross-Encoder reranking (experimental, OFF by default)."""

from app.rag.legal_v2.rerank.config import (
    CrossEncoderConfig,
    cross_encoder_config_from_env,
    cross_encoder_enabled,
)
from app.rag.legal_v2.rerank.service import (
    CrossEncoderRerankingService,
    get_cross_encoder_reranking_service,
    reset_cross_encoder_reranking_service_for_tests,
)

__all__ = [
    "CrossEncoderConfig",
    "CrossEncoderRerankingService",
    "cross_encoder_config_from_env",
    "cross_encoder_enabled",
    "get_cross_encoder_reranking_service",
    "reset_cross_encoder_reranking_service_for_tests",
]
