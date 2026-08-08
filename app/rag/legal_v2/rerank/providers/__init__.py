"""Legal v2 reranker providers."""

from app.rag.legal_v2.rerank.providers.cross_encoder import SentenceTransformersCrossEncoderProvider

__all__ = ["SentenceTransformersCrossEncoderProvider"]
