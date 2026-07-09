"""BGE-M3 embedder wrapper with optional Redis/in-memory embedding cache."""

from __future__ import annotations

from typing import Any

from app.rag.retrieval.bge_m3_embedder import BgeM3Embedder
from app.rag.retrieval.embedding_cache import (
    EmbeddingCache,
    EmbeddingCacheBuild,
    EmbeddingCacheConfig,
    embed_texts_with_cache,
    embedding_cache_config_from_env,
    get_query_vector_cached,
)
from app.rag.retrieval.production_profile import ProductionRetrievalConfig


class CachedBgeM3Embedder:
    def __init__(
        self,
        inner: BgeM3Embedder,
        *,
        cache: EmbeddingCache,
        cache_config: EmbeddingCacheConfig,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._cache_config = cache_config

    @property
    def loaded(self) -> bool:
        return self._inner.loaded

    def load(self) -> None:
        self._inner.load()

    def embed_query(self, query: str) -> list[float]:
        return get_query_vector_cached(
            cache=self._cache,
            config=self._cache_config,
            query=query,
            embed_fn=self._inner.embed_query,
        )

    def embed_texts(
        self,
        texts: list[str],
        *,
        content_checksums: list[str] | None = None,
        batch_size: int = 1,
    ) -> list[list[float]]:
        if not self._cache_config.enabled:
            return self._inner.embed_texts(texts)

        def encode_batch(batch: list[str]) -> list[list[float]]:
            return self._inner.embed_texts(batch)

        return embed_texts_with_cache(
            texts=texts,
            content_checksums=content_checksums,
            cache=self._cache,
            config=self._cache_config,
            encode_batch=encode_batch,
            batch_size=max(1, batch_size),
        )


def build_cached_bge_m3_embedder(
    config: ProductionRetrievalConfig,
    *,
    cache_build: EmbeddingCacheBuild | None = None,
    model: Any | None = None,
) -> tuple[CachedBgeM3Embedder, EmbeddingCacheBuild]:
    inner = BgeM3Embedder(config, model=model)
    if cache_build is None:
        cache_build = _default_cache_build(config)
    cache_config = embedding_cache_config_from_env(
        profile_name=config.profile.name,
        embedding_model=config.profile.embedding_model,
        embedding_dim=config.profile.embedding_dimension,
    )
    return (
        CachedBgeM3Embedder(inner, cache=cache_build.cache, cache_config=cache_config),
        cache_build,
    )


def _default_cache_build(config: ProductionRetrievalConfig) -> EmbeddingCacheBuild:
    from app.rag.retrieval.embedding_cache import build_embedding_cache

    return build_embedding_cache(
        profile_name=config.profile.name,
        embedding_model=config.profile.embedding_model,
        embedding_dim=config.profile.embedding_dimension,
    )
