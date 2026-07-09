from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.rag.retrieval.cached_bge_m3_embedder import CachedBgeM3Embedder
from app.rag.retrieval.embedding_cache import (
    EmbeddingCacheBuild,
    EmbeddingCacheConfig,
    InMemoryEmbeddingCache,
    NullEmbeddingCache,
    RedisEmbeddingCache,
    build_embedding_cache,
    chunk_cache_key,
    embed_texts_with_cache,
    get_query_vector_cached,
    normalize_text_for_embedding_cache,
    query_cache_key,
    text_checksum,
)
from app.rag.retrieval.errors import RetrievalConfigurationError
from app.rag.retrieval.production_profile import BGE_M3_DENSE_BM25_RRF, ProductionRetrievalConfig
from app.rag.retrieval.bge_m3_embedder import BgeM3Embedder


def _cache_config(*, enabled: bool = True) -> EmbeddingCacheConfig:
    return EmbeddingCacheConfig(
        enabled=enabled,
        redis_url="redis://fake:6379/1",
        key_prefix="nalus:embedding",
        profile_name=BGE_M3_DENSE_BM25_RRF.name,
        embedding_model=BGE_M3_DENSE_BM25_RRF.embedding_model,
        embedding_dim=BGE_M3_DENSE_BM25_RRF.embedding_dimension,
        query_ttl_seconds=3600,
        chunk_ttl_seconds=None,
        fail_open_on_redis_error=False,
    )


def test_normalize_query_text_collapses_whitespace() -> None:
    assert normalize_text_for_embedding_cache("  právo   na   proces  ") == "právo na proces"


def test_query_cache_key_is_deterministic() -> None:
    config = _cache_config()
    checksum = text_checksum("právo na proces")
    assert query_cache_key(config=config, query_checksum=checksum) == (
        f"nalus:embedding:{config.profile_name}:query:{checksum}"
    )


def test_chunk_cache_key_uses_content_checksum() -> None:
    config = _cache_config()
    checksum = "abc123"
    assert chunk_cache_key(config=config, content_checksum=checksum) == (
        f"nalus:embedding:{config.profile_name}:chunk:{checksum}"
    )


def test_wrong_vector_dimension_rejected_on_set() -> None:
    cache = InMemoryEmbeddingCache()
    config = _cache_config()
    with pytest.raises(RetrievalConfigurationError, match="dimension mismatch"):
        cache.set_vector(
            "key",
            vector=[0.1] * 768,
            source="query",
            checksum="abc",
            expected=config,
            ttl_seconds=60,
        )


def test_wrong_vector_dimension_rejected_on_get() -> None:
    cache = InMemoryEmbeddingCache()
    config = _cache_config()
    payload = {
        "profile_name": config.profile_name,
        "embedding_model": config.embedding_model,
        "embedding_dim": config.embedding_dim,
        "source": "query",
        "checksum": "abc",
        "vector": [0.1] * 768,
        "created_at": "2026-01-01T00:00:00+00:00",
        "schema_version": 1,
    }
    cache._store["bad"] = json.dumps(payload)
    with pytest.raises(RetrievalConfigurationError, match="dimension mismatch"):
        cache.get_vector("bad", expected=config)


def test_corrupted_payload_raises_clear_error() -> None:
    cache = InMemoryEmbeddingCache()
    config = _cache_config()
    cache._store["bad"] = "{not-json"
    with pytest.raises(RetrievalConfigurationError, match="Corrupted embedding cache payload"):
        cache.get_vector("bad", expected=config)


def test_redis_embedding_cache_uses_fake_client() -> None:
    client = MagicMock()
    client.get.return_value = None
    cache = RedisEmbeddingCache(client)
    config = _cache_config()
    assert cache.get_vector("missing", expected=config) is None
    cache.set_vector(
        "key",
        vector=[0.0] * 1024,
        source="query",
        checksum="abc",
        expected=config,
        ttl_seconds=60,
    )
    client.set.assert_called_once()


def test_cache_disabled_preserves_embedder_behavior(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.setenv("EMBEDDING_CACHE_ENABLED", "0")
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    config = ProductionRetrievalConfig(
        profile=BGE_M3_DENSE_BM25_RRF,
        qdrant_collection="test",
        bm25_sidecar_path=tmp_path / "bm25.sqlite",
        bm25_index_id=BGE_M3_DENSE_BM25_RRF.name,
        model_path=str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        device="cpu",
    )
    model = MagicMock()
    model.encode.return_value = [[0.5] * 1024]
    embedder = CachedBgeM3Embedder(
        BgeM3Embedder(config, model=model),
        cache=NullEmbeddingCache(),
        cache_config=_cache_config(enabled=False),
    )
    vector = embedder.embed_query("test query")
    assert len(vector) == 1024
    model.encode.assert_called_once()


def test_query_cache_hit_skips_embedder() -> None:
    cache = InMemoryEmbeddingCache()
    config = _cache_config()
    query = "právo na spravedlivý proces"
    checksum = text_checksum(normalize_text_for_embedding_cache(query))
    key = query_cache_key(config=config, query_checksum=checksum)
    cache.set_vector(
        key,
        vector=[0.25] * 1024,
        source="query",
        checksum=checksum,
        expected=config,
        ttl_seconds=60,
    )
    embed_fn = MagicMock()
    vector = get_query_vector_cached(
        cache=cache,
        config=config,
        query=query,
        embed_fn=embed_fn,
    )
    assert vector == [0.25] * 1024
    embed_fn.assert_not_called()


def test_query_cache_miss_calls_embedder_once_and_stores() -> None:
    cache = InMemoryEmbeddingCache()
    config = _cache_config()
    embed_fn = MagicMock(return_value=[0.75] * 1024)
    vector = get_query_vector_cached(
        cache=cache,
        config=config,
        query="dotaz",
        embed_fn=embed_fn,
    )
    assert vector == [0.75] * 1024
    embed_fn.assert_called_once()
    second = get_query_vector_cached(
        cache=cache,
        config=config,
        query="dotaz",
        embed_fn=embed_fn,
    )
    assert second == [0.75] * 1024
    embed_fn.assert_called_once()


def test_chunk_cache_hit_avoids_reencoding() -> None:
    cache = InMemoryEmbeddingCache()
    config = _cache_config()
    checksum = "chunk-checksum-1"
    key = chunk_cache_key(config=config, content_checksum=checksum)
    cache.set_vector(
        key,
        vector=[0.4] * 1024,
        source="chunk",
        checksum=checksum,
        expected=config,
        ttl_seconds=None,
    )
    encode_batch = MagicMock()
    vectors = embed_texts_with_cache(
        texts=["chunk text"],
        content_checksums=[checksum],
        cache=cache,
        config=config,
        encode_batch=encode_batch,
        batch_size=8,
    )
    assert vectors == [[0.4] * 1024]
    encode_batch.assert_not_called()


def test_build_embedding_cache_disabled_without_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EMBEDDING_CACHE_ENABLED", raising=False)
    build = build_embedding_cache(
        profile_name=BGE_M3_DENSE_BM25_RRF.name,
        embedding_model=BGE_M3_DENSE_BM25_RRF.embedding_model,
        embedding_dim=BGE_M3_DENSE_BM25_RRF.embedding_dimension,
    )
    assert build.enabled is False
    assert build.backend == "none"
    assert isinstance(build.cache, NullEmbeddingCache)


def test_build_embedding_cache_enabled_fails_without_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_CACHE_ENABLED", "1")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:9")
    with pytest.raises(RetrievalConfigurationError, match="Redis is unavailable"):
        build_embedding_cache(
            profile_name=BGE_M3_DENSE_BM25_RRF.name,
            embedding_model=BGE_M3_DENSE_BM25_RRF.embedding_model,
            embedding_dim=BGE_M3_DENSE_BM25_RRF.embedding_dimension,
        )


def test_payload_includes_profile_model_and_dim() -> None:
    cache = InMemoryEmbeddingCache()
    config = _cache_config()
    cache.set_vector(
        "key",
        vector=[0.1] * 1024,
        source="chunk",
        checksum="abc",
        expected=config,
        ttl_seconds=None,
    )
    raw = json.loads(cache._store["key"])
    assert raw["profile_name"] == config.profile_name
    assert raw["embedding_model"] == config.embedding_model
    assert raw["embedding_dim"] == 1024
    assert raw["schema_version"] == 1
