"""Redis-backed BGE-M3 embedding cache for query and chunk vectors."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from app.core.logging import get_logger
from app.rag.retrieval.errors import RetrievalConfigurationError

logger = get_logger(__name__)

SCHEMA_VERSION = 1
DEFAULT_KEY_PREFIX = "nalus:embedding"
DEFAULT_QUERY_TTL_SECONDS = 604_800
DEFAULT_CHUNK_TTL_SECONDS = 0


@dataclass(frozen=True)
class EmbeddingCacheConfig:
    enabled: bool
    redis_url: str
    key_prefix: str
    profile_name: str
    embedding_model: str
    embedding_dim: int
    query_ttl_seconds: int | None
    chunk_ttl_seconds: int | None
    fail_open_on_redis_error: bool


@dataclass(frozen=True)
class EmbeddingCacheBuild:
    cache: "EmbeddingCache"
    backend: str
    enabled: bool
    error: str | None = None


@dataclass(frozen=True)
class CachedEmbeddingPayload:
    profile_name: str
    embedding_model: str
    embedding_dim: int
    source: str
    checksum: str
    vector: list[float]
    created_at: str
    schema_version: int

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "source": self.source,
            "checksum": self.checksum,
            "vector": self.vector,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "CachedEmbeddingPayload":
        return cls(
            profile_name=str(payload.get("profile_name") or ""),
            embedding_model=str(payload.get("embedding_model") or ""),
            embedding_dim=int(payload.get("embedding_dim") or 0),
            source=str(payload.get("source") or ""),
            checksum=str(payload.get("checksum") or ""),
            vector=[float(value) for value in payload.get("vector") or []],
            created_at=str(payload.get("created_at") or ""),
            schema_version=int(payload.get("schema_version") or 0),
        )


class EmbeddingCache(Protocol):
    def get_vector(self, key: str, *, expected: EmbeddingCacheConfig) -> list[float] | None: ...

    def set_vector(
        self,
        key: str,
        *,
        vector: list[float],
        source: str,
        checksum: str,
        expected: EmbeddingCacheConfig,
        ttl_seconds: int | None,
    ) -> None: ...


class NullEmbeddingCache:
    def get_vector(self, key: str, *, expected: EmbeddingCacheConfig) -> list[float] | None:
        del key, expected
        return None

    def set_vector(
        self,
        key: str,
        *,
        vector: list[float],
        source: str,
        checksum: str,
        expected: EmbeddingCacheConfig,
        ttl_seconds: int | None,
    ) -> None:
        del key, vector, source, checksum, expected, ttl_seconds


class InMemoryEmbeddingCache:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get_vector(self, key: str, *, expected: EmbeddingCacheConfig) -> list[float] | None:
        raw_value = self._store.get(key)
        if raw_value is None:
            return None
        return _parse_cached_vector(raw_value, expected=expected, key=key)

    def set_vector(
        self,
        key: str,
        *,
        vector: list[float],
        source: str,
        checksum: str,
        expected: EmbeddingCacheConfig,
        ttl_seconds: int | None,
    ) -> None:
        del ttl_seconds
        payload = _build_payload(
            vector=vector,
            source=source,
            checksum=checksum,
            expected=expected,
        )
        self._store[key] = json.dumps(payload.to_json_dict(), ensure_ascii=False, separators=(",", ":"))


class RedisEmbeddingCache:
    def __init__(self, client: Any) -> None:
        self._client = client

    def get_vector(self, key: str, *, expected: EmbeddingCacheConfig) -> list[float] | None:
        raw_value = self._client.get(key)
        if not raw_value:
            return None
        return _parse_cached_vector(str(raw_value), expected=expected, key=key)

    def set_vector(
        self,
        key: str,
        *,
        vector: list[float],
        source: str,
        checksum: str,
        expected: EmbeddingCacheConfig,
        ttl_seconds: int | None,
    ) -> None:
        payload = _build_payload(
            vector=vector,
            source=source,
            checksum=checksum,
            expected=expected,
        )
        serialized = json.dumps(payload.to_json_dict(), ensure_ascii=False, separators=(",", ":"))
        if ttl_seconds is None:
            self._client.set(key, serialized)
            return
        self._client.set(key, serialized, ex=ttl_seconds)


def normalize_text_for_embedding_cache(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def text_checksum(text: str) -> str:
    normalized = normalize_text_for_embedding_cache(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def query_cache_key(*, config: EmbeddingCacheConfig, query_checksum: str) -> str:
    return (
        f"{config.key_prefix}:{config.profile_name}:query:{query_checksum}"
    )


def chunk_cache_key(*, config: EmbeddingCacheConfig, content_checksum: str) -> str:
    return (
        f"{config.key_prefix}:{config.profile_name}:chunk:{content_checksum}"
    )


def embedding_cache_config_from_env(
    *,
    profile_name: str,
    embedding_model: str,
    embedding_dim: int,
) -> EmbeddingCacheConfig:
    return EmbeddingCacheConfig(
        enabled=_read_bool_env("EMBEDDING_CACHE_ENABLED", default=False),
        redis_url=_resolve_redis_url(),
        key_prefix=os.getenv("EMBEDDING_CACHE_KEY_PREFIX", DEFAULT_KEY_PREFIX).strip()
        or DEFAULT_KEY_PREFIX,
        profile_name=profile_name,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        query_ttl_seconds=_read_ttl_env("QUERY_EMBEDDING_CACHE_TTL_SECONDS", DEFAULT_QUERY_TTL_SECONDS),
        chunk_ttl_seconds=_read_ttl_env("CHUNK_EMBEDDING_CACHE_TTL_SECONDS", DEFAULT_CHUNK_TTL_SECONDS),
        fail_open_on_redis_error=_read_bool_env("EMBEDDING_CACHE_FAIL_OPEN_ON_REDIS_ERROR", default=False),
    )


def build_embedding_cache(
    *,
    profile_name: str,
    embedding_model: str,
    embedding_dim: int,
) -> EmbeddingCacheBuild:
    config = embedding_cache_config_from_env(
        profile_name=profile_name,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
    )
    if not config.enabled:
        return EmbeddingCacheBuild(
            cache=NullEmbeddingCache(),
            backend="none",
            enabled=False,
        )

    try:
        from redis import Redis

        client = Redis.from_url(config.redis_url, decode_responses=True)
        client.ping()
        return EmbeddingCacheBuild(
            cache=RedisEmbeddingCache(client),
            backend="redis",
            enabled=True,
        )
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        if config.fail_open_on_redis_error:
            logger.warning(
                "[embedding_cache] redis unavailable (%s); cache disabled (fail-open)",
                error,
            )
            return EmbeddingCacheBuild(
                cache=NullEmbeddingCache(),
                backend="redis",
                enabled=False,
                error=error,
            )
        raise RetrievalConfigurationError(
            f"EMBEDDING_CACHE_ENABLED=1 but Redis is unavailable: {error}"
        ) from exc


def get_query_vector_cached(
    *,
    cache: EmbeddingCache,
    config: EmbeddingCacheConfig,
    query: str,
    embed_fn: Any,
) -> list[float]:
    if not config.enabled:
        return embed_fn(query)

    normalized = normalize_text_for_embedding_cache(query)
    checksum = text_checksum(normalized)
    key = query_cache_key(config=config, query_checksum=checksum)
    cached = cache.get_vector(key, expected=config)
    if cached is not None:
        return cached

    vector = embed_fn(normalized)
    _validate_vector_dim(vector, expected_dim=config.embedding_dim, label="query")
    cache.set_vector(
        key,
        vector=vector,
        source="query",
        checksum=checksum,
        expected=config,
        ttl_seconds=config.query_ttl_seconds,
    )
    return vector


def embed_texts_with_cache(
    *,
    texts: list[str],
    content_checksums: list[str] | None,
    cache: EmbeddingCache,
    config: EmbeddingCacheConfig,
    encode_batch: Any,
    batch_size: int,
) -> list[list[float]]:
    if not texts:
        return []
    if not config.enabled:
        return _encode_in_batches(texts, encode_batch=encode_batch, batch_size=batch_size)

    if content_checksums is not None and len(content_checksums) != len(texts):
        raise RetrievalConfigurationError(
            "content_checksums length must match texts length for chunk embedding cache."
        )

    vectors: list[list[float] | None] = [None] * len(texts)
    miss_indices: list[int] = []
    miss_texts: list[str] = []

    for index, text in enumerate(texts):
        if content_checksums is not None:
            checksum = str(content_checksums[index]).strip() or text_checksum(text)
        else:
            checksum = text_checksum(text)
        if not checksum:
            raise RetrievalConfigurationError("Chunk content checksum must not be empty.")
        key = chunk_cache_key(config=config, content_checksum=checksum)
        cached = cache.get_vector(key, expected=config)
        if cached is not None:
            vectors[index] = cached
            continue
        miss_indices.append(index)
        miss_texts.append(text)

    if miss_texts:
        encoded = _encode_in_batches(miss_texts, encode_batch=encode_batch, batch_size=batch_size)
        for index, vector in zip(miss_indices, encoded, strict=True):
            _validate_vector_dim(vector, expected_dim=config.embedding_dim, label="chunk")
            vectors[index] = vector
            checksum = (
                str(content_checksums[index]).strip() or text_checksum(texts[index])
                if content_checksums is not None
                else text_checksum(texts[index])
            )
            key = chunk_cache_key(config=config, content_checksum=checksum)
            cache.set_vector(
                key,
                vector=vector,
                source="chunk",
                checksum=checksum,
                expected=config,
                ttl_seconds=config.chunk_ttl_seconds,
            )

    return [vector for vector in vectors if vector is not None]


def _encode_in_batches(texts: list[str], *, encode_batch: Any, batch_size: int) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(encode_batch(batch))
    return vectors


def _build_payload(
    *,
    vector: list[float],
    source: str,
    checksum: str,
    expected: EmbeddingCacheConfig,
) -> CachedEmbeddingPayload:
    _validate_vector_dim(vector, expected_dim=expected.embedding_dim, label=source)
    return CachedEmbeddingPayload(
        profile_name=expected.profile_name,
        embedding_model=expected.embedding_model,
        embedding_dim=expected.embedding_dim,
        source=source,
        checksum=checksum,
        vector=vector,
        created_at=datetime.now(timezone.utc).isoformat(),
        schema_version=SCHEMA_VERSION,
    )


def _parse_cached_vector(
    raw_value: str,
    *,
    expected: EmbeddingCacheConfig,
    key: str,
) -> list[float]:
    try:
        payload = CachedEmbeddingPayload.from_json_dict(json.loads(raw_value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RetrievalConfigurationError(
            f"Corrupted embedding cache payload at key {key!r}: {exc}"
        ) from exc

    if payload.schema_version != SCHEMA_VERSION:
        raise RetrievalConfigurationError(
            f"Unsupported embedding cache schema at key {key!r}: {payload.schema_version}"
        )
    if payload.profile_name != expected.profile_name:
        raise RetrievalConfigurationError(
            f"Embedding cache profile mismatch at key {key!r}: "
            f"{payload.profile_name!r} != {expected.profile_name!r}"
        )
    if payload.embedding_model != expected.embedding_model:
        raise RetrievalConfigurationError(
            f"Embedding cache model mismatch at key {key!r}: "
            f"{payload.embedding_model!r} != {expected.embedding_model!r}"
        )
    if payload.embedding_dim != expected.embedding_dim:
        raise RetrievalConfigurationError(
            f"Embedding cache dimension mismatch at key {key!r}: "
            f"{payload.embedding_dim} != {expected.embedding_dim}"
        )
    _validate_vector_dim(payload.vector, expected_dim=expected.embedding_dim, label=key)
    return payload.vector


def _validate_vector_dim(vector: list[float], *, expected_dim: int, label: str) -> None:
    if len(vector) != expected_dim:
        raise RetrievalConfigurationError(
            f"Embedding vector dimension mismatch for {label}: {len(vector)} != {expected_dim}"
        )


def _resolve_redis_url() -> str:
    explicit = os.getenv("REDIS_URL", "").strip()
    if explicit:
        return explicit
    return os.getenv("RAG_QUERY_CACHE_URL", "redis://redis:6379/0").strip()


def _read_bool_env(name: str, *, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RetrievalConfigurationError(f"{name} must be a boolean value.")


def _read_ttl_env(name: str, default: int) -> int | None:
    raw_value = os.getenv(name)
    if raw_value is None:
        raw_value = str(default)
    ttl_seconds = int(raw_value.strip())
    if ttl_seconds <= 0:
        return None
    return ttl_seconds
