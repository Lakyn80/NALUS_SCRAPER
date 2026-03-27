from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Protocol

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CachedQueryResponse:
    answer: str
    sources: list[str]
    plan_steps: list[str]


@dataclass(frozen=True)
class QueryCacheBuild:
    cache: "BaseQueryCache"
    backend: str
    error: str | None = None


class BaseQueryCache(Protocol):
    def get(self, key: str) -> CachedQueryResponse | None: ...

    def set(
        self,
        key: str,
        value: CachedQueryResponse,
        *,
        ttl_seconds: int | None = None,
    ) -> None: ...

    def close(self) -> None: ...


class NullQueryCache:
    def get(self, key: str) -> CachedQueryResponse | None:
        del key
        return None

    def set(
        self,
        key: str,
        value: CachedQueryResponse,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        del key, value, ttl_seconds

    def close(self) -> None:
        return None


class RedisQueryCache:
    def __init__(self, url: str) -> None:
        from redis import Redis

        self._client = Redis.from_url(url, decode_responses=True)

    def get(self, key: str) -> CachedQueryResponse | None:
        raw_value = self._client.get(key)
        if not raw_value:
            return None
        try:
            payload = json.loads(raw_value)
            return CachedQueryResponse(
                answer=str(payload.get("answer", "")),
                sources=[str(item) for item in payload.get("sources", [])],
                plan_steps=[str(item) for item in payload.get("plan_steps", [])],
            )
        except (TypeError, ValueError, AttributeError) as exc:
            logger.warning("[query_cache] invalid cached payload key=%s error=%s", key, exc)
            return None

    def set(
        self,
        key: str,
        value: CachedQueryResponse,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        payload = json.dumps(asdict(value), ensure_ascii=False, separators=(",", ":"))
        if ttl_seconds is None:
            self._client.set(key, payload)
            return
        self._client.set(key, payload, ex=ttl_seconds)

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


def build_query_cache() -> QueryCacheBuild:
    backend = os.getenv("RAG_QUERY_CACHE_BACKEND", "none").strip().lower() or "none"
    if backend == "none":
        return QueryCacheBuild(cache=NullQueryCache(), backend="none")

    if backend != "redis":
        error = f"Unsupported RAG_QUERY_CACHE_BACKEND={backend!r}"
        logger.warning("[query_cache] %s", error)
        return QueryCacheBuild(cache=NullQueryCache(), backend=backend, error=error)

    url = os.getenv("RAG_QUERY_CACHE_URL", "redis://redis:6379/0").strip()
    try:
        cache = RedisQueryCache(url)
        return QueryCacheBuild(cache=cache, backend="redis")
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        logger.warning("[query_cache] redis unavailable (%s); cache disabled", error)
        return QueryCacheBuild(cache=NullQueryCache(), backend="redis", error=error)


def query_cache_ttl_seconds() -> int | None:
    raw_value = os.getenv("RAG_QUERY_CACHE_TTL_SECONDS", "").strip()
    if not raw_value:
        return 604800
    ttl_seconds = int(raw_value)
    if ttl_seconds <= 0:
        return None
    return ttl_seconds


def build_cache_key(
    query: str,
    *,
    corpus_version: str,
) -> str:
    payload = {
        "query": " ".join(query.split()).strip().lower(),
        "corpus_version": corpus_version,
        "provider": os.getenv("LLM_PROVIDER", "").strip().lower(),
        "model": _configured_llm_model(),
        "prompt_version": os.getenv("RAG_QUERY_CACHE_PROMPT_VERSION", "v1").strip() or "v1",
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "rag-query:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _configured_llm_model() -> str:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider == "deepseek":
        return os.getenv("LLM_MODEL_DEEPSEEK", "").strip()
    if provider == "openai":
        return os.getenv("LLM_MODEL_OPENAI", "").strip()
    if provider == "claude":
        return os.getenv("LLM_MODEL_CLAUDE", "").strip()
    return ""
