from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Protocol

from app.core.logging import get_logger
from app.rag.clarification.models import CachedClarificationEntry, ClarificationDecision
from app.rag.clarification.text_utils import simplify_text

logger = get_logger(__name__)

CACHE_VERSION = "v1"
RULES_VERSION = "v1"
DEFAULT_TTL_SECONDS = 604800


class BaseClarificationCache(Protocol):
    def get(self, key: str) -> CachedClarificationEntry | None: ...

    def set(
        self,
        key: str,
        value: CachedClarificationEntry,
        *,
        ttl_seconds: int | None = None,
    ) -> None: ...

    def close(self) -> None: ...


class NullClarificationCache:
    def get(self, key: str) -> CachedClarificationEntry | None:
        del key
        return None

    def set(
        self,
        key: str,
        value: CachedClarificationEntry,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        del key, value, ttl_seconds

    def close(self) -> None:
        return None


class InMemoryClarificationCache:
    """Process-local cache for tests."""

    def __init__(self) -> None:
        self._store: dict[str, CachedClarificationEntry] = {}

    def get(self, key: str) -> CachedClarificationEntry | None:
        return self._store.get(key)

    def set(
        self,
        key: str,
        value: CachedClarificationEntry,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        del ttl_seconds
        self._store[key] = value

    def close(self) -> None:
        self._store.clear()


class RedisClarificationCache:
    """Redis payload cache for clarification decisions (not court-document vectors)."""

    def __init__(self, url: str) -> None:
        from redis import Redis

        self._client = Redis.from_url(url, decode_responses=True)

    def get(self, key: str) -> CachedClarificationEntry | None:
        raw_value = self._client.get(key)
        if not raw_value:
            return None
        try:
            payload = json.loads(raw_value)
            return CachedClarificationEntry(
                query_signature=str(payload.get("query_signature", "")),
                ambiguity_types=list(payload.get("ambiguity_types", [])),
                missing_slots=list(payload.get("missing_slots", [])),
                clarification_question_cs=str(payload.get("clarification_question_cs", "")),
                detected_issue=str(payload.get("detected_issue", "")),
                recommended_next_action=payload.get("recommended_next_action", "ask_user"),
                created_at=str(payload.get("created_at", "")),
                rules_version=str(payload.get("rules_version", RULES_VERSION)),
                reason_cs=str(payload.get("reason_cs", "")),
            )
        except (TypeError, ValueError, AttributeError) as exc:
            logger.warning("[clarification_cache] invalid payload key=%s error=%s", key, exc)
            return None

    def set(
        self,
        key: str,
        value: CachedClarificationEntry,
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


def normalize_query_for_cache(query: str) -> str:
    return simplify_text(query)


def build_exact_query_cache_key(query: str) -> str:
    normalized = normalize_query_for_cache(query)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"legal_query_clarification:{CACHE_VERSION}:{digest}"


def build_pattern_cache_key(*, query_signature: str) -> str:
    digest = hashlib.sha256(query_signature.encode("utf-8")).hexdigest()
    return f"legal_query_clarification:{CACHE_VERSION}:pattern:{digest}"


# Backward-compatible alias used by existing tests.
def build_clarification_cache_key(*, query_signature: str) -> str:
    return build_pattern_cache_key(query_signature=query_signature)


def clarification_cache_ttl_seconds() -> int | None:
    raw_value = os.getenv("LEGAL_QUERY_CLARIFICATION_CACHE_TTL_SECONDS", "").strip()
    if not raw_value:
        return DEFAULT_TTL_SECONDS
    ttl_seconds = int(raw_value)
    if ttl_seconds <= 0:
        return None
    return ttl_seconds


def build_clarification_cache() -> tuple[BaseClarificationCache, str]:
    backend = os.getenv("LEGAL_QUERY_CLARIFICATION_CACHE_BACKEND", "none").strip().lower() or "none"
    if backend == "none":
        return NullClarificationCache(), "none"
    if backend == "memory":
        return InMemoryClarificationCache(), "memory"
    if backend != "redis":
        logger.warning("[clarification_cache] unsupported backend=%s", backend)
        return NullClarificationCache(), backend

    url = os.getenv(
        "LEGAL_QUERY_CLARIFICATION_CACHE_URL",
        os.getenv("RAG_QUERY_CACHE_URL", "redis://redis:6379/0"),
    ).strip()
    try:
        return RedisClarificationCache(url), "redis"
    except Exception as exc:  # noqa: BLE001
        logger.warning("[clarification_cache] redis unavailable (%s); cache disabled", exc)
        return NullClarificationCache(), "redis"


def cache_entry_from_decision(decision: ClarificationDecision) -> CachedClarificationEntry:
    return CachedClarificationEntry(
        query_signature=decision.query_signature,
        ambiguity_types=list(decision.ambiguity_types),
        missing_slots=list(decision.missing_slots),
        clarification_question_cs=decision.clarification_question_cs,
        detected_issue=decision.ambiguity_types[0] if decision.ambiguity_types else "none",
        recommended_next_action=decision.recommended_next_action,
        created_at=datetime.now(timezone.utc).isoformat(),
        rules_version=RULES_VERSION,
        reason_cs=decision.reason_cs,
    )


def decision_from_cache_entry(
    entry: CachedClarificationEntry,
    *,
    cache_key: str,
    detected_legal_domain: str = "unknown",
    detected_procedure_stage: str = "unknown",
    semantic_cache_hit: bool = False,
) -> ClarificationDecision:
    return ClarificationDecision(
        decision="ask_clarifying_question",
        confidence=0.85,
        ambiguity_types=list(entry.ambiguity_types),
        missing_slots=list(entry.missing_slots),
        detected_legal_domain=detected_legal_domain,  # type: ignore[arg-type]
        detected_procedure_stage=detected_procedure_stage,  # type: ignore[arg-type]
        clarification_question_cs=entry.clarification_question_cs,
        reason_cs=entry.reason_cs or "Použita dříve uložená upřesňující otázka pro podobný nejednoznačný dotaz.",
        cache_key=cache_key,
        query_signature=entry.query_signature,
        recommended_next_action=entry.recommended_next_action,
        cache_hit=True,
        semantic_cache_hit=semantic_cache_hit,
        llm_called=False,
    )
