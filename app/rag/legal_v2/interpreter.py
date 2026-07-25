from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from app.core.logging import get_logger
from app.rag.legal_v2.query_spec import QuerySpecV2, build_query_spec_v2
from app.rag.llm.config import effective_llm_config_from_env
from app.rag.llm.provider_factory import get_text_llm
from app.rag.llm.providers._base import LLMProviderError

logger = get_logger(__name__)


@dataclass(frozen=True)
class QueryInterpretation:
    status: str
    query_spec: QuerySpecV2 | None
    reason: str | None = None
    provider_name: str = "unknown"
    model: str | None = None
    latency_ms: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)
    estimated_cost: float = 0.0
    provider_error: dict[str, Any] = field(default_factory=dict)


class QuerySpecProvider(Protocol):
    provider_name: str

    def interpret(self, original_query: str, *, timeout_seconds: float | None = None) -> dict[str, Any] | str:
        ...


class DeterministicQuerySpecProvider:
    provider_name = "deterministic_query_spec_provider"

    def __init__(self, payload: dict[str, Any] | str | None = None, *, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error
        self.calls = 0

    def interpret(self, original_query: str, *, timeout_seconds: float | None = None) -> dict[str, Any] | str:
        del timeout_seconds
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.payload is not None:
            return self.payload
        return build_query_spec_v2(original_query).to_dict()


class DeepSeekQuerySpecProvider:
    provider_name = "deepseek_query_spec_v2"

    def __init__(self, api_key: str, *, model: str | None = None) -> None:
        config = effective_llm_config_from_env()
        self.model = model or config.deepseek_model
        self._llm = get_text_llm(
            "deepseek",
            api_key,
            model=self.model,
            timeout=config.timeout_seconds,
            max_tokens=config.legal_v2_max_tokens,
            max_retries=config.retry_count,
            raise_on_error=True,
            json_response=True,
        )

    def interpret(self, original_query: str, *, timeout_seconds: float | None = None) -> dict[str, Any] | str:
        del timeout_seconds
        return self._llm.generate_text(_query_spec_prompt(original_query))


def interpret_query_spec_v2(
    original_query: str,
    *,
    provider: QuerySpecProvider | None = None,
    timeout_seconds: float | None = None,
    allow_deterministic_fallback: bool = True,
) -> QueryInterpretation:
    started = time.perf_counter()
    provider = provider or _provider_from_env()
    if provider is None:
        if not allow_deterministic_fallback:
            return _failed("provider_not_configured", started)
        spec = build_query_spec_v2(original_query)
        return QueryInterpretation(
            status=_status_for_spec(spec),
            query_spec=spec,
            reason="deterministic_fallback",
            provider_name="deterministic_fallback",
            latency_ms=_elapsed_ms(started),
        )
    try:
        raw = provider.interpret(original_query, timeout_seconds=timeout_seconds)
    except TimeoutError:
        return _failed("query_interpreter_timeout", started, provider)
    except LLMProviderError as exc:
        return _failed(
            f"query_interpreter_provider_error:{exc.safe_reason}",
            started,
            provider,
            provider_error=exc.to_safe_dict(),
        )
    except Exception as exc:  # noqa: BLE001
        return _failed(f"query_interpreter_provider_error:{exc.__class__.__name__}", started, provider)
    payload = _json_payload(raw)
    if payload is None:
        return _failed("query_interpreter_invalid_json", started, provider)
    repair_reason: str | None = None
    try:
        spec = QuerySpecV2.from_dict(payload)
    except Exception as exc:  # noqa: BLE001
        spec = _repair_provider_query_spec(original_query, payload)
        repair_reason = f"query_interpreter_schema_repaired:{exc.__class__.__name__}"
    validation_error = validate_query_spec_preservation(original_query, spec)
    if validation_error is not None:
        return _failed(validation_error, started, provider)
    return QueryInterpretation(
        status=_status_for_spec(spec),
        query_spec=spec,
        reason=repair_reason,
        provider_name=getattr(provider, "provider_name", provider.__class__.__name__),
        model=getattr(provider, "model", None),
        latency_ms=_elapsed_ms(started),
        token_usage=_estimate_tokens(original_query, payload),
    )


def validate_query_spec_preservation(original_query: str, spec: QuerySpecV2) -> str | None:
    deterministic = build_query_spec_v2(original_query)
    if spec.original_query != original_query:
        return "original_query_changed"
    if not spec.retrieval_queries or spec.retrieval_queries[0] != original_query:
        return "original_query_not_first_retrieval_query"
    expected_roles = {entity.role for entity in deterministic.entities if entity.role}
    actual_roles = {entity.role for entity in spec.entities if entity.role}
    if "mother" in expected_roles and "mother" not in actual_roles:
        return "explicit_mother_role_lost"
    if "father" in expected_roles and "father" not in actual_roles:
        return "explicit_father_role_lost"
    if deterministic.origin and not spec.origin:
        return "origin_lost"
    if deterministic.destination and not spec.destination:
        return "destination_lost"
    if deterministic.origin and spec.origin and deterministic.origin.normalized_text != spec.origin.normalized_text:
        return "origin_changed"
    if deterministic.destination and spec.destination and deterministic.destination.normalized_text != spec.destination.normalized_text:
        return "destination_changed"
    if deterministic.negations and not spec.negations:
        return "negation_lost"
    if deterministic.hard_constraints and not spec.hard_constraints:
        return "hard_constraints_lost"
    return None


def _provider_from_env() -> QuerySpecProvider | None:
    if os.getenv("NALUS_LEGAL_V2_QUERY_PROVIDER", "deepseek").lower() != "deepseek":
        return None
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key or api_key == "your-api-key-here":
        return None
    return DeepSeekQuerySpecProvider(api_key)


def _json_payload(raw: dict[str, Any] | str) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _status_for_spec(spec: QuerySpecV2) -> str:
    if spec.requires_verification and not spec.hard_constraints:
        return "unverifiable_query"
    return "ok"


def _failed(
    reason: str,
    started: float,
    provider: QuerySpecProvider | None = None,
    *,
    provider_error: dict[str, Any] | None = None,
) -> QueryInterpretation:
    return QueryInterpretation(
        status="failed",
        query_spec=None,
        reason=reason,
        provider_name=getattr(provider, "provider_name", "unknown"),
        model=getattr(provider, "model", None),
        latency_ms=_elapsed_ms(started),
        token_usage={},
        estimated_cost=0.0,
        provider_error=dict(provider_error or {}),
    )


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _estimate_tokens(original_query: str, payload: dict[str, Any]) -> dict[str, int]:
    output = json.dumps(payload, ensure_ascii=False)
    return {
        "input_tokens_estimated": max(1, len(original_query) // 4),
        "output_tokens_estimated": max(1, len(output) // 4),
    }


def _repair_provider_query_spec(
    original_query: str,
    payload: dict[str, Any],
) -> QuerySpecV2:
    deterministic = build_query_spec_v2(original_query)
    provider_queries = [
        str(item).strip()
        for item in payload.get("retrieval_queries") or []
        if str(item).strip()
    ]
    retrieval_queries = _dedupe_preserve_order(
        [original_query, *deterministic.retrieval_queries, *provider_queries]
    )[:8]
    structured_query = {
        **deterministic.structured_query,
        "provider_schema_repaired": True,
    }
    return replace(
        deterministic,
        retrieval_queries=retrieval_queries,
        structured_query=structured_query,
    )


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _query_spec_prompt(original_query: str) -> str:
    return (
        "Return strict JSON for schema legal_query_spec_v2. Preserve every explicit hard fact. "
        "Do not broaden precise queries. The JSON must include original_query, normalized_query, "
        "structured_query, retrieval_queries, intent, entities, events, relations, locations, "
        "origin, destination, movement_direction, date_ranges, durations, legal_provisions, "
        "courts, document_types, procedural_posture, decision_outcome, negations, modalities, "
        "source_of_claims, cited_cases, current_case_identifiers, hard_constraints, "
        "soft_constraints, negative_constraints, ambiguities, requires_verification. "
        f"Original query: {original_query}"
    )
