from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from app.core.logging import get_logger
from app.rag.legal_v2.query_spec import QueryConstraint, QuerySpecV2, build_query_spec_v2
from app.rag.legal_v2.structured_output import extract_json_object
from app.rag.legal_v2.eval_budget import BudgetOperation, budget_operation_context
from app.rag.llm.config import effective_llm_config_from_env
from app.rag.llm.provider_factory import get_text_llm
from app.rag.llm.providers._base import LLMProviderError
from app.rag.llm.providers.deepseek import DeepSeekThinkingMode

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
    _MIN_MAX_TOKENS = 6000
    _THINKING_MAX_TOKENS = 8000
    _QUALITY_TIMEOUT_SECONDS = 120.0

    def __init__(
        self,
        api_key: str,
        *,
        model: str | None = None,
        thinking: DeepSeekThinkingMode = DeepSeekThinkingMode.ENABLED,
        timeout_seconds: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        config = effective_llm_config_from_env()
        self._api_key = api_key
        self.model = model or config.deepseek_model
        self.thinking = DeepSeekThinkingMode(thinking)
        # Structured-output retry is handled by interpret_query_spec_v2 (max 2 provider calls).
        self._retry_count = 0
        if timeout_seconds is not None:
            default_timeout = float(timeout_seconds)
        elif self.thinking is DeepSeekThinkingMode.ENABLED:
            default_timeout = float(
                os.getenv(
                    "NALUS_LEGAL_V2_QUERYSPEC_TIMEOUT_SECONDS",
                    str(self._QUALITY_TIMEOUT_SECONDS),
                )
            )
        else:
            default_timeout = float(config.timeout_seconds)
        self._default_timeout_seconds = default_timeout
        if max_tokens is not None:
            self.max_tokens = int(max_tokens)
        elif self.thinking is DeepSeekThinkingMode.ENABLED:
            self.max_tokens = self._THINKING_MAX_TOKENS
        else:
            self.max_tokens = max(config.legal_v2_max_tokens, self._MIN_MAX_TOKENS)
        self.last_meta = None
        self._llm = self._make_llm(timeout_seconds=self._default_timeout_seconds)

    def _make_llm(self, *, timeout_seconds: float) -> Any:
        return get_text_llm(
            "deepseek",
            self._api_key,
            model=self.model,
            timeout=timeout_seconds,
            max_tokens=self.max_tokens,
            max_retries=self._retry_count,
            raise_on_error=True,
            json_response=True,
            thinking=self.thinking,
        )

    def interpret(self, original_query: str, *, timeout_seconds: float | None = None) -> dict[str, Any] | str:
        llm = self._llm
        if timeout_seconds is not None and float(timeout_seconds) != self._default_timeout_seconds:
            llm = self._make_llm(timeout_seconds=float(timeout_seconds))
        try:
            with budget_operation_context(BudgetOperation.QUERYSPEC):
                text = llm.generate_text(_query_spec_prompt(original_query))
        finally:
            self.last_meta = getattr(llm, "last_meta", None)
        return text


def interpret_query_spec_v2(
    original_query: str,
    *,
    provider: QuerySpecProvider | None = None,
    timeout_seconds: float | None = None,
    allow_deterministic_fallback: bool = True,
    max_provider_attempts: int = 2,
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
    attempts = max(1, min(2, int(max_provider_attempts)))
    last_failure: QueryInterpretation | None = None
    for attempt in range(1, attempts + 1):
        interpretation = _interpret_once(
            original_query,
            provider=provider,
            timeout_seconds=timeout_seconds,
            started=started,
            attempt=attempt,
        )
        if interpretation.query_spec is not None:
            return interpretation
        last_failure = interpretation
        if attempt >= attempts or not _is_retryable_query_spec_failure(interpretation.reason):
            return interpretation
    assert last_failure is not None
    return last_failure


def _is_retryable_query_spec_failure(reason: str | None) -> bool:
    text = str(reason or "").lower()
    return (
        text == "query_interpreter_timeout"
        or text == "query_interpreter_invalid_json"
        or "empty_message_content" in text
        or "network_error" in text
        or text.endswith(":timeout")
    )


def _interpret_once(
    original_query: str,
    *,
    provider: QuerySpecProvider,
    timeout_seconds: float | None,
    started: float,
    attempt: int,
) -> QueryInterpretation:
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
    spec, merged_fields = _merge_deterministic_fallbacks(original_query, spec)
    if merged_fields:
        merge_note = "query_interpreter_merged:" + ",".join(merged_fields)
        repair_reason = f"{repair_reason};{merge_note}" if repair_reason else merge_note
        logger.info("legal_v2.query_spec_merged fields=%s", ",".join(merged_fields))
    validation_error = validate_query_spec_preservation(original_query, spec)
    if validation_error is not None:
        return _failed(validation_error, started, provider)
    if attempt > 1:
        repair_reason = (
            f"{repair_reason};query_interpreter_retried" if repair_reason else "query_interpreter_retried"
        )
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
    timeout = float(
        os.getenv(
            "NALUS_LEGAL_V2_QUERYSPEC_TIMEOUT_SECONDS",
            str(DeepSeekQuerySpecProvider._QUALITY_TIMEOUT_SECONDS),
        )
    )
    return DeepSeekQuerySpecProvider(
        api_key,
        thinking=DeepSeekThinkingMode.ENABLED,
        timeout_seconds=timeout,
    )


def _json_payload(raw: dict[str, Any] | str) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    return extract_json_object(raw).payload


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
        provider_error=provider_error or {},
    )


def _repair_provider_query_spec(original_query: str, payload: dict[str, Any]) -> QuerySpecV2:
    base = build_query_spec_v2(original_query)
    structured = dict(base.structured_query)
    structured["provider_schema_repaired"] = True
    if isinstance(payload.get("structured_query"), dict):
        structured.update({key: value for key, value in payload["structured_query"].items() if value is not None})
    return replace(
        base,
        structured_query=structured,
        retrieval_queries=[original_query, *[q for q in base.retrieval_queries[1:]]],
    )


def _merge_deterministic_fallbacks(
    original_query: str, spec: QuerySpecV2
) -> tuple[QuerySpecV2, list[str]]:
    """Fill fields the model dropped that the deterministic parser still found."""
    deterministic = build_query_spec_v2(original_query)
    updates: dict[str, Any] = {}
    repaired: list[str] = []

    if spec.original_query != original_query:
        updates["original_query"] = original_query
        repaired.append("original_query")

    queries = list(spec.retrieval_queries)
    if not queries or queries[0] != original_query:
        updates["retrieval_queries"] = [
            original_query,
            *[query for query in queries if query != original_query],
        ]
        repaired.append("retrieval_queries")

    if deterministic.origin and (
        not spec.origin
        or deterministic.origin.normalized_text != spec.origin.normalized_text
    ):
        updates["origin"] = deterministic.origin
        repaired.append("origin")
    if deterministic.destination and (
        not spec.destination
        or deterministic.destination.normalized_text != spec.destination.normalized_text
    ):
        updates["destination"] = deterministic.destination
        repaired.append("destination")

    if deterministic.hard_constraints:
        merged = _union_constraints(spec.hard_constraints, deterministic.hard_constraints)
        if len(merged) != len(spec.hard_constraints):
            updates["hard_constraints"] = merged
            repaired.append("hard_constraints")

    if deterministic.negations and not spec.negations:
        updates["negations"] = list(deterministic.negations)
        repaired.append("negations")

    actual_roles = {entity.role for entity in spec.entities if entity.role}
    missing = [
        entity
        for entity in deterministic.entities
        if entity.role in {"mother", "father", "child"} and entity.role not in actual_roles
    ]
    if missing:
        updates["entities"] = [*spec.entities, *missing]
        repaired.append("entity_roles")

    if not updates:
        return spec, []
    return replace(spec, **updates), repaired


def _union_constraints(
    primary: list[QueryConstraint], extra: list[QueryConstraint]
) -> list[QueryConstraint]:
    seen = {(item.category, item.attribute, item.normalized_value) for item in primary}
    result = list(primary)
    for constraint in extra:
        key = (constraint.category, constraint.attribute, constraint.normalized_value)
        if key in seen:
            continue
        seen.add(key)
        result.append(constraint)
    return result


def _query_spec_prompt(original_query: str) -> str:
    schema_hint = {
        "original_query": original_query,
        "normalized_query": "string",
        "intent": (
            "legal_document_search|legal_provision_search|case_law_search|"
            "fact_pattern_match|case_citation_search|unknown"
        ),
        "entities": [{"text": "string", "role": "mother|father|child|other|null"}],
        "locations": [{"text": "string", "normalized_text": "string"}],
        "origin": {"text": "string", "normalized_text": "string"},
        "destination": {"text": "string", "normalized_text": "string"},
        "hard_constraints": [
            {
                "category": (
                    "entity|event|relation|location|date_range|duration|legal_provision|"
                    "court|document_type|procedural_posture|decision_outcome|negation|"
                    "modality|source_of_claim|cited_case|current_case"
                ),
                "value": "string",
                "normalized_value": "string",
                "polarity": "hard",
                "constraint_id": "string",
            }
        ],
        "soft_constraints": [],
        "negative_constraints": [],
        "retrieval_queries": [original_query],
        "requires_verification": True,
        "ambiguities": [],
        "negations": [],
        "courts": [],
        "procedural_posture": [],
    }
    return (
        "Extract a structured Legal Retrieval v2 QuerySpec JSON object for the user query. "
        "Preserve actors, countries, jurisdictions, and legal concepts. "
        "Do not invent unsupported facts. Keep the original query as the first retrieval query. "
        "Use ambiguities when the query is underspecified; do not invent unsupported intents. "
        "Return JSON only.\n"
        f"Schema hint: {json.dumps(schema_hint, ensure_ascii=False)}\n"
        f"User query: {original_query}"
    )


def _estimate_tokens(original_query: str, payload: dict[str, Any]) -> dict[str, int]:
    encoded = json.dumps(payload, ensure_ascii=False)
    return {
        "prompt_chars": len(original_query),
        "completion_chars": len(encoded),
    }


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0
