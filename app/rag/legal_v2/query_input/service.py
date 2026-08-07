"""QueryInputService: validate → classify → normalize → optional condense."""

from __future__ import annotations

import hashlib
import time
from typing import Any

from app.core.logging import get_logger
from app.rag.legal_v2.query_input.classifier import classify_input
from app.rag.legal_v2.query_input.config import LongInputConfig, long_input_config_from_env
from app.rag.legal_v2.query_input.errors import (
    CondensationFailedError,
    InputTooLargeError,
    NoUsefulContentError,
    UnsupportedCondensationModeError,
)
from app.rag.legal_v2.query_input.models import (
    CondensationMethod,
    InputClassification,
    PreparedQuery,
    SearchBrief,
)
from app.rag.legal_v2.query_input.normalizer import normalize_legal_input
from app.rag.legal_v2.query_input.providers.base import CondensationRequest
from app.rag.legal_v2.query_input.providers.extractive import ExtractiveSearchBriefProvider
from app.rag.legal_v2.query_input.providers.precise_llm import PreciseLLMSearchBriefProvider

logger = get_logger(__name__)


def _passthrough_brief(
    *,
    original: str,
    normalized: str,
    config: LongInputConfig,
    latency_ms: float,
) -> SearchBrief:
    signature = hashlib.sha256(
        f"{normalized}\npassthrough\n{config.policy_version}".encode("utf-8")
    ).hexdigest()
    return SearchBrief(
        original_length=len(original),
        normalized_length=len(normalized),
        brief_text=normalized,
        method=CondensationMethod.PASSTHROUGH,
        was_condensed=False,
        policy_version=config.policy_version,
        brief_signature=signature,
        condensation_latency_ms=latency_ms,
    )


class QueryInputService:
    def __init__(self, config: LongInputConfig | None = None) -> None:
        self._config = config or long_input_config_from_env()
        self._extractive = ExtractiveSearchBriefProvider()
        self._precise = PreciseLLMSearchBriefProvider()

    @property
    def config(self) -> LongInputConfig:
        return self._config

    def prepare(self, raw_query: str) -> PreparedQuery:
        started = time.perf_counter()
        original = (raw_query or "").strip()
        if not original:
            raise ValueError("query must not be blank")

        # Feature OFF: preserve historical Stage 1 8k behavior.
        if not self._config.enabled:
            if len(original) > self._config.stage1_retrieval_char_limit:
                raise InputTooLargeError(
                    f"query exceeds maximum length of {self._config.stage1_retrieval_char_limit}"
                )
            cleaned = " ".join(original.split())
            latency_ms = (time.perf_counter() - started) * 1000.0
            brief = _passthrough_brief(
                original=original,
                normalized=cleaned,
                config=self._config,
                latency_ms=latency_ms,
            )
            return PreparedQuery(
                original_query=original,
                retrieval_query=cleaned,
                classification=InputClassification.SHORT_QUERY
                if len(cleaned) < self._config.char_threshold
                else InputClassification.LONG_LEGAL_INPUT,
                was_condensed=False,
                condensation_method=CondensationMethod.NONE,
                brief=brief,
                diagnostics={
                    "feature_enabled": False,
                    "preprocessing_latency_ms": latency_ms,
                },
            )

        if len(original) > self._config.raw_hard_char_limit:
            raise InputTooLargeError(
                f"query exceeds hard limit of {self._config.raw_hard_char_limit} characters"
            )

        classification = classify_input(original, self._config)
        if classification.classification == InputClassification.EMPTY:
            raise ValueError("query must not be blank")
        if classification.classification == InputClassification.OVERSIZED_INPUT:
            raise InputTooLargeError(
                f"query exceeds hard limit of {self._config.raw_hard_char_limit} characters"
            )

        normalized = normalize_legal_input(original)
        if not normalized:
            raise NoUsefulContentError("Normalized input is empty.")

        if classification.classification == InputClassification.SHORT_QUERY:
            # Keep short retrieval text close to original whitespace collapse.
            retrieval = " ".join(original.split())
            latency_ms = (time.perf_counter() - started) * 1000.0
            brief = _passthrough_brief(
                original=original,
                normalized=retrieval,
                config=self._config,
                latency_ms=latency_ms,
            )
            self._record_metrics(
                classification=classification.classification.value,
                method="passthrough",
                status="ok",
                latency_ms=latency_ms,
                original_chars=len(original),
                brief_chars=len(retrieval),
                condensed=False,
            )
            return PreparedQuery(
                original_query=original,
                retrieval_query=retrieval,
                classification=InputClassification.SHORT_QUERY,
                was_condensed=False,
                condensation_method=CondensationMethod.PASSTHROUGH,
                brief=brief,
                diagnostics={
                    "feature_enabled": True,
                    "classification_reasons": list(classification.reasons),
                    "word_count": classification.word_count,
                    "paragraph_count": classification.paragraph_count,
                    "preprocessing_latency_ms": latency_ms,
                },
            )

        # LONG_LEGAL_INPUT
        try:
            brief = self._condense(original=original, normalized=normalized)
        except UnsupportedCondensationModeError:
            raise
        except (CondensationFailedError, NoUsefulContentError) as exc:
            # Safe fallback within Stage 1 limit: normalized original truncated at sentence boundary.
            logger.warning(
                "[legal_v2.query_input] condensation failed (%s); falling back to bounded normalized text",
                type(exc).__name__,
            )
            fallback = normalized
            if len(fallback) > self._config.stage1_retrieval_char_limit:
                cut = fallback.rfind(". ", 0, self._config.stage1_retrieval_char_limit)
                fallback = fallback[: cut + 1] if cut > 200 else fallback[: self._config.stage1_retrieval_char_limit]
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._record_metrics(
                classification="long_legal_input",
                method=self._config.method,
                status="fallback",
                latency_ms=latency_ms,
                original_chars=len(original),
                brief_chars=len(fallback),
                condensed=False,
            )
            brief = _passthrough_brief(
                original=original,
                normalized=fallback,
                config=self._config,
                latency_ms=latency_ms,
            )
            return PreparedQuery(
                original_query=original,
                retrieval_query=fallback,
                classification=InputClassification.LONG_LEGAL_INPUT,
                was_condensed=False,
                condensation_method=CondensationMethod.PASSTHROUGH,
                brief=brief,
                diagnostics={
                    "feature_enabled": True,
                    "classification_reasons": list(classification.reasons),
                    "condensation_error_type": type(exc).__name__,
                    "fallback": True,
                    "preprocessing_latency_ms": latency_ms,
                },
            )

        retrieval = brief.brief_text.strip()
        if len(retrieval) > self._config.stage1_retrieval_char_limit:
            retrieval = retrieval[: self._config.stage1_retrieval_char_limit].rsplit(" ", 1)[0]

        latency_ms = (time.perf_counter() - started) * 1000.0
        self._record_metrics(
            classification="long_legal_input",
            method=brief.method.value,
            status="ok",
            latency_ms=latency_ms,
            original_chars=len(original),
            brief_chars=len(retrieval),
            condensed=True,
        )
        return PreparedQuery(
            original_query=original,
            retrieval_query=retrieval,
            classification=InputClassification.LONG_LEGAL_INPUT,
            was_condensed=True,
            condensation_method=brief.method,
            brief=brief,
            diagnostics={
                "feature_enabled": True,
                "classification_reasons": list(classification.reasons),
                "word_count": classification.word_count,
                "paragraph_count": classification.paragraph_count,
                "preprocessing_latency_ms": latency_ms,
            },
        )

    def _condense(self, *, original: str, normalized: str) -> SearchBrief:
        method = self._config.method
        request = CondensationRequest(
            raw_text=original,
            normalized_text=normalized,
            config=self._config,
        )
        if method == "extractive":
            return self._extractive.condense(request)
        if method == "precise":
            return self._precise.condense(request)
        raise UnsupportedCondensationModeError(f"Unsupported condensation method: {method}")

    def _record_metrics(
        self,
        *,
        classification: str,
        method: str,
        status: str,
        latency_ms: float,
        original_chars: int,
        brief_chars: int,
        condensed: bool,
    ) -> None:
        try:
            from app.observability import legal_v2_metrics as metrics

            metrics.record_long_input(
                classification=classification,
                method=method,
                status=status,
                latency_ms=latency_ms,
                original_chars=original_chars,
                brief_chars=brief_chars,
                condensed=condensed,
            )
        except Exception:  # noqa: BLE001
            # Observability must never break retrieval.
            return


_SERVICE: QueryInputService | None = None


def get_query_input_service(config: LongInputConfig | None = None) -> QueryInputService:
    global _SERVICE
    if config is not None:
        return QueryInputService(config=config)
    if _SERVICE is None:
        _SERVICE = QueryInputService()
    return _SERVICE


def reset_query_input_service_for_tests() -> None:
    global _SERVICE
    _SERVICE = None
