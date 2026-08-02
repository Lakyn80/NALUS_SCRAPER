"""
DeepSeek LLM adapters.

DeepSeekLLM    — BaseLLM    — structured RAG answering
DeepSeekTextLLM — BaseTextLLM — plain text generation (rewrite / planner / synthesis)

Endpoint: https://api.deepseek.com/chat/completions
Auth:     Authorization: Bearer <api_key>
"""

from __future__ import annotations

import os
from enum import Enum

from app.core.logging import get_logger
from app.rag.llm.base import BaseLLM
from app.rag.llm.models import LLMInput, LLMOutput
from app.rag.llm.config import DEEPSEEK_DEFAULT_MODEL
from app.rag.llm.deepseek_eval_budget import (
    BudgetExhaustedError,
    get_budget_tracker,
    reserve_for_prompt,
)
from app.rag.llm.deepseek_pricing import parse_deepseek_usage
from app.rag.llm.providers._base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_RETRY,
    DEFAULT_TIMEOUT,
    HTTPClient,
    LLMProviderError,
    LLMResponseStructureError,
    build_rag_messages,
    build_text_messages,
    empty_output,
    parse_rag_response,
)
from app.rag.rewrite.query_rewrite_service import BaseTextLLM

logger = get_logger(__name__)

_ENDPOINT = "https://api.deepseek.com/chat/completions"
_DEFAULT_MODEL = os.getenv("LLM_MODEL_DEEPSEEK", DEEPSEEK_DEFAULT_MODEL)


class DeepSeekThinkingMode(str, Enum):
    """Per-request DeepSeek v4 thinking-mode control for direct HTTP calls."""

    PROVIDER_DEFAULT = "provider_default"
    ENABLED = "enabled"
    DISABLED = "disabled"


# ---------------------------------------------------------------------------
# Shared header builder
# ---------------------------------------------------------------------------


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


# ---------------------------------------------------------------------------
# DeepSeekLLM — BaseLLM
# ---------------------------------------------------------------------------


class DeepSeekLLM(BaseLLM):
    """DeepSeek adapter for structured RAG answering (BaseLLM contract)."""

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = DEFAULT_RETRY,
        raise_on_error: bool = False,
        json_response: bool = False,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._raise_on_error = raise_on_error
        self._json_response = json_response
        self._http = HTTPClient(
            provider="deepseek",
            headers=_headers(api_key),
            timeout=timeout,
            max_retries=max_retries,
            raise_on_error=raise_on_error,
        )

    def generate(self, data: LLMInput) -> LLMOutput:
        messages = build_rag_messages(data.query, data.chunks)
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": 0.0,
        }
        chunk_ids = [c.id for c in data.chunks[:5]]

        resp = self._http.post(_ENDPOINT, payload)
        if resp is None:
            return empty_output()

        try:
            body = resp.json()
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError, TypeError):
            error = LLMResponseStructureError(
                provider="deepseek",
                model=self._model,
                operation="rag_generate",
            )
            logger.warning("[llm] provider=deepseek error=%s", error.safe_reason)
            if self._raise_on_error:
                raise error
            return empty_output()

        return parse_rag_response(text, chunk_ids)


# ---------------------------------------------------------------------------
# DeepSeekTextLLM — BaseTextLLM
# ---------------------------------------------------------------------------


class DeepSeekTextLLM(BaseTextLLM):
    """DeepSeek adapter for plain text generation (BaseTextLLM contract)."""

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = DEFAULT_RETRY,
        raise_on_error: bool = False,
        json_response: bool = False,
        thinking: DeepSeekThinkingMode = DeepSeekThinkingMode.PROVIDER_DEFAULT,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._raise_on_error = raise_on_error
        self._json_response = json_response
        self._thinking = DeepSeekThinkingMode(thinking)
        self.last_meta: dict | None = None
        self._http = HTTPClient(
            provider="deepseek",
            headers=_headers(api_key),
            timeout=timeout,
            max_retries=max_retries,
            raise_on_error=raise_on_error,
        )

    def generate_text(self, prompt: str) -> str:
        messages = build_text_messages(prompt)
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": 0.0,
        }
        if self._json_response:
            payload["response_format"] = {"type": "json_object"}
        if self._thinking is DeepSeekThinkingMode.ENABLED:
            payload["thinking"] = {"type": "enabled"}
        elif self._thinking is DeepSeekThinkingMode.DISABLED:
            payload["thinking"] = {"type": "disabled"}

        self.last_meta = None
        tracker = get_budget_tracker()
        reservation_id: str | None = None
        if tracker is not None:
            reservation_id = reserve_for_prompt(
                tracker,
                prompt=prompt,
                model=self._model,
                max_tokens=self._max_tokens,
            )

        try:
            resp = self._http.post(_ENDPOINT, payload)
            if resp is None:
                if tracker is not None and reservation_id is not None:
                    tracker.release_failure(reservation_id)
                return ""

            try:
                body = resp.json()
                first = body["choices"][0]
                message = first["message"]
                content = message.get("content")
            except (KeyError, IndexError, ValueError, TypeError):
                if tracker is not None and reservation_id is not None:
                    tracker.release_failure(reservation_id)
                error = LLMResponseStructureError(
                    provider="deepseek",
                    model=self._model,
                    operation="text_generate",
                )
                logger.warning("[llm] provider=deepseek error=%s", error.safe_reason)
                if self._raise_on_error:
                    raise error
                return ""
            if not isinstance(content, str):
                if tracker is not None and reservation_id is not None:
                    tracker.release_failure(reservation_id)
                error = LLMResponseStructureError(
                    provider="deepseek",
                    model=self._model,
                    operation="text_generate",
                )
                logger.warning("[llm] provider=deepseek error=%s", error.safe_reason)
                if self._raise_on_error:
                    raise error
                return ""
            text = content.strip()
            usage = parse_deepseek_usage(body, configured_model=self._model)
            actual_model = str(body.get("model") or self._model)
            self.last_meta = {
                "configured_model": self._model,
                "actual_model": actual_model,
                "usage": None if usage is None else usage.to_dict(),
                "usage_missing": usage is None,
            }
            if tracker is not None and reservation_id is not None:
                if usage is None:
                    tracker.commit_missing_usage(reservation_id)
                else:
                    tracker.commit_success(reservation_id, usage)
            if not text:
                category = "empty_message_content"
                if message.get("tool_calls"):
                    category = "tool_call_instead_of_content"
                elif message.get("refusal"):
                    category = "provider_refusal"
                content_error = LLMProviderError(
                    provider="deepseek",
                    category=category,
                    message="Provider returned HTTP 2xx with no usable message.content.",
                    model=self._model,
                    operation="text_generate",
                )
                logger.warning("[llm] provider=deepseek error=%s", content_error.safe_reason)
                if self._raise_on_error:
                    raise content_error
                return ""
            return text
        except BudgetExhaustedError:
            raise
        except Exception:
            if tracker is not None and reservation_id is not None:
                tracker.release_failure(reservation_id)
            raise