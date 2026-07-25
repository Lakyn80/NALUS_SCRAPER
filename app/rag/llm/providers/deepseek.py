"""
DeepSeek LLM adapters.

DeepSeekLLM    — BaseLLM    — structured RAG answering
DeepSeekTextLLM — BaseTextLLM — plain text generation (rewrite / planner / synthesis)

Endpoint: https://api.deepseek.com/chat/completions
Auth:     Authorization: Bearer <api_key>
"""

from __future__ import annotations

import os

from app.core.logging import get_logger
from app.rag.llm.base import BaseLLM
from app.rag.llm.models import LLMInput, LLMOutput
from app.rag.llm.config import DEEPSEEK_DEFAULT_MODEL
from app.rag.llm.providers._base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_RETRY,
    DEFAULT_TIMEOUT,
    HTTPClient,
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

        resp = self._http.post(_ENDPOINT, payload)
        if resp is None:
            return ""

        try:
            body = resp.json()
            return body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, ValueError, TypeError):
            error = LLMResponseStructureError(
                provider="deepseek",
                model=self._model,
                operation="text_generate",
            )
            logger.warning("[llm] provider=deepseek error=%s", error.safe_reason)
            if self._raise_on_error:
                raise error
            return ""
