"""
Claude (Anthropic) LLM adapters.

ClaudeLLM    — BaseLLM    — structured RAG answering
ClaudeTextLLM — BaseTextLLM — plain text generation (rewrite / planner / synthesis)

Endpoint: https://api.anthropic.com/v1/messages
Auth:     x-api-key: <api_key>
Format:   Anthropic Messages API (different from OpenAI chat/completions)
"""

from __future__ import annotations

import os

from app.core.logging import get_logger
from app.rag.llm.base import BaseLLM
from app.rag.llm.models import LLMInput, LLMOutput
from app.rag.llm.providers._base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_RETRY,
    DEFAULT_TIMEOUT,
    HTTPClient,
    build_rag_messages,
    build_text_messages,
    empty_output,
    parse_rag_response,
    to_claude_payload,
)
from app.rag.rewrite.query_rewrite_service import BaseTextLLM

logger = get_logger(__name__)

_ENDPOINT = "https://api.anthropic.com/v1/messages"
_DEFAULT_MODEL = os.getenv("LLM_MODEL_CLAUDE", "claude-3-haiku-20240307")
_ANTHROPIC_VERSION = "2023-06-01"


# ---------------------------------------------------------------------------
# Shared header builder
# ---------------------------------------------------------------------------


def _headers(api_key: str) -> dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": _ANTHROPIC_VERSION,
        "Content-Type": "application/json",
    }


def _extract_claude_text(body: dict) -> str:
    """Extract text from Claude /v1/messages response body."""
    content = body.get("content", [])
    if not content or not isinstance(content, list):
        return ""
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            return str(block.get("text", "")).strip()
    return ""


# ---------------------------------------------------------------------------
# ClaudeLLM — BaseLLM
# ---------------------------------------------------------------------------


class ClaudeLLM(BaseLLM):
    """Claude adapter for structured RAG answering (BaseLLM contract)."""

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = DEFAULT_RETRY,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._http = HTTPClient(
            provider="claude",
            headers=_headers(api_key),
            timeout=timeout,
            max_retries=max_retries,
        )

    def generate(self, data: LLMInput) -> LLMOutput:
        messages = build_rag_messages(data.query, data.chunks)
        payload = to_claude_payload(messages, self._model, self._max_tokens)
        chunk_ids = [c.id for c in data.chunks[:5]]

        resp = self._http.post(_ENDPOINT, payload)
        if resp is None:
            return empty_output()

        try:
            body = resp.json()
            text = _extract_claude_text(body)
        except (ValueError, TypeError):
            logger.warning("[llm] provider=claude error=invalid response structure")
            return empty_output()

        if not text:
            logger.warning("[llm] provider=claude error=empty response content")
            return empty_output()

        return parse_rag_response(text, chunk_ids)


# ---------------------------------------------------------------------------
# ClaudeTextLLM — BaseTextLLM
# ---------------------------------------------------------------------------


class ClaudeTextLLM(BaseTextLLM):
    """Claude adapter for plain text generation (BaseTextLLM contract)."""

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = DEFAULT_RETRY,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._http = HTTPClient(
            provider="claude",
            headers=_headers(api_key),
            timeout=timeout,
            max_retries=max_retries,
        )

    def generate_text(self, prompt: str) -> str:
        messages = build_text_messages(prompt)
        payload = to_claude_payload(messages, self._model, self._max_tokens)

        resp = self._http.post(_ENDPOINT, payload)
        if resp is None:
            return ""

        try:
            body = resp.json()
            return _extract_claude_text(body)
        except (ValueError, TypeError):
            logger.warning("[llm] provider=claude error=invalid response structure")
            return ""
