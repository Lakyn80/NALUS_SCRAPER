"""
Shared HTTP infrastructure, prompt building, and response parsing
for all LLM provider adapters.

Internal module — import only from sibling provider files.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from app.core.logging import get_logger
from app.rag.llm.models import LLMOutput

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# ENV-based defaults
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "10"))
DEFAULT_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "512"))
DEFAULT_RETRY: int = int(os.getenv("LLM_RETRY", "2"))

# ---------------------------------------------------------------------------
# Prompt limits
# ---------------------------------------------------------------------------

_MAX_CHUNKS = 5
_MAX_CHUNK_CHARS = 800
_RETRY_DELAYS = [0.5, 1.0]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_RAG_SYSTEM = (
    "Jsi právní asistent specializovaný na judikaturu Ústavního soudu České republiky. "
    "Odpovídej výhradně v češtině. Buď věcný a přesný."
)

_RAG_USER_TEMPLATE = (
    "Na základě níže uvedených úryvků z rozhodnutí Ústavního soudu odpověz na otázku.\n\n"
    "Odpověz VÝHRADNĚ ve formátu JSON (žádný markdown, žádné komentáře):\n"
    "{{\n"
    '  "answer": "<odpověď v češtině>",\n'
    '  "reasoning": "<zdůvodnění výběru zdrojů>",\n'
    '  "sources": ["<id1>", "<id2>"],\n'
    '  "confidence": <číslo 0.0 až 1.0>\n'
    "}}\n\n"
    "Otázka: {query}\n\n"
    "Úryvky:\n{chunks}"
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def empty_output() -> LLMOutput:
    """Return a safe empty LLMOutput (fresh instance every call)."""
    return LLMOutput(answer="", reasoning="", sources=[], confidence=0.0)


def build_rag_messages(query: str, chunks: list) -> list[dict[str, str]]:
    """Build OpenAI-compatible message list for structured RAG answering."""
    safe_q = _escape(query)
    top = chunks[:_MAX_CHUNKS]
    parts = [f"[{c.id}] {_escape(c.text[:_MAX_CHUNK_CHARS])}" for c in top]
    chunks_text = "\n\n".join(parts) or "(žádné úryvky)"
    user_content = _RAG_USER_TEMPLATE.format(query=safe_q, chunks=chunks_text)
    return [
        {"role": "system", "content": _RAG_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def build_text_messages(prompt: str) -> list[dict[str, str]]:
    """Build OpenAI-compatible message list for plain text generation."""
    return [{"role": "user", "content": prompt}]


def to_claude_payload(
    messages: list[dict[str, str]], model: str, max_tokens: int
) -> dict[str, Any]:
    """Convert OpenAI-style messages to Claude /v1/messages request payload."""
    system = ""
    user_messages: list[dict[str, str]] = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            user_messages.append({"role": m["role"], "content": m["content"]})
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": user_messages,
    }
    if system:
        payload["system"] = system
    return payload


def parse_rag_response(text: str, chunk_ids: list[str]) -> LLMOutput:
    """Parse structured JSON from LLM response with plain-text fallback."""
    text = text.strip()
    if not text:
        return empty_output()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:]
        if inner and inner[-1].strip() in ("```", "```json"):
            inner = inner[:-1]
        text = "\n".join(inner).strip()
        if not text:
            return empty_output()

    try:
        data = json.loads(text)
        answer = str(data.get("answer", "")).strip()
        if not answer:
            return empty_output()
        reasoning = str(data.get("reasoning", "")).strip()
        raw_sources = data.get("sources", [])
        sources = [str(s) for s in raw_sources] if isinstance(raw_sources, list) else []
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        return LLMOutput(
            answer=answer,
            reasoning=reasoning,
            sources=sources,
            confidence=confidence,
        )
    except (json.JSONDecodeError, TypeError, AttributeError):
        # Plain text fallback: use raw text as answer
        return LLMOutput(
            answer=text,
            reasoning="",
            sources=chunk_ids[:3],
            confidence=0.0,
        )


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


class HTTPClient:
    """Shared HTTP client with retry, exponential backoff, and structured logging.

    Thread-safe for reads (no mutable state after construction).
    Each adapter instance owns its own HTTPClient instance.
    """

    def __init__(
        self,
        provider: str,
        headers: dict[str, str],
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_RETRY,
    ) -> None:
        self.provider = provider
        self._max_retries = max_retries
        self._client = httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(timeout),
        )

    def post(self, url: str, payload: dict[str, Any]) -> httpx.Response | None:
        """POST with retry on 5xx / timeout / network error.

        Returns the response on the first successful (non-5xx) response.
        Returns None after all retries are exhausted.
        """
        model = str(payload.get("model", ""))
        input_length = len(json.dumps(payload, ensure_ascii=False))
        retry_count = 0
        start = time.monotonic()

        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.post(url, json=payload)
            except httpx.TimeoutException as exc:
                if attempt < self._max_retries:
                    retry_count += 1
                    time.sleep(_RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)])
                    continue
                logger.warning("[llm] provider=%s error=timeout: %s", self.provider, exc)
                return None
            except httpx.RequestError as exc:
                if attempt < self._max_retries:
                    retry_count += 1
                    time.sleep(_RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)])
                    continue
                logger.warning("[llm] provider=%s error=network: %s", self.provider, exc)
                return None

            if resp.status_code >= 500:
                if attempt < self._max_retries:
                    retry_count += 1
                    time.sleep(_RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)])
                    continue
                logger.warning(
                    "[llm] provider=%s error=HTTP %d", self.provider, resp.status_code
                )
                return None

            latency_ms = int((time.monotonic() - start) * 1000)
            logger.info(
                "[llm] provider=%s model=%s latency_ms=%d input_length=%d"
                " output_length=%d retry_count=%d",
                self.provider,
                model,
                latency_ms,
                input_length,
                len(resp.text),
                retry_count,
            )
            return resp

        return None  # unreachable but satisfies type checker

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _escape(text: str) -> str:
    """Neutralise format-string markers to prevent prompt injection."""
    return text.replace("{", "(").replace("}", ")")
