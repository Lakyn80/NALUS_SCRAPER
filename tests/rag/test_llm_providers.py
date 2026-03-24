"""
Unit tests for LLM provider adapters and the provider factory.

All HTTP calls are mocked — no real API calls made.

Run:
    pytest tests/rag/test_llm_providers.py -v
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.rag.llm.models import LLMInput, LLMOutput
from app.rag.llm.provider_factory import get_llm, get_text_llm
from app.rag.llm.providers.claude import ClaudeLLM, ClaudeTextLLM
from app.rag.llm.providers.deepseek import DeepSeekLLM, DeepSeekTextLLM
from app.rag.llm.providers.openai import OpenAILLM, OpenAITextLLM
from app.rag.retrieval.models import RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(id: str = "1", text: str = "právní text") -> RetrievedChunk:
    return RetrievedChunk(id=id, text=text, score=0.9, source="keyword")


def _llm_input(*chunk_ids: str) -> LLMInput:
    return LLMInput(
        query="únos dítěte do zahraničí",
        chunks=[_chunk(cid) for cid in chunk_ids] if chunk_ids else [_chunk()],
    )


# --- OpenAI / DeepSeek response envelope ---

def _openai_envelope(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


# --- Claude response envelope ---

def _claude_envelope(content: str) -> dict:
    return {"content": [{"type": "text", "text": content}]}


# --- Structured JSON that the LLM returns as its content ---

def _rag_json(
    answer: str = "Odpověď",
    reasoning: str = "Zdůvodnění",
    sources: list[str] | None = None,
    confidence: float = 0.85,
) -> str:
    return json.dumps(
        {
            "answer": answer,
            "reasoning": reasoning,
            "sources": sources if sources is not None else ["1"],
            "confidence": confidence,
        },
        ensure_ascii=False,
    )


# --- Mock httpx.Client factory ---

def _mock_client(json_data: dict | None = None, status_code: int = 200, text: str = ""):
    """Return a patched httpx.Client class whose instance.post() returns a mock response."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = status_code
    mock_resp.text = text if json_data is None else json.dumps(json_data)
    if json_data is not None:
        mock_resp.json.return_value = json_data
    else:
        mock_resp.json.side_effect = ValueError("no JSON")

    mock_instance = MagicMock()
    mock_instance.post.return_value = mock_resp

    mock_class = MagicMock()
    mock_class.return_value = mock_instance
    return mock_class, mock_instance


def _timeout_client():
    """httpx.Client whose post() always raises TimeoutException."""
    mock_instance = MagicMock()
    mock_instance.post.side_effect = httpx.TimeoutException("timed out")
    mock_class = MagicMock()
    mock_class.return_value = mock_instance
    return mock_class


def _network_error_client():
    """httpx.Client whose post() always raises RequestError."""
    mock_instance = MagicMock()
    mock_instance.post.side_effect = httpx.RequestError("connection refused")
    mock_class = MagicMock()
    mock_class.return_value = mock_instance
    return mock_class


# ---------------------------------------------------------------------------
# Provider Factory
# ---------------------------------------------------------------------------


class TestGetLLM:
    def test_deepseek_returns_deepseek_llm(self) -> None:
        assert isinstance(get_llm("deepseek", "key"), DeepSeekLLM)

    def test_openai_returns_openai_llm(self) -> None:
        assert isinstance(get_llm("openai", "key"), OpenAILLM)

    def test_claude_returns_claude_llm(self) -> None:
        assert isinstance(get_llm("claude", "key"), ClaudeLLM)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm("mistral", "key")


class TestGetTextLLM:
    def test_deepseek_returns_deepseek_text_llm(self) -> None:
        assert isinstance(get_text_llm("deepseek", "key"), DeepSeekTextLLM)

    def test_openai_returns_openai_text_llm(self) -> None:
        assert isinstance(get_text_llm("openai", "key"), OpenAITextLLM)

    def test_claude_returns_claude_text_llm(self) -> None:
        assert isinstance(get_text_llm("claude", "key"), ClaudeTextLLM)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_text_llm("llama", "key")


# ---------------------------------------------------------------------------
# DeepSeek — BaseLLM
# ---------------------------------------------------------------------------


class TestDeepSeekLLM:
    def test_returns_llm_output_type(self) -> None:
        mock_class, _ = _mock_client(_openai_envelope(_rag_json()))
        with patch("httpx.Client", mock_class):
            result = DeepSeekLLM(api_key="k").generate(_llm_input())
        assert isinstance(result, LLMOutput)

    def test_parses_answer(self) -> None:
        mock_class, _ = _mock_client(_openai_envelope(_rag_json(answer="Právní závěr")))
        with patch("httpx.Client", mock_class):
            result = DeepSeekLLM(api_key="k").generate(_llm_input())
        assert result.answer == "Právní závěr"

    def test_parses_confidence(self) -> None:
        mock_class, _ = _mock_client(_openai_envelope(_rag_json(confidence=0.75)))
        with patch("httpx.Client", mock_class):
            result = DeepSeekLLM(api_key="k").generate(_llm_input())
        assert result.confidence == pytest.approx(0.75)

    def test_parses_sources(self) -> None:
        mock_class, _ = _mock_client(_openai_envelope(_rag_json(sources=["A", "B"])))
        with patch("httpx.Client", mock_class):
            result = DeepSeekLLM(api_key="k").generate(_llm_input("A", "B"))
        assert "A" in result.sources
        assert "B" in result.sources

    def test_fallback_on_timeout(self) -> None:
        with patch("httpx.Client", _timeout_client()), patch("time.sleep"):
            result = DeepSeekLLM(api_key="k", max_retries=1).generate(_llm_input())
        assert result == LLMOutput(answer="", reasoning="", sources=[], confidence=0.0)

    def test_fallback_on_network_error(self) -> None:
        with patch("httpx.Client", _network_error_client()), patch("time.sleep"):
            result = DeepSeekLLM(api_key="k", max_retries=1).generate(_llm_input())
        assert result.answer == ""

    def test_fallback_on_5xx(self) -> None:
        mock_class, _ = _mock_client(status_code=503)
        with patch("httpx.Client", mock_class), patch("time.sleep"):
            result = DeepSeekLLM(api_key="k", max_retries=1).generate(_llm_input())
        assert result.answer == ""

    def test_fallback_on_invalid_json_content(self) -> None:
        mock_class, _ = _mock_client(_openai_envelope("not valid json {{ oops"))
        with patch("httpx.Client", mock_class):
            result = DeepSeekLLM(api_key="k").generate(_llm_input())
        # plain-text fallback: raw text becomes answer
        assert isinstance(result.answer, str)
        assert result.answer != ""

    def test_fallback_on_missing_choices_key(self) -> None:
        mock_class, _ = _mock_client({"unexpected": "structure"})
        with patch("httpx.Client", mock_class):
            result = DeepSeekLLM(api_key="k").generate(_llm_input())
        assert result.answer == ""

    def test_never_raises(self) -> None:
        with patch("httpx.Client", _timeout_client()), patch("time.sleep"):
            # Must not raise under any circumstances
            result = DeepSeekLLM(api_key="k", max_retries=0).generate(_llm_input())
        assert isinstance(result, LLMOutput)


# ---------------------------------------------------------------------------
# DeepSeek — BaseTextLLM
# ---------------------------------------------------------------------------


class TestDeepSeekTextLLM:
    def test_returns_str(self) -> None:
        mock_class, _ = _mock_client(_openai_envelope("přepsaný dotaz"))
        with patch("httpx.Client", mock_class):
            result = DeepSeekTextLLM(api_key="k").generate_text("dotaz")
        assert isinstance(result, str)

    def test_returns_content(self) -> None:
        mock_class, _ = _mock_client(_openai_envelope("výstup textu"))
        with patch("httpx.Client", mock_class):
            result = DeepSeekTextLLM(api_key="k").generate_text("prompt")
        assert result == "výstup textu"

    def test_fallback_on_timeout(self) -> None:
        with patch("httpx.Client", _timeout_client()), patch("time.sleep"):
            result = DeepSeekTextLLM(api_key="k", max_retries=1).generate_text("p")
        assert result == ""

    def test_fallback_on_invalid_structure(self) -> None:
        mock_class, _ = _mock_client({"no_choices": True})
        with patch("httpx.Client", mock_class):
            result = DeepSeekTextLLM(api_key="k").generate_text("p")
        assert result == ""

    def test_never_raises(self) -> None:
        with patch("httpx.Client", _network_error_client()), patch("time.sleep"):
            result = DeepSeekTextLLM(api_key="k", max_retries=0).generate_text("p")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# OpenAI — BaseLLM
# ---------------------------------------------------------------------------


class TestOpenAILLM:
    def test_returns_llm_output_type(self) -> None:
        mock_class, _ = _mock_client(_openai_envelope(_rag_json()))
        with patch("httpx.Client", mock_class):
            result = OpenAILLM(api_key="k").generate(_llm_input())
        assert isinstance(result, LLMOutput)

    def test_parses_answer(self) -> None:
        mock_class, _ = _mock_client(_openai_envelope(_rag_json(answer="Závěr soudu")))
        with patch("httpx.Client", mock_class):
            result = OpenAILLM(api_key="k").generate(_llm_input())
        assert result.answer == "Závěr soudu"

    def test_fallback_on_timeout(self) -> None:
        with patch("httpx.Client", _timeout_client()), patch("time.sleep"):
            result = OpenAILLM(api_key="k", max_retries=1).generate(_llm_input())
        assert result.answer == ""

    def test_fallback_on_5xx(self) -> None:
        mock_class, _ = _mock_client(status_code=500)
        with patch("httpx.Client", mock_class), patch("time.sleep"):
            result = OpenAILLM(api_key="k", max_retries=1).generate(_llm_input())
        assert result.answer == ""

    def test_fallback_on_missing_structure(self) -> None:
        mock_class, _ = _mock_client({"data": []})
        with patch("httpx.Client", mock_class):
            result = OpenAILLM(api_key="k").generate(_llm_input())
        assert result.answer == ""

    def test_never_raises(self) -> None:
        with patch("httpx.Client", _network_error_client()), patch("time.sleep"):
            result = OpenAILLM(api_key="k", max_retries=0).generate(_llm_input())
        assert isinstance(result, LLMOutput)


# ---------------------------------------------------------------------------
# OpenAI — BaseTextLLM
# ---------------------------------------------------------------------------


class TestOpenAITextLLM:
    def test_returns_str(self) -> None:
        mock_class, _ = _mock_client(_openai_envelope("výsledek"))
        with patch("httpx.Client", mock_class):
            result = OpenAITextLLM(api_key="k").generate_text("dotaz")
        assert isinstance(result, str)

    def test_returns_content(self) -> None:
        mock_class, _ = _mock_client(_openai_envelope("právní termín"))
        with patch("httpx.Client", mock_class):
            result = OpenAITextLLM(api_key="k").generate_text("p")
        assert result == "právní termín"

    def test_fallback_on_network_error(self) -> None:
        with patch("httpx.Client", _network_error_client()), patch("time.sleep"):
            result = OpenAITextLLM(api_key="k", max_retries=0).generate_text("p")
        assert result == ""

    def test_never_raises(self) -> None:
        with patch("httpx.Client", _timeout_client()), patch("time.sleep"):
            result = OpenAITextLLM(api_key="k", max_retries=0).generate_text("p")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Claude — BaseLLM
# ---------------------------------------------------------------------------


class TestClaudeLLM:
    def test_returns_llm_output_type(self) -> None:
        mock_class, _ = _mock_client(_claude_envelope(_rag_json()))
        with patch("httpx.Client", mock_class):
            result = ClaudeLLM(api_key="k").generate(_llm_input())
        assert isinstance(result, LLMOutput)

    def test_parses_answer(self) -> None:
        mock_class, _ = _mock_client(_claude_envelope(_rag_json(answer="Claude odpověď")))
        with patch("httpx.Client", mock_class):
            result = ClaudeLLM(api_key="k").generate(_llm_input())
        assert result.answer == "Claude odpověď"

    def test_parses_confidence(self) -> None:
        mock_class, _ = _mock_client(_claude_envelope(_rag_json(confidence=0.6)))
        with patch("httpx.Client", mock_class):
            result = ClaudeLLM(api_key="k").generate(_llm_input())
        assert result.confidence == pytest.approx(0.6)

    def test_fallback_on_timeout(self) -> None:
        with patch("httpx.Client", _timeout_client()), patch("time.sleep"):
            result = ClaudeLLM(api_key="k", max_retries=1).generate(_llm_input())
        assert result.answer == ""

    def test_fallback_on_5xx(self) -> None:
        mock_class, _ = _mock_client(status_code=529)
        with patch("httpx.Client", mock_class), patch("time.sleep"):
            result = ClaudeLLM(api_key="k", max_retries=1).generate(_llm_input())
        assert result.answer == ""

    def test_fallback_on_empty_content_list(self) -> None:
        mock_class, _ = _mock_client({"content": []})
        with patch("httpx.Client", mock_class):
            result = ClaudeLLM(api_key="k").generate(_llm_input())
        assert result.answer == ""

    def test_fallback_on_missing_content_key(self) -> None:
        mock_class, _ = _mock_client({"type": "message", "role": "assistant"})
        with patch("httpx.Client", mock_class):
            result = ClaudeLLM(api_key="k").generate(_llm_input())
        assert result.answer == ""

    def test_never_raises(self) -> None:
        with patch("httpx.Client", _network_error_client()), patch("time.sleep"):
            result = ClaudeLLM(api_key="k", max_retries=0).generate(_llm_input())
        assert isinstance(result, LLMOutput)


# ---------------------------------------------------------------------------
# Claude — BaseTextLLM
# ---------------------------------------------------------------------------


class TestClaudeTextLLM:
    def test_returns_str(self) -> None:
        mock_class, _ = _mock_client(_claude_envelope("výstup"))
        with patch("httpx.Client", mock_class):
            result = ClaudeTextLLM(api_key="k").generate_text("prompt")
        assert isinstance(result, str)

    def test_returns_content(self) -> None:
        mock_class, _ = _mock_client(_claude_envelope("přepsaný dotaz"))
        with patch("httpx.Client", mock_class):
            result = ClaudeTextLLM(api_key="k").generate_text("p")
        assert result == "přepsaný dotaz"

    def test_fallback_on_timeout(self) -> None:
        with patch("httpx.Client", _timeout_client()), patch("time.sleep"):
            result = ClaudeTextLLM(api_key="k", max_retries=1).generate_text("p")
        assert result == ""

    def test_fallback_on_missing_content(self) -> None:
        mock_class, _ = _mock_client({"type": "message"})
        with patch("httpx.Client", mock_class):
            result = ClaudeTextLLM(api_key="k").generate_text("p")
        assert result == ""

    def test_never_raises(self) -> None:
        with patch("httpx.Client", _network_error_client()), patch("time.sleep"):
            result = ClaudeTextLLM(api_key="k", max_retries=0).generate_text("p")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Response normalization
# ---------------------------------------------------------------------------


class TestResponseNormalization:
    """Tests for parse_rag_response shared parser via real adapter calls."""

    def _run_deepseek(self, content: str) -> LLMOutput:
        mock_class, _ = _mock_client(_openai_envelope(content))
        with patch("httpx.Client", mock_class):
            return DeepSeekLLM(api_key="k").generate(_llm_input("X"))

    def test_json_with_all_fields(self) -> None:
        result = self._run_deepseek(_rag_json("Odpověď", "Důvod", ["X"], 0.9))
        assert result.answer == "Odpověď"
        assert result.reasoning == "Důvod"
        assert result.sources == ["X"]
        assert result.confidence == pytest.approx(0.9)

    def test_json_missing_reasoning_defaults_empty(self) -> None:
        content = json.dumps({"answer": "A", "sources": [], "confidence": 0.5})
        result = self._run_deepseek(content)
        assert result.reasoning == ""

    def test_json_missing_confidence_defaults_zero(self) -> None:
        content = json.dumps({"answer": "A", "reasoning": "R", "sources": []})
        result = self._run_deepseek(content)
        assert result.confidence == pytest.approx(0.0)

    def test_confidence_clamped_above_one(self) -> None:
        content = json.dumps({"answer": "A", "reasoning": "", "sources": [], "confidence": 5.0})
        result = self._run_deepseek(content)
        assert result.confidence == pytest.approx(1.0)

    def test_confidence_clamped_below_zero(self) -> None:
        content = json.dumps({"answer": "A", "reasoning": "", "sources": [], "confidence": -1.0})
        result = self._run_deepseek(content)
        assert result.confidence == pytest.approx(0.0)

    def test_plain_text_fallback(self) -> None:
        result = self._run_deepseek("Toto není JSON, pouze text odpovědi.")
        assert result.answer == "Toto není JSON, pouze text odpovědi."
        assert result.confidence == pytest.approx(0.0)

    def test_markdown_fenced_json_stripped(self) -> None:
        content = "```json\n" + _rag_json(answer="fenced") + "\n```"
        result = self._run_deepseek(content)
        assert result.answer == "fenced"

    def test_empty_content_returns_empty_output(self) -> None:
        result = self._run_deepseek("")
        assert result.answer == ""
        assert result.sources == []

    def test_empty_answer_field_returns_empty_output(self) -> None:
        content = json.dumps({"answer": "", "reasoning": "R", "sources": ["1"], "confidence": 0.9})
        result = self._run_deepseek(content)
        assert result.answer == ""

    def test_sources_not_list_defaults_empty(self) -> None:
        content = json.dumps({"answer": "A", "reasoning": "", "sources": "not-a-list", "confidence": 0.5})
        result = self._run_deepseek(content)
        assert result.sources == []


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


class TestRetryBehaviour:
    def test_retries_on_5xx_then_succeeds(self) -> None:
        ok_resp = MagicMock(spec=httpx.Response)
        ok_resp.status_code = 200
        ok_resp.text = json.dumps(_openai_envelope(_rag_json()))
        ok_resp.json.return_value = _openai_envelope(_rag_json())

        fail_resp = MagicMock(spec=httpx.Response)
        fail_resp.status_code = 503
        fail_resp.text = "error"

        mock_instance = MagicMock()
        mock_instance.post.side_effect = [fail_resp, ok_resp]
        mock_class = MagicMock()
        mock_class.return_value = mock_instance

        with patch("httpx.Client", mock_class), patch("time.sleep"):
            result = DeepSeekLLM(api_key="k", max_retries=2).generate(_llm_input())

        assert result.answer != ""
        assert mock_instance.post.call_count == 2

    def test_exhausts_retries_on_persistent_5xx(self) -> None:
        fail_resp = MagicMock(spec=httpx.Response)
        fail_resp.status_code = 500
        fail_resp.text = "error"

        mock_instance = MagicMock()
        mock_instance.post.return_value = fail_resp
        mock_class = MagicMock()
        mock_class.return_value = mock_instance

        with patch("httpx.Client", mock_class), patch("time.sleep"):
            result = DeepSeekLLM(api_key="k", max_retries=2).generate(_llm_input())

        assert result.answer == ""
        # 1 initial + 2 retries = 3 attempts
        assert mock_instance.post.call_count == 3

    def test_retries_on_timeout_then_succeeds(self) -> None:
        ok_resp = MagicMock(spec=httpx.Response)
        ok_resp.status_code = 200
        ok_resp.text = json.dumps(_openai_envelope(_rag_json()))
        ok_resp.json.return_value = _openai_envelope(_rag_json())

        mock_instance = MagicMock()
        mock_instance.post.side_effect = [
            httpx.TimeoutException("timeout"),
            ok_resp,
        ]
        mock_class = MagicMock()
        mock_class.return_value = mock_instance

        with patch("httpx.Client", mock_class), patch("time.sleep"):
            result = DeepSeekLLM(api_key="k", max_retries=2).generate(_llm_input())

        assert result.answer != ""
        assert mock_instance.post.call_count == 2

    def test_zero_retries_no_sleep_called(self) -> None:
        with patch("httpx.Client", _timeout_client()) as _, patch("time.sleep") as mock_sleep:
            DeepSeekLLM(api_key="k", max_retries=0).generate(_llm_input())
        mock_sleep.assert_not_called()

    def test_retry_uses_sleep_with_backoff(self) -> None:
        fail_resp = MagicMock(spec=httpx.Response)
        fail_resp.status_code = 500
        fail_resp.text = "error"

        mock_instance = MagicMock()
        mock_instance.post.return_value = fail_resp
        mock_class = MagicMock()
        mock_class.return_value = mock_instance

        with patch("httpx.Client", mock_class), patch("time.sleep") as mock_sleep:
            DeepSeekLLM(api_key="k", max_retries=2).generate(_llm_input())

        assert mock_sleep.call_count == 2
        calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert calls[0] == pytest.approx(0.5)
        assert calls[1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestLogging:
    def test_info_logged_on_success(self, caplog) -> None:
        import logging

        mock_class, _ = _mock_client(_openai_envelope(_rag_json()))
        with patch("httpx.Client", mock_class):
            with caplog.at_level(logging.INFO, logger="app.rag.llm.providers._base"):
                DeepSeekLLM(api_key="k").generate(_llm_input())
        msgs = [r.getMessage() for r in caplog.records]
        assert any("[llm]" in m and "provider=deepseek" in m for m in msgs)

    def test_warning_logged_on_timeout(self, caplog) -> None:
        import logging

        with patch("httpx.Client", _timeout_client()), patch("time.sleep"):
            with caplog.at_level(logging.WARNING, logger="app.rag.llm.providers._base"):
                DeepSeekLLM(api_key="k", max_retries=0).generate(_llm_input())
        msgs = [r.getMessage() for r in caplog.records]
        assert any("[llm]" in m and "timeout" in m for m in msgs)

    def test_warning_logged_on_5xx(self, caplog) -> None:
        import logging

        mock_class, _ = _mock_client(status_code=500)
        with patch("httpx.Client", mock_class), patch("time.sleep"):
            with caplog.at_level(logging.WARNING, logger="app.rag.llm.providers._base"):
                DeepSeekLLM(api_key="k", max_retries=0).generate(_llm_input())
        msgs = [r.getMessage() for r in caplog.records]
        assert any("[llm]" in m and "HTTP 500" in m for m in msgs)
