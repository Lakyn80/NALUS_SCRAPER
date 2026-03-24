"""
LLM Provider Factory.

Returns the correct adapter instance for the requested provider.
Reads configuration from environment variables when not passed explicitly.

Usage:
    import os
    from app.rag.llm.provider_factory import get_llm, get_text_llm

    provider = os.getenv("LLM_PROVIDER", "deepseek")
    api_key  = os.getenv("LLM_API_KEY", "")

    text_llm = get_text_llm(provider, api_key)   # BaseTextLLM
    llm      = get_llm(provider, api_key)         # BaseLLM

    # Plug into existing services without modification:
    rewrite_service = QueryRewriteService(text_llm)
    planner         = PlannerService(text_llm)
    synthesis       = SynthesisService(text_llm)
    llm_service     = LLMService(llm)
"""

from __future__ import annotations

from app.rag.llm.base import BaseLLM
from app.rag.llm.providers.claude import ClaudeLLM, ClaudeTextLLM
from app.rag.llm.providers.deepseek import DeepSeekLLM, DeepSeekTextLLM
from app.rag.llm.providers.openai import OpenAILLM, OpenAITextLLM
from app.rag.rewrite.query_rewrite_service import BaseTextLLM

_SUPPORTED = ("deepseek", "openai", "claude")


def get_llm(provider: str, api_key: str) -> BaseLLM:
    """Return a BaseLLM adapter for the given provider.

    Args:
        provider: One of "deepseek", "openai", "claude".
        api_key:  Provider API key.

    Raises:
        ValueError: If provider is not recognised.
    """
    if provider == "deepseek":
        return DeepSeekLLM(api_key=api_key)
    if provider == "openai":
        return OpenAILLM(api_key=api_key)
    if provider == "claude":
        return ClaudeLLM(api_key=api_key)
    raise ValueError(
        f"Unknown LLM provider: {provider!r}. Supported: {', '.join(_SUPPORTED)}"
    )


def get_text_llm(provider: str, api_key: str) -> BaseTextLLM:
    """Return a BaseTextLLM adapter for the given provider.

    Args:
        provider: One of "deepseek", "openai", "claude".
        api_key:  Provider API key.

    Raises:
        ValueError: If provider is not recognised.
    """
    if provider == "deepseek":
        return DeepSeekTextLLM(api_key=api_key)
    if provider == "openai":
        return OpenAITextLLM(api_key=api_key)
    if provider == "claude":
        return ClaudeTextLLM(api_key=api_key)
    raise ValueError(
        f"Unknown LLM provider: {provider!r}. Supported: {', '.join(_SUPPORTED)}"
    )
