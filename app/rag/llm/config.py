from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Mapping

DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_PROVIDER = "deepseek"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_TOKENS = 512
DEFAULT_RETRY = 2
LEGAL_V2_MIN_MAX_TOKENS = 2400


@dataclass(frozen=True)
class EffectiveLLMConfig:
    provider: str
    deepseek_model: str
    timeout_seconds: float
    max_tokens: int
    legal_v2_max_tokens: int
    retry_count: int
    api_key_configured: bool
    api_key_length: int
    api_key_prefix: str
    sources: dict[str, str]

    def to_safe_dict(self) -> dict[str, object]:
        return asdict(self)


def effective_llm_config_from_env(
    env: Mapping[str, str] | None = None,
) -> EffectiveLLMConfig:
    values = env or os.environ
    provider, provider_source = _string_value(
        values, "LLM_PROVIDER", DEFAULT_PROVIDER
    )
    deepseek_model, deepseek_model_source = _string_value(
        values, "LLM_MODEL_DEEPSEEK", DEEPSEEK_DEFAULT_MODEL
    )
    timeout, timeout_source = _float_value(
        values, "LLM_TIMEOUT", DEFAULT_TIMEOUT_SECONDS
    )
    max_tokens, max_tokens_source = _int_value(
        values, "LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS
    )
    legal_v2_max_tokens, legal_v2_max_tokens_source = _legal_v2_max_tokens(
        values, max_tokens
    )
    retry_count, retry_source = _int_value(values, "LLM_RETRY", DEFAULT_RETRY)
    api_key = str(values.get("LLM_API_KEY", ""))
    stripped_key = api_key.strip()
    return EffectiveLLMConfig(
        provider=provider.lower(),
        deepseek_model=deepseek_model,
        timeout_seconds=timeout,
        max_tokens=max_tokens,
        legal_v2_max_tokens=legal_v2_max_tokens,
        retry_count=retry_count,
        api_key_configured=bool(stripped_key)
        and stripped_key != "your-api-key-here",
        api_key_length=len(api_key),
        api_key_prefix=api_key[:3],
        sources={
            "LLM_PROVIDER": provider_source,
            "LLM_MODEL_DEEPSEEK": deepseek_model_source,
            "LLM_TIMEOUT": timeout_source,
            "LLM_MAX_TOKENS": max_tokens_source,
            "NALUS_LEGAL_V2_LLM_MAX_TOKENS": legal_v2_max_tokens_source,
            "LLM_RETRY": retry_source,
            "LLM_API_KEY": "environment" if "LLM_API_KEY" in values else "unset",
        },
    )


def _string_value(
    env: Mapping[str, str], key: str, default: str
) -> tuple[str, str]:
    value = str(env.get(key, "")).strip()
    if value:
        return value, "environment"
    return default, "default"


def _float_value(
    env: Mapping[str, str], key: str, default: float
) -> tuple[float, str]:
    value = str(env.get(key, "")).strip()
    if not value:
        return default, "default"
    try:
        parsed = float(value)
    except ValueError:
        return default, "invalid_defaulted"
    return parsed, "environment"


def _int_value(env: Mapping[str, str], key: str, default: int) -> tuple[int, str]:
    value = str(env.get(key, "")).strip()
    if not value:
        return default, "default"
    try:
        parsed = int(value)
    except ValueError:
        return default, "invalid_defaulted"
    return parsed, "environment"


def _legal_v2_max_tokens(
    env: Mapping[str, str],
    general_max_tokens: int,
) -> tuple[int, str]:
    value = str(env.get("NALUS_LEGAL_V2_LLM_MAX_TOKENS", "")).strip()
    if value:
        try:
            parsed = int(value)
        except ValueError:
            return max(general_max_tokens, LEGAL_V2_MIN_MAX_TOKENS), "invalid_defaulted"
        return parsed, "environment"
    if general_max_tokens < LEGAL_V2_MIN_MAX_TOKENS:
        return LEGAL_V2_MIN_MAX_TOKENS, "derived_minimum_for_legal_v2"
    return general_max_tokens, "LLM_MAX_TOKENS"
