from __future__ import annotations

from app.rag.llm.config import (
    DEEPSEEK_DEFAULT_MODEL,
    LEGAL_V2_MIN_MAX_TOKENS,
    effective_llm_config_from_env,
)


def test_effective_config_uses_deepseek_v4_flash_default() -> None:
    config = effective_llm_config_from_env({})

    assert config.provider == "deepseek"
    assert config.deepseek_model == DEEPSEEK_DEFAULT_MODEL
    assert config.legal_v2_max_tokens == LEGAL_V2_MIN_MAX_TOKENS
    assert config.sources["LLM_MODEL_DEEPSEEK"] == "default"


def test_effective_config_prefers_runtime_environment() -> None:
    config = effective_llm_config_from_env(
        {
            "LLM_PROVIDER": "deepseek",
            "LLM_MODEL_DEEPSEEK": "deepseek-v4-pro",
            "LLM_TIMEOUT": "30",
            "LLM_MAX_TOKENS": "800",
            "LLM_RETRY": "1",
            "LLM_API_KEY": "sk-test-redacted",
        }
    )

    assert config.deepseek_model == "deepseek-v4-pro"
    assert config.timeout_seconds == 30.0
    assert config.max_tokens == 800
    assert config.legal_v2_max_tokens == LEGAL_V2_MIN_MAX_TOKENS
    assert config.retry_count == 1
    assert config.api_key_configured is True
    assert config.api_key_length == len("sk-test-redacted")
    assert config.api_key_prefix == "sk-"


def test_effective_config_does_not_read_env_example() -> None:
    config = effective_llm_config_from_env({"LLM_API_KEY": "your-api-key-here"})

    assert config.api_key_configured is False
    assert config.sources["LLM_API_KEY"] == "environment"


def test_legal_v2_max_tokens_can_be_overridden() -> None:
    config = effective_llm_config_from_env(
        {
            "LLM_MAX_TOKENS": "800",
            "NALUS_LEGAL_V2_LLM_MAX_TOKENS": "3200",
        }
    )

    assert config.max_tokens == 800
    assert config.legal_v2_max_tokens == 3200
    assert config.sources["NALUS_LEGAL_V2_LLM_MAX_TOKENS"] == "environment"
