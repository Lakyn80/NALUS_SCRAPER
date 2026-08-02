"""DeepSeek usage parsing and USD pricing for evaluation budget accounting.

Pricing table version: deepseek_v4_2026_07_31.
Does not log prompts, responses, reasoning content, or API keys.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

PRICING_TABLE_VERSION = "deepseek_v4_2026_07_31"

# USD per 1M tokens
_FLASH_CACHE_HIT = 0.0028
_FLASH_CACHE_MISS = 0.14
_FLASH_OUTPUT = 0.28
_PRO_CACHE_HIT = 0.003625
_PRO_CACHE_MISS = 0.435
_PRO_OUTPUT = 0.87

_MODEL_RATES: dict[str, tuple[float, float, float]] = {
    "deepseek-v4-flash": (_FLASH_CACHE_HIT, _FLASH_CACHE_MISS, _FLASH_OUTPUT),
    "deepseek-v4-pro": (_PRO_CACHE_HIT, _PRO_CACHE_MISS, _PRO_OUTPUT),
}


@dataclass(frozen=True)
class DeepSeekUsage:
    """Normalized provider usage from a successful DeepSeek response."""

    model: str
    prompt_tokens: int
    prompt_cache_hit_tokens: int
    prompt_cache_miss_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UnknownDeepSeekModelError(ValueError):
    """Raised when pricing cannot be resolved for a model id."""


def normalize_deepseek_model(model: str | None) -> str:
    text = str(model or "").strip().lower()
    if not text:
        raise UnknownDeepSeekModelError("empty_model")
    if "v4-pro" in text or text.endswith("-pro") or "/deepseek-v4-pro" in text:
        return "deepseek-v4-pro"
    if "v4-flash" in text or text.endswith("-flash") or "/deepseek-v4-flash" in text:
        return "deepseek-v4-flash"
    if text in _MODEL_RATES:
        return text
    raise UnknownDeepSeekModelError(text)


def rates_for_model(model: str) -> tuple[float, float, float]:
    key = normalize_deepseek_model(model)
    return _MODEL_RATES[key]


def estimate_uncached_prompt_tokens(prompt: str) -> int:
    """Conservative character-based estimate; treated as cache-miss for reservation."""
    length = len(prompt or "")
    return max(1, (length + 3) // 4)


def parse_deepseek_usage(body: dict[str, Any], *, configured_model: str) -> DeepSeekUsage | None:
    """Extract usage from a provider response body. Returns None when incomplete."""
    if not isinstance(body, dict):
        return None
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None
    try:
        prompt_raw = usage.get("prompt_tokens")
        completion_raw = usage.get("completion_tokens")
        total_raw = usage.get("total_tokens")
        if prompt_raw is None or completion_raw is None or total_raw is None:
            return None
        prompt_tokens = int(prompt_raw)
        completion_tokens = int(completion_raw)
        total_tokens = int(total_raw)
    except (TypeError, ValueError):
        return None
    if prompt_tokens < 0 or completion_tokens < 0 or total_tokens < 0:
        return None

    hit = _optional_nonneg_int(usage.get("prompt_cache_hit_tokens"))
    miss = _optional_nonneg_int(usage.get("prompt_cache_miss_tokens"))
    if hit is None and miss is None:
        # Older shape: treat all prompt tokens as cache miss (conservative accounting).
        hit = 0
        miss = prompt_tokens
    elif hit is None:
        hit = max(0, prompt_tokens - int(miss or 0))
    elif miss is None:
        miss = max(0, prompt_tokens - int(hit or 0))
    assert hit is not None and miss is not None

    details = usage.get("completion_tokens_details")
    reasoning = 0
    if isinstance(details, dict):
        reasoning = _optional_nonneg_int(details.get("reasoning_tokens")) or 0
    else:
        reasoning = _optional_nonneg_int(usage.get("reasoning_tokens")) or 0
    # Diagnostic only; never add on top of completion_tokens for billing.
    reasoning = min(reasoning, completion_tokens)

    actual_model = str(body.get("model") or configured_model or "").strip()
    if not actual_model:
        return None
    try:
        normalize_deepseek_model(actual_model)
    except UnknownDeepSeekModelError:
        return None

    return DeepSeekUsage(
        model=actual_model,
        prompt_tokens=prompt_tokens,
        prompt_cache_hit_tokens=int(hit),
        prompt_cache_miss_tokens=int(miss),
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning,
        total_tokens=total_tokens,
    )


def calculate_usage_cost_usd(usage: DeepSeekUsage) -> float:
    """Bill completion_tokens once; reasoning_tokens are not added again."""
    hit_rate, miss_rate, output_rate = rates_for_model(usage.model)
    cost = (
        (usage.prompt_cache_hit_tokens / 1_000_000.0) * hit_rate
        + (usage.prompt_cache_miss_tokens / 1_000_000.0) * miss_rate
        + (usage.completion_tokens / 1_000_000.0) * output_rate
    )
    return round(cost, 10)


def calculate_reservation_cost_usd(
    *,
    model: str,
    estimated_uncached_prompt_tokens: int,
    max_tokens: int,
) -> float:
    """Conservative pre-call maximum: all prompt tokens as cache-miss + full max_tokens output."""
    _, miss_rate, output_rate = rates_for_model(model)
    prompt_tokens = max(0, int(estimated_uncached_prompt_tokens))
    completion_cap = max(0, int(max_tokens))
    cost = (prompt_tokens / 1_000_000.0) * miss_rate + (completion_cap / 1_000_000.0) * output_rate
    return round(cost, 10)


def _optional_nonneg_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return None
    return number
