"""Deterministic tests for DeepSeek pricing and evaluation budget guards.

No real provider HTTP calls.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.rag.llm.deepseek_eval_budget import (
    BudgetExhaustedError,
    BudgetLimits,
    BudgetOperation,
    BudgetStopReason,
    EvalBudgetTracker,
    bind_budget_tracker,
    budget_operation_context,
    build_evaluation_fingerprint,
    fingerprints_compatible,
)
from app.rag.llm.deepseek_pricing import (
    PRICING_TABLE_VERSION,
    DeepSeekUsage,
    calculate_reservation_cost_usd,
    calculate_usage_cost_usd,
    parse_deepseek_usage,
)
from app.rag.llm.providers.deepseek import DeepSeekTextLLM


def _usage(
    *,
    model: str = "deepseek-v4-flash",
    hit: int = 0,
    miss: int = 0,
    completion: int = 0,
    reasoning: int = 0,
) -> DeepSeekUsage:
    prompt = hit + miss
    return DeepSeekUsage(
        model=model,
        prompt_tokens=prompt,
        prompt_cache_hit_tokens=hit,
        prompt_cache_miss_tokens=miss,
        completion_tokens=completion,
        reasoning_tokens=reasoning,
        total_tokens=prompt + completion,
    )


def test_flash_cache_hit_pricing() -> None:
    cost = calculate_usage_cost_usd(_usage(hit=1_000_000, miss=0, completion=0))
    assert cost == pytest.approx(0.0028)


def test_flash_cache_miss_pricing() -> None:
    cost = calculate_usage_cost_usd(_usage(hit=0, miss=1_000_000, completion=0))
    assert cost == pytest.approx(0.14)


def test_flash_output_pricing() -> None:
    cost = calculate_usage_cost_usd(_usage(hit=0, miss=0, completion=1_000_000))
    assert cost == pytest.approx(0.28)


def test_pro_pricing() -> None:
    cost = calculate_usage_cost_usd(
        _usage(model="deepseek-v4-pro", hit=1_000_000, miss=1_000_000, completion=1_000_000)
    )
    assert cost == pytest.approx(0.003625 + 0.435 + 0.87)


def test_mixed_cache_hit_and_miss_pricing() -> None:
    cost = calculate_usage_cost_usd(_usage(hit=500_000, miss=500_000, completion=0))
    assert cost == pytest.approx((0.0028 * 0.5) + (0.14 * 0.5))


def test_reasoning_tokens_not_double_counted() -> None:
    with_reasoning = calculate_usage_cost_usd(
        _usage(hit=0, miss=0, completion=1000, reasoning=800)
    )
    without = calculate_usage_cost_usd(_usage(hit=0, miss=0, completion=1000, reasoning=0))
    assert with_reasoning == without


def test_request_blocked_before_budget_overflow() -> None:
    tracker = EvalBudgetTracker(
        limits=BudgetLimits(max_cost_usd=0.0001),
        configured_model="deepseek-v4-flash",
    )
    with pytest.raises(BudgetExhaustedError) as exc:
        tracker.reserve(
            operation=BudgetOperation.QUERYSPEC,
            model="deepseek-v4-flash",
            estimated_uncached_prompt_tokens=100_000,
            max_tokens=8000,
            query_id="uq_001",
        )
    assert exc.value.stop_reason == BudgetStopReason.COST_BUDGET_EXHAUSTED.value
    assert tracker.reserved_maximum_cost_usd == 0.0


def test_concurrent_reservations_cannot_overspend() -> None:
    tracker = EvalBudgetTracker(
        limits=BudgetLimits(max_cost_usd=0.001),
        configured_model="deepseek-v4-flash",
    )
    # One reservation max ~ 0.00014 (1000 miss) + 0.00028 (1000 out) = 0.00042
    # Two would exceed 0.001 if each reserves ~0.00056 with larger tokens.
    barrier = threading.Barrier(4)
    outcomes: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        try:
            reservation_id = tracker.reserve(
                operation=BudgetOperation.FAST_VERIFIER,
                model="deepseek-v4-flash",
                estimated_uncached_prompt_tokens=2_000,
                max_tokens=1_000,
                query_id="q",
            )
            with lock:
                outcomes.append(f"ok:{reservation_id}")
        except BudgetExhaustedError as exc:
            with lock:
                outcomes.append(exc.stop_reason)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: worker(), range(4)))

    ok = [item for item in outcomes if item.startswith("ok:")]
    blocked = [item for item in outcomes if item == BudgetStopReason.COST_BUDGET_EXHAUSTED.value]
    assert len(ok) + len(blocked) == 4
    assert len(ok) >= 1
    assert len(blocked) >= 1
    # Active reservations must fit under budget.
    assert tracker.actual_cost_usd + tracker.reserved_maximum_cost_usd <= 0.001 + 1e-12


def test_reservation_release_after_success() -> None:
    tracker = EvalBudgetTracker(
        limits=BudgetLimits(max_cost_usd=1.0),
        configured_model="deepseek-v4-flash",
    )
    reservation_id = tracker.reserve(
        operation=BudgetOperation.QUERYSPEC,
        model="deepseek-v4-flash",
        estimated_uncached_prompt_tokens=1000,
        max_tokens=100,
        query_id="uq_x",
    )
    assert tracker.reserved_maximum_cost_usd > 0
    cost = tracker.commit_success(reservation_id, _usage(hit=1000, miss=0, completion=10))
    assert cost > 0
    assert tracker.reserved_maximum_cost_usd == 0.0
    assert tracker.actual_cost_usd == cost


def test_reservation_release_after_failure() -> None:
    tracker = EvalBudgetTracker(
        limits=BudgetLimits(max_cost_usd=1.0),
        configured_model="deepseek-v4-flash",
    )
    reservation_id = tracker.reserve(
        operation=BudgetOperation.FAST_VERIFIER,
        model="deepseek-v4-flash",
        estimated_uncached_prompt_tokens=1000,
        max_tokens=100,
    )
    tracker.release_failure(reservation_id)
    assert tracker.reserved_maximum_cost_usd == 0.0
    assert tracker.actual_cost_usd == 0.0
    assert tracker.summary()["failed_provider_operations"] == 1


def test_missing_usage_fail_closed_charges_reserved_and_stops() -> None:
    tracker = EvalBudgetTracker(
        limits=BudgetLimits(max_cost_usd=1.0),
        configured_model="deepseek-v4-flash",
    )
    reservation_id = tracker.reserve(
        operation=BudgetOperation.THINKING_FALLBACK,
        model="deepseek-v4-flash",
        estimated_uncached_prompt_tokens=500,
        max_tokens=100,
        query_id="uq_y",
    )
    reserved = tracker.reserved_maximum_cost_usd
    charged = tracker.commit_missing_usage(reservation_id)
    assert charged == reserved
    assert tracker.actual_cost_usd == charged
    assert tracker.stop_reason == BudgetStopReason.PROVIDER_USAGE_MISSING.value
    with pytest.raises(BudgetExhaustedError):
        tracker.reserve(
            operation=BudgetOperation.QUERYSPEC,
            model="deepseek-v4-flash",
            estimated_uncached_prompt_tokens=1,
            max_tokens=1,
        )


@pytest.mark.parametrize(
    ("limit_field", "operation", "reason"),
    [
        ("max_provider_calls", BudgetOperation.QUERYSPEC, BudgetStopReason.PROVIDER_CALL_BUDGET_EXHAUSTED),
        ("max_queryspec_calls", BudgetOperation.QUERYSPEC, BudgetStopReason.QUERYSPEC_CALL_BUDGET_EXHAUSTED),
        (
            "max_fast_verifier_calls",
            BudgetOperation.FAST_VERIFIER,
            BudgetStopReason.FAST_VERIFIER_CALL_BUDGET_EXHAUSTED,
        ),
        (
            "max_thinking_fallback_calls",
            BudgetOperation.THINKING_FALLBACK,
            BudgetStopReason.THINKING_FALLBACK_CALL_BUDGET_EXHAUSTED,
        ),
    ],
)
def test_provider_call_limits(limit_field: str, operation: BudgetOperation, reason: BudgetStopReason) -> None:
    limits = BudgetLimits(**{limit_field: 1})
    tracker = EvalBudgetTracker(limits=limits, configured_model="deepseek-v4-flash")
    reservation_id = tracker.reserve(
        operation=operation,
        model="deepseek-v4-flash",
        estimated_uncached_prompt_tokens=10,
        max_tokens=10,
    )
    tracker.commit_success(reservation_id, _usage(hit=0, miss=10, completion=1))
    with pytest.raises(BudgetExhaustedError) as exc:
        tracker.reserve(
            operation=operation,
            model="deepseek-v4-flash",
            estimated_uncached_prompt_tokens=10,
            max_tokens=10,
        )
    assert exc.value.stop_reason == reason.value


def test_resume_fingerprint_validation() -> None:
    limits = BudgetLimits(max_cost_usd=1.0, max_provider_calls=10)
    fp = build_evaluation_fingerprint(
        benchmark_checksum="abc",
        runtime_policy_fingerprint="pol",
        model_identity="deepseek-v4-flash",
        pricing_table_version=PRICING_TABLE_VERSION,
        budget_limits=limits,
        index_identity={"qdrant_collection": "c", "bm25_index_id": "b"},
    )
    assert fingerprints_compatible(fp, dict(fp))
    other = dict(fp)
    other["budget_configuration"] = BudgetLimits(max_cost_usd=2.0).to_dict()
    assert not fingerprints_compatible(fp, other)


def test_partial_artifact_persistence(tmp_path: Path) -> None:
    from scripts.legal_v2 import evaluate_thinking_hybrid_smoke as smoke

    tracker = EvalBudgetTracker(
        limits=BudgetLimits(max_cost_usd=0.5),
        configured_model="deepseek-v4-flash",
    )
    reservation_id = tracker.reserve(
        operation=BudgetOperation.QUERYSPEC,
        model="deepseek-v4-flash",
        estimated_uncached_prompt_tokens=100,
        max_tokens=50,
        query_id="uq_001",
    )
    tracker.commit_success(reservation_id, _usage(hit=0, miss=100, completion=20))
    fp = build_evaluation_fingerprint(
        benchmark_checksum="x",
        runtime_policy_fingerprint="y",
        model_identity="deepseek-v4-flash",
        pricing_table_version=PRICING_TABLE_VERSION,
        budget_limits=tracker.limits,
        index_identity={"qdrant_collection": "c", "bm25_index_id": "b"},
    )
    json_path = tmp_path / "partial.json"
    md_path = tmp_path / "partial.md"
    artifact = smoke._write_checkpoint(
        output_dir=tmp_path,
        json_path=json_path,
        md_path=md_path,
        rows=[
            {
                "id": "uq_001",
                "queryspec_schema_valid": True,
                "fast_verifier_results": [],
                "thinking_verifier_results": [],
                "fast_verifier_calls": 0,
                "thinking_fallback_calls": 0,
                "queryspec_calls": 1,
                "false_approvals": 0,
                "false_rejections": 0,
                "prompt_injection_success": 0,
                "wrong_index_identity": 0,
                "status": "budget_stopped",
                "budget_stopped": True,
                "total_latency_ms": 10.0,
            }
        ],
        query_limit=16,
        stop_reason=BudgetStopReason.COST_BUDGET_EXHAUSTED.value,
        started=0.0,
        resume_kept=0,
        pending_total=16,
        budget_tracker=tracker,
        evaluation_fingerprint=fp,
        budget_limits=tracker.limits,
    )
    assert json_path.exists() and md_path.exists()
    assert artifact["summary"]["stop_reason"] == BudgetStopReason.COST_BUDGET_EXHAUSTED.value
    assert artifact["summary"]["budget_limited"] is True
    assert artifact["evaluation_fingerprint"]["pricing_table_version"] == PRICING_TABLE_VERSION
    assert artifact["summary"]["actual_cost_usd"] > 0


def test_deepseek_text_llm_records_usage_without_real_http() -> None:
    body = {
        "model": "deepseek-v4-flash",
        "choices": [{"message": {"content": "{\"ok\": true}"}}],
        "usage": {
            "prompt_tokens": 100,
            "prompt_cache_hit_tokens": 80,
            "prompt_cache_miss_tokens": 20,
            "completion_tokens": 10,
            "total_tokens": 110,
            "completion_tokens_details": {"reasoning_tokens": 4},
        },
    }
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = body
    mock_resp.text = json.dumps(body)
    mock_resp.headers = {}
    mock_resp.raise_for_status.return_value = None
    mock_instance = MagicMock()
    mock_instance.post.return_value = mock_resp
    mock_class = MagicMock(return_value=mock_instance)

    tracker = EvalBudgetTracker(
        limits=BudgetLimits(max_cost_usd=1.0),
        configured_model="deepseek-v4-flash",
    )
    with patch("httpx.Client", mock_class):
        with bind_budget_tracker(tracker):
            with budget_operation_context(BudgetOperation.QUERYSPEC, query_id="uq_t"):
                llm = DeepSeekTextLLM(api_key="k", max_retries=0, raise_on_error=True)
                text = llm.generate_text('{"hello":"world"}')
    assert text
    assert llm.last_meta is not None
    assert llm.last_meta["usage"]["prompt_cache_hit_tokens"] == 80
    assert llm.last_meta["usage"]["reasoning_tokens"] == 4
    assert tracker.actual_cost_usd > 0
    assert tracker.summary()["queryspec_calls"] == 1


def test_parse_usage_requires_fields() -> None:
    assert parse_deepseek_usage({"model": "deepseek-v4-flash"}, configured_model="deepseek-v4-flash") is None


def test_reservation_cost_uses_uncached_miss_and_max_tokens() -> None:
    cost = calculate_reservation_cost_usd(
        model="deepseek-v4-flash",
        estimated_uncached_prompt_tokens=1_000_000,
        max_tokens=1_000_000,
    )
    assert cost == pytest.approx(0.14 + 0.28)


def test_pricing_table_version_constant() -> None:
    assert PRICING_TABLE_VERSION == "deepseek_v4_2026_07_31"
