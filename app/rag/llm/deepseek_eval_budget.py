"""Thread-safe DeepSeek evaluation budget tracker with pre-call reservations.

Used by Legal Retrieval v2 evaluation runners. Inactive unless bound via context.
Does not change retrieval quality logic.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterator

from app.rag.llm.deepseek_pricing import (
    PRICING_TABLE_VERSION,
    DeepSeekUsage,
    calculate_reservation_cost_usd,
    calculate_usage_cost_usd,
    estimate_uncached_prompt_tokens,
)


class BudgetOperation(str, Enum):
    QUERYSPEC = "queryspec"
    FAST_VERIFIER = "fast_verifier"
    THINKING_FALLBACK = "thinking_fallback"
    OTHER = "other"


class BudgetStopReason(str, Enum):
    COST_BUDGET_EXHAUSTED = "cost_budget_exhausted"
    PROVIDER_CALL_BUDGET_EXHAUSTED = "provider_call_budget_exhausted"
    QUERYSPEC_CALL_BUDGET_EXHAUSTED = "queryspec_call_budget_exhausted"
    FAST_VERIFIER_CALL_BUDGET_EXHAUSTED = "fast_verifier_call_budget_exhausted"
    THINKING_FALLBACK_CALL_BUDGET_EXHAUSTED = "thinking_fallback_call_budget_exhausted"
    PROVIDER_USAGE_MISSING = "provider_usage_missing"


class BudgetExhaustedError(RuntimeError):
    """Raised when a pre-call reservation cannot be granted."""

    def __init__(self, stop_reason: str) -> None:
        self.stop_reason = stop_reason
        super().__init__(stop_reason)


@dataclass(frozen=True)
class BudgetLimits:
    max_cost_usd: float | None = None
    max_provider_calls: int | None = None
    max_queryspec_calls: int | None = None
    max_fast_verifier_calls: int | None = None
    max_thinking_fallback_calls: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def any_limit_set(self) -> bool:
        return any(
            value is not None
            for value in (
                self.max_cost_usd,
                self.max_provider_calls,
                self.max_queryspec_calls,
                self.max_fast_verifier_calls,
                self.max_thinking_fallback_calls,
            )
        )


@dataclass
class _Reservation:
    reservation_id: str
    operation: BudgetOperation
    model: str
    query_id: str | None
    reserved_cost_usd: float


@dataclass
class EvalBudgetTracker:
    """Concurrent-safe budget ledger for evaluation provider calls."""

    limits: BudgetLimits
    configured_model: str
    pricing_table_version: str = PRICING_TABLE_VERSION
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _actual_cost_usd: float = 0.0
    _reserved_cost_usd: float = 0.0
    _reservations: dict[str, _Reservation] = field(default_factory=dict, repr=False)
    _stop_reason: str | None = None
    _provider_calls: int = 0
    _calls_by_operation: dict[str, int] = field(
        default_factory=lambda: {
            BudgetOperation.QUERYSPEC.value: 0,
            BudgetOperation.FAST_VERIFIER.value: 0,
            BudgetOperation.THINKING_FALLBACK.value: 0,
            BudgetOperation.OTHER.value: 0,
        }
    )
    _failed_operations: int = 0
    _usage_missing_events: int = 0
    _prompt_cache_hit_tokens: int = 0
    _prompt_cache_miss_tokens: int = 0
    _completion_tokens: int = 0
    _reasoning_tokens: int = 0
    _total_tokens: int = 0
    _cost_by_operation: dict[str, float] = field(
        default_factory=lambda: {
            BudgetOperation.QUERYSPEC.value: 0.0,
            BudgetOperation.FAST_VERIFIER.value: 0.0,
            BudgetOperation.THINKING_FALLBACK.value: 0.0,
            BudgetOperation.OTHER.value: 0.0,
        }
    )
    _cost_by_model: dict[str, float] = field(default_factory=dict)
    _cost_by_query_id: dict[str, float] = field(default_factory=dict)
    _actual_models: set[str] = field(default_factory=set)
    _completed_query_ids: set[str] = field(default_factory=set)

    @property
    def stop_reason(self) -> str | None:
        with self._lock:
            return self._stop_reason

    @property
    def is_stopped(self) -> bool:
        return self.stop_reason is not None

    @property
    def actual_cost_usd(self) -> float:
        with self._lock:
            return round(self._actual_cost_usd, 10)

    @property
    def reserved_maximum_cost_usd(self) -> float:
        with self._lock:
            return round(self._reserved_cost_usd, 10)

    def reserve(
        self,
        *,
        operation: BudgetOperation,
        model: str,
        estimated_uncached_prompt_tokens: int,
        max_tokens: int,
        query_id: str | None = None,
    ) -> str:
        reserved_cost = calculate_reservation_cost_usd(
            model=model,
            estimated_uncached_prompt_tokens=estimated_uncached_prompt_tokens,
            max_tokens=max_tokens,
        )
        with self._lock:
            if self._stop_reason is not None:
                raise BudgetExhaustedError(self._stop_reason)
            stop = self._projected_call_limit_stop_locked(operation)
            if stop is not None:
                self._stop_reason = stop
                raise BudgetExhaustedError(stop)
            if self.limits.max_cost_usd is not None:
                projected = self._actual_cost_usd + self._reserved_cost_usd + reserved_cost
                if projected > float(self.limits.max_cost_usd) + 1e-12:
                    self._stop_reason = BudgetStopReason.COST_BUDGET_EXHAUSTED.value
                    raise BudgetExhaustedError(self._stop_reason)
            reservation_id = uuid.uuid4().hex
            self._reservations[reservation_id] = _Reservation(
                reservation_id=reservation_id,
                operation=operation,
                model=model,
                query_id=query_id,
                reserved_cost_usd=reserved_cost,
            )
            self._reserved_cost_usd = round(self._reserved_cost_usd + reserved_cost, 10)
            return reservation_id

    def commit_success(self, reservation_id: str, usage: DeepSeekUsage) -> float:
        cost = calculate_usage_cost_usd(usage)
        with self._lock:
            reservation = self._reservations.pop(reservation_id, None)
            if reservation is None:
                raise KeyError(f"unknown_reservation:{reservation_id}")
            self._reserved_cost_usd = round(
                max(0.0, self._reserved_cost_usd - reservation.reserved_cost_usd),
                10,
            )
            self._actual_cost_usd = round(self._actual_cost_usd + cost, 10)
            self._provider_calls += 1
            self._calls_by_operation[reservation.operation.value] = (
                self._calls_by_operation.get(reservation.operation.value, 0) + 1
            )
            self._prompt_cache_hit_tokens += usage.prompt_cache_hit_tokens
            self._prompt_cache_miss_tokens += usage.prompt_cache_miss_tokens
            self._completion_tokens += usage.completion_tokens
            self._reasoning_tokens += usage.reasoning_tokens
            self._total_tokens += usage.total_tokens
            self._actual_models.add(usage.model)
            op = reservation.operation.value
            self._cost_by_operation[op] = round(self._cost_by_operation.get(op, 0.0) + cost, 10)
            self._cost_by_model[usage.model] = round(
                self._cost_by_model.get(usage.model, 0.0) + cost,
                10,
            )
            if reservation.query_id:
                qid = reservation.query_id
                self._cost_by_query_id[qid] = round(
                    self._cost_by_query_id.get(qid, 0.0) + cost,
                    10,
                )
                self._completed_query_ids.add(qid)
            if (
                self.limits.max_cost_usd is not None
                and self._actual_cost_usd > float(self.limits.max_cost_usd) + 1e-12
                and not self._reservations
                and self._stop_reason is None
            ):
                self._stop_reason = BudgetStopReason.COST_BUDGET_EXHAUSTED.value
            stop = self._exhausted_after_success_locked()
            if stop is not None and self._stop_reason is None and not self._reservations:
                self._stop_reason = stop
            return cost

    def commit_missing_usage(self, reservation_id: str) -> float:
        """Fail closed: charge reserved maximum and stop further paid calls."""
        with self._lock:
            reservation = self._reservations.pop(reservation_id, None)
            if reservation is None:
                raise KeyError(f"unknown_reservation:{reservation_id}")
            cost = reservation.reserved_cost_usd
            self._reserved_cost_usd = round(
                max(0.0, self._reserved_cost_usd - reservation.reserved_cost_usd),
                10,
            )
            self._actual_cost_usd = round(self._actual_cost_usd + cost, 10)
            self._provider_calls += 1
            self._usage_missing_events += 1
            self._calls_by_operation[reservation.operation.value] = (
                self._calls_by_operation.get(reservation.operation.value, 0) + 1
            )
            op = reservation.operation.value
            self._cost_by_operation[op] = round(self._cost_by_operation.get(op, 0.0) + cost, 10)
            self._cost_by_model[reservation.model] = round(
                self._cost_by_model.get(reservation.model, 0.0) + cost,
                10,
            )
            if reservation.query_id:
                qid = reservation.query_id
                self._cost_by_query_id[qid] = round(
                    self._cost_by_query_id.get(qid, 0.0) + cost,
                    10,
                )
            self._stop_reason = BudgetStopReason.PROVIDER_USAGE_MISSING.value
            return cost

    def release_failure(self, reservation_id: str) -> None:
        with self._lock:
            reservation = self._reservations.pop(reservation_id, None)
            if reservation is None:
                return
            self._reserved_cost_usd = round(
                max(0.0, self._reserved_cost_usd - reservation.reserved_cost_usd),
                10,
            )
            self._failed_operations += 1

    def mark_query_completed(self, query_id: str) -> None:
        with self._lock:
            if query_id:
                self._completed_query_ids.add(query_id)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            most_expensive_query_id = None
            if self._cost_by_query_id:
                most_expensive_query_id = max(
                    self._cost_by_query_id.items(),
                    key=lambda item: item[1],
                )[0]
            completed = len(self._completed_query_ids)
            avg = (
                round(self._actual_cost_usd / completed, 10) if completed > 0 else None
            )
            budget = self.limits.max_cost_usd
            remaining = (
                None
                if budget is None
                else round(float(budget) - self._actual_cost_usd - self._reserved_cost_usd, 10)
            )
            return {
                "configured_model": self.configured_model,
                "actual_models_observed": sorted(self._actual_models),
                "pricing_table_version": self.pricing_table_version,
                "total_provider_calls": self._provider_calls,
                "queryspec_calls": self._calls_by_operation.get(
                    BudgetOperation.QUERYSPEC.value, 0
                ),
                "fast_verifier_calls": self._calls_by_operation.get(
                    BudgetOperation.FAST_VERIFIER.value, 0
                ),
                "thinking_fallback_calls": self._calls_by_operation.get(
                    BudgetOperation.THINKING_FALLBACK.value, 0
                ),
                "failed_provider_operations": self._failed_operations,
                "usage_missing_events": self._usage_missing_events,
                "prompt_cache_hit_tokens": self._prompt_cache_hit_tokens,
                "prompt_cache_miss_tokens": self._prompt_cache_miss_tokens,
                "completion_tokens": self._completion_tokens,
                "reasoning_tokens_diagnostic": self._reasoning_tokens,
                "total_tokens": self._total_tokens,
                "actual_cost_usd": round(self._actual_cost_usd, 10),
                "reserved_maximum_cost_usd": round(self._reserved_cost_usd, 10),
                "cost_by_operation_type": dict(self._cost_by_operation),
                "cost_by_model": dict(self._cost_by_model),
                "cost_by_query_id": dict(self._cost_by_query_id),
                "most_expensive_query_id": most_expensive_query_id,
                "average_cost_per_completed_query": avg,
                "configured_cost_budget_usd": budget,
                "budget_remaining_usd": remaining,
                "budget_limits": self.limits.to_dict(),
                "stop_reason": self._stop_reason,
            }

    def _projected_call_limit_stop_locked(self, operation: BudgetOperation) -> str | None:
        limits = self.limits
        if limits.max_provider_calls is not None:
            projected = self._provider_calls + len(self._reservations) + 1
            if projected > int(limits.max_provider_calls):
                return BudgetStopReason.PROVIDER_CALL_BUDGET_EXHAUSTED.value

        mapping = {
            BudgetOperation.QUERYSPEC: (
                limits.max_queryspec_calls,
                BudgetStopReason.QUERYSPEC_CALL_BUDGET_EXHAUSTED.value,
            ),
            BudgetOperation.FAST_VERIFIER: (
                limits.max_fast_verifier_calls,
                BudgetStopReason.FAST_VERIFIER_CALL_BUDGET_EXHAUSTED.value,
            ),
            BudgetOperation.THINKING_FALLBACK: (
                limits.max_thinking_fallback_calls,
                BudgetStopReason.THINKING_FALLBACK_CALL_BUDGET_EXHAUSTED.value,
            ),
        }
        limit_and_reason = mapping.get(operation)
        if limit_and_reason is None:
            return None
        limit, reason = limit_and_reason
        if limit is None:
            return None
        current_op = self._calls_by_operation.get(operation.value, 0)
        active_op = sum(
            1 for item in self._reservations.values() if item.operation == operation
        )
        if current_op + active_op + 1 > int(limit):
            return reason
        return None

    def _exhausted_after_success_locked(self) -> str | None:
        limits = self.limits
        if limits.max_provider_calls is not None and self._provider_calls >= int(
            limits.max_provider_calls
        ):
            return BudgetStopReason.PROVIDER_CALL_BUDGET_EXHAUSTED.value
        if limits.max_queryspec_calls is not None and self._calls_by_operation.get(
            BudgetOperation.QUERYSPEC.value, 0
        ) >= int(limits.max_queryspec_calls):
            return BudgetStopReason.QUERYSPEC_CALL_BUDGET_EXHAUSTED.value
        if limits.max_fast_verifier_calls is not None and self._calls_by_operation.get(
            BudgetOperation.FAST_VERIFIER.value, 0
        ) >= int(limits.max_fast_verifier_calls):
            return BudgetStopReason.FAST_VERIFIER_CALL_BUDGET_EXHAUSTED.value
        if limits.max_thinking_fallback_calls is not None and self._calls_by_operation.get(
            BudgetOperation.THINKING_FALLBACK.value, 0
        ) >= int(limits.max_thinking_fallback_calls):
            return BudgetStopReason.THINKING_FALLBACK_CALL_BUDGET_EXHAUSTED.value
        return None


_budget_tracker_var: ContextVar[EvalBudgetTracker | None] = ContextVar(
    "legal_v2_eval_budget_tracker",
    default=None,
)
_budget_operation_var: ContextVar[BudgetOperation] = ContextVar(
    "legal_v2_eval_budget_operation",
    default=BudgetOperation.OTHER,
)
_budget_query_id_var: ContextVar[str | None] = ContextVar(
    "legal_v2_eval_budget_query_id",
    default=None,
)


def get_budget_tracker() -> EvalBudgetTracker | None:
    return _budget_tracker_var.get()


def get_budget_operation() -> BudgetOperation:
    return _budget_operation_var.get()


def get_budget_query_id() -> str | None:
    return _budget_query_id_var.get()


@contextmanager
def bind_budget_tracker(tracker: EvalBudgetTracker | None) -> Iterator[None]:
    token = _budget_tracker_var.set(tracker)
    try:
        yield
    finally:
        _budget_tracker_var.reset(token)


_UNSET = object()


@contextmanager
def budget_operation_context(
    operation: BudgetOperation,
    *,
    query_id: str | None | object = _UNSET,
) -> Iterator[None]:
    op_token = _budget_operation_var.set(operation)
    q_token = None
    if query_id is not _UNSET:
        q_token = _budget_query_id_var.set(query_id)  # type: ignore[arg-type]
    try:
        yield
    finally:
        _budget_operation_var.reset(op_token)
        if q_token is not None:
            _budget_query_id_var.reset(q_token)


def build_evaluation_fingerprint(
    *,
    benchmark_checksum: str,
    runtime_policy_fingerprint: str,
    model_identity: str,
    pricing_table_version: str,
    budget_limits: BudgetLimits,
    index_identity: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "benchmark_checksum": benchmark_checksum,
        "runtime_policy_fingerprint": runtime_policy_fingerprint,
        "model_identity": model_identity,
        "pricing_table_version": pricing_table_version,
        "budget_configuration": budget_limits.to_dict(),
        "index_identity": index_identity,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    payload["fingerprint_sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return payload


def fingerprints_compatible(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    keys = (
        "benchmark_checksum",
        "runtime_policy_fingerprint",
        "model_identity",
        "pricing_table_version",
        "budget_configuration",
        "index_identity",
    )
    for key in keys:
        if expected.get(key) != actual.get(key):
            return False
    return True


def checksum_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def reserve_for_prompt(
    tracker: EvalBudgetTracker,
    *,
    prompt: str,
    model: str,
    max_tokens: int,
) -> str:
    return tracker.reserve(
        operation=get_budget_operation(),
        model=model,
        estimated_uncached_prompt_tokens=estimate_uncached_prompt_tokens(prompt),
        max_tokens=max_tokens,
        query_id=get_budget_query_id(),
    )
