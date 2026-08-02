"""Compatibility re-export for Legal v2 evaluation budget helpers."""

from __future__ import annotations

from app.rag.llm.deepseek_eval_budget import (
    BudgetExhaustedError,
    BudgetLimits,
    BudgetOperation,
    BudgetStopReason,
    EvalBudgetTracker,
    bind_budget_tracker,
    budget_operation_context,
    build_evaluation_fingerprint,
    checksum_text,
    fingerprints_compatible,
    get_budget_operation,
    get_budget_query_id,
    get_budget_tracker,
    reserve_for_prompt,
)
from app.rag.llm.deepseek_pricing import PRICING_TABLE_VERSION

__all__ = [
    "PRICING_TABLE_VERSION",
    "BudgetExhaustedError",
    "BudgetLimits",
    "BudgetOperation",
    "BudgetStopReason",
    "EvalBudgetTracker",
    "bind_budget_tracker",
    "budget_operation_context",
    "build_evaluation_fingerprint",
    "checksum_text",
    "fingerprints_compatible",
    "get_budget_operation",
    "get_budget_query_id",
    "get_budget_tracker",
    "reserve_for_prompt",
]
