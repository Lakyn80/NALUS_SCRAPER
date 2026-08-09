"""Focused unit tests for chunking A/B QA helpers (no network, no models)."""

from __future__ import annotations

from scripts.legal_v2.run_chunking_ab_pilot_300_chunk_qa import (
    _classify,
    _percentile,
    _prefix_overlap_chars,
    _stats,
)


def test_percentile_and_stats_basic() -> None:
    values = [10, 20, 30, 40, 50]
    assert _percentile(values, 0.5) == 30
    stats = _stats(values)
    assert stats["count"] == 5
    assert stats["median"] == 30
    assert stats["max"] == 50


def test_prefix_overlap_chars() -> None:
    assert _prefix_overlap_chars("abcDEF", "DEFxyz") == 3
    assert _prefix_overlap_chars("abc", "xyz") == 0


def test_classify_pass_and_expansion() -> None:
    summary_a = {
        "total_child_chunks": 100,
        "ce_tokens_per_chunk": {"fraction_would_truncate_under_ce_max_length": 0.01},
    }
    summary_b = {
        "total_child_chunks": 110,
        "ce_tokens_per_chunk": {"fraction_would_truncate_under_ce_max_length": 0.02},
    }
    assert _classify(summary_a, summary_b, {"blocked": False, "structural_regression": False}) == (
        "CHUNK_QA_PASS"
    )
    summary_b2 = {
        "total_child_chunks": 160,
        "ce_tokens_per_chunk": {"fraction_would_truncate_under_ce_max_length": 0.02},
    }
    assert (
        _classify(summary_a, summary_b2, {"blocked": False, "structural_regression": False})
        == "CHUNK_B_EXCESSIVE_EXPANSION"
    )
    assert (
        _classify(summary_a, summary_b, {"blocked": True, "structural_regression": False})
        == "CHUNK_EXPERIMENT_BLOCKED"
    )
    # High absolute truncation on both sides, but B better than A → PASS.
    summary_a_hi = {
        "total_child_chunks": 100,
        "ce_tokens_per_chunk": {"fraction_would_truncate_under_ce_max_length": 0.37},
    }
    summary_b_hi = {
        "total_child_chunks": 92,
        "ce_tokens_per_chunk": {"fraction_would_truncate_under_ce_max_length": 0.31},
    }
    assert (
        _classify(summary_a_hi, summary_b_hi, {"blocked": False, "structural_regression": False})
        == "CHUNK_QA_PASS"
    )
    # B materially worse than A → TRUNCATION_RISK.
    summary_b_worse = {
        "total_child_chunks": 100,
        "ce_tokens_per_chunk": {"fraction_would_truncate_under_ce_max_length": 0.20},
    }
    summary_a_low = {
        "total_child_chunks": 100,
        "ce_tokens_per_chunk": {"fraction_would_truncate_under_ce_max_length": 0.05},
    }
    assert (
        _classify(summary_a_low, summary_b_worse, {"blocked": False, "structural_regression": False})
        == "CHUNK_B_TRUNCATION_RISK"
    )
