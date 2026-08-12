"""Unit tests for chunk QA v8 classifier helpers and text-loss gates."""

from __future__ import annotations

import os

from app.rag.legal_v2.ingest.chunkers.contextual_packed_v1 import ContextualPackedConfigV1
from scripts.legal_v2.run_chunking_ab_pilot_300_chunk_qa import _policy_hash
from scripts.legal_v2.run_chunking_ab_pilot_300_chunk_qa_v8 import (
    EXPECTED_B_POLICY_HASH,
    _classify_v8,
    _git_head,
    _section_boundary_class,
)


def test_b_policy_hash_frozen() -> None:
    assert _policy_hash(ContextualPackedConfigV1()) == EXPECTED_B_POLICY_HASH


def test_classify_v8_pass() -> None:
    a = {
        "total_child_chunks": 100,
        "ce_tokens_per_chunk": {"fraction_would_truncate_under_ce_max_length": 0.30},
    }
    b = {
        "total_child_chunks": 90,
        "ce_tokens_per_chunk": {"fraction_would_truncate_under_ce_max_length": 0.25},
    }
    assert (
        _classify_v8(
            a, b, {"blocked": False, "structural_regression": False}, "SECTION_BOUNDARIES_CLEAN"
        )
        == "CHUNK_QA_PASS_V8"
    )


def test_section_boundary_clean() -> None:
    assert (
        _section_boundary_class(
            header_share=0.025,
            header_suspicion_docs=1,
            docs=300,
            section_violations_b=0,
            deep_header_flags=1,
        )
        == "SECTION_BOUNDARIES_NOISY_BUT_USABLE"
    )
    assert (
        _section_boundary_class(
            header_share=0.02,
            header_suspicion_docs=0,
            docs=300,
            section_violations_b=0,
            deep_header_flags=0,
        )
        == "SECTION_BOUNDARIES_CLEAN"
    )


def test_git_head_explicit_override() -> None:
    assert _git_head("abc123def") == "abc123def"


def test_git_head_env_override(monkeypatch) -> None:
    monkeypatch.setenv("LEGAL_V2_GIT_COMMIT", "envcommit99")
    assert _git_head(None) == "envcommit99"
    monkeypatch.delenv("LEGAL_V2_GIT_COMMIT", raising=False)


def test_verification_artifact_all_false_positives() -> None:
    import json
    from pathlib import Path

    path = Path(
        "artifacts/legal_v2/chunking_ab_pilot_300_v1/chunk_qa_v8/"
        "lost_paragraph_heuristic_verification.json"
    )
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["confirmed_text_loss_a"] == 0
    assert payload["confirmed_text_loss_b"] == 0
    assert payload["heuristic_alerts_a"] == 3
    assert payload["heuristic_alerts_b"] == 2
    assert all(c["classification"] == "FALSE_POSITIVE_HEURISTIC" for c in payload["cases"])
