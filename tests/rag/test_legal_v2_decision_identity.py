from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.rag.legal_v2.benchmark.case_similarity_eval import (
    FAILURE_EXPECTED_ECLI_MISSING_FROM_INDEX,
    FAILURE_MISSING_VERIFIED_ECLI_IN_BENCHMARK,
    FAILURE_RETRIEVED_RESULT_MISSING_ECLI,
    corpus_presence_summary,
    evaluate_ranked_documents,
)
from app.rag.legal_v2.benchmark.case_similarity_golden import (
    HardNegativeRationale,
    load_case_similarity_golden_jsonl,
)
from app.rag.legal_v2.benchmark.case_similarity_identity import (
    load_case_similarity_identity_map,
)
from app.rag.legal_v2.identity import (
    DecisionIdentityError,
    IDENTITY_STATUS_BLOCKED_MISSING_ECLI,
    IDENTITY_STATUS_VERIFIED,
    is_valid_ecli,
    production_identity_fields,
    resolve_production_document_id,
    validate_decision_identity,
)
from app.rag.legal_v2.ingest.indexing import payload_for_child_chunk
from app.rag.legal_v2.ingest.chunking import RetrievalChildChunk
from app.rag.legal_v2.models import SectionType
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PILOT_PATH = PROJECT_ROOT / "benchmarks" / "legal_v2" / "case_similarity_golden_v1_pilot.jsonl"
IDENTITY_PATH = PROJECT_ROOT / "benchmarks" / "legal_v2" / "case_similarity_document_identity_v1.json"


def test_valid_verified_ecli() -> None:
    ecli = "ECLI:CZ:US:2024:3.US.3203.24.1"
    assert is_valid_ecli(ecli)
    assert validate_decision_identity(ecli=ecli, canonical_document_id=ecli) == ecli


def test_plenary_ecli_with_hyphen_is_valid() -> None:
    assert is_valid_ecli("ECLI:CZ:US:2025:Pl.US-st.61.25.1")


def test_missing_ecli_rejected() -> None:
    with pytest.raises(DecisionIdentityError, match="required"):
        validate_decision_identity(ecli=None, canonical_document_id=None)


def test_malformed_ecli_rejected() -> None:
    with pytest.raises(DecisionIdentityError, match="malformed"):
        validate_decision_identity(ecli="not-an-ecli", canonical_document_id="not-an-ecli")


def test_canonical_must_equal_ecli() -> None:
    with pytest.raises(DecisionIdentityError, match="must equal"):
        validate_decision_identity(
            ecli="ECLI:CZ:US:2024:3.US.3203.24.1",
            canonical_document_id="ECLI:CZ:US:2024:1.US.3299.24.1",
        )


def test_production_identity_keeps_source_doc_secondary() -> None:
    fields = production_identity_fields(
        ecli="ECLI:CZ:US:2024:3.US.3203.24.1",
        source_document_id="doc-0a90125eb71851b4",
    )
    assert fields["document_id"] == "ECLI:CZ:US:2024:3.US.3203.24.1"
    assert fields["canonical_document_id"] == fields["ecli"]
    assert fields["source_document_id"] == "doc-0a90125eb71851b4"


def test_identity_map_no_source_to_multiple_eclis() -> None:
    mapping = load_case_similarity_identity_map(IDENTITY_PATH)
    assert len(mapping) == len({row["source_document_id"] for row in mapping.values()})


def test_identity_map_no_conflicting_ecli_aliases() -> None:
    mapping = load_case_similarity_identity_map(IDENTITY_PATH)
    by_ecli: dict[str, str] = {}
    for source_id, row in mapping.items():
        if row.get("identity_status") != IDENTITY_STATUS_VERIFIED:
            continue
        ecli = row["ecli"].casefold()
        assert ecli not in by_ecli
        by_ecli[ecli] = source_id


def test_pilot_primaries_and_references_carry_identity() -> None:
    items = load_case_similarity_golden_jsonl(PILOT_PATH)
    assert len(items) == 20
    for item in items:
        assert item.source_document_id.startswith("doc-")
        if item.primary_identity_status == IDENTITY_STATUS_VERIFIED:
            assert item.expected_primary_ecli
            assert item.expected_primary_canonical_document_id == item.expected_primary_ecli
        else:
            assert item.primary_identity_status == IDENTITY_STATUS_BLOCKED_MISSING_ECLI
            assert item.expected_primary_ecli is None
        for row in item.accepted_alternative_rationales:
            if row.identity_status == IDENTITY_STATUS_VERIFIED:
                assert row.ecli and row.canonical_document_id == row.ecli
        for row in item.hard_negative_rationales:
            if row.identity_status == IDENTITY_STATUS_VERIFIED:
                assert row.ecli and row.canonical_document_id == row.ecli


def test_chunk_payload_uses_ecli_as_document_id() -> None:
    ecli = "ECLI:CZ:US:2024:3.US.3203.24.1"
    chunk = RetrievalChildChunk(
        chunk_id=f"{ecli}::p:00008",
        document_id=ecli,
        chunk_index=8,
        text="odůvodnění",
        token_count=2,
        paragraph_ids=["p1"],
        paragraph_texts={"p1": "odůvodnění"},
        paragraph_original_texts={"p1": "odůvodnění"},
        source_spans=[],
        section_type=SectionType.COURT_REASONING,
        start_offset=0,
        end_offset=10,
        source_order=8,
        heading_context=[],
        metadata={
            "ecli": ecli,
            "canonical_document_id": ecli,
            "source_document_id": "doc-0a90125eb71851b4",
            "court": "constitutional_court",
            "case_reference": "III.ÚS 3203/24",
            "decision_date": "2024-12-20",
        },
    )
    payload = payload_for_child_chunk(chunk)
    assert payload["document_id"] == ecli
    assert payload["canonical_document_id"] == ecli
    assert payload["ecli"] == ecli
    assert payload["source_document_id"] == "doc-0a90125eb71851b4"


def test_resolve_production_document_id_prefers_ecli() -> None:
    assert (
        resolve_production_document_id(
            {
                "document_id": "doc-aaaa",
                "source_document_id": "doc-aaaa",
                "ecli": "ECLI:CZ:US:2024:3.US.3203.24.1",
            }
        )
        == "ECLI:CZ:US:2024:3.US.3203.24.1"
    )


def test_aggregation_key_groups_same_ecli() -> None:
    from app.rag.legal_v2.identity import ecli_key

    left = "ECLI:CZ:US:2024:3.US.3203.24.1"
    right = "ecli:cz:us:2024:3.us.3203.24.1"
    assert ecli_key(left) == ecli_key(right)


def test_evaluator_compares_ecli_not_doc_star() -> None:
    row = evaluate_ranked_documents(
        query_id="q1",
        query="q",
        query_style="client_narrative",
        difficulty="easy",
        expected_primary_document_id="ECLI:CZ:US:2024:3.US.3203.24.1",
        accepted_alternative_document_ids=[],
        hard_negative_document_ids=["ECLI:CZ:US:2024:1.US.3299.24.1"],
        hard_negative_evaluable=True,
        hard_negative_blocker=None,
        ranked_document_ids=[
            "ECLI:CZ:US:2024:3.US.3203.24.1",
            "ECLI:CZ:US:2024:1.US.3299.24.1",
        ],
        expected_primary_source_document_id="doc-0a90125eb71851b4",
        expected_primary_ecli="ECLI:CZ:US:2024:3.US.3203.24.1",
    )
    assert row.primary_rank == 1
    assert row.hit_at_1 is True
    assert row.retrieved_eclis[0].startswith("ECLI:")
    assert row.expected_primary_source_document_id == "doc-0a90125eb71851b4"


def test_evaluator_succeeds_without_source_doc_in_qdrant_results() -> None:
    row = evaluate_ranked_documents(
        query_id="q1",
        query="q",
        query_style="client_narrative",
        difficulty="easy",
        expected_primary_document_id="ECLI:CZ:US:2024:3.US.3203.24.1",
        accepted_alternative_document_ids=[],
        hard_negative_document_ids=[],
        hard_negative_evaluable=True,
        hard_negative_blocker=None,
        ranked_document_ids=["ECLI:CZ:US:2024:3.US.3203.24.1"],
        expected_primary_ecli="ECLI:CZ:US:2024:3.US.3203.24.1",
    )
    assert row.hit_at_1 is True


def test_compatibility_uses_ecli_presence() -> None:
    items = load_case_similarity_golden_jsonl(PILOT_PATH)
    present = {
        item.expected_primary_ecli
        for item in items
        if item.expected_primary_ecli
    }
    summary = corpus_presence_summary(items=items, present_document_ids=present)
    assert summary["primary_documents_present"] == len(present)
    assert summary["primary_documents_missing"] == 20 - len(present)


def test_golden_schema_rejects_canonical_mismatch() -> None:
    from tests.rag import test_legal_v2_case_similarity_eval as eval_tests

    with pytest.raises(ValidationError):
        eval_tests._minimal_item(
            expected_primary_ecli="ECLI:CZ:US:2024:3.US.3203.24.1",
            expected_primary_canonical_document_id="ECLI:CZ:US:2024:1.US.3299.24.1",
        )


def test_hard_negative_same_ecli_as_primary_rejected() -> None:
    from tests.rag import test_legal_v2_case_similarity_eval as eval_tests

    with pytest.raises(ValidationError):
        eval_tests._minimal_item(
            hard_negative_rationales=[
                HardNegativeRationale(
                    document_id="doc-bbbbbbbbbbbbbbbb",
                    looks_similar_because="looks similar",
                    materially_incorrect_because="wrong",
                    ecli="ECLI:CZ:US:2024:3.US.3203.24.1",
                    canonical_document_id="ECLI:CZ:US:2024:3.US.3203.24.1",
                    identity_status="verified",
                )
            ]
        )


def test_failure_constants_exist() -> None:
    assert FAILURE_MISSING_VERIFIED_ECLI_IN_BENCHMARK
    assert FAILURE_EXPECTED_ECLI_MISSING_FROM_INDEX
    assert FAILURE_RETRIEVED_RESULT_MISSING_ECLI
