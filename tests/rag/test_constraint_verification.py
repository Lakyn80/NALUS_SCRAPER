from app.rag.retrieval.constraint_config import ConstraintRetrievalConfig
from app.rag.retrieval.constraint_models import DocumentDecisionStatus
from app.rag.retrieval.constraint_verification import verify_document_constraints
from app.rag.retrieval.document_retrieval import DocumentSearchResult
from app.rag.retrieval.full_document import (
    FullDocumentChunk,
    FullDocumentDiagnostics,
    FullDocumentResult,
)
from app.rag.retrieval.structured_query import interpret_structured_query


def _candidate(document_id: str = "ECLI:CZ:US:2024:1.US.1.24.1") -> DocumentSearchResult:
    return DocumentSearchResult(
        document_id=document_id,
        score=0.8,
        best_passages=[],
        metadata={"document_id": document_id, "court_name": "Ústavní soud"},
        candidate_chunk_count=1,
        best_chunk_score=0.8,
    )


def _document(
    text: str,
    *,
    document_id: str = "ECLI:CZ:US:2024:1.US.1.24.1",
) -> FullDocumentResult:
    chunks = [
        FullDocumentChunk(
            chunk_id="chunk-0",
            chunk_index=0,
            text=text,
            metadata={"document_id": document_id, "chunk_index": 0},
        )
    ]
    return FullDocumentResult(
        document_id=document_id,
        metadata={"document_id": document_id, "court_name": "Ústavní soud"},
        full_text=text,
        chunks=chunks,
        source_url=None,
        provenance_status="overeno",
        full_text_availability_status="available",
        diagnostics=FullDocumentDiagnostics(
            collection_name="test",
            chunk_count=len(chunks),
            missing_chunk_indexes=[],
            duplicate_chunk_indexes=[],
            all_chunks_have_index=True,
            reconstruction_method="test",
            max_chunks=2000,
        ),
    )


def test_verifies_citizenship_grant_for_russian_applicant() -> None:
    structured = interpret_structured_query("udělení českého občanství ruskému občanu")
    result = verify_document_constraints(
        structured_query=structured,
        candidate=_candidate(),
        document=_document(
            "Stěžovatel je státní občan Ruské federace. "
            "Podal žádost o udělení státního občanství České republiky."
        ),
        config=ConstraintRetrievalConfig(),
    )

    assert result.decision_status == DocumentDecisionStatus.VERIFIED_MATCH
    assert {item.status.value for item in result.constraint_results} == {"matched"}
    assert result.supporting_passages


def test_rejects_citizenship_document_when_requested_nationality_not_proven() -> None:
    structured = interpret_structured_query("udělení českého občanství ruskému občanu")
    result = verify_document_constraints(
        structured_query=structured,
        candidate=_candidate(),
        document=_document(
            "Stěžovatel je občan Ukrajiny. Řízení se týká žádosti o udělení státního občanství České republiky."
        ),
        config=ConstraintRetrievalConfig(),
    )

    assert result.decision_status == DocumentDecisionStatus.EXCLUDED_HARD_MISMATCH


def test_rejects_partial_citizenship_topic_without_grant_event() -> None:
    structured = interpret_structured_query("udělení českého občanství ruskému občanu")
    result = verify_document_constraints(
        structured_query=structured,
        candidate=_candidate(),
        document=_document(
            "Stěžovatel namítal otázku osvědčení o státním občanství. Text neřeší žádost o udělení."
        ),
        config=ConstraintRetrievalConfig(),
    )

    assert result.decision_status in {
        DocumentDecisionStatus.EXCLUDED_NOT_PROVEN,
        DocumentDecisionStatus.EXCLUDED_HARD_MISMATCH,
    }


def test_verifies_child_abduction_destination_and_parent_role() -> None:
    structured = interpret_structured_query("mezinárodní únos dítěte matkou do Ruska")
    result = verify_document_constraints(
        structured_query=structured,
        candidate=_candidate(),
        document=_document(
            "Věc se týká mezinárodního únosu dítěte. Matka dítě neoprávněně přemístila do Ruska "
            "a soud řešil navrácení dítěte podle Haagské úmluvy."
        ),
        config=ConstraintRetrievalConfig(),
    )

    assert result.decision_status == DocumentDecisionStatus.VERIFIED_MATCH


def test_missing_full_text_is_insufficient_evidence() -> None:
    structured = interpret_structured_query("udělení českého občanství ruskému občanu")
    result = verify_document_constraints(
        structured_query=structured,
        candidate=_candidate("DOC-1"),
        document=None,
        config=ConstraintRetrievalConfig(),
    )

    assert result.decision_status == DocumentDecisionStatus.EXCLUDED_INSUFFICIENT_EVIDENCE
