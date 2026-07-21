from prometheus_client import generate_latest

from app.observability.constraint_retrieval_metrics import record_constraint_retrieval_metrics
from app.rag.retrieval.constraint_models import (
    ConstraintCategory,
    ConstraintEvidence,
    ConstraintRetrievalDiagnostics,
    ConstraintRetrievalResult,
    ConstraintVerificationResult,
    ConstraintVerificationStatus,
    DocumentDecisionStatus,
    InterpretationStatus,
    StructuredQuery,
    VerificationMethod,
    VerifiedDocument,
)


def test_constraint_retrieval_metrics_use_only_bounded_labels() -> None:
    result = ConstraintRetrievalResult(
        structured_query=StructuredQuery(
            intent="constraint_aware_document_retrieval",
            status=InterpretationStatus.STRUCTURED,
            constraints=[],
        ),
        verified_documents=[
            VerifiedDocument(
                document_id="ECLI:CZ:US:2024:SECRET.1",
                score=0.9,
                decision_status=DocumentDecisionStatus.VERIFIED_MATCH,
                constraint_results=[
                    ConstraintVerificationResult(
                        constraint_id="nationality:applicant:ru",
                        category=ConstraintCategory.NATIONALITY,
                        status=ConstraintVerificationStatus.MATCHED,
                        required_value="RU",
                        detected_value="RU",
                        evidence=[
                            ConstraintEvidence(
                                document_id="ECLI:CZ:US:2024:SECRET.1",
                                chunk_id="chunk-secret",
                                quote="raw query and ecli must not become metric labels",
                            )
                        ],
                        verification_method=VerificationMethod.DETERMINISTIC_EVIDENCE,
                        confidence=0.9,
                        reason="matched",
                    )
                ],
                supporting_passages=[],
            )
        ],
        rejected_documents=[],
        diagnostics=ConstraintRetrievalDiagnostics(
            query_interpretation_status=InterpretationStatus.STRUCTURED,
            hard_constraint_count=1,
            soft_constraint_count=0,
            candidate_chunks_retrieved=1,
            candidate_documents_produced=1,
            documents_verified=1,
            verified_document_count=1,
            excluded_hard_mismatch_count=0,
            excluded_not_proven_count=0,
            verification_error_count=0,
            final_document_count=1,
            retrieval_latency_ms=1.0,
            verification_latency_ms=2.0,
            total_latency_ms=3.0,
            latency_budget_ms=1000,
            latency_budget_exceeded=False,
        ),
    )

    record_constraint_retrieval_metrics(
        result,
        endpoint="/api/rag/retrieve-verified",
    )

    output = generate_latest().decode("utf-8")
    assert "nalus_constraint_retrieval_requests_total" in output
    assert 'endpoint="/api/rag/retrieve-verified"' in output
    assert 'category="nationality"' in output
    assert "ECLI:CZ:US:2024:SECRET.1" not in output
    assert "chunk-secret" not in output
    assert "raw query" not in output
