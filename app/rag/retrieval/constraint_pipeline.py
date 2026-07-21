"""Additive constraint-aware document retrieval pipeline."""

from __future__ import annotations

import time
from typing import Callable

from app.core.logging import get_logger
from app.core.tracing import trace_event
from app.rag.retrieval.constraint_config import ConstraintRetrievalConfig
from app.rag.retrieval.constraint_models import (
    ConstraintRetrievalDiagnostics,
    ConstraintRetrievalResult,
    DocumentDecisionStatus,
    VerifiedDocument,
)
from app.rag.retrieval.constraint_verification import verify_document_constraints
from app.rag.retrieval.document_retrieval import (
    DocumentRetrievalConfig,
    build_document_level_results,
)
from app.rag.retrieval.full_document import FullDocumentStore
from app.rag.retrieval.models import RetrievedChunk
from app.rag.retrieval.structured_query import interpret_structured_query

logger = get_logger(__name__)

CandidateRetriever = Callable[[str, int], list[RetrievedChunk]]
CandidateFilter = Callable[[RetrievedChunk], bool]


def retrieve_verified_documents(
    *,
    query: str,
    retriever: CandidateRetriever,
    full_document_store: FullDocumentStore,
    config: ConstraintRetrievalConfig,
    candidate_filter: CandidateFilter | None = None,
) -> ConstraintRetrievalResult:
    config.validate()
    started = time.perf_counter()
    structured_query = interpret_structured_query(query)

    retrieval_started = time.perf_counter()
    candidate_chunks = retriever(query, config.max_candidate_chunks)
    retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000
    if candidate_filter is not None:
        candidate_chunks = [chunk for chunk in candidate_chunks if candidate_filter(chunk)]

    aggregation_config = DocumentRetrievalConfig(
        enabled=True,
        max_candidate_chunks=config.max_candidate_chunks,
        max_returned_documents=config.max_candidate_documents,
        max_supporting_chunks_per_document=config.max_supporting_chunks,
        document_relevance_threshold=0.0,
        latency_budget_ms=config.total_latency_budget_ms,
    )
    candidate_documents = build_document_level_results(
        candidate_chunks=candidate_chunks,
        config=aggregation_config,
        retrieval_latency_ms=retrieval_latency_ms,
    ).documents

    verification_started = time.perf_counter()
    verified: list[VerifiedDocument] = []
    rejected: list[VerifiedDocument] = []
    latency_budget_exceeded = False

    for candidate in candidate_documents[: config.max_candidate_documents]:
        if _latency_budget_exceeded(started, config.total_latency_budget_ms):
            latency_budget_exceeded = True
            break
        try:
            document = full_document_store.get(candidate.document_id)
            verification = verify_document_constraints(
                structured_query=structured_query,
                candidate=candidate,
                document=document,
                config=config,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[constraint-retrieval] document verification failed document_id_length=%d error=%s",
                len(candidate.document_id),
                exc,
            )
            verification = VerifiedDocument(
                document_id=candidate.document_id,
                score=candidate.score,
                decision_status=DocumentDecisionStatus.VERIFICATION_ERROR,
                constraint_results=[],
                supporting_passages=[],
                metadata=candidate.metadata,
                candidate_chunk_count=candidate.candidate_chunk_count,
            )

        if verification.decision_status == DocumentDecisionStatus.VERIFIED_MATCH:
            verified.append(verification)
        else:
            rejected.append(verification)

    verification_latency_ms = (time.perf_counter() - verification_started) * 1000
    verified.sort(key=lambda document: (-document.score, document.document_id))
    rejected.sort(key=lambda document: (-document.score, document.document_id))
    final_verified = verified[: config.max_returned_documents]
    total_latency_ms = (time.perf_counter() - started) * 1000
    if _latency_budget_exceeded(started, config.total_latency_budget_ms):
        latency_budget_exceeded = True

    diagnostics = ConstraintRetrievalDiagnostics(
        query_interpretation_status=structured_query.status,
        hard_constraint_count=len(structured_query.hard_constraints),
        soft_constraint_count=len(structured_query.soft_constraints),
        candidate_chunks_retrieved=len(candidate_chunks),
        candidate_documents_produced=len(candidate_documents),
        documents_verified=len(verified) + len(rejected),
        verified_document_count=len(verified),
        excluded_hard_mismatch_count=_count_decisions(
            rejected,
            DocumentDecisionStatus.EXCLUDED_HARD_MISMATCH,
        ),
        excluded_not_proven_count=_count_decisions(
            rejected,
            DocumentDecisionStatus.EXCLUDED_NOT_PROVEN,
        ),
        verification_error_count=_count_decisions(
            rejected,
            DocumentDecisionStatus.VERIFICATION_ERROR,
        ),
        final_document_count=len(final_verified),
        retrieval_latency_ms=retrieval_latency_ms,
        verification_latency_ms=verification_latency_ms,
        total_latency_ms=total_latency_ms,
        latency_budget_ms=config.total_latency_budget_ms,
        latency_budget_exceeded=latency_budget_exceeded,
    )

    trace_event(
        logger,
        "constraint_retrieval.done",
        interpretation_status=structured_query.status.value,
        hard_constraint_count=diagnostics.hard_constraint_count,
        candidate_chunks_retrieved=diagnostics.candidate_chunks_retrieved,
        candidate_documents_produced=diagnostics.candidate_documents_produced,
        documents_verified=diagnostics.documents_verified,
        final_document_count=diagnostics.final_document_count,
        latency_budget_exceeded=diagnostics.latency_budget_exceeded,
    )
    logger.info(
        "[constraint-retrieval] candidates=%d documents=%d verified=%d rejected=%d final=%d",
        diagnostics.candidate_chunks_retrieved,
        diagnostics.candidate_documents_produced,
        diagnostics.verified_document_count,
        len(rejected),
        diagnostics.final_document_count,
    )
    return ConstraintRetrievalResult(
        structured_query=structured_query,
        verified_documents=final_verified,
        rejected_documents=rejected,
        diagnostics=diagnostics,
    )


def _count_decisions(
    documents: list[VerifiedDocument],
    status: DocumentDecisionStatus,
) -> int:
    return sum(1 for document in documents if document.decision_status == status)


def _latency_budget_exceeded(started: float, budget_ms: int | None) -> bool:
    if budget_ms is None:
        return False
    return (time.perf_counter() - started) * 1000 > budget_ms
