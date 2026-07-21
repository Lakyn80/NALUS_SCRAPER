"""Prometheus metrics for constraint-aware retrieval.

Labels are intentionally bounded. Do not add raw queries, document ids, ECLI
values, chunk ids, evidence text, or user content as labels.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

from app.rag.retrieval.constraint_models import ConstraintRetrievalResult

REQUESTS_TOTAL = Counter(
    "nalus_constraint_retrieval_requests_total",
    "Constraint-aware retrieval endpoint requests.",
    ("endpoint", "status"),
)

DOCUMENTS_TOTAL = Counter(
    "nalus_constraint_retrieval_documents_total",
    "Constraint-aware retrieval document decisions.",
    ("endpoint", "decision_status"),
)

VERIFICATIONS_TOTAL = Counter(
    "nalus_constraint_verifications_total",
    "Constraint verification results.",
    ("category", "status", "method"),
)

LATENCY_SECONDS = Histogram(
    "nalus_constraint_retrieval_latency_seconds",
    "Constraint-aware retrieval total latency.",
    ("endpoint", "status"),
)


def record_constraint_retrieval_metrics(
    result: ConstraintRetrievalResult,
    *,
    endpoint: str,
    status: str = "success",
) -> None:
    REQUESTS_TOTAL.labels(endpoint=endpoint, status=status).inc()
    if result.diagnostics.total_latency_ms is not None:
        LATENCY_SECONDS.labels(endpoint=endpoint, status=status).observe(
            max(0.0, result.diagnostics.total_latency_ms / 1000)
        )
    for document in [*result.verified_documents, *result.rejected_documents]:
        DOCUMENTS_TOTAL.labels(
            endpoint=endpoint,
            decision_status=document.decision_status.value,
        ).inc()
        for verification in document.constraint_results:
            VERIFICATIONS_TOTAL.labels(
                category=verification.category.value,
                status=verification.status.value,
                method=verification.verification_method.value,
            ).inc()


def record_constraint_retrieval_error(
    *,
    endpoint: str,
    status: str,
) -> None:
    REQUESTS_TOTAL.labels(endpoint=endpoint, status=status).inc()
