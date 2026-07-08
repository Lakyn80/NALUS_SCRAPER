"""Legal query clarification gate for long-form client questions.

Court decisions are still retrieved via Qdrant. Redis caches clarification payloads.
Optional Qdrant collection ``legal_query_clarification_patterns`` stores only
clarification-pattern embeddings for similar ambiguous-query reuse.
"""

from app.rag.clarification.cache import build_clarification_cache
from app.rag.clarification.models import (
    AmbiguityType,
    ClarificationDecision,
    LegalDomain,
    ProcedureStage,
    RetrievalHitSummary,
)
from app.rag.clarification.orchestrator import ClarificationTrace, ClarifyingOrchestratorService
from app.rag.clarification.qdrant_patterns import build_clarification_pattern_index
from app.rag.clarification.service import LegalQueryClarificationService

__all__ = [
    "AmbiguityType",
    "ClarificationDecision",
    "ClarificationTrace",
    "ClarifyingOrchestratorService",
    "LegalDomain",
    "LegalQueryClarificationService",
    "ProcedureStage",
    "RetrievalHitSummary",
    "build_clarification_cache",
    "build_clarification_pattern_index",
]
