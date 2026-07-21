"""Typed models for additive constraint-aware legal retrieval.

The models in this module are intentionally provider-agnostic. They can be
populated by deterministic parsing or, in a later rollout, by a strictly
validated LLM structured-output interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ConstraintRequirement(str, Enum):
    HARD = "hard"
    SOFT = "soft"
    UNRESOLVED = "unresolved"


class ConstraintCategory(str, Enum):
    COURT = "court"
    LEGAL_EVENT = "legal_event"
    LEGAL_TOPIC = "legal_topic"
    ACTOR_ROLE = "actor_role"
    AFFECTED_PARTY_ROLE = "affected_party_role"
    NATIONALITY = "nationality"
    COUNTRY_RELATION = "country_relation"
    PROCEDURAL_STAGE = "procedural_stage"
    DATE = "date"
    DOCUMENT_TYPE = "document_type"
    EXCLUSION = "exclusion"


class RelationPredicate(str, Enum):
    HAS_NATIONALITY = "has_nationality"
    APPLIED_FOR = "applied_for"
    DESTINATION_COUNTRY = "destination_country"
    WRONGFULLY_REMOVED_OR_RETAINED = "wrongfully_removed_or_retained"
    CONCERNS_LEGAL_EVENT = "concerns_legal_event"


class InterpretationStatus(str, Enum):
    STRUCTURED = "structured"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class ConstraintVerificationStatus(str, Enum):
    MATCHED = "matched"
    MISMATCH = "mismatch"
    NOT_PROVEN = "not_proven"
    NOT_APPLICABLE = "not_applicable"


class VerificationMethod(str, Enum):
    TRUSTED_METADATA = "trusted_metadata"
    DETERMINISTIC_EVIDENCE = "deterministic_evidence"
    DETERMINISTIC_RELATION = "deterministic_relation"
    NOT_EVALUATED = "not_evaluated"


class DocumentDecisionStatus(str, Enum):
    VERIFIED_MATCH = "verified_match"
    EXCLUDED_HARD_MISMATCH = "excluded_hard_mismatch"
    EXCLUDED_NOT_PROVEN = "excluded_not_proven"
    EXCLUDED_INSUFFICIENT_EVIDENCE = "excluded_insufficient_evidence"
    VERIFICATION_TIMEOUT = "verification_timeout"
    VERIFICATION_ERROR = "verification_error"


@dataclass(frozen=True)
class StructuredEntity:
    id: str
    entity_type: str
    role: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StructuredRelation:
    subject: str
    predicate: RelationPredicate
    object: str
    requirement: ConstraintRequirement = ConstraintRequirement.HARD


@dataclass(frozen=True)
class StructuredConstraint:
    id: str
    category: ConstraintCategory
    value: str
    requirement: ConstraintRequirement
    relation: StructuredRelation | None = None
    description: str = ""


@dataclass(frozen=True)
class StructuredQuery:
    intent: str
    status: InterpretationStatus
    constraints: list[StructuredConstraint]
    entities: list[StructuredEntity] = field(default_factory=list)
    relations: list[StructuredRelation] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)
    retrieval_expansions: list[str] = field(default_factory=list)
    interpreter: str = "deterministic_v1"

    @property
    def hard_constraints(self) -> list[StructuredConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if constraint.requirement == ConstraintRequirement.HARD
        ]

    @property
    def soft_constraints(self) -> list[StructuredConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if constraint.requirement == ConstraintRequirement.SOFT
        ]


@dataclass(frozen=True)
class ConstraintEvidence:
    document_id: str
    chunk_id: str | None
    quote: str
    source_field: str | None = None


@dataclass(frozen=True)
class ConstraintVerificationResult:
    constraint_id: str
    category: ConstraintCategory
    status: ConstraintVerificationStatus
    required_value: str
    detected_value: str | None
    evidence: list[ConstraintEvidence]
    verification_method: VerificationMethod
    confidence: float
    reason: str


@dataclass(frozen=True)
class VerifiedDocument:
    document_id: str
    score: float
    decision_status: DocumentDecisionStatus
    constraint_results: list[ConstraintVerificationResult]
    supporting_passages: list[ConstraintEvidence]
    metadata: dict[str, Any] = field(default_factory=dict)
    candidate_chunk_count: int = 0


@dataclass(frozen=True)
class ConstraintRetrievalDiagnostics:
    query_interpretation_status: InterpretationStatus
    hard_constraint_count: int
    soft_constraint_count: int
    candidate_chunks_retrieved: int
    candidate_documents_produced: int
    documents_verified: int
    verified_document_count: int
    excluded_hard_mismatch_count: int
    excluded_not_proven_count: int
    verification_error_count: int
    final_document_count: int
    retrieval_latency_ms: float | None
    verification_latency_ms: float | None
    total_latency_ms: float | None
    latency_budget_ms: int | None
    latency_budget_exceeded: bool


@dataclass(frozen=True)
class ConstraintRetrievalResult:
    structured_query: StructuredQuery
    verified_documents: list[VerifiedDocument]
    rejected_documents: list[VerifiedDocument]
    diagnostics: ConstraintRetrievalDiagnostics
