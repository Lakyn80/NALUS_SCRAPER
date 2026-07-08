from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

DecisionType = Literal["proceed_to_retrieval", "ask_clarifying_question"]
LegalDomain = Literal["criminal", "civil", "execution", "family", "administrative", "unknown"]
ProcedureStage = Literal["first_instance", "appeal", "dovolani", "execution", "unknown"]
AmbiguityType = Literal[
    "legal_domain_ambiguous",
    "procedure_stage_ambiguous",
    "role_ambiguous",
    "remedy_ambiguous",
    "jurisdiction_or_court_ambiguous",
    "retrieval_domain_mismatch",
]


@dataclass(frozen=True)
class RetrievalHitSummary:
    rank: int
    document_id: str
    score: float = 0.0


@dataclass(frozen=True)
class RuleAssessment:
    domain_confidence: float
    procedure_confidence: float
    role_confidence: float
    remedy_confidence: float
    jurisdiction_confidence: float
    detected_legal_domain: LegalDomain
    detected_procedure_stage: ProcedureStage
    ambiguity_types: tuple[AmbiguityType, ...]
    missing_slots: tuple[str, ...]
    query_signature: str
    reason_cs: str


@dataclass(frozen=True)
class ClarificationDecision:
    decision: DecisionType
    confidence: float
    ambiguity_types: list[AmbiguityType]
    missing_slots: list[str]
    detected_legal_domain: LegalDomain
    detected_procedure_stage: ProcedureStage
    clarification_question_cs: str
    reason_cs: str
    cache_key: str
    query_signature: str
    recommended_next_action: Literal["ask_user", "run_retrieval"] = "run_retrieval"
    cache_hit: bool = False
    semantic_cache_hit: bool = False
    llm_called: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClarificationDecision":
        return cls(
            decision=payload["decision"],
            confidence=float(payload.get("confidence", 0.0)),
            ambiguity_types=list(payload.get("ambiguity_types", [])),
            missing_slots=list(payload.get("missing_slots", [])),
            detected_legal_domain=payload.get("detected_legal_domain", "unknown"),
            detected_procedure_stage=payload.get("detected_procedure_stage", "unknown"),
            clarification_question_cs=str(payload.get("clarification_question_cs", "")),
            reason_cs=str(payload.get("reason_cs", "")),
            cache_key=str(payload.get("cache_key", "")),
            query_signature=str(payload.get("query_signature", "")),
            recommended_next_action=payload.get("recommended_next_action", "run_retrieval"),
            cache_hit=bool(payload.get("cache_hit", False)),
            semantic_cache_hit=bool(payload.get("semantic_cache_hit", False)),
            llm_called=bool(payload.get("llm_called", False)),
        )


@dataclass
class CachedClarificationEntry:
    query_signature: str
    ambiguity_types: list[AmbiguityType]
    missing_slots: list[str]
    clarification_question_cs: str
    detected_issue: str
    recommended_next_action: Literal["ask_user", "run_retrieval"]
    created_at: str = ""
    rules_version: str = ""
    reason_cs: str = ""
