"""Legal Retrieval v2 verification package.

Public symbols remain importable from ``app.rag.legal_v2.verifier`` via the
compatibility shim. Prefer this package for new code.
"""

from app.rag.legal_v2.verify.verifier import (
    CandidateDocumentForVerification,
    ConstraintVerificationResult,
    ConstraintVerificationStatus,
    DeepSeekSemanticVerifierProvider,
    DeterministicFakeVerifier,
    EvidenceCoverageVerifier,
    EvidenceWindowForConstraint,
    RelevanceClassification,
    SemanticVerifierProvider,
    SemanticVerifierResult,
    VerificationDecision,
    VerifierDiagnostics,
    apply_thinking_promotion_policy,
    deterministic_verification_gate,
    run_semantic_verifier,
    thinking_promotion_allows_verified_match,
    validate_verifier_payload,
    verifier_diagnostics,
)

__all__ = [
    "CandidateDocumentForVerification",
    "ConstraintVerificationResult",
    "ConstraintVerificationStatus",
    "DeepSeekSemanticVerifierProvider",
    "DeterministicFakeVerifier",
    "EvidenceCoverageVerifier",
    "EvidenceWindowForConstraint",
    "RelevanceClassification",
    "SemanticVerifierProvider",
    "SemanticVerifierResult",
    "VerificationDecision",
    "VerifierDiagnostics",
    "apply_thinking_promotion_policy",
    "deterministic_verification_gate",
    "run_semantic_verifier",
    "thinking_promotion_allows_verified_match",
    "validate_verifier_payload",
    "verifier_diagnostics",
]
