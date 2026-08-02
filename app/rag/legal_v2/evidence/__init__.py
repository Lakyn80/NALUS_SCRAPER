"""Legal Retrieval v2 evidence-selection package.

The package path ``app.rag.legal_v2.evidence`` replaces the former module file.
"""

from app.rag.legal_v2.evidence.selection import (
    PREFERRED_SECTIONS,
    RESTRICTED_SECTIONS,
    CandidateEvidenceDocument,
    effective_source_of_claim,
    looks_like_court_holding_text,
    select_evidence_windows,
    source_of_claim_for_section,
)

__all__ = [
    "PREFERRED_SECTIONS",
    "RESTRICTED_SECTIONS",
    "CandidateEvidenceDocument",
    "effective_source_of_claim",
    "looks_like_court_holding_text",
    "select_evidence_windows",
    "source_of_claim_for_section",
]
