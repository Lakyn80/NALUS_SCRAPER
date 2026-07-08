from __future__ import annotations

from app.rag.clarification.models import AmbiguityType, LegalDomain, RetrievalHitSummary
from app.rag.clarification.text_utils import simplify_text


def infer_domain_from_document_id(document_id: str) -> LegalDomain:
    normalized = simplify_text(document_id).replace(" ", "")
    if any(marker in normalized for marker in ("tdo", "tz", "pzo")):
        return "criminal"
    if any(marker in normalized for marker in ("cdo", "nscr")):
        return "civil"
    if "nd" in normalized:
        return "execution"
    if any(marker in normalized for marker in ("ncr", "rod")):
        return "family"
    if any(marker in normalized for marker in ("aos", "aso")):
        return "administrative"
    return "unknown"


def infer_domains_from_hits(hits: list[RetrievalHitSummary]) -> list[LegalDomain]:
    return [infer_domain_from_document_id(hit.document_id) for hit in hits]


def detect_retrieval_domain_mismatch(
    hits: list[RetrievalHitSummary],
    *,
    query_domain: LegalDomain = "unknown",
) -> tuple[bool, AmbiguityType | None, str]:
    if not hits:
        return False, None, ""

    domains = [domain for domain in infer_domains_from_hits(hits) if domain != "unknown"]
    if len(domains) < 2:
        return False, None, ""

    top_domain = infer_domain_from_document_id(hits[0].document_id)
    if top_domain == "unknown":
        return False, None, ""

    other_domains = {
        infer_domain_from_document_id(hit.document_id)
        for hit in hits[1:5]
        if infer_domain_from_document_id(hit.document_id) != "unknown"
    }
    if top_domain in other_domains:
        reason = (
            f"Top výsledek je z domény {top_domain}, ale další hity obsahují jiné právní domény "
            f"({', '.join(sorted(other_domains))})."
        )
        return True, "retrieval_domain_mismatch", reason

    if query_domain != "unknown" and top_domain != query_domain:
        reason = (
            f"Dotaz naznačuje doménu {query_domain}, ale top-1 výsledek je z domény {top_domain}."
        )
        return True, "retrieval_domain_mismatch", reason

    return False, None, ""


def is_cdo_tdo_mismatch(hits: list[RetrievalHitSummary]) -> bool:
    if not hits:
        return False
    top_id = hits[0].document_id.upper()
    if "CDO" not in top_id:
        return False
    for hit in hits[1:5]:
        if "TDO" in hit.document_id.upper():
            return True
    return False
