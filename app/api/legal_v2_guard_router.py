"""Disabled-by-default guard for the Legal Retrieval v2 API route."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.observability.legal_v2_metrics import record_request
from app.rag.legal_v2.pipeline import legal_v2_search_enabled

router = APIRouter(prefix="/api/rag", tags=["rag"])


class LegalV2GuardSearchRequest(BaseModel):
    query: str
    sources: list[str] | None = None
    max_results: int = 10
    debug: bool = False


@router.post("/search-v2")
def search_v2_disabled_guard(req: LegalV2GuardSearchRequest) -> None:
    del req
    endpoint_label = "/api/rag/search-v2"
    if not legal_v2_search_enabled():
        record_request(endpoint=endpoint_label, status="disabled")
        raise HTTPException(
            status_code=404,
            detail=(
                "Legal Retrieval v2 search is disabled. Set "
                "NALUS_LEGAL_V2_SEARCH_ENABLED=1 to enable the isolated endpoint."
            ),
        )
    record_request(endpoint=endpoint_label, status="unavailable")
    raise HTTPException(
        status_code=503,
        detail="Legal Retrieval v2 search is enabled but no runtime route is registered.",
    )
