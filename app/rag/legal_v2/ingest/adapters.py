from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from app.rag.legal_v2.models import MetadataProvenance
from app.rag.legal_v2.parser import parse_legal_document


@dataclass(frozen=True)
class LegalSourceDocument:
    document_id: str
    source: str
    text: str
    metadata: dict[str, Any]
    origin_path: str


class LegalSourceAdapter(Protocol):
    source_name: str

    def parse(self, document: LegalSourceDocument):
        ...


class GenericLegalAdapter:
    source_name = "generic"

    def parse(self, document: LegalSourceDocument):
        metadata = dict(document.metadata)
        metadata.setdefault("source", document.source)
        metadata.setdefault("origin_path", document.origin_path)
        return parse_legal_document(
            document_id=document.document_id,
            text=document.text,
            metadata=metadata,
            provenance=MetadataProvenance(
                source=document.source,
                extraction_method="legal_v2_generic_adapter",
                document_version=metadata.get("document_version"),
                source_url=metadata.get("text_url") or metadata.get("detail_url"),
            ),
        )


class NalusConstitutionalAdapter(GenericLegalAdapter):
    source_name = "constitutional"

    def parse(self, document: LegalSourceDocument):
        metadata = dict(document.metadata)
        metadata.setdefault("court", "Ústavní soud")
        metadata.setdefault("document_type", metadata.get("decision_form") or "decision")
        return parse_legal_document(
            document_id=document.document_id,
            text=document.text,
            metadata=metadata,
            provenance=MetadataProvenance(
                source="constitutional",
                extraction_method="legal_v2_nalus_constitutional_adapter",
                document_version=metadata.get("document_version"),
                source_url=metadata.get("text_url") or metadata.get("detail_url"),
            ),
        )


class SupremeCourtAdapter(GenericLegalAdapter):
    source_name = "supreme"

    def parse(self, document: LegalSourceDocument):
        metadata = dict(document.metadata)
        metadata.setdefault("court", "Nejvyšší soud")
        metadata.setdefault("document_type", metadata.get("document_type") or "decision")
        return parse_legal_document(
            document_id=document.document_id,
            text=document.text,
            metadata=metadata,
            provenance=MetadataProvenance(
                source="supreme",
                extraction_method="legal_v2_supreme_court_adapter",
                document_version=metadata.get("document_version"),
                source_url=metadata.get("source_url"),
            ),
        )


class LegalAdapterRegistry:
    def __init__(self, adapters: list[LegalSourceAdapter] | None = None) -> None:
        adapters = adapters or [
            NalusConstitutionalAdapter(),
            SupremeCourtAdapter(),
            GenericLegalAdapter(),
        ]
        self._adapters = {adapter.source_name: adapter for adapter in adapters}
        self._fallback = self._adapters["generic"]

    def adapter_for(self, source: str) -> LegalSourceAdapter:
        normalized = source.strip().lower()
        if normalized in {"nalus", "usoud", "constitutional", "ústavní soud", "ustavni soud"}:
            return self._adapters.get("constitutional", self._fallback)
        if normalized in {"nsoud", "supreme", "nejvyšší soud", "nejvyssi soud"}:
            return self._adapters.get("supreme", self._fallback)
        return self._fallback

