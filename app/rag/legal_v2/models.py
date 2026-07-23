from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SectionType(str, Enum):
    HEADER = "header"
    PARTICIPANTS = "participants"
    PROCEDURAL_HISTORY = "procedural_history"
    FACTS = "facts"
    PARTY_ARGUMENTS = "party_arguments"
    LEGAL_FRAMEWORK = "legal_framework"
    CITED_CASE = "cited_case"
    COURT_REASONING = "court_reasoning"
    OPERATIVE_PART = "operative_part"
    INSTRUCTION = "instruction"
    OTHER = "other"


@dataclass(frozen=True)
class MetadataProvenance:
    source: str
    extraction_method: str
    document_version: str | None = None
    source_url: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LegalParagraph:
    document_id: str
    paragraph_id: str
    paragraph_index: int
    original_text: str
    normalized_text: str
    section_type: SectionType
    start_offset: int
    end_offset: int
    source_order: int
    heading_context: list[str]
    is_boilerplate: bool
    is_citation_block: bool
    language: str
    metadata_provenance: MetadataProvenance
    numbering: str | None = None


@dataclass(frozen=True)
class LegalDocumentStructure:
    document_id: str
    normalized_text: str
    paragraphs: list[LegalParagraph]
    diagnostics: "ParagraphParsingDiagnostics"
    metadata: dict[str, Any] = field(default_factory=dict)

    def reconstruct_text(self) -> str:
        return "\n\n".join(paragraph.original_text for paragraph in self.paragraphs)


@dataclass(frozen=True)
class ParagraphParsingDiagnostics:
    paragraph_count: int
    numbered_paragraph_count: int
    heading_count: int
    boilerplate_count: int
    citation_block_count: int
    damaged_formatting_detected: bool
    fallback_paragraphs_created: int
    section_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)


_SPACE_RE = re.compile(r"\s+")


def normalize_legal_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    return _SPACE_RE.sub(" ", text).strip()


def stable_paragraph_id(
    *,
    document_id: str,
    paragraph_index: int,
    normalized_text: str,
    document_version: str | None = None,
) -> str:
    payload = "|".join(
        [
            document_id,
            str(paragraph_index),
            document_version or "",
            normalized_text,
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{document_id}:p:{paragraph_index:05d}:{digest}"


def stable_chunk_id(
    *,
    document_id: str,
    chunk_index: int,
    paragraph_ids: list[str],
    chunk_type: str,
) -> str:
    payload = "|".join([document_id, chunk_type, str(chunk_index), *paragraph_ids])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{document_id}:{chunk_type}:{chunk_index:05d}:{digest}"
