"""Map current legal_v2 parser/chunker outputs onto canonical schema v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from app.rag.legal_v2.audit import CHUNKER_VERSION, PARSER_VERSION
from app.rag.legal_v2.ingest.chunking import HierarchicalChunkingResult, ParentEvidenceWindow, RetrievalChildChunk
from app.rag.legal_v2.models import LegalDocumentStructure, LegalParagraph
from app.rag.legal_v2.schema.canonical_v1 import (
    DEFAULT_CHUNKING_PROFILE,
    CanonicalBlock,
    CanonicalChildChunk,
    CanonicalDocument,
    CanonicalDocumentBundle,
    CanonicalParentContext,
    content_checksum,
)


@dataclass(frozen=True)
class LineInventoryRow:
    """Optional review/export line row used to populate line ranges and classes."""

    line_number: int
    text: str
    parser_block_id: str | None = None
    parser_class: str | None = None


def map_document_metadata(
    document: LegalDocumentStructure,
    *,
    parser_profile: str = PARSER_VERSION,
    source_document_id: str | None = None,
    source_checksum: str | None = None,
) -> CanonicalDocument:
    metadata = dict(document.metadata or {})
    return CanonicalDocument(
        document_id=document.document_id,
        source_document_id=_first_str(
            source_document_id,
            metadata.get("source_document_id"),
            metadata.get("source_id"),
            metadata.get("origin_source_id"),
        ),
        ecli=_first_str(metadata.get("ecli")),
        case_number=_first_str(
            metadata.get("case_number"),
            metadata.get("case_reference"),
            metadata.get("spisova_znacka"),
        ),
        court=_first_str(metadata.get("court")),
        court_chamber=_first_str(metadata.get("court_chamber"), metadata.get("senat")),
        decision_type=_first_str(
            metadata.get("decision_type"),
            metadata.get("document_type"),
            metadata.get("decision_form"),
        ),
        decision_date=_first_str(metadata.get("decision_date")),
        jurisdiction=str(metadata.get("jurisdiction") or "CZ"),
        language=str(metadata.get("language") or _document_language(document) or "cs"),
        source_url=_first_str(
            metadata.get("source_url"),
            metadata.get("text_url"),
            metadata.get("detail_url"),
        ),
        source_checksum=_first_str(source_checksum, metadata.get("source_checksum"), metadata.get("document_content_hash")),
        parser_profile=parser_profile,
    )


def map_blocks_from_paragraphs(
    document: LegalDocumentStructure,
    *,
    line_inventory: Sequence[LineInventoryRow] | None = None,
) -> list[CanonicalBlock]:
    lines_by_block = _group_lines_by_block(line_inventory or [])
    blocks: list[CanonicalBlock] = []
    for paragraph in document.paragraphs:
        line_rows = lines_by_block.get(paragraph.paragraph_id, [])
        line_numbers = [row.line_number for row in line_rows]
        line_classes = [row.parser_class for row in line_rows if row.parser_class]
        section_path = list(paragraph.heading_context) if paragraph.heading_context else [paragraph.section_type.value]
        primary_class = line_classes[0] if line_classes else paragraph.section_type.value
        blocks.append(
            CanonicalBlock(
                block_id=paragraph.paragraph_id,
                document_id=paragraph.document_id,
                block_index=paragraph.paragraph_index,
                line_start=min(line_numbers) if line_numbers else None,
                line_end=max(line_numbers) if line_numbers else None,
                start_offset=paragraph.start_offset,
                end_offset=paragraph.end_offset,
                raw_text=paragraph.original_text,
                normalized_text=paragraph.normalized_text,
                primary_class=primary_class,
                all_line_classes=list(line_classes) if line_classes else [paragraph.section_type.value],
                section_path=section_path,
                heading_context=list(paragraph.heading_context),
                paragraph_number=paragraph.numbering,
                hierarchy_level=None,
                parent_block_id=None,
                source_checksum=content_checksum(paragraph.original_text),
            )
        )
    return blocks


def map_child_chunk(
    child: RetrievalChildChunk,
    *,
    blocks_by_id: Mapping[str, CanonicalBlock],
    parent_id: str | None,
    chunking_profile: str = DEFAULT_CHUNKING_PROFILE,
    paragraphs_by_id: Mapping[str, LegalParagraph] | None = None,
) -> CanonicalChildChunk:
    source_block_ids = list(child.paragraph_ids)
    line_starts = [
        blocks_by_id[block_id].line_start
        for block_id in source_block_ids
        if block_id in blocks_by_id and blocks_by_id[block_id].line_start is not None
    ]
    line_ends = [
        blocks_by_id[block_id].line_end
        for block_id in source_block_ids
        if block_id in blocks_by_id and blocks_by_id[block_id].line_end is not None
    ]
    primary_number = None
    if paragraphs_by_id:
        for block_id in source_block_ids:
            paragraph = paragraphs_by_id.get(block_id)
            if paragraph and paragraph.numbering:
                primary_number = paragraph.numbering
                break
    return CanonicalChildChunk(
        chunk_id=child.chunk_id,
        document_id=child.document_id,
        source_block_ids=source_block_ids,
        line_start=min(line_starts) if line_starts else None,
        line_end=max(line_ends) if line_ends else None,
        start_offset=child.start_offset,
        end_offset=child.end_offset,
        chunk_text=child.text,
        embedding_text=child.text,
        section_path=[child.section_type.value],
        heading_context=list(child.heading_context),
        primary_paragraph_number=primary_number,
        parent_id=parent_id,
        token_count=child.token_count,
        chunking_profile=chunking_profile,
        content_checksum=content_checksum(child.text),
    )


def map_parent_window(
    window: ParentEvidenceWindow,
    *,
    blocks_by_id: Mapping[str, CanonicalBlock],
    chunking_profile: str = DEFAULT_CHUNKING_PROFILE,
) -> CanonicalParentContext:
    line_starts = [
        blocks_by_id[block_id].line_start
        for block_id in window.paragraph_ids
        if block_id in blocks_by_id and blocks_by_id[block_id].line_start is not None
    ]
    line_ends = [
        blocks_by_id[block_id].line_end
        for block_id in window.paragraph_ids
        if block_id in blocks_by_id and blocks_by_id[block_id].line_end is not None
    ]
    section_path = [section.value for section in window.section_types]
    context_type = section_path[0] if section_path else "evidence_window"
    if window.truncated:
        context_type = f"{context_type}:truncated"
    return CanonicalParentContext(
        parent_id=window.window_id,
        document_id=window.document_id,
        child_ids=list(window.child_chunk_ids),
        line_start=min(line_starts) if line_starts else None,
        line_end=max(line_ends) if line_ends else None,
        start_offset=window.start_offset,
        end_offset=window.end_offset,
        parent_text=window.text,
        section_path=section_path,
        context_type=context_type,
        token_count=window.token_count,
        content_checksum=content_checksum(window.text),
    )


def map_legal_v2_bundle(
    document: LegalDocumentStructure,
    chunking: HierarchicalChunkingResult | None = None,
    *,
    line_inventory: Sequence[LineInventoryRow] | None = None,
    parser_profile: str = PARSER_VERSION,
    chunking_profile: str = DEFAULT_CHUNKING_PROFILE,
    source_document_id: str | None = None,
    source_checksum: str | None = None,
) -> CanonicalDocumentBundle:
    """Bridge legal_v2 parse (+ optional hierarchical chunks) to canonical v1."""
    _ = CHUNKER_VERSION  # documented related constant; profile string is explicit
    canonical_document = map_document_metadata(
        document,
        parser_profile=parser_profile,
        source_document_id=source_document_id,
        source_checksum=source_checksum,
    )
    blocks = map_blocks_from_paragraphs(document, line_inventory=line_inventory)
    blocks_by_id = {block.block_id: block for block in blocks}
    paragraphs_by_id = {paragraph.paragraph_id: paragraph for paragraph in document.paragraphs}

    children: list[CanonicalChildChunk] = []
    parents: list[CanonicalParentContext] = []
    if chunking is not None:
        child_to_parent = _child_to_parent_map(chunking.parent_windows)
        for child in chunking.child_chunks:
            children.append(
                map_child_chunk(
                    child,
                    blocks_by_id=blocks_by_id,
                    parent_id=child_to_parent.get(child.chunk_id),
                    chunking_profile=chunking_profile,
                    paragraphs_by_id=paragraphs_by_id,
                )
            )
        for window in chunking.parent_windows:
            parents.append(
                map_parent_window(
                    window,
                    blocks_by_id=blocks_by_id,
                    chunking_profile=chunking_profile,
                )
            )

    return CanonicalDocumentBundle(
        document=canonical_document,
        blocks=blocks,
        children=children,
        parents=parents,
        chunking_profile=chunking_profile if chunking is not None else None,
    )


def line_inventory_from_review_rows(rows: Sequence[Mapping[str, Any]]) -> list[LineInventoryRow]:
    inventory: list[LineInventoryRow] = []
    for row in rows:
        line_number = row.get("raw_line_number", row.get("line_number"))
        if line_number is None:
            continue
        inventory.append(
            LineInventoryRow(
                line_number=int(line_number),
                text=str(row.get("raw_text") or row.get("text") or ""),
                parser_block_id=_first_str(row.get("parser_block_id"), row.get("stable_block_id")),
                parser_class=_first_str(
                    row.get("parser_proposed_line_class"),
                    row.get("parser_class"),
                    row.get("primary_class"),
                ),
            )
        )
    return inventory


def _child_to_parent_map(windows: Sequence[ParentEvidenceWindow]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for window in windows:
        for child_id in window.child_chunk_ids:
            mapping.setdefault(child_id, window.window_id)
    return mapping


def _group_lines_by_block(rows: Sequence[LineInventoryRow]) -> dict[str, list[LineInventoryRow]]:
    grouped: dict[str, list[LineInventoryRow]] = {}
    for row in rows:
        if not row.parser_block_id:
            continue
        grouped.setdefault(row.parser_block_id, []).append(row)
    for block_id, items in grouped.items():
        grouped[block_id] = sorted(items, key=lambda item: item.line_number)
    return grouped


def _document_language(document: LegalDocumentStructure) -> str | None:
    if not document.paragraphs:
        return None
    return document.paragraphs[0].language or None


def _first_str(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


__all__ = [
    "LineInventoryRow",
    "map_document_metadata",
    "map_blocks_from_paragraphs",
    "map_child_chunk",
    "map_parent_window",
    "map_legal_v2_bundle",
    "line_inventory_from_review_rows",
]
