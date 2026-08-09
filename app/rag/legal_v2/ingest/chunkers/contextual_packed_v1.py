"""Legal Contextual Packed chunker v1 (experiment candidate B).

Frozen policy (do not tune after inventory freeze):
- soft min 300 / soft target 650 / hard max 850 (native \\w+ units, same as A)
- pack complete paragraphs within the same SectionType
- overlap: at most one complete previous paragraph if <= 150 units; else none
- oversized paragraphs: sentence → punctuation → token-safe split
- emits children only (empty parent_windows) for the retrieval contract
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.rag.legal_v2.ingest.chunkers.names import CHUNKER_B_CONTEXTUAL_PACKED_V1
from app.rag.legal_v2.ingest.chunking import (
    ChunkingDiagnostics,
    HierarchicalChunkingResult,
    RetrievalChildChunk,
    SourceSpan,
)
from app.rag.legal_v2.models import (
    LegalDocumentStructure,
    LegalParagraph,
    stable_chunk_id,
)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ0-9])")
_CLAUSE_RE = re.compile(r"(?<=[;:])\s+")


@dataclass(frozen=True)
class ContextualPackedConfigV1:
    soft_min_tokens: int = 300
    soft_target_tokens: int = 650
    hard_max_tokens: int = 850
    overlap_max_tokens: int = 150
    chunker_version: str = CHUNKER_B_CONTEXTUAL_PACKED_V1

    def validate(self) -> None:
        if self.soft_min_tokens < 1:
            raise ValueError("soft_min_tokens must be positive")
        if self.soft_target_tokens < self.soft_min_tokens:
            raise ValueError("soft_target_tokens must be >= soft_min_tokens")
        if self.hard_max_tokens < self.soft_target_tokens:
            raise ValueError("hard_max_tokens must be >= soft_target_tokens")
        if self.overlap_max_tokens < 0:
            raise ValueError("overlap_max_tokens must be >= 0")


@dataclass(frozen=True)
class _Unit:
    text: str
    paragraph: LegalParagraph
    token_count: int
    sentence_index: int | None = None
    is_complete_paragraph: bool = True


def build_contextual_packed_chunks_v1(
    document: LegalDocumentStructure,
    *,
    config: ContextualPackedConfigV1 | None = None,
) -> HierarchicalChunkingResult:
    config = config or ContextualPackedConfigV1()
    config.validate()
    units, split_overlong = _paragraph_units(document.paragraphs, config)
    child_chunks = _pack_units(
        document_id=document.document_id,
        units=units,
        config=config,
    )
    section_distribution: dict[str, int] = {}
    for chunk in child_chunks:
        key = chunk.section_type.value
        section_distribution[key] = section_distribution.get(key, 0) + 1
    return HierarchicalChunkingResult(
        child_chunks=child_chunks,
        parent_windows=[],
        diagnostics=ChunkingDiagnostics(
            paragraph_count=len(document.paragraphs),
            child_chunk_count=len(child_chunks),
            parent_window_count=0,
            split_overlong_paragraph_count=split_overlong,
            merged_short_paragraph_count=0,
            average_child_tokens=_average([chunk.token_count for chunk in child_chunks]),
            average_parent_tokens=0.0,
            section_distribution=section_distribution,
        ),
    )


def _paragraph_units(
    paragraphs: list[LegalParagraph],
    config: ContextualPackedConfigV1,
) -> tuple[list[_Unit], int]:
    units: list[_Unit] = []
    split_count = 0
    for paragraph in paragraphs:
        tokens = _token_count(paragraph.normalized_text)
        if tokens <= config.hard_max_tokens:
            units.append(
                _Unit(
                    text=paragraph.normalized_text,
                    paragraph=paragraph,
                    token_count=tokens,
                    is_complete_paragraph=True,
                )
            )
            continue
        split_count += 1
        for sentence_index, piece in enumerate(
            _split_oversized(paragraph.normalized_text, config.hard_max_tokens)
        ):
            units.append(
                _Unit(
                    text=piece,
                    paragraph=paragraph,
                    token_count=_token_count(piece),
                    sentence_index=sentence_index,
                    is_complete_paragraph=False,
                )
            )
    return units, split_count


def _pack_units(
    *,
    document_id: str,
    units: list[_Unit],
    config: ContextualPackedConfigV1,
) -> list[RetrievalChildChunk]:
    chunks: list[RetrievalChildChunk] = []
    current: list[_Unit] = []
    last_pack: list[_Unit] = []

    def flush() -> None:
        nonlocal current, last_pack
        if not current:
            return
        chunks.append(_make_child(document_id, len(chunks), current, config=config))
        last_pack = list(current)
        current = []

    for unit in units:
        if not current:
            current = [unit]
            continue

        same_section = all(
            item.paragraph.section_type == unit.paragraph.section_type for item in current
        )
        if not same_section:
            flush()
            current = [unit]
            continue

        proposed = sum(item.token_count for item in current) + unit.token_count
        current_tokens = sum(item.token_count for item in current)

        if proposed <= config.soft_target_tokens:
            current.append(unit)
            continue

        if current_tokens < config.soft_min_tokens and proposed <= config.hard_max_tokens:
            current.append(unit)
            continue

        flush()
        overlap = _select_overlap(last_pack, unit, config)
        current = [*overlap, unit] if overlap else [unit]

    if current:
        flush()
    return chunks


def _select_overlap(
    previous_pack: list[_Unit],
    next_unit: _Unit,
    config: ContextualPackedConfigV1,
) -> list[_Unit]:
    """Whole-paragraph overlap or nothing — never trim a paragraph for overlap."""
    if not previous_pack:
        return []
    last = previous_pack[-1]
    if last.paragraph.section_type != next_unit.paragraph.section_type:
        return []
    if not last.is_complete_paragraph:
        return []
    if last.token_count > config.overlap_max_tokens:
        return []
    if last.token_count + next_unit.token_count > config.hard_max_tokens:
        return []
    return [last]


def _make_child(
    document_id: str,
    chunk_index: int,
    units: list[_Unit],
    *,
    config: ContextualPackedConfigV1,
) -> RetrievalChildChunk:
    paragraph_ids: list[str] = []
    paragraph_texts: dict[str, str] = {}
    paragraph_original_texts: dict[str, str] = {}
    spans: list[SourceSpan] = []
    for unit in units:
        if unit.paragraph.paragraph_id not in paragraph_ids:
            paragraph_ids.append(unit.paragraph.paragraph_id)
            paragraph_texts[unit.paragraph.paragraph_id] = unit.paragraph.normalized_text
            paragraph_original_texts[unit.paragraph.paragraph_id] = (
                unit.paragraph.original_text
            )
        spans.append(
            SourceSpan(
                paragraph_id=unit.paragraph.paragraph_id,
                paragraph_index=unit.paragraph.paragraph_index,
                start_offset=unit.paragraph.start_offset,
                end_offset=unit.paragraph.end_offset,
                sentence_index=unit.sentence_index,
            )
        )
    text = "\n\n".join(unit.text for unit in units)
    return RetrievalChildChunk(
        chunk_id=stable_chunk_id(
            document_id=document_id,
            chunk_index=chunk_index,
            paragraph_ids=paragraph_ids,
            chunk_type=config.chunker_version,
        ),
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        token_count=_token_count(text),
        paragraph_ids=paragraph_ids,
        paragraph_texts=paragraph_texts,
        paragraph_original_texts=paragraph_original_texts,
        source_spans=spans,
        section_type=units[0].paragraph.section_type,
        start_offset=min(unit.paragraph.start_offset for unit in units),
        end_offset=max(unit.paragraph.end_offset for unit in units),
        source_order=min(unit.paragraph.source_order for unit in units),
        heading_context=list(units[-1].paragraph.heading_context),
        metadata={
            "section_type": units[0].paragraph.section_type.value,
            "paragraph_ids": paragraph_ids,
            "source_order": min(unit.paragraph.source_order for unit in units),
            "chunker_version": config.chunker_version,
        },
    )


def _split_oversized(text: str, hard_max: int) -> list[str]:
    parts: list[str] = []
    for sentence in _split_sentence_aware(text):
        if _token_count(sentence) <= hard_max:
            parts.append(sentence)
            continue
        for clause in _split_clause_aware(sentence):
            if _token_count(clause) <= hard_max:
                parts.append(clause)
            else:
                parts.extend(_split_by_tokens(clause, hard_max))
    return parts or [text]


def _split_sentence_aware(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]
    return parts or [text]


def _split_clause_aware(text: str) -> list[str]:
    parts = [part.strip() for part in _CLAUSE_RE.split(text) if part.strip()]
    return parts or [text]


def _split_by_tokens(text: str, max_tokens: int) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for piece in text.split():
        piece_tokens = _token_count(piece)
        if piece_tokens > max_tokens:
            if current:
                parts.append(" ".join(current))
                current = []
                current_tokens = 0
            matches = list(_TOKEN_RE.finditer(piece))
            if not matches:
                parts.append(piece)
                continue
            for index in range(0, len(matches), max_tokens):
                group = matches[index : index + max_tokens]
                parts.append(piece[group[0].start() : group[-1].end()])
            continue
        if current and current_tokens + piece_tokens > max_tokens:
            parts.append(" ".join(current))
            current = [piece]
            current_tokens = piece_tokens
            continue
        current.append(piece)
        current_tokens += piece_tokens
    if current:
        parts.append(" ".join(current))
    return parts


def _token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


def _average(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0
