from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.rag.legal_v2.models import (
    LegalDocumentStructure,
    LegalParagraph,
    SectionType,
    stable_chunk_id,
)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ0-9])")


@dataclass(frozen=True)
class HierarchicalChunkConfig:
    child_target_min_tokens: int = 250
    child_target_max_tokens: int = 500
    child_hard_max_tokens: int = 650
    parent_target_min_tokens: int = 800
    parent_target_max_tokens: int = 1500
    parent_hard_max_tokens: int = 1800
    min_short_paragraph_tokens: int = 80

    def validate(self) -> None:
        if self.child_target_min_tokens < 1:
            raise ValueError("child_target_min_tokens must be positive.")
        if self.child_target_max_tokens < self.child_target_min_tokens:
            raise ValueError("child_target_max_tokens must be >= child_target_min_tokens.")
        if self.child_hard_max_tokens < self.child_target_max_tokens:
            raise ValueError("child_hard_max_tokens must be >= child_target_max_tokens.")
        if self.parent_target_max_tokens < self.parent_target_min_tokens:
            raise ValueError("parent_target_max_tokens must be >= parent_target_min_tokens.")
        if self.parent_hard_max_tokens < self.parent_target_max_tokens:
            raise ValueError("parent_hard_max_tokens must be >= parent_target_max_tokens.")


@dataclass(frozen=True)
class SourceSpan:
    paragraph_id: str
    paragraph_index: int
    start_offset: int
    end_offset: int
    sentence_index: int | None = None


@dataclass(frozen=True)
class RetrievalChildChunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    token_count: int
    paragraph_ids: list[str]
    paragraph_texts: dict[str, str]
    paragraph_original_texts: dict[str, str]
    source_spans: list[SourceSpan]
    section_type: SectionType
    start_offset: int
    end_offset: int
    source_order: int
    heading_context: list[str]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ParentEvidenceWindow:
    window_id: str
    document_id: str
    text: str
    token_count: int
    paragraph_ids: list[str]
    child_chunk_ids: list[str]
    section_types: list[SectionType]
    start_offset: int
    end_offset: int
    truncated: bool = False


@dataclass(frozen=True)
class ChunkingDiagnostics:
    paragraph_count: int
    child_chunk_count: int
    parent_window_count: int
    split_overlong_paragraph_count: int
    merged_short_paragraph_count: int
    average_child_tokens: float
    average_parent_tokens: float
    section_distribution: dict[str, int]
    reconstruction_failures: int = 0


@dataclass(frozen=True)
class HierarchicalChunkingResult:
    child_chunks: list[RetrievalChildChunk]
    parent_windows: list[ParentEvidenceWindow]
    diagnostics: ChunkingDiagnostics

    def reconstruct_text(self) -> str:
        ordered: dict[str, tuple[int, str]] = {}
        for chunk in self.child_chunks:
            span_indexes = {
                span.paragraph_id: span.paragraph_index for span in chunk.source_spans
            }
            for paragraph_id in chunk.paragraph_ids:
                paragraph_text = chunk.paragraph_original_texts.get(
                    paragraph_id
                ) or chunk.paragraph_texts.get(paragraph_id)
                paragraph_index = span_indexes.get(paragraph_id)
                if paragraph_text is None or paragraph_index is None:
                    continue
                ordered.setdefault(paragraph_id, (paragraph_index, paragraph_text))
        return "\n\n".join(
            text for _, text in sorted(ordered.values(), key=lambda item: item[0])
        )


def build_hierarchical_chunks(
    document: LegalDocumentStructure,
    *,
    config: HierarchicalChunkConfig | None = None,
) -> HierarchicalChunkingResult:
    config = config or HierarchicalChunkConfig()
    config.validate()
    child_units, split_overlong = _paragraph_units(document.paragraphs, config)
    child_chunks, merged_short = _build_child_chunks(
        document_id=document.document_id,
        units=child_units,
        config=config,
    )
    parent_windows = build_parent_windows(document, child_chunks, config=config)
    section_distribution: dict[str, int] = {}
    for chunk in child_chunks:
        section_distribution[chunk.section_type.value] = (
            section_distribution.get(chunk.section_type.value, 0) + 1
        )
    return HierarchicalChunkingResult(
        child_chunks=child_chunks,
        parent_windows=parent_windows,
        diagnostics=ChunkingDiagnostics(
            paragraph_count=len(document.paragraphs),
            child_chunk_count=len(child_chunks),
            parent_window_count=len(parent_windows),
            split_overlong_paragraph_count=split_overlong,
            merged_short_paragraph_count=merged_short,
            average_child_tokens=_average([chunk.token_count for chunk in child_chunks]),
            average_parent_tokens=_average([window.token_count for window in parent_windows]),
            section_distribution=section_distribution,
        ),
    )


def build_parent_windows(
    document: LegalDocumentStructure,
    child_chunks: list[RetrievalChildChunk],
    *,
    config: HierarchicalChunkConfig,
) -> list[ParentEvidenceWindow]:
    windows: list[ParentEvidenceWindow] = []
    paragraphs_by_id = {paragraph.paragraph_id: paragraph for paragraph in document.paragraphs}
    for child in child_chunks:
        anchor_indexes = [
            paragraphs_by_id[paragraph_id].paragraph_index
            for paragraph_id in child.paragraph_ids
            if paragraph_id in paragraphs_by_id
        ]
        if not anchor_indexes:
            continue
        section = child.section_type
        start_index = min(anchor_indexes)
        end_index = max(anchor_indexes)
        selected = [
            paragraph
            for paragraph in document.paragraphs[start_index : end_index + 1]
            if paragraph.section_type == section
        ]
        left = start_index - 1
        right = end_index + 1
        while _token_count(_paragraph_text(selected)) < config.parent_target_min_tokens:
            added = False
            if left >= 0 and document.paragraphs[left].section_type == section:
                candidate = [document.paragraphs[left], *selected]
                if _token_count(_paragraph_text(candidate)) <= config.parent_hard_max_tokens:
                    selected = candidate
                    added = True
                left -= 1
            if _token_count(_paragraph_text(selected)) >= config.parent_target_min_tokens:
                break
            if right < len(document.paragraphs) and document.paragraphs[right].section_type == section:
                candidate = [*selected, document.paragraphs[right]]
                if _token_count(_paragraph_text(candidate)) <= config.parent_hard_max_tokens:
                    selected = candidate
                    added = True
                right += 1
            if not added:
                break
        windows.append(_make_parent_window(document.document_id, len(windows), child, selected, config=config))
    return windows


def expand_parent_window(
    document: LegalDocumentStructure,
    anchor_child: RetrievalChildChunk,
    *,
    config: HierarchicalChunkConfig | None = None,
) -> ParentEvidenceWindow:
    config = config or HierarchicalChunkConfig()
    windows = build_parent_windows(document, [anchor_child], config=config)
    if not windows:
        raise ValueError("Cannot expand parent window without paragraph evidence.")
    return windows[0]


@dataclass(frozen=True)
class _ChunkUnit:
    text: str
    paragraph: LegalParagraph
    token_count: int
    sentence_index: int | None = None


def _paragraph_units(
    paragraphs: list[LegalParagraph],
    config: HierarchicalChunkConfig,
) -> tuple[list[_ChunkUnit], int]:
    units: list[_ChunkUnit] = []
    split_count = 0
    for paragraph in paragraphs:
        tokens = _token_count(paragraph.normalized_text)
        if tokens <= config.child_hard_max_tokens:
            units.append(_ChunkUnit(paragraph.normalized_text, paragraph, tokens))
            continue
        split_count += 1
        for sentence_index, sentence in enumerate(_split_sentence_aware(paragraph.normalized_text)):
            sentence_tokens = _token_count(sentence)
            if sentence_tokens <= config.child_hard_max_tokens:
                units.append(_ChunkUnit(sentence, paragraph, sentence_tokens, sentence_index))
            else:
                units.extend(
                    _ChunkUnit(part, paragraph, _token_count(part), sentence_index)
                    for part in _split_by_tokens(sentence, config.child_hard_max_tokens)
                )
    return units, split_count


def _build_child_chunks(
    *,
    document_id: str,
    units: list[_ChunkUnit],
    config: HierarchicalChunkConfig,
) -> tuple[list[RetrievalChildChunk], int]:
    chunks: list[RetrievalChildChunk] = []
    current: list[_ChunkUnit] = []
    merged_short = 0
    for unit in units:
        if not current:
            current = [unit]
            continue
        same_section = all(item.paragraph.section_type == unit.paragraph.section_type for item in current)
        proposed_tokens = sum(item.token_count for item in current) + unit.token_count
        if same_section and proposed_tokens <= config.child_target_max_tokens:
            if unit.token_count < config.min_short_paragraph_tokens:
                merged_short += 1
            current.append(unit)
            continue
        if same_section and sum(item.token_count for item in current) < config.child_target_min_tokens:
            if proposed_tokens <= config.child_hard_max_tokens:
                current.append(unit)
                continue
        chunks.append(_make_child_chunk(document_id, len(chunks), current))
        overlap = _overlap_units(current, unit, config)
        current = [*overlap, unit] if overlap else [unit]
    if current:
        chunks.append(_make_child_chunk(document_id, len(chunks), current))
    return chunks, merged_short


def _overlap_units(
    previous: list[_ChunkUnit],
    next_unit: _ChunkUnit,
    config: HierarchicalChunkConfig,
) -> list[_ChunkUnit]:
    if not previous:
        return []
    last = previous[-1]
    if last.paragraph.section_type != next_unit.paragraph.section_type:
        return []
    if last.token_count + next_unit.token_count > config.child_hard_max_tokens:
        return []
    return [last]


def _make_child_chunk(
    document_id: str,
    chunk_index: int,
    units: list[_ChunkUnit],
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
            chunk_type="child",
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
        heading_context=units[-1].paragraph.heading_context,
        metadata={
            "section_type": units[0].paragraph.section_type.value,
            "paragraph_ids": paragraph_ids,
            "source_order": min(unit.paragraph.source_order for unit in units),
        },
    )


def _make_parent_window(
    document_id: str,
    window_index: int,
    child: RetrievalChildChunk,
    paragraphs: list[LegalParagraph],
    *,
    config: HierarchicalChunkConfig,
) -> ParentEvidenceWindow:
    paragraph_ids = [paragraph.paragraph_id for paragraph in paragraphs]
    text = _paragraph_text(paragraphs)
    truncated = False
    start_offset = min(paragraph.start_offset for paragraph in paragraphs)
    end_offset = max(paragraph.end_offset for paragraph in paragraphs)
    if _token_count(text) > config.parent_hard_max_tokens:
        text = child.text
        truncated = True
        start_offset = child.start_offset
        end_offset = child.end_offset
    return ParentEvidenceWindow(
        window_id=stable_chunk_id(
            document_id=document_id,
            chunk_index=window_index,
            paragraph_ids=[child.chunk_id, *paragraph_ids],
            chunk_type="parent",
        ),
        document_id=document_id,
        text=text,
        token_count=_token_count(text),
        paragraph_ids=paragraph_ids,
        child_chunk_ids=[child.chunk_id],
        section_types=sorted({paragraph.section_type for paragraph in paragraphs}, key=lambda item: item.value),
        start_offset=start_offset,
        end_offset=end_offset,
        truncated=truncated,
    )


def _split_sentence_aware(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]
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
            parts.extend(_split_piece_by_token_spans(piece, max_tokens))
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


def _split_piece_by_token_spans(text: str, max_tokens: int) -> list[str]:
    matches = list(_TOKEN_RE.finditer(text))
    if not matches:
        return [text]
    parts: list[str] = []
    for index in range(0, len(matches), max_tokens):
        group = matches[index : index + max_tokens]
        parts.append(text[group[0].start() : group[-1].end()])
    return parts


def _token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


def _paragraph_text(paragraphs: list[LegalParagraph]) -> str:
    return "\n\n".join(paragraph.normalized_text for paragraph in paragraphs)


def _average(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0
