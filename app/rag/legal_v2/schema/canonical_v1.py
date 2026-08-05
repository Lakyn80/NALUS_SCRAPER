"""Canonical document → block → child → parent schema v1.

Phase 2 experiment/pilot contract. Does not replace production Qdrant payloads.
See docs/architecture/CANONICAL_BLOCK_CHUNK_SCHEMA_V1.md.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Mapping

SCHEMA_VERSION = "nalus-canonical-block-chunk.v1"
DEFAULT_CHUNKING_PROFILE = "legal_v2_hierarchical_parent_child_v1"
DEFAULT_PARSER_PROFILE = "legal-decision-parser.cz-courts.v7"


def content_checksum(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_block_id(
    *,
    document_id: str,
    block_index: int,
    normalized_text: str,
    document_version: str | None = None,
) -> str:
    payload = "|".join(
        [
            document_id,
            str(block_index),
            document_version or "",
            normalized_text,
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{document_id}:b:{block_index:05d}:{digest}"


def stable_child_chunk_id(
    *,
    document_id: str,
    chunk_index: int,
    source_block_ids: list[str],
    chunking_profile: str,
) -> str:
    payload = "|".join(
        [document_id, "child", chunking_profile, str(chunk_index), *source_block_ids]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{document_id}:c:{chunk_index:05d}:{digest}"


def stable_parent_id(
    *,
    document_id: str,
    parent_index: int,
    child_ids: list[str],
    chunking_profile: str,
) -> str:
    payload = "|".join(
        [document_id, "parent", chunking_profile, str(parent_index), *child_ids]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{document_id}:p:{parent_index:05d}:{digest}"


@dataclass(frozen=True)
class CanonicalDocument:
    document_id: str
    source_document_id: str | None = None
    ecli: str | None = None
    case_number: str | None = None
    court: str | None = None
    court_chamber: str | None = None
    decision_type: str | None = None
    decision_date: str | None = None
    jurisdiction: str = "CZ"
    language: str = "cs"
    source_url: str | None = None
    source_checksum: str | None = None
    parser_profile: str = DEFAULT_PARSER_PROFILE


@dataclass(frozen=True)
class CanonicalBlock:
    block_id: str
    document_id: str
    block_index: int
    raw_text: str
    normalized_text: str
    primary_class: str
    source_checksum: str
    line_start: int | None = None
    line_end: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    all_line_classes: list[str] = field(default_factory=list)
    section_path: list[str] = field(default_factory=list)
    heading_context: list[str] = field(default_factory=list)
    paragraph_number: str | None = None
    hierarchy_level: int | None = None
    parent_block_id: str | None = None
    citations: list[str] = field(default_factory=list)
    statutes: list[str] = field(default_factory=list)
    case_references: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.block_index < 0:
            raise ValueError("block_index must be >= 0.")
        if self.line_start is not None and self.line_end is not None and self.line_start > self.line_end:
            raise ValueError("line_start must be <= line_end.")
        if self.start_offset is not None and self.end_offset is not None and self.start_offset > self.end_offset:
            raise ValueError("start_offset must be <= end_offset.")


@dataclass(frozen=True)
class CanonicalChildChunk:
    chunk_id: str
    document_id: str
    source_block_ids: list[str]
    chunk_text: str
    embedding_text: str
    token_count: int
    chunking_profile: str
    content_checksum: str
    line_start: int | None = None
    line_end: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    section_path: list[str] = field(default_factory=list)
    heading_context: list[str] = field(default_factory=list)
    primary_paragraph_number: str | None = None
    parent_id: str | None = None

    def __post_init__(self) -> None:
        if not self.source_block_ids:
            raise ValueError("source_block_ids must not be empty.")
        if self.token_count < 0:
            raise ValueError("token_count must be >= 0.")
        if not self.chunking_profile.strip():
            raise ValueError("chunking_profile must be non-empty.")


@dataclass(frozen=True)
class CanonicalParentContext:
    parent_id: str
    document_id: str
    child_ids: list[str]
    parent_text: str
    context_type: str
    token_count: int
    content_checksum: str
    line_start: int | None = None
    line_end: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    section_path: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.child_ids:
            raise ValueError("child_ids must not be empty.")
        if self.token_count < 0:
            raise ValueError("token_count must be >= 0.")
        if not self.context_type.strip():
            raise ValueError("context_type must be non-empty.")


@dataclass(frozen=True)
class CanonicalDocumentBundle:
    document: CanonicalDocument
    blocks: list[CanonicalBlock]
    children: list[CanonicalChildChunk] = field(default_factory=list)
    parents: list[CanonicalParentContext] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    chunking_profile: str | None = None

    def blocks_by_id(self) -> dict[str, CanonicalBlock]:
        return {block.block_id: block for block in self.blocks}

    def children_by_id(self) -> dict[str, CanonicalChildChunk]:
        return {child.chunk_id: child for child in self.children}

    def parents_by_id(self) -> dict[str, CanonicalParentContext]:
        return {parent.parent_id: parent for parent in self.parents}


@dataclass(frozen=True)
class ReconstructionReport:
    ok: bool
    child_reconstruction_failures: list[str] = field(default_factory=list)
    parent_child_inconsistencies: list[str] = field(default_factory=list)
    duplicate_ids: list[str] = field(default_factory=list)
    cross_document_refs: list[str] = field(default_factory=list)
    missing_block_refs: list[str] = field(default_factory=list)

    @property
    def failure_count(self) -> int:
        return (
            len(self.child_reconstruction_failures)
            + len(self.parent_child_inconsistencies)
            + len(self.duplicate_ids)
            + len(self.cross_document_refs)
            + len(self.missing_block_refs)
        )


def reconstruct_child_text(
    child: CanonicalChildChunk,
    blocks_by_id: Mapping[str, CanonicalBlock],
    *,
    separator: str = "\n\n",
) -> str:
    parts: list[str] = []
    for block_id in child.source_block_ids:
        block = blocks_by_id.get(block_id)
        if block is None:
            raise KeyError(f"missing source block: {block_id}")
        parts.append(block.raw_text)
    return separator.join(parts)


def validate_bundle_invariants(bundle: CanonicalDocumentBundle) -> ReconstructionReport:
    document_id = bundle.document.document_id
    child_failures: list[str] = []
    parent_failures: list[str] = []
    duplicate_ids: list[str] = []
    cross_docs: list[str] = []
    missing_blocks: list[str] = []

    block_ids = [block.block_id for block in bundle.blocks]
    child_ids = [child.chunk_id for child in bundle.children]
    parent_ids = [parent.parent_id for parent in bundle.parents]
    for label, values in (
        ("block", block_ids),
        ("child", child_ids),
        ("parent", parent_ids),
    ):
        seen: set[str] = set()
        for value in values:
            if value in seen:
                duplicate_ids.append(f"{label}:{value}")
            seen.add(value)

    blocks_by_id = bundle.blocks_by_id()
    children_by_id = bundle.children_by_id()

    for block in bundle.blocks:
        if block.document_id != document_id:
            cross_docs.append(f"block:{block.block_id}")

    for child in bundle.children:
        if child.document_id != document_id:
            cross_docs.append(f"child:{child.chunk_id}")
            continue
        missing = [block_id for block_id in child.source_block_ids if block_id not in blocks_by_id]
        if missing:
            missing_blocks.append(f"{child.chunk_id}:{','.join(missing)}")
            continue
        try:
            assembled = reconstruct_child_text(child, blocks_by_id)
        except KeyError as exc:
            child_failures.append(f"{child.chunk_id}:{exc}")
            continue
        if any(blocks_by_id[block_id].raw_text.strip() for block_id in child.source_block_ids):
            if not assembled.strip():
                child_failures.append(f"{child.chunk_id}:empty_reconstruction")
        if len(child.source_block_ids) != len(set(child.source_block_ids)):
            child_failures.append(f"{child.chunk_id}:duplicate_source_block_ids")
        if content_checksum(child.chunk_text) != child.content_checksum:
            child_failures.append(f"{child.chunk_id}:content_checksum_mismatch")
        # Chunk text may use normalized units; require each source block's
        # normalized or raw text to appear after whitespace normalization.
        chunk_norm = _normalize_ws(child.chunk_text)
        for block_id in child.source_block_ids:
            block = blocks_by_id[block_id]
            candidates = [
                _normalize_ws(block.raw_text),
                _normalize_ws(block.normalized_text),
            ]
            if any(candidate and candidate in chunk_norm for candidate in candidates):
                continue
            # Split overlong paragraphs may place only a sentence in the child.
            if block.normalized_text and any(
                _normalize_ws(part) in chunk_norm
                for part in block.normalized_text.split(".")
                if len(part.strip()) >= 24
            ):
                continue
            child_failures.append(f"{child.chunk_id}:missing_block_text:{block_id}")
            break

    child_to_parents: dict[str, list[str]] = {}
    for parent in bundle.parents:
        if parent.document_id != document_id:
            cross_docs.append(f"parent:{parent.parent_id}")
            continue
        if content_checksum(parent.parent_text) != parent.content_checksum:
            parent_failures.append(f"{parent.parent_id}:content_checksum_mismatch")
        for child_id in parent.child_ids:
            child = children_by_id.get(child_id)
            if child is None:
                parent_failures.append(f"{parent.parent_id}:missing_child:{child_id}")
                continue
            if child.document_id != document_id:
                cross_docs.append(f"parent:{parent.parent_id}:child:{child_id}")
                continue
            child_to_parents.setdefault(child_id, []).append(parent.parent_id)
            if child.parent_id is not None and child.parent_id != parent.parent_id:
                parent_failures.append(
                    f"{parent.parent_id}:child_parent_mismatch:{child_id}->{child.parent_id}"
                )

    for child in bundle.children:
        if child.parent_id is None:
            continue
        claimed = child_to_parents.get(child.chunk_id, [])
        if child.parent_id not in claimed:
            parent_failures.append(
                f"{child.chunk_id}:parent_not_claiming_child:{child.parent_id}"
            )

    ok = not (
        child_failures or parent_failures or duplicate_ids or cross_docs or missing_blocks
    )
    return ReconstructionReport(
        ok=ok,
        child_reconstruction_failures=child_failures,
        parent_child_inconsistencies=parent_failures,
        duplicate_ids=duplicate_ids,
        cross_document_refs=cross_docs,
        missing_block_refs=missing_blocks,
    )


def bundle_to_dict(bundle: CanonicalDocumentBundle) -> dict[str, Any]:
    return {
        "schema_version": bundle.schema_version,
        "chunking_profile": bundle.chunking_profile,
        "document": _to_plain(bundle.document),
        "blocks": [_to_plain(block) for block in bundle.blocks],
        "children": [_to_plain(child) for child in bundle.children],
        "parents": [_to_plain(parent) for parent in bundle.parents],
    }


def bundle_from_dict(payload: Mapping[str, Any]) -> CanonicalDocumentBundle:
    document = CanonicalDocument(**dict(payload["document"]))
    blocks = [CanonicalBlock(**dict(item)) for item in payload.get("blocks", [])]
    children = [CanonicalChildChunk(**dict(item)) for item in payload.get("children", [])]
    parents = [CanonicalParentContext(**dict(item)) for item in payload.get("parents", [])]
    return CanonicalDocumentBundle(
        document=document,
        blocks=blocks,
        children=children,
        parents=parents,
        schema_version=str(payload.get("schema_version") or SCHEMA_VERSION),
        chunking_profile=payload.get("chunking_profile"),
    )


def bundle_to_json(bundle: CanonicalDocumentBundle, *, indent: int = 2) -> str:
    return json.dumps(bundle_to_dict(bundle), ensure_ascii=False, indent=indent, sort_keys=True)


def bundle_from_json(raw: str) -> CanonicalDocumentBundle:
    return bundle_from_dict(json.loads(raw))


def _normalize_ws(value: str) -> str:
    return " ".join(value.split())


def _to_plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _to_plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    return value


__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_CHUNKING_PROFILE",
    "DEFAULT_PARSER_PROFILE",
    "CanonicalDocument",
    "CanonicalBlock",
    "CanonicalChildChunk",
    "CanonicalParentContext",
    "CanonicalDocumentBundle",
    "ReconstructionReport",
    "content_checksum",
    "stable_block_id",
    "stable_child_chunk_id",
    "stable_parent_id",
    "reconstruct_child_text",
    "validate_bundle_invariants",
    "bundle_to_dict",
    "bundle_from_dict",
    "bundle_to_json",
    "bundle_from_json",
]
