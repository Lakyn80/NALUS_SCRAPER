from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.rag.legal_v2.ingest.adapters import LegalAdapterRegistry, LegalSourceDocument
from app.rag.legal_v2.ingest.chunking import HierarchicalChunkConfig, build_hierarchical_chunks

PARSER_VERSION = "legal-paragraph-parser.v3"
CHUNKER_VERSION = "legal_v2_hierarchical_chunker_v1"


@dataclass(frozen=True)
class ParsedDocumentAudit:
    document_id: str
    source: str
    adapter: str
    status: str
    reasons: list[str] = field(default_factory=list)
    paragraph_count: int = 0
    child_chunk_count: int = 0
    parent_window_count: int = 0
    boilerplate_count: int = 0
    citation_block_count: int = 0
    fallback_parser_count: int = 0
    reconstruction_failures: int = 0
    offset_failures: int = 0
    boundary_violations: int = 0
    overlong_paragraphs: int = 0
    overlong_chunks: int = 0
    empty_chunks: int = 0
    duplicate_ids: int = 0
    token_stats: dict[str, float] = field(default_factory=dict)
    section_distribution: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class CorpusAuditReport:
    summary: dict[str, Any]
    documents: list[ParsedDocumentAudit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": dict(self.summary),
            "documents": [asdict(document) for document in self.documents],
        }


def audit_documents(
    documents: list[LegalSourceDocument],
    *,
    config: HierarchicalChunkConfig | None = None,
    registry: LegalAdapterRegistry | None = None,
) -> CorpusAuditReport:
    started_at = _utc_now()
    config = config or HierarchicalChunkConfig()
    registry = registry or LegalAdapterRegistry()
    audits: list[ParsedDocumentAudit] = []
    for source_document in documents:
        audits.append(_audit_one(source_document, config=config, registry=registry))
    finished_at = _utc_now()
    summary = _summary(audits)
    summary.update(
        {
            "schema": "legal_v2_parse_only_audit",
            "status": "pass" if summary["failed_documents"] == 0 else "fail",
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": _duration_ms(started_at, finished_at),
            "parser_version": PARSER_VERSION,
            "chunker_version": CHUNKER_VERSION,
            "deepseek_calls": 0,
            "embeddings_created": 0,
            "qdrant_writes": 0,
            "bm25_writes": 0,
        }
    )
    return CorpusAuditReport(summary=summary, documents=audits)


def write_audit_report(report: CorpusAuditReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "legal_v2_parse_audit.json"
    markdown_path = output_dir / "legal_v2_parse_audit.md"
    payload = report.to_dict()
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    return json_path, markdown_path


def _audit_one(
    document: LegalSourceDocument,
    *,
    config: HierarchicalChunkConfig,
    registry: LegalAdapterRegistry,
) -> ParsedDocumentAudit:
    adapter = registry.adapter_for(document.source)
    try:
        parsed = adapter.parse(document)
        chunked = build_hierarchical_chunks(parsed, config=config)
        reasons = _validate_structure(document.text, parsed, chunked, config=config)
        status = "pass" if not reasons else "fail"
        token_counts = [chunk.token_count for chunk in chunked.child_chunks]
        return ParsedDocumentAudit(
            document_id=document.document_id,
            source=document.source,
            adapter=adapter.source_name,
            status=status,
            reasons=reasons,
            paragraph_count=len(parsed.paragraphs),
            child_chunk_count=len(chunked.child_chunks),
            parent_window_count=len(chunked.parent_windows),
            boilerplate_count=parsed.diagnostics.boilerplate_count,
            citation_block_count=parsed.diagnostics.citation_block_count,
            fallback_parser_count=parsed.diagnostics.fallback_paragraphs_created,
            reconstruction_failures=int("reconstruction_order_mismatch" in reasons),
            offset_failures=sum(1 for reason in reasons if reason.startswith("offset_")),
            boundary_violations=sum(1 for reason in reasons if reason.startswith("boundary_")),
            overlong_paragraphs=sum(
                1 for paragraph in parsed.paragraphs if len(paragraph.normalized_text.split()) > config.child_hard_max_tokens
            ),
            overlong_chunks=sum(1 for chunk in chunked.child_chunks if chunk.token_count > config.child_hard_max_tokens),
            empty_chunks=sum(1 for chunk in chunked.child_chunks if not chunk.text.strip()),
            duplicate_ids=_duplicate_count(
                [p.paragraph_id for p in parsed.paragraphs]
                + [c.chunk_id for c in chunked.child_chunks]
                + [w.window_id for w in chunked.parent_windows]
            ),
            token_stats={
                "min": float(min(token_counts)) if token_counts else 0.0,
                "max": float(max(token_counts)) if token_counts else 0.0,
                "avg": sum(token_counts) / len(token_counts) if token_counts else 0.0,
            },
            section_distribution=dict(chunked.diagnostics.section_distribution),
        )
    except Exception as exc:  # noqa: BLE001 - audit reports exceptions as failed documents.
        return ParsedDocumentAudit(
            document_id=document.document_id,
            source=document.source,
            adapter=getattr(adapter, "source_name", "unknown"),
            status="fail",
            reasons=[f"exception:{exc.__class__.__name__}"],
        )


def _validate_structure(document_text: str, parsed: Any, chunked: Any, *, config: HierarchicalChunkConfig) -> list[str]:
    reasons: list[str] = []
    if not parsed.paragraphs:
        reasons.append("empty_or_malformed_document")
    paragraph_ids = [paragraph.paragraph_id for paragraph in parsed.paragraphs]
    if _duplicate_count(paragraph_ids):
        reasons.append("duplicate_paragraph_ids")
    if [paragraph.paragraph_index for paragraph in parsed.paragraphs] != list(range(len(parsed.paragraphs))):
        reasons.append("paragraph_indexes_not_ordered")
    normalized_source_len = len(parsed.normalized_text)
    previous_end = -1
    for paragraph in parsed.paragraphs:
        if paragraph.start_offset < 0 or paragraph.end_offset < paragraph.start_offset or paragraph.end_offset > normalized_source_len:
            reasons.append("offset_invalid")
        if paragraph.start_offset < previous_end:
            reasons.append("offset_overlap")
        previous_end = max(previous_end, paragraph.end_offset)
    reconstructed = parsed.reconstruct_text()
    if parsed.paragraphs and not all(paragraph.original_text in reconstructed for paragraph in parsed.paragraphs[:10]):
        reasons.append("reconstruction_order_mismatch")
    known_paragraph_ids = set(paragraph_ids)
    for chunk in chunked.child_chunks:
        if not chunk.text.strip():
            reasons.append("empty_chunk")
        if chunk.token_count > config.child_hard_max_tokens:
            reasons.append("boundary_overlong_chunk")
        if not set(chunk.paragraph_ids).issubset(known_paragraph_ids):
            reasons.append("boundary_chunk_unknown_paragraph")
    for window in chunked.parent_windows:
        if not set(window.paragraph_ids).issubset(known_paragraph_ids):
            reasons.append("boundary_window_unknown_paragraph")
        for child_id in window.child_chunk_ids:
            child = next((chunk for chunk in chunked.child_chunks if chunk.chunk_id == child_id), None)
            if child is None or not set(child.paragraph_ids).issubset(set(window.paragraph_ids)):
                reasons.append("boundary_parent_window_missing_child")
    if not str(document_text or "").strip():
        reasons.append("empty_source_text")
    return sorted(set(reasons))


def _summary(audits: list[ParsedDocumentAudit]) -> dict[str, Any]:
    return {
        "total_documents": len(audits),
        "successfully_parsed_documents": sum(1 for item in audits if item.status == "pass"),
        "blocked_documents": sum(1 for item in audits if item.status == "blocked"),
        "failed_documents": sum(1 for item in audits if item.status == "fail"),
        "paragraph_count": sum(item.paragraph_count for item in audits),
        "child_chunk_count": sum(item.child_chunk_count for item in audits),
        "parent_window_count": sum(item.parent_window_count for item in audits),
        "section_distribution": _merge_counts(item.section_distribution for item in audits),
        "boilerplate_count": sum(item.boilerplate_count for item in audits),
        "citation_block_count": sum(item.citation_block_count for item in audits),
        "fallback_parser_count": sum(item.fallback_parser_count for item in audits),
        "reconstruction_failures": sum(item.reconstruction_failures for item in audits),
        "offset_failures": sum(item.offset_failures for item in audits),
        "boundary_violations": sum(item.boundary_violations for item in audits),
        "overlong_paragraphs": sum(item.overlong_paragraphs for item in audits),
        "overlong_chunks": sum(item.overlong_chunks for item in audits),
        "empty_chunks": sum(item.empty_chunks for item in audits),
        "duplicate_ids": sum(item.duplicate_ids for item in audits),
        "documents_excluded_from_indexing": [
            {"document_id": item.document_id, "reasons": item.reasons}
            for item in audits
            if item.status != "pass"
        ],
    }


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Legal Retrieval v2 parse-only audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Total documents: {summary['total_documents']}",
        f"- Successfully parsed: {summary['successfully_parsed_documents']}",
        f"- Failed: {summary['failed_documents']}",
        f"- Blocked: {summary['blocked_documents']}",
        f"- Paragraphs: {summary['paragraph_count']}",
        f"- Child chunks: {summary['child_chunk_count']}",
        f"- Parent windows: {summary['parent_window_count']}",
        f"- DeepSeek calls: {summary['deepseek_calls']}",
        f"- Embeddings created: {summary['embeddings_created']}",
        f"- Qdrant writes: {summary['qdrant_writes']}",
        "",
        "## Excluded documents",
        "",
    ]
    excluded = summary["documents_excluded_from_indexing"]
    if not excluded:
        lines.append("- None")
    else:
        for item in excluded[:100]:
            lines.append(f"- `{item['document_id']}`: {', '.join(item['reasons'])}")
    return "\n".join(lines) + "\n"


def _merge_counts(values: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        for key, count in value.items():
            result[key] = result.get(key, 0) + int(count)
    return result


def _duplicate_count(values: list[str]) -> int:
    return len(values) - len(set(values))


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _duration_ms(started_at: str, finished_at: str) -> float:
    started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    finished = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
    return (finished - started).total_seconds() * 1000

