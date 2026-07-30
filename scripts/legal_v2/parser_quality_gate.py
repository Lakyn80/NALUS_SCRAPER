from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.adapters import LegalAdapterRegistry, LegalSourceDocument  # noqa: E402
from app.rag.legal_v2.chunking import build_hierarchical_chunks  # noqa: E402
from app.rag.legal_v2.models import SectionType  # noqa: E402
from app.rag.legal_v2.sources import discover_source_documents  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a human-reviewable Legal v2 parser QA artifact.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "artifacts/legal_v2/parser_quality_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/legal_v2/parser_quality_gate")
    parser.add_argument("--limit", type=int, default=12)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    documents = _select_representative_documents(_qa_candidate_documents(args.limit), limit=args.limit)
    registry = LegalAdapterRegistry()
    statuses = _load_statuses(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reviewed = []
    lines = [
        "# Legal Retrieval v2 parser quality gate",
        "",
        "Generated samples are evidence for manual review. Items remain `needs_review` unless the review manifest explicitly marks them otherwise.",
        "",
    ]
    for document in documents:
        parsed = registry.adapter_for(document.source).parse(document)
        chunked = build_hierarchical_chunks(parsed)
        status = statuses.get(document.document_id, "needs_review")
        categories = _categories(document, parsed, chunked)
        review = {
            "document_id": document.document_id,
            "court": parsed.metadata.get("court") or document.source,
            "source": document.source,
            "source_path": document.origin_path,
            "categories": categories,
            "beginning_correctly_parsed": "needs_review",
            "end_correctly_parsed": "needs_review",
            "headings_correctly_classified": "needs_review",
            "numbered_paragraphs_preserved": "needs_review",
            "legal_reasoning_preserved": "needs_review",
            "boilerplate_classification_acceptable": "needs_review",
            "reconstruction_identical_or_explainably_normalized": "needs_review",
            "child_chunks_preserve_meaning": "needs_review",
            "parent_evidence_windows_preserve_context": "needs_review",
            "no_cross_document_mixing": "needs_review",
            "review_status": status,
            "review_reason": "Generated parser and chunk evidence requires human confirmation.",
        }
        reviewed.append(review)
        lines.extend(
            [
                f"## {document.document_id}",
                "",
                f"- Court: `{review['court']}`",
                f"- Source: `{document.source}`",
                f"- Source path: `{document.origin_path}`",
                f"- Categories: `{', '.join(categories)}`",
                f"- Review status: `{status}`",
                f"- Review reason: {review['review_reason']}",
                f"- Paragraphs: {len(parsed.paragraphs)}",
                f"- Child chunks: {len(chunked.child_chunks)}",
                f"- Parent windows: {len(chunked.parent_windows)}",
                f"- Reconstruction identical or explainably normalized: `{review['reconstruction_identical_or_explainably_normalized']}`",
                f"- No cross-document mixing: `{review['no_cross_document_mixing']}`",
                "",
                "### Beginning",
                "",
            ]
        )
        for paragraph in parsed.paragraphs[:5]:
            lines.append(_paragraph_line(paragraph))
        lines.extend(["", "### End", ""])
        for paragraph in parsed.paragraphs[-5:]:
            lines.append(_paragraph_line(paragraph))
        lines.extend(
            [
                "",
                "### Parsed Sections",
                "",
            ]
        )
        for paragraph in parsed.paragraphs[:80]:
            lines.append(_paragraph_line(paragraph))
        lines.extend(["", "### Child Chunks", ""])
        for chunk in chunked.child_chunks[:40]:
            lines.append(f"- `{chunk.chunk_id}` paragraphs={chunk.paragraph_ids} tokens={chunk.token_count}")
        lines.extend(["", "### Parent Windows", ""])
        for window in chunked.parent_windows[:40]:
            lines.append(f"- `{window.window_id}` paragraphs={window.paragraph_ids} tokens={window.token_count}")
        lines.append("")
    status_counts = {status: sum(1 for item in reviewed if item["review_status"] == status) for status in ("approved", "rejected", "needs_review")}
    payload = {
        "summary": {
            "status": "needs_review" if status_counts["needs_review"] else "reviewed",
            "reviewed_documents": len(reviewed),
            "approved": status_counts["approved"],
            "rejected": status_counts["rejected"],
            "needs_review": status_counts["needs_review"],
            "selection_categories": sorted({category for item in reviewed for category in item["categories"]}),
        },
        "documents": reviewed,
    }
    (args.output_dir / "parser_quality_gate.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "parser_quality_gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_dir / "parser_quality_gate.md")
    return 0


def _load_statuses(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    statuses = {}
    for item in data.get("documents", []):
        status = str(item.get("review_status") or "needs_review")
        if status not in {"approved", "rejected", "needs_review"}:
            status = "needs_review"
        statuses[str(item.get("document_id"))] = status
    return statuses


def _snippet(text: str, limit: int = 260) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 3] + "..."


def _paragraph_line(paragraph) -> str:
    return (
        f"- `{paragraph.paragraph_id}` `{paragraph.section_type.value}` "
        f"boilerplate={paragraph.is_boilerplate} citation={paragraph.is_citation_block}: "
        f"{_snippet(paragraph.normalized_text)}"
    )


def _select_representative_documents(documents, *, limit: int) -> list:
    if limit <= 0:
        return []
    selected: list = []
    seen: set[str] = set()

    def add(document) -> None:
        if len(selected) < limit and document.document_id not in seen:
            selected.append(document)
            seen.add(document.document_id)

    for source in ("constitutional", "supreme"):
        candidates = [document for document in documents if document.source == source]
        if not candidates:
            continue
        by_length = sorted(candidates, key=lambda item: len(item.text))
        add(by_length[0])
        long_bounded = next((document for document in by_length if 5000 <= len(document.text.split()) <= 15000), None)
        add(long_bounded or by_length[-1])
        older = sorted(candidates, key=lambda item: _year(item.metadata) or 9999)
        recent = sorted(candidates, key=lambda item: _year(item.metadata) or 0, reverse=True)
        add(older[0])
        add(recent[0])

    registry = LegalAdapterRegistry()
    wanted = {
        "numbered_paragraphs",
        "damaged_formatting",
        "citations",
        "long_factual_section",
        "long_legal_reasoning",
        "boilerplate",
    }
    covered: set[str] = set()
    for document in documents:
        if len(selected) >= limit or wanted.issubset(covered):
            break
        try:
            parsed = registry.adapter_for(document.source).parse(document)
            chunked = build_hierarchical_chunks(parsed)
        except Exception:  # noqa: BLE001 - QA selection skips malformed candidates.
            continue
        categories = set(_categories(document, parsed, chunked))
        if categories.intersection(wanted - covered):
            add(document)
            covered.update(categories)

    for document in documents:
        if len(selected) >= limit:
            break
        add(document)
    return selected


def _qa_candidate_documents(limit: int) -> list:
    nalus_limit = max(500, limit * 20)
    nsoud_limit = max(50, limit)
    documents = discover_source_documents(limit=nalus_limit)
    for year in (1993, 2005, 2016, 2024, 2026):
        documents.extend(_load_nalus_year_candidates(year, per_year_limit=20))
    documents.extend(
        discover_source_documents(
            batches_dir=PROJECT_ROOT / ".missing-legal-v2-nalus-batches-for-nsoud-only",
            limit=nsoud_limit,
        )
    )
    return documents


def _load_nalus_year_candidates(year: int, *, per_year_limit: int) -> list[LegalSourceDocument]:
    files = sorted((PROJECT_ROOT / "batches").glob(f"year_{year}_*.json"))
    documents: list[LegalSourceDocument] = []
    seen: set[str] = set()
    for file_path in files:
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            document_id = _document_identity(item)
            text = str(item.get("full_text") or "").strip()
            if not document_id or not text or document_id in seen:
                continue
            metadata = dict(item)
            metadata["source"] = "constitutional"
            documents.append(
                LegalSourceDocument(
                    document_id=document_id,
                    source="constitutional",
                    text=text,
                    metadata=metadata,
                    origin_path=str(file_path),
                )
            )
            seen.add(document_id)
            if len(documents) >= per_year_limit:
                return documents
    return documents


def _categories(document, parsed, chunked) -> list[str]:
    categories: set[str] = {document.source}
    if len(document.text.split()) < 500:
        categories.add("short_judgment")
    if len(document.text.split()) > 5000:
        categories.add("long_judgment")
    if parsed.diagnostics.numbered_paragraph_count:
        categories.add("numbered_paragraphs")
    if parsed.diagnostics.damaged_formatting_detected:
        categories.add("damaged_formatting")
    if parsed.diagnostics.citation_block_count:
        categories.add("citations")
    if parsed.diagnostics.boilerplate_count:
        categories.add("boilerplate")
    if any(chunk.section_type == SectionType.FACTS and chunk.token_count > 300 for chunk in chunked.child_chunks):
        categories.add("long_factual_section")
    reasoning_tokens = sum(
        chunk.token_count
        for chunk in chunked.child_chunks
        if chunk.section_type == SectionType.COURT_REASONING
    )
    raw_reasoning_tokens = _raw_reasoning_token_count(document.text)
    if reasoning_tokens > 100 or raw_reasoning_tokens > 300:
        categories.add("long_legal_reasoning")
    year = _year(document.metadata)
    if year is not None and year <= 2005:
        categories.add("older_decision")
    if year is not None and year >= 2024:
        categories.add("recent_decision")
    return sorted(categories)


def _document_identity(item: dict) -> str:
    for key in ("ecli", "source_document_id", "document_id", "case_reference", "spisova_znacka", "result_id"):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return ""


def _raw_reasoning_token_count(text: str) -> int:
    marker = "odůvodnění"
    normalized = text.lower()
    index = normalized.find(marker)
    if index < 0:
        index = normalized.find("oduvodneni")
    if index < 0:
        return 0
    return len(text[index:].split())


def _year(metadata: dict) -> int | None:
    for key in ("decision_date", "date", "publication_date", "scraped_at"):
        value = str(metadata.get(key) or "")
        for part in value.replace(".", " ").replace("-", " ").split():
            if part.isdigit() and 1900 <= int(part) <= 2099:
                return int(part)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
