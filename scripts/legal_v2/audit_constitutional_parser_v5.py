from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.audit import PARSER_VERSION  # noqa: E402
from app.rag.legal_v2.ingest.parser import parse_legal_document  # noqa: E402
from scripts.legal_v2.parser_review.manifest import load_design_documents  # noqa: E402
from scripts.legal_v2.parser_review.snapshot import (  # noqa: E402
    _boundary_before,
    _line_class,
    _line_offsets,
    _paragraph_for_line,
    _raw_lines,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "legal_v2" / "constitutional_parser_v5"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Constitutional Court parser v5 against a saved v4 baseline.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-json", type=Path, default=DEFAULT_OUTPUT_DIR / "constitutional_v4_baseline.json")
    args = parser.parse_args(argv)
    result = build_audit(args.output_dir, args.baseline_json)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["summary"]["status"] == "pass" else 1


def build_audit(output_dir: Path, baseline_json: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = _load_baseline(baseline_json)
    _, design_documents = load_design_documents()
    documents = [document for document in design_documents if document.court == "constitutional_court"]
    if len(documents) != 10:
        raise ValueError(f"Expected 10 Constitutional Court design documents, found {len(documents)}")
    current = [_document_result(document) for document in documents]
    by_source = {document["source_id"]: document for document in baseline["documents"]}
    changed_boundaries: list[dict[str, Any]] = []
    changed_classes: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for new_doc in current:
        old_doc = by_source[new_doc["source_id"]]
        boundary_changes = _changed_boundaries(old_doc, new_doc)
        class_changes = _changed_classes(old_doc, new_doc)
        changed_boundaries.extend(boundary_changes)
        changed_classes.extend(class_changes)
        comparisons.append(
            {
                "document_id": new_doc["document_id"],
                "source_id": new_doc["source_id"],
                "review_number": new_doc["review_number"],
                "raw_line_count": new_doc["raw_line_count"],
                "old_block_count": old_doc["block_count"],
                "new_block_count": new_doc["block_count"],
                "changed_boundaries": len(boundary_changes),
                "changed_classes": len(class_changes),
                "text_conservation": new_doc["conservation"],
                "duplication_failures": new_doc["duplication_failures"],
                "ordering_failures": new_doc["ordering_failures"],
                "suspicious_overmerges": _suspicious_overmerges(new_doc),
                "suspicious_undersplits": _suspicious_undersplits(new_doc),
            }
        )
    doc2 = next(document for document in current if document["review_number"] == 2)
    summary = _summary(baseline, current, comparisons, changed_boundaries, changed_classes, doc2)
    payload = {
        "schema_version": "constitutional-parser-v5-audit.v1",
        "old_parser_profile": baseline.get("parser_profile"),
        "new_parser_profile": PARSER_VERSION,
        "summary": summary,
        "documents": comparisons,
    }
    _write_json(output_dir / "document2_golden_result.json", doc2)
    _write_json(output_dir / "constitutional_v4_vs_v5.json", payload)
    _write_jsonl(output_dir / "constitutional_changed_boundaries.jsonl", changed_boundaries)
    _write_jsonl(output_dir / "constitutional_changed_classes.jsonl", changed_classes)
    _write_json(output_dir / "constitutional_v5_acceptance.json", {"summary": summary, "document2": doc2})
    (output_dir / "constitutional_v4_vs_v5.md").write_text(_comparison_markdown(payload), encoding="utf-8")
    (output_dir / "constitutional_v5_acceptance.md").write_text(_acceptance_markdown(summary), encoding="utf-8")
    return payload


def _document_result(document: Any) -> dict[str, Any]:
    raw_lines = _raw_lines(document)
    text = "\n".join(raw_lines)
    parsed = parse_legal_document(
        document_id=document.source_id,
        text=text,
        metadata={"court": document.court, "source_url": document.source_url},
    )
    line_offsets = _line_offsets(raw_lines)
    block_for_line = [_paragraph_for_line(parsed.paragraphs, start, end) for start, end in line_offsets]
    line_rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.strip():
            continue
        paragraph = block_for_line[line_number - 1]
        line_rows.append(
            {
                "line": line_number,
                "text": raw_line,
                "block_index": paragraph.paragraph_index if paragraph else None,
                "class": _line_class(raw_line, paragraph, parsed, line_number, block_for_line),
                "boundary_before": _boundary_before(line_number, block_for_line),
            }
        )
    block_ranges: list[list[int]] = []
    for paragraph in parsed.paragraphs:
        line_numbers = [row["line"] for row in line_rows if row["block_index"] == paragraph.paragraph_index]
        block_ranges.append([min(line_numbers), max(line_numbers)])
    source = _non_whitespace(text)
    reconstructed = _non_whitespace(parsed.reconstruct_text())
    return {
        "document_id": document.review_id,
        "source_id": document.source_id,
        "review_number": document.review_number,
        "raw_line_count": len([line for line in raw_lines if line.strip()]),
        "block_count": len(parsed.paragraphs),
        "block_ranges": block_ranges,
        "lines": line_rows,
        "conservation": source == reconstructed,
        "duplication_failures": int(source != reconstructed),
        "ordering_failures": int([paragraph.start_offset for paragraph in parsed.paragraphs] != sorted(paragraph.start_offset for paragraph in parsed.paragraphs)),
    }


def _changed_boundaries(old_doc: dict[str, Any], new_doc: dict[str, Any]) -> list[dict[str, Any]]:
    old_by_line = {row["line"]: row for row in old_doc["lines"]}
    changes: list[dict[str, Any]] = []
    for row in new_doc["lines"]:
        line = int(row["line"])
        if line == 1:
            continue
        old = old_by_line[line]
        if bool(old["boundary_before"]) != bool(row["boundary_before"]):
            changes.append(
                {
                    "document_id": new_doc["document_id"],
                    "source_id": new_doc["source_id"],
                    "before_line": line - 1,
                    "after_line": line,
                    "old_boundary": bool(old["boundary_before"]),
                    "new_boundary": bool(row["boundary_before"]),
                    "after_text": row["text"],
                }
            )
    return changes


def _changed_classes(old_doc: dict[str, Any], new_doc: dict[str, Any]) -> list[dict[str, Any]]:
    old_by_line = {row["line"]: row for row in old_doc["lines"]}
    changes: list[dict[str, Any]] = []
    for row in new_doc["lines"]:
        old = old_by_line[row["line"]]
        if old["class"] != row["class"]:
            changes.append(
                {
                    "document_id": new_doc["document_id"],
                    "source_id": new_doc["source_id"],
                    "line": row["line"],
                    "old_class": old["class"],
                    "new_class": row["class"],
                    "text": row["text"],
                }
            )
    return changes


def _suspicious_overmerges(document: dict[str, Any]) -> list[dict[str, Any]]:
    lines = document["lines"]
    by_line = {int(row["line"]): row for row in lines}
    reasons: list[dict[str, Any]] = []
    for line in range(2, len(lines) + 1):
        before = by_line[line - 1]["text"]
        after = by_line[line]["text"]
        boundary = bool(by_line[line]["boundary_before"])
        if _must_split(before, after) and not boundary:
            reasons.append({"line": line, "code": "expected_split_missing", "before": before, "after": after})
    return reasons


def _suspicious_undersplits(document: dict[str, Any]) -> list[dict[str, Any]]:
    lines = document["lines"]
    by_line = {int(row["line"]): row for row in lines}
    reasons: list[dict[str, Any]] = []
    for line in range(2, len(lines) + 1):
        before = by_line[line - 1]["text"]
        after = by_line[line]["text"]
        boundary = bool(by_line[line]["boundary_before"])
        if _must_merge(before, after) and boundary:
            reasons.append({"line": line, "code": "expected_merge_missing", "before": before, "after": after})
    return reasons


def _must_split(before: str, after: str) -> bool:
    return bool(
        _re(r"^NALUS\s*-").match(before) and _re(r"^[IVXLCDM]+\.?\s*ÚS\s+\d+/\d+").match(after)
        or _re(r"^Ústavního soudu$").match(before) and _re(r"^Ústavní soud rozhodl\b").match(after)
        or before.rstrip().endswith("takto:")
        or _re(r"^Odůvodnění:?$").match(before)
        or _re(r"^Poučení:").match(after)
        or _re(r"^V Brně dne").match(after)
        or _re(r"\bv\.\s*r\.$").search(after)
    )


def _must_merge(before: str, after: str) -> bool:
    return bool(
        _re(r"^(?:USNESENÍ|NÁLEZ)$").match(before) and _re(r"^Ústavního soudu$").match(after)
        or _re(r"\bv\.\s*r\.$").search(before) and _re(r"^(?:soudce zpravodaj|soudkyně zpravodajka|předseda senátu|předsedkyně senátu)$").match(after)
    )


def _summary(
    baseline: dict[str, Any],
    current: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    changed_boundaries: list[dict[str, Any]],
    changed_classes: list[dict[str, Any]],
    doc2: dict[str, Any],
) -> dict[str, Any]:
    suspicious_overmerges = sum(len(row["suspicious_overmerges"]) for row in comparisons)
    suspicious_undersplits = sum(len(row["suspicious_undersplits"]) for row in comparisons)
    conservation_failures = sum(1 for row in current if not row["conservation"])
    duplication_failures = sum(int(row["duplication_failures"]) for row in current)
    ordering_failures = sum(int(row["ordering_failures"]) for row in current)
    doc2_pass = doc2["block_count"] == 11 and doc2["block_ranges"] == [[1, 1], [2, 2], [3, 3], [4, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 13]]
    status = "pass" if doc2_pass and not conservation_failures and not duplication_failures and not ordering_failures and not suspicious_overmerges and not suspicious_undersplits else "fail"
    return {
        "status": status,
        "documents": len(current),
        "old_parser_profile": baseline.get("parser_profile"),
        "new_parser_profile": PARSER_VERSION,
        "parser_exceptions": 0,
        "old_blocks": sum(int(row["block_count"]) for row in baseline["documents"]),
        "new_blocks": sum(int(row["block_count"]) for row in current),
        "changed_boundaries": len(changed_boundaries),
        "changed_classes": len(changed_classes),
        "conservation_failures": conservation_failures,
        "duplication_failures": duplication_failures,
        "ordering_failures": ordering_failures,
        "suspicious_overmerges": suspicious_overmerges,
        "suspicious_undersplits": suspicious_undersplits,
        "document2_blocks": doc2["block_count"],
        "document2_block_ranges": doc2["block_ranges"],
        "remaining_review_queue": len(changed_boundaries) + len(changed_classes),
    }


def _comparison_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Constitutional parser v4 vs v5",
        "",
        f"- Status: `{summary['status']}`",
        f"- Documents: `{summary['documents']}`",
        f"- Old blocks: `{summary['old_blocks']}`",
        f"- New blocks: `{summary['new_blocks']}`",
        f"- Changed boundaries: `{summary['changed_boundaries']}`",
        f"- Changed classes: `{summary['changed_classes']}`",
        f"- Remaining review queue: `{summary['remaining_review_queue']}`",
        "",
        "## Documents",
        "",
    ]
    for row in payload["documents"]:
        lines.append(
            f"- `{row['source_id']}`: blocks {row['old_block_count']} -> {row['new_block_count']}; "
            f"boundaries `{row['changed_boundaries']}`, classes `{row['changed_classes']}`"
        )
    return "\n".join(lines) + "\n"


def _acceptance_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Constitutional parser v5 acceptance",
            "",
            f"- Status: `{summary['status']}`",
            f"- Parser exceptions: `{summary['parser_exceptions']}`",
            f"- Conservation failures: `{summary['conservation_failures']}`",
            f"- Duplication failures: `{summary['duplication_failures']}`",
            f"- Ordering failures: `{summary['ordering_failures']}`",
            f"- Suspicious overmerges: `{summary['suspicious_overmerges']}`",
            f"- Suspicious undersplits: `{summary['suspicious_undersplits']}`",
            f"- Document 2 blocks: `{summary['document2_blocks']}`",
            f"- Document 2 ranges: `{summary['document2_block_ranges']}`",
        ]
    ) + "\n"


def _load_baseline(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing v4 baseline: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _non_whitespace(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _re(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


if __name__ == "__main__":
    raise SystemExit(main())
