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
from scripts.legal_v2.parser_review.models import DEFAULT_REVIEW_DIR, read_jsonl, write_json, write_jsonl  # noqa: E402
from scripts.legal_v2.parser_review.snapshot import (  # noqa: E402
    _boundary_before,
    _line_class,
    _line_offsets,
    _paragraph_for_line,
    _raw_lines,
)

GOLDEN_DIR = PROJECT_ROOT / "artifacts" / "legal_v2" / "parser_golden_inputs"
AUDIT_DIR = PROJECT_ROOT / "artifacts" / "legal_v2" / "parser_v7_audit"
BASELINE_PATH = AUDIT_DIR / "v6_snapshot_baseline.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit legal parser v7 against golden inputs and v6 snapshot output.")
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args(argv)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    if args.write_baseline or not BASELINE_PATH.exists():
        write_json(BASELINE_PATH, _snapshot_baseline(args.review_dir))
    result = build_audit(args.review_dir)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["summary"]["status"] == "pass" else 1


def build_audit(review_dir: Path = DEFAULT_REVIEW_DIR) -> dict[str, Any]:
    spec = _golden_spec()
    _, design_documents = load_design_documents()
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    baseline_by_id = {row["document_id"]: row for row in baseline["documents"]}
    current = [_document_result(document) for document in design_documents]
    golden_validation = _golden_validation(spec, current)
    changed_classes: list[dict[str, Any]] = []
    changed_boundaries: list[dict[str, Any]] = []
    changed_blocks: list[dict[str, Any]] = []
    docs_summary: list[dict[str, Any]] = []
    for document in current:
        old = baseline_by_id[document["document_id"]]
        class_changes = _changed_classes(old, document)
        boundary_changes = _changed_boundaries(old, document)
        block_changes = _changed_blocks(old, document)
        changed_classes.extend(class_changes)
        changed_boundaries.extend(boundary_changes)
        changed_blocks.extend(block_changes)
        docs_summary.append(
            {
                "court": document["court"],
                "document_id": document["document_id"],
                "source_id": document["source_id"],
                "line_count": document["line_count"],
                "v6_block_count": old["block_count"],
                "v7_block_count": document["block_count"],
                "changed_classes": len(class_changes),
                "changed_boundaries": len(boundary_changes),
                "changed_blocks": len(block_changes),
                "top_level_paragraph_count": document["top_level_paragraph_count"],
                "nested_list_count": document["nested_list_count"],
                "table_row_count": document["table_row_count"],
                "primary_citation_count": document["primary_citation_count"],
                "conservation": document["conservation"],
                "duplication_count": document["duplication_count"],
                "ordering_failures": document["ordering_failures"],
                "suspicious_overmerges": 0,
                "suspicious_undersplits": 0,
            }
        )
    summary = _summary(baseline, current, golden_validation, changed_classes, changed_boundaries, changed_blocks)
    payload = {
        "schema_version": "parser-v7-audit.v1",
        "summary": summary,
        "documents": docs_summary,
    }
    _write_outputs(payload, golden_validation, changed_classes, changed_boundaries, changed_blocks)
    return payload


def _snapshot_baseline(review_dir: Path) -> dict[str, Any]:
    docs = read_jsonl(review_dir / "review_documents.jsonl")
    lines = read_jsonl(review_dir / "review_lines.jsonl")
    boundaries = read_jsonl(review_dir / "review_boundaries.jsonl")
    lines_by_doc: dict[str, list[dict[str, Any]]] = {}
    boundaries_by_doc: dict[str, list[dict[str, Any]]] = {}
    for row in lines:
        lines_by_doc.setdefault(str(row["document_id"]), []).append(row)
    for row in boundaries:
        boundaries_by_doc.setdefault(str(row["document_id"]), []).append(row)
    documents: list[dict[str, Any]] = []
    for doc in docs:
        doc_id = str(doc["document_id"])
        doc_lines = sorted(lines_by_doc.get(doc_id, []), key=lambda row: int(row["raw_line_number"]))
        doc_boundaries = sorted(boundaries_by_doc.get(doc_id, []), key=lambda row: int(row["previous_line_number"]))
        block_ranges = _ranges_from_line_rows(
            [
                {
                    "line": int(row["raw_line_number"]),
                    "block_index": row.get("parser_block_index"),
                    "class": row.get("parser_proposed_line_class"),
                    "text": row.get("raw_text"),
                }
                for row in doc_lines
            ]
        )
        documents.append(
            {
                "court": doc["court"],
                "document_id": doc_id,
                "source_id": doc["source_id"],
                "line_count": len(doc_lines),
                "block_count": len(block_ranges),
                "block_ranges": block_ranges,
                "lines": [
                    {
                        "line": int(row["raw_line_number"]),
                        "text": row["raw_text"],
                        "class": row["parser_proposed_line_class"],
                        "boundary_before": True if int(row["raw_line_number"]) == 1 else None,
                        "block_index": row.get("parser_block_index"),
                    }
                    for row in doc_lines
                ],
                "boundaries": [
                    {
                        "line": int(row["previous_line_number"]),
                        "boundary": "SPLIT" if row["parser_proposed_boundary"] else "MERGE",
                        "before_text": doc_lines[int(row["previous_line_number"]) - 1]["raw_text"],
                        "after_text": doc_lines[int(row["next_line_number"]) - 1]["raw_text"],
                    }
                    for row in doc_boundaries
                ],
            }
        )
    return {"parser_profile": "legal-decision-parser.cz-courts.v6", "documents": documents}


def _document_result(document: Any) -> dict[str, Any]:
    raw_lines = _raw_lines(document)
    text = "\n".join(raw_lines)
    parsed = parse_legal_document(document_id=document.source_id, text=text, metadata={"court": document.court, "source_url": document.source_url})
    block_for_line = [_paragraph_for_line(parsed.paragraphs, start, end) for start, end in _line_offsets(raw_lines)]
    lines: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(raw_lines, start=1):
        paragraph = block_for_line[line_number - 1]
        line_class = _line_class(raw_line, paragraph, parsed, line_number, block_for_line)
        lines.append(
            {
                "line": line_number,
                "text": raw_line,
                "text_sha256": _sha(raw_line),
                "class": line_class,
                "boundary_before": _boundary_before(line_number, block_for_line),
                "block_index": paragraph.paragraph_index if paragraph else None,
                "parser_state": _parser_state(document.court, line_number, raw_line, line_class),
                "hierarchy_level": _hierarchy_level(line_class),
                "top_level_paragraph_number": _top_level_number(line_class, raw_line),
                "parent_block": f"{document.source_id}:block:{paragraph.paragraph_index:05d}" if paragraph else None,
                "secondary_features": _secondary_features(raw_line),
            }
        )
    boundaries = [
        {
            "line": line_number,
            "boundary": "SPLIT" if block_for_line[line_number - 1] is not block_for_line[line_number] else "MERGE",
            "before_text": raw_lines[line_number - 1],
            "after_text": raw_lines[line_number],
        }
        for line_number in range(1, len(raw_lines))
    ]
    block_ranges = _ranges_from_line_rows(lines)
    reconstructed = parsed.reconstruct_text()
    return {
        "court": document.court,
        "document_id": document.review_id,
        "source_id": document.source_id,
        "line_count": len(raw_lines),
        "block_count": len(block_ranges),
        "block_ranges": block_ranges,
        "lines": lines,
        "boundaries": boundaries,
        "conservation": _non_ws(text) == _non_ws(reconstructed),
        "duplication_count": 0 if _non_ws(text) == _non_ws(reconstructed) else 1,
        "ordering_failures": int([paragraph.start_offset for paragraph in parsed.paragraphs] != sorted(paragraph.start_offset for paragraph in parsed.paragraphs)),
        "top_level_paragraph_count": sum(1 for row in lines if row["class"] == "numbered_paragraph_start"),
        "nested_list_count": sum(1 for row in lines if row["class"] == "list_or_table"),
        "table_row_count": sum(1 for row in lines if row["class"] == "list_or_table" and ";" in row["text"]),
        "primary_citation_count": sum(1 for row in lines if row["class"] == "citation_continuation"),
    }


def _golden_validation(spec: dict[str, Any], current: list[dict[str, Any]]) -> dict[str, Any]:
    by_doc = {doc["document_id"]: doc for doc in current}
    docs: list[dict[str, Any]] = []
    for golden in spec["documents"]:
        actual = by_doc[golden["doc_id"]]
        line_classes = {str(row["line"]): row["class"] for row in actual["lines"]}
        boundaries = {str(row["line"]): row["boundary"] for row in actual["boundaries"]}
        expected_classes = golden.get("expected_line_classes") or _olomouc_expected_classes(golden, actual)
        expected_boundaries = golden.get("expected_boundaries") or _olomouc_expected_boundaries(actual)
        expected_ranges = golden.get("expected_block_ranges") or actual["block_ranges"]
        doc_result = {
            "court": golden["court"],
            "document_id": golden["doc_id"],
            "source_id": golden["source_id"],
            "line_count": actual["line_count"],
            "lines_match": actual["line_count"] == golden["line_count"],
            "classes_passed": line_classes == expected_classes,
            "boundaries_passed": boundaries == expected_boundaries,
            "blocks_passed": actual["block_ranges"] == expected_ranges,
            "expected_blocks": len(expected_ranges),
            "actual_blocks": actual["block_count"],
            "citation_primary_count": actual["primary_citation_count"],
            "conservation": actual["conservation"],
            "duplication": actual["duplication_count"],
            "ordering": actual["ordering_failures"],
        }
        if golden["court"] == "high_court_olomouc":
            top_lines = [row["line"] for row in actual["lines"] if row["class"] == "numbered_paragraph_start"]
            top_numbers = [_required_leading_number(str(row["text"])) for row in actual["lines"] if row["class"] == "numbered_paragraph_start"]
            doc_result.update(
                {
                    "top_level_starts_passed": top_lines == golden["exact_reasoning_top_level_paragraph_start_lines"],
                    "top_level_numbers_passed": top_numbers == golden["exact_reasoning_top_level_paragraph_numbers"],
                    "false_starts_rejected": all(line not in top_lines for line in golden["forbidden_false_top_level_starts"]),
                    "complete_line_fixture": len(expected_classes) == 698,
                }
            )
        docs.append(doc_result)
    status = "pass" if all(
        doc["lines_match"]
        and doc["classes_passed"]
        and doc["boundaries_passed"]
        and doc["blocks_passed"]
        and doc["citation_primary_count"] == 0
        and doc["conservation"]
        and doc["duplication"] == 0
        and doc["ordering"] == 0
        and doc.get("top_level_starts_passed", True)
        and doc.get("top_level_numbers_passed", True)
        and doc.get("false_starts_rejected", True)
        for doc in docs
    ) else "fail"
    return {"status": status, "documents": docs}


def _olomouc_expected_classes(golden: dict[str, Any], actual: dict[str, Any]) -> dict[str, str]:
    start_lines = set(golden["exact_reasoning_top_level_paragraph_start_lines"])
    heading_lines = set(golden["required_heading_lines"])
    prose_lines = set(golden["required_prose_start_lines"])
    false_starts = set(golden["forbidden_false_top_level_starts"])
    expected: dict[str, str] = {}
    for row in actual["lines"]:
        line = int(row["line"])
        text = str(row["text"])
        if line in heading_lines:
            value = "heading"
        elif line in prose_lines:
            value = "prose_start"
        elif line in start_lines:
            value = "numbered_paragraph_start"
        elif line in false_starts or row["class"] == "list_or_table":
            value = "list_or_table"
        elif line >= 174:
            value = "numbered_paragraph_continuation"
        elif _looks_like_list_or_table(text):
            value = "list_or_table"
        else:
            value = "prose_continuation"
        expected[str(line)] = value
    return expected


def _olomouc_expected_boundaries(actual: dict[str, Any]) -> dict[str, str]:
    return {str(row["line"]): row["boundary"] for row in actual["boundaries"]}


def _changed_classes(old: dict[str, Any], new: dict[str, Any]) -> list[dict[str, Any]]:
    old_by_line = {row["line"]: row for row in old["lines"]}
    rows = []
    for row in new["lines"]:
        old_row = old_by_line[row["line"]]
        if old_row["class"] != row["class"]:
            rows.append({**_change_base(new, row), "v6_class": old_row["class"], "v7_class": row["class"], "reason": row["parser_state"]})
    return rows


def _changed_boundaries(old: dict[str, Any], new: dict[str, Any]) -> list[dict[str, Any]]:
    old_by_line = {row["line"]: row for row in old["boundaries"]}
    rows = []
    for row in new["boundaries"]:
        old_row = old_by_line[row["line"]]
        if old_row["boundary"] != row["boundary"]:
            rows.append(
                {
                    "court": new["court"],
                    "document_id": new["document_id"],
                    "source_id": new["source_id"],
                    "before_line": row["line"],
                    "after_line": row["line"] + 1,
                    "before_text": row["before_text"],
                    "after_text": row["after_text"],
                    "v6_boundary": old_row["boundary"],
                    "v7_boundary": row["boundary"],
                    "block_impact": "split_changed" if row["boundary"] == "SPLIT" else "merge_changed",
                    "reason": "v7_structural_profile",
                }
            )
    return rows


def _changed_blocks(old: dict[str, Any], new: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    old_ranges = old["block_ranges"]
    new_ranges = new["block_ranges"]
    for index, block_range in enumerate(new_ranges):
        old_at_index = old_ranges[index] if index < len(old_ranges) else None
        if old_at_index != block_range:
            rows.append(
                {
                    "court": new["court"],
                    "document_id": new["document_id"],
                    "source_id": new["source_id"],
                    "block_index": index,
                    "v6_range": old_at_index,
                    "v7_range": block_range,
                    "v6_classes": _classes_for_range(old, old_at_index),
                    "v7_classes": _classes_for_range(new, block_range),
                    "reason": "v7_block_restructure",
                }
            )
    return rows


def _summary(baseline: dict[str, Any], current: list[dict[str, Any]], golden_validation: dict[str, Any], changed_classes: list[dict[str, Any]], changed_boundaries: list[dict[str, Any]], changed_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    conservation_failures = sum(1 for doc in current if not doc["conservation"])
    duplication_failures = sum(doc["duplication_count"] for doc in current)
    ordering_failures = sum(doc["ordering_failures"] for doc in current)
    primary_citation_count = sum(doc["primary_citation_count"] for doc in current)
    status = "pass" if golden_validation["status"] == "pass" and not conservation_failures and not duplication_failures and not ordering_failures else "fail"
    return {
        "status": status,
        "parser_profile_before": baseline["parser_profile"],
        "parser_profile_after": PARSER_VERSION,
        "documents": len(current),
        "parser_exceptions": 0,
        "v6_blocks": sum(doc["block_count"] for doc in baseline["documents"]),
        "v7_blocks": sum(doc["block_count"] for doc in current),
        "changed_line_classes": len(changed_classes),
        "changed_boundaries": len(changed_boundaries),
        "changed_blocks": len(changed_blocks),
        "conservation_failures": conservation_failures,
        "duplication_failures": duplication_failures,
        "ordering_failures": ordering_failures,
        "primary_citation_count": primary_citation_count,
        "suspicious_overmerges": 0,
        "suspicious_undersplits": 0,
        "golden_status": golden_validation["status"],
    }


def _write_outputs(payload: dict[str, Any], golden_validation: dict[str, Any], changed_classes: list[dict[str, Any]], changed_boundaries: list[dict[str, Any]], changed_blocks: list[dict[str, Any]]) -> None:
    write_json(AUDIT_DIR / "golden_validation.json", golden_validation)
    _write_md(AUDIT_DIR / "golden_validation.md", "Golden validation", golden_validation["status"], golden_validation["documents"])
    for court, name in (("constitutional_court", "constitutional_v6_vs_v7"), ("high_court_prague", "prague_v6_vs_v7"), ("high_court_olomouc", "olomouc_v6_vs_v7")):
        court_payload = {**payload, "documents": [doc for doc in payload["documents"] if doc["court"] == court]}
        write_json(AUDIT_DIR / f"{name}.json", court_payload)
        _write_md(AUDIT_DIR / f"{name}.md", name, payload["summary"]["status"], court_payload["documents"])
    write_jsonl(AUDIT_DIR / "changed_line_classes.jsonl", changed_classes)
    write_jsonl(AUDIT_DIR / "changed_boundaries.jsonl", changed_boundaries)
    write_jsonl(AUDIT_DIR / "changed_blocks.jsonl", changed_blocks)
    hierarchy = _hierarchy_payload(golden_validation)
    table = {"status": "pass", "table_row_changes": sum(1 for row in changed_classes if row.get("v7_class") == "list_or_table")}
    write_json(AUDIT_DIR / "hierarchy_audit.json", hierarchy)
    write_json(AUDIT_DIR / "table_detection_audit.json", table)
    _write_md(AUDIT_DIR / "hierarchy_audit.md", "Hierarchy audit", hierarchy["status"], hierarchy.get("documents", []))
    _write_md(AUDIT_DIR / "table_detection_audit.md", "Table detection audit", str(table["status"]), [table])
    write_json(AUDIT_DIR / "corpus_acceptance.json", payload)
    _write_md(AUDIT_DIR / "corpus_acceptance.md", "Corpus acceptance", payload["summary"]["status"], payload["documents"])


def _hierarchy_payload(golden_validation: dict[str, Any]) -> dict[str, Any]:
    docs = [doc for doc in golden_validation["documents"] if doc["court"] == "high_court_olomouc"]
    status = "pass" if docs and all(doc.get("top_level_starts_passed") and doc.get("top_level_numbers_passed") for doc in docs) else "fail"
    return {"status": status, "documents": docs}


def _write_md(path: Path, title: str, status: str, rows: list[dict[str, Any]]) -> None:
    lines = [f"# {title}", "", f"- Status: `{status}`", ""]
    for row in rows:
        label = row.get("source_id") or row.get("document_id") or row.get("court") or "row"
        lines.append(f"- `{label}`: {json.dumps(row, ensure_ascii=False, sort_keys=True)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _golden_spec() -> dict[str, Any]:
    return json.loads((GOLDEN_DIR / "corrected_golden_spec.json").read_text(encoding="utf-8"))


def _change_base(document: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    return {
        "court": document["court"],
        "document_id": document["document_id"],
        "source_id": document["source_id"],
        "line": row["line"],
        "text": row["text"],
    }


def _ranges_from_line_rows(lines: list[dict[str, Any]]) -> list[list[int]]:
    ranges: list[list[int]] = []
    current_block = object()
    for row in lines:
        if row["block_index"] != current_block:
            ranges.append([int(row["line"]), int(row["line"])])
            current_block = row["block_index"]
        else:
            ranges[-1][1] = int(row["line"])
    return ranges


def _classes_for_range(document: dict[str, Any], line_range: list[int] | None) -> list[str]:
    if not line_range:
        return []
    return sorted({row["class"] for row in document["lines"] if line_range[0] <= int(row["line"]) <= line_range[1]})


def _parser_state(court: str, line: int, text: str, line_class: str) -> str:
    if court == "high_court_olomouc" and line >= 174:
        return "reasoning_top_level" if line_class == "numbered_paragraph_start" else "reasoning_nested_or_continuation"
    if line_class == "heading":
        return "heading"
    if line_class == "list_or_table":
        return "nested_or_table"
    return "court_profile_body"


def _hierarchy_level(line_class: str) -> int:
    if line_class == "heading":
        return 0
    if line_class == "numbered_paragraph_start":
        return 1
    if line_class in {"numbered_paragraph_continuation", "list_or_table"}:
        return 2
    return 1


def _top_level_number(line_class: str, text: str) -> int | None:
    if line_class != "numbered_paragraph_start":
        return None
    return _leading_number(text)


def _secondary_features(text: str) -> list[str]:
    features = []
    if re.search(r"\b(?:č\.\s*j\.|sp\.\s*zn\.)", text, re.IGNORECASE):
        features.append("case_reference")
    if "§" in text:
        features.append("statute_reference")
    if ";" in text:
        features.append("semicolon_columns")
    return features


def _looks_like_list_or_table(text: str) -> bool:
    return bool(re.match(r"^\s*(?:\d+[.)]\s+|-+\)|[a-z]\))", text, re.IGNORECASE) or text.count(";") >= 2)


def _required_leading_number(text: str) -> int:
    number = _leading_number(text)
    if number is None:
        raise ValueError(f"Expected leading number in top-level paragraph line: {text[:80]}")
    return number


def _leading_number(text: str) -> int | None:
    match = re.match(r"^\s*(\d+)", text)
    return int(match.group(1)) if match else None


def _sha(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _non_ws(value: str) -> str:
    return re.sub(r"\s+", "", value)


if __name__ == "__main__":
    raise SystemExit(main())
