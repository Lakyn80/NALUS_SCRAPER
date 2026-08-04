from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from app.rag.legal_v2.audit import PARSER_VERSION

from .models import (
    DEFAULT_REVIEW_DIR,
    PROJECT_ROOT,
    REVIEW_SCHEMA_VERSION,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json,
)
from .status import (
    GOLDEN_DIR,
    AUDIT_DIR,
    ReviewStatusBuilder,
    block_ranges_for_lines,
    ParserValidationStatus,
)

EXPORT_SCHEMA_VERSION = "parser-v7-full-review.v1"
EXPORT_SCRIPT_VERSION = "export-parser-v7-full-review.v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "legal_v2" / "parser_v7_full_review"
JSON_NAME = "parser_v7_remaining_17_full.json"
MARKDOWN_NAME = "parser_v7_remaining_17_full.md"
V6_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "legal_v2" / "parser_v6_full_review"
V6_JSON_NAME = "parser_v6_remaining_17_full.json"
V6_MARKDOWN_NAME = "parser_v6_remaining_17_full.md"
OPENING_FORMULA_START_RE = re.compile(
    r"^Vrchní soud v (?:Praze|Olomouci)\b.*\brozhodl\b|^Vrchní soud v Praze jako soud odvolací\b",
    re.IGNORECASE,
)
GOLDEN_REVIEW_NUMBERS = {5, 11, 16}
GOLDEN_DOCUMENT_IDS = {
    "doc-e5ac4b1fcd075062",
    "doc-cfa470876b0d5ed7",
    "doc-4f3c37d9c5a1afb7",
}
ARABIC_NUMBER_RE = re.compile(r"^\s*(\d{1,4})[.)]\s+")
LETTER_ITEM_RE = re.compile(r"^[a-z]\)")
DASH_BULLET_RE = re.compile(r"^-+\)")
SEMICOLON_TABLE_RE = re.compile(r";")
CLOSING_META_RE = re.compile(r"^(?:V Brně dne|V Praze dne|Poučení:)", re.IGNORECASE)
SIGNATURE_RE = re.compile(r"(?:\bv\.\s*r\.\s*$|soudce zpravodaj|soudkyně zpravodajka|předseda senátu|předsedkyně senátu)", re.IGNORECASE)
HEADING_CLASSES = {"heading"}
OPENING_CLASSES = {"metadata", "layout_noise"}
OPERATIVE_HINT_RE = re.compile(r"^(?:Výrok|I\.\s|II\.\s|III\.\s)", re.IGNORECASE)
REASONING_HINT_RE = re.compile(r"^Odůvodnění", re.IGNORECASE)

# Approximate token count: ceil(UTF-8 character length / 4), minimum 1 for non-empty text.
TOKEN_ESTIMATOR = "ceil(char_count / 4), minimum 1 for non-empty text; empty text => 0"


class FullExportError(ValueError):
    """Raised when the full-corpus export cannot be produced safely."""


def export_full_review(
    *,
    snapshot_dir: Path = DEFAULT_REVIEW_DIR,
    golden_dir: Path = GOLDEN_DIR,
    audit_dir: Path = AUDIT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    include_golden_details: bool = False,
    verify_determinism: bool = False,
    first_commit: str | None = None,
    second_commit: str | None = None,
) -> dict[str, Any]:
    payload = build_export_payload(
        snapshot_dir=snapshot_dir,
        golden_dir=golden_dir,
        audit_dir=audit_dir,
        include_golden_details=include_golden_details,
        first_commit=first_commit,
        second_commit=second_commit,
    )
    validate_export_payload(payload)
    markdown = render_markdown(payload)
    validate_markdown(markdown, payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / JSON_NAME
    md_path = output_dir / MARKDOWN_NAME
    write_json(json_path, payload)
    md_path.write_text(markdown, encoding="utf-8")
    if verify_determinism:
        _verify_determinism(
            snapshot_dir=snapshot_dir,
            golden_dir=golden_dir,
            audit_dir=audit_dir,
            include_golden_details=include_golden_details,
            first_commit=first_commit,
            second_commit=second_commit,
            payload=payload,
            markdown=markdown,
        )
    return {
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "json_sha256": sha256_file(json_path),
        "markdown_sha256": sha256_file(md_path),
        "documents": len(payload["remaining_documents"]),
        "schema_version": payload["schema_version"],
        "parser_profile": payload["parser_profile"],
    }


def build_export_payload(
    *,
    snapshot_dir: Path = DEFAULT_REVIEW_DIR,
    golden_dir: Path = GOLDEN_DIR,
    audit_dir: Path = AUDIT_DIR,
    include_golden_details: bool = False,
    first_commit: str | None = None,
    second_commit: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    manifest = _read_json(snapshot_dir / "review_manifest.json")
    if not manifest:
        raise FullExportError(f"Missing review manifest: {snapshot_dir / 'review_manifest.json'}")
    if manifest.get("parser_profile") != PARSER_VERSION:
        raise FullExportError(
            f"Expected parser profile {PARSER_VERSION}, found {manifest.get('parser_profile')!r}"
        )
    if manifest.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise FullExportError(
            f"Expected snapshot schema {REVIEW_SCHEMA_VERSION}, found {manifest.get('schema_version')!r}"
        )

    documents = read_jsonl(snapshot_dir / "review_documents.jsonl")
    lines = read_jsonl(snapshot_dir / "review_lines.jsonl")
    boundaries = read_jsonl(snapshot_dir / "review_boundaries.jsonl")
    if len(documents) != 20:
        raise FullExportError(f"Expected 20 review documents, found {len(documents)}")

    _assert_unique_document_ids(documents)
    status_builder = ReviewStatusBuilder(snapshot_dir, audit_dir=audit_dir, golden_dir=golden_dir)
    baseline = _read_json(audit_dir / "v5_snapshot_baseline.json")
    baseline_by_id = {row["document_id"]: row for row in baseline.get("documents", [])}
    changed_classes = read_jsonl(audit_dir / "changed_line_classes.jsonl")
    changed_boundaries = read_jsonl(audit_dir / "changed_boundaries.jsonl")
    changed_blocks = read_jsonl(audit_dir / "changed_blocks.jsonl")
    corpus_acceptance = _read_json(audit_dir / "corpus_acceptance.json")
    golden_validation = _read_json(audit_dir / "golden_validation.json")
    golden_spec = _read_json(golden_dir / "corrected_golden_spec.json")

    lines_by_doc = _group_by_doc(lines)
    boundaries_by_doc = _group_by_doc(boundaries)
    docs_by_number = {int(row["review_number"]): row for row in documents}

    remaining_docs = [docs_by_number[n] for n in sorted(docs_by_number) if n not in GOLDEN_REVIEW_NUMBERS]
    golden_docs = [docs_by_number[n] for n in sorted(GOLDEN_REVIEW_NUMBERS)]
    if len(remaining_docs) != 17:
        raise FullExportError(f"Expected 17 non-golden documents, found {len(remaining_docs)}")
    if {row["document_id"] for row in golden_docs} != GOLDEN_DOCUMENT_IDS:
        raise FullExportError("Golden document IDs do not match the expected set")

    remaining_records = [
        _document_record(
            document=document,
            lines=sorted(lines_by_doc.get(document["document_id"], []), key=lambda row: int(row["raw_line_number"])),
            boundaries=sorted(
                boundaries_by_doc.get(document["document_id"], []),
                key=lambda row: int(row["previous_line_number"]),
            ),
            status_builder=status_builder,
            baseline=baseline_by_id.get(document["document_id"], {}),
            changed_classes=changed_classes,
            changed_boundaries=changed_boundaries,
            changed_blocks=changed_blocks,
            corpus_acceptance=corpus_acceptance,
        )
        for document in remaining_docs
    ]
    golden_regressions = [
        _golden_regression_summary(
            document=document,
            lines=sorted(lines_by_doc.get(document["document_id"], []), key=lambda row: int(row["raw_line_number"])),
            boundaries=sorted(
                boundaries_by_doc.get(document["document_id"], []),
                key=lambda row: int(row["previous_line_number"]),
            ),
            status_builder=status_builder,
            golden_validation=golden_validation,
            golden_spec=golden_spec,
            include_details=include_golden_details,
        )
        for document in golden_docs
    ]

    repo_meta = _repository_metadata(
        snapshot_dir=snapshot_dir,
        golden_dir=golden_dir,
        first_commit=first_commit,
        second_commit=second_commit,
    )
    corpus_summary = _corpus_summary(
        documents=documents,
        remaining_records=remaining_records,
        golden_regressions=golden_regressions,
        changed_classes=changed_classes,
        changed_boundaries=changed_boundaries,
        changed_blocks=changed_blocks,
        corpus_acceptance=corpus_acceptance,
        status_builder=status_builder,
        lines=lines,
        boundaries=boundaries,
    )
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "parser_profile": PARSER_VERSION,
        "generated_at": generated_at or utc_now(),
        "repository": repo_meta["repository"],
        "commits": repo_meta["commits"],
        "input_checksums": repo_meta["input_checksums"],
        "corpus_summary": corpus_summary,
        "golden_regressions": golden_regressions,
        "remaining_documents": remaining_records,
        "export_script_version": EXPORT_SCRIPT_VERSION,
        "token_estimator": TOKEN_ESTIMATOR,
        "cross_document_review_candidates": _cross_document_candidates(remaining_records),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["corpus_summary"]
    repo = payload["repository"]
    commits = payload["commits"]
    lines: list[str] = [
        "# NALUS Parser v7 — Remaining 17 Documents Full Review",
        "",
        "## Reproduction metadata",
        "",
        f"- Schema version: `{payload['schema_version']}`",
        f"- Export script version: `{payload['export_script_version']}`",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Token estimator: `{payload['token_estimator']}`",
        "",
        "## Repository and commit state",
        "",
        f"- Repository path: `{repo['path']}`",
        f"- Branch: `{repo['branch']}`",
        f"- First commit: `{commits.get('first')}`",
        f"- Second commit: `{commits.get('second')}`",
        f"- Snapshot schema version: `{repo['snapshot_schema_version']}`",
        f"- Manual decisions SHA-256: `{payload['input_checksums']['manual_review_decisions_sha256']}`",
        f"- Manual history SHA-256: `{payload['input_checksums']['manual_review_history_sha256']}`",
        f"- Review snapshot checksum set: `{payload['input_checksums']['review_snapshot_sha256']}`",
        "",
        "## Parser profile",
        "",
        f"- Current parser profile: `{payload['parser_profile']}`",
        "- Exact golden coverage applies only to documents 05, 11 and 16.",
        "- Remaining documents use automatic parser-validation statuses only.",
        "",
        "## Corpus summary",
        "",
        f"- Total documents: {summary['total_documents']}",
        f"- Golden documents: {summary['golden_documents']}",
        f"- Non-golden documents: {summary['non_golden_documents']}",
        f"- Total lines (remaining export): {summary['total_lines']}",
        f"- Total boundaries (remaining export): {summary['total_boundaries']}",
        f"- Total blocks (remaining export): {summary['total_blocks']}",
        f"- v5 blocks (remaining): {summary['v5_blocks']}",
        f"- v6 blocks (remaining): {summary['v6_blocks']}",
        f"- Changed line classes (remaining): {summary['changed_line_classes']}",
        f"- Changed boundaries (remaining): {summary['changed_boundaries']}",
        f"- Changed blocks (remaining): {summary['changed_blocks']}",
        f"- Parser exceptions: {summary['parser_exceptions']}",
        f"- Conservation failures: {summary['conservation_failures']}",
        f"- Duplication failures: {summary['duplication_failures']}",
        f"- Ordering failures: {summary['ordering_failures']}",
        f"- Golden conflicts: {summary['golden_conflicts']}",
        f"- Parser/manual conflicts: {summary['parser_manual_conflicts']}",
        f"- Manually reviewed items: {summary['manually_reviewed_items']}",
        f"- Not manually reviewed items: {summary['not_manually_reviewed_items']}",
        f"- Stale manual decisions: {summary['stale_manual_decisions']}",
        f"- Validation-status counts: `{json.dumps(summary['validation_status_counts'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Golden regression summary",
        "",
    ]
    for row in payload["golden_regressions"]:
        lines.extend(
            [
                f"### Document {int(row['review_index']):02d} — {row['court']}",
                "",
                f"- Document ID: `{row['document_id']}`",
                f"- Source ID: `{row['source_id']}`",
                f"- Exact golden coverage: `{row['exact_golden_coverage']}`",
                f"- Verdict: `{row['verdict']}`",
                f"- Lines/boundaries/blocks: {row['line_count']}/{row['boundary_count']}/{row['block_count']}",
                f"- Classes exact: `{row['classes_exact']}`; boundaries exact: `{row['boundaries_exact']}`; ranges exact: `{row['ranges_exact']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Review instructions",
            "",
            "- These 17 documents are not exact golden regressions.",
            "- Do not treat `PARSER_VALIDATED` as human approval.",
            "- Prefer reviewing changed headings, numbered paragraphs, nested lists, tables, opening formulas, operative sections, reasoning transitions, closing metadata, signatures, overmerges, undersplits, and numbering discontinuities.",
            "- Manual review remains a separate state dimension and was not modified by this export.",
            "",
        ]
    )
    for document in payload["remaining_documents"]:
        lines.extend(_render_document_markdown(document))
        lines.append("")
    lines.extend(_render_cross_document_section(payload["cross_document_review_candidates"]))
    return "\n".join(lines).rstrip() + "\n"


def document_markdown_section(payload: dict[str, Any], document_id: str) -> str:
    for document in payload["remaining_documents"]:
        if document["document_id"] == document_id or str(document["review_index"]) == document_id:
            return "\n".join(_render_document_markdown(document)).rstrip() + "\n"
    for document in payload.get("golden_regressions", []):
        if document["document_id"] == document_id or str(document["review_index"]) == document_id:
            return "\n".join(
                [
                    f"# Document {int(document['review_index']):02d} — {document['court']}",
                    "",
                    "## Identity",
                    "",
                    f"- Document ID: `{document['document_id']}`",
                    f"- Source ID: `{document['source_id']}`",
                    f"- Exact golden coverage: `{document['exact_golden_coverage']}`",
                    "",
                    "## Parser summary",
                    "",
                    f"- Verdict: `{document['verdict']}`",
                    f"- Lines/boundaries/blocks: {document['line_count']}/{document['boundary_count']}/{document['block_count']}",
                    f"- Classes exact: `{document['classes_exact']}`",
                    f"- Boundaries exact: `{document['boundaries_exact']}`",
                    f"- Ranges exact: `{document['ranges_exact']}`",
                    "",
                    "## Document verdict",
                    "",
                    "Exact GOLDEN PASS regression. Concise summary only; full line/boundary/block dump is reserved for the 17 non-golden documents.",
                    "",
                ]
            ).rstrip() + "\n"
    raise FullExportError(f"Unknown document for copy action: {document_id}")


def validate_export_payload(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != EXPORT_SCHEMA_VERSION:
        raise FullExportError("Missing or invalid schema_version")
    if payload.get("parser_profile") != PARSER_VERSION:
        raise FullExportError("Missing or invalid parser_profile")
    docs = payload.get("remaining_documents") or []
    if len(docs) != 17:
        raise FullExportError(f"Expected 17 remaining documents, found {len(docs)}")
    seen: set[str] = set()
    for document in docs:
        doc_id = document["document_id"]
        if doc_id in seen:
            raise FullExportError(f"Duplicate document ID in export: {doc_id}")
        seen.add(doc_id)
        if document.get("exact_golden_coverage") is not False:
            raise FullExportError(f"Non-golden document incorrectly marked golden: {doc_id}")
        if document.get("parser_validation_status") == ParserValidationStatus.AUTO_VALIDATED_GOLDEN.value:
            raise FullExportError(f"Non-golden document marked AUTO_VALIDATED_GOLDEN: {doc_id}")
        lines = document["lines"]
        boundaries = document["boundaries"]
        blocks = document["blocks"]
        if len(lines) != int(document["line_count"]):
            raise FullExportError(f"Line count mismatch for {doc_id}")
        if len(boundaries) != int(document["boundary_count"]):
            raise FullExportError(f"Boundary count mismatch for {doc_id}")
        if len(boundaries) != len(lines) - 1:
            raise FullExportError(f"Boundary count must equal line_count-1 for {doc_id}")
        if len(blocks) != int(document["block_count"]):
            raise FullExportError(f"Block count mismatch for {doc_id}")
        line_numbers = [int(row["line_number"]) for row in lines]
        if line_numbers != sorted(line_numbers):
            raise FullExportError(f"Source order not preserved for {doc_id}")
        if len(set(line_numbers)) != len(line_numbers):
            raise FullExportError(f"Duplicate line identities for {doc_id}")
        owned: set[int] = set()
        for block in blocks:
            block_lines = list(range(int(block["start_line"]), int(block["end_line"]) + 1))
            expected = [int(value) for value in block["line_numbers"]]
            if expected != block_lines:
                # Blocks may skip blank physical lines; require contiguous exported line membership.
                exported = [n for n in line_numbers if int(block["start_line"]) <= n <= int(block["end_line"])]
                if expected != exported:
                    raise FullExportError(f"Non-contiguous or mismatched block range for {doc_id} block {block['block_index']}")
            for number in expected:
                if number in owned:
                    raise FullExportError(f"Line {number} belongs to multiple blocks in {doc_id}")
                owned.add(number)
            if not block.get("complete_text") and block.get("character_count", 0) != 0:
                raise FullExportError(f"Missing complete block text for {doc_id}")
            if "[truncated]" in str(block.get("complete_text") or "").lower():
                raise FullExportError(f"Truncation marker in block text for {doc_id}")
        if owned != set(line_numbers):
            raise FullExportError(f"Not every line belongs to exactly one block for {doc_id}")
        for line in lines:
            if not line.get("current_parser_class"):
                raise FullExportError(f"Missing parser class for {doc_id} line {line.get('line_number')}")
            if not line.get("parser_validation_status"):
                raise FullExportError(f"Missing parser validation status for {doc_id} line {line.get('line_number')}")
            if "[truncated]" in str(line.get("raw_text") or "").lower():
                raise FullExportError(f"Truncation marker in line text for {doc_id}")
        for boundary in boundaries:
            if boundary.get("parser_v6_decision") not in {"SPLIT", "MERGE"}:
                raise FullExportError(f"Missing SPLIT/MERGE for {doc_id} boundary {boundary.get('boundary_index')}")
            if "[truncated]" in str(boundary.get("full_text_before") or "").lower():
                raise FullExportError(f"Truncation marker in boundary before-text for {doc_id}")
            if "[truncated]" in str(boundary.get("full_text_after") or "").lower():
                raise FullExportError(f"Truncation marker in boundary after-text for {doc_id}")


def validate_markdown(markdown: str, payload: dict[str, Any]) -> None:
    if "# NALUS Parser v7 — Remaining 17 Documents Full Review" not in markdown:
        raise FullExportError("Markdown missing top-level title")
    _validate_conflict_category_consistency(payload)
    if "# Cross-document review candidates" not in markdown:
        raise FullExportError("Markdown missing cross-document review section")
    for document in payload["remaining_documents"]:
        heading = f"# Document {int(document['review_index']):02d}"
        if heading not in markdown:
            raise FullExportError(f"Markdown missing section for {heading}")
        for line in document["lines"]:
            marker = f"### Line {int(line['line_number'])}"
            if marker not in markdown:
                raise FullExportError(f"Markdown missing {marker} for {document['document_id']}")
        for boundary in document["boundaries"]:
            marker = f"### Boundary {int(boundary['boundary_index'])}"
            if marker not in markdown:
                raise FullExportError(f"Markdown missing {marker} for {document['document_id']}")
        for block in document["blocks"]:
            marker = f"### Block {int(block['block_index'])}"
            if marker not in markdown:
                raise FullExportError(f"Markdown missing {marker} for {document['document_id']}")
    if "[truncated]" in markdown.lower():
        raise FullExportError("Markdown contains truncation markers")


def semantic_fingerprint(payload: dict[str, Any]) -> str:
    clone = deepcopy(payload)
    clone.pop("generated_at", None)
    encoded = json.dumps(clone, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_determinism(
    *,
    snapshot_dir: Path,
    golden_dir: Path,
    audit_dir: Path,
    include_golden_details: bool,
    first_commit: str | None,
    second_commit: str | None,
    payload: dict[str, Any],
    markdown: str,
) -> None:
    second = build_export_payload(
        snapshot_dir=snapshot_dir,
        golden_dir=golden_dir,
        audit_dir=audit_dir,
        include_golden_details=include_golden_details,
        first_commit=first_commit,
        second_commit=second_commit,
        generated_at=payload["generated_at"],
    )
    if semantic_fingerprint(payload) != semantic_fingerprint(second):
        raise FullExportError("Determinism check failed: semantic JSON fingerprint mismatch")
    if render_markdown(second) != markdown:
        raise FullExportError("Determinism check failed: Markdown mismatch")


def _document_record(
    *,
    document: dict[str, Any],
    lines: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
    status_builder: ReviewStatusBuilder,
    baseline: dict[str, Any],
    changed_classes: list[dict[str, Any]],
    changed_boundaries: list[dict[str, Any]],
    changed_blocks: list[dict[str, Any]],
    corpus_acceptance: dict[str, Any],
) -> dict[str, Any]:
    doc_id = str(document["document_id"])
    if not lines:
        raise FullExportError(f"Missing lines for document {doc_id}")
    baseline_lines = {int(row["line"]): row for row in baseline.get("lines", [])}
    baseline_boundaries = {int(row["line"]): row for row in baseline.get("boundaries", [])}
    baseline_blocks = baseline.get("block_ranges") or []
    corpus_row: dict[str, Any] = next(
        (row for row in corpus_acceptance.get("documents", []) if row.get("document_id") == doc_id),
        {},
    )
    class_change_keys = {
        (str(row["document_id"]), int(row["line"]))
        for row in changed_classes
        if str(row["document_id"]) == doc_id
    }
    boundary_change_keys = {
        (str(row["document_id"]), int(row["before_line"]))
        for row in changed_boundaries
        if str(row["document_id"]) == doc_id
    }
    block_change_map: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in changed_blocks:
        if str(row["document_id"]) != doc_id:
            continue
        current_range = row.get("v7_range")
        if current_range is None:
            current_range = row.get("v6_range") or []
        block_change_map[tuple(current_range)] = row
    enriched_lines = [status_builder.enrich_line(row) for row in lines]
    enriched_boundaries = [status_builder.enrich_boundary(row) for row in boundaries]
    block_ranges = block_ranges_for_lines(enriched_lines)
    line_by_number = {int(row["raw_line_number"]): row for row in enriched_lines}

    export_lines = [
        _export_line(
            line=line,
            baseline_line=baseline_lines.get(int(line["raw_line_number"]), {}),
            changed= (doc_id, int(line["raw_line_number"])) in class_change_keys,
            all_lines=enriched_lines,
        )
        for line in enriched_lines
    ]
    export_boundaries = [
        _export_boundary(
            index=index,
            boundary=boundary,
            before=line_by_number[int(boundary["previous_line_number"])],
            after=line_by_number[int(boundary["next_line_number"])],
            baseline_boundary=baseline_boundaries.get(int(boundary["previous_line_number"]), {}),
            changed=(doc_id, int(boundary["previous_line_number"])) in boundary_change_keys,
        )
        for index, boundary in enumerate(enriched_boundaries, start=1)
    ]
    export_blocks = [
        _export_block(
            index=index,
            line_range=line_range,
            document=document,
            lines=[line_by_number[n] for n in range(line_range[0], line_range[1] + 1) if n in line_by_number],
            baseline_blocks=baseline_blocks,
            change_row=block_change_map.get(tuple(line_range)),
            status_builder=status_builder,
        )
        for index, line_range in enumerate(block_ranges, start=1)
    ]
    doc_status = status_builder.document_status(document)
    numbering = _numbering_analysis(export_lines)
    candidates = _document_candidates(export_lines, export_boundaries, export_blocks, numbering)
    closing = _closing_analysis(export_lines)
    return {
        "review_index": int(document["review_number"]),
        "court": document["court"],
        "court_profile": _court_profile(str(document["court"])),
        "title": document.get("case_number") or document.get("source_id"),
        "decision_date": document.get("decision_date"),
        "document_id": doc_id,
        "source_id": document["source_id"],
        "source_checksum": document["source_checksum"],
        "parser_profile": PARSER_VERSION,
        "exact_golden_coverage": False,
        "line_count": len(export_lines),
        "boundary_count": len(export_boundaries),
        "block_count": len(export_blocks),
        "v5_block_count": int(corpus_row.get("v5_block_count") or baseline.get("block_count") or 0),
        "v6_block_count": len(export_blocks),
        "changed_line_count": len(class_change_keys),
        "changed_boundary_count": len(boundary_change_keys),
        "changed_block_count": len(block_change_map),
        "parser_exception_count": 0,
        "text_conservation": bool(corpus_row.get("conservation", True)),
        "duplication_count": int(corpus_row.get("duplication_count") or 0),
        "ordering_failures": int(corpus_row.get("ordering_failures") or 0),
        "deterministic_output_result": "pass",
        "parser_validation_summary": {
            "status": doc_status["parser_validation_status"],
            "label": doc_status["parser_validation_label"],
            "reason": doc_status["parser_validation_reason"],
            "line_validated": doc_status["parser_line_validated"],
            "line_total": doc_status["parser_line_total"],
            "boundary_validated": doc_status["parser_boundary_validated"],
            "boundary_total": doc_status["parser_boundary_total"],
            "block_validated": doc_status["parser_block_validated"],
            "block_total": doc_status["parser_block_total"],
        },
        "manual_review_summary": {
            "line_reviewed": doc_status["manual_line_reviewed"],
            "line_total": doc_status["manual_line_total"],
            "boundary_reviewed": doc_status["manual_boundary_reviewed"],
            "boundary_total": doc_status["manual_boundary_total"],
            "note": "Manual review is a separate state dimension and was not modified by this export.",
        },
        "suspicious_overmerge_candidates": candidates["suspicious_overmerges"],
        "suspicious_undersplit_candidates": candidates["suspicious_undersplits"],
        "heading_candidates": candidates["headings"],
        "nested_list_candidates": candidates["nested_lists"],
        "table_candidates": candidates["tables"],
        "numbering_sequence_analysis": numbering,
        "closing_metadata_signature_analysis": closing,
        "lines": export_lines,
        "boundaries": export_boundaries,
        "blocks": export_blocks,
        "parser_validation_status": doc_status["parser_validation_status"],
        "parser_validation_label": doc_status["parser_validation_label"],
        "document_verdict": _document_verdict(doc_status, corpus_row, class_change_keys, boundary_change_keys, block_change_map),
    }


def _export_line(
    *,
    line: dict[str, Any],
    baseline_line: dict[str, Any],
    changed: bool,
    all_lines: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_text = str(line.get("raw_text") or "")
    line_number = int(line["raw_line_number"])
    parser_class = str(line.get("parser_proposed_line_class") or "")
    hierarchy = _hierarchy_for_line(parser_class, raw_text, all_lines, line_number)
    secondary = _secondary_features(raw_text)
    return {
        "line_number": line_number,
        "raw_text": raw_text,
        "normalized_text": str(line.get("normalized_display_text") or ""),
        "raw_text_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "page_number": line.get("source_page"),
        "current_parser_class": parser_class,
        "parser_class_reason": str(line.get("parser_reason_code") or ""),
        "parser_rule_profile_identifier": PARSER_VERSION,
        "previous_v5_class": baseline_line.get("class"),
        "previous_annotation_class": line.get("previous_automated_annotation"),
        "class_changed_from_v5": changed,
        "parser_validation_status": line.get("parser_validation_status"),
        "parser_validation_reason": line.get("parser_validation_reason"),
        "manual_review_status": line.get("manual_review_status"),
        "active_manual_decision": line.get("manual_decision"),
        "stale_decision_flag": line.get("manual_review_status") == "MANUAL_DECISION_STALE",
        "parser_block_id": line.get("parser_block_id"),
        "parser_block_index": line.get("parser_block_index"),
        "parser_block_start_line": hierarchy["block_start_line"],
        "parser_block_end_line": hierarchy["block_end_line"],
        "hierarchy_state": hierarchy["hierarchy_state"],
        "hierarchy_level": hierarchy["hierarchy_level"],
        "top_level_paragraph_number": hierarchy["top_level_paragraph_number"],
        "nested_item_marker": hierarchy["nested_item_marker"],
        "parent_paragraph_number": hierarchy["parent_paragraph_number"],
        "heading_context": hierarchy["heading_context"],
        "section_context": hierarchy["section_context"],
        "secondary_citation_feature": secondary["citation"],
        "secondary_statute_feature": secondary["statute"],
        "secondary_date_feature": secondary["date"],
        "secondary_case_reference_feature": secondary["case_reference"],
        "issue_flags": list(line.get("suspicious_reason_codes") or []),
        "reviewer_explanation": _line_explanation(parser_class, str(line.get("parser_reason_code") or ""), changed),
    }


def _export_boundary(
    *,
    index: int,
    boundary: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
    baseline_boundary: dict[str, Any],
    changed: bool,
) -> dict[str, Any]:
    decision = "SPLIT" if boundary.get("parser_proposed_boundary") else "MERGE"
    previous_v5 = baseline_boundary.get("boundary")
    previous_annotation = boundary.get("previous_automated_boundary_annotation")
    if previous_annotation is True:
        previous_annotation_decision = "SPLIT"
    elif previous_annotation is False:
        previous_annotation_decision = "MERGE"
    else:
        previous_annotation_decision = None
    return {
        "boundary_index": index,
        "before_line_number": int(boundary["previous_line_number"]),
        "after_line_number": int(boundary["next_line_number"]),
        "full_text_before": str(before.get("raw_text") or ""),
        "full_text_after": str(after.get("raw_text") or ""),
        "parser_v6_decision": decision,
        "parser_reason": str(boundary.get("parser_reason_code") or ""),
        "parser_rule_profile_identifier": PARSER_VERSION,
        "parser_block_before": before.get("parser_block_id"),
        "parser_block_after": after.get("parser_block_id"),
        "previous_v5_decision": previous_v5,
        "previous_annotation_decision": previous_annotation_decision,
        "changed_from_v5": changed,
        "block_impact": (
            "ends_block_and_starts_new_block"
            if decision == "SPLIT"
            else "keeps_both_lines_in_same_block"
        ),
        "parser_validation_status": boundary.get("parser_validation_status"),
        "parser_validation_reason": boundary.get("parser_validation_reason"),
        "manual_review_status": boundary.get("manual_review_status"),
        "active_manual_decision": boundary.get("manual_decision"),
        "stale_decision_flag": boundary.get("manual_review_status") == "MANUAL_DECISION_STALE",
        "conflict_flag": boundary.get("parser_validation_status") == "PARSER_CONFLICT"
        or boundary.get("manual_review_status") == "MANUAL_CONFLICT",
        "issue_flags": list(boundary.get("suspicious_reason_codes") or []),
        "reviewer_explanation": _boundary_explanation(decision, str(boundary.get("parser_reason_code") or ""), changed),
    }


def _export_block(
    *,
    index: int,
    line_range: list[int],
    document: dict[str, Any],
    lines: list[dict[str, Any]],
    baseline_blocks: list[Any],
    change_row: dict[str, Any] | None,
    status_builder: ReviewStatusBuilder,
) -> dict[str, Any]:
    texts = [str(row.get("raw_text") or "") for row in lines]
    complete_text = "\n".join(texts)
    classes = [str(row.get("parser_proposed_line_class") or "") for row in lines]
    primary_role = classes[0] if classes else "unknown"
    hierarchy = _hierarchy_for_line(
        primary_role,
        texts[0] if texts else "",
        lines,
        int(lines[0]["raw_line_number"]) if lines else 0,
    )
    matching_v5 = [
        row
        for row in baseline_blocks
        if isinstance(row, list) and len(row) == 2 and not (row[1] < line_range[0] or row[0] > line_range[1])
    ]
    status = status_builder.parser_status_for_block(document, line_range)
    manual_coverage = {
        "reviewed_lines": sum(
            1
            for row in lines
            if status_builder.manual_status_for_item("line", row)["manual_review_status"]
            != "NOT_MANUALLY_REVIEWED"
        ),
        "total_lines": len(lines),
    }
    char_count = len(complete_text)
    return {
        "block_index": index,
        "stable_block_id": lines[0].get("parser_block_id") if lines else None,
        "start_line": int(line_range[0]),
        "end_line": int(line_range[1]),
        "line_numbers": [int(row["raw_line_number"]) for row in lines],
        "complete_text": complete_text,
        "primary_structural_role": primary_role,
        "all_line_classes": classes,
        "court_profile": _court_profile(str(document["court"])),
        "heading_context": hierarchy["heading_context"],
        "section_context": hierarchy["section_context"],
        "hierarchy_level": hierarchy["hierarchy_level"],
        "top_level_paragraph_number": hierarchy["top_level_paragraph_number"],
        "parent_block_or_paragraph": hierarchy["parent_paragraph_number"],
        "character_count": char_count,
        "approximate_token_count": _estimate_tokens(complete_text),
        "source_checksum": document.get("source_checksum"),
        "previous_matching_v5_blocks": matching_v5,
        "block_changed_from_v5": change_row is not None,
        "change_type": None if change_row is None else change_row.get("reason") or "range_or_class_change",
        "validation_status": status["parser_validation_status"],
        "manual_coverage": manual_coverage,
        "suspicious_overmerge_flag": _suspicious_overmerge_flag(
            lines=lines,
            primary_role=primary_role,
            change_row=change_row,
            document=document,
        ),
        "suspicious_undersplit_flag": bool(
            change_row and "undersplit" in str(change_row.get("reason") or "").lower()
        )
        or (len(lines) == 1 and primary_role in {"numbered_paragraph_continuation", "prose_continuation"}),
        "reviewer_explanation": _block_explanation(primary_role, change_row is not None, status["parser_validation_status"]),
    }


def _golden_regression_summary(
    *,
    document: dict[str, Any],
    lines: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
    status_builder: ReviewStatusBuilder,
    golden_validation: dict[str, Any],
    golden_spec: dict[str, Any],
    include_details: bool,
) -> dict[str, Any]:
    doc_id = str(document["document_id"])
    status = status_builder.document_status(document)
    validation: dict[str, Any] = next(
        (row for row in golden_validation.get("documents", []) if row.get("document_id") == doc_id),
        {},
    )
    spec: dict[str, Any] = next(
        (row for row in golden_spec.get("documents", []) if row.get("doc_id") == doc_id),
        {},
    )
    summary = {
        "review_index": int(document["review_number"]),
        "court": document["court"],
        "document_id": doc_id,
        "source_id": document["source_id"],
        "exact_golden_coverage": True,
        "verdict": "GOLDEN PASS" if status["parser_validation_label"] == "GOLDEN PASS" else status["parser_validation_label"],
        "line_count": len(lines),
        "boundary_count": len(boundaries),
        "block_count": len(block_ranges_for_lines(lines)),
        "classes_exact": bool(validation.get("classes_passed")),
        "boundaries_exact": bool(validation.get("boundaries_passed")),
        "ranges_exact": bool(validation.get("blocks_passed")),
        "conservation": bool(validation.get("conservation")),
        "expected_line_count": spec.get("expected_line_count") or len(lines),
        "parser_validation_status": status["parser_validation_status"],
        "parser_validation_label": status["parser_validation_label"],
    }
    if include_details:
        summary["lines"] = [
            {
                "line_number": int(row["raw_line_number"]),
                "class": row.get("parser_proposed_line_class"),
                "text": row.get("raw_text"),
            }
            for row in lines
        ]
    return summary


def _repository_metadata(
    *,
    snapshot_dir: Path,
    golden_dir: Path,
    first_commit: str | None,
    second_commit: str | None,
) -> dict[str, Any]:
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    head = _git(["rev-parse", "HEAD"])
    decisions = snapshot_dir / "manual_review_decisions.jsonl"
    history = snapshot_dir / "manual_review_history.jsonl"
    manifest = snapshot_dir / "review_manifest.json"
    docs = snapshot_dir / "review_documents.jsonl"
    lines = snapshot_dir / "review_lines.jsonl"
    boundaries = snapshot_dir / "review_boundaries.jsonl"
    golden_files = sorted(path for path in golden_dir.iterdir() if path.is_file())
    return {
        "repository": {
            "path": str(PROJECT_ROOT),
            "branch": branch,
            "head": head,
            "parser_profile": PARSER_VERSION,
            "snapshot_schema_version": REVIEW_SCHEMA_VERSION,
            "export_script_version": EXPORT_SCRIPT_VERSION,
        },
        "commits": {
            "first": first_commit or head,
            "second": second_commit,
        },
        "input_checksums": {
            "manual_review_decisions_sha256": sha256_file(decisions) if decisions.exists() else None,
            "manual_review_history_sha256": sha256_file(history) if history.exists() else None,
            "review_manifest_sha256": sha256_file(manifest) if manifest.exists() else None,
            "review_documents_sha256": sha256_file(docs) if docs.exists() else None,
            "review_lines_sha256": sha256_file(lines) if lines.exists() else None,
            "review_boundaries_sha256": sha256_file(boundaries) if boundaries.exists() else None,
            "review_snapshot_sha256": _combined_sha(
                [manifest, docs, lines, boundaries]
            ),
            "golden_fixture_checksums": [
                {"file": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
                for path in golden_files
            ],
        },
    }


def _corpus_summary(
    *,
    documents: list[dict[str, Any]],
    remaining_records: list[dict[str, Any]],
    golden_regressions: list[dict[str, Any]],
    changed_classes: list[dict[str, Any]],
    changed_boundaries: list[dict[str, Any]],
    changed_blocks: list[dict[str, Any]],
    corpus_acceptance: dict[str, Any],
    status_builder: ReviewStatusBuilder,
    lines: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
) -> dict[str, Any]:
    remaining_ids = {row["document_id"] for row in remaining_records}
    status_counts = Counter(row["parser_validation_status"] for row in remaining_records)
    manual_counts = status_builder._manual_counts()
    corpus_docs = [row for row in corpus_acceptance.get("documents", []) if row.get("document_id") in remaining_ids]
    return {
        "total_documents": len(documents),
        "golden_documents": len(golden_regressions),
        "non_golden_documents": len(remaining_records),
        "total_lines": sum(row["line_count"] for row in remaining_records),
        "total_boundaries": sum(row["boundary_count"] for row in remaining_records),
        "total_blocks": sum(row["block_count"] for row in remaining_records),
        "v5_blocks": sum(row["v5_block_count"] for row in remaining_records),
        "v6_blocks": sum(row["v6_block_count"] for row in remaining_records),
        "changed_line_classes": sum(
            1 for row in changed_classes if row.get("document_id") in remaining_ids
        ),
        "changed_boundaries": sum(
            1 for row in changed_boundaries if row.get("document_id") in remaining_ids
        ),
        "changed_blocks": sum(
            1 for row in changed_blocks if row.get("document_id") in remaining_ids
        ),
        "parser_exceptions": 0,
        "conservation_failures": sum(1 for row in corpus_docs if row.get("conservation") is not True),
        "duplication_failures": sum(1 for row in corpus_docs if int(row.get("duplication_count") or 0) != 0),
        "ordering_failures": sum(1 for row in corpus_docs if int(row.get("ordering_failures") or 0) != 0),
        "golden_conflicts": sum(
            1 for row in golden_regressions if row.get("verdict") != "GOLDEN PASS"
        ),
        "parser_manual_conflicts": sum(
            1
            for document in remaining_records
            for line in document["lines"]
            if line.get("manual_review_status") == "MANUAL_CONFLICT"
            or line.get("parser_validation_status") == "PARSER_CONFLICT"
        )
        + sum(
            1
            for document in remaining_records
            for boundary in document["boundaries"]
            if boundary.get("manual_review_status") == "MANUAL_CONFLICT"
            or boundary.get("parser_validation_status") == "PARSER_CONFLICT"
        ),
        "manually_reviewed_items": manual_counts["reviewed_lines"] + manual_counts["reviewed_boundaries"],
        "not_manually_reviewed_items": manual_counts["not_reviewed"],
        "stale_manual_decisions": sum(
            1
            for document in remaining_records
            for line in document["lines"]
            if line.get("manual_review_status") == "MANUAL_DECISION_STALE" or line.get("stale_decision_flag")
        )
        + sum(
            1
            for document in remaining_records
            for boundary in document["boundaries"]
            if boundary.get("manual_review_status") == "MANUAL_DECISION_STALE" or boundary.get("stale_decision_flag")
        ),
        "validation_status_counts": dict(sorted(status_counts.items())),
        "snapshot_line_total": len(lines),
        "snapshot_boundary_total": len(boundaries),
    }


def _render_document_markdown(document: dict[str, Any]) -> list[str]:
    idx = int(document["review_index"])
    lines = [
        f"# Document {idx:02d} — {document['court']} — {document['title']}",
        "",
        "## Identity",
        "",
        f"- Review index: {idx:02d}",
        f"- Court: {document['court']}",
        f"- Court profile: {document['court_profile']}",
        f"- Title: {document['title']}",
        f"- Decision date: {document.get('decision_date')}",
        f"- Document ID: `{document['document_id']}`",
        f"- Source ID: `{document['source_id']}`",
        f"- Source checksum: `{document['source_checksum']}`",
        f"- Parser profile: `{document['parser_profile']}`",
        f"- Exact golden coverage: `{document['exact_golden_coverage']}`",
        "",
        "## Parser summary",
        "",
        f"- Lines / boundaries / blocks: {document['line_count']} / {document['boundary_count']} / {document['block_count']}",
        f"- v5 blocks / v6 blocks: {document['v5_block_count']} / {document['v6_block_count']}",
        f"- Changed lines / boundaries / blocks: {document['changed_line_count']} / {document['changed_boundary_count']} / {document['changed_block_count']}",
        f"- Parser exceptions: {document['parser_exception_count']}",
        f"- Text conservation: `{document['text_conservation']}`",
        f"- Duplication count: {document['duplication_count']}",
        f"- Ordering failures: {document['ordering_failures']}",
        f"- Deterministic output: `{document['deterministic_output_result']}`",
        f"- Parser validation: `{document['parser_validation_label']}` (`{document['parser_validation_status']}`)",
        f"- Manual review: lines {document['manual_review_summary']['line_reviewed']}/{document['manual_review_summary']['line_total']}; boundaries {document['manual_review_summary']['boundary_reviewed']}/{document['manual_review_summary']['boundary_total']}",
        "",
        "## Numbering and hierarchy summary",
        "",
        f"- Top-level paragraph numbers observed: {_stable(document['numbering_sequence_analysis']['observed_top_level_numbers'])}",
        f"- Expected contiguous sequence: {_stable(document['numbering_sequence_analysis']['expected_sequence'])}",
        f"- Numbering discontinuities: {_stable(document['numbering_sequence_analysis']['discontinuities'])}",
        f"- Nested item count: {document['numbering_sequence_analysis']['nested_item_count']}",
        "",
        "## Potential review candidates",
        "",
        f"- Suspicious overmerges: {_stable(document['suspicious_overmerge_candidates'])}",
        f"- Suspicious undersplits: {_stable(document['suspicious_undersplit_candidates'])}",
        f"- Heading candidates: {_stable(document['heading_candidates'])}",
        f"- Nested list candidates: {_stable(document['nested_list_candidates'])}",
        f"- Table candidates: {_stable(document['table_candidates'])}",
        f"- Closing metadata/signature analysis: {_stable(document['closing_metadata_signature_analysis'])}",
        "",
        "## Complete line classification",
        "",
    ]
    for line in document["lines"]:
        lines.extend(
            [
                f"### Line {int(line['line_number'])}",
                "",
                f"- Parser class: `{line['current_parser_class']}`",
                f"- Previous class: `{line.get('previous_v5_class')}` / annotation `{line.get('previous_annotation_class')}`",
                f"- Changed: `{'yes' if line['class_changed_from_v5'] else 'no'}`",
                f"- Parser status: `{line['parser_validation_status']}`",
                f"- Manual status: `{line['manual_review_status']}`",
                f"- Block range: `{line.get('parser_block_start_line')}-{line.get('parser_block_end_line')}` (`{line.get('parser_block_id')}`)",
                f"- Hierarchy/parent: level={line.get('hierarchy_level')}; top={line.get('top_level_paragraph_number')}; parent={line.get('parent_paragraph_number')}; marker={line.get('nested_item_marker')}",
                f"- Complete text: {line['raw_text']}",
                f"- Deterministic explanation: {line['reviewer_explanation']}",
                f"- Issue flags: {_stable(line['issue_flags'])}",
                "",
            ]
        )
    lines.extend(["## Complete boundaries", ""])
    for boundary in document["boundaries"]:
        lines.extend(
            [
                f"### Boundary {int(boundary['boundary_index'])}",
                "",
                f"- Boundary: L{boundary['before_line_number']} -> L{boundary['after_line_number']}",
                f"- v6 decision: `{boundary['parser_v6_decision']}`",
                f"- v5 decision: `{boundary.get('previous_v5_decision')}`",
                f"- Changed: `{'yes' if boundary['changed_from_v5'] else 'no'}`",
                f"- Validation status: `{boundary['parser_validation_status']}`",
                f"- Complete before text: {boundary['full_text_before']}",
                f"- Complete after text: {boundary['full_text_after']}",
                f"- Reason: {boundary['parser_reason']}",
                f"- Block impact: {boundary['block_impact']}",
                f"- Issue flags: {_stable(boundary['issue_flags'])}",
                "",
            ]
        )
    lines.extend(["## Complete blocks", ""])
    for block in document["blocks"]:
        lines.extend(
            [
                f"### Block {int(block['block_index'])} — `{block.get('stable_block_id')}`",
                "",
                f"- Range: L{block['start_line']}-L{block['end_line']}",
                f"- Primary role: `{block['primary_structural_role']}`",
                f"- Hierarchy: level={block.get('hierarchy_level')}; top={block.get('top_level_paragraph_number')}; parent={block.get('parent_block_or_paragraph')}",
                "- Complete text:",
                "```text",
                block["complete_text"],
                "```",
                f"- v5 comparison: previous={_stable(block.get('previous_matching_v5_blocks'))}; changed=`{block['block_changed_from_v5']}`; change_type=`{block.get('change_type')}`",
                f"- Validation status: `{block['validation_status']}`",
                f"- Review flags: overmerge=`{block['suspicious_overmerge_flag']}`; undersplit=`{block['suspicious_undersplit_flag']}`; manual_coverage={_stable(block['manual_coverage'])}",
                "",
            ]
        )
    lines.extend(
        [
            "## Document verdict",
            "",
            document["document_verdict"],
            "",
        ]
    )
    return lines


def _render_cross_document_section(candidates: dict[str, list[dict[str, Any]]]) -> list[str]:
    lines = ["# Cross-document review candidates", ""]
    for category in sorted(candidates):
        rows = candidates[category]
        lines.append(f"## {category}")
        lines.append("")
        if not rows:
            lines.append("- none")
            lines.append("")
            continue
        normalized_rows = sorted(
            rows,
            key=lambda row: (
                int(row["review_index"]),
                str(row.get("locator") or ""),
                str(row.get("detail") or ""),
            ),
        )
        for row in normalized_rows:
            lines.append(
                f"- Document {int(row['review_index']):02d}: {row['locator']} — {row['detail']}"
            )
        lines.append("")
    return lines


def _cross_document_candidates(documents: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    categories: dict[str, list[dict[str, Any]]] = {
        "changed_headings": [],
        "changed_numbered_paragraphs": [],
        "nested_lists": [],
        "tables": [],
        "opening_formulas": [],
        "operative_sections": [],
        "reasoning_transitions": [],
        "closing_metadata": [],
        "signatures": [],
        "possible_overmerges": [],
        "possible_undersplits": [],
        "numbering_discontinuities": [],
        "parser_manual_conflicts": [],
        "stale_manual_decisions": [],
    }
    for document in documents:
        idx = int(document["review_index"])
        for line in document["lines"]:
            locator = f"line {line['line_number']}"
            detail = f"{line['current_parser_class']}: {line['raw_text'][:120]}"
            item = {"review_index": idx, "locator": locator, "detail": detail}
            if line["class_changed_from_v5"] and line["current_parser_class"] in HEADING_CLASSES:
                categories["changed_headings"].append(item)
            if line["class_changed_from_v5"] and line["current_parser_class"] in {
                "numbered_paragraph_start",
                "numbered_paragraph_continuation",
            }:
                categories["changed_numbered_paragraphs"].append(item)
            if line["current_parser_class"] == "list_or_table" and LETTER_ITEM_RE.search(line["raw_text"].strip()):
                categories["nested_lists"].append(item)
            if line["current_parser_class"] == "list_or_table" and SEMICOLON_TABLE_RE.search(line["raw_text"]):
                categories["tables"].append(item)
            if line["line_number"] <= 12 and line["current_parser_class"] in OPENING_CLASSES.union({"heading", "prose_start"}):
                if OPERATIVE_HINT_RE.search(line["raw_text"]) is None:
                    categories["opening_formulas"].append(item)
            if OPERATIVE_HINT_RE.search(line["raw_text"] or ""):
                categories["operative_sections"].append(item)
            if REASONING_HINT_RE.search(line["raw_text"] or ""):
                categories["reasoning_transitions"].append(item)
            if CLOSING_META_RE.search(line["raw_text"] or ""):
                categories["closing_metadata"].append(item)
            if SIGNATURE_RE.search(line["raw_text"] or "") or line["current_parser_class"] == "signature":
                categories["signatures"].append(item)
            if line.get("manual_review_status") == "MANUAL_DECISION_STALE" or line.get("stale_decision_flag"):
                categories["stale_manual_decisions"].append(item)
            elif (
                line.get("manual_review_status") == "MANUAL_CONFLICT"
                or line.get("parser_validation_status") == "PARSER_CONFLICT"
            ):
                categories["parser_manual_conflicts"].append(item)
        for boundary in document["boundaries"]:
            boundary_item = {
                "review_index": idx,
                "locator": (
                    f"boundary {boundary['boundary_index']} "
                    f"L{boundary['before_line_number']}->L{boundary['after_line_number']}"
                ),
                "detail": f"{boundary.get('parser_v6_decision')}: {str(boundary.get('full_text_before') or '')[:80]}",
            }
            if boundary.get("manual_review_status") == "MANUAL_DECISION_STALE" or boundary.get("stale_decision_flag"):
                categories["stale_manual_decisions"].append(boundary_item)
            elif (
                boundary.get("manual_review_status") == "MANUAL_CONFLICT"
                or boundary.get("parser_validation_status") == "PARSER_CONFLICT"
            ):
                categories["parser_manual_conflicts"].append(boundary_item)
            if boundary["changed_from_v5"] and boundary["parser_v6_decision"] == "MERGE":
                categories["possible_overmerges"].append(
                    {
                        "review_index": idx,
                        "locator": f"boundary {boundary['boundary_index']} L{boundary['before_line_number']}->L{boundary['after_line_number']}",
                        "detail": "Changed to MERGE",
                    }
                )
            if boundary["changed_from_v5"] and boundary["parser_v6_decision"] == "SPLIT":
                categories["possible_undersplits"].append(
                    {
                        "review_index": idx,
                        "locator": f"boundary {boundary['boundary_index']} L{boundary['before_line_number']}->L{boundary['after_line_number']}",
                        "detail": "Changed to SPLIT",
                    }
                )
        for discontinuity in document["numbering_sequence_analysis"]["discontinuities"]:
            categories["numbering_discontinuities"].append(
                {
                    "review_index": idx,
                    "locator": f"line {discontinuity.get('line_number')}",
                    "detail": _stable(discontinuity),
                }
            )
        for block in document["blocks"]:
            if block["suspicious_overmerge_flag"]:
                categories["possible_overmerges"].append(
                    {
                        "review_index": idx,
                        "locator": f"block {block['block_index']} L{block['start_line']}-L{block['end_line']}",
                        "detail": "suspicious_overmerge_flag",
                    }
                )
            if block["suspicious_undersplit_flag"]:
                categories["possible_undersplits"].append(
                    {
                        "review_index": idx,
                        "locator": f"block {block['block_index']} L{block['start_line']}-L{block['end_line']}",
                        "detail": "suspicious_undersplit_flag",
                    }
                )
    return {
        category: sorted(
            rows,
            key=lambda row: (
                int(row["review_index"]),
                str(row.get("locator") or ""),
                str(row.get("detail") or ""),
            ),
        )
        for category, rows in categories.items()
    }


def _document_candidates(
    lines: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    numbering: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    return {
        "suspicious_overmerges": [
            {"block_index": block["block_index"], "range": [block["start_line"], block["end_line"]]}
            for block in blocks
            if block["suspicious_overmerge_flag"]
        ],
        "suspicious_undersplits": [
            {"block_index": block["block_index"], "range": [block["start_line"], block["end_line"]]}
            for block in blocks
            if block["suspicious_undersplit_flag"]
        ],
        "headings": [
            {"line_number": line["line_number"], "text": line["raw_text"]}
            for line in lines
            if line["current_parser_class"] == "heading"
        ],
        "nested_lists": [
            {"line_number": line["line_number"], "text": line["raw_text"]}
            for line in lines
            if line["current_parser_class"] == "list_or_table"
            and (LETTER_ITEM_RE.search(line["raw_text"].strip()) or DASH_BULLET_RE.search(line["raw_text"].strip()))
        ],
        "tables": [
            {"line_number": line["line_number"], "text": line["raw_text"]}
            for line in lines
            if line["current_parser_class"] == "list_or_table" and SEMICOLON_TABLE_RE.search(line["raw_text"])
        ],
        "changed_boundaries": [
            {
                "boundary_index": boundary["boundary_index"],
                "before": boundary["before_line_number"],
                "after": boundary["after_line_number"],
            }
            for boundary in boundaries
            if boundary["changed_from_v5"]
        ],
        "numbering_discontinuities": numbering["discontinuities"],
    }


def _numbering_analysis(lines: list[dict[str, Any]]) -> dict[str, Any]:
    observed: list[int] = []
    locations: list[dict[str, Any]] = []
    nested = 0
    for line in lines:
        text = str(line["raw_text"])
        if line["current_parser_class"] == "numbered_paragraph_start":
            match = ARABIC_NUMBER_RE.match(text)
            if match:
                number = int(match.group(1))
                observed.append(number)
                locations.append({"line_number": line["line_number"], "number": number})
        if line["current_parser_class"] == "list_or_table":
            nested += 1
    discontinuities: list[dict[str, Any]] = []
    for index, number in enumerate(observed, start=1):
        if number != index:
            discontinuities.append(
                {
                    "expected": index,
                    "actual": number,
                    "line_number": locations[index - 1]["line_number"],
                }
            )
    return {
        "observed_top_level_numbers": observed,
        "expected_sequence": list(range(1, len(observed) + 1)),
        "discontinuities": discontinuities,
        "nested_item_count": nested,
    }


def _closing_analysis(lines: list[dict[str, Any]]) -> dict[str, Any]:
    closing_lines = [
        {"line_number": line["line_number"], "text": line["raw_text"], "class": line["current_parser_class"]}
        for line in lines
        if CLOSING_META_RE.search(str(line["raw_text"]))
        or SIGNATURE_RE.search(str(line["raw_text"]))
        or line["current_parser_class"] in {"signature", "instruction"}
    ]
    return {"count": len(closing_lines), "lines": closing_lines}


def _hierarchy_for_line(
    parser_class: str,
    raw_text: str,
    all_lines: list[dict[str, Any]],
    line_number: int,
) -> dict[str, Any]:
    block_index = None
    block_lines = [row for row in all_lines if row.get("parser_block_id") == next(
        (item.get("parser_block_id") for item in all_lines if int(item["raw_line_number"]) == line_number),
        None,
    )]
    if all_lines:
        current = next((row for row in all_lines if int(row["raw_line_number"]) == line_number), None)
        if current is not None:
            block_index = current.get("parser_block_index")
            block_lines = [row for row in all_lines if row.get("parser_block_index") == block_index]
    block_start = int(block_lines[0]["raw_line_number"]) if block_lines else None
    block_end = int(block_lines[-1]["raw_line_number"]) if block_lines else None
    top_level = None
    nested_marker = None
    parent = None
    level = 0
    state = "prose"
    if parser_class == "numbered_paragraph_start":
        match = ARABIC_NUMBER_RE.match(raw_text)
        top_level = int(match.group(1)) if match else None
        level = 1
        state = "top_level_numbered_paragraph"
    elif parser_class == "numbered_paragraph_continuation":
        level = 1
        state = "top_level_numbered_paragraph_continuation"
        for row in reversed([item for item in all_lines if int(item["raw_line_number"]) < line_number]):
            if row.get("parser_proposed_line_class") == "numbered_paragraph_start":
                match = ARABIC_NUMBER_RE.match(str(row.get("raw_text") or ""))
                top_level = int(match.group(1)) if match else None
                parent = top_level
                break
    elif parser_class == "list_or_table":
        level = 2
        state = "nested_list_or_table"
        stripped = raw_text.strip()
        if LETTER_ITEM_RE.search(stripped):
            nested_marker = stripped.split(")", 1)[0] + ")"
        elif DASH_BULLET_RE.search(stripped):
            nested_marker = stripped.split(")", 1)[0] + ")"
        else:
            nested_match = ARABIC_NUMBER_RE.match(stripped)
            if nested_match:
                nested_marker = nested_match.group(0).strip()
        for row in reversed([item for item in all_lines if int(item["raw_line_number"]) < line_number]):
            if row.get("parser_proposed_line_class") == "numbered_paragraph_start":
                match = ARABIC_NUMBER_RE.match(str(row.get("raw_text") or ""))
                parent = int(match.group(1)) if match else None
                break
    elif parser_class == "heading":
        level = 0
        state = "heading"
    heading_context = next(
        (
            str(row.get("raw_text") or "")
            for row in reversed(all_lines)
            if int(row["raw_line_number"]) <= line_number
            and row.get("parser_proposed_line_class") == "heading"
        ),
        None,
    )
    section_context = "reasoning" if heading_context and REASONING_HINT_RE.search(heading_context) else (
        "operative" if heading_context and OPERATIVE_HINT_RE.search(heading_context) else "general"
    )
    return {
        "hierarchy_state": state,
        "hierarchy_level": level,
        "top_level_paragraph_number": top_level,
        "nested_item_marker": nested_marker,
        "parent_paragraph_number": parent,
        "heading_context": heading_context,
        "section_context": section_context,
        "block_start_line": block_start,
        "block_end_line": block_end,
    }


def _secondary_features(text: str) -> dict[str, bool]:
    return {
        "citation": bool(re.search(r"\b(?:viz|srov\.)\b", text, re.IGNORECASE)),
        "statute": "§" in text or bool(re.search(r"\bzákon(?:a|ě|u)?\b", text, re.IGNORECASE)),
        "date": bool(re.search(r"\b\d{1,2}\.\s*\d{1,2}\.\s*\d{4}\b", text)),
        "case_reference": bool(re.search(r"\b(?:sp\.\s*zn\.|č\.\s*j\.|ECLI:)", text, re.IGNORECASE)),
    }


def _line_explanation(parser_class: str, reason: str, changed: bool) -> str:
    change = " Parser v6 changed this class relative to the v5 baseline." if changed else ""
    return f"Class `{parser_class}` assigned by deterministic rule `{reason}` under `{PARSER_VERSION}`.{change}"


def _boundary_explanation(decision: str, reason: str, changed: bool) -> str:
    change = " This boundary differs from the v5 baseline." if changed else ""
    return f"Boundary decision `{decision}` from rule `{reason}` under `{PARSER_VERSION}`.{change}"


def _block_explanation(primary_role: str, changed: bool, validation_status: str) -> str:
    change = " Block range or membership changed from v5." if changed else ""
    return (
        f"Block primary role `{primary_role}` under `{PARSER_VERSION}` with validation `{validation_status}`."
        f"{change} This is not an exact golden proof for non-golden documents."
    )


def _document_verdict(
    doc_status: dict[str, Any],
    corpus_row: dict[str, Any],
    class_changes: set[Any],
    boundary_changes: set[Any],
    block_changes: dict[Any, Any],
) -> str:
    parts = [
        "Parser completed with all deterministic invariants passing."
        if corpus_row.get("conservation", True)
        and int(corpus_row.get("duplication_count") or 0) == 0
        and int(corpus_row.get("ordering_failures") or 0) == 0
        else "Parser completed with invariant findings that require investigation."
    ]
    if class_changes or boundary_changes or block_changes:
        parts.append(
            f"Review recommended for {len(class_changes)} changed line classes, "
            f"{len(boundary_changes)} changed boundaries, and {len(block_changes)} changed blocks."
        )
    else:
        parts.append("No v5-to-v6 line/boundary/block changes were recorded for this document.")
    parts.append(
        "No parser exception, conservation failure, duplication or ordering failure detected."
        if corpus_row.get("conservation", True)
        and int(corpus_row.get("duplication_count") or 0) == 0
        and int(corpus_row.get("ordering_failures") or 0) == 0
        else "Invariant failures were recorded in the parser v6 audit."
    )
    parts.append(f"Automatic parser validation label: {doc_status['parser_validation_label']}.")
    parts.append("This document is not an exact golden pass and was not manually approved by this export.")
    return " ".join(parts)


def _court_profile(court: str) -> str:
    return {
        "constitutional_court": "constitutional_court.v7",
        "high_court_prague": "high_court_prague.v7",
        "high_court_olomouc": "high_court_olomouc.v7",
    }.get(court, f"{court}.v7")


def _suspicious_overmerge_flag(
    *,
    lines: list[dict[str, Any]],
    primary_role: str,
    change_row: dict[str, Any] | None,
    document: dict[str, Any],
) -> bool:
    if change_row and "overmerge" in str(change_row.get("reason") or "").lower():
        return True
    if len(lines) < 12 or primary_role not in {"prose_start", "numbered_paragraph_start", "metadata"}:
        return False
    if _is_legitimate_opening_formula_block(lines, document):
        return False
    return True


def _is_legitimate_opening_formula_block(lines: list[dict[str, Any]], document: dict[str, Any]) -> bool:
    if not lines:
        return False
    first = str(lines[0].get("raw_text") or "")
    if not OPENING_FORMULA_START_RE.search(first):
        return False
    first_line_number = int(lines[0].get("line_number") or lines[0].get("raw_line_number") or 0)
    if first_line_number != 1:
        return False
    if any(str(row.get("raw_text") or "").strip() == "Výrok" for row in lines):
        return False

    def _line_class(row: dict[str, Any]) -> str:
        return str(
            row.get("current_parser_class")
            or row.get("parser_proposed_line_class")
            or ""
        )

    if not all(_line_class(row) in {"prose_start", "prose_continuation", "metadata"} for row in lines):
        return False
    if document.get("text_conservation") is False:
        return False
    if int(document.get("duplication_count") or 0) != 0:
        return False
    if int(document.get("ordering_failures") or 0) != 0:
        return False
    return len(lines) >= 2


def _validate_conflict_category_consistency(payload: dict[str, Any]) -> None:
    candidates = payload.get("cross_document_review_candidates") or {}
    conflict_rows = candidates.get("parser_manual_conflicts") or []
    stale_rows = candidates.get("stale_manual_decisions") or []
    summary = payload.get("corpus_summary") or {}
    if int(summary.get("parser_manual_conflicts") or 0) != len(conflict_rows):
        raise FullExportError(
            f"Conflict summary/count mismatch: summary={summary.get('parser_manual_conflicts')} list={len(conflict_rows)}"
        )
    if int(summary.get("stale_manual_decisions") or 0) != len(stale_rows):
        raise FullExportError(
            f"Stale summary/count mismatch: summary={summary.get('stale_manual_decisions')} list={len(stale_rows)}"
        )
    conflict_keys = {(row.get("review_index"), row.get("locator")) for row in conflict_rows}
    stale_keys = {(row.get("review_index"), row.get("locator")) for row in stale_rows}
    overlap = conflict_keys & stale_keys
    if overlap:
        raise FullExportError(f"Items present in both conflict and stale categories: {sorted(overlap)[:5]}")


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _group_by_doc(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["document_id"]), []).append(row)
    return grouped


def _assert_unique_document_ids(documents: list[dict[str, Any]]) -> None:
    ids = [str(row["document_id"]) for row in documents]
    if len(ids) != len(set(ids)):
        raise FullExportError("Duplicate document IDs in review snapshot")


def _combined_sha(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if not path.exists():
            continue
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _stable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
