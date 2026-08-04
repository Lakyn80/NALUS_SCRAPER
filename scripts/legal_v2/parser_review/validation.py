from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .manifest import load_design_documents, parser_git_identity
from .models import DEFAULT_REVIEW_DIR, REVIEW_SCHEMA_VERSION, read_jsonl
from .progress import compute_progress, duplicate_active_decisions, write_progress_files


def validate_review(review_dir: Path = DEFAULT_REVIEW_DIR, *, strict_complete: bool = False, write_summary: bool = False) -> dict[str, Any]:
    snapshot_errors: list[dict[str, Any]] = []
    completion_errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    _, documents = load_design_documents()
    expected_documents = {item.review_id for item in documents}
    expected_checksums = {item.review_id: item.source_checksum for item in documents}
    manifest_path = review_dir / "review_manifest.json"
    if not manifest_path.exists():
        snapshot_errors.append({"code": "missing_review_manifest", "path": str(manifest_path)})
        return _result(snapshot_errors, completion_errors, warnings, None, summary_written=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != REVIEW_SCHEMA_VERSION:
        snapshot_errors.append({"code": "manifest_schema_mismatch"})
    git_identity = parser_git_identity()
    if manifest.get("head") != git_identity["head"]:
        snapshot_errors.append({"code": "parser_git_identity_changed"})
    if manifest.get("parser_profile") != git_identity["parser_profile"]:
        snapshot_errors.append({"code": "parser_profile_changed"})
    rows_by_name: dict[str, list[dict[str, Any]]] = {}
    for name in ("review_documents.jsonl", "review_lines.jsonl", "review_boundaries.jsonl", "manual_review_decisions.jsonl", "manual_review_history.jsonl"):
        try:
            rows_by_name[name] = read_jsonl(review_dir / name)
        except ValueError as exc:
            snapshot_errors.append({"code": "invalid_jsonl", "path": name, "message": str(exc)})
    docs = rows_by_name.get("review_documents.jsonl", [])
    if {str(row.get("document_id")) for row in docs} != expected_documents:
        snapshot_errors.append({"code": "review_document_identity_mismatch"})
    lines = rows_by_name.get("review_lines.jsonl", [])
    boundaries = rows_by_name.get("review_boundaries.jsonl", [])
    line_ids = {str(row.get("item_id")) for row in lines}
    boundary_ids = {str(row.get("item_id")) for row in boundaries}
    if len(line_ids) != len(lines):
        snapshot_errors.append({"code": "duplicate_line_ids"})
    if len(boundary_ids) != len(boundaries):
        snapshot_errors.append({"code": "duplicate_boundary_ids"})
    if not lines:
        snapshot_errors.append({"code": "review_lines_missing"})
    if not boundaries:
        snapshot_errors.append({"code": "review_boundaries_missing"})
    for row in lines + boundaries + rows_by_name.get("manual_review_decisions.jsonl", []):
        if row.get("schema_version") != REVIEW_SCHEMA_VERSION:
            snapshot_errors.append({"code": "schema_mismatch", "item_id": row.get("item_id")})
    duplicate_decisions = duplicate_active_decisions(review_dir)
    if duplicate_decisions:
        completion_errors.append({"code": "duplicate_active_decisions", "count": len(duplicate_decisions)})
    stale_checksum_count = 0
    stale_profile_count = 0
    stale_identity_count = 0
    unresolved_line_count = 0
    unresolved_boundary_count = 0
    for decision in rows_by_name.get("manual_review_decisions.jsonl", []):
        item_type = decision.get("item_type")
        item_id = str(decision.get("item_id"))
        document_id = str(decision.get("document_id") or "")
        if item_type == "line" and item_id not in line_ids:
            completion_errors.append({"code": "unknown_decision_line", "item_id": item_id})
        if item_type == "boundary" and item_id not in boundary_ids:
            completion_errors.append({"code": "unknown_decision_boundary", "item_id": item_id})
        if document_id in expected_checksums and decision.get("source_checksum") != expected_checksums[document_id]:
            stale_checksum_count += 1
        if decision.get("parser_profile") != manifest.get("parser_profile"):
            stale_profile_count += 1
        if decision.get("parser_git_identity") != manifest.get("head"):
            stale_identity_count += 1
        if item_type == "line" and decision.get("decision_status") == "unresolved":
            unresolved_line_count += 1
        if item_type == "boundary" and decision.get("decision_status") == "unresolved":
            unresolved_boundary_count += 1
    progress = write_progress_files(review_dir) if write_summary else compute_progress(review_dir)
    if progress["line_pending"] > 0:
        completion_errors.append({"code": "manual_line_decisions_missing", "remaining": progress["line_pending"]})
    if progress["boundary_pending"] > 0:
        completion_errors.append({"code": "manual_boundary_decisions_missing", "remaining": progress["boundary_pending"]})
    if progress["incomplete_documents"] > 0:
        completion_errors.append({"code": "manual_documents_incomplete", "remaining": progress["incomplete_documents"]})
    if unresolved_line_count:
        completion_errors.append({"code": "manual_line_decisions_unresolved", "count": unresolved_line_count})
    if unresolved_boundary_count:
        completion_errors.append({"code": "manual_boundary_decisions_unresolved", "count": unresolved_boundary_count})
    if stale_checksum_count:
        completion_errors.append({"code": "stale_source_checksum_decisions", "count": stale_checksum_count})
    if stale_profile_count:
        completion_errors.append({"code": "stale_parser_profile_decisions", "count": stale_profile_count})
    if stale_identity_count:
        completion_errors.append({"code": "stale_parser_git_identity_decisions", "count": stale_identity_count})
    return _result(snapshot_errors, completion_errors, warnings, progress, summary_written=write_summary)


def _result(
    snapshot_errors: list[dict[str, Any]],
    completion_errors: list[dict[str, Any]],
    warnings: list[str],
    progress: dict[str, Any] | None,
    *,
    summary_written: bool,
) -> dict[str, Any]:
    snapshot_status = "pass" if not snapshot_errors else "fail"
    completion_status = "pass" if not completion_errors else "fail"
    status = "pass" if snapshot_status == "pass" and completion_status == "pass" else "fail"
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "status": status,
        "snapshot_integrity": {
            "status": snapshot_status,
            "errors": snapshot_errors,
        },
        "manual_review_completion": {
            "status": completion_status,
            "errors": completion_errors,
        },
        "errors": [*snapshot_errors, *completion_errors],
        "warnings": warnings,
        "progress": progress,
        "summary_written": summary_written,
    }
