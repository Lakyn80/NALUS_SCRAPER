from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import REVIEW_SCHEMA_VERSION, read_jsonl, write_json


def latest_decisions(review_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for record in read_jsonl(review_dir / "manual_review_decisions.jsonl"):
        key = (str(record.get("item_type")), str(record.get("item_id")))
        if int(record.get("revision_number", 0)) >= int(latest.get(key, {}).get("revision_number", 0)):
            latest[key] = record
    return latest


def duplicate_active_decisions(review_dir: Path) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    duplicates: list[dict[str, Any]] = []
    for record in read_jsonl(review_dir / "manual_review_decisions.jsonl"):
        key = (str(record.get("item_type")), str(record.get("item_id")))
        if key in seen:
            duplicates.append(record)
        seen.add(key)
    return duplicates


def apply_manual_status(review_dir: Path, records: list[dict[str, Any]], item_type: str) -> list[dict[str, Any]]:
    latest = latest_decisions(review_dir)
    output: list[dict[str, Any]] = []
    for record in records:
        decision = latest.get((item_type, str(record.get("item_id"))))
        copy = dict(record)
        copy["manual_decision_status"] = decision.get("decision_status") if decision else "pending"
        if decision:
            copy["manual_class"] = decision.get("manual_class")
            copy["manual_boundary_decision"] = decision.get("manual_boundary_decision")
            copy["manual_revision_number"] = decision.get("revision_number")
        output.append(copy)
    return output


def compute_progress(review_dir: Path) -> dict[str, Any]:
    documents = read_jsonl(review_dir / "review_documents.jsonl")
    lines = read_jsonl(review_dir / "review_lines.jsonl")
    boundaries = read_jsonl(review_dir / "review_boundaries.jsonl")
    latest = latest_decisions(review_dir)
    assisted_decisions = {
        key: decision
        for key, decision in latest.items()
        if str(decision.get("interface") or "").startswith("assisted_") or decision.get("assisted_batch_id")
    }
    line_done = {item_id for item_type, item_id in latest if item_type == "line"}
    boundary_done = {item_id for item_type, item_id in latest if item_type == "boundary"}
    unresolved_lines = {
        item_id
        for (item_type, item_id), decision in latest.items()
        if item_type == "line" and decision.get("decision_status") == "unresolved"
    }
    unresolved_boundaries = {
        item_id
        for (item_type, item_id), decision in latest.items()
        if item_type == "boundary" and decision.get("decision_status") == "unresolved"
    }
    by_doc: dict[str, dict[str, Any]] = {}
    for document in documents:
        by_doc[str(document["document_id"])] = {
            "document_id": document["document_id"],
            "review_number": document["review_number"],
            "source_id": document["source_id"],
            "court": document["court"],
            "line_total": 0,
            "line_reviewed": 0,
            "boundary_total": 0,
            "boundary_reviewed": 0,
            "line_unresolved": 0,
            "boundary_unresolved": 0,
        }
    for line in lines:
        row = by_doc[str(line["document_id"])]
        row["line_total"] += 1
        item_id = str(line["item_id"])
        row["line_reviewed"] += int(item_id in line_done)
        row["line_unresolved"] += int(item_id in unresolved_lines)
    for boundary in boundaries:
        row = by_doc[str(boundary["document_id"])]
        row["boundary_total"] += 1
        item_id = str(boundary["item_id"])
        row["boundary_reviewed"] += int(item_id in boundary_done)
        row["boundary_unresolved"] += int(item_id in unresolved_boundaries)
    incomplete_documents = [
        row
        for row in by_doc.values()
        if row["line_reviewed"] < row["line_total"]
        or row["boundary_reviewed"] < row["boundary_total"]
        or row["line_unresolved"]
        or row["boundary_unresolved"]
    ]
    summary = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "snapshot_status": "ready",
        "document_count": len(documents),
        "line_total": len(lines),
        "line_reviewed": len(line_done),
        "line_pending": max(0, len(lines) - len(line_done)),
        "line_unresolved": len(unresolved_lines),
        "boundary_total": len(boundaries),
        "boundary_reviewed": len(boundary_done),
        "boundary_pending": max(0, len(boundaries) - len(boundary_done)),
        "boundary_unresolved": len(unresolved_boundaries),
        "unresolved_items": len(unresolved_lines) + len(unresolved_boundaries),
        "incomplete_documents": len(incomplete_documents),
        "manual_review_complete": len(incomplete_documents) == 0,
        "manual_review_status": "complete" if len(incomplete_documents) == 0 else "incomplete",
        "decision_records": len(latest),
        "manual_individual_decision_records": len(latest) - len(assisted_decisions),
        "assisted_batch_decision_records": len(assisted_decisions),
        "documents": list(by_doc.values()),
    }
    return summary


def write_progress_files(review_dir: Path) -> dict[str, Any]:
    progress = compute_progress(review_dir)
    write_json(review_dir / "review_progress.json", progress)
    lines = [
        "# Manual Parser Review Summary",
        "",
        f"- Schema: `{REVIEW_SCHEMA_VERSION}`",
        f"- Documents: `{progress['document_count']}`",
        f"- Snapshot: `{progress['snapshot_status']}`",
        f"- Manual review: `{progress['manual_review_status']}`",
        f"- Lines reviewed: `{progress['line_reviewed']}/{progress['line_total']}`",
        f"- Boundaries reviewed: `{progress['boundary_reviewed']}/{progress['boundary_total']}`",
        f"- Incomplete documents: `{progress['incomplete_documents']}`",
        f"- Unresolved items: `{progress['unresolved_items']}`",
        f"- Individual decisions: `{progress['manual_individual_decision_records']}`",
        f"- Assisted batch decisions: `{progress['assisted_batch_decision_records']}`",
        "",
        "## Documents",
        "",
    ]
    for row in progress["documents"]:
        lines.append(
            f"- {row['review_number']:02d} `{row['document_id']}` `{row['court']}`: "
            f"lines {row['line_reviewed']}/{row['line_total']}, "
            f"boundaries {row['boundary_reviewed']}/{row['boundary_total']}"
        )
    (review_dir / "manual_review_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return progress
