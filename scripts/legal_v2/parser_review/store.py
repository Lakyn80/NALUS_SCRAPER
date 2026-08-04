from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from .manifest import parser_git_identity
from .models import (
    DECISION_STATUSES,
    DEFAULT_REVIEW_DIR,
    INTERFACES,
    ITEM_TYPES,
    MANUAL_BOUNDARY_DECISIONS,
    MANUAL_LINE_CLASSES,
    REVIEW_SCHEMA_VERSION,
    REVIEW_TOOL_VERSION,
    read_jsonl,
    utc_now,
)
from .progress import write_progress_files


class StoreLock:
    def __init__(self, path: Path, timeout_s: float = 10.0) -> None:
        self.path = path
        self.timeout_s = timeout_s
        self.fd: int | None = None

    def __enter__(self) -> "StoreLock":
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, str(os.getpid()).encode("ascii"))
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for review store lock: {self.path}")
                time.sleep(0.05)

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def append_decision(review_dir: Path, request: dict[str, Any]) -> dict[str, Any]:
    review_dir.mkdir(parents=True, exist_ok=True)
    tmp_files = sorted(review_dir.glob("manual_review_decisions.jsonl.tmp-*"))
    if tmp_files:
        valid_tmp = [str(path.name) for path in tmp_files if _valid_jsonl(path)]
        invalid_tmp = [str(path.name) for path in tmp_files if not _valid_jsonl(path)]
        if invalid_tmp:
            raise ValueError(f"Abandoned invalid temporary store files exist: {invalid_tmp}")
        if valid_tmp:
            raise ValueError(f"Abandoned temporary store files exist and require manual inspection: {valid_tmp}")
    with StoreLock(review_dir / "manual_review_decisions.lock"):
        active_path = review_dir / "manual_review_decisions.jsonl"
        history_path = review_dir / "manual_review_history.jsonl"
        active_records = read_jsonl(active_path)
        _validate_existing_records(active_records)
        record = _decision_record(review_dir, request, active_records)
        latest_by_key = {(row["item_type"], row["item_id"]): row for row in active_records}
        latest_by_key[(record["item_type"], record["item_id"])] = record
        ordered = sorted(latest_by_key.values(), key=lambda row: (str(row["document_id"]), str(row["item_type"]), str(row["item_id"])))
        tmp_path = review_dir / f"manual_review_decisions.jsonl.tmp-{os.getpid()}"
        with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in ordered:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, active_path)
        with history_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        write_progress_files(review_dir)
        return record


def _decision_record(review_dir: Path, request: dict[str, Any], active_records: list[dict[str, Any]]) -> dict[str, Any]:
    item_type = str(request.get("item_type") or "")
    item_id = str(request.get("item_id") or "")
    document_id = str(request.get("document_id") or "")
    decision_status = str(request.get("decision_status") or "")
    interface = str(request.get("interface") or "")
    if item_type not in ITEM_TYPES:
        raise ValueError(f"Unsupported item_type: {item_type}")
    if decision_status not in DECISION_STATUSES:
        raise ValueError(f"Unsupported decision_status: {decision_status}")
    if interface not in INTERFACES:
        raise ValueError(f"Unsupported interface: {interface}")
    manifest = json.loads((review_dir / "review_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError("Unsupported review manifest schema")
    item = _load_item(review_dir, item_type, item_id, document_id)
    source_checksum = str(request.get("source_checksum") or item.get("source_checksum") or "")
    if source_checksum != item.get("source_checksum"):
        raise ValueError("Stale source checksum")
    git_identity = parser_git_identity()
    if str(request.get("parser_profile") or git_identity["parser_profile"]) != manifest.get("parser_profile"):
        raise ValueError("Stale parser profile")
    if str(request.get("parser_git_identity") or git_identity["head"]) != manifest.get("head"):
        raise ValueError("Stale parser Git identity")
    manual_class = request.get("manual_class")
    manual_boundary_decision = request.get("manual_boundary_decision")
    if item_type == "line":
        manual_class = manual_class or item.get("parser_proposed_line_class") or "unresolved"
        if manual_class not in MANUAL_LINE_CLASSES:
            manual_class = "unresolved" if decision_status == "accepted" else manual_class
        if manual_class not in MANUAL_LINE_CLASSES:
            raise ValueError(f"Unsupported manual class: {manual_class}")
    if item_type == "boundary":
        manual_boundary_decision = manual_boundary_decision or "preserve_parser"
        if manual_boundary_decision not in MANUAL_BOUNDARY_DECISIONS:
            raise ValueError(f"Unsupported manual boundary decision: {manual_boundary_decision}")
    if item_type == "document_approval":
        manual_class = manual_class or "unresolved"
        manual_boundary_decision = manual_boundary_decision or "unresolved"
    revision_number = 1 + max(
        (int(row.get("revision_number", 0)) for row in active_records if row.get("item_type") == item_type and row.get("item_id") == item_id),
        default=0,
    )
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "document_id": item["document_id"],
        "source_checksum": source_checksum,
        "parser_profile": manifest["parser_profile"],
        "parser_git_identity": manifest["head"],
        "item_type": item_type,
        "item_id": item_id,
        "raw_line_number": item.get("raw_line_number") or item.get("previous_line_number"),
        "parser_proposal": _parser_proposal(item),
        "previous_automated_annotation": item.get("previous_automated_annotation")
        or item.get("previous_automated_boundary_annotation"),
        "manual_class": manual_class,
        "manual_boundary_decision": manual_boundary_decision,
        "decision_status": decision_status,
        "reviewer_comment": str(request.get("reviewer_comment") or ""),
        "revision_number": revision_number,
        "timestamp": utc_now(),
        "interface": interface,
        "review_tool_version": REVIEW_TOOL_VERSION,
        "assisted_rule_id": request.get("assisted_rule_id"),
        "assisted_batch_id": request.get("assisted_batch_id"),
        "assisted_source_evidence_document_ids": list(request.get("assisted_source_evidence_document_ids") or []),
        "pre_batch_decision": request.get("pre_batch_decision"),
    }


def _load_item(review_dir: Path, item_type: str, item_id: str, document_id: str) -> dict[str, Any]:
    if item_type == "line":
        rows = read_jsonl(review_dir / "review_lines.jsonl")
    elif item_type == "boundary":
        rows = read_jsonl(review_dir / "review_boundaries.jsonl")
    else:
        rows = read_jsonl(review_dir / "review_documents.jsonl")
        for row in rows:
            if row.get("document_id") == document_id:
                row = dict(row)
                row["item_id"] = document_id
                return row
        raise ValueError(f"Unknown document for approval: {document_id}")
    for row in rows:
        if row.get("item_id") == item_id:
            if document_id and row.get("document_id") != document_id:
                raise ValueError("Decision document_id does not match item")
            return row
    raise ValueError(f"Unknown {item_type} item: {item_id}")


def _parser_proposal(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item.get(key)
        for key in (
            "parser_proposed_line_class",
            "parser_proposed_boundary",
            "parser_proposed_boundary_before",
            "parser_proposed_boundary_after",
            "parser_proposed_boundary_type",
            "parser_reason_code",
        )
        if key in item
    }


def _validate_existing_records(records: list[dict[str, Any]]) -> None:
    for row in records:
        if row.get("schema_version") != REVIEW_SCHEMA_VERSION:
            raise ValueError("Existing store contains incompatible schema version")
        if row.get("item_type") not in ITEM_TYPES:
            raise ValueError("Existing store contains unsupported item_type")


def _valid_jsonl(path: Path) -> bool:
    try:
        read_jsonl(path)
    except ValueError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Append a parser-review manual decision.")
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--decision-json", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = json.loads(args.decision_json.read_text(encoding="utf-8"))
    record = append_decision(args.review_dir, payload)
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
