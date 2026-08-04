from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .assisted import load_assisted_artifacts, occurrences_for_rule
from .models import DEFAULT_REVIEW_DIR, REVIEW_SCHEMA_VERSION, read_jsonl, utc_now
from .progress import write_progress_files
from .store import StoreLock, _decision_record, _validate_existing_records


def apply_batch(
    review_dir: Path = DEFAULT_REVIEW_DIR,
    *,
    rule_id: str,
    confirmation: str,
    interface: str = "assisted_cli",
) -> dict[str, Any]:
    occurrences = [item for item in occurrences_for_rule(review_dir, rule_id) if not item.get("excluded")]
    expected = f"APPLY {rule_id} {len(occurrences)}"
    if confirmation != expected:
        raise ValueError(f"Confirmation mismatch. Expected: {expected}")
    if not occurrences:
        raise ValueError(f"No applicable occurrences for rule: {rule_id}")
    artifacts = load_assisted_artifacts(review_dir)
    rule = next((item for item in artifacts["rules"] if item["rule_id"] == rule_id), None)
    if not rule or rule["confidence"] != "SAFE":
        raise ValueError("Only SAFE rules may be batch-applied")
    batch_id = f"batch-{rule_id.removeprefix('rule-')}-{utc_now().replace(':', '').replace('-', '')}"
    return _write_batch(review_dir, rule=rule, occurrences=occurrences, batch_id=batch_id, interface=interface)


def revert_batch(
    review_dir: Path = DEFAULT_REVIEW_DIR,
    *,
    batch_id: str,
    confirmation: str,
    interface: str = "assisted_cli",
) -> dict[str, Any]:
    expected = f"REVERT {batch_id}"
    if confirmation != expected:
        raise ValueError(f"Confirmation mismatch. Expected: {expected}")
    with StoreLock(review_dir / "manual_review_decisions.lock"):
        active_path = review_dir / "manual_review_decisions.jsonl"
        history_path = review_dir / "manual_review_history.jsonl"
        active = read_jsonl(active_path)
        history = read_jsonl(history_path)
        _validate_existing_records(active)
        batch_records = [row for row in active if row.get("assisted_batch_id") == batch_id]
        if not batch_records:
            raise ValueError(f"No active records found for batch: {batch_id}")
        restored: dict[tuple[str, str], dict[str, Any] | None] = {}
        for record in batch_records:
            key = (str(record["item_type"]), str(record["item_id"]))
            later_independent = [
                row
                for row in history
                if row.get("item_type") == key[0]
                and row.get("item_id") == key[1]
                and int(row.get("revision_number", 0)) > int(record.get("revision_number", 0))
                and row.get("assisted_batch_id") != batch_id
            ]
            if later_independent:
                raise ValueError(f"Cannot revert {batch_id}; later independent decision exists for {key[1]}")
            previous = record.get("pre_batch_decision")
            restored[key] = previous if isinstance(previous, dict) else None
        latest_by_key = {(str(row["item_type"]), str(row["item_id"])): row for row in active}
        for key, previous in restored.items():
            if previous is None:
                latest_by_key.pop(key, None)
            else:
                latest_by_key[key] = previous
        revert_events = [
            {
                "schema_version": REVIEW_SCHEMA_VERSION,
                "event_type": "assisted_batch_revert",
                "item_type": key[0],
                "item_id": key[1],
                "assisted_batch_id": batch_id,
                "interface": interface,
                "timestamp": utc_now(),
            }
            for key in restored
        ]
        _replace_store(active_path, sorted(latest_by_key.values(), key=lambda row: (str(row["document_id"]), str(row["item_type"]), str(row["item_id"]))))
        with history_path.open("a", encoding="utf-8", newline="\n") as handle:
            for event in revert_events:
                handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _append_batch_log(review_dir, {"event": "revert", "batch_id": batch_id, "count": len(revert_events), "timestamp": utc_now()})
        write_progress_files(review_dir)
        return {"status": "reverted", "batch_id": batch_id, "reverted_count": len(revert_events)}


def _write_batch(review_dir: Path, *, rule: dict[str, Any], occurrences: list[dict[str, Any]], batch_id: str, interface: str) -> dict[str, Any]:
    with StoreLock(review_dir / "manual_review_decisions.lock"):
        active_path = review_dir / "manual_review_decisions.jsonl"
        history_path = review_dir / "manual_review_history.jsonl"
        active = read_jsonl(active_path)
        _validate_existing_records(active)
        latest_by_key = {(str(row["item_type"]), str(row["item_id"])): row for row in active}
        new_records: list[dict[str, Any]] = []
        working = list(active)
        for occurrence in occurrences:
            key = (str(occurrence["item_type"]), str(occurrence["item_id"]))
            if key in latest_by_key:
                raise ValueError(f"Existing manual decision would be overwritten: {key[1]}")
            request = {
                "item_type": occurrence["item_type"],
                "item_id": occurrence["item_id"],
                "document_id": occurrence["document_id"],
                "source_checksum": occurrence["source_checksum"],
                "decision_status": "overridden",
                "manual_class": occurrence.get("proposed_manual_class"),
                "manual_boundary_decision": occurrence.get("proposed_boundary_decision"),
                "interface": interface,
                "assisted_rule_id": rule["rule_id"],
                "assisted_batch_id": batch_id,
                "assisted_source_evidence_document_ids": rule["source_document_ids"],
                "pre_batch_decision": latest_by_key.get(key),
            }
            record = _decision_record(review_dir, request, working)
            new_records.append(record)
            working.append(record)
            latest_by_key[key] = record
        _replace_store(active_path, sorted(latest_by_key.values(), key=lambda row: (str(row["document_id"]), str(row["item_type"]), str(row["item_id"]))))
        with history_path.open("a", encoding="utf-8", newline="\n") as handle:
            for record in new_records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _append_batch_log(review_dir, {"event": "apply", "batch_id": batch_id, "rule_id": rule["rule_id"], "count": len(new_records), "timestamp": utc_now()})
        write_progress_files(review_dir)
        return {"status": "applied", "batch_id": batch_id, "applied_count": len(new_records), "rule_id": rule["rule_id"]}


def _replace_store(path: Path, records: list[dict[str, Any]]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _append_batch_log(review_dir: Path, payload: dict[str, Any]) -> None:
    path = review_dir / "assisted" / "batch_application_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
