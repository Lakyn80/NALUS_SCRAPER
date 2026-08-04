from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import PROJECT_ROOT, REVIEW_SCHEMA_VERSION, read_jsonl
from .progress import apply_manual_status, compute_progress
from .assisted import build_assisted_review, load_assisted_artifacts, occurrences_for_rule
from .batches import apply_batch, revert_batch
from .status import ReviewStatusBuilder
from .store import append_decision


class ReviewApi:
    def __init__(self, review_dir: Path) -> None:
        self.review_dir = review_dir

    def get(self, path: str, params: dict[str, list[str]]) -> tuple[int, dict[str, Any]]:
        if path == "/api/manifest":
            return 200, self._json("review_manifest.json")
        if path == "/api/progress":
            progress = compute_progress(self.review_dir)
            return 200, self._status_builder().progress(progress)
        if path == "/api/documents":
            documents = read_jsonl(self.review_dir / "review_documents.jsonl")
            builder = self._status_builder()
            return 200, {"schema_version": REVIEW_SCHEMA_VERSION, "documents": [{**row, **builder.document_status(row)} for row in documents]}
        if path == "/api/document":
            doc = self._document(_param(params, "id"))
            return 200, {"schema_version": REVIEW_SCHEMA_VERSION, "document": {**doc, **self._status_builder().document_status(doc)}}
        if path == "/api/lines":
            document_id = self._document(_param(params, "document_id"))["document_id"]
            rows = [row for row in read_jsonl(self.review_dir / "review_lines.jsonl") if row["document_id"] == document_id]
            rows = apply_manual_status(self.review_dir, rows, "line")
            builder = self._status_builder()
            return 200, {"schema_version": REVIEW_SCHEMA_VERSION, "lines": [builder.enrich_line(row) for row in rows]}
        if path == "/api/boundaries":
            document_id = self._document(_param(params, "document_id"))["document_id"]
            rows = [row for row in read_jsonl(self.review_dir / "review_boundaries.jsonl") if row["document_id"] == document_id]
            rows = apply_manual_status(self.review_dir, rows, "boundary")
            builder = self._status_builder()
            return 200, {"schema_version": REVIEW_SCHEMA_VERSION, "boundaries": [builder.enrich_boundary(row) for row in rows]}
        if path == "/api/boundary-cards":
            document_id = self._document(_param(params, "document_id"))["document_id"]
            return 200, {
                "schema_version": REVIEW_SCHEMA_VERSION,
                "boundaries": boundary_cards(self.review_dir, document_id),
            }
        if path == "/api/problems":
            return 200, {"schema_version": REVIEW_SCHEMA_VERSION, **self._problems(_optional_param(params, "document_id"))}
        if path == "/api/summary":
            summary = (self.review_dir / "manual_review_summary.md").read_text(encoding="utf-8")
            return 200, {"schema_version": REVIEW_SCHEMA_VERSION, "markdown": summary}
        if path == "/api/assisted/summary":
            return 200, {"schema_version": REVIEW_SCHEMA_VERSION, "summary": load_assisted_artifacts(self.review_dir)["summary"]}
        if path == "/api/assisted/rules":
            artifacts = load_assisted_artifacts(self.review_dir)
            return 200, {"schema_version": REVIEW_SCHEMA_VERSION, "rules": artifacts["rules"], "batches": artifacts["batches"]}
        if path == "/api/parser-v5/changes":
            document_id = _optional_param(params, "document_id")
            return 200, {"schema_version": REVIEW_SCHEMA_VERSION, **self._parser_v5_changes(document_id)}
        if path == "/api/parser-v6/changes":
            document_id = _optional_param(params, "document_id")
            return 200, {"schema_version": REVIEW_SCHEMA_VERSION, **self._parser_v6_changes(document_id)}
        if path.startswith("/api/assisted/rules/"):
            parts = [part for part in path.split("/") if part]
            if len(parts) >= 4:
                rule_id = parts[3]
                if len(parts) == 5 and parts[4] == "occurrences":
                    return 200, {"schema_version": REVIEW_SCHEMA_VERSION, "occurrences": occurrences_for_rule(self.review_dir, rule_id)}
                artifacts = load_assisted_artifacts(self.review_dir)
                rule = next((item for item in artifacts["rules"] if item["rule_id"] == rule_id), None)
                if rule:
                    return 200, {"schema_version": REVIEW_SCHEMA_VERSION, "rule": rule}
            return 404, {"schema_version": REVIEW_SCHEMA_VERSION, "error": "rule_not_found"}
        return 404, {"schema_version": REVIEW_SCHEMA_VERSION, "error": "not_found"}

    def post(self, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if path == "/api/decision":
            payload = dict(payload)
            payload["interface"] = "html"
            return 200, {"schema_version": REVIEW_SCHEMA_VERSION, "decision": append_decision(self.review_dir, payload)}
        if path == "/api/assisted/apply":
            try:
                result = apply_batch(
                    self.review_dir,
                    rule_id=str(payload.get("rule_id") or ""),
                    confirmation=str(payload.get("confirmation") or ""),
                    interface="assisted_html",
                )
                build_assisted_review(review_dir=self.review_dir)
                return 200, {"schema_version": REVIEW_SCHEMA_VERSION, "result": result}
            except ValueError as exc:
                return 400, {"schema_version": REVIEW_SCHEMA_VERSION, "error": str(exc)}
        if path == "/api/assisted/revert":
            try:
                result = revert_batch(
                    self.review_dir,
                    batch_id=str(payload.get("batch_id") or ""),
                    confirmation=str(payload.get("confirmation") or ""),
                    interface="assisted_html",
                )
                build_assisted_review(review_dir=self.review_dir)
                return 200, {"schema_version": REVIEW_SCHEMA_VERSION, "result": result}
            except ValueError as exc:
                return 400, {"schema_version": REVIEW_SCHEMA_VERSION, "error": str(exc)}
        return 404, {"schema_version": REVIEW_SCHEMA_VERSION, "error": "not_found"}

    def _json(self, name: str) -> dict[str, Any]:
        return json.loads((self.review_dir / name).read_text(encoding="utf-8"))

    def _parser_v5_changes(self, document_id: str | None) -> dict[str, Any]:
        audit_dir = PROJECT_ROOT / "artifacts" / "legal_v2" / "constitutional_parser_v5"
        boundaries = read_jsonl(audit_dir / "constitutional_changed_boundaries.jsonl")
        classes = read_jsonl(audit_dir / "constitutional_changed_classes.jsonl")
        if document_id:
            boundaries = [row for row in boundaries if row.get("document_id") == document_id]
            classes = [row for row in classes if row.get("document_id") == document_id]
        return {
            "changed_boundaries": boundaries,
            "changed_classes": classes,
            "boundary_count": len(boundaries),
            "class_count": len(classes),
        }

    def _parser_v6_changes(self, document_id: str | None) -> dict[str, Any]:
        audit_dir = PROJECT_ROOT / "artifacts" / "legal_v2" / "parser_v6_audit"
        classes = read_jsonl(audit_dir / "changed_line_classes.jsonl")
        boundaries = read_jsonl(audit_dir / "changed_boundaries.jsonl")
        blocks = read_jsonl(audit_dir / "changed_blocks.jsonl")
        if document_id:
            classes = [row for row in classes if row.get("document_id") == document_id]
            boundaries = [row for row in boundaries if row.get("document_id") == document_id]
            blocks = [row for row in blocks if row.get("document_id") == document_id]
        builder = self._status_builder()
        line_rows = {
            (str(row["document_id"]), int(row["raw_line_number"])): row
            for row in read_jsonl(self.review_dir / "review_lines.jsonl")
        }
        boundary_rows = {
            (str(row["document_id"]), int(row["previous_line_number"])): row
            for row in read_jsonl(self.review_dir / "review_boundaries.jsonl")
        }
        document_rows = {str(row["document_id"]): row for row in read_jsonl(self.review_dir / "review_documents.jsonl")}
        enriched_classes = [
            {**row, **builder.parser_status_for_line(line_rows[(str(row["document_id"]), int(row["line"]))]), **builder.manual_status_for_item("line", line_rows[(str(row["document_id"]), int(row["line"]))])}
            for row in classes
            if (str(row["document_id"]), int(row["line"])) in line_rows
        ]
        enriched_boundaries = [
            {**row, **builder.parser_status_for_boundary(boundary_rows[(str(row["document_id"]), int(row["before_line"]))]), **builder.manual_status_for_item("boundary", boundary_rows[(str(row["document_id"]), int(row["before_line"]))])}
            for row in boundaries
            if (str(row["document_id"]), int(row["before_line"])) in boundary_rows
        ]
        enriched_blocks = [
            {**row, **builder.parser_status_for_block(document_rows[str(row["document_id"])], row["v6_range"])}
            for row in blocks
            if str(row["document_id"]) in document_rows
        ]
        return {
            "changed_classes": enriched_classes,
            "changed_boundaries": enriched_boundaries,
            "changed_blocks": enriched_blocks,
            "class_count": len(classes),
            "boundary_count": len(boundaries),
            "block_count": len(blocks),
        }

    def _problems(self, document_id: str | None) -> dict[str, Any]:
        builder = self._status_builder()
        lines = [builder.enrich_line(row) for row in read_jsonl(self.review_dir / "review_lines.jsonl")]
        boundaries = [builder.enrich_boundary(row) for row in read_jsonl(self.review_dir / "review_boundaries.jsonl")]
        if document_id:
            lines = [row for row in lines if row.get("document_id") == document_id]
            boundaries = [row for row in boundaries if row.get("document_id") == document_id]
        parser_conflicts = [
            row for row in [*lines, *boundaries] if row.get("parser_validation_status") == "PARSER_CONFLICT"
        ]
        manual_conflicts = [
            row for row in [*lines, *boundaries] if row.get("manual_review_status") in {"MANUAL_CONFLICT", "MANUAL_DECISION_STALE"}
        ]
        return {
            "parser_conflicts": parser_conflicts,
            "manual_conflicts": manual_conflicts,
            "parser_conflict_count": len(parser_conflicts),
            "manual_conflict_count": len(manual_conflicts),
        }

    def _document(self, document_id: str) -> dict[str, Any]:
        for row in read_jsonl(self.review_dir / "review_documents.jsonl"):
            if document_id in {str(row["document_id"]), str(row["source_id"]), str(row["review_number"])}:
                return row
        raise ValueError(f"Unknown document: {document_id}")

    def _status_builder(self) -> ReviewStatusBuilder:
        return ReviewStatusBuilder(self.review_dir)


def _param(params: dict[str, list[str]], name: str) -> str:
    values = params.get(name) or []
    if not values:
        raise ValueError(f"Missing required query parameter: {name}")
    return values[0]


def _optional_param(params: dict[str, list[str]], name: str) -> str | None:
    values = params.get(name) or []
    return values[0] if values else None


def boundary_cards(review_dir: Path, document_id: str) -> list[dict[str, Any]]:
    lines = [row for row in read_jsonl(review_dir / "review_lines.jsonl") if row["document_id"] == document_id]
    boundaries = [row for row in read_jsonl(review_dir / "review_boundaries.jsonl") if row["document_id"] == document_id]
    lines = apply_manual_status(review_dir, lines, "line")
    boundaries = apply_manual_status(review_dir, boundaries, "boundary")
    builder = ReviewStatusBuilder(review_dir)
    lines = [builder.enrich_line(row) for row in lines]
    boundaries = [builder.enrich_boundary(row) for row in boundaries]
    by_number = {int(row["raw_line_number"]): row for row in lines}
    cards: list[dict[str, Any]] = []
    for index, boundary in enumerate(boundaries, start=1):
        before = by_number.get(int(boundary["previous_line_number"]))
        after = by_number.get(int(boundary["next_line_number"]))
        if before is None or after is None:
            continue
        parser_display = _boundary_display(bool(boundary.get("parser_proposed_boundary")))
        previous_value = boundary.get("previous_automated_boundary_annotation")
        previous_display = _boundary_display(previous_value if isinstance(previous_value, bool) else None)
        manual = _manual_boundary(boundary, parser_display)
        cards.append(
            {
                "boundary_id": boundary["item_id"],
                "boundary_number": index,
                "document_id": document_id,
                "status": boundary.get("manual_decision_status", "pending"),
                "parser_validation_status": boundary["parser_validation_status"],
                "parser_validation_label": boundary["parser_validation_label"],
                "parser_validation_reason": boundary["parser_validation_reason"],
                "golden_covered": boundary["golden_covered"],
                "golden_match": boundary["golden_match"],
                "invariant_validated": boundary["invariant_validated"],
                "manual_review_status": boundary["manual_review_status"],
                "manual_review_label": boundary["manual_review_label"],
                "manual_revision": boundary["manual_revision"],
                "before": _line_summary(before),
                "after": _line_summary(after),
                "context_before": [_line_summary(row) for row in lines if int(before["raw_line_number"]) - 2 <= int(row["raw_line_number"]) < int(before["raw_line_number"])],
                "context_after": [_line_summary(row) for row in lines if int(after["raw_line_number"]) < int(row["raw_line_number"]) <= int(after["raw_line_number"]) + 2],
                "parser_boundary": parser_display,
                "previous_boundary": previous_display,
                "manual_decision": manual,
                "reason_code": boundary.get("parser_reason_code"),
                "suspicious_reasons": list(boundary.get("suspicious_reason_codes") or []),
                "conflict": _conflict_text(parser_display, previous_display),
                "same_parser_block": before.get("parser_block_id") == after.get("parser_block_id"),
                "parser_block_context": _parser_block_context(before, after, parser_display),
                "source_checksum": boundary.get("source_checksum"),
            }
        )
    return cards


def _line_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "line_number": row.get("raw_line_number"),
        "page": row.get("source_page"),
        "raw_text": row.get("raw_text"),
        "parser_class": row.get("parser_proposed_line_class"),
        "previous_class": row.get("previous_automated_annotation"),
        "parser_block_id": row.get("parser_block_id"),
        "parser_block_index": row.get("parser_block_index"),
        "item_id": row.get("item_id"),
        "parser_validation_status": row.get("parser_validation_status"),
        "parser_validation_label": row.get("parser_validation_label"),
        "parser_validation_reason": row.get("parser_validation_reason"),
        "manual_review_status": row.get("manual_review_status"),
        "manual_review_label": row.get("manual_review_label"),
    }


def _boundary_display(value: bool | None) -> dict[str, Any]:
    if value is True:
        return {
            "value": True,
            "display": "SPLIT",
            "explanation": "The following line starts a new block.",
        }
    if value is False:
        return {
            "value": False,
            "display": "MERGE",
            "explanation": "The following line remains in the same block as the preceding line.",
        }
    return {
        "value": None,
        "display": "UNKNOWN",
        "explanation": "No previous boundary decision is available.",
    }


def _manual_boundary(boundary: dict[str, Any], parser_display: dict[str, Any]) -> dict[str, Any]:
    stored = boundary.get("manual_boundary_decision")
    if stored == "split":
        display = "SPLIT"
        explanation = f"Force SPLIT before line {boundary['next_line_number']}."
    elif stored == "merge":
        display = "MERGE"
        explanation = f"Force MERGE with line {boundary['previous_line_number']}."
    elif stored == "unresolved":
        display = "UNRESOLVED"
        explanation = "Mark this boundary unresolved."
    elif stored == "preserve_parser":
        display = parser_display["display"]
        explanation = f"preserve_parser -> {display}. Accept parser result: {display}."
    else:
        display = "PENDING"
        explanation = f"No manual decision yet. preserve_parser would save: {parser_display['display']}."
    return {"stored_value": stored, "effective_display": display, "explanation": explanation}


def _conflict_text(parser_display: dict[str, Any], previous_display: dict[str, Any]) -> dict[str, Any]:
    if previous_display["display"] == "UNKNOWN":
        return {"has_conflict": False, "text": f"Parser says {parser_display['display']}; no previous boundary annotation is available."}
    if parser_display["display"] == previous_display["display"]:
        return {"has_conflict": False, "text": f"Parser and previous annotation both say {parser_display['display']}."}
    return {
        "has_conflict": True,
        "text": f"Parser says {parser_display['display']}; previous annotation says {previous_display['display']}.",
    }


def _parser_block_context(before: dict[str, Any], after: dict[str, Any], parser_display: dict[str, Any]) -> str:
    before_block = str(before.get("parser_block_id") or "unknown")
    after_block = str(after.get("parser_block_id") or "unknown")
    if parser_display["display"] == "MERGE":
        return f"Current parser block {before_block} contains both lines."
    return f"Parser ends block {before_block} after line {before['raw_line_number']} and starts block {after_block} at line {after['raw_line_number']}."
