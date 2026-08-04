from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .full_export import (
    DEFAULT_OUTPUT_DIR,
    JSON_NAME,
    MARKDOWN_NAME,
    V6_OUTPUT_DIR,
    V6_JSON_NAME,
    V6_MARKDOWN_NAME,
    GOLDEN_DOCUMENT_IDS,
    GOLDEN_REVIEW_NUMBERS,
    build_export_payload,
    document_markdown_section,
)
from .models import PROJECT_ROOT, REVIEW_SCHEMA_VERSION, read_jsonl
from .progress import apply_manual_status, compute_progress
from .assisted import build_assisted_review, load_assisted_artifacts, occurrences_for_rule
from .batches import apply_batch, revert_batch
from .status import AUDIT_DIR, V6_AUDIT_DIR, ReviewStatusBuilder, block_ranges_for_lines
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
        if path == "/api/parser-v7/changes":
            document_id = _optional_param(params, "document_id")
            return 200, {
                "schema_version": REVIEW_SCHEMA_VERSION,
                **self._parser_changes(document_id, audit_dir=AUDIT_DIR, current_range_key="v7_range"),
            }
        if path == "/api/parser-v6/changes":
            document_id = _optional_param(params, "document_id")
            return 200, {
                "schema_version": REVIEW_SCHEMA_VERSION,
                **self._parser_changes(document_id, audit_dir=V6_AUDIT_DIR, current_range_key="v6_range"),
            }
        if path == "/api/full-corpus-v7":
            return 200, {
                "schema_version": REVIEW_SCHEMA_VERSION,
                **self._full_corpus(
                    export_dir=DEFAULT_OUTPUT_DIR,
                    json_name=JSON_NAME,
                    markdown_name=MARKDOWN_NAME,
                    audit_dir=AUDIT_DIR,
                ),
            }
        if path == "/api/full-corpus-v6":
            return 200, {
                "schema_version": REVIEW_SCHEMA_VERSION,
                **self._full_corpus(
                    export_dir=V6_OUTPUT_DIR,
                    json_name=V6_JSON_NAME,
                    markdown_name=V6_MARKDOWN_NAME,
                    audit_dir=V6_AUDIT_DIR,
                ),
            }
        if path == "/api/full-corpus-v7/document-markdown":
            document_id = _param(params, "document_id")
            return 200, {
                "schema_version": REVIEW_SCHEMA_VERSION,
                **self._document_review_markdown(
                    document_id,
                    export_dir=DEFAULT_OUTPUT_DIR,
                    json_name=JSON_NAME,
                ),
            }
        if path == "/api/full-corpus-v6/document-markdown":
            document_id = _param(params, "document_id")
            return 200, {
                "schema_version": REVIEW_SCHEMA_VERSION,
                **self._document_review_markdown(
                    document_id,
                    export_dir=V6_OUTPUT_DIR,
                    json_name=V6_JSON_NAME,
                ),
            }
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

    def _parser_changes(
        self,
        document_id: str | None,
        *,
        audit_dir: Path,
        current_range_key: str,
    ) -> dict[str, Any]:
        classes = read_jsonl(audit_dir / "changed_line_classes.jsonl")
        boundaries = read_jsonl(audit_dir / "changed_boundaries.jsonl")
        blocks = read_jsonl(audit_dir / "changed_blocks.jsonl")
        if document_id:
            classes = [row for row in classes if row.get("document_id") == document_id]
            boundaries = [row for row in boundaries if row.get("document_id") == document_id]
            blocks = [row for row in blocks if row.get("document_id") == document_id]
        builder = ReviewStatusBuilder(self.review_dir, audit_dir=audit_dir)
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
            {
                **row,
                **builder.parser_status_for_line(line_rows[(str(row["document_id"]), int(row["line"]))]),
                **builder.manual_status_for_item("line", line_rows[(str(row["document_id"]), int(row["line"]))]),
            }
            for row in classes
            if (str(row["document_id"]), int(row["line"])) in line_rows
        ]
        enriched_boundaries = [
            {
                **row,
                **builder.parser_status_for_boundary(
                    boundary_rows[(str(row["document_id"]), int(row["before_line"]))]
                ),
                **builder.manual_status_for_item(
                    "boundary",
                    boundary_rows[(str(row["document_id"]), int(row["before_line"]))],
                ),
            }
            for row in boundaries
            if (str(row["document_id"]), int(row["before_line"])) in boundary_rows
        ]
        enriched_blocks = []
        for row in blocks:
            doc = document_rows.get(str(row["document_id"]))
            if doc is None:
                continue
            current_range = row.get(current_range_key) or row.get("v7_range") or row.get("v6_range")
            if not isinstance(current_range, list) or len(current_range) != 2:
                continue
            enriched_blocks.append(
                {**row, **builder.parser_status_for_block(doc, [int(current_range[0]), int(current_range[1])])}
            )
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

    def _full_corpus(
        self,
        *,
        export_dir: Path,
        json_name: str,
        markdown_name: str,
        audit_dir: Path,
    ) -> dict[str, Any]:
        builder = ReviewStatusBuilder(self.review_dir, audit_dir=audit_dir)
        documents = read_jsonl(self.review_dir / "review_documents.jsonl")
        enriched = []
        for row in documents:
            status = builder.document_status(row)
            review_number = int(row["review_number"])
            is_golden = review_number in GOLDEN_REVIEW_NUMBERS or str(row["document_id"]) in GOLDEN_DOCUMENT_IDS
            lines = [line for line in builder.lines if line["document_id"] == row["document_id"]]
            boundaries = [boundary for boundary in builder.boundaries if boundary["document_id"] == row["document_id"]]
            changed_lines = sum(
                1
                for line in lines
                if (str(row["document_id"]), int(line["raw_line_number"])) in builder._changed_class_keys
            )
            changed_boundaries = sum(
                1
                for boundary in boundaries
                if (str(row["document_id"]), int(boundary["previous_line_number"])) in builder._changed_boundary_keys
            )
            block_ranges = block_ranges_for_lines(lines)
            changed_blocks = sum(
                1
                for line_range in block_ranges
                if (str(row["document_id"]), (int(line_range[0]), int(line_range[1]))) in builder._changed_block_keys
            )
            enriched.append(
                {
                    **row,
                    **status,
                    "exact_golden_coverage": is_golden,
                    "corpus_group": "golden" if is_golden else "remaining",
                    "changed_line_count": changed_lines,
                    "changed_boundary_count": changed_boundaries,
                    "changed_block_count": changed_blocks,
                    "hierarchy_summary": {
                        "block_count": len(block_ranges),
                        "numbered_paragraph_count": sum(
                            1 for line in lines if line.get("parser_proposed_line_class") == "numbered_paragraph_start"
                        ),
                        "list_or_table_count": sum(
                            1 for line in lines if line.get("parser_proposed_line_class") == "list_or_table"
                        ),
                        "heading_count": sum(1 for line in lines if line.get("parser_proposed_line_class") == "heading"),
                    },
                    "potential_review_candidates": {
                        "changed_lines": changed_lines,
                        "changed_boundaries": changed_boundaries,
                        "changed_blocks": changed_blocks,
                        "review_recommended": status["parser_validation_status"] == "PARSER_CHANGED_NEEDS_REVIEW",
                    },
                    "display_parser_label": "GOLDEN PASS"
                    if is_golden and status["parser_validation_label"] == "GOLDEN PASS"
                    else (
                        "REVIEW RECOMMENDED"
                        if status["parser_validation_status"] == "PARSER_CHANGED_NEEDS_REVIEW"
                        else status["parser_validation_label"]
                    ),
                }
            )
        json_path = export_dir / json_name
        md_path = export_dir / markdown_name
        return {
            "documents": enriched,
            "golden_documents": [row for row in enriched if row["corpus_group"] == "golden"],
            "remaining_documents": [row for row in enriched if row["corpus_group"] == "remaining"],
            "exports": {
                "json_path": str(json_path.relative_to(PROJECT_ROOT)) if json_path.exists() else None,
                "markdown_path": str(md_path.relative_to(PROJECT_ROOT)) if md_path.exists() else None,
                "json_url": f"/exports/{json_name}",
                "markdown_url": f"/exports/{markdown_name}",
                "json_exists": json_path.exists(),
                "markdown_exists": md_path.exists(),
            },
        }

    def _document_review_markdown(
        self,
        document_id: str,
        *,
        export_dir: Path = DEFAULT_OUTPUT_DIR,
        json_name: str = JSON_NAME,
    ) -> dict[str, Any]:
        document = self._document(document_id)
        export_path = export_dir / json_name
        if export_path.exists():
            payload = json.loads(export_path.read_text(encoding="utf-8"))
        else:
            payload = build_export_payload(snapshot_dir=self.review_dir)
        markdown = document_markdown_section(payload, str(document["document_id"]))
        return {
            "document_id": document["document_id"],
            "review_number": document["review_number"],
            "markdown": markdown,
        }


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
