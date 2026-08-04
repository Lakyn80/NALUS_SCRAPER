from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from app.rag.legal_v2.audit import PARSER_VERSION

from .models import PROJECT_ROOT, sha256_file, read_jsonl


class ParserValidationStatus(str, Enum):
    AUTO_VALIDATED_GOLDEN = "AUTO_VALIDATED_GOLDEN"
    PARSER_VALIDATED = "PARSER_VALIDATED"
    PARSER_CHANGED_NEEDS_REVIEW = "PARSER_CHANGED_NEEDS_REVIEW"
    PARSER_CONFLICT = "PARSER_CONFLICT"
    PARSER_UNVALIDATED = "PARSER_UNVALIDATED"


class ManualReviewStatus(str, Enum):
    NOT_MANUALLY_REVIEWED = "NOT_MANUALLY_REVIEWED"
    MANUALLY_ACCEPTED = "MANUALLY_ACCEPTED"
    MANUALLY_OVERRIDDEN = "MANUALLY_OVERRIDDEN"
    MANUAL_DECISION_STALE = "MANUAL_DECISION_STALE"
    MANUAL_CONFLICT = "MANUAL_CONFLICT"


PARSER_LABELS = {
    ParserValidationStatus.AUTO_VALIDATED_GOLDEN: "AUTO-VALIDATED · GOLDEN v6",
    ParserValidationStatus.PARSER_VALIDATED: "PARSER VALIDATED",
    ParserValidationStatus.PARSER_CHANGED_NEEDS_REVIEW: "CHANGED BY PARSER v6 · REVIEW RECOMMENDED",
    ParserValidationStatus.PARSER_CONFLICT: "CONFLICT",
    ParserValidationStatus.PARSER_UNVALIDATED: "PARSER NOT VALIDATED",
}

MANUAL_LABELS = {
    ManualReviewStatus.NOT_MANUALLY_REVIEWED: "Manual review: not performed",
    ManualReviewStatus.MANUALLY_ACCEPTED: "Manually accepted",
    ManualReviewStatus.MANUALLY_OVERRIDDEN: "Manually overridden",
    ManualReviewStatus.MANUAL_DECISION_STALE: "Manual decision stale",
    ManualReviewStatus.MANUAL_CONFLICT: "Manual conflict",
}

GOLDEN_DIR = PROJECT_ROOT / "artifacts" / "legal_v2" / "parser_golden_inputs"
AUDIT_DIR = PROJECT_ROOT / "artifacts" / "legal_v2" / "parser_v6_audit"


class ReviewStatusBuilder:
    def __init__(
        self,
        review_dir: Path,
        *,
        audit_dir: Path = AUDIT_DIR,
        golden_dir: Path = GOLDEN_DIR,
    ) -> None:
        self.review_dir = review_dir
        self.audit_dir = audit_dir
        self.golden_dir = golden_dir
        self.manifest = _read_json(review_dir / "review_manifest.json")
        self.documents = read_jsonl(review_dir / "review_documents.jsonl")
        self.lines = read_jsonl(review_dir / "review_lines.jsonl")
        self.boundaries = read_jsonl(review_dir / "review_boundaries.jsonl")
        self.latest = _latest_decisions(review_dir)
        self.golden_spec = _read_json(golden_dir / "corrected_golden_spec.json")
        self.golden_validation = _read_json(audit_dir / "golden_validation.json")
        self.corpus_acceptance = _read_json(audit_dir / "corpus_acceptance.json")
        self.golden_checksums = _read_json(audit_dir / "golden_input_checksums.json")
        self.changed_classes = read_jsonl(audit_dir / "changed_line_classes.jsonl")
        self.changed_boundaries = read_jsonl(audit_dir / "changed_boundaries.jsonl")
        self.changed_blocks = read_jsonl(audit_dir / "changed_blocks.jsonl")
        self._golden_by_doc = {row["doc_id"]: row for row in self.golden_spec.get("documents", [])}
        self._golden_validation_by_doc = {
            row["document_id"]: row for row in self.golden_validation.get("documents", [])
        }
        self._corpus_by_doc = {
            row["document_id"]: row for row in self.corpus_acceptance.get("documents", [])
        }
        self._documents_by_id = {str(row["document_id"]): row for row in self.documents}
        self._changed_class_keys = {
            (str(row["document_id"]), int(row["line"])) for row in self.changed_classes
        }
        self._changed_boundary_keys = {
            (str(row["document_id"]), int(row["before_line"])) for row in self.changed_boundaries
        }
        self._changed_block_keys = {
            (str(row["document_id"]), _range_tuple(row.get("v6_range"))) for row in self.changed_blocks
        }
        self._golden_inputs_valid = self._golden_input_checksums_match()
        self._parser_profile_valid = (
            self.manifest.get("parser_profile") == self.golden_spec.get("target_parser_profile") == PARSER_VERSION
        )

    def document_status(self, document: dict[str, Any]) -> dict[str, Any]:
        doc_id = str(document["document_id"])
        line_rows = [row for row in self.lines if row["document_id"] == doc_id]
        boundary_rows = [row for row in self.boundaries if row["document_id"] == doc_id]
        block_ranges = block_ranges_for_lines(line_rows)
        line_statuses = [self.parser_status_for_line(row)["parser_validation_status"] for row in line_rows]
        boundary_statuses = [self.parser_status_for_boundary(row)["parser_validation_status"] for row in boundary_rows]
        block_statuses = [
            self.parser_status_for_block(document, line_range)["parser_validation_status"]
            for line_range in block_ranges
        ]
        parser_badge = _document_badge(line_statuses, boundary_statuses, block_statuses)
        manual_line_reviewed = sum(1 for row in line_rows if ("line", str(row["item_id"])) in self.latest)
        manual_boundary_reviewed = sum(1 for row in boundary_rows if ("boundary", str(row["item_id"])) in self.latest)
        result = {
            "parser_validation_status": parser_badge["status"],
            "parser_validation_label": parser_badge["label"],
            "parser_validation_reason": parser_badge["reason"],
            "parser_profile": self.manifest.get("parser_profile"),
            "golden_covered": doc_id in self._golden_by_doc,
            "golden_match": parser_badge["status"] == ParserValidationStatus.AUTO_VALIDATED_GOLDEN.value,
            "invariant_validated": parser_badge["status"] in {
                ParserValidationStatus.AUTO_VALIDATED_GOLDEN.value,
                ParserValidationStatus.PARSER_VALIDATED.value,
                ParserValidationStatus.PARSER_CHANGED_NEEDS_REVIEW.value,
            },
            "parser_line_validated": sum(1 for value in line_statuses if _validated_status(value)),
            "parser_line_total": len(line_rows),
            "parser_boundary_validated": sum(1 for value in boundary_statuses if _validated_status(value)),
            "parser_boundary_total": len(boundary_rows),
            "parser_block_validated": sum(1 for value in block_statuses if _validated_status(value)),
            "parser_block_total": len(block_statuses),
            "manual_line_reviewed": manual_line_reviewed,
            "manual_line_total": len(line_rows),
            "manual_boundary_reviewed": manual_boundary_reviewed,
            "manual_boundary_total": len(boundary_rows),
        }
        return result

    def parser_status_for_line(self, line: dict[str, Any]) -> dict[str, Any]:
        doc_id = str(line["document_id"])
        line_number = int(line["raw_line_number"])
        if not self._source_identity_ok(doc_id, line):
            return _parser_payload(
                ParserValidationStatus.PARSER_CONFLICT,
                "Snapshot source checksum differs from the parser-derived line item.",
                doc_id in self._golden_by_doc,
                False,
            )
        status, reason, golden_covered, golden_match = self._base_parser_status(
            doc_id=doc_id,
            changed_key=(doc_id, line_number),
            changed_keys=self._changed_class_keys,
            item_kind="line",
        )
        golden = self._golden_by_doc.get(doc_id)
        if golden and self._golden_prerequisites_ok(doc_id):
            expected = (golden.get("expected_line_classes") or {}).get(str(line_number))
            if expected is not None and expected != line.get("parser_proposed_line_class"):
                status = ParserValidationStatus.PARSER_CONFLICT
                reason = "Golden fixture expected a different parser v6 line class."
                golden_match = False
        return _parser_payload(status, reason, golden_covered, golden_match)

    def parser_status_for_boundary(self, boundary: dict[str, Any]) -> dict[str, Any]:
        doc_id = str(boundary["document_id"])
        line_number = int(boundary["previous_line_number"])
        if not self._source_identity_ok(doc_id, boundary):
            return _parser_payload(
                ParserValidationStatus.PARSER_CONFLICT,
                "Snapshot source checksum differs from the parser-derived boundary item.",
                doc_id in self._golden_by_doc,
                False,
            )
        status, reason, golden_covered, golden_match = self._base_parser_status(
            doc_id=doc_id,
            changed_key=(doc_id, line_number),
            changed_keys=self._changed_boundary_keys,
            item_kind="boundary",
        )
        golden = self._golden_by_doc.get(doc_id)
        if golden and self._golden_prerequisites_ok(doc_id):
            expected = (golden.get("expected_boundaries") or {}).get(str(line_number))
            actual = "SPLIT" if boundary.get("parser_proposed_boundary") else "MERGE"
            if expected is not None and expected != actual:
                status = ParserValidationStatus.PARSER_CONFLICT
                reason = "Golden fixture expected a different parser v6 boundary."
                golden_match = False
        return _parser_payload(status, reason, golden_covered, golden_match)

    def parser_status_for_block(self, document: dict[str, Any], line_range: list[int]) -> dict[str, Any]:
        doc_id = str(document["document_id"])
        status, reason, golden_covered, golden_match = self._base_parser_status(
            doc_id=doc_id,
            changed_key=(doc_id, _range_tuple(line_range)),
            changed_keys=self._changed_block_keys,
            item_kind="block",
        )
        golden = self._golden_by_doc.get(doc_id)
        if golden and self._golden_prerequisites_ok(doc_id):
            expected_ranges = golden.get("expected_block_ranges") or []
            if expected_ranges and line_range not in expected_ranges:
                status = ParserValidationStatus.PARSER_CONFLICT
                reason = "Golden fixture expected a different parser v6 block range."
                golden_match = False
        return _parser_payload(status, reason, golden_covered, golden_match)

    def manual_status_for_item(self, item_type: str, item: dict[str, Any]) -> dict[str, Any]:
        decision = self.latest.get((item_type, str(item.get("item_id"))))
        if decision is None:
            return _manual_payload(ManualReviewStatus.NOT_MANUALLY_REVIEWED, None)
        if (
            decision.get("parser_profile") != self.manifest.get("parser_profile")
            or decision.get("source_checksum") != item.get("source_checksum")
        ):
            return _manual_payload(ManualReviewStatus.MANUAL_DECISION_STALE, decision)
        if decision.get("decision_status") == "accepted":
            return _manual_payload(ManualReviewStatus.MANUALLY_ACCEPTED, decision)
        if decision.get("decision_status") == "overridden":
            return _manual_payload(ManualReviewStatus.MANUALLY_OVERRIDDEN, decision)
        return _manual_payload(ManualReviewStatus.MANUAL_CONFLICT, decision)

    def enrich_line(self, line: dict[str, Any]) -> dict[str, Any]:
        row = dict(line)
        row.update(self.parser_status_for_line(line))
        row.update(self.manual_status_for_item("line", line))
        return row

    def enrich_boundary(self, boundary: dict[str, Any]) -> dict[str, Any]:
        row = dict(boundary)
        row.update(self.parser_status_for_boundary(boundary))
        row.update(self.manual_status_for_item("boundary", boundary))
        return row

    def progress(self, manual_progress: dict[str, Any]) -> dict[str, Any]:
        line_statuses = [self.parser_status_for_line(row)["parser_validation_status"] for row in self.lines]
        boundary_statuses = [self.parser_status_for_boundary(row)["parser_validation_status"] for row in self.boundaries]
        block_statuses: list[str] = []
        for document in self.documents:
            doc_lines = [row for row in self.lines if row["document_id"] == document["document_id"]]
            block_statuses.extend(
                self.parser_status_for_block(document, line_range)["parser_validation_status"]
                for line_range in block_ranges_for_lines(doc_lines)
            )
        manual_counts = self._manual_counts()
        enriched_docs = []
        for document in manual_progress["documents"]:
            source = next(row for row in self.documents if row["document_id"] == document["document_id"])
            enriched_docs.append({**document, **self.document_status(source)})
        return {
            **manual_progress,
            "documents": enriched_docs,
            "parser_validation": {
                "line_total": len(line_statuses),
                "line_validated": sum(1 for value in line_statuses if _validated_status(value)),
                "boundary_total": len(boundary_statuses),
                "boundary_validated": sum(1 for value in boundary_statuses if _validated_status(value)),
                "block_total": len(block_statuses),
                "block_validated": sum(1 for value in block_statuses if _validated_status(value)),
                "golden_covered_items": sum(
                    1
                    for value in [*line_statuses, *boundary_statuses, *block_statuses]
                    if value == ParserValidationStatus.AUTO_VALIDATED_GOLDEN.value
                ),
                "invariant_covered_items": sum(
                    1
                    for value in [*line_statuses, *boundary_statuses, *block_statuses]
                    if value == ParserValidationStatus.PARSER_VALIDATED.value
                ),
                "review_recommended": sum(
                    1
                    for value in [*line_statuses, *boundary_statuses, *block_statuses]
                    if value == ParserValidationStatus.PARSER_CHANGED_NEEDS_REVIEW.value
                ),
                "conflicts": sum(
                    1
                    for value in [*line_statuses, *boundary_statuses, *block_statuses]
                    if value == ParserValidationStatus.PARSER_CONFLICT.value
                ),
                "unvalidated": sum(
                    1
                    for value in [*line_statuses, *boundary_statuses, *block_statuses]
                    if value == ParserValidationStatus.PARSER_UNVALIDATED.value
                ),
            },
            "manual_review": manual_counts,
        }

    def _base_parser_status(
        self,
        *,
        doc_id: str,
        changed_key: tuple[str, int] | tuple[str, tuple[int, int] | None],
        changed_keys: set[Any],
        item_kind: str,
    ) -> tuple[ParserValidationStatus, str, bool, bool]:
        if not self._parser_profile_valid:
            return ParserValidationStatus.PARSER_CONFLICT, "Parser profile is not the expected v6 profile.", False, False
        if doc_id in self._golden_by_doc:
            if self._golden_prerequisites_ok(doc_id):
                return (
                    ParserValidationStatus.AUTO_VALIDATED_GOLDEN,
                    f"Golden validation passed for this exact v6 {item_kind}.",
                    True,
                    True,
                )
            return ParserValidationStatus.PARSER_CONFLICT, "Golden validation prerequisites failed.", True, False
        corpus = self._corpus_by_doc.get(doc_id)
        if not corpus:
            return ParserValidationStatus.PARSER_UNVALIDATED, "No parser v6 audit information exists for this document.", False, False
        if not _corpus_doc_valid(corpus):
            return ParserValidationStatus.PARSER_CONFLICT, "Deterministic parser invariant validation failed.", False, False
        if changed_key in changed_keys:
            return (
                ParserValidationStatus.PARSER_CHANGED_NEEDS_REVIEW,
                "Parser v6 changed this item from v5; deterministic invariants pass but human review is recommended.",
                False,
                False,
            )
        return (
            ParserValidationStatus.PARSER_VALIDATED,
            "Validated by deterministic parser invariants; not manually reviewed.",
            False,
            False,
        )

    def _golden_prerequisites_ok(self, doc_id: str) -> bool:
        validation = self._golden_validation_by_doc.get(doc_id)
        return bool(
            self._golden_inputs_valid
            and validation
            and self.golden_validation.get("status") == "pass"
            and validation.get("lines_match")
            and validation.get("classes_passed")
            and validation.get("boundaries_passed")
            and validation.get("blocks_passed")
            and validation.get("conservation")
            and validation.get("duplication") == 0
            and validation.get("ordering") == 0
            and validation.get("citation_primary_count") == 0
        )

    def _source_identity_ok(self, doc_id: str, item: dict[str, Any]) -> bool:
        document = self._documents_by_id.get(doc_id)
        return bool(document and item.get("source_checksum") == document.get("source_checksum"))

    def _golden_input_checksums_match(self) -> bool:
        for row in self.golden_checksums.get("files", []):
            path = self.golden_dir / str(row["file"])
            if not path.exists() or path.stat().st_size != int(row["size"]):
                return False
            if sha256_file(path).upper() != str(row["sha256"]).upper():
                return False
        return True

    def _manual_counts(self) -> dict[str, int]:
        counts = {
            "reviewed_lines": 0,
            "reviewed_boundaries": 0,
            "accepted": 0,
            "overridden": 0,
            "stale": 0,
            "not_reviewed": 0,
            "conflicts": 0,
        }
        all_items = [("line", row) for row in self.lines] + [("boundary", row) for row in self.boundaries]
        for item_type, item in all_items:
            status = self.manual_status_for_item(item_type, item)["manual_review_status"]
            if item_type == "line" and status != ManualReviewStatus.NOT_MANUALLY_REVIEWED.value:
                counts["reviewed_lines"] += 1
            if item_type == "boundary" and status != ManualReviewStatus.NOT_MANUALLY_REVIEWED.value:
                counts["reviewed_boundaries"] += 1
            if status == ManualReviewStatus.MANUALLY_ACCEPTED.value:
                counts["accepted"] += 1
            elif status == ManualReviewStatus.MANUALLY_OVERRIDDEN.value:
                counts["overridden"] += 1
            elif status == ManualReviewStatus.MANUAL_DECISION_STALE.value:
                counts["stale"] += 1
            elif status == ManualReviewStatus.MANUAL_CONFLICT.value:
                counts["conflicts"] += 1
            else:
                counts["not_reviewed"] += 1
        return counts


def block_ranges_for_lines(lines: list[dict[str, Any]]) -> list[list[int]]:
    ranges: list[list[int]] = []
    current = object()
    for row in sorted(lines, key=lambda item: int(item["raw_line_number"])):
        block_index = row.get("parser_block_index")
        line_number = int(row["raw_line_number"])
        if block_index != current:
            ranges.append([line_number, line_number])
            current = block_index
        else:
            ranges[-1][1] = line_number
    return ranges


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_decisions(review_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for record in read_jsonl(review_dir / "manual_review_decisions.jsonl"):
        key = (str(record.get("item_type")), str(record.get("item_id")))
        if int(record.get("revision_number", 0)) >= int(latest.get(key, {}).get("revision_number", 0)):
            latest[key] = record
    return latest


def _corpus_doc_valid(row: dict[str, Any]) -> bool:
    return bool(
        row.get("conservation") is True
        and row.get("duplication_count") == 0
        and row.get("ordering_failures") == 0
        and row.get("primary_citation_count") == 0
    )


def _parser_payload(
    status: ParserValidationStatus,
    reason: str,
    golden_covered: bool,
    golden_match: bool,
) -> dict[str, Any]:
    return {
        "parser_validation_status": status.value,
        "parser_validation_label": PARSER_LABELS[status],
        "parser_validation_reason": reason,
        "parser_profile": PARSER_VERSION,
        "golden_covered": golden_covered,
        "golden_match": golden_match,
        "invariant_validated": status
        in {
            ParserValidationStatus.AUTO_VALIDATED_GOLDEN,
            ParserValidationStatus.PARSER_VALIDATED,
            ParserValidationStatus.PARSER_CHANGED_NEEDS_REVIEW,
        },
    }


def _manual_payload(status: ManualReviewStatus, decision: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "manual_review_status": status.value,
        "manual_review_label": MANUAL_LABELS[status],
        "manual_decision": decision.get("decision_status") if decision else None,
        "manual_comment": decision.get("reviewer_comment") if decision else None,
        "manual_revision": decision.get("revision_number") if decision else None,
    }


def _document_badge(line_statuses: list[str], boundary_statuses: list[str], block_statuses: list[str]) -> dict[str, str]:
    values = [*line_statuses, *boundary_statuses, *block_statuses]
    if not values:
        return {
            "status": ParserValidationStatus.PARSER_UNVALIDATED.value,
            "label": "PARSER NOT VALIDATED",
            "reason": "No parser-derived items were found.",
        }
    if any(value == ParserValidationStatus.PARSER_CONFLICT.value for value in values):
        return {
            "status": ParserValidationStatus.PARSER_CONFLICT.value,
            "label": "CONFLICT",
            "reason": "At least one parser-validation item is in conflict.",
        }
    if all(value == ParserValidationStatus.AUTO_VALIDATED_GOLDEN.value for value in values):
        return {
            "status": ParserValidationStatus.AUTO_VALIDATED_GOLDEN.value,
            "label": "GOLDEN PASS",
            "reason": "All document lines, boundaries, and blocks are covered by passing v6 golden validation.",
        }
    if any(value == ParserValidationStatus.PARSER_CHANGED_NEEDS_REVIEW.value for value in values):
        return {
            "status": ParserValidationStatus.PARSER_CHANGED_NEEDS_REVIEW.value,
            "label": "REVIEW RECOMMENDED",
            "reason": "Parser v6 changed at least one non-golden item from v5.",
        }
    if all(value == ParserValidationStatus.PARSER_VALIDATED.value for value in values):
        return {
            "status": ParserValidationStatus.PARSER_VALIDATED.value,
            "label": "PARSER VALIDATED",
            "reason": "All document items passed deterministic parser invariants.",
        }
    return {
        "status": ParserValidationStatus.PARSER_UNVALIDATED.value,
        "label": "PARSER NOT VALIDATED",
        "reason": "Some parser-derived items lack validation evidence.",
    }


def _range_tuple(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    return int(value[0]), int(value[1])


def _validated_status(value: str) -> bool:
    return value in {
        ParserValidationStatus.AUTO_VALIDATED_GOLDEN.value,
        ParserValidationStatus.PARSER_VALIDATED.value,
        ParserValidationStatus.PARSER_CHANGED_NEEDS_REVIEW.value,
    }
