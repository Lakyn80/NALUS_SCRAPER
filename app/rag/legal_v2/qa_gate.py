from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INITIAL_INDEX_QA_POLICY_VERSION = "legal_v2_initial_index_qa_v1"
LEGAL_V2_INITIAL_INDEX_QA_POLICY_VERSION = INITIAL_INDEX_QA_POLICY_VERSION
GATE_POLICY_VERSION = INITIAL_INDEX_QA_POLICY_VERSION

_GENERATOR_PLACEHOLDER_REASON = "Generated parser and chunk evidence requires human confirmation."


@dataclass(frozen=True)
class ParserQaGatePolicy:
    policy_version: str = INITIAL_INDEX_QA_POLICY_VERSION
    minimum_sample_count: int = 30
    required_review_coverage: float = 1.0
    required_approval_rate: float = 1.0
    max_rejected_count: int = 0
    max_needs_review_count: int = 0
    max_reconstruction_failures: int = 0
    max_boundary_violations: int = 0
    max_duplicate_ids: int = 0
    max_cross_document_mixing_count: int = 0
    max_unresolved_blocking_defects: int = 0


@dataclass(frozen=True)
class ParserQaGateDecision:
    sample_count: int
    reviewed_count: int
    approved_count: int
    rejected_count: int
    needs_review_count: int
    review_coverage: float
    approval_rate: float
    full_parse_audit_status: str
    reconstruction_failures: int
    boundary_violations: int
    duplicate_ids: int
    cross_document_mixing: int
    unresolved_blocking_defects: int
    source_incomplete_count: int
    duplicate_source_identifier_count: int
    reviewed_source_incomplete_count: int
    reviewed_duplicate_source_identifier_count: int
    gate_policy_version: str
    final_decision: str
    smoke_index_permitted: bool
    blocking_reasons: list[str] = field(default_factory=list)
    invalid_reasons: list[str] = field(default_factory=list)
    recommended_smoke_document_ids: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: _utc_now())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_parser_qa_gate(
    *,
    parser_quality: dict[str, Any],
    manual_review_summary: dict[str, Any],
    parse_audit: dict[str, Any],
    source_inventory: dict[str, Any],
    policy: ParserQaGatePolicy | None = None,
) -> ParserQaGateDecision:
    policy = policy or ParserQaGatePolicy()
    invalid: list[str] = []
    blocking: list[str] = []
    if not policy.policy_version:
        invalid.append("missing_policy_version")

    documents = parser_quality.get("documents")
    if not isinstance(documents, list):
        documents = []
        invalid.append("parser_quality_documents_missing_or_invalid")

    sample_count = len(documents)
    approved_count = 0
    rejected_count = 0
    needs_review_count = 0
    reviewed_count = 0
    cross_document_mixing = 0
    unresolved_defects = 0
    reviewed_source_incomplete_count = 0
    reviewed_duplicate_source_identifier_count = 0

    for index, item in enumerate(documents):
        if not isinstance(item, dict):
            invalid.append(f"sample_{index}_not_object")
            continue
        status = str(item.get("review_status") or "").strip()
        reason = str(item.get("review_reason") or "").strip()
        if not status:
            continue
        if status not in {"approved", "rejected", "needs_review"}:
            invalid.append(f"sample_{index}_invalid_review_status")
            continue
        if reason and reason != _GENERATOR_PLACEHOLDER_REASON:
            reviewed_count += 1
        approved_count += int(status == "approved")
        rejected_count += int(status == "rejected")
        needs_review_count += int(status == "needs_review")
        if item.get("no_cross_document_mixing") is not True:
            cross_document_mixing += 1
        defects = item.get("identified_defects")
        if defects:
            unresolved_defects += len(defects if isinstance(defects, list) else [defects])
        if not _has_any_field(item, ("beginning_preserved", "beginning_correctly_parsed")):
            invalid.append(f"sample_{index}_missing_beginning_check")
        if not _has_any_field(item, ("ending_preserved", "end_correctly_parsed")):
            invalid.append(f"sample_{index}_missing_ending_check")
        for field_name in ("legal_reasoning_preserved", "operative_part_preserved"):
            if field_name not in item:
                invalid.append(f"sample_{index}_missing_{field_name}")
        source_status = str(item.get("source_completeness_status") or "").strip()
        duplicate_status = str(item.get("duplicate_source_identifier_status") or "").strip()
        if not source_status:
            invalid.append(f"sample_{index}_missing_source_completeness_status")
        elif source_status != "complete_from_available_source":
            reviewed_source_incomplete_count += 1
        if not duplicate_status:
            invalid.append(f"sample_{index}_missing_duplicate_source_identifier_status")
        elif duplicate_status != "none":
            reviewed_duplicate_source_identifier_count += 1

    audit_summary = _summary(parse_audit)
    inventory_summary = source_inventory
    full_parse_audit_status = str(audit_summary.get("status") or "missing")
    reconstruction_failures = _int(audit_summary.get("reconstruction_failures"))
    boundary_violations = _int(audit_summary.get("boundary_violations"))
    duplicate_ids = _int(audit_summary.get("duplicate_ids"))
    source_incomplete_count = _required_int(
        inventory_summary,
        "documents_missing_complete_text",
        invalid,
    )
    duplicate_source_identifier_count = _required_int(
        inventory_summary,
        "duplicate_source_document_identifiers",
        invalid,
    )

    parser_summary = _summary(parser_quality)
    summary = manual_review_summary.get("summary") if "summary" in manual_review_summary else manual_review_summary
    if not isinstance(summary, dict):
        invalid.append("manual_review_summary_missing_or_invalid")
        summary = {}
    artifact_policy_version = str(
        summary.get("gate_policy_version") or parser_summary.get("gate_policy_version") or ""
    )
    if artifact_policy_version != policy.policy_version:
        invalid.append("missing_or_unknown_gate_policy_version")
    if _int(summary.get("approved")) not in {0, approved_count}:
        invalid.append("manual_review_approved_count_mismatch")
    if _int(summary.get("rejected")) not in {0, rejected_count}:
        invalid.append("manual_review_rejected_count_mismatch")
    if _int(summary.get("needs_review")) not in {0, needs_review_count}:
        invalid.append("manual_review_needs_review_count_mismatch")

    review_coverage = reviewed_count / sample_count if sample_count else 0.0
    approval_rate = approved_count / sample_count if sample_count else 0.0

    _block_if(blocking, sample_count < policy.minimum_sample_count, "minimum_sample_count_not_met")
    _block_if(blocking, review_coverage < policy.required_review_coverage, "manual_review_coverage_not_100_percent")
    _block_if(blocking, approval_rate < policy.required_approval_rate, "approval_rate_not_100_percent")
    _block_if(blocking, rejected_count > policy.max_rejected_count, "rejected_samples_present")
    _block_if(blocking, needs_review_count > policy.max_needs_review_count, "needs_review_samples_present")
    _block_if(blocking, full_parse_audit_status != "pass", "full_parse_audit_not_pass")
    _block_if(blocking, reconstruction_failures > policy.max_reconstruction_failures, "reconstruction_failures_present")
    _block_if(blocking, boundary_violations > policy.max_boundary_violations, "boundary_violations_present")
    _block_if(blocking, duplicate_ids > policy.max_duplicate_ids, "duplicate_paragraph_or_chunk_ids_present")
    _block_if(blocking, cross_document_mixing > policy.max_cross_document_mixing_count, "cross_document_mixing_present")
    _block_if(blocking, unresolved_defects > policy.max_unresolved_blocking_defects, "unresolved_blocking_defects_present")

    if invalid:
        final_decision = "invalid"
    elif blocking:
        final_decision = "blocked"
    else:
        final_decision = "pass"

    return ParserQaGateDecision(
        sample_count=sample_count,
        reviewed_count=reviewed_count,
        approved_count=approved_count,
        rejected_count=rejected_count,
        needs_review_count=needs_review_count,
        review_coverage=review_coverage,
        approval_rate=approval_rate,
        full_parse_audit_status=full_parse_audit_status,
        reconstruction_failures=reconstruction_failures,
        boundary_violations=boundary_violations,
        duplicate_ids=duplicate_ids,
        cross_document_mixing=cross_document_mixing,
        unresolved_blocking_defects=unresolved_defects,
        source_incomplete_count=source_incomplete_count,
        duplicate_source_identifier_count=duplicate_source_identifier_count,
        reviewed_source_incomplete_count=reviewed_source_incomplete_count,
        reviewed_duplicate_source_identifier_count=reviewed_duplicate_source_identifier_count,
        gate_policy_version=policy.policy_version,
        final_decision=final_decision,
        smoke_index_permitted=final_decision == "pass",
        blocking_reasons=blocking,
        invalid_reasons=invalid,
        recommended_smoke_document_ids=_recommended_smoke_document_ids(documents),
    )


def evaluate_initial_index_qa_gate(
    *,
    parser_quality_path: Path | None = None,
    parser_quality_gate_path: Path | None = None,
    manual_review_summary_path: Path,
    parse_audit_path: Path,
    source_inventory_path: Path,
    policy: ParserQaGatePolicy | None = None,
) -> ParserQaGateDecision:
    parser_quality_source = parser_quality_gate_path or parser_quality_path
    if parser_quality_source is None:
        raise TypeError("parser_quality_path or parser_quality_gate_path is required")
    try:
        return evaluate_parser_qa_gate(
            parser_quality=load_json(parser_quality_source),
            manual_review_summary=load_json(manual_review_summary_path),
            parse_audit=load_json(parse_audit_path),
            source_inventory=load_json(source_inventory_path),
            policy=policy,
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return ParserQaGateDecision(
            sample_count=0,
            reviewed_count=0,
            approved_count=0,
            rejected_count=0,
            needs_review_count=0,
            review_coverage=0.0,
            approval_rate=0.0,
            full_parse_audit_status="unknown",
            reconstruction_failures=0,
            boundary_violations=0,
            duplicate_ids=0,
            cross_document_mixing=0,
            unresolved_blocking_defects=0,
            source_incomplete_count=0,
            duplicate_source_identifier_count=0,
            reviewed_source_incomplete_count=0,
            reviewed_duplicate_source_identifier_count=0,
            gate_policy_version=(policy or ParserQaGatePolicy()).policy_version,
            final_decision="invalid",
            smoke_index_permitted=False,
            invalid_reasons=[f"malformed_artifact:{exc.__class__.__name__}"],
            recommended_smoke_document_ids=[],
        )


def write_gate_decision(decision: ParserQaGateDecision, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "gate_decision.json"
    markdown_path = output_dir / "gate_decision.md"
    payload = decision.to_dict()
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    (output_dir / "smoke_document_ids.txt").write_text(
        "\n".join(payload.get("recommended_smoke_document_ids") or []) + "\n",
        encoding="utf-8",
    )
    return json_path, markdown_path


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary", payload)
    return summary if isinstance(summary, dict) else {}


def _required_int(payload: dict[str, Any], key: str, invalid: list[str]) -> int:
    if key not in payload:
        invalid.append(f"source_inventory_missing_{key}")
        return 0
    return _int(payload.get(key))


def _has_any_field(payload: dict[str, Any], field_names: tuple[str, ...]) -> bool:
    return any(field_name in payload for field_name in field_names)


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _block_if(reasons: list[str], condition: bool, reason: str) -> None:
    if condition:
        reasons.append(reason)


def _recommended_smoke_document_ids(documents: list[Any], *, limit: int = 20) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()

    def add(item: dict[str, Any]) -> None:
        document_id = str(item.get("document_id") or "").strip()
        if (
            len(selected) < limit
            and document_id
            and document_id not in seen
            and item.get("review_status") == "approved"
            and str(item.get("source_completeness_status") or "complete_from_available_source")
            == "complete_from_available_source"
            and str(item.get("duplicate_source_identifier_status") or "none") == "none"
        ):
            selected.append(document_id)
            seen.add(document_id)

    for category in (
        "constitutional",
        "supreme",
        "short_judgment",
        "long_judgment",
        "numbered_paragraphs",
        "citations",
        "long_legal_reasoning",
        "damaged_formatting",
    ):
        for item in documents:
            if isinstance(item, dict) and category in set(item.get("categories") or []):
                before = len(selected)
                add(item)
                if len(selected) > before:
                    break
    for item in documents:
        if isinstance(item, dict):
            add(item)
    return selected


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Legal Retrieval v2 parser QA gate decision",
        "",
        f"- Policy: `{payload['gate_policy_version']}`",
        f"- Final decision: `{payload['final_decision']}`",
        f"- Smoke index permitted: `{payload['smoke_index_permitted']}`",
        f"- Samples: {payload['sample_count']}",
        f"- Reviewed: {payload['reviewed_count']}",
        f"- Approved: {payload['approved_count']}",
        f"- Rejected: {payload['rejected_count']}",
        f"- Needs review: {payload['needs_review_count']}",
        f"- Review coverage: {payload['review_coverage']:.4f}",
        f"- Approval rate: {payload['approval_rate']:.4f}",
        f"- Full parse audit: `{payload['full_parse_audit_status']}`",
        f"- Reconstruction failures: {payload['reconstruction_failures']}",
        f"- Boundary violations: {payload['boundary_violations']}",
        f"- Duplicate IDs: {payload['duplicate_ids']}",
        f"- Cross-document mixing: {payload['cross_document_mixing']}",
        f"- Unresolved blocking defects: {payload['unresolved_blocking_defects']}",
        f"- Source incomplete count: {payload['source_incomplete_count']}",
        f"- Duplicate source identifier count: {payload['duplicate_source_identifier_count']}",
        f"- Reviewed source incomplete count: {payload['reviewed_source_incomplete_count']}",
        f"- Reviewed duplicate source identifier count: {payload['reviewed_duplicate_source_identifier_count']}",
        "",
        "## Blocking reasons",
        "",
    ]
    if payload["blocking_reasons"]:
        lines.extend(f"- `{reason}`" for reason in payload["blocking_reasons"])
    else:
        lines.append("- None")
    lines.extend(["", "## Invalid reasons", ""])
    if payload["invalid_reasons"]:
        lines.extend(f"- `{reason}`" for reason in payload["invalid_reasons"])
    else:
        lines.append("- None")
    lines.extend(["", "## Recommended smoke document IDs", ""])
    if payload["recommended_smoke_document_ids"]:
        lines.extend(f"- `{document_id}`" for document_id in payload["recommended_smoke_document_ids"])
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
