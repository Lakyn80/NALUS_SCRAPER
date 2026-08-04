from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import DEFAULT_REVIEW_DIR, REVIEW_SCHEMA_VERSION, utc_now, write_json, write_jsonl
from .progress import latest_decisions
from .rules import (
    Rule,
    boundary_signature,
    boundary_to_bool,
    line_rule_type_for_signature,
    line_signature,
    matches_line_signature,
    rule_id,
)

ASSISTED_DIR_NAME = "assisted"


def build_assisted_review(*, review_dir: Path = DEFAULT_REVIEW_DIR, court: str | None = None, write_artifacts: bool = True) -> dict[str, Any]:
    manifest = _read_json(review_dir / "review_manifest.json")
    documents = _read_jsonl(review_dir / "review_documents.jsonl")
    lines = _read_jsonl(review_dir / "review_lines.jsonl")
    boundaries = _read_jsonl(review_dir / "review_boundaries.jsonl")
    latest = latest_decisions(review_dir)
    completed = _completed_documents(
        documents,
        lines,
        boundaries,
        latest,
        parser_profile=str(manifest.get("parser_profile") or ""),
        parser_head=str(manifest.get("head") or ""),
    )
    if court:
        completed = [doc for doc in completed if doc["court"] == court]
    rules = _derive_rules(completed, lines, boundaries, latest)
    suggestions = _match_rules(rules, documents, lines, boundaries, latest, court=court)
    batches = _batches(rules, suggestions)
    summary = _summary(documents, lines, boundaries, completed, rules, suggestions, batches, latest, court=court)
    if write_artifacts:
        assisted_dir = review_dir / ASSISTED_DIR_NAME
        assisted_dir.mkdir(parents=True, exist_ok=True)
        write_json(assisted_dir / "assisted_rules.json", {"schema_version": REVIEW_SCHEMA_VERSION, "rules": [rule.to_dict() for rule in rules]})
        write_jsonl(assisted_dir / "assisted_suggestions.jsonl", suggestions)
        write_jsonl(assisted_dir / "assisted_batches.jsonl", batches)
        write_json(assisted_dir / "assisted_summary.json", summary)
        _write_summary_md(assisted_dir / "assisted_summary.md", summary)
        (assisted_dir / "batch_application_log.jsonl").touch(exist_ok=True)
    return {"summary": summary, "rules": [rule.to_dict() for rule in rules], "suggestions": suggestions, "batches": batches}


def load_assisted_artifacts(review_dir: Path = DEFAULT_REVIEW_DIR) -> dict[str, Any]:
    assisted_dir = review_dir / ASSISTED_DIR_NAME
    if not (assisted_dir / "assisted_rules.json").exists():
        return build_assisted_review(review_dir=review_dir)
    return {
        "summary": json.loads((assisted_dir / "assisted_summary.json").read_text(encoding="utf-8")),
        "rules": json.loads((assisted_dir / "assisted_rules.json").read_text(encoding="utf-8"))["rules"],
        "suggestions": _read_jsonl(assisted_dir / "assisted_suggestions.jsonl"),
        "batches": _read_jsonl(assisted_dir / "assisted_batches.jsonl"),
    }


def occurrences_for_rule(review_dir: Path, rule_id_value: str) -> list[dict[str, Any]]:
    artifacts = load_assisted_artifacts(review_dir)
    return [item for item in artifacts["suggestions"] if item["rule_id"] == rule_id_value]


def _completed_documents(
    documents: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
    latest: dict[tuple[str, str], dict[str, Any]],
    *,
    parser_profile: str,
    parser_head: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for document in documents:
        doc_id = str(document["document_id"])
        doc_lines = [line for line in lines if line["document_id"] == doc_id]
        doc_boundaries = [boundary for boundary in boundaries if boundary["document_id"] == doc_id]
        line_decisions = [latest.get(("line", str(line["item_id"]))) for line in doc_lines]
        boundary_decisions = [latest.get(("boundary", str(boundary["item_id"]))) for boundary in doc_boundaries]
        decisions = [decision for decision in [*line_decisions, *boundary_decisions] if decision]
        complete = (
            len(line_decisions) == len(doc_lines)
            and len(boundary_decisions) == len(doc_boundaries)
            and all(line_decisions)
            and all(boundary_decisions)
            and all(decision.get("decision_status") != "unresolved" for decision in decisions)
            and all(decision.get("source_checksum") == document.get("source_checksum") for decision in decisions)
            and all(decision.get("parser_profile") == parser_profile for decision in decisions)
            and all(decision.get("parser_git_identity") == parser_head for decision in decisions)
        )
        if complete:
            result.append(document)
    return result


def _derive_rules(
    completed: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
    latest: dict[tuple[str, str], dict[str, Any]],
) -> list[Rule]:
    evidence: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    completed_ids = {doc["document_id"] for doc in completed}
    for line in lines:
        if line["document_id"] not in completed_ids:
            continue
        decision = latest.get(("line", str(line["item_id"])))
        manual_class = str(decision.get("manual_class") or "") if decision else ""
        signature = line_signature(line, manual_class)
        if not signature:
            continue
        key = (str(line["court"]), "line", manual_class, signature)
        row = evidence.setdefault(key, {"source_document_ids": set(), "conflicts": [], "examples": []})
        row["source_document_ids"].add(line["document_id"])
        row["examples"].append(line)
    by_doc_line = {(line["document_id"], int(line["raw_line_number"])): line for line in lines}
    for boundary in boundaries:
        if boundary["document_id"] not in completed_ids:
            continue
        decision = latest.get(("boundary", str(boundary["item_id"])))
        manual_boundary = str(decision.get("manual_boundary_decision") or "") if decision else ""
        if manual_boundary not in {"split", "merge", "preserve_parser"}:
            continue
        before = by_doc_line.get((boundary["document_id"], int(boundary["previous_line_number"])))
        after = by_doc_line.get((boundary["document_id"], int(boundary["next_line_number"])))
        if not before or not after:
            continue
        signature = boundary_signature(before, after)
        if not signature:
            continue
        effective_boundary = bool(boundary.get("parser_proposed_boundary")) if manual_boundary == "preserve_parser" else boundary_to_bool(manual_boundary)
        target = "split" if effective_boundary else "merge"
        key = (str(before["court"]), "boundary", target, signature)
        row = evidence.setdefault(key, {"source_document_ids": set(), "conflicts": [], "examples": []})
        row["source_document_ids"].add(boundary["document_id"])
        row["examples"].append(boundary)
    targets_by_signature: dict[tuple[str, str, str], set[str]] = {}
    for court, item_type, target, signature in evidence:
        targets_by_signature.setdefault((court, item_type, signature), set()).add(target)
    rules: list[Rule] = []
    for (court, item_type, target, signature), row in sorted(evidence.items()):
        rule_type = line_rule_type_for_signature(signature) if item_type == "line" else "boundary_context"
        conflicting_targets = sorted(targets_by_signature[(court, item_type, signature)] - {target})
        conflicts = list(row["conflicts"])
        if conflicting_targets:
            conflicts.append({"code": "conflicting_completed_manual_evidence", "conflicting_target_values": conflicting_targets})
        confidence = "SAFE" if not conflicts and row["source_document_ids"] else "BLOCKED"
        rules.append(
            Rule(
                rule_id=rule_id(court=court, item_type=item_type, rule_type=rule_type, target_value=target, signature=signature),
                court=court,
                confidence=confidence,
                rule_type=rule_type,
                item_type=item_type,
                target_value=target,
                source_document_ids=sorted(row["source_document_ids"]),
                rationale=_rationale(item_type, rule_type, target, signature),
                pattern={"signature": signature},
                conflicts=conflicts,
            )
        )
    return rules


def _match_rules(
    rules: list[Rule],
    documents: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
    latest: dict[tuple[str, str], dict[str, Any]],
    *,
    court: str | None,
) -> list[dict[str, Any]]:
    docs_by_id = {doc["document_id"]: doc for doc in documents}
    by_doc_line = {(line["document_id"], int(line["raw_line_number"])): line for line in lines}
    suggestions: list[dict[str, Any]] = []
    for rule in rules:
        if court and rule.court != court:
            continue
        if rule.item_type == "line":
            for line in lines:
                if line["court"] != rule.court or not matches_line_signature(line, str(rule.pattern["signature"])):
                    continue
                existing = latest.get(("line", str(line["item_id"])))
                suggestions.append(_line_occurrence(rule, line, docs_by_id[str(line["document_id"])], existing))
        else:
            for boundary in boundaries:
                doc = docs_by_id[str(boundary["document_id"])]
                if doc["court"] != rule.court:
                    continue
                before = by_doc_line.get((boundary["document_id"], int(boundary["previous_line_number"])))
                after = by_doc_line.get((boundary["document_id"], int(boundary["next_line_number"])))
                if not before or not after or boundary_signature(before, after) != rule.pattern["signature"]:
                    continue
                existing = latest.get(("boundary", str(boundary["item_id"])))
                suggestions.append(_boundary_occurrence(rule, boundary, before, after, doc, existing))
    return suggestions


def _line_occurrence(rule: Rule, line: dict[str, Any], document: dict[str, Any], existing: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "rule_id": rule.rule_id,
        "item_type": "line",
        "item_id": line["item_id"],
        "document_id": line["document_id"],
        "document_review_number": document["review_number"],
        "court": line["court"],
        "confidence": rule.confidence,
        "excluded": existing is not None or rule.confidence != "SAFE",
        "excluded_reason": "existing_manual_decision" if existing else ("not_safe" if rule.confidence != "SAFE" else None),
        "raw_text": line["raw_text"],
        "context": [],
        "parser_proposal": line.get("parser_proposed_line_class"),
        "previous_annotation": line.get("previous_automated_annotation"),
        "proposed_manual_class": rule.target_value,
        "proposed_boundary_decision": None,
        "suspicious_reasons": list(line.get("suspicious_reason_codes") or []),
        "existing_manual_decision": existing,
        "source_checksum": line.get("source_checksum"),
    }


def _boundary_occurrence(rule: Rule, boundary: dict[str, Any], before: dict[str, Any], after: dict[str, Any], document: dict[str, Any], existing: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "rule_id": rule.rule_id,
        "item_type": "boundary",
        "item_id": boundary["item_id"],
        "document_id": boundary["document_id"],
        "document_review_number": document["review_number"],
        "court": document["court"],
        "confidence": rule.confidence,
        "excluded": existing is not None or rule.confidence != "SAFE",
        "excluded_reason": "existing_manual_decision" if existing else ("not_safe" if rule.confidence != "SAFE" else None),
        "raw_text": f"L{before['raw_line_number']}: {before['raw_text']}\nL{after['raw_line_number']}: {after['raw_text']}",
        "context": [before, after],
        "parser_proposal": "split" if boundary.get("parser_proposed_boundary") else "merge",
        "previous_annotation": "split" if boundary.get("previous_automated_boundary_annotation") else "merge",
        "proposed_manual_class": None,
        "proposed_boundary_decision": rule.target_value,
        "suspicious_reasons": list(boundary.get("suspicious_reason_codes") or []),
        "existing_manual_decision": existing,
        "source_checksum": boundary.get("source_checksum"),
    }


def _batches(rules: list[Rule], suggestions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rule in rules:
        occurrences = [item for item in suggestions if item["rule_id"] == rule.rule_id and not item["excluded"]]
        excluded = [item for item in suggestions if item["rule_id"] == rule.rule_id and item["excluded"]]
        rows.append(
            {
                "schema_version": REVIEW_SCHEMA_VERSION,
                "batch_id": f"batch-{rule.rule_id.removeprefix('rule-')}",
                "rule_id": rule.rule_id,
                "confidence": rule.confidence,
                "apply_allowed": rule.confidence == "SAFE" and bool(occurrences),
                "occurrence_count": len(occurrences),
                "excluded_count": len(excluded),
                "confirmation": f"APPLY {rule.rule_id} {len(occurrences)}",
            }
        )
    return rows


def _summary(
    documents: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
    completed: list[dict[str, Any]],
    rules: list[Rule],
    suggestions: list[dict[str, Any]],
    batches: list[dict[str, Any]],
    latest: dict[tuple[str, str], dict[str, Any]],
    *,
    court: str | None,
) -> dict[str, Any]:
    by_confidence: dict[str, int] = {}
    by_court: dict[str, int] = {}
    for rule in rules:
        by_confidence[rule.confidence] = by_confidence.get(rule.confidence, 0) + 1
        by_court[rule.court] = by_court.get(rule.court, 0) + 1
    completed_courts = {doc["court"] for doc in completed}
    gated = sorted({doc["court"] for doc in documents if doc["court"] not in completed_courts})
    total_pending = _pending_item_count(documents, lines, boundaries, latest, court=court)
    safe_batch_reduction = sum(1 for item in suggestions if item["confidence"] == "SAFE" and not item["excluded"])
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "completed_evidence_documents": [
            {"document_id": doc["document_id"], "review_number": doc["review_number"], "court": doc["court"]}
            for doc in completed
        ],
        "rules_total": len(rules),
        "rules_by_confidence": by_confidence,
        "rules_by_court": by_court,
        "safe_rules": by_confidence.get("SAFE", 0),
        "review_rules": by_confidence.get("REVIEW", 0),
        "blocked_rules": by_confidence.get("BLOCKED", 0),
        "matching_pending_line_items": sum(1 for item in suggestions if item["item_type"] == "line" and not item["excluded"]),
        "matching_pending_boundaries": sum(1 for item in suggestions if item["item_type"] == "boundary" and not item["excluded"]),
        "constitutional_court_matches": sum(1 for item in suggestions if item["court"] == "constitutional_court" and not item["excluded"]),
        "high_court_gated": gated,
        "applicable_batch_count": sum(1 for batch in batches if batch["apply_allowed"]),
        "estimated_safe_batch_reduction": safe_batch_reduction,
        "estimated_remaining_manual_items_after_safe_batches": max(0, total_pending - safe_batch_reduction),
    }


def _rationale(item_type: str, rule_type: str, target: str, signature: str) -> str:
    if item_type == "line":
        return f"{rule_type} from completed manual evidence maps {signature} to {target}."
    return f"Boundary context from completed manual evidence maps {signature} to {target.upper()}."


def _write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Assisted Parser Review Summary",
        "",
        f"- Completed evidence documents: `{len(summary['completed_evidence_documents'])}`",
        f"- Rules total: `{summary['rules_total']}`",
        f"- SAFE rules: `{summary['safe_rules']}`",
        f"- REVIEW rules: `{summary['review_rules']}`",
        f"- BLOCKED rules: `{summary['blocked_rules']}`",
        f"- Matching pending line items: `{summary['matching_pending_line_items']}`",
        f"- Matching pending boundaries: `{summary['matching_pending_boundaries']}`",
        f"- Estimated safe batch reduction: `{summary['estimated_safe_batch_reduction']}`",
        f"- Estimated remaining manual items after safe batches: `{summary['estimated_remaining_manual_items_after_safe_batches']}`",
        f"- High court gated: `{', '.join(summary['high_court_gated']) or 'none'}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pending_item_count(
    documents: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
    latest: dict[tuple[str, str], dict[str, Any]],
    *,
    court: str | None,
) -> int:
    docs_by_id = {str(doc["document_id"]): doc for doc in documents}
    line_count = sum(
        1
        for line in lines
        if (court is None or line["court"] == court) and ("line", str(line["item_id"])) not in latest
    )
    boundary_count = sum(
        1
        for boundary in boundaries
        if (court is None or docs_by_id[str(boundary["document_id"])]["court"] == court)
        and ("boundary", str(boundary["item_id"])) not in latest
    )
    return line_count + boundary_count


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
