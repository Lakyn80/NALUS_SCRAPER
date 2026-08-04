from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .models import digest_text

SAFE_LINE_CLASSES = {"layout_noise", "metadata", "heading", "instruction", "signature"}
FORMAL_TEXT_RE = re.compile(
    r"^(?:NALUS - databáze rozhodnutí Ústavního soudu|Česká republika|Ústavního soudu|USNESENÍ|NÁLEZ|Odůvodnění:|soudce zpravodaj)$",
    re.IGNORECASE,
)
CASE_DATE_RE = re.compile(r"^[IVX]+\.\s*ÚS\s+\d+/\d+\s+ze dne\s+\d{1,2}\.\s*\d{1,2}\.\s*\d{4}$", re.IGNORECASE)
PLACE_DATE_RE = re.compile(r"^V Brně dne\s+\d{1,2}\.\s*\d{1,2}\.\s*\d{4}$", re.IGNORECASE)
POUCENI_RE = re.compile(r"^Poučení:\s+.+", re.IGNORECASE)


@dataclass(frozen=True)
class Rule:
    rule_id: str
    court: str
    confidence: str
    rule_type: str
    item_type: str
    target_value: str
    source_document_ids: list[str]
    rationale: str
    pattern: dict[str, Any]
    conflicts: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "court": self.court,
            "confidence": self.confidence,
            "rule_type": self.rule_type,
            "item_type": self.item_type,
            "target_value": self.target_value,
            "source_document_ids": list(self.source_document_ids),
            "rationale": self.rationale,
            "pattern": dict(self.pattern),
            "conflicts": list(self.conflicts),
        }


def normalize_line(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def boundary_to_bool(value: str) -> bool:
    if value == "split":
        return True
    if value in {"merge", "preserve_parser"}:
        return False
    raise ValueError(f"Unsupported boundary decision for rule matching: {value}")


def line_signature(line: dict[str, Any], manual_class: str | None = None) -> str:
    normalized = normalize_line(str(line.get("raw_text") or ""))
    if FORMAL_TEXT_RE.match(normalized):
        return f"exact:{normalized.casefold()}"
    if CASE_DATE_RE.match(normalized):
        return "template:constitutional_case_date"
    if PLACE_DATE_RE.match(normalized):
        return "template:brno_place_date"
    if POUCENI_RE.match(normalized):
        return "template:pouceni_prefix"
    previous = str(line.get("previous_automated_annotation") or "")
    parser = str(line.get("parser_proposed_line_class") or "")
    if manual_class in SAFE_LINE_CLASSES and previous in {"decision_type", "court_identifier", "case_identifier", "section_heading", "signature_block"}:
        return f"structural:{previous}:{parser}"
    return ""


def line_rule_type_for_signature(signature: str) -> str:
    if signature.startswith("exact:"):
        return "exact_normalized_line"
    if signature.startswith("template:"):
        return "anchored_structural_template"
    return "anchored_structural_template"


def rule_id(*, court: str, item_type: str, rule_type: str, target_value: str, signature: str) -> str:
    return f"rule-{digest_text(court, item_type, rule_type, target_value, signature, length=16)}"


def matches_line_signature(line: dict[str, Any], signature: str) -> bool:
    return line_signature(line) == signature


def boundary_signature(before_line: dict[str, Any], after_line: dict[str, Any]) -> str:
    left = line_signature(before_line)
    right = line_signature(after_line)
    if left and right:
        return f"{left}=>{right}"
    left_prev = str(before_line.get("previous_automated_annotation") or "")
    right_prev = str(after_line.get("previous_automated_annotation") or "")
    if left_prev and right_prev and left_prev != "statute_reference" and right_prev != "statute_reference":
        return f"previous:{left_prev}=>{right_prev}"
    return ""
