"""Domain models for long-input preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class InputClassification(str, Enum):
    SHORT_QUERY = "short_query"
    LONG_LEGAL_INPUT = "long_legal_input"
    OVERSIZED_INPUT = "oversized_input"
    EMPTY = "empty"


class CondensationMethod(str, Enum):
    NONE = "none"
    PASSTHROUGH = "passthrough"
    EXTRACTIVE = "extractive"
    PRECISE = "precise"


@dataclass(frozen=True)
class ScoredSentence:
    text: str
    score: float
    segment_index: int
    sentence_index: int
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchBrief:
    original_length: int
    normalized_length: int
    brief_text: str
    method: CondensationMethod
    was_condensed: bool
    policy_version: str
    brief_signature: str

    facts: tuple[str, ...] = ()
    legal_issues: tuple[str, ...] = ()
    procedural_signals: tuple[str, ...] = ()
    requested_focus: tuple[str, ...] = ()
    negative_focus: tuple[str, ...] = ()

    source_segments_used: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    suppressed_identifier_count: int = 0
    segments_examined: int = 0
    segments_selected: int = 0
    condensation_latency_ms: float = 0.0

    def diagnostics(self, *, include_brief_text: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "original_length": self.original_length,
            "normalized_length": self.normalized_length,
            "brief_length": len(self.brief_text),
            "was_condensed": self.was_condensed,
            "method": self.method.value,
            "policy_version": self.policy_version,
            "brief_signature": self.brief_signature,
            "segments_examined": self.segments_examined,
            "segments_selected": self.segments_selected,
            "suppressed_identifier_count": self.suppressed_identifier_count,
            "condensation_latency_ms": self.condensation_latency_ms,
            "facts_count": len(self.facts),
            "legal_issues_count": len(self.legal_issues),
            "procedural_signals_count": len(self.procedural_signals),
            "requested_focus_count": len(self.requested_focus),
            "negative_focus_count": len(self.negative_focus),
            "warnings": list(self.warnings),
        }
        if include_brief_text:
            payload["condensed_query"] = self.brief_text
            payload["facts"] = list(self.facts)
            payload["legal_issues"] = list(self.legal_issues)
            payload["procedural_signals"] = list(self.procedural_signals)
            payload["requested_focus"] = list(self.requested_focus)
            payload["negative_focus"] = list(self.negative_focus)
            payload["source_segments_used"] = list(self.source_segments_used)
        return payload


@dataclass(frozen=True)
class PreparedQuery:
    original_query: str
    retrieval_query: str
    classification: InputClassification
    was_condensed: bool
    condensation_method: CondensationMethod
    brief: SearchBrief | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def input_processing_diagnostics(self, *, include_brief_text: bool = False) -> dict[str, Any]:
        payload = {
            "classification": self.classification.value,
            "was_condensed": self.was_condensed,
            "method": self.condensation_method.value,
            "original_length": len(self.original_query),
            "retrieval_query_length": len(self.retrieval_query),
            **dict(self.diagnostics),
        }
        if self.brief is not None:
            payload.update(self.brief.diagnostics(include_brief_text=include_brief_text))
        return payload
