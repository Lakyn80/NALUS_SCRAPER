"""Retrieval-golden v1 models and deterministic validation (Step 4A pilot)."""

from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ALLOWED_SPLITS = frozenset({"development", "validation", "locked_holdout"})
ALLOWED_DIFFICULTIES = frozenset({"easy", "medium", "hard"})
ALLOWED_QUERY_TYPES = frozenset(
    {
        "legal_rule",
        "client_paraphrase",
        "fact_specific",
        "procedural",
        "court_reasoning",
        "operative_outcome",
        "concept_distinction",
        "corpus_negative",
    }
)

_WS_RE = re.compile(r"\s+")


def normalize_evidence_text(value: str) -> str:
    return _WS_RE.sub(" ", (value or "").strip())


def normalize_query_text(value: str) -> str:
    return normalize_evidence_text(value).casefold()


class RetrievalGoldenSplit(str, Enum):
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    LOCKED_HOLDOUT = "locked_holdout"


class InspectedNegativeCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    block_id: str
    rank: int = Field(ge=1)
    overlap_score: float | None = None
    rejection_reason: str


class RetrievalGoldenItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str
    query: str
    split: str
    is_negative: bool
    query_type: str
    difficulty: str
    court: str | None = None
    jurisdiction: str = "CZ"
    legal_area: str | None = None
    source_document_id: str | None = None
    expected_document_ids: list[str] = Field(default_factory=list)
    primary_expected_block_id: str | None = None
    expected_block_ids: list[str] = Field(default_factory=list)
    accepted_alternative_block_ids: list[str] = Field(default_factory=list)
    hard_negative_block_ids: list[str] = Field(default_factory=list)
    evidence_excerpt: str | None = None
    grounding_note: str | None = None
    negative_rationale: str | None = None
    inspected_negative_candidates: list[InspectedNegativeCandidate] = Field(default_factory=list)

    @field_validator("split")
    @classmethod
    def _validate_split(cls, value: str) -> str:
        if value not in ALLOWED_SPLITS:
            raise ValueError(f"split must be one of {sorted(ALLOWED_SPLITS)}")
        return value

    @field_validator("difficulty")
    @classmethod
    def _validate_difficulty(cls, value: str) -> str:
        if value not in ALLOWED_DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {sorted(ALLOWED_DIFFICULTIES)}")
        return value

    @field_validator("query_type")
    @classmethod
    def _validate_query_type(cls, value: str) -> str:
        if value not in ALLOWED_QUERY_TYPES:
            raise ValueError(f"query_type must be one of {sorted(ALLOWED_QUERY_TYPES)}")
        return value

    @field_validator("query_id", "query")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("must be non-empty")
        return text

    @model_validator(mode="after")
    def _cross_field_rules(self) -> "RetrievalGoldenItem":
        if self.is_negative:
            if self.expected_document_ids or self.expected_block_ids or self.primary_expected_block_id:
                raise ValueError("negative items must not declare expected documents/blocks")
            if self.accepted_alternative_block_ids or self.hard_negative_block_ids:
                raise ValueError("negative items must not declare alternative/hard-negative blocks")
            if self.evidence_excerpt:
                raise ValueError("negative items must not include evidence_excerpt")
            if not (self.negative_rationale or "").strip():
                raise ValueError("negative items require negative_rationale")
            if not self.inspected_negative_candidates:
                raise ValueError("negative items require inspected_negative_candidates")
            if self.query_type != "corpus_negative":
                raise ValueError("negative items must use query_type=corpus_negative")
            return self

        if not self.expected_document_ids:
            raise ValueError("positive items require expected_document_ids")
        if not self.expected_block_ids:
            raise ValueError("positive items require expected_block_ids")
        if not self.primary_expected_block_id:
            raise ValueError("positive items require primary_expected_block_id")
        if self.primary_expected_block_id not in self.expected_block_ids:
            raise ValueError("primary_expected_block_id must be listed in expected_block_ids")
        if not self.source_document_id:
            raise ValueError("positive items require source_document_id")
        if self.source_document_id not in self.expected_document_ids:
            raise ValueError("source_document_id must be in expected_document_ids")
        if not (self.evidence_excerpt or "").strip():
            raise ValueError("positive items require evidence_excerpt")
        if self.negative_rationale or self.inspected_negative_candidates:
            raise ValueError("positive items must not include negative fields")
        overlap = set(self.expected_block_ids) & set(self.hard_negative_block_ids)
        if overlap:
            raise ValueError(f"hard negatives overlap expected blocks: {sorted(overlap)}")
        alt_overlap = set(self.accepted_alternative_block_ids) & set(self.hard_negative_block_ids)
        if alt_overlap:
            raise ValueError(f"hard negatives overlap alternatives: {sorted(alt_overlap)}")
        return self


class RetrievalGoldenValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    query_id: str | None = None


class RetrievalGoldenValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_path: str
    ok: bool
    item_count: int
    positive_count: int
    negative_count: int
    issues: list[RetrievalGoldenValidationIssue] = Field(default_factory=list)

    @property
    def failure_count(self) -> int:
        return len(self.issues)


def load_retrieval_golden_jsonl(path: Path | str) -> list[RetrievalGoldenItem]:
    dataset_path = Path(path)
    items: list[RetrievalGoldenItem] = []
    for line_no, raw in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{dataset_path}:{line_no}: invalid JSON ({exc})") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{dataset_path}:{line_no}: expected object")
        items.append(RetrievalGoldenItem.model_validate(payload))
    return items


def validate_retrieval_golden_dataset(
    items: Sequence[RetrievalGoldenItem],
    *,
    blocks_by_id: Mapping[str, Any] | None = None,
    expected_total: int | None = 30,
    expected_positive: int | None = 29,
    expected_negative: int | None = 1,
    dataset_path: str = "",
) -> RetrievalGoldenValidationReport:
    issues: list[RetrievalGoldenValidationIssue] = []
    positives = [item for item in items if not item.is_negative]
    negatives = [item for item in items if item.is_negative]

    if expected_total is not None and len(items) != expected_total:
        issues.append(
            RetrievalGoldenValidationIssue(
                code="count_total",
                message=f"expected {expected_total} items, found {len(items)}",
            )
        )
    if expected_positive is not None and len(positives) != expected_positive:
        issues.append(
            RetrievalGoldenValidationIssue(
                code="count_positive",
                message=f"expected {expected_positive} positives, found {len(positives)}",
            )
        )
    if expected_negative is not None and len(negatives) != expected_negative:
        issues.append(
            RetrievalGoldenValidationIssue(
                code="count_negative",
                message=f"expected {expected_negative} negatives, found {len(negatives)}",
            )
        )

    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    for item in items:
        if item.query_id in seen_ids:
            issues.append(
                RetrievalGoldenValidationIssue(
                    code="duplicate_query_id",
                    message=f"duplicate query_id {item.query_id}",
                    query_id=item.query_id,
                )
            )
        seen_ids.add(item.query_id)
        norm_q = normalize_query_text(item.query)
        if norm_q in seen_queries:
            issues.append(
                RetrievalGoldenValidationIssue(
                    code="duplicate_query_text",
                    message=f"duplicate normalized query text for {item.query_id}",
                    query_id=item.query_id,
                )
            )
        seen_queries.add(norm_q)

    if blocks_by_id is not None:
        for item in positives:
            for document_id in item.expected_document_ids:
                if not any(
                    getattr(block, "document_id", None) == document_id
                    or (isinstance(block, Mapping) and block.get("document_id") == document_id)
                    for block in blocks_by_id.values()
                ):
                    # document existence is implied by its blocks; check via primary block
                    pass
            for block_id in (
                list(item.expected_block_ids)
                + list(item.accepted_alternative_block_ids)
                + list(item.hard_negative_block_ids)
            ):
                if block_id not in blocks_by_id:
                    issues.append(
                        RetrievalGoldenValidationIssue(
                            code="missing_block",
                            message=f"block_id not found in corpus: {block_id}",
                            query_id=item.query_id,
                        )
                    )
            primary = blocks_by_id.get(item.primary_expected_block_id) if item.primary_expected_block_id else None
            if primary is None:
                issues.append(
                    RetrievalGoldenValidationIssue(
                        code="missing_primary_block",
                        message=f"primary block missing: {item.primary_expected_block_id}",
                        query_id=item.query_id,
                    )
                )
            else:
                primary_doc = getattr(primary, "document_id", None) or (
                    primary.get("document_id") if isinstance(primary, Mapping) else None
                )
                if primary_doc not in item.expected_document_ids:
                    issues.append(
                        RetrievalGoldenValidationIssue(
                            code="primary_document_mismatch",
                            message="primary block document_id not in expected_document_ids",
                            query_id=item.query_id,
                        )
                    )
                raw_text = getattr(primary, "raw_text", None) or (
                    primary.get("raw_text") if isinstance(primary, Mapping) else ""
                )
                if not evidence_excerpt_in_block(item.evidence_excerpt or "", str(raw_text or "")):
                    issues.append(
                        RetrievalGoldenValidationIssue(
                            code="evidence_excerpt_mismatch",
                            message="evidence_excerpt is not a normalized substring of primary block raw_text",
                            query_id=item.query_id,
                        )
                    )
            for block_id in item.accepted_alternative_block_ids:
                if block_id not in blocks_by_id:
                    continue
            for block_id in item.hard_negative_block_ids:
                if block_id in item.expected_block_ids:
                    issues.append(
                        RetrievalGoldenValidationIssue(
                            code="hard_negative_collision",
                            message=f"hard negative also expected: {block_id}",
                            query_id=item.query_id,
                        )
                    )

        for item in negatives:
            for candidate in item.inspected_negative_candidates:
                if candidate.block_id not in blocks_by_id:
                    issues.append(
                        RetrievalGoldenValidationIssue(
                            code="missing_negative_candidate_block",
                            message=f"inspected candidate block missing: {candidate.block_id}",
                            query_id=item.query_id,
                        )
                    )

    return RetrievalGoldenValidationReport(
        dataset_path=dataset_path,
        ok=not issues,
        item_count=len(items),
        positive_count=len(positives),
        negative_count=len(negatives),
        issues=issues,
    )


def evidence_excerpt_in_block(excerpt: str, block_text: str) -> bool:
    excerpt_n = normalize_evidence_text(excerpt)
    block_n = normalize_evidence_text(block_text)
    if not excerpt_n:
        return False
    return excerpt_n in block_n


def write_jsonl(path: Path | str, items: Iterable[RetrievalGoldenItem]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [item.model_dump_json(exclude_none=False) for item in items]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


DEFAULT_PILOT_DATASET = (
    Path(__file__).resolve().parents[4]
    / "benchmarks"
    / "legal_v2"
    / "retrieval_golden_v1_pilot.jsonl"
)

__all__ = [
    "ALLOWED_SPLITS",
    "ALLOWED_DIFFICULTIES",
    "ALLOWED_QUERY_TYPES",
    "RetrievalGoldenSplit",
    "InspectedNegativeCandidate",
    "RetrievalGoldenItem",
    "RetrievalGoldenValidationIssue",
    "RetrievalGoldenValidationReport",
    "DEFAULT_PILOT_DATASET",
    "normalize_evidence_text",
    "normalize_query_text",
    "load_retrieval_golden_jsonl",
    "validate_retrieval_golden_dataset",
    "evidence_excerpt_in_block",
    "write_jsonl",
]
