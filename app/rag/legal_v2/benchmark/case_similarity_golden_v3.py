"""Case-similarity graded multi-relevance golden v3."""

from __future__ import annotations

import json
from enum import IntEnum
from pathlib import Path
from typing import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.rag.legal_v2.benchmark.case_similarity_golden_v2 import (
    EXPECTED_DEV_COUNT,
    EXPECTED_QUERY_COUNT,
    EXPECTED_TEST_COUNT,
    audit_query_leakage,
    count_words,
)
from app.rag.legal_v2.identity import is_valid_ecli, normalize_ecli

SCHEMA_VERSION = "nalus-case-similarity-golden.v3"
BENCHMARK_TYPE = "case_similarity_graded_retrieval"
BENCHMARK_SCOPE = "current_full_A_constitutional_court_only"
MIN_QUERY_WORDS = 20
MAX_QUERY_WORDS = 180
BINARY_RELEVANCE_THRESHOLD = 2


class RelevanceGrade(IntEnum):
    NOT_RELEVANT = 0
    PARTIALLY_RELEVANT = 1
    RELEVANT = 2
    HIGHLY_RELEVANT = 3


GRADE_LABELS: dict[int, str] = {
    0: "NOT_RELEVANT",
    1: "PARTIALLY_RELEVANT",
    2: "RELEVANT",
    3: "HIGHLY_RELEVANT",
}


class RelevanceJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    ecli: str | None = None
    grade: int
    label: str
    review_reason: str = ""
    review_status: str = "pending"

    @field_validator("grade")
    @classmethod
    def _validate_grade(cls, value: int) -> int:
        if value not in {0, 1, 2, 3}:
            raise ValueError("grade must be 0, 1, 2, or 3")
        return value

    @field_validator("label")
    @classmethod
    def _validate_label(cls, value: str, info) -> str:
        grade = info.data.get("grade")
        expected = GRADE_LABELS.get(grade, "")
        if expected and value != expected:
            return expected
        return value


class CaseSimilarityGoldenV3Item(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str
    schema_version: str = SCHEMA_VERSION
    benchmark_type: str = BENCHMARK_TYPE
    benchmark_scope: str = BENCHMARK_SCOPE
    split: str
    language: str = "cs"
    query_text: str
    query_type: str
    legal_area: str
    legacy_primary_document_id: str
    legacy_primary_ecli: str | None = None
    expected_court: str = "constitutional_court"
    expected_source: str = "constitutional"
    expected_year: int | None = None
    document_type: str | None = None
    case_reference: str | None = None
    query_review_status: str = "pending"
    query_review_notes: str = ""
    legacy_v2_query_text: str = ""
    relevance_judgments: list[RelevanceJudgment] = Field(default_factory=list)

    @field_validator("legacy_primary_ecli")
    @classmethod
    def _validate_ecli(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_ecli(value)
        if not is_valid_ecli(normalized):
            raise ValueError(f"invalid ECLI: {value}")
        return normalized


def load_case_similarity_golden_v3_jsonl(path: Path | str) -> list[CaseSimilarityGoldenV3Item]:
    items: list[CaseSimilarityGoldenV3Item] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            items.append(CaseSimilarityGoldenV3Item.model_validate(payload))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return items


def write_case_similarity_golden_v3_jsonl(
    path: Path | str,
    items: Iterable[CaseSimilarityGoldenV3Item],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = [item.model_dump_json() for item in items]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def validate_v3_split_counts(items: Sequence[CaseSimilarityGoldenV3Item]) -> None:
    if len(items) != EXPECTED_QUERY_COUNT:
        raise ValueError(f"expected {EXPECTED_QUERY_COUNT} queries, got {len(items)}")
    dev = sum(1 for item in items if item.split == "dev")
    test = sum(1 for item in items if item.split == "test")
    if dev != EXPECTED_DEV_COUNT or test != EXPECTED_TEST_COUNT:
        raise ValueError(f"expected {EXPECTED_DEV_COUNT} dev / {EXPECTED_TEST_COUNT} test, got {dev}/{test}")


DEFAULT_V3_DATASET = (
    Path(__file__).resolve().parents[4]
    / "benchmarks"
    / "legal_v2"
    / "case_similarity_golden_v3_graded.jsonl"
)

__all__ = [
    "SCHEMA_VERSION",
    "BENCHMARK_TYPE",
    "BENCHMARK_SCOPE",
    "BINARY_RELEVANCE_THRESHOLD",
    "RelevanceGrade",
    "GRADE_LABELS",
    "RelevanceJudgment",
    "CaseSimilarityGoldenV3Item",
    "load_case_similarity_golden_v3_jsonl",
    "write_case_similarity_golden_v3_jsonl",
    "validate_v3_split_counts",
    "audit_query_leakage",
    "count_words",
    "DEFAULT_V3_DATASET",
]
