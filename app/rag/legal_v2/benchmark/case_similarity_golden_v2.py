"""Case-similarity document-retrieval golden v2 (full-corpus compatible)."""

from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.rag.legal_v2.identity import is_valid_ecli, normalize_ecli

SCHEMA_VERSION = "nalus-case-similarity-golden.v2"
BENCHMARK_TYPE = "case_similarity_document_retrieval"
EXPECTED_QUERY_COUNT = 60
EXPECTED_DEV_COUNT = 40
EXPECTED_TEST_COUNT = 20
ALLOWED_SPLITS = frozenset({"dev", "test"})
ALLOWED_QUERY_TYPES = frozenset({"lexical_friendly", "semantic", "mixed"})
MIN_QUERY_WORDS = 40
MAX_QUERY_WORDS = 200

_ECLI_RE = re.compile(r"\bECLI:[A-Z]{2}:[A-Z]{2}:[0-9]{4}:[^\s,;]+", re.IGNORECASE)
_CASE_REF_RE = re.compile(
    r"(?:"
    r"\b(?:I{1,3}|IV|V{0,3}|VI{0,3}|IX|X{0,3})\.?\s*ÚS\s+\d+/\d+\b"
    r"|\bPl\.?\s*ÚS\s+\d+/\d+\b"
    r"|\bsp\.\s*zn\.\s*[^\s,]{3,}"
    r")",
    re.IGNORECASE,
)


class QueryType(str, Enum):
    LEXICAL_FRIENDLY = "lexical_friendly"
    SEMANTIC = "semantic"
    MIXED = "mixed"


class CaseSimilarityGoldenV2Item(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str
    schema_version: str = SCHEMA_VERSION
    benchmark_type: str = BENCHMARK_TYPE
    split: str
    language: str = "cs"
    query_text: str
    expected_primary_document_id: str
    expected_primary_ecli: str | None = None
    expected_court: str = "constitutional_court"
    expected_source: str = "constitutional"
    expected_year: int | None = None
    expected_relevant_document_ids: list[str] = Field(default_factory=list)
    relevance_notes: str
    query_type: str
    legal_area: str
    document_type: str | None = None
    case_reference: str | None = None
    target_selection_method: str = "corpus_stratified_v2"
    human_review_status: str = "PENDING_HUMAN_REVIEW"

    @field_validator("split")
    @classmethod
    def _validate_split(cls, value: str) -> str:
        if value not in ALLOWED_SPLITS:
            raise ValueError(f"split must be one of {sorted(ALLOWED_SPLITS)}")
        return value

    @field_validator("query_type")
    @classmethod
    def _validate_query_type(cls, value: str) -> str:
        if value not in ALLOWED_QUERY_TYPES:
            raise ValueError(f"query_type must be one of {sorted(ALLOWED_QUERY_TYPES)}")
        return value

    @field_validator("expected_primary_ecli")
    @classmethod
    def _validate_ecli(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_ecli(value)
        if not is_valid_ecli(normalized):
            raise ValueError(f"invalid ECLI: {value}")
        return normalized


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text.strip()))


def assign_frozen_splits(query_ids: list[str]) -> dict[str, str]:
    """Deterministic 40 dev / 20 test holdout before any retrieval tuning."""
    ranked = sorted(query_ids, key=lambda item: hashlib.sha256(item.encode("utf-8")).hexdigest())
    test_ids = set(ranked[:EXPECTED_TEST_COUNT])
    return {query_id: ("test" if query_id in test_ids else "dev") for query_id in query_ids}


def load_case_similarity_golden_v2_jsonl(path: Path | str) -> list[CaseSimilarityGoldenV2Item]:
    items: list[CaseSimilarityGoldenV2Item] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            items.append(CaseSimilarityGoldenV2Item.model_validate(payload))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return items


def write_case_similarity_golden_v2_jsonl(
    path: Path | str,
    items: Iterable[CaseSimilarityGoldenV2Item],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = [item.model_dump_json() for item in items]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def audit_query_leakage(query_text: str, *, target_ecli: str | None, case_reference: str | None) -> list[str]:
    issues: list[str] = []
    if _ECLI_RE.search(query_text):
        issues.append("ecli_leakage")
    if _CASE_REF_RE.search(query_text):
        issues.append("case_reference_leakage")
    if target_ecli and normalize_ecli(target_ecli) in query_text.upper().replace(" ", ""):
        issues.append("target_ecli_substring")
    if case_reference:
        folded_ref = re.sub(r"[^a-z0-9]+", "", case_reference.casefold())
        folded_query = re.sub(r"[^a-z0-9]+", "", query_text.casefold())
        if folded_ref and len(folded_ref) >= 6 and folded_ref in folded_query:
            issues.append("case_reference_folded_leakage")
    word_count = count_words(query_text)
    if word_count < MIN_QUERY_WORDS:
        issues.append("query_too_short")
    if word_count > MAX_QUERY_WORDS:
        issues.append("query_too_long")
    return issues


DEFAULT_V2_DATASET = (
    Path(__file__).resolve().parents[4]
    / "benchmarks"
    / "legal_v2"
    / "case_similarity_golden_v2_full_corpus.jsonl"
)

__all__ = [
    "SCHEMA_VERSION",
    "BENCHMARK_TYPE",
    "EXPECTED_QUERY_COUNT",
    "CaseSimilarityGoldenV2Item",
    "QueryType",
    "assign_frozen_splits",
    "audit_query_leakage",
    "load_case_similarity_golden_v2_jsonl",
    "write_case_similarity_golden_v2_jsonl",
    "DEFAULT_V2_DATASET",
]
