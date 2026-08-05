"""Case-similarity document-retrieval golden v1 (document-level pilot)."""

from __future__ import annotations

import json
import re
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.rag.legal_v2.benchmark.retrieval_golden import (
    evidence_excerpt_in_block,
    normalize_evidence_text,
    normalize_query_text,
)
from app.rag.legal_v2.identity import (
    IDENTITY_STATUS_BLOCKED_MISSING_ECLI,
    IDENTITY_STATUS_VERIFIED,
    ecli_key,
    eclis_equal,
    is_valid_ecli,
    normalize_ecli,
    validate_decision_identity,
)

SCHEMA_VERSION = "nalus-case-similarity-golden.v1"
BENCHMARK_TYPE = "case_similarity_document_retrieval"
HUMAN_REVIEW_STATUS = "PENDING_HUMAN_REVIEW"
EXPECTED_PILOT_COUNT = 20
MIN_SUPPORTING_BLOCKS = 2
MAX_SUPPORTING_BLOCKS = 5
MIN_HARD_NEGATIVES = 1
MAX_HARD_NEGATIVES = 3
MAX_ACCEPTED_ALTERNATIVES = 2
MIN_QUERY_WORDS = 60
MAX_QUERY_WORDS = 180
MIN_QUERY_SENTENCES = 3
MAX_QUERY_SENTENCES = 8
COPIED_SENTENCE_MIN_TOKENS = 12
HARD_NEGATIVE_BLOCKER_INSUFFICIENT_SAME_DOMAIN_CORPUS = "insufficient_same_domain_corpus"
ALLOWED_HARD_NEGATIVE_BLOCKERS = frozenset(
    {
        HARD_NEGATIVE_BLOCKER_INSUFFICIENT_SAME_DOMAIN_CORPUS,
    }
)

ALLOWED_SPLITS = frozenset({"development"})
ALLOWED_DIFFICULTIES = frozenset({"easy", "medium", "hard"})
ALLOWED_QUERY_STYLES = frozenset(
    {
        "client_narrative",
        "noisy_client_narrative",
        "multi_issue_client_narrative",
        "concise_case_description",
    }
)
EXPECTED_QUERY_STYLE_COUNTS = {
    "client_narrative": 8,
    "noisy_client_narrative": 4,
    "multi_issue_client_narrative": 4,
    "concise_case_description": 4,
}

_WS_RE = re.compile(r"\s+")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")
_TOKEN_RE = re.compile(r"[0-9A-Za-zÁ-Žá-ž§]+", re.UNICODE)
_DOC_ID_RE = re.compile(r"\bdoc-[0-9a-f]{8,}\b", re.IGNORECASE)
_BLOCK_ID_RE = re.compile(r"\bdoc-[0-9a-f]+:p:\d{5}:[0-9a-f]+\b", re.IGNORECASE)
_FILENAME_RE = re.compile(r"\braw_numbered\.txt\b|\breview_lines\.jsonl\b", re.IGNORECASE)
_ANON_LEAK_RE = re.compile(r"\bjméno\s+příjmení\b|\bprávnická\s+osoba\b", re.IGNORECASE)
_CASE_REF_RE = re.compile(
    r"(?:"
    r"\b(?:I{1,3}|IV|V{0,3}|VI{0,3}|IX|X{0,3})\.?\s*ÚS\s+\d+/\d+\b"
    r"|\b\d+\s+Cmo\s+\d+/\d+(?:-\d+)?\b"
    r"|\b\d+\s+Co\s+\d+/\d+(?:-\d+)?\b"
    r"|\b\d+\s+TO\s+\d+/\d+(?:-\d+)?\b"
    r"|\bsp\.\s*zn\.\s*[^\s,]{3,}"
    r")",
    re.IGNORECASE,
)


class CaseSimilarityQueryStyle(str, Enum):
    CLIENT_NARRATIVE = "client_narrative"
    NOISY_CLIENT_NARRATIVE = "noisy_client_narrative"
    MULTI_ISSUE_CLIENT_NARRATIVE = "multi_issue_client_narrative"
    CONCISE_CASE_DESCRIPTION = "concise_case_description"


class AnswerEvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_id: str
    excerpt: str

    @field_validator("block_id", "excerpt")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("must be non-empty")
        return text


class HardNegativeRationale(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    looks_similar_because: str
    materially_incorrect_because: str
    ecli: str | None = None
    canonical_document_id: str | None = None
    identity_status: str | None = None

    @field_validator("document_id", "looks_similar_because", "materially_incorrect_because")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("must be non-empty")
        return text


class AlternativeRationale(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    rationale: str
    ecli: str | None = None
    canonical_document_id: str | None = None
    identity_status: str | None = None

    @field_validator("document_id", "rationale")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("must be non-empty")
        return text


class CaseSimilarityProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    builder: str
    corpus_role: str = "reviewed_pool"
    review_number: int = Field(ge=1)
    source_case_number: str | None = None
    source_court: str | None = None
    notes: str | None = None


class CaseSimilarityGoldenItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark_id: str
    schema_version: str = SCHEMA_VERSION
    benchmark_type: str = BENCHMARK_TYPE
    split: str = "development"
    language: str = "cs"
    query: str
    query_style: str
    difficulty: str
    source_document_id: str
    expected_document_ids: list[str]
    expected_primary_ecli: str | None = None
    expected_primary_canonical_document_id: str | None = None
    primary_identity_status: str = "verified"
    accepted_alternative_document_ids: list[str] = Field(default_factory=list)
    hard_negative_document_ids: list[str]
    supporting_block_ids: list[str]
    answer_evidence: list[AnswerEvidenceItem]
    factual_facets: list[str] = Field(default_factory=list)
    legal_issue_facets: list[str] = Field(default_factory=list)
    procedural_facets: list[str] = Field(default_factory=list)
    similarity_rationale: str
    hard_negative_rationales: list[HardNegativeRationale]
    accepted_alternative_rationales: list[AlternativeRationale] = Field(default_factory=list)
    hard_negative_evaluable: bool = True
    hard_negative_blocker: str | None = None
    provenance: CaseSimilarityProvenance
    human_review_status: str = HUMAN_REVIEW_STATUS
    notes: str | None = None

    @field_validator(
        "benchmark_id",
        "query",
        "query_style",
        "difficulty",
        "source_document_id",
        "similarity_rationale",
        "schema_version",
        "benchmark_type",
        "split",
        "language",
        "human_review_status",
    )
    @classmethod
    def _non_empty_str(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("must be non-empty")
        return text

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

    @field_validator("query_style")
    @classmethod
    def _validate_query_style(cls, value: str) -> str:
        if value not in ALLOWED_QUERY_STYLES:
            raise ValueError(f"query_style must be one of {sorted(ALLOWED_QUERY_STYLES)}")
        return value

    @model_validator(mode="after")
    def _cross_field_rules(self) -> "CaseSimilarityGoldenItem":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        if self.benchmark_type != BENCHMARK_TYPE:
            raise ValueError(f"benchmark_type must be {BENCHMARK_TYPE}")
        if self.language != "cs":
            raise ValueError("language must be cs")
        if self.human_review_status != HUMAN_REVIEW_STATUS:
            raise ValueError(f"human_review_status must be {HUMAN_REVIEW_STATUS}")
        if len(self.expected_document_ids) != 1:
            raise ValueError("expected_document_ids must contain exactly one document")
        if self.expected_document_ids[0] != self.source_document_id:
            raise ValueError("source_document_id must equal the sole expected document")
        if not (MIN_SUPPORTING_BLOCKS <= len(self.supporting_block_ids) <= MAX_SUPPORTING_BLOCKS):
            raise ValueError(
                f"supporting_block_ids must contain {MIN_SUPPORTING_BLOCKS}-{MAX_SUPPORTING_BLOCKS} ids"
            )
        if len(set(self.supporting_block_ids)) != len(self.supporting_block_ids):
            raise ValueError("supporting_block_ids must be unique")
        if not (MIN_HARD_NEGATIVES <= len(self.hard_negative_document_ids) <= MAX_HARD_NEGATIVES):
            raise ValueError(
                f"hard_negative_document_ids must contain {MIN_HARD_NEGATIVES}-{MAX_HARD_NEGATIVES} ids"
            )
        if len(set(self.hard_negative_document_ids)) != len(self.hard_negative_document_ids):
            raise ValueError("hard_negative_document_ids must be unique")
        if len(self.accepted_alternative_document_ids) > MAX_ACCEPTED_ALTERNATIVES:
            raise ValueError(
                f"accepted_alternative_document_ids must contain at most {MAX_ACCEPTED_ALTERNATIVES}"
            )
        if len(set(self.accepted_alternative_document_ids)) != len(
            self.accepted_alternative_document_ids
        ):
            raise ValueError("accepted_alternative_document_ids must be unique")

        expected = set(self.expected_document_ids)
        alts = set(self.accepted_alternative_document_ids)
        hards = set(self.hard_negative_document_ids)
        if expected & alts or expected & hards or alts & hards:
            raise ValueError("expected, alternative and hard-negative document sets must not overlap")

        evidence_blocks = {item.block_id for item in self.answer_evidence}
        if evidence_blocks != set(self.supporting_block_ids):
            raise ValueError("answer_evidence block_ids must exactly match supporting_block_ids")

        rationale_docs = {item.document_id for item in self.hard_negative_rationales}
        if rationale_docs != hards:
            raise ValueError("hard_negative_rationales must cover exactly hard_negative_document_ids")

        alt_rationale_docs = {item.document_id for item in self.accepted_alternative_rationales}
        if alt_rationale_docs != alts:
            raise ValueError(
                "accepted_alternative_rationales must cover exactly accepted_alternative_document_ids"
            )

        words = count_words(self.query)
        if not (MIN_QUERY_WORDS <= words <= MAX_QUERY_WORDS):
            raise ValueError(
                f"query word count must be {MIN_QUERY_WORDS}-{MAX_QUERY_WORDS}, found {words}"
            )
        sentences = count_sentences(self.query)
        if not (MIN_QUERY_SENTENCES <= sentences <= MAX_QUERY_SENTENCES):
            raise ValueError(
                f"query sentence count must be {MIN_QUERY_SENTENCES}-{MAX_QUERY_SENTENCES}, "
                f"found {sentences}"
            )

        if self.hard_negative_evaluable and self.hard_negative_blocker is not None:
            raise ValueError(
                "hard_negative_blocker must be null when hard_negative_evaluable is true"
            )
        if not self.hard_negative_evaluable:
            if not self.hard_negative_blocker:
                raise ValueError(
                    "hard_negative_blocker is required when hard_negative_evaluable is false"
                )
            if self.hard_negative_blocker not in ALLOWED_HARD_NEGATIVE_BLOCKERS:
                raise ValueError(
                    "hard_negative_blocker must be one of "
                    f"{sorted(ALLOWED_HARD_NEGATIVE_BLOCKERS)}"
                )

        if self.primary_identity_status == IDENTITY_STATUS_VERIFIED:
            validate_decision_identity(
                ecli=self.expected_primary_ecli,
                canonical_document_id=self.expected_primary_canonical_document_id,
            )
        elif self.primary_identity_status == IDENTITY_STATUS_BLOCKED_MISSING_ECLI:
            if self.expected_primary_ecli is not None or self.expected_primary_canonical_document_id is not None:
                raise ValueError(
                    "blocked primary identity must leave expected_primary_ecli and "
                    "expected_primary_canonical_document_id null"
                )
        else:
            raise ValueError(
                f"unsupported primary_identity_status {self.primary_identity_status!r}"
            )

        for row in self.hard_negative_rationales:
            _validate_reference_identity(
                source_document_id=row.document_id,
                ecli=row.ecli,
                canonical_document_id=row.canonical_document_id,
                identity_status=row.identity_status,
                label="hard_negative",
            )
        for row in self.accepted_alternative_rationales:
            _validate_reference_identity(
                source_document_id=row.document_id,
                ecli=row.ecli,
                canonical_document_id=row.canonical_document_id,
                identity_status=row.identity_status,
                label="accepted_alternative",
            )

        primary_ecli = normalize_ecli(self.expected_primary_ecli) if self.expected_primary_ecli else None
        if primary_ecli:
            for row in [*self.hard_negative_rationales, *self.accepted_alternative_rationales]:
                if row.ecli and eclis_equal(row.ecli, primary_ecli):
                    raise ValueError(
                        "hard-negative/alternative ECLI must not equal primary ECLI"
                    )
            alt_eclis = [
                normalize_ecli(row.ecli)
                for row in self.accepted_alternative_rationales
                if row.ecli
            ]
            for row in self.hard_negative_rationales:
                if row.ecli and any(eclis_equal(row.ecli, alt) for alt in alt_eclis):
                    raise ValueError(
                        "hard-negative ECLI must not equal an accepted-alternative ECLI"
                    )
        return self


def _validate_reference_identity(
    *,
    source_document_id: str,
    ecli: str | None,
    canonical_document_id: str | None,
    identity_status: str | None,
    label: str,
) -> None:
    status = identity_status or IDENTITY_STATUS_VERIFIED
    if status == IDENTITY_STATUS_VERIFIED:
        try:
            validate_decision_identity(ecli=ecli, canonical_document_id=canonical_document_id)
        except ValueError as exc:
            raise ValueError(f"{label} {source_document_id}: {exc}") from exc
    elif status == IDENTITY_STATUS_BLOCKED_MISSING_ECLI:
        if ecli is not None or canonical_document_id is not None:
            raise ValueError(
                f"{label} {source_document_id}: blocked identity must have null ecli/canonical"
            )
    else:
        raise ValueError(f"{label} {source_document_id}: unsupported identity_status {status!r}")


class CaseSimilarityValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    benchmark_id: str | None = None
    severity: str = "error"


class CaseSimilarityValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_path: str
    ok: bool
    item_count: int
    issues: list[CaseSimilarityValidationIssue] = Field(default_factory=list)
    warnings: list[CaseSimilarityValidationIssue] = Field(default_factory=list)

    @property
    def failure_count(self) -> int:
        return len(self.issues)


def count_words(text: str) -> int:
    return len([token for token in _TOKEN_RE.findall(text or "") if token])


def count_sentences(text: str) -> int:
    parts = [part.strip() for part in _SENTENCE_RE.findall(text or "") if part.strip()]
    return len(parts)


def tokenize_normalized(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(normalize_evidence_text(text))]


def longest_verbatim_sentence_overlap_tokens(query: str, block_text: str) -> int:
    """Longest complete query sentence (≥12 tokens) that appears verbatim in block_text.

    Used for the hard leakage failure rule only. Shorter overlaps are reported by
    ``longest_contiguous_normalized_token_overlap``.
    """
    block_n = normalize_query_text(block_text)
    best = 0
    for sentence in _SENTENCE_RE.findall(query or ""):
        tokens = tokenize_normalized(sentence)
        if len(tokens) < COPIED_SENTENCE_MIN_TOKENS:
            continue
        sentence_n = normalize_query_text(sentence)
        if sentence_n and sentence_n in block_n:
            best = max(best, len(tokens))
    return best


def longest_contiguous_normalized_token_overlap(
    query: str,
    block_text: str,
) -> tuple[int, str]:
    """Longest contiguous normalized query-token run also contiguous in block_text."""
    query_tokens = tokenize_normalized(query)
    block_tokens = tokenize_normalized(block_text)
    if not query_tokens or not block_tokens:
        return 0, ""
    best_len = 0
    best_span: list[str] = []
    block_len = len(block_tokens)
    for i, start_token in enumerate(query_tokens):
        if len(query_tokens) - i <= best_len:
            break
        for j, block_token in enumerate(block_tokens):
            if block_token != start_token:
                continue
            k = 0
            while (
                i + k < len(query_tokens)
                and j + k < block_len
                and query_tokens[i + k] == block_tokens[j + k]
            ):
                k += 1
            if k > best_len:
                best_len = k
                best_span = query_tokens[i : i + k]
    return best_len, " ".join(best_span)


def best_supporting_block_token_overlap(
    query: str,
    blocks: Mapping[str, Any] | Sequence[Any],
    supporting_block_ids: Sequence[str],
) -> tuple[int, str, str | None]:
    """Return (token_count, overlap_text, supporting_block_id) for the best overlap."""
    best_count = 0
    best_text = ""
    best_block_id: str | None = None
    for block_id in supporting_block_ids:
        if isinstance(blocks, Mapping):
            block = blocks.get(block_id)
        else:
            block = next(
                (item for item in blocks if getattr(item, "block_id", None) == block_id),
                None,
            )
        if block is None:
            continue
        raw_text = getattr(block, "raw_text", None) or (
            block.get("raw_text") if isinstance(block, Mapping) else ""
        )
        count, text = longest_contiguous_normalized_token_overlap(query, str(raw_text or ""))
        if count > best_count:
            best_count = count
            best_text = text
            best_block_id = block_id
    return best_count, best_text, best_block_id


def detect_query_leakage(
    query: str,
    *,
    document_ids: Sequence[str] | None = None,
    case_numbers: Sequence[str | None] | None = None,
    source_ids: Sequence[str | None] | None = None,
) -> list[str]:
    leaks: list[str] = []
    if _DOC_ID_RE.search(query):
        leaks.append("document_id_pattern")
    if _BLOCK_ID_RE.search(query):
        leaks.append("block_id_pattern")
    if _FILENAME_RE.search(query):
        leaks.append("source_filename")
    if _ANON_LEAK_RE.search(query):
        leaks.append("anonymization_placeholder")
    if _CASE_REF_RE.search(query):
        leaks.append("case_reference_pattern")
    for document_id in document_ids or []:
        if document_id and document_id in query:
            leaks.append(f"literal_document_id:{document_id}")
    for case_number in case_numbers or []:
        if case_number and case_number in query:
            leaks.append(f"literal_case_number:{case_number}")
    for source_id in source_ids or []:
        if source_id and source_id in query:
            leaks.append(f"literal_source_id:{source_id}")
    return leaks


def load_case_similarity_golden_jsonl(path: Path | str) -> list[CaseSimilarityGoldenItem]:
    dataset_path = Path(path)
    items: list[CaseSimilarityGoldenItem] = []
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
        items.append(CaseSimilarityGoldenItem.model_validate(payload))
    return items


def write_case_similarity_jsonl(path: Path | str, items: Iterable[CaseSimilarityGoldenItem]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [item.model_dump_json(exclude_none=False) for item in items]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def validate_case_similarity_dataset(
    items: Sequence[CaseSimilarityGoldenItem],
    *,
    corpus_documents: Sequence[Any] | None = None,
    blocks_by_id: Mapping[str, Any] | None = None,
    expected_document_ids: Sequence[str] | None = None,
    dataset_path: str = "",
    rebuild_bytes: bytes | None = None,
    tracked_bytes: bytes | None = None,
) -> CaseSimilarityValidationReport:
    issues: list[CaseSimilarityValidationIssue] = []
    warnings: list[CaseSimilarityValidationIssue] = []

    def err(code: str, message: str, benchmark_id: str | None = None) -> None:
        issues.append(
            CaseSimilarityValidationIssue(
                code=code,
                message=message,
                benchmark_id=benchmark_id,
                severity="error",
            )
        )

    def warn(code: str, message: str, benchmark_id: str | None = None) -> None:
        warnings.append(
            CaseSimilarityValidationIssue(
                code=code,
                message=message,
                benchmark_id=benchmark_id,
                severity="warning",
            )
        )

    if len(items) != EXPECTED_PILOT_COUNT:
        err("count_total", f"expected {EXPECTED_PILOT_COUNT} items, found {len(items)}")

    seen_ids: set[str] = set()
    source_docs: list[str] = []
    style_counts: Counter[str] = Counter()

    corpus_doc_ids: set[str] = set()
    case_by_doc: dict[str, str | None] = {}
    source_id_by_doc: dict[str, str | None] = {}
    court_by_doc: dict[str, str | None] = {}
    if corpus_documents is not None:
        for ref in corpus_documents:
            document_id = getattr(ref, "document_id", None)
            if document_id:
                corpus_doc_ids.add(str(document_id))
                case_by_doc[str(document_id)] = getattr(ref, "case_number", None)
                source_id_by_doc[str(document_id)] = getattr(ref, "source_id", None)
                court_by_doc[str(document_id)] = getattr(ref, "court", None)

    expected_pool = set(expected_document_ids or corpus_doc_ids)

    for item in items:
        if item.benchmark_id in seen_ids:
            err("duplicate_benchmark_id", f"duplicate benchmark_id {item.benchmark_id}", item.benchmark_id)
        seen_ids.add(item.benchmark_id)
        source_docs.append(item.source_document_id)
        style_counts[item.query_style] += 1

        if item.split != "development":
            err("split", "split must be development", item.benchmark_id)
        if item.benchmark_type != BENCHMARK_TYPE:
            err("benchmark_type", f"benchmark_type must be {BENCHMARK_TYPE}", item.benchmark_id)
        if item.human_review_status != HUMAN_REVIEW_STATUS:
            err(
                "human_review_status",
                f"human_review_status must be {HUMAN_REVIEW_STATUS}",
                item.benchmark_id,
            )

        if len(item.expected_document_ids) != 1 or item.expected_document_ids[0] != item.source_document_id:
            err(
                "expected_source_mismatch",
                "source_document_id must equal the sole expected document",
                item.benchmark_id,
            )

        all_docs = (
            list(item.expected_document_ids)
            + list(item.accepted_alternative_document_ids)
            + list(item.hard_negative_document_ids)
        )
        for document_id in all_docs:
            if corpus_doc_ids and document_id not in corpus_doc_ids:
                err("missing_document", f"document not in corpus: {document_id}", item.benchmark_id)

        if not (MIN_SUPPORTING_BLOCKS <= len(item.supporting_block_ids) <= MAX_SUPPORTING_BLOCKS):
            err(
                "supporting_block_count",
                f"supporting blocks must be {MIN_SUPPORTING_BLOCKS}-{MAX_SUPPORTING_BLOCKS}",
                item.benchmark_id,
            )
        if not item.hard_negative_document_ids:
            err("missing_hard_negative", "hard-negative documents required", item.benchmark_id)
        if len(item.hard_negative_document_ids) > MAX_HARD_NEGATIVES:
            err("too_many_hard_negatives", "more than three hard negatives", item.benchmark_id)

        expected_set = set(item.expected_document_ids)
        alt_set = set(item.accepted_alternative_document_ids)
        hard_set = set(item.hard_negative_document_ids)
        if expected_set & alt_set or expected_set & hard_set or alt_set & hard_set:
            err("document_set_overlap", "expected/alternative/hard-negative overlap", item.benchmark_id)

        if item.hard_negative_evaluable and item.hard_negative_blocker is not None:
            err(
                "hard_negative_state_conflict",
                "hard_negative_evaluable=true cannot set a blocker",
                item.benchmark_id,
            )
        if not item.hard_negative_evaluable:
            if not item.hard_negative_blocker:
                err(
                    "hard_negative_blocker_missing",
                    "hard_negative_evaluable=false requires hard_negative_blocker",
                    item.benchmark_id,
                )
            elif item.hard_negative_blocker not in ALLOWED_HARD_NEGATIVE_BLOCKERS:
                err(
                    "hard_negative_blocker_invalid",
                    f"unsupported hard_negative_blocker {item.hard_negative_blocker}",
                    item.benchmark_id,
                )

        rationale_docs = {row.document_id for row in item.hard_negative_rationales}
        if rationale_docs != hard_set:
            err(
                "hard_negative_rationale_missing",
                "hard-negative rationales must cover hard_negative_document_ids",
                item.benchmark_id,
            )

        words = count_words(item.query)
        if not (MIN_QUERY_WORDS <= words <= MAX_QUERY_WORDS):
            err(
                "query_word_count",
                f"query word count out of range: {words}",
                item.benchmark_id,
            )
        sentences = count_sentences(item.query)
        if not (MIN_QUERY_SENTENCES <= sentences <= MAX_QUERY_SENTENCES):
            err(
                "query_sentence_count",
                f"query sentence count out of range: {sentences}",
                item.benchmark_id,
            )

        leaks = detect_query_leakage(
            item.query,
            document_ids=all_docs,
            case_numbers=[case_by_doc.get(doc) for doc in all_docs],
            source_ids=[source_id_by_doc.get(doc) for doc in all_docs],
        )
        for leak in leaks:
            err("query_leakage", f"query leakage: {leak}", item.benchmark_id)

        if re.search(r"\bco rozhodl soud\b|\bjak soud rozhodl\b|\bv tomto rozsudku\b", item.query, re.I):
            warn("exam_question_tone", "query may sound like a legal exam question", item.benchmark_id)
        if len(re.findall(r"§\s*\d+", item.query)) >= 3:
            warn("excessive_statutes", "query contains many statute numbers", item.benchmark_id)

        if blocks_by_id is not None:
            caption_only = True
            for block_id in item.supporting_block_ids:
                block = blocks_by_id.get(block_id)
                if block is None:
                    err("missing_block", f"supporting block missing: {block_id}", item.benchmark_id)
                    continue
                block_doc = getattr(block, "document_id", None) or (
                    block.get("document_id") if isinstance(block, Mapping) else None
                )
                if block_doc != item.source_document_id:
                    err(
                        "supporting_block_wrong_document",
                        f"supporting block {block_id} belongs to {block_doc}",
                        item.benchmark_id,
                    )
                primary_class = getattr(block, "primary_class", None) or (
                    block.get("primary_class") if isinstance(block, Mapping) else None
                )
                raw_text = getattr(block, "raw_text", None) or (
                    block.get("raw_text") if isinstance(block, Mapping) else ""
                )
                if primary_class not in {"header", "instruction", "other"} and len(
                    normalize_evidence_text(str(raw_text or ""))
                ) >= 80:
                    caption_only = False
                overlap = longest_verbatim_sentence_overlap_tokens(item.query, str(raw_text or ""))
                if overlap >= COPIED_SENTENCE_MIN_TOKENS:
                    err(
                        "copied_sentence",
                        f"query sentence of {overlap} tokens appears verbatim in supporting block",
                        item.benchmark_id,
                    )
            if caption_only and item.supporting_block_ids:
                err(
                    "supporting_blocks_caption_only",
                    "supporting blocks must not consist only of captions/headings/instructions",
                    item.benchmark_id,
                )

            for evidence in item.answer_evidence:
                block = blocks_by_id.get(evidence.block_id)
                if block is None:
                    continue
                raw_text = getattr(block, "raw_text", None) or (
                    block.get("raw_text") if isinstance(block, Mapping) else ""
                )
                if not evidence_excerpt_in_block(evidence.excerpt, str(raw_text or "")):
                    err(
                        "evidence_excerpt_mismatch",
                        f"answer_evidence excerpt not in block {evidence.block_id}",
                        item.benchmark_id,
                    )

        primary_court = court_by_doc.get(item.source_document_id)
        for hard_id in item.hard_negative_document_ids:
            hard_court = court_by_doc.get(hard_id)
            if primary_court and hard_court and primary_court != hard_court:
                # not automatically wrong, just warn when fields differ wildly
                if {primary_court, hard_court} == {"constitutional_court", "high_court_olomouc"} and (
                    "criminal" in (getattr(item, "procedural_facets", []) or [])
                ):
                    pass
            if primary_court and hard_court:
                civilish = {"constitutional_court", "high_court_prague", "high_court_olomouc"}
                if primary_court in civilish and hard_court in civilish:
                    continue

        if item.accepted_alternative_document_ids and not item.accepted_alternative_rationales:
            warn(
                "weak_alternative_rationale",
                "accepted alternatives present without strong rationale objects",
                item.benchmark_id,
            )

    if len(source_docs) != len(set(source_docs)):
        dupes = [doc for doc, count in Counter(source_docs).items() if count > 1]
        err("source_document_reuse", f"source documents used more than once: {dupes}")

    if expected_pool:
        missing = sorted(expected_pool - set(source_docs))
        unexpected = sorted(set(source_docs) - expected_pool)
        if missing:
            err("missing_reviewed_document", f"reviewed documents missing as sources: {missing}")
        if unexpected:
            err("unexpected_source_document", f"unexpected source documents: {unexpected}")
        if expected_pool and len(expected_pool) != EXPECTED_PILOT_COUNT:
            err(
                "reviewed_pool_size",
                f"expected reviewed pool size {EXPECTED_PILOT_COUNT}, found {len(expected_pool)}",
            )

    for style, expected_count in EXPECTED_QUERY_STYLE_COUNTS.items():
        if style_counts.get(style, 0) != expected_count:
            err(
                "query_style_distribution",
                f"query_style {style}: expected {expected_count}, found {style_counts.get(style, 0)}",
            )

    # Similar query pairs warning
    norms = [(item.benchmark_id, set(tokenize_normalized(item.query))) for item in items]
    for idx, (left_id, left_tokens) in enumerate(norms):
        if not left_tokens:
            continue
        for right_id, right_tokens in norms[idx + 1 :]:
            if not right_tokens:
                continue
            overlap = len(left_tokens & right_tokens) / float(len(left_tokens | right_tokens))
            if overlap >= 0.72:
                warn(
                    "similar_query_pair",
                    f"high query Jaccard overlap {overlap:.2f} between {left_id} and {right_id}",
                    left_id,
                )

    if rebuild_bytes is not None and tracked_bytes is not None and rebuild_bytes != tracked_bytes:
        err("deterministic_rebuild", "builder output is not byte-identical to tracked dataset")

    # Identity-map / ECLI dataset rules
    try:
        from app.rag.legal_v2.benchmark.case_similarity_identity import (
            load_case_similarity_identity_map,
        )

        identity_map = load_case_similarity_identity_map()
    except Exception as exc:  # noqa: BLE001
        err("identity_map_load", f"failed to load identity map: {exc}")
        identity_map = {}

    if identity_map:
        source_to_ecli: dict[str, str] = {}
        ecli_to_sources: dict[str, list[str]] = {}
        for source_id, row in identity_map.items():
            status = row.get("identity_status")
            ecli = row.get("ecli")
            if status == IDENTITY_STATUS_VERIFIED and ecli:
                key = ecli_key(ecli)
                source_to_ecli[source_id] = normalize_ecli(ecli)
                ecli_to_sources.setdefault(key, []).append(source_id)
            elif status == IDENTITY_STATUS_BLOCKED_MISSING_ECLI and ecli is not None:
                err(
                    "blocked_identity_has_ecli",
                    f"blocked identity row {source_id} must not carry an ecli",
                )
        for ecli_norm, sources in ecli_to_sources.items():
            if len(sources) > 1:
                err(
                    "duplicate_ecli_mapping",
                    f"ECLI {ecli_norm} mapped to multiple sources: {sources}",
                )

        for item in items:
            if item.source_document_id not in identity_map:
                err(
                    "primary_identity_unmapped",
                    f"primary {item.source_document_id} missing from identity map",
                    item.benchmark_id,
                )
            mapped = identity_map.get(item.source_document_id) or {}
            if item.primary_identity_status == IDENTITY_STATUS_VERIFIED:
                if not item.expected_primary_ecli:
                    err(
                        "missing_verified_ecli",
                        "verified primary missing expected_primary_ecli",
                        item.benchmark_id,
                    )
                elif ecli_key(item.expected_primary_ecli) != ecli_key(mapped.get("ecli")):
                    err(
                        "primary_ecli_map_mismatch",
                        "expected_primary_ecli does not match identity map",
                        item.benchmark_id,
                    )
                if item.source_document_id.startswith("doc-") and not item.expected_primary_ecli:
                    err(
                        "doc_id_used_as_canonical",
                        "row uses only doc-* without verified ECLI for evaluation",
                        item.benchmark_id,
                    )
            for row in item.hard_negative_rationales + item.accepted_alternative_rationales:
                if row.document_id not in identity_map:
                    err(
                        "reference_identity_unmapped",
                        f"{row.document_id} missing from identity map",
                        item.benchmark_id,
                    )
                    continue
                mapped_ref = identity_map[row.document_id]
                if row.identity_status == IDENTITY_STATUS_VERIFIED:
                    if ecli_key(row.ecli) != ecli_key(mapped_ref.get("ecli")):
                        err(
                            "reference_ecli_map_mismatch",
                            f"{row.document_id} ecli does not match identity map",
                            item.benchmark_id,
                        )

    return CaseSimilarityValidationReport(
        dataset_path=dataset_path,
        ok=not issues,
        item_count=len(items),
        issues=issues,
        warnings=warnings,
    )


DEFAULT_PILOT_DATASET = (
    Path(__file__).resolve().parents[4]
    / "benchmarks"
    / "legal_v2"
    / "case_similarity_golden_v1_pilot.jsonl"
)

__all__ = [
    "SCHEMA_VERSION",
    "BENCHMARK_TYPE",
    "HUMAN_REVIEW_STATUS",
    "EXPECTED_PILOT_COUNT",
    "EXPECTED_QUERY_STYLE_COUNTS",
    "CaseSimilarityQueryStyle",
    "AnswerEvidenceItem",
    "HardNegativeRationale",
    "AlternativeRationale",
    "CaseSimilarityProvenance",
    "CaseSimilarityGoldenItem",
    "CaseSimilarityValidationIssue",
    "CaseSimilarityValidationReport",
    "DEFAULT_PILOT_DATASET",
    "count_words",
    "count_sentences",
    "tokenize_normalized",
    "longest_verbatim_sentence_overlap_tokens",
    "longest_contiguous_normalized_token_overlap",
    "best_supporting_block_token_overlap",
    "detect_query_leakage",
    "load_case_similarity_golden_jsonl",
    "write_case_similarity_jsonl",
    "validate_case_similarity_dataset",
    "normalize_evidence_text",
    "normalize_query_text",
    "evidence_excerpt_in_block",
]
