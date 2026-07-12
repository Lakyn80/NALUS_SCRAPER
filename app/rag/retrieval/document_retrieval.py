"""Document-level aggregation for additive legal retrieval.

This module intentionally works on already retrieved candidate chunks. It does
not call Qdrant, BM25, embeddings, rerankers, or LLMs. The caller owns candidate
retrieval and can keep the existing chunk-level path unchanged.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.core.logging import get_logger
from app.core.tracing import trace_event
from app.rag.retrieval.errors import RetrievalConfigurationError
from app.rag.retrieval.models import RetrievedChunk

logger = get_logger(__name__)

DOCUMENT_SCORING_BEST_PLUS_AVERAGE = "best_plus_average_top_chunks"
DOCUMENT_SCORING_STRATEGIES = frozenset({DOCUMENT_SCORING_BEST_PLUS_AVERAGE})

_MAX_CANDIDATE_CHUNKS_UPPER_BOUND = 10_000
_MAX_RETURNED_DOCUMENTS_UPPER_BOUND = 1_000
_MAX_SUPPORTING_CHUNKS_UPPER_BOUND = 20
_SAFE_DOCUMENT_METADATA_KEYS = frozenset(
    {
        "authority_level",
        "case_number",
        "case_reference",
        "court",
        "court_name",
        "date",
        "decision_date",
        "document_id",
        "ecli",
        "legal_area",
        "reference",
        "source",
        "source_document_id",
        "title",
        "url",
    }
)


@dataclass(frozen=True)
class DocumentRetrievalConfig:
    enabled: bool = False
    max_candidate_chunks: int = 200
    max_returned_documents: int = 50
    max_supporting_chunks_per_document: int = 3
    document_relevance_threshold: float = 0.0
    scoring_strategy: str = DOCUMENT_SCORING_BEST_PLUS_AVERAGE
    latency_budget_ms: int | None = None

    def validate(self) -> None:
        _validate_int_range(
            "max_candidate_chunks",
            self.max_candidate_chunks,
            minimum=1,
            maximum=_MAX_CANDIDATE_CHUNKS_UPPER_BOUND,
        )
        _validate_int_range(
            "max_returned_documents",
            self.max_returned_documents,
            minimum=1,
            maximum=_MAX_RETURNED_DOCUMENTS_UPPER_BOUND,
        )
        _validate_int_range(
            "max_supporting_chunks_per_document",
            self.max_supporting_chunks_per_document,
            minimum=1,
            maximum=_MAX_SUPPORTING_CHUNKS_UPPER_BOUND,
        )
        if not math.isfinite(self.document_relevance_threshold):
            raise RetrievalConfigurationError("document_relevance_threshold must be finite.")
        if self.document_relevance_threshold < 0:
            raise RetrievalConfigurationError("document_relevance_threshold must be >= 0.")
        if self.scoring_strategy not in DOCUMENT_SCORING_STRATEGIES:
            allowed = ", ".join(sorted(DOCUMENT_SCORING_STRATEGIES))
            raise RetrievalConfigurationError(
                f"Unsupported document scoring strategy {self.scoring_strategy!r}; "
                f"expected one of: {allowed}."
            )
        if self.latency_budget_ms is not None:
            _validate_int_range(
                "latency_budget_ms",
                self.latency_budget_ms,
                minimum=1,
                maximum=300_000,
            )


@dataclass(frozen=True)
class SupportingPassage:
    chunk_id: str
    text: str
    score: float
    source: str | None
    chunk_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentSearchResult:
    document_id: str
    score: float
    best_passages: list[SupportingPassage]
    metadata: dict[str, Any] = field(default_factory=dict)
    candidate_chunk_count: int = 0
    best_chunk_score: float = 0.0


@dataclass(frozen=True)
class DocumentRetrievalDiagnostics:
    candidate_chunks_retrieved: int
    unique_documents_produced: int
    duplicate_document_hits_removed: int
    duplicate_chunks_removed: int
    chunks_missing_document_id: int
    documents_filtered: int
    final_document_count: int
    scoring_strategy: str
    document_relevance_threshold: float
    max_candidate_chunks: int
    max_returned_documents: int
    max_supporting_chunks_per_document: int
    retrieval_latency_ms: float | None = None
    aggregation_latency_ms: float | None = None
    latency_budget_ms: int | None = None
    latency_budget_exceeded: bool = False


@dataclass(frozen=True)
class DocumentRetrievalResult:
    documents: list[DocumentSearchResult]
    diagnostics: DocumentRetrievalDiagnostics


@dataclass(frozen=True)
class _DocumentGroup:
    document_id: str
    chunks: list[RetrievedChunk]


class DocumentScorer(Protocol):
    strategy_name: str

    def score(self, group: _DocumentGroup, *, top_n: int) -> float:
        """Return a deterministic document-level score for one document group."""


class BestPlusAverageTopChunksScorer:
    """Default deterministic scoring strategy.

    The score combines the best chunk score with the average of the top-N chunk
    scores. A single lucky chunk cannot fully define the strategy when multiple
    supporting chunks are present, while single-chunk documents remain valid.
    """

    strategy_name = DOCUMENT_SCORING_BEST_PLUS_AVERAGE

    def score(self, group: _DocumentGroup, *, top_n: int) -> float:
        scores = sorted((float(chunk.score) for chunk in group.chunks), reverse=True)
        if not scores:
            return 0.0
        bounded_top_n = max(1, min(top_n, len(scores)))
        best_score = scores[0]
        average_top_score = sum(scores[:bounded_top_n]) / bounded_top_n
        return (0.7 * best_score) + (0.3 * average_top_score)


def document_retrieval_config_from_env() -> DocumentRetrievalConfig:
    config = DocumentRetrievalConfig(
        enabled=_read_bool_env("NALUS_DOCUMENT_RETRIEVAL_ENABLED", default=False),
        max_candidate_chunks=_read_int_env(
            "NALUS_DOCUMENT_MAX_CANDIDATE_CHUNKS",
            default=200,
        ),
        max_returned_documents=_read_int_env(
            "NALUS_DOCUMENT_MAX_RETURNED_DOCUMENTS",
            default=50,
        ),
        max_supporting_chunks_per_document=_read_int_env(
            "NALUS_DOCUMENT_MAX_SUPPORTING_CHUNKS_PER_DOCUMENT",
            default=3,
        ),
        document_relevance_threshold=_read_float_env(
            "NALUS_DOCUMENT_RELEVANCE_THRESHOLD",
            default=0.0,
        ),
        scoring_strategy=os.getenv(
            "NALUS_DOCUMENT_SCORING_STRATEGY",
            DOCUMENT_SCORING_BEST_PLUS_AVERAGE,
        ).strip(),
        latency_budget_ms=_read_optional_int_env("NALUS_DOCUMENT_LATENCY_BUDGET_MS"),
    )
    config.validate()
    return config


def build_document_level_results(
    *,
    candidate_chunks: list[RetrievedChunk],
    config: DocumentRetrievalConfig,
    retrieval_latency_ms: float | None = None,
) -> DocumentRetrievalResult:
    config.validate()
    aggregation_started = time.perf_counter()
    bounded_candidates = candidate_chunks[: config.max_candidate_chunks]
    groups, duplicate_chunks_removed, chunks_missing_document_id = group_chunks_by_document(
        bounded_candidates
    )
    scorer = _scorer_for_strategy(config.scoring_strategy)

    documents: list[DocumentSearchResult] = []
    for group in groups:
        score = scorer.score(group, top_n=config.max_supporting_chunks_per_document)
        if score < config.document_relevance_threshold:
            continue
        documents.append(
            DocumentSearchResult(
                document_id=group.document_id,
                score=score,
                best_passages=_best_supporting_passages(
                    group.chunks,
                    max_passages=config.max_supporting_chunks_per_document,
                ),
                metadata=_document_metadata(group),
                candidate_chunk_count=len(group.chunks),
                best_chunk_score=max(float(chunk.score) for chunk in group.chunks),
            )
        )

    documents.sort(key=lambda document: (-document.score, document.document_id))
    filtered_count = len(groups) - len(documents)
    documents = documents[: config.max_returned_documents]
    aggregation_latency_ms = (time.perf_counter() - aggregation_started) * 1000
    total_latency_ms = (retrieval_latency_ms or 0.0) + aggregation_latency_ms
    latency_budget_exceeded = (
        config.latency_budget_ms is not None and total_latency_ms > config.latency_budget_ms
    )
    duplicate_document_hits_removed = len(bounded_candidates) - len(groups) - chunks_missing_document_id

    diagnostics = DocumentRetrievalDiagnostics(
        candidate_chunks_retrieved=len(bounded_candidates),
        unique_documents_produced=len(groups),
        duplicate_document_hits_removed=max(0, duplicate_document_hits_removed),
        duplicate_chunks_removed=duplicate_chunks_removed,
        chunks_missing_document_id=chunks_missing_document_id,
        documents_filtered=filtered_count,
        final_document_count=len(documents),
        scoring_strategy=config.scoring_strategy,
        document_relevance_threshold=config.document_relevance_threshold,
        max_candidate_chunks=config.max_candidate_chunks,
        max_returned_documents=config.max_returned_documents,
        max_supporting_chunks_per_document=config.max_supporting_chunks_per_document,
        retrieval_latency_ms=retrieval_latency_ms,
        aggregation_latency_ms=aggregation_latency_ms,
        latency_budget_ms=config.latency_budget_ms,
        latency_budget_exceeded=latency_budget_exceeded,
    )
    trace_event(
        logger,
        "document_retrieval.aggregate",
        candidate_chunks_retrieved=diagnostics.candidate_chunks_retrieved,
        unique_documents_produced=diagnostics.unique_documents_produced,
        duplicate_document_hits_removed=diagnostics.duplicate_document_hits_removed,
        duplicate_chunks_removed=diagnostics.duplicate_chunks_removed,
        documents_filtered=diagnostics.documents_filtered,
        final_document_count=diagnostics.final_document_count,
        aggregation_latency_ms=round(aggregation_latency_ms, 3),
    )
    logger.info(
        "[document-retrieval] candidates=%d documents=%d filtered=%d final=%d",
        diagnostics.candidate_chunks_retrieved,
        diagnostics.unique_documents_produced,
        diagnostics.documents_filtered,
        diagnostics.final_document_count,
    )
    return DocumentRetrievalResult(documents=documents, diagnostics=diagnostics)


def group_chunks_by_document(
    chunks: list[RetrievedChunk],
) -> tuple[list[_DocumentGroup], int, int]:
    grouped: dict[str, dict[str, RetrievedChunk]] = {}
    chunks_missing_document_id = 0
    duplicate_chunks_removed = 0

    for chunk in chunks:
        document_id = canonical_document_id(chunk)
        if document_id is None:
            chunks_missing_document_id += 1
            continue
        chunk_key = str(chunk.id).strip()
        if not chunk_key:
            chunks_missing_document_id += 1
            continue

        document_chunks = grouped.setdefault(document_id, {})
        existing = document_chunks.get(chunk_key)
        if existing is None or chunk.score > existing.score:
            if existing is not None:
                duplicate_chunks_removed += 1
            document_chunks[chunk_key] = chunk
        else:
            duplicate_chunks_removed += 1

    groups = [
        _DocumentGroup(
            document_id=document_id,
            chunks=sorted(document_chunks.values(), key=lambda item: (-item.score, item.id)),
        )
        for document_id, document_chunks in grouped.items()
    ]
    groups.sort(key=lambda group: group.document_id)
    return groups, duplicate_chunks_removed, chunks_missing_document_id


def canonical_document_id(chunk: RetrievedChunk) -> str | None:
    metadata = dict(chunk.metadata or {})
    for key in ("source_document_id", "document_id", "ecli", "case_reference", "reference"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _best_supporting_passages(
    chunks: list[RetrievedChunk],
    *,
    max_passages: int,
) -> list[SupportingPassage]:
    passages: list[SupportingPassage] = []
    seen_text: set[str] = set()
    for chunk in sorted(chunks, key=lambda item: (-item.score, item.id)):
        normalized_text = " ".join(chunk.text.split()).strip().lower()
        if not normalized_text or normalized_text in seen_text:
            continue
        seen_text.add(normalized_text)
        passages.append(
            SupportingPassage(
                chunk_id=chunk.id,
                text=chunk.text,
                score=chunk.score,
                source=chunk.source,
                chunk_index=_coerce_optional_int((chunk.metadata or {}).get("chunk_index")),
                metadata=_safe_passage_metadata(chunk),
            )
        )
        if len(passages) >= max_passages:
            break
    return passages


def _document_metadata(group: _DocumentGroup) -> dict[str, Any]:
    best_chunk = sorted(group.chunks, key=lambda item: (-item.score, item.id))[0]
    metadata = _safe_document_metadata(best_chunk)
    metadata.setdefault("document_id", group.document_id)
    return metadata


def _safe_document_metadata(chunk: RetrievedChunk) -> dict[str, Any]:
    metadata = dict(chunk.metadata or {})
    return {
        key: value
        for key, value in metadata.items()
        if key in _SAFE_DOCUMENT_METADATA_KEYS and value is not None and value != ""
    }


def _safe_passage_metadata(chunk: RetrievedChunk) -> dict[str, Any]:
    metadata = _safe_document_metadata(chunk)
    chunk_index = _coerce_optional_int((chunk.metadata or {}).get("chunk_index"))
    if chunk_index is not None:
        metadata["chunk_index"] = chunk_index
    return metadata


def _scorer_for_strategy(strategy: str) -> DocumentScorer:
    if strategy == DOCUMENT_SCORING_BEST_PLUS_AVERAGE:
        return BestPlusAverageTopChunksScorer()
    allowed = ", ".join(sorted(DOCUMENT_SCORING_STRATEGIES))
    raise RetrievalConfigurationError(
        f"Unsupported document scoring strategy {strategy!r}; expected one of: {allowed}."
    )


def _read_bool_env(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RetrievalConfigurationError(f"{name} must be a boolean value.")


def _read_int_env(name: str, *, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RetrievalConfigurationError(f"{name} must be an integer.") from exc


def _read_optional_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise RetrievalConfigurationError(f"{name} must be an integer.") from exc


def _read_float_env(name: str, *, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RetrievalConfigurationError(f"{name} must be a number.") from exc


def _validate_int_range(name: str, value: int, *, minimum: int, maximum: int) -> None:
    if value < minimum or value > maximum:
        raise RetrievalConfigurationError(f"{name} must be between {minimum} and {maximum}.")


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
