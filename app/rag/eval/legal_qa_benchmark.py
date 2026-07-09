"""Retrieval-only legal Q&A benchmark harness for BGE-M3 hybrid RAG."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from app.rag.retrieval.errors import RetrievalConfigurationError
from app.rag.retrieval.models import RetrievedChunk

PRODUCTION_COLLECTION_DENYLIST = frozenset(
    {"nalus", "nalus_live", "nalus_stable_20260326"}
)
REQUIRED_DATASET_FIELDS = (
    "id",
    "corpus",
    "question",
    "expected_answer_points",
    "expected_source_constraints",
    "expected_keywords",
    "forbidden_answer_patterns",
    "difficulty",
    "legal_topic",
    "evaluation_type",
    "source_pending",
)
ALLOWED_CORPORA = frozenset({"usoud", "nsoud", "mixed"})
ALLOWED_DIFFICULTIES = frozenset({"easy", "medium", "hard"})
ALLOWED_EVALUATION_TYPES = frozenset({"retrieval", "synthesis"})


class RetrievalSearchFn(Protocol):
    def __call__(self, query: str, top_k: int) -> list[RetrievedChunk]: ...


@dataclass(frozen=True)
class SourceConstraints:
    court: str | None = None
    source: str | None = None
    case_reference: str | None = None
    source_document_id: str | None = None
    decision_date: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SourceConstraints":
        return cls(
            court=_nullable_str(payload.get("court")),
            source=_nullable_str(payload.get("source")),
            case_reference=_nullable_str(payload.get("case_reference")),
            source_document_id=_nullable_str(payload.get("source_document_id")),
            decision_date=_nullable_str(payload.get("decision_date")),
        )

    def active_constraints(self) -> dict[str, str]:
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None
        }


@dataclass(frozen=True)
class LegalQaItem:
    id: str
    corpus: str
    question: str
    expected_answer_points: list[str]
    expected_source_constraints: SourceConstraints
    expected_keywords: list[str]
    forbidden_answer_patterns: list[str]
    difficulty: str
    legal_topic: str
    evaluation_type: str
    source_pending: bool

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LegalQaItem":
        validate_dataset_item(payload)
        return cls(
            id=str(payload["id"]),
            corpus=str(payload["corpus"]),
            question=str(payload["question"]),
            expected_answer_points=[str(item) for item in payload["expected_answer_points"]],
            expected_source_constraints=SourceConstraints.from_dict(
                dict(payload.get("expected_source_constraints") or {})
            ),
            expected_keywords=[str(item) for item in payload["expected_keywords"]],
            forbidden_answer_patterns=[str(item) for item in payload.get("forbidden_answer_patterns") or []],
            difficulty=str(payload["difficulty"]),
            legal_topic=str(payload["legal_topic"]),
            evaluation_type=str(payload["evaluation_type"]),
            source_pending=bool(payload["source_pending"]),
        )


@dataclass(frozen=True)
class RetrievedHitRecord:
    rank: int
    chunk_id: str
    text_snippet: str
    score: float
    source: str
    dense_score: float | None
    bm25_score: float | None
    rrf_score: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuestionRetrievalResult:
    item: LegalQaItem
    hits: list[RetrievedHitRecord]
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    hit_at_10: bool
    keyword_coverage: float
    source_constraint_match: float | None
    passed: bool
    failure_reason: str | None = None


@dataclass(frozen=True)
class BenchmarkMetrics:
    question_count: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    hit_at_10: float
    mean_keyword_coverage: float
    mean_source_constraint_match: float | None
    pass_rate: float


def validate_dataset_item(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise RetrievalConfigurationError("Dataset item must be a JSON object.")
    missing = [field_name for field_name in REQUIRED_DATASET_FIELDS if field_name not in payload]
    if missing:
        raise RetrievalConfigurationError(
            "Dataset item is missing required fields: " + ", ".join(missing)
        )
    if not str(payload.get("id") or "").strip():
        raise RetrievalConfigurationError("Dataset item id must not be empty.")
    if not str(payload.get("question") or "").strip():
        raise RetrievalConfigurationError(f"Dataset item {payload.get('id')!r} has empty question.")
    corpus = str(payload.get("corpus") or "")
    if corpus not in ALLOWED_CORPORA:
        raise RetrievalConfigurationError(f"Unsupported corpus {corpus!r} in item {payload.get('id')!r}.")
    difficulty = str(payload.get("difficulty") or "")
    if difficulty not in ALLOWED_DIFFICULTIES:
        raise RetrievalConfigurationError(
            f"Unsupported difficulty {difficulty!r} in item {payload.get('id')!r}."
        )
    evaluation_type = str(payload.get("evaluation_type") or "")
    if evaluation_type not in ALLOWED_EVALUATION_TYPES:
        raise RetrievalConfigurationError(
            f"Unsupported evaluation_type {evaluation_type!r} in item {payload.get('id')!r}."
        )
    keywords = payload.get("expected_keywords")
    if not isinstance(keywords, list) or not keywords:
        raise RetrievalConfigurationError(
            f"Dataset item {payload.get('id')!r} must include non-empty expected_keywords."
        )
    constraints = payload.get("expected_source_constraints")
    if not isinstance(constraints, dict):
        raise RetrievalConfigurationError(
            f"Dataset item {payload.get('id')!r} must include expected_source_constraints object."
        )


def load_dataset(path: Path, *, limit: int | None = None) -> list[LegalQaItem]:
    items: list[LegalQaItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RetrievalConfigurationError(
                    f"Invalid JSON in dataset {path} line {line_number}: {exc}"
                ) from exc
            items.append(LegalQaItem.from_dict(payload))
            if limit is not None and len(items) >= limit:
                break
    if not items:
        raise RetrievalConfigurationError(f"Dataset {path} contains no items.")
    return items


def validate_collection_name(collection_name: str) -> None:
    normalized = collection_name.strip()
    if not normalized:
        raise RetrievalConfigurationError("Collection name must not be empty.")
    if normalized in PRODUCTION_COLLECTION_DENYLIST or normalized.startswith("nalus_stable_"):
        raise RetrievalConfigurationError(f"Refusing protected collection: {normalized}")


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def keyword_coverage(expected_keywords: list[str], hits: list[RetrievedHitRecord]) -> float:
    if not expected_keywords:
        return 0.0
    haystack = " ".join(
        f"{hit.text_snippet} {json.dumps(hit.metadata, ensure_ascii=False)}" for hit in hits
    )
    normalized_haystack = normalize_for_match(haystack)
    matched = sum(
        1 for keyword in expected_keywords if normalize_for_match(keyword) in normalized_haystack
    )
    return matched / len(expected_keywords)


def keywords_hit(expected_keywords: list[str], hits: list[RetrievedHitRecord]) -> bool:
    return keyword_coverage(expected_keywords, hits) > 0.0


def source_constraint_match_ratio(
    constraints: SourceConstraints,
    hits: list[RetrievedHitRecord],
) -> float | None:
    active = constraints.active_constraints()
    if not active:
        return None
    for hit in hits:
        if _hit_matches_constraints(hit, active):
            return 1.0
    return 0.0


def _hit_matches_constraints(hit: RetrievedHitRecord, active: dict[str, str]) -> bool:
    metadata = hit.metadata
    for field_name, expected in active.items():
        actual = metadata.get(field_name)
        if actual is None:
            return False
        if normalize_for_match(str(actual)) != normalize_for_match(expected):
            return False
    return True


def evaluate_question(
    item: LegalQaItem,
    hits: list[RetrievedHitRecord],
) -> QuestionRetrievalResult:
    if item.source_pending or not item.expected_source_constraints.active_constraints():
        passed = keywords_hit(item.expected_keywords, hits)
        source_match = None
        failure_reason = None if passed else "No expected keyword found in top hits."
    else:
        source_match = source_constraint_match_ratio(item.expected_source_constraints, hits)
        passed = source_match == 1.0
        failure_reason = None if passed else "Source constraints not satisfied in top hits."

    return QuestionRetrievalResult(
        item=item,
        hits=hits,
        hit_at_1=keywords_hit(item.expected_keywords, hits[:1]),
        hit_at_3=keywords_hit(item.expected_keywords, hits[:3]),
        hit_at_5=keywords_hit(item.expected_keywords, hits[:5]),
        hit_at_10=keywords_hit(item.expected_keywords, hits[:10]),
        keyword_coverage=keyword_coverage(item.expected_keywords, hits),
        source_constraint_match=source_match,
        passed=passed,
        failure_reason=failure_reason,
    )


def aggregate_metrics(results: list[QuestionRetrievalResult]) -> BenchmarkMetrics:
    count = len(results)
    source_values = [
        value
        for value in (result.source_constraint_match for result in results)
        if value is not None
    ]
    return BenchmarkMetrics(
        question_count=count,
        hit_at_1=sum(result.hit_at_1 for result in results) / count,
        hit_at_3=sum(result.hit_at_3 for result in results) / count,
        hit_at_5=sum(result.hit_at_5 for result in results) / count,
        hit_at_10=sum(result.hit_at_10 for result in results) / count,
        mean_keyword_coverage=sum(result.keyword_coverage for result in results) / count,
        mean_source_constraint_match=(
            sum(source_values) / len(source_values) if source_values else None
        ),
        pass_rate=sum(result.passed for result in results) / count,
    )


def chunk_to_hit_record(rank: int, chunk: RetrievedChunk) -> RetrievedHitRecord:
    metadata = dict(chunk.metadata)
    score_components = dict(metadata.get("score_components") or {})
    return RetrievedHitRecord(
        rank=rank,
        chunk_id=str(chunk.id),
        text_snippet=_snippet(chunk.text),
        score=float(chunk.score),
        source=str(chunk.source),
        dense_score=_optional_float(score_components.get("dense")),
        bm25_score=_optional_float(score_components.get("bm25")),
        rrf_score=_optional_float(metadata.get("rrf_score")),
        metadata=metadata,
    )


def run_retrieval_benchmark(
    *,
    items: list[LegalQaItem],
    search_fn: RetrievalSearchFn,
    top_k: int,
) -> list[QuestionRetrievalResult]:
    results: list[QuestionRetrievalResult] = []
    for item in items:
        if item.evaluation_type != "retrieval":
            raise RetrievalConfigurationError(
                f"Item {item.id} has evaluation_type={item.evaluation_type!r}; "
                "only retrieval is supported in v1."
            )
        raw_hits = search_fn(item.question, top_k)
        hits = [chunk_to_hit_record(rank, chunk) for rank, chunk in enumerate(raw_hits, start=1)]
        results.append(evaluate_question(item, hits))
    return results


def write_run_outputs(
    *,
    output_dir: Path,
    dataset_path: Path,
    collection_name: str,
    top_k: int,
    retrieval_only: bool,
    use_redis_cache: bool,
    results: list[QuestionRetrievalResult],
    metrics: BenchmarkMetrics,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    results_path = output_dir / "retrieval_results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(
                json.dumps(
                    {
                        "id": result.item.id,
                        "corpus": result.item.corpus,
                        "question": result.item.question,
                        "legal_topic": result.item.legal_topic,
                        "difficulty": result.item.difficulty,
                        "source_pending": result.item.source_pending,
                        "passed": result.passed,
                        "failure_reason": result.failure_reason,
                        "hit_at_1": result.hit_at_1,
                        "hit_at_3": result.hit_at_3,
                        "hit_at_5": result.hit_at_5,
                        "hit_at_10": result.hit_at_10,
                        "keyword_coverage": result.keyword_coverage,
                        "source_constraint_match": result.source_constraint_match,
                        "hits": [asdict(hit) for hit in result.hits],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics_payload = {
        "generated_at": generated_at,
        "dataset": str(dataset_path),
        "collection_name": collection_name,
        "top_k": top_k,
        "retrieval_only": retrieval_only,
        "use_redis_cache": use_redis_cache,
        **asdict(metrics),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    failures = [result for result in results if not result.passed]
    failures_lines = [
        "# Retrieval benchmark failures",
        "",
        f"Generated: {generated_at}",
        f"Failed: {len(failures)} / {len(results)}",
        "",
    ]
    for result in failures:
        failures_lines.extend(
            [
                f"## {result.item.id}",
                f"- Question: {result.item.question}",
                f"- Reason: {result.failure_reason}",
                f"- Keyword coverage: {result.keyword_coverage:.2f}",
                "",
            ]
        )
    (output_dir / "failures.md").write_text("\n".join(failures_lines), encoding="utf-8")

    summary_lines = [
        "# Retrieval benchmark summary",
        "",
        f"- Generated: {generated_at}",
        f"- Dataset: `{dataset_path}`",
        f"- Collection: `{collection_name}`",
        f"- Questions: {metrics.question_count}",
        f"- Top-k: {top_k}",
        f"- Retrieval only: {retrieval_only}",
        f"- Redis cache: {use_redis_cache}",
        "",
        "## Metrics",
        "",
        f"- hit@1: {metrics.hit_at_1:.3f}",
        f"- hit@3: {metrics.hit_at_3:.3f}",
        f"- hit@5: {metrics.hit_at_5:.3f}",
        f"- hit@10: {metrics.hit_at_10:.3f}",
        f"- mean keyword coverage: {metrics.mean_keyword_coverage:.3f}",
        f"- pass rate: {metrics.pass_rate:.3f}",
        "",
    ]
    if metrics.mean_source_constraint_match is not None:
        summary_lines.append(
            f"- mean source constraint match: {metrics.mean_source_constraint_match:.3f}"
        )
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def build_hybrid_retriever(
    *,
    collection_name: str,
    qdrant_url: str,
    use_redis_cache: bool,
    bm25_sidecar_path: str | Path | None = None,
) -> RetrievalSearchFn:
    validate_collection_name(collection_name)
    if use_redis_cache:
        os.environ["EMBEDDING_CACHE_ENABLED"] = "1"
    else:
        os.environ["EMBEDDING_CACHE_ENABLED"] = "0"

    from qdrant_client import QdrantClient

    from app.api.startup import _build_production_retrieval
    from app.rag.retrieval.production_profile import production_retrieval_config_from_env

    os.environ["QDRANT_COLLECTION_NAME"] = collection_name
    if bm25_sidecar_path is not None:
        resolved_bm25 = Path(bm25_sidecar_path).resolve()
        if not resolved_bm25.exists():
            raise RetrievalConfigurationError(f"BM25 sidecar not found: {resolved_bm25}")
        os.environ["BM25_SIDECAR_PATH"] = str(resolved_bm25)
    config = production_retrieval_config_from_env()
    if config.qdrant_collection != collection_name:
        raise RetrievalConfigurationError(
            f"Configured collection {config.qdrant_collection!r} != requested {collection_name!r}"
        )
    if bm25_sidecar_path is not None and config.bm25_sidecar_path.resolve() != Path(bm25_sidecar_path).resolve():
        raise RetrievalConfigurationError(
            "BM25 sidecar path mismatch between CLI override and production config."
        )
    client = QdrantClient(url=qdrant_url, timeout=30)
    retriever, _cache_build = _build_production_retrieval(client, config)
    return retriever.search


def _snippet(text: str, *, limit: int = 240) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _nullable_str(value: Any) -> str | None:
    cleaned = str(value or "").strip()
    return cleaned or None
