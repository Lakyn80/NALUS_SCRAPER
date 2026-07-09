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
ALLOWED_TARGET_CORPORA = frozenset({"usoud", "nsoud", "both", "ambiguous"})
ALLOWED_DIFFICULTIES = frozenset({"easy", "medium", "hard"})
ALLOWED_EVALUATION_TYPES = frozenset({"retrieval", "synthesis"})
MIXED_MERGE_RRF_K = 60


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
    expected_target_corpus: str | None = None

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
            expected_target_corpus=_nullable_str(payload.get("expected_target_corpus")),
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
    source_hit_at_1: bool | None = None
    source_hit_at_3: bool | None = None
    source_hit_at_5: bool | None = None
    passed: bool = False
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
    source_pending_count: int = 0
    gold_question_count: int = 0
    source_hit_at_1: float | None = None
    source_hit_at_3: float | None = None
    source_hit_at_5: float | None = None


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
    validate_mixed_dataset_item(payload)


def validate_mixed_dataset_item(payload: dict[str, Any]) -> None:
    corpus = str(payload.get("corpus") or "")
    if corpus != "mixed":
        return
    target = str(payload.get("expected_target_corpus") or "").strip()
    if not target:
        raise RetrievalConfigurationError(
            f"Mixed dataset item {payload.get('id')!r} must include expected_target_corpus."
        )
    if target not in ALLOWED_TARGET_CORPORA:
        raise RetrievalConfigurationError(
            f"Unsupported expected_target_corpus {target!r} in item {payload.get('id')!r}."
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


def _metadata_field_value(metadata: dict[str, Any], field_name: str) -> Any:
    chunk_meta = _parse_chunk_metadata(metadata)
    if field_name == "source_document_id":
        for key in ("source_document_id", "document_id", "ecli"):
            value = metadata.get(key) or chunk_meta.get(key) or chunk_meta.get("source_document_id")
            if value not in {None, ""}:
                return value
        return None
    if field_name == "case_reference":
        for key in ("case_reference", "spisova_znacka", "case_number"):
            value = metadata.get(key) or chunk_meta.get(key)
            if value not in {None, ""}:
                return value
        return None
    if field_name == "source":
        value = metadata.get("source") or metadata.get("retrieved_corpus")
        if str(value or "").strip() == "usoud":
            return metadata.get("source") or "usoud / nalus"
        return value
    if field_name == "court":
        corpus = str(metadata.get("retrieved_corpus") or "").strip()
        if corpus == "usoud":
            return "Ústavní soud"
        if corpus == "nsoud":
            return "Nejvyšší soud"
        return metadata.get("court")
    return metadata.get(field_name)


def _hit_matches_constraints(hit: RetrievedHitRecord, active: dict[str, str]) -> bool:
    metadata = hit.metadata
    for field_name, expected in active.items():
        actual = _metadata_field_value(metadata, field_name)
        if actual is None:
            return False
        if normalize_for_match(str(actual)) != normalize_for_match(expected):
            return False
    return True


def _source_hit_at_k(
    constraints: SourceConstraints,
    hits: list[RetrievedHitRecord],
    k: int,
) -> bool | None:
    if not constraints.active_constraints():
        return None
    return source_constraint_match_ratio(constraints, hits[:k]) == 1.0


def evaluate_question(
    item: LegalQaItem,
    hits: list[RetrievedHitRecord],
) -> QuestionRetrievalResult:
    constraints = item.expected_source_constraints
    has_gold = not item.source_pending and bool(constraints.active_constraints())
    source_hit_1 = _source_hit_at_k(constraints, hits, 1)
    source_hit_3 = _source_hit_at_k(constraints, hits, 3)
    source_hit_5 = _source_hit_at_k(constraints, hits, 5)

    if item.source_pending or not constraints.active_constraints():
        passed = keywords_hit(item.expected_keywords, hits)
        source_match = None
        failure_reason = None if passed else "No expected keyword found in top hits."
    else:
        source_match = source_constraint_match_ratio(constraints, hits)
        keyword_ok = keywords_hit(item.expected_keywords, hits)
        passed = source_match == 1.0 and keyword_ok
        if not keyword_ok:
            failure_reason = "No expected keyword found in top hits."
        elif source_match != 1.0:
            failure_reason = "Source constraints not satisfied in top hits."
        else:
            failure_reason = None

    return QuestionRetrievalResult(
        item=item,
        hits=hits,
        hit_at_1=keywords_hit(item.expected_keywords, hits[:1]),
        hit_at_3=keywords_hit(item.expected_keywords, hits[:3]),
        hit_at_5=keywords_hit(item.expected_keywords, hits[:5]),
        hit_at_10=keywords_hit(item.expected_keywords, hits[:10]),
        keyword_coverage=keyword_coverage(item.expected_keywords, hits),
        source_constraint_match=source_match,
        source_hit_at_1=source_hit_1 if has_gold else None,
        source_hit_at_3=source_hit_3 if has_gold else None,
        source_hit_at_5=source_hit_5 if has_gold else None,
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
    gold_results = [result for result in results if result.source_hit_at_1 is not None]

    def _mean_source_hit(field: str) -> float | None:
        if not gold_results:
            return None
        return sum(bool(getattr(result, field)) for result in gold_results) / len(gold_results)

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
        source_pending_count=sum(1 for result in results if result.item.source_pending),
        gold_question_count=len(gold_results),
        source_hit_at_1=_mean_source_hit("source_hit_at_1"),
        source_hit_at_3=_mean_source_hit("source_hit_at_3"),
        source_hit_at_5=_mean_source_hit("source_hit_at_5"),
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
                        "source_hit_at_1": result.source_hit_at_1,
                        "source_hit_at_3": result.source_hit_at_3,
                        "source_hit_at_5": result.source_hit_at_5,
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
    if metrics.gold_question_count:
        summary_lines.append(f"- gold questions: {metrics.gold_question_count}")
    if metrics.source_hit_at_1 is not None:
        summary_lines.append(f"- source_hit@1: {metrics.source_hit_at_1:.3f}")
    if metrics.source_hit_at_3 is not None:
        summary_lines.append(f"- source_hit@3: {metrics.source_hit_at_3:.3f}")
    if metrics.source_hit_at_5 is not None:
        summary_lines.append(f"- source_hit@5: {metrics.source_hit_at_5:.3f}")
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


@dataclass(frozen=True)
class MixedRetrievedHitRecord:
    rank: int
    chunk_id: str
    text_snippet: str
    score: float
    source: str
    retrieved_corpus: str
    collection_name: str
    source_document_id: str | None
    ecli: str | None
    case_reference: str | None
    dense_score: float | None
    bm25_score: float | None
    rrf_score: float | None
    corpus_rank: int
    combined_rrf_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MixedQuestionRetrievalResult:
    item: LegalQaItem
    hits: list[MixedRetrievedHitRecord]
    retrieval_hit_at_1: bool
    retrieval_hit_at_3: bool
    retrieval_hit_at_5: bool
    retrieval_hit_at_10: bool
    corpus_hit_at_1: bool | None
    corpus_hit_at_3: bool | None
    corpus_hit_at_5: bool | None
    keyword_coverage: float
    source_constraint_match: float | None = None
    source_hit_at_1: bool | None = None
    source_hit_at_3: bool | None = None
    source_hit_at_5: bool | None = None
    passed: bool = False
    usoud_at_1: bool = False
    nsoud_at_1: bool = False
    failure_reason: str | None = None


@dataclass(frozen=True)
class MixedBenchmarkMetrics:
    question_count: int
    corpus_hit_at_1: float
    corpus_hit_at_3: float
    corpus_hit_at_5: float
    retrieval_hit_at_1: float
    retrieval_hit_at_3: float
    retrieval_hit_at_5: float
    retrieval_hit_at_10: float
    mean_keyword_coverage: float
    pass_rate: float
    usoud_win_rate_at_1: float
    nsoud_win_rate_at_1: float
    ambiguous_count: int
    source_pending_count: int
    corpus_scored_question_count: int
    gold_question_count: int = 0
    mean_source_constraint_match: float | None = None
    source_hit_at_1: float | None = None
    source_hit_at_3: float | None = None
    source_hit_at_5: float | None = None


@dataclass(frozen=True)
class MixedTwoPassConfig:
    usoud_collection_name: str
    nsoud_collection_name: str
    usoud_bm25_sidecar_path: Path
    nsoud_bm25_sidecar_path: Path
    qdrant_url: str
    use_redis_cache: bool


class MixedTwoPassSearchFn(Protocol):
    def __call__(self, query: str, top_k: int) -> list[MixedRetrievedHitRecord]: ...


def resolve_bm25_sidecar_path(
    *,
    collection_name: str,
    explicit_path: str | Path | None,
) -> Path:
    if explicit_path is not None:
        resolved = Path(explicit_path).resolve()
        if not resolved.exists():
            raise RetrievalConfigurationError(f"BM25 sidecar not found: {resolved}")
        return resolved

    from app.rag.retrieval.production_profile import production_retrieval_config_from_env

    previous_collection = os.environ.get("QDRANT_COLLECTION_NAME")
    os.environ["QDRANT_COLLECTION_NAME"] = collection_name
    try:
        config = production_retrieval_config_from_env()
    finally:
        if previous_collection is None:
            os.environ.pop("QDRANT_COLLECTION_NAME", None)
        else:
            os.environ["QDRANT_COLLECTION_NAME"] = previous_collection

    if config.qdrant_collection == collection_name and config.bm25_sidecar_path.exists():
        return config.bm25_sidecar_path.resolve()

    default_path = Path("storage/rag/bm25") / f"{collection_name}.sqlite"
    if default_path.exists():
        return default_path.resolve()

    raise RetrievalConfigurationError(
        f"BM25 sidecar for collection {collection_name!r} not found. "
        f"Expected configured path {config.bm25_sidecar_path} or {default_path}. "
        "Pass an explicit --*-bm25-sidecar-path flag."
    )


def _parse_chunk_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    raw = metadata.get("chunk_metadata")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return dict(decoded) if isinstance(decoded, dict) else {}
    return {}


def _extract_document_fields(metadata: dict[str, Any]) -> dict[str, str | None]:
    chunk_meta = _parse_chunk_metadata(metadata)
    document_id = _nullable_str(metadata.get("document_id")) or _nullable_str(
        metadata.get("source_document_id")
    ) or _nullable_str(chunk_meta.get("source_document_id"))
    ecli = _nullable_str(metadata.get("ecli"))
    if not ecli and document_id and document_id.startswith("ECLI:"):
        ecli = document_id
    case_reference = (
        _nullable_str(metadata.get("case_reference"))
        or _nullable_str(metadata.get("spisova_znacka"))
        or _nullable_str(chunk_meta.get("case_number"))
    )
    return {
        "source_document_id": document_id,
        "ecli": ecli,
        "case_reference": case_reference,
    }


def chunk_to_mixed_hit_record(
    *,
    rank: int,
    chunk: RetrievedChunk,
    retrieved_corpus: str,
    collection_name: str,
    corpus_rank: int,
    combined_rrf_score: float,
) -> MixedRetrievedHitRecord:
    metadata = dict(chunk.metadata)
    score_components = dict(metadata.get("score_components") or {})
    doc_fields = _extract_document_fields(metadata)
    return MixedRetrievedHitRecord(
        rank=rank,
        chunk_id=str(chunk.id),
        text_snippet=_snippet(chunk.text),
        score=float(chunk.score),
        source=str(chunk.source),
        retrieved_corpus=retrieved_corpus,
        collection_name=collection_name,
        source_document_id=doc_fields["source_document_id"],
        ecli=doc_fields["ecli"],
        case_reference=doc_fields["case_reference"],
        dense_score=_optional_float(score_components.get("dense")),
        bm25_score=_optional_float(score_components.get("bm25")),
        rrf_score=_optional_float(metadata.get("rrf_score")),
        corpus_rank=corpus_rank,
        combined_rrf_score=combined_rrf_score,
        metadata=metadata,
    )


def merge_two_pass_hits(
    *,
    usoud_hits: list[RetrievedChunk],
    nsoud_hits: list[RetrievedChunk],
    usoud_collection_name: str,
    nsoud_collection_name: str,
    top_k: int,
    rrf_k: int = MIXED_MERGE_RRF_K,
) -> list[MixedRetrievedHitRecord]:
    sortable: list[tuple[float, str, str, str, MixedRetrievedHitRecord]] = []

    for corpus_rank, chunk in enumerate(usoud_hits, start=1):
        combined_rrf = 1.0 / (rrf_k + corpus_rank)
        hit = chunk_to_mixed_hit_record(
            rank=0,
            chunk=chunk,
            retrieved_corpus="usoud",
            collection_name=usoud_collection_name,
            corpus_rank=corpus_rank,
            combined_rrf_score=combined_rrf,
        )
        sortable.append(
            (
                combined_rrf,
                hit.retrieved_corpus,
                hit.source_document_id or "",
                hit.chunk_id,
                hit,
            )
        )

    for corpus_rank, chunk in enumerate(nsoud_hits, start=1):
        combined_rrf = 1.0 / (rrf_k + corpus_rank)
        hit = chunk_to_mixed_hit_record(
            rank=0,
            chunk=chunk,
            retrieved_corpus="nsoud",
            collection_name=nsoud_collection_name,
            corpus_rank=corpus_rank,
            combined_rrf_score=combined_rrf,
        )
        sortable.append(
            (
                combined_rrf,
                hit.retrieved_corpus,
                hit.source_document_id or "",
                hit.chunk_id,
                hit,
            )
        )

    sortable.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    merged: list[MixedRetrievedHitRecord] = []
    for final_rank, (_, _, _, _, hit) in enumerate(sortable[:top_k], start=1):
        merged.append(
            MixedRetrievedHitRecord(
                rank=final_rank,
                chunk_id=hit.chunk_id,
                text_snippet=hit.text_snippet,
                score=hit.combined_rrf_score,
                source=hit.source,
                retrieved_corpus=hit.retrieved_corpus,
                collection_name=hit.collection_name,
                source_document_id=hit.source_document_id,
                ecli=hit.ecli,
                case_reference=hit.case_reference,
                dense_score=hit.dense_score,
                bm25_score=hit.bm25_score,
                rrf_score=hit.rrf_score,
                corpus_rank=hit.corpus_rank,
                combined_rrf_score=hit.combined_rrf_score,
                metadata=hit.metadata,
            )
        )
    return merged


def _mixed_hits_to_keyword_records(hits: list[MixedRetrievedHitRecord]) -> list[RetrievedHitRecord]:
    return [
        RetrievedHitRecord(
            rank=hit.rank,
            chunk_id=hit.chunk_id,
            text_snippet=hit.text_snippet,
            score=hit.score,
            source=hit.source,
            dense_score=hit.dense_score,
            bm25_score=hit.bm25_score,
            rrf_score=hit.rrf_score,
            metadata={
                **hit.metadata,
                "retrieved_corpus": hit.retrieved_corpus,
                "collection_name": hit.collection_name,
                "source_document_id": hit.source_document_id,
                "ecli": hit.ecli,
                "case_reference": hit.case_reference,
            },
        )
        for hit in hits
    ]


def _corpora_in_hits(hits: list[MixedRetrievedHitRecord], k: int) -> set[str]:
    return {hit.retrieved_corpus for hit in hits[:k]}


def corpus_hit_at_k(
    expected_target_corpus: str | None,
    hits: list[MixedRetrievedHitRecord],
    k: int,
) -> bool | None:
    if not expected_target_corpus or expected_target_corpus == "ambiguous":
        return None
    corpora = _corpora_in_hits(hits, k)
    if expected_target_corpus == "usoud":
        return "usoud" in corpora
    if expected_target_corpus == "nsoud":
        return "nsoud" in corpora
    if expected_target_corpus == "both":
        return "usoud" in corpora and "nsoud" in corpora
    return None


def evaluate_mixed_question(
    item: LegalQaItem,
    hits: list[MixedRetrievedHitRecord],
) -> MixedQuestionRetrievalResult:
    keyword_hits = _mixed_hits_to_keyword_records(hits)
    constraints = item.expected_source_constraints
    has_gold = not item.source_pending and bool(constraints.active_constraints())
    source_hit_1 = _source_hit_at_k(constraints, keyword_hits, 1)
    source_hit_3 = _source_hit_at_k(constraints, keyword_hits, 3)
    source_hit_5 = _source_hit_at_k(constraints, keyword_hits, 5)
    keyword_ok = keywords_hit(item.expected_keywords, keyword_hits)

    if item.source_pending or not constraints.active_constraints():
        passed = keyword_ok
        source_match = None
        failure_reason = None if passed else "No expected keyword found in top hits."
    else:
        source_match = source_constraint_match_ratio(constraints, keyword_hits)
        passed = source_match == 1.0 and keyword_ok
        if not keyword_ok:
            failure_reason = "No expected keyword found in top hits."
        elif source_match != 1.0:
            failure_reason = "Source constraints not satisfied in top hits."
        else:
            failure_reason = None

    return MixedQuestionRetrievalResult(
        item=item,
        hits=hits,
        retrieval_hit_at_1=keywords_hit(item.expected_keywords, keyword_hits[:1]),
        retrieval_hit_at_3=keywords_hit(item.expected_keywords, keyword_hits[:3]),
        retrieval_hit_at_5=keywords_hit(item.expected_keywords, keyword_hits[:5]),
        retrieval_hit_at_10=keywords_hit(item.expected_keywords, keyword_hits[:10]),
        corpus_hit_at_1=corpus_hit_at_k(item.expected_target_corpus, hits, 1),
        corpus_hit_at_3=corpus_hit_at_k(item.expected_target_corpus, hits, 3),
        corpus_hit_at_5=corpus_hit_at_k(item.expected_target_corpus, hits, 5),
        keyword_coverage=keyword_coverage(item.expected_keywords, keyword_hits),
        source_constraint_match=source_match,
        source_hit_at_1=source_hit_1 if has_gold else None,
        source_hit_at_3=source_hit_3 if has_gold else None,
        source_hit_at_5=source_hit_5 if has_gold else None,
        passed=passed,
        usoud_at_1=bool(hits) and hits[0].retrieved_corpus == "usoud",
        nsoud_at_1=bool(hits) and hits[0].retrieved_corpus == "nsoud",
        failure_reason=failure_reason,
    )


def aggregate_mixed_metrics(results: list[MixedQuestionRetrievalResult]) -> MixedBenchmarkMetrics:
    count = len(results)
    corpus_scored = [result for result in results if result.corpus_hit_at_1 is not None]
    corpus_count = len(corpus_scored)

    def _mean_corpus(field: str) -> float:
        if corpus_count == 0:
            return 0.0
        return sum(bool(getattr(result, field)) for result in corpus_scored) / corpus_count

    def _mean_source_hit(field: str) -> float | None:
        gold_results = [result for result in results if getattr(result, field) is not None]
        if not gold_results:
            return None
        return sum(bool(getattr(result, field)) for result in gold_results) / len(gold_results)

    source_values = [
        value
        for value in (result.source_constraint_match for result in results)
        if value is not None
    ]
    gold_results = [result for result in results if result.source_hit_at_1 is not None]

    return MixedBenchmarkMetrics(
        question_count=count,
        corpus_hit_at_1=_mean_corpus("corpus_hit_at_1"),
        corpus_hit_at_3=_mean_corpus("corpus_hit_at_3"),
        corpus_hit_at_5=_mean_corpus("corpus_hit_at_5"),
        retrieval_hit_at_1=sum(result.retrieval_hit_at_1 for result in results) / count,
        retrieval_hit_at_3=sum(result.retrieval_hit_at_3 for result in results) / count,
        retrieval_hit_at_5=sum(result.retrieval_hit_at_5 for result in results) / count,
        retrieval_hit_at_10=sum(result.retrieval_hit_at_10 for result in results) / count,
        mean_keyword_coverage=sum(result.keyword_coverage for result in results) / count,
        pass_rate=sum(result.passed for result in results) / count,
        usoud_win_rate_at_1=sum(result.usoud_at_1 for result in results) / count,
        nsoud_win_rate_at_1=sum(result.nsoud_at_1 for result in results) / count,
        ambiguous_count=sum(
            1 for result in results if result.item.expected_target_corpus == "ambiguous"
        ),
        source_pending_count=sum(1 for result in results if result.item.source_pending),
        corpus_scored_question_count=corpus_count,
        gold_question_count=len(gold_results),
        mean_source_constraint_match=(
            sum(source_values) / len(source_values) if source_values else None
        ),
        source_hit_at_1=_mean_source_hit("source_hit_at_1"),
        source_hit_at_3=_mean_source_hit("source_hit_at_3"),
        source_hit_at_5=_mean_source_hit("source_hit_at_5"),
    )


def build_mixed_two_pass_search_fn(config: MixedTwoPassConfig) -> MixedTwoPassSearchFn:
    validate_collection_name(config.usoud_collection_name)
    validate_collection_name(config.nsoud_collection_name)
    usoud_search = build_hybrid_retriever(
        collection_name=config.usoud_collection_name,
        qdrant_url=config.qdrant_url,
        use_redis_cache=config.use_redis_cache,
        bm25_sidecar_path=config.usoud_bm25_sidecar_path,
    )
    nsoud_search = build_hybrid_retriever(
        collection_name=config.nsoud_collection_name,
        qdrant_url=config.qdrant_url,
        use_redis_cache=config.use_redis_cache,
        bm25_sidecar_path=config.nsoud_bm25_sidecar_path,
    )

    def search_fn(query: str, top_k: int) -> list[MixedRetrievedHitRecord]:
        usoud_hits = usoud_search(query, top_k)
        nsoud_hits = nsoud_search(query, top_k)
        return merge_two_pass_hits(
            usoud_hits=usoud_hits,
            nsoud_hits=nsoud_hits,
            usoud_collection_name=config.usoud_collection_name,
            nsoud_collection_name=config.nsoud_collection_name,
            top_k=top_k,
        )

    return search_fn


def run_mixed_retrieval_benchmark(
    *,
    items: list[LegalQaItem],
    search_fn: MixedTwoPassSearchFn,
    top_k: int,
) -> list[MixedQuestionRetrievalResult]:
    results: list[MixedQuestionRetrievalResult] = []
    for item in items:
        if item.corpus != "mixed":
            raise RetrievalConfigurationError(
                f"Mixed benchmark item {item.id} has corpus={item.corpus!r}; expected 'mixed'."
            )
        if item.evaluation_type != "retrieval":
            raise RetrievalConfigurationError(
                f"Item {item.id} has evaluation_type={item.evaluation_type!r}; "
                "only retrieval is supported in v1."
            )
        hits = search_fn(item.question, top_k)
        results.append(evaluate_mixed_question(item, hits))
    return results


def write_mixed_run_outputs(
    *,
    output_dir: Path,
    dataset_path: Path,
    config: MixedTwoPassConfig,
    top_k: int,
    retrieval_only: bool,
    results: list[MixedQuestionRetrievalResult],
    metrics: MixedBenchmarkMetrics,
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
                        "expected_target_corpus": result.item.expected_target_corpus,
                        "question": result.item.question,
                        "legal_topic": result.item.legal_topic,
                        "difficulty": result.item.difficulty,
                        "source_pending": result.item.source_pending,
                        "passed": result.passed,
                        "failure_reason": result.failure_reason,
                        "retrieval_hit_at_1": result.retrieval_hit_at_1,
                        "retrieval_hit_at_3": result.retrieval_hit_at_3,
                        "retrieval_hit_at_5": result.retrieval_hit_at_5,
                        "retrieval_hit_at_10": result.retrieval_hit_at_10,
                        "corpus_hit_at_1": result.corpus_hit_at_1,
                        "corpus_hit_at_3": result.corpus_hit_at_3,
                        "corpus_hit_at_5": result.corpus_hit_at_5,
                        "keyword_coverage": result.keyword_coverage,
                        "source_constraint_match": result.source_constraint_match,
                        "source_hit_at_1": result.source_hit_at_1,
                        "source_hit_at_3": result.source_hit_at_3,
                        "source_hit_at_5": result.source_hit_at_5,
                        "usoud_at_1": result.usoud_at_1,
                        "nsoud_at_1": result.nsoud_at_1,
                        "hits": [asdict(hit) for hit in result.hits],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics_payload = {
        "generated_at": generated_at,
        "dataset": str(dataset_path),
        "mode": "mixed_two_pass",
        "usoud_collection_name": config.usoud_collection_name,
        "nsoud_collection_name": config.nsoud_collection_name,
        "usoud_bm25_sidecar_path": str(config.usoud_bm25_sidecar_path),
        "nsoud_bm25_sidecar_path": str(config.nsoud_bm25_sidecar_path),
        "top_k": top_k,
        "retrieval_only": retrieval_only,
        "use_redis_cache": config.use_redis_cache,
        "mixed_merge_rrf_k": MIXED_MERGE_RRF_K,
        **asdict(metrics),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    failures = [result for result in results if not result.passed]
    failures_lines = [
        "# Mixed retrieval benchmark failures",
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
                f"- Expected target corpus: {result.item.expected_target_corpus}",
                f"- Reason: {result.failure_reason}",
                f"- Keyword coverage: {result.keyword_coverage:.2f}",
                "",
            ]
        )
    (output_dir / "failures.md").write_text("\n".join(failures_lines), encoding="utf-8")

    summary_lines = [
        "# Mixed retrieval benchmark summary",
        "",
        f"- Generated: {generated_at}",
        f"- Dataset: `{dataset_path}`",
        f"- Mode: two-pass mixed retrieval",
        f"- ÚS collection: `{config.usoud_collection_name}`",
        f"- NSoud collection: `{config.nsoud_collection_name}`",
        f"- ÚS BM25 sidecar: `{config.usoud_bm25_sidecar_path}`",
        f"- NSoud BM25 sidecar: `{config.nsoud_bm25_sidecar_path}`",
        f"- Questions: {metrics.question_count}",
        f"- Corpus-scored questions: {metrics.corpus_scored_question_count}",
        f"- Ambiguous questions: {metrics.ambiguous_count}",
        f"- Source pending: {metrics.source_pending_count}",
        f"- Top-k: {top_k}",
        f"- Retrieval only: {retrieval_only}",
        f"- Redis cache: {config.use_redis_cache}",
        "",
        "## Metrics",
        "",
        f"- corpus_hit@1: {metrics.corpus_hit_at_1:.3f}",
        f"- corpus_hit@3: {metrics.corpus_hit_at_3:.3f}",
        f"- corpus_hit@5: {metrics.corpus_hit_at_5:.3f}",
        f"- retrieval_hit@1: {metrics.retrieval_hit_at_1:.3f}",
        f"- retrieval_hit@3: {metrics.retrieval_hit_at_3:.3f}",
        f"- retrieval_hit@5: {metrics.retrieval_hit_at_5:.3f}",
        f"- retrieval_hit@10: {metrics.retrieval_hit_at_10:.3f}",
        f"- mean keyword coverage: {metrics.mean_keyword_coverage:.3f}",
        f"- pass rate: {metrics.pass_rate:.3f}",
        f"- usoud_win_rate@1: {metrics.usoud_win_rate_at_1:.3f}",
        f"- nsoud_win_rate@1: {metrics.nsoud_win_rate_at_1:.3f}",
        "",
    ]
    if metrics.gold_question_count:
        summary_lines.append(f"- gold questions: {metrics.gold_question_count}")
    if metrics.mean_source_constraint_match is not None:
        summary_lines.append(
            f"- mean source constraint match: {metrics.mean_source_constraint_match:.3f}"
        )
    if metrics.source_hit_at_1 is not None:
        summary_lines.append(f"- source_hit@1: {metrics.source_hit_at_1:.3f}")
    if metrics.source_hit_at_3 is not None:
        summary_lines.append(f"- source_hit@3: {metrics.source_hit_at_3:.3f}")
    if metrics.source_hit_at_5 is not None:
        summary_lines.append(f"- source_hit@5: {metrics.source_hit_at_5:.3f}")
    summary_lines.append("")
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


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
