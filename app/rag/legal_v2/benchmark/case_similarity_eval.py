"""Document-level case-similarity retrieval evaluation metrics (offline)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

FAILURE_EXPECTED_DOCUMENT_MISSING = "expected_document_missing_from_corpus"
FAILURE_POSITIVE_NOT_RETRIEVED = "positive_not_retrieved"
FAILURE_HARD_NEGATIVE_ABOVE_POSITIVE = "hard_negative_ranked_above_positive"
FAILURE_RETRIEVAL_ERROR = "retrieval_error"
FAILURE_INVALID_BENCHMARK_ROW = "invalid_benchmark_row"
FAILURE_DOCUMENT_ID_MAPPING_ERROR = "document_id_mapping_error"
FAILURE_AGGREGATION_ERROR = "aggregation_error"
FAILURE_MISSING_VERIFIED_ECLI_IN_BENCHMARK = "missing_verified_ecli_in_benchmark"
FAILURE_EXPECTED_ECLI_MISSING_FROM_INDEX = "expected_ecli_missing_from_index"
FAILURE_RETRIEVED_RESULT_MISSING_ECLI = "retrieved_result_missing_ecli"
FAILURE_CANONICAL_IDENTITY_MISMATCH = "canonical_identity_mismatch"


class RetrievedDocumentScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int
    document_id: str
    ecli: str | None = None
    canonical_document_id: str | None = None
    source_document_id: str | None = None
    score: float | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    reranker_score: float | None = None


class CaseSimilarityQueryEvalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str
    query: str
    query_style: str
    difficulty: str
    expected_primary_document_id: str
    expected_primary_source_document_id: str | None = None
    expected_primary_ecli: str | None = None
    accepted_alternative_document_ids: list[str] = Field(default_factory=list)
    hard_negative_document_ids: list[str] = Field(default_factory=list)
    hard_negative_evaluable: bool = True
    hard_negative_blocker: str | None = None
    retrieved_document_ids: list[str] = Field(default_factory=list)
    retrieved_eclis: list[str] = Field(default_factory=list)
    retrieved_results: list[RetrievedDocumentScore] = Field(default_factory=list)
    primary_rank: int | None = None
    best_accepted_alternative_rank: int | None = None
    best_positive_rank: int | None = None
    best_positive_document_id: str | None = None
    hard_negative_ranks: dict[str, int | None] = Field(default_factory=dict)
    hard_negative_before_positive: bool | None = None
    hit_at_1: bool = False
    hit_at_3: bool = False
    hit_at_5: bool = False
    hit_at_10: bool = False
    reciprocal_rank: float = 0.0
    corpus_compatible: bool = True
    failure_type: str | None = None
    error: str | None = None


class CaseSimilarityAggregateMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_queries: int = 0
    valid_queries: int = 0
    evaluable_positive_retrieval_queries: int = 0
    corpus_index_failures: int = 0
    retrieval_execution_failures: int = 0
    hit_at_1: float | None = None
    hit_at_3: float | None = None
    hit_at_5: float | None = None
    hit_at_10: float | None = None
    mrr: float | None = None
    primary_only_hit_at_1: float | None = None
    primary_only_hit_at_3: float | None = None
    primary_only_hit_at_5: float | None = None
    primary_only_hit_at_10: float | None = None
    accepted_alternative_wins: int = 0
    no_positive_in_top_10: int = 0
    by_difficulty: dict[str, dict[str, float | int | None]] = Field(default_factory=dict)
    by_query_style: dict[str, dict[str, float | int | None]] = Field(default_factory=dict)
    hard_negative_evaluable_query_count: int = 0
    hard_negative_blocked_query_count: int = 0
    hard_negative_outrank_count: int = 0
    hard_negative_outrank_rate: float | None = None
    hard_negative_outrank_query_ids: list[str] = Field(default_factory=list)
    missing_hard_negative_document_count: int = 0


def _id_match_key(document_id: str) -> str:
    text = str(document_id or "").strip()
    if text.upper().startswith("ECLI:"):
        from app.rag.legal_v2.identity import ecli_key

        return ecli_key(text)
    return text


def first_rank(document_id: str, ranked_ids: Sequence[str]) -> int | None:
    target = _id_match_key(document_id)
    if not target:
        return None
    for index, candidate in enumerate(ranked_ids, start=1):
        if _id_match_key(candidate) == target:
            return index
    return None


def dedupe_document_ids(document_ids: Sequence[str]) -> list[str]:
    """Preserve first-seen order when collapsing duplicate chunk→document IDs."""
    seen: set[str] = set()
    out: list[str] = []
    for document_id in document_ids:
        key = _id_match_key(document_id)
        if not document_id or not key or key in seen:
            continue
        seen.add(key)
        out.append(document_id)
    return out


def evaluate_ranked_documents(
    *,
    query_id: str,
    query: str,
    query_style: str,
    difficulty: str,
    expected_primary_document_id: str,
    accepted_alternative_document_ids: Sequence[str],
    hard_negative_document_ids: Sequence[str],
    hard_negative_evaluable: bool,
    hard_negative_blocker: str | None,
    ranked_document_ids: Sequence[str],
    retrieved_results: Sequence[RetrievedDocumentScore] | None = None,
    corpus_compatible: bool = True,
    failure_type: str | None = None,
    error: str | None = None,
    top_k: int = 10,
    expected_primary_source_document_id: str | None = None,
    expected_primary_ecli: str | None = None,
) -> CaseSimilarityQueryEvalResult:
    ranked = dedupe_document_ids(ranked_document_ids)[:top_k]
    results = list(retrieved_results or [])
    if not results:
        results = [
            RetrievedDocumentScore(rank=index, document_id=document_id)
            for index, document_id in enumerate(ranked, start=1)
        ]

    primary_rank = first_rank(expected_primary_document_id, ranked)
    alt_ranks = [
        first_rank(document_id, ranked) for document_id in accepted_alternative_document_ids
    ]
    alt_ranks_present = [rank for rank in alt_ranks if rank is not None]
    best_alt_rank = min(alt_ranks_present) if alt_ranks_present else None
    best_alt_id = None
    if best_alt_rank is not None:
        for document_id in accepted_alternative_document_ids:
            if first_rank(document_id, ranked) == best_alt_rank:
                best_alt_id = document_id
                break

    positive_candidates: list[tuple[int, str]] = []
    if primary_rank is not None:
        positive_candidates.append((primary_rank, expected_primary_document_id))
    if best_alt_rank is not None and best_alt_id is not None:
        positive_candidates.append((best_alt_rank, best_alt_id))
    positive_candidates.sort(key=lambda item: item[0])
    best_positive_rank = positive_candidates[0][0] if positive_candidates else None
    best_positive_id = positive_candidates[0][1] if positive_candidates else None

    hard_ranks = {
        document_id: first_rank(document_id, ranked) for document_id in hard_negative_document_ids
    }
    hard_before_positive: bool | None
    if not hard_negative_evaluable:
        hard_before_positive = None
    elif best_positive_rank is None:
        if any(rank is not None for rank in hard_ranks.values()):
            hard_before_positive = True
        else:
            hard_before_positive = False
    else:
        hard_before_positive = any(
            rank is not None and rank < best_positive_rank for rank in hard_ranks.values()
        )

    hit1 = best_positive_rank == 1
    hit3 = best_positive_rank is not None and best_positive_rank <= 3
    hit5 = best_positive_rank is not None and best_positive_rank <= 5
    hit10 = best_positive_rank is not None and best_positive_rank <= 10
    mrr = (1.0 / best_positive_rank) if best_positive_rank else 0.0

    resolved_failure = failure_type
    if resolved_failure is None and corpus_compatible and error is None:
        if best_positive_rank is None:
            resolved_failure = FAILURE_POSITIVE_NOT_RETRIEVED
        elif hard_negative_evaluable and hard_before_positive:
            resolved_failure = FAILURE_HARD_NEGATIVE_ABOVE_POSITIVE

    retrieved_eclis = [doc_id for doc_id in ranked if str(doc_id).upper().startswith("ECLI:")]

    return CaseSimilarityQueryEvalResult(
        query_id=query_id,
        query=query,
        query_style=query_style,
        difficulty=difficulty,
        expected_primary_document_id=expected_primary_document_id,
        expected_primary_source_document_id=expected_primary_source_document_id,
        expected_primary_ecli=expected_primary_ecli or (
            expected_primary_document_id
            if str(expected_primary_document_id).upper().startswith("ECLI:")
            else None
        ),
        accepted_alternative_document_ids=list(accepted_alternative_document_ids),
        hard_negative_document_ids=list(hard_negative_document_ids),
        hard_negative_evaluable=hard_negative_evaluable,
        hard_negative_blocker=hard_negative_blocker,
        retrieved_document_ids=ranked,
        retrieved_eclis=retrieved_eclis,
        retrieved_results=results[:top_k],
        primary_rank=primary_rank,
        best_accepted_alternative_rank=best_alt_rank,
        best_positive_rank=best_positive_rank,
        best_positive_document_id=best_positive_id,
        hard_negative_ranks=hard_ranks,
        hard_negative_before_positive=hard_before_positive,
        hit_at_1=hit1,
        hit_at_3=hit3,
        hit_at_5=hit5,
        hit_at_10=hit10,
        reciprocal_rank=mrr,
        corpus_compatible=corpus_compatible,
        failure_type=resolved_failure,
        error=error,
    )


def _group_positive_metrics(
    rows: Sequence[CaseSimilarityQueryEvalResult],
) -> dict[str, float | int | None]:
    if not rows:
        return {
            "count": 0,
            "hit_at_1": None,
            "hit_at_3": None,
            "hit_at_5": None,
            "hit_at_10": None,
            "mrr": None,
        }
    n = len(rows)
    return {
        "count": n,
        "hit_at_1": sum(1 for row in rows if row.hit_at_1) / n,
        "hit_at_3": sum(1 for row in rows if row.hit_at_3) / n,
        "hit_at_5": sum(1 for row in rows if row.hit_at_5) / n,
        "hit_at_10": sum(1 for row in rows if row.hit_at_10) / n,
        "mrr": sum(row.reciprocal_rank for row in rows) / n,
    }


def aggregate_case_similarity_metrics(
    results: Sequence[CaseSimilarityQueryEvalResult],
    *,
    missing_hard_negative_document_count: int = 0,
) -> CaseSimilarityAggregateMetrics:
    total = len(results)
    corpus_failures = sum(
        1
        for row in results
        if not row.corpus_compatible
        or row.failure_type
        in {FAILURE_EXPECTED_DOCUMENT_MISSING, FAILURE_DOCUMENT_ID_MAPPING_ERROR}
    )
    retrieval_failures = sum(
        1 for row in results if row.failure_type == FAILURE_RETRIEVAL_ERROR or row.error
    )
    evaluable = [
        row
        for row in results
        if row.corpus_compatible and row.failure_type != FAILURE_RETRIEVAL_ERROR and not row.error
    ]
    valid = [
        row
        for row in results
        if row.failure_type not in {FAILURE_INVALID_BENCHMARK_ROW, FAILURE_AGGREGATION_ERROR}
    ]

    def _rate(predicate) -> float | None:
        if not evaluable:
            return None
        return sum(1 for row in evaluable if predicate(row)) / len(evaluable)

    primary_only = [
        row
        for row in evaluable
        if row.primary_rank is not None
        and (
            row.best_accepted_alternative_rank is None
            or row.primary_rank <= row.best_accepted_alternative_rank
        )
    ]
    # Primary-only Hit@K uses primary_rank on all evaluable rows.
    def _primary_rate(limit: int) -> float | None:
        if not evaluable:
            return None
        return sum(
            1 for row in evaluable if row.primary_rank is not None and row.primary_rank <= limit
        ) / len(evaluable)

    alt_wins = sum(
        1
        for row in evaluable
        if row.best_accepted_alternative_rank is not None
        and (
            row.primary_rank is None or row.best_accepted_alternative_rank < row.primary_rank
        )
    )
    no_pos = sum(1 for row in evaluable if row.best_positive_rank is None)

    hn_evaluable_rows = [row for row in evaluable if row.hard_negative_evaluable]
    hn_blocked_rows = [row for row in results if not row.hard_negative_evaluable]
    hn_outrank_ids = [
        row.query_id
        for row in hn_evaluable_rows
        if row.hard_negative_before_positive is True
    ]
    hn_rate = (
        (len(hn_outrank_ids) / len(hn_evaluable_rows)) if hn_evaluable_rows else None
    )

    by_diff: dict[str, list[CaseSimilarityQueryEvalResult]] = defaultdict(list)
    by_style: dict[str, list[CaseSimilarityQueryEvalResult]] = defaultdict(list)
    for row in evaluable:
        by_diff[row.difficulty].append(row)
        by_style[row.query_style].append(row)

    return CaseSimilarityAggregateMetrics(
        total_queries=total,
        valid_queries=len(valid),
        evaluable_positive_retrieval_queries=len(evaluable),
        corpus_index_failures=corpus_failures,
        retrieval_execution_failures=retrieval_failures,
        hit_at_1=_rate(lambda row: row.hit_at_1),
        hit_at_3=_rate(lambda row: row.hit_at_3),
        hit_at_5=_rate(lambda row: row.hit_at_5),
        hit_at_10=_rate(lambda row: row.hit_at_10),
        mrr=(sum(row.reciprocal_rank for row in evaluable) / len(evaluable)) if evaluable else None,
        primary_only_hit_at_1=_primary_rate(1),
        primary_only_hit_at_3=_primary_rate(3),
        primary_only_hit_at_5=_primary_rate(5),
        primary_only_hit_at_10=_primary_rate(10),
        accepted_alternative_wins=alt_wins,
        no_positive_in_top_10=no_pos,
        by_difficulty={key: _group_positive_metrics(rows) for key, rows in sorted(by_diff.items())},
        by_query_style={key: _group_positive_metrics(rows) for key, rows in sorted(by_style.items())},
        hard_negative_evaluable_query_count=len(hn_evaluable_rows),
        hard_negative_blocked_query_count=len(hn_blocked_rows),
        hard_negative_outrank_count=len(hn_outrank_ids),
        hard_negative_outrank_rate=hn_rate,
        hard_negative_outrank_query_ids=hn_outrank_ids,
        missing_hard_negative_document_count=missing_hard_negative_document_count,
    )


def corpus_presence_summary(
    *,
    items: Sequence[Any],
    present_document_ids: Iterable[str],
) -> dict[str, Any]:
    """Summarize corpus presence using production ECLI identity when available.

    ``present_document_ids`` should contain indexed production IDs (ECLI preferred).
    Golden ``doc-*`` IDs are matched via ``expected_primary_ecli`` /
    rationale ``ecli`` fields, not by treating ``doc-*`` as production IDs.
    """
    from app.rag.legal_v2.identity import ecli_key, is_valid_ecli

    present_keys = {_id_match_key(doc_id) for doc_id in present_document_ids if doc_id}
    primary_present = 0
    primary_missing = 0
    alt_present = 0
    alt_missing = 0
    hn_present = 0
    hn_missing = 0
    hn_evaluable = 0
    hn_blocked = 0
    details: list[dict[str, Any]] = []

    def _lookup_id(source_document_id: str, ecli: str | None) -> tuple[str, bool]:
        if ecli and is_valid_ecli(ecli):
            key = ecli_key(ecli)
            return normalize_display_id(ecli), key in present_keys
        key = _id_match_key(source_document_id)
        return source_document_id, key in present_keys

    def normalize_display_id(value: str) -> str:
        from app.rag.legal_v2.identity import normalize_ecli

        return normalize_ecli(value) if is_valid_ecli(value) else value

    for item in items:
        primary_ecli = getattr(item, "expected_primary_ecli", None)
        primary_id, primary_ok = _lookup_id(item.source_document_id, primary_ecli)
        primary_present += int(primary_ok)
        primary_missing += int(not primary_ok)

        alt_rows = list(getattr(item, "accepted_alternative_rationales", []) or [])
        alt_by_source = {row.document_id: getattr(row, "ecli", None) for row in alt_rows}
        alt_status: dict[str, bool] = {}
        for doc in item.accepted_alternative_document_ids:
            _, ok = _lookup_id(doc, alt_by_source.get(doc))
            alt_status[doc] = ok

        hn_rows = list(getattr(item, "hard_negative_rationales", []) or [])
        hn_by_source = {row.document_id: getattr(row, "ecli", None) for row in hn_rows}
        hn_status: dict[str, bool] = {}
        for doc in item.hard_negative_document_ids:
            _, ok = _lookup_id(doc, hn_by_source.get(doc))
            hn_status[doc] = ok

        alt_present += sum(1 for ok in alt_status.values() if ok)
        alt_missing += sum(1 for ok in alt_status.values() if not ok)
        hn_present += sum(1 for ok in hn_status.values() if ok)
        hn_missing += sum(1 for ok in hn_status.values() if not ok)
        if item.hard_negative_evaluable:
            hn_evaluable += 1
        else:
            hn_blocked += 1
        details.append(
            {
                "benchmark_id": item.benchmark_id,
                "primary": {
                    "source_document_id": item.source_document_id,
                    "ecli": primary_ecli,
                    "match_id": primary_id,
                    "present": primary_ok,
                },
                "accepted_alternatives": [
                    {
                        "source_document_id": doc,
                        "ecli": alt_by_source.get(doc),
                        "present": ok,
                    }
                    for doc, ok in alt_status.items()
                ],
                "hard_negatives": [
                    {
                        "source_document_id": doc,
                        "ecli": hn_by_source.get(doc),
                        "present": ok,
                    }
                    for doc, ok in hn_status.items()
                ],
                "hard_negative_evaluable": item.hard_negative_evaluable,
                "hard_negative_blocker": item.hard_negative_blocker,
            }
        )
    return {
        "primary_documents_present": primary_present,
        "primary_documents_missing": primary_missing,
        "accepted_alternatives_present": alt_present,
        "accepted_alternatives_missing": alt_missing,
        "hard_negatives_present": hn_present,
        "hard_negatives_missing": hn_missing,
        "hard_negative_evaluable_entries": hn_evaluable,
        "hard_negative_blocked_entries": hn_blocked,
        "details": details,
    }
