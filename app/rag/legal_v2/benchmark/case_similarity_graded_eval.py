"""Graded multi-relevance evaluation metrics for case-similarity golden v3."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from app.rag.legal_v2.benchmark.case_similarity_eval import (
    CaseSimilarityQueryEvalResult,
    dedupe_document_ids,
    evaluate_ranked_documents,
    first_rank,
)
from app.rag.legal_v2.benchmark.case_similarity_golden_v3 import (
    BINARY_RELEVANCE_THRESHOLD,
    GRADE_LABELS,
)

JUDGMENT_EXPLICIT_NOT_RELEVANT = "explicit_grade_0"
JUDGMENT_GRADED = "graded"
JUDGMENT_UNJUDGED = "unjudged"


@dataclass(frozen=True)
class QrelEntry:
    query_id: str
    document_id: str
    grade: int
    judgment_state: str = JUDGMENT_GRADED
    review_reason: str = ""


@dataclass
class GradedQueryEvalResult:
    query_id: str
    ranked_document_ids: list[str]
    qrels: dict[str, QrelEntry]
    ndcg_at_5: float | None = None
    ndcg_at_10: float | None = None
    ndcg_at_20: float | None = None
    precision_at_5: float | None = None
    precision_at_10: float | None = None
    recall_at_10: float | None = None
    recall_at_20: float | None = None
    recall_at_50: float | None = None
    mrr_highly_relevant: float = 0.0
    success_at_10_highly_relevant: bool = False
    legacy_primary_rank: int | None = None
    legacy_primary_hit_at_1: bool = False
    legacy_primary_hit_at_3: bool = False
    legacy_primary_hit_at_10: bool = False
    legacy_primary_reciprocal_rank: float = 0.0
    judged_relevant_count: int = 0
    judged_highly_relevant_count: int = 0
    explicit_not_relevant_count: int = 0
    unjudged_in_pool_count: int = 0
    eval_mode: str = "pooled_graded"
    notes: list[str] = field(default_factory=list)


@dataclass
class GradedAggregateMetrics:
    total_queries: int = 0
    ndcg_at_5: float | None = None
    ndcg_at_10: float | None = None
    ndcg_at_20: float | None = None
    precision_at_5: float | None = None
    precision_at_10: float | None = None
    recall_at_10: float | None = None
    recall_at_20: float | None = None
    recall_at_50: float | None = None
    mrr_highly_relevant: float | None = None
    success_at_10_highly_relevant: float | None = None
    legacy_primary_hit_at_1: float | None = None
    legacy_primary_hit_at_10: float | None = None
    legacy_primary_mrr: float | None = None
    total_reviewed_judgments: int = 0
    total_pending_judgments: int = 0


def grade_label(grade: int) -> str:
    return GRADE_LABELS.get(grade, "UNKNOWN")


def is_binary_relevant(grade: int) -> bool:
    return grade >= BINARY_RELEVANCE_THRESHOLD


def build_qrel_map(
    entries: Sequence[QrelEntry],
) -> dict[str, QrelEntry]:
    out: dict[str, QrelEntry] = {}
    for entry in entries:
        out[_match_key(entry.document_id)] = entry
    return out


def _match_key(document_id: str) -> str:
    from app.rag.legal_v2.benchmark.case_similarity_eval import _id_match_key

    return _id_match_key(document_id)


def _gain(grade: int) -> float:
    if grade <= 0:
        return 0.0
    return float(2**grade - 1)


def compute_dcg(relevances: Sequence[float], k: int) -> float:
    total = 0.0
    for index, rel in enumerate(relevances[:k], start=1):
        total += rel / math.log2(index + 1)
    return total


def compute_ndcg_at_k(
    ranked_document_ids: Sequence[str],
    qrels: Mapping[str, QrelEntry],
    *,
    k: int,
    unjudged_default_grade: int = 0,
) -> tuple[float, str]:
    """Pooled nDCG: unjudged documents in the ranked list use ``unjudged_default_grade``."""
    keys = [_match_key(doc_id) for doc_id in dedupe_document_ids(ranked_document_ids)[:k]]
    gains = []
    has_unjudged = False
    for key in keys:
        entry = qrels.get(key)
        if entry is None:
            has_unjudged = True
            gains.append(_gain(unjudged_default_grade))
        else:
            gains.append(_gain(entry.grade))
    ideal = sorted((_gain(entry.grade) for entry in qrels.values() if entry.judgment_state == JUDGMENT_GRADED), reverse=True)
    ideal_gains = ideal[:k]
    while len(ideal_gains) < k:
        ideal_gains.append(0.0)
    dcg = compute_dcg(gains, k)
    idcg = compute_dcg(ideal_gains, k)
    if idcg <= 0:
        return 0.0, "pooled_unjudged_as_0" if has_unjudged else "no_relevant_judgments"
    return dcg / idcg, "pooled_unjudged_as_0" if has_unjudged else "fully_judged_ideal"


def compute_precision_at_k(
    ranked_document_ids: Sequence[str],
    qrels: Mapping[str, QrelEntry],
    *,
    k: int,
) -> float | None:
    ranked = dedupe_document_ids(ranked_document_ids)[:k]
    if not ranked:
        return None
    hits = 0
    for doc_id in ranked:
        entry = qrels.get(_match_key(doc_id))
        if entry and is_binary_relevant(entry.grade):
            hits += 1
    return hits / len(ranked)


def compute_recall_at_k(
    ranked_document_ids: Sequence[str],
    qrels: Mapping[str, QrelEntry],
    *,
    k: int,
) -> float | None:
    relevant_ids = {
        key for key, entry in qrels.items() if entry.judgment_state == JUDGMENT_GRADED and is_binary_relevant(entry.grade)
    }
    if not relevant_ids:
        return None
    ranked = dedupe_document_ids(ranked_document_ids)[:k]
    retrieved = {_match_key(doc_id) for doc_id in ranked}
    return len(relevant_ids & retrieved) / len(relevant_ids)


def first_highly_relevant_rank(
    ranked_document_ids: Sequence[str],
    qrels: Mapping[str, QrelEntry],
) -> int | None:
    for index, doc_id in enumerate(dedupe_document_ids(ranked_document_ids), start=1):
        entry = qrels.get(_match_key(doc_id))
        if entry and entry.grade >= 3:
            return index
    return None


def evaluate_graded_query(
    *,
    query_id: str,
    ranked_document_ids: Sequence[str],
    qrel_entries: Sequence[QrelEntry],
    legacy_primary_document_id: str,
    legacy_query: str = "",
    legacy_query_style: str = "mixed",
) -> GradedQueryEvalResult:
    qrels = build_qrel_map(qrel_entries)
    ranked = dedupe_document_ids(ranked_document_ids)
    ndcg5, _ = compute_ndcg_at_k(ranked, qrels, k=5)
    ndcg10, _ = compute_ndcg_at_k(ranked, qrels, k=10)
    ndcg20, _ = compute_ndcg_at_k(ranked, qrels, k=20)
    hr_rank = first_highly_relevant_rank(ranked, qrels)
    legacy = evaluate_ranked_documents(
        query_id=query_id,
        query=legacy_query,
        query_style=legacy_query_style,
        difficulty="medium",
        expected_primary_document_id=legacy_primary_document_id,
        accepted_alternative_document_ids=[],
        hard_negative_document_ids=[],
        hard_negative_evaluable=False,
        hard_negative_blocker=None,
        ranked_document_ids=ranked,
        top_k=max(50, len(ranked)),
        expected_primary_ecli=legacy_primary_document_id,
    )
    judged_relevant = sum(
        1 for entry in qrels.values() if entry.judgment_state == JUDGMENT_GRADED and is_binary_relevant(entry.grade)
    )
    judged_highly = sum(
        1 for entry in qrels.values() if entry.judgment_state == JUDGMENT_GRADED and entry.grade >= 3
    )
    explicit_zero = sum(
        1 for entry in qrels.values() if entry.judgment_state == JUDGMENT_EXPLICIT_NOT_RELEVANT or entry.grade == 0
    )
    ranked_keys = {_match_key(doc_id) for doc_id in ranked}
    unjudged = sum(1 for key in ranked_keys if key not in qrels)
    notes: list[str] = []
    if unjudged:
        notes.append(f"pooled_eval_unjudged_in_ranked_list={unjudged}")
    return GradedQueryEvalResult(
        query_id=query_id,
        ranked_document_ids=ranked,
        qrels=qrels,
        ndcg_at_5=ndcg5,
        ndcg_at_10=ndcg10,
        ndcg_at_20=ndcg20,
        precision_at_5=compute_precision_at_k(ranked, qrels, k=5),
        precision_at_10=compute_precision_at_k(ranked, qrels, k=10),
        recall_at_10=compute_recall_at_k(ranked, qrels, k=10),
        recall_at_20=compute_recall_at_k(ranked, qrels, k=20),
        recall_at_50=compute_recall_at_k(ranked, qrels, k=50),
        mrr_highly_relevant=(1.0 / hr_rank) if hr_rank else 0.0,
        success_at_10_highly_relevant=bool(hr_rank and hr_rank <= 10),
        legacy_primary_rank=legacy.primary_rank,
        legacy_primary_hit_at_1=legacy.hit_at_1,
        legacy_primary_hit_at_3=legacy.hit_at_3,
        legacy_primary_hit_at_10=legacy.hit_at_10,
        legacy_primary_reciprocal_rank=legacy.reciprocal_rank,
        judged_relevant_count=judged_relevant,
        judged_highly_relevant_count=judged_highly,
        explicit_not_relevant_count=explicit_zero,
        unjudged_in_pool_count=unjudged,
        notes=notes,
    )


def aggregate_graded_metrics(results: Sequence[GradedQueryEvalResult]) -> GradedAggregateMetrics:
    if not results:
        return GradedAggregateMetrics()

    def _mean(values: Sequence[float | None]) -> float | None:
        present = [value for value in values if value is not None]
        return (sum(present) / len(present)) if present else None

    return GradedAggregateMetrics(
        total_queries=len(results),
        ndcg_at_5=_mean([row.ndcg_at_5 for row in results]),
        ndcg_at_10=_mean([row.ndcg_at_10 for row in results]),
        ndcg_at_20=_mean([row.ndcg_at_20 for row in results]),
        precision_at_5=_mean([row.precision_at_5 for row in results]),
        precision_at_10=_mean([row.precision_at_10 for row in results]),
        recall_at_10=_mean([row.recall_at_10 for row in results]),
        recall_at_20=_mean([row.recall_at_20 for row in results]),
        recall_at_50=_mean([row.recall_at_50 for row in results]),
        mrr_highly_relevant=_mean([row.mrr_highly_relevant for row in results]),
        success_at_10_highly_relevant=_mean([1.0 if row.success_at_10_highly_relevant else 0.0 for row in results]),
        legacy_primary_hit_at_1=_mean([1.0 if row.legacy_primary_hit_at_1 else 0.0 for row in results]),
        legacy_primary_hit_at_10=_mean([1.0 if row.legacy_primary_hit_at_10 else 0.0 for row in results]),
        legacy_primary_mrr=_mean([row.legacy_primary_reciprocal_rank for row in results]),
    )


def infer_relevance_from_retrieval_rank(*args, **kwargs) -> None:
    """Guardrail: relevance must never be inferred from retrieval rank."""
    raise RuntimeError("Automatic relevance assignment from retrieval rank is forbidden.")
