"""Document-level score aggregation and deterministic ordering."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from app.rag.legal_v2.rerank.models import (
    RerankCandidate,
    RerankScore,
    RerankedDocument,
)


def aggregate_max_passage_scores(
    candidates: Sequence[RerankCandidate],
    scores: Sequence[RerankScore],
) -> tuple[RerankedDocument, ...]:
    """Aggregate passage CE scores with policy=max; deterministic ties."""
    by_ecli: dict[str, list[RerankScore]] = defaultdict(list)
    for score in scores:
        by_ecli[score.ecli].append(score)

    ranked_rows: list[tuple[float, int, str, RerankCandidate, tuple[RerankScore, ...]]] = []
    for candidate in candidates:
        passage_scores = tuple(by_ecli.get(candidate.ecli, ()))
        if passage_scores:
            ce_score = max(item.score for item in passage_scores)
        else:
            # No scorable passages: keep at bottom relative to scored docs,
            # but still deterministic via Stage1 rank.
            ce_score = float("-inf")
        ranked_rows.append(
            (
                ce_score,
                candidate.stage1_rank,
                candidate.ecli,
                candidate,
                passage_scores,
            )
        )

    # CE score desc, Stage1 rank asc, ECLI lex
    ranked_rows.sort(key=lambda row: (-row[0], row[1], row[2]))

    documents: list[RerankedDocument] = []
    for index, (ce_score, _s1_rank, _ecli, candidate, passage_scores) in enumerate(
        ranked_rows, start=1
    ):
        finite_score = 0.0 if ce_score == float("-inf") else float(ce_score)
        documents.append(
            RerankedDocument(
                ecli=candidate.ecli,
                stage1_rank=candidate.stage1_rank,
                stage1_score=candidate.stage1_score,
                ce_rank=index,
                ce_score=finite_score,
                passage_scores=passage_scores,
                dense_rank=candidate.dense_rank,
                bm25_rank=candidate.bm25_rank,
                rrf_score=candidate.rrf_score,
                metadata=dict(candidate.metadata),
            )
        )
    return tuple(documents)
