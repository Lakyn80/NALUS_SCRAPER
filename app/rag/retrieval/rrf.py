from __future__ import annotations

from dataclasses import replace

from app.rag.retrieval.models import RetrievedChunk


def rrf_fuse(
    ranked_lists: list[list[RetrievedChunk]],
    *,
    top_k: int,
    rrf_k: int,
) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    best_chunk: dict[str, RetrievedChunk] = {}
    score_components: dict[str, dict[str, float]] = {}

    for results in ranked_lists:
        for rank, chunk in enumerate(results, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (rrf_k + rank)
            current = best_chunk.get(chunk.id)
            if current is None or chunk.score > current.score:
                best_chunk[chunk.id] = chunk
            source = chunk.source or "unknown"
            score_components.setdefault(chunk.id, {})[source] = chunk.score

    fused: list[RetrievedChunk] = []
    for chunk_id, score in scores.items():
        chunk = best_chunk[chunk_id]
        metadata = dict(chunk.metadata)
        metadata["score_components"] = score_components.get(chunk_id, {})
        metadata["rrf_score"] = score
        fused.append(replace(chunk, score=score, source="hybrid", metadata=metadata))

    fused.sort(key=lambda chunk: (-chunk.score, chunk.id))
    return fused[:top_k]
