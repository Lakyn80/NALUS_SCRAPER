#!/usr/bin/env python3
"""Dense-only baseline for Legal v2 full-corpus golden v2 (no tuning)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.benchmark.case_similarity_eval import (  # noqa: E402
    CaseSimilarityQueryEvalResult,
    RetrievedDocumentScore,
    aggregate_case_similarity_metrics,
    evaluate_ranked_documents,
)
from app.rag.legal_v2.benchmark.case_similarity_golden_v2 import (  # noqa: E402
    DEFAULT_V2_DATASET,
    load_case_similarity_golden_v2_jsonl,
)
from app.rag.legal_v2.identity import normalize_ecli  # noqa: E402
from app.rag.legal_v2.indexing import LEGAL_V2_PROFILE  # noqa: E402
from app.rag.legal_v2.query_spec import build_query_spec_v2  # noqa: E402
from app.rag.legal_v2.retrieve.retriever import (  # noqa: E402
    LegalV2RetrieverConfig,
    build_live_legal_v2_retriever,
)
from app.rag.retrieval.bge_m3_embedder import BgeM3Embedder  # noqa: E402
from app.rag.retrieval.production_profile import ProductionRetrievalConfig  # noqa: E402
from scripts.legal_v2.evaluate_case_similarity_golden_v1 import (  # noqa: E402
    _list_indexed_document_ids,
    _stage1_docs_from_retrieval,
)

DEFAULT_COLLECTION = "nalus_legal_paragraph_chunks_v2_chunk_ab_v8_a_current_full"
DEFAULT_ARTIFACTS = PROJECT_ROOT / "artifacts" / "legal_v2" / "golden_v2_full_corpus"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://nalus-scraper-qdrant-1:6333"))
    parser.add_argument("--qdrant-collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_V2_DATASET)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args(argv)


def _embedder_config(config: LegalV2RetrieverConfig) -> ProductionRetrievalConfig:
    return ProductionRetrievalConfig(
        profile=LEGAL_V2_PROFILE,
        qdrant_collection=config.qdrant_collection,
        bm25_sidecar_path=config.bm25_sidecar_path,
        bm25_index_id=config.bm25_index_id,
        model_path=config.model_path,
        local_files_only=True,
        trust_remote_code=False,
        device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        candidate_multiplier=1,
        min_candidate_count=1,
        max_candidate_count=max(config.dense_candidate_chunks, config.bm25_candidate_chunks),
        lexical_filter_enabled=False,
    )


def _evaluate_split(
    *,
    retriever: Any,
    items: list[Any],
    index_doc_ids: set[str],
    top_k: int,
) -> tuple[list[CaseSimilarityQueryEvalResult], list[dict[str, Any]], list[float]]:
    results: list[CaseSimilarityQueryEvalResult] = []
    per_query: list[dict[str, Any]] = []
    latencies: list[float] = []
    for item in items:
        primary = normalize_ecli(item.expected_primary_ecli or item.expected_primary_document_id)
        started = time.perf_counter()
        retrieval = retriever.retrieve(build_query_spec_v2(item.query_text))
        docs = _stage1_docs_from_retrieval(
            retrieval.documents,
            limit=top_k,
            evidence_limit=3,
            prefer_chunk_evidence=True,
        )
        latencies.append((time.perf_counter() - started) * 1000.0)
        ranked_eclis: list[str] = []
        retrieved_results: list[RetrievedDocumentScore] = []
        for doc in docs:
            ecli_n = normalize_ecli(getattr(doc, "ecli", None) or getattr(doc, "document_id", "") or "")
            if not ecli_n or ecli_n in ranked_eclis:
                continue
            ranked_eclis.append(ecli_n)
            retrieved_results.append(
                RetrievedDocumentScore(
                    rank=len(retrieved_results) + 1,
                    document_id=ecli_n,
                    ecli=ecli_n,
                    canonical_document_id=ecli_n,
                    score=getattr(doc, "score", None),
                    fusion_score=getattr(doc, "rrf_score", None),
                )
            )
        eval_result = evaluate_ranked_documents(
            query_id=item.query_id,
            query=item.query_text,
            query_style=item.query_type,
            difficulty="medium",
            expected_primary_document_id=primary,
            accepted_alternative_document_ids=[
                normalize_ecli(value) for value in item.expected_relevant_document_ids if value
            ],
            hard_negative_document_ids=[],
            hard_negative_evaluable=False,
            hard_negative_blocker=None,
            ranked_document_ids=ranked_eclis,
            retrieved_results=retrieved_results,
            corpus_compatible=primary in index_doc_ids,
            top_k=min(10, top_k),
            expected_primary_source_document_id=item.expected_primary_document_id,
            expected_primary_ecli=primary,
        )
        results.append(eval_result)
        per_query.append(
            {
                "query_id": item.query_id,
                "split": item.split,
                "primary_rank": eval_result.primary_rank,
                "hit_at_1": eval_result.hit_at_1,
                "hit_at_3": eval_result.hit_at_3,
                "hit_at_5": eval_result.hit_at_5,
                "hit_at_10": eval_result.hit_at_10,
                "reciprocal_rank": eval_result.reciprocal_rank,
                "corpus_compatible": eval_result.corpus_compatible,
                "failure_type": eval_result.failure_type,
                "latency_ms": round(latencies[-1], 3),
            }
        )
    return results, per_query, latencies


def _split_metrics(results: list[CaseSimilarityQueryEvalResult]) -> dict[str, Any]:
    metrics = aggregate_case_similarity_metrics(results, missing_hard_negative_document_count=0)
    return metrics.model_dump()


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return round(ordered[index], 3)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    items = load_case_similarity_golden_v2_jsonl(args.benchmark)
    try:
        from qdrant_client import QdrantClient
    except ModuleNotFoundError as exc:
        raise SystemExit("Run inside API container.") from exc

    client = QdrantClient(url=args.qdrant_url, timeout=120)
    index_doc_ids = _list_indexed_document_ids(client, args.qdrant_collection)
    config = LegalV2RetrieverConfig(
        qdrant_collection=args.qdrant_collection,
        bm25_sidecar_path=Path(
            os.getenv(
                "NALUS_LEGAL_V2_FAST_BM25_SIDECAR_PATH",
                "/app/storage/rag/bm25/nalus_legal_paragraph_bm25_v2_chunk_ab_v8_a_current_full.sqlite",
            )
        ),
        bm25_index_id=os.getenv(
            "NALUS_LEGAL_V2_FAST_BM25_INDEX_ID",
            "nalus_legal_paragraph_bm25_v2_chunk_ab_v8_a_current_full",
        ),
        bm25_enabled=False,
        dense_candidate_chunks=80,
        fused_candidate_chunks=120,
        candidate_documents=50,
        model_path=os.getenv(
            "EMBEDDING_MODEL_NAME",
            "/root/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/"
            "5617a9f61b028005a4858fdac845db406aefb181",
        ),
    )
    retriever = build_live_legal_v2_retriever(client, BgeM3Embedder(_embedder_config(config)), config)

    dev_items = [item for item in items if item.split == "dev"]
    test_items = [item for item in items if item.split == "test"]
    dev_results, dev_per_query, dev_lat = _evaluate_split(
        retriever=retriever, items=dev_items, index_doc_ids=index_doc_ids, top_k=args.top_k
    )
    test_results, test_per_query, test_lat = _evaluate_split(
        retriever=retriever, items=test_items, index_doc_ids=index_doc_ids, top_k=args.top_k
    )

    payload = {
        "mode": "dense_only",
        "collection": args.qdrant_collection,
        "benchmark": str(args.benchmark),
        "rrf_k": LEGAL_V2_PROFILE.rrf_k,
        "dev": {
            "count": len(dev_items),
            "metrics": _split_metrics(dev_results),
            "latency_ms": {
                "p50": _percentile(dev_lat, 0.5),
                "p95": _percentile(dev_lat, 0.95),
                "mean": round(sum(dev_lat) / len(dev_lat), 3) if dev_lat else None,
            },
        },
        "test": {
            "count": len(test_items),
            "metrics": _split_metrics(test_results),
            "latency_ms": {
                "p50": _percentile(test_lat, 0.5),
                "p95": _percentile(test_lat, 0.95),
                "mean": round(sum(test_lat) / len(test_lat), 3) if test_lat else None,
            },
        },
    }

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (args.artifacts_dir / "dense_baseline.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    per_query_path = args.artifacts_dir / "per_query_dense.jsonl"
    per_query_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in dev_per_query + test_per_query) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
