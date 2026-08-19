#!/usr/bin/env python3
"""Build Legal v2 golden v3 graded multi-relevance benchmark from frozen v2 input."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.benchmark.case_similarity_golden_v2 import (  # noqa: E402
    DEFAULT_V2_DATASET,
    load_case_similarity_golden_v2_jsonl,
)
from app.rag.legal_v2.benchmark.case_similarity_golden_v3 import (  # noqa: E402
    BENCHMARK_SCOPE,
    DEFAULT_V3_DATASET,
    GRADE_LABELS,
    CaseSimilarityGoldenV3Item,
    validate_v3_split_counts,
    write_case_similarity_golden_v3_jsonl,
)
from app.rag.legal_v2.benchmark.case_similarity_query_audit import (  # noqa: E402
    audit_and_rewrite_v2_query,
    classify_query,
)
from app.rag.legal_v2.identity import is_valid_ecli, normalize_ecli  # noqa: E402
from app.rag.legal_v2.indexing import LEGAL_V2_PROFILE  # noqa: E402
from app.rag.legal_v2.query_spec import build_query_spec_v2  # noqa: E402
from app.rag.legal_v2.retrieve.retriever import (  # noqa: E402
    LegalV2RetrieverConfig,
    build_live_legal_v2_retriever,
)
from app.rag.retrieval.bge_m3_embedder import BgeM3Embedder  # noqa: E402
from app.rag.retrieval.production_profile import ProductionRetrievalConfig  # noqa: E402
from scripts.legal_v2.prepare_case_similarity_golden_v2_human_review import (  # noqa: E402
    _fetch_target_context,
    _verify_corpus_scope,
)

DEFAULT_COLLECTION = "nalus_legal_paragraph_chunks_v2_chunk_ab_v8_a_current_full"
DEFAULT_ARTIFACTS = PROJECT_ROOT / "artifacts" / "legal_v2" / "golden_v3_graded"
POOL_DEPTH = 20


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-benchmark", type=Path, default=DEFAULT_V2_DATASET)
    parser.add_argument("--v3-benchmark", type=Path, default=DEFAULT_V3_DATASET)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://nalus-scraper-qdrant-1:6333"))
    parser.add_argument("--qdrant-collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--skip-pooling", action="store_true", help="Skip retrieval candidate pooling.")
    parser.add_argument("--pool-depth", type=int, default=POOL_DEPTH)
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


def _document_ranks_from_chunks(chunks: list[Any], *, limit: int) -> list[tuple[str, int]]:
    ranks: list[tuple[str, int]] = []
    seen: set[str] = set()
    for index, chunk in enumerate(chunks, start=1):
        meta = getattr(chunk, "metadata", None) or getattr(chunk, "payload", None) or {}
        if not isinstance(meta, dict):
            meta = {}
        raw = meta.get("ecli") or meta.get("document_id") or getattr(chunk, "document_id", "") or ""
        ecli = normalize_ecli(str(raw)) if is_valid_ecli(str(raw)) else ""
        if not ecli or ecli in seen:
            continue
        seen.add(ecli)
        ranks.append((ecli, index))
        if len(ranks) >= limit:
            break
    return ranks


def _retriever_config(collection: str) -> LegalV2RetrieverConfig:
    return LegalV2RetrieverConfig(
        qdrant_collection=collection,
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
        model_path=os.getenv(
            "EMBEDDING_MODEL_NAME",
            "/root/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181",
        ),
        dense_candidate_chunks=80,
        bm25_candidate_chunks=80,
        fused_candidate_chunks=120,
        candidate_documents=50,
        returned_verified_documents=10,
        bm25_enabled=True,
    )


def _build_v3_items(v2_items: list[Any], client: Any, collection: str) -> tuple[list[CaseSimilarityGoldenV3Item], list[dict[str, Any]]]:
    v3_items: list[CaseSimilarityGoldenV3Item] = []
    query_reviews: list[dict[str, Any]] = []
    for item in v2_items:
        ecli = normalize_ecli(item.expected_primary_ecli or item.expected_primary_document_id)
        ctx = _fetch_target_context(client, collection, ecli) if client else {}
        audit = audit_and_rewrite_v2_query(item, target_reasoning=ctx.get("target_reasoning_excerpt", ""))
        review_status = audit.status
        if review_status == "edited":
            review_status = "needs_edit"
        query_reviews.append(
            {
                "query_id": item.query_id,
                "split": item.split,
                "legacy_v2_query_text": item.query_text,
                "query_text": audit.rewritten_text,
                "review_status": "approved" if audit.status == "approved" else ("edited" if audit.rewrite_applied else review_status),
                "audit_flags": audit.flags,
                "reviewer_notes": "",
                "revised_query_text": audit.rewritten_text if audit.rewrite_applied else "",
            }
        )
        v3_items.append(
            CaseSimilarityGoldenV3Item(
                query_id=item.query_id,
                split=item.split,
                query_text=audit.rewritten_text,
                query_type=item.query_type,
                legal_area=item.legal_area,
                legacy_primary_document_id=ecli,
                legacy_primary_ecli=ecli,
                expected_court=item.expected_court,
                expected_source=item.expected_source,
                expected_year=item.expected_year,
                document_type=item.document_type,
                case_reference=item.case_reference,
                query_review_status="approved" if audit.status == "approved" else "edited",
                legacy_v2_query_text=item.query_text,
                relevance_judgments=[],
            )
        )
    validate_v3_split_counts(v3_items)
    return v3_items, query_reviews


def _pool_candidates_for_query(
    retriever: Any,
    *,
    query_text: str,
    legacy_ecli: str,
    depth: int,
) -> dict[str, Any]:
    result = retriever.retrieve(build_query_spec_v2(query_text))
    dense = _document_ranks_from_chunks(result.dense_results, limit=depth)
    bm25 = _document_ranks_from_chunks(result.bm25_results, limit=depth)
    hybrid = _document_ranks_from_chunks(result.fused_results, limit=depth)
    dense_map = {doc: rank for doc, rank in dense}
    bm25_map = {doc: rank for doc, rank in bm25}
    hybrid_map = {doc: rank for doc, rank in hybrid}
    all_docs: list[str] = []
    seen: set[str] = set()
    for doc, _ in dense + bm25 + hybrid:
        if doc not in seen:
            seen.add(doc)
            all_docs.append(doc)
    if legacy_ecli not in seen:
        all_docs.insert(0, legacy_ecli)
        seen.add(legacy_ecli)
    return {
        "candidates": all_docs,
        "dense_map": dense_map,
        "bm25_map": bm25_map,
        "hybrid_map": hybrid_map,
    }


def _write_query_review_md(path: Path, rows: list[dict[str, Any]]) -> None:
    chunks = [
        "# Legal v2 Golden v3 — Query Review",
        "",
        f"**Benchmark scope:** `{BENCHMARK_SCOPE}`",
        "",
    ]
    for row in rows:
        chunks.extend(
            [
                f"## {row['query_id']} — {row['split'].upper()}",
                "",
                "**Rewritten query:**",
                row["query_text"],
                "",
                "**Legacy v2 query:**",
                row["legacy_v2_query_text"][:500] + ("..." if len(row["legacy_v2_query_text"]) > 500 else ""),
                "",
                f"**Review status:** {row['review_status']}",
                f"**Flags:** {', '.join(row.get('audit_flags') or []) or 'none'}",
                "",
                "**Reviewer:**",
                "- [ ] approved",
                "- [ ] needs edit",
                "- [ ] rejected",
                "",
                "**Notes:**",
                "",
                "---",
                "",
            ]
        )
    path.write_text("\n".join(chunks), encoding="utf-8")


def _candidate_metadata(client: Any, collection: str, ecli: str) -> dict[str, Any]:
    ctx = _fetch_target_context(client, collection, ecli)
    from qdrant_client.http import models as qmodels

    filt = qmodels.Filter(must=[qmodels.FieldCondition(key="ecli", match=qmodels.MatchValue(value=ecli))])
    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=filt,
        limit=8,
        with_payload=["case_reference", "document_type", "decision_date", "court", "source"],
        with_vectors=False,
    )
    payload = (points[0].payload if points else {}) or {}
    return {
        "document_id": ecli,
        "ecli": ecli,
        "case_reference": payload.get("case_reference"),
        "court": payload.get("court") or "constitutional_court",
        "decision_date": payload.get("decision_date"),
        "document_type": payload.get("document_type"),
        "candidate_summary": ctx.get("target_decision_summary", ""),
        "central_legal_issue": ctx.get("central_legal_issue", ""),
        "reasoning_excerpt": ctx.get("target_reasoning_excerpt", ""),
    }


def _write_relevance_review_md(path: Path, queue_rows: list[dict[str, Any]]) -> None:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in queue_rows:
        by_query[row["query_id"]].append(row)
    chunks = [
        "# Legal v2 Golden v3 — Relevance Review",
        "",
        "Dense/BM25/hybrid ranks are discovery metadata only — do not use as relevance oracle.",
        "",
    ]
    for query_id in sorted(by_query):
        rows = by_query[query_id]
        first = rows[0]
        chunks.extend(
            [
                f"# {query_id} — {first['split'].upper()}",
                "",
                "## Query",
                first["query_text"],
                "",
            ]
        )
        for index, row in enumerate(rows, start=1):
            chunks.extend(
                [
                    f"### Candidate {index}",
                    f"ECLI: {row.get('ecli') or row.get('document_id')}",
                    f"Court: {row.get('court')}",
                    f"Date: {row.get('decision_date') or row.get('expected_year') or '?'}",
                    "",
                    "Why it may be relevant:",
                    row.get("candidate_summary") or row.get("central_legal_issue") or "",
                    "",
                    "Reasoning excerpt:",
                    row.get("reasoning_excerpt") or "",
                    "",
                    "Discovered by:",
                    f"- dense rank: {row.get('dense_rank')}",
                    f"- BM25 rank: {row.get('bm25_rank')}",
                    f"- hybrid rank: {row.get('hybrid_rank')}",
                    "",
                    "Manual judgment:",
                    "- [ ] 3 HIGHLY_RELEVANT",
                    "- [ ] 2 RELEVANT",
                    "- [ ] 1 PARTIALLY_RELEVANT",
                    "- [ ] 0 NOT_RELEVANT",
                    "",
                    "Notes:",
                    "",
                    "---",
                    "",
                ]
            )
    path.write_text("\n".join(chunks), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    v2_items = load_case_similarity_golden_v2_jsonl(args.v2_benchmark)

    client = None
    retriever = None
    corpus_scope: dict[str, Any] = {"benchmark_scope": BENCHMARK_SCOPE}
    if not args.skip_pooling:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=args.qdrant_url, timeout=120)
        corpus_scope = _verify_corpus_scope(client, args.qdrant_collection)
        config = _retriever_config(args.qdrant_collection)
        retriever = build_live_legal_v2_retriever(
            QdrantClient(url=args.qdrant_url, timeout=120),
            BgeM3Embedder(_embedder_config(config)),
            config,
        )

    v3_items, query_reviews = _build_v3_items(v2_items, client, args.qdrant_collection)
    write_case_similarity_golden_v3_jsonl(args.v3_benchmark, v3_items)

    query_review_path = args.artifacts_dir / "query_review.jsonl"
    with query_review_path.open("w", encoding="utf-8") as handle:
        for row in query_reviews:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_query_review_md(args.artifacts_dir / "QUERY_REVIEW.md", query_reviews)

    candidate_pool_rows: list[dict[str, Any]] = []
    relevance_queue_rows: list[dict[str, Any]] = []
    overlap_stats = Counter()
    if retriever and client:
        for item in v3_items:
            pool = _pool_candidates_for_query(
                retriever,
                query_text=item.query_text,
                legacy_ecli=item.legacy_primary_ecli or item.legacy_primary_document_id,
                depth=args.pool_depth,
            )
            dense_only = 0
            bm25_only = 0
            hybrid_only = 0
            for doc in pool["candidates"]:
                in_d = doc in pool["dense_map"]
                in_b = doc in pool["bm25_map"]
                in_h = doc in pool["hybrid_map"]
                if in_d and not in_b and not in_h:
                    dense_only += 1
                if in_b and not in_d and not in_h:
                    bm25_only += 1
                if in_h and not in_d and not in_b:
                    hybrid_only += 1
            candidate_pool_rows.append(
                {
                    "query_id": item.query_id,
                    "split": item.split,
                    "query_text": item.query_text,
                    "legacy_primary_ecli": item.legacy_primary_ecli,
                    "candidates": pool["candidates"],
                    "dense_ranks": pool["dense_map"],
                    "bm25_ranks": pool["bm25_map"],
                    "hybrid_ranks": pool["hybrid_map"],
                    "candidate_count": len(pool["candidates"]),
                }
            )
            for doc in pool["candidates"]:
                meta = _candidate_metadata(client, args.qdrant_collection, doc)
                relevance_queue_rows.append(
                    {
                        "query_id": item.query_id,
                        "split": item.split,
                        "query_text": item.query_text,
                        "document_id": doc,
                        "ecli": doc,
                        "case_reference": meta.get("case_reference"),
                        "court": meta.get("court"),
                        "decision_date": meta.get("decision_date"),
                        "document_type": meta.get("document_type"),
                        "candidate_summary": meta.get("candidate_summary"),
                        "central_legal_issue": meta.get("central_legal_issue"),
                        "reasoning_excerpt": meta.get("reasoning_excerpt"),
                        "found_by_dense": doc in pool["dense_map"],
                        "dense_rank": pool["dense_map"].get(doc),
                        "found_by_bm25": doc in pool["bm25_map"],
                        "bm25_rank": pool["bm25_map"].get(doc),
                        "found_by_hybrid": doc in pool["hybrid_map"],
                        "hybrid_rank": pool["hybrid_map"].get(doc),
                        "relevance_grade": None,
                        "relevance_label": None,
                        "reviewer_notes": "",
                        "review_status": "pending",
                    }
                )
            overlap_stats["dense_only"] += dense_only
            overlap_stats["bm25_only"] += bm25_only
            overlap_stats["hybrid_only"] += hybrid_only

        with (args.artifacts_dir / "candidate_pool.jsonl").open("w", encoding="utf-8") as handle:
            for row in candidate_pool_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (args.artifacts_dir / "relevance_review_queue.jsonl").open("w", encoding="utf-8") as handle:
            for row in relevance_queue_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        _write_relevance_review_md(args.artifacts_dir / "RELEVANCE_REVIEW.md", relevance_queue_rows)

    qrels_template = args.artifacts_dir / "qrels_template.jsonl"
    qrels_template.write_text(
        "# Empty template — only human-reviewed judgments may enter frozen qrels.\n",
        encoding="utf-8",
    )

    evaluator_config = {
        "schema_version": "nalus-case-similarity-graded-eval.v1",
        "binary_relevance_threshold": 2,
        "grades": dict(GRADE_LABELS),
        "unjudged_handling": {
            "in_pooled_ndcg": "treat_unjudged_as_grade_0_with_report_flag",
            "explicit_grade_0": "judged_not_relevant",
            "unjudged": "not_assumed_irrelevant_outside_pool",
        },
        "primary_metrics": ["ndcg@5", "ndcg@10", "ndcg@20"],
        "secondary_metrics": [
            "precision@5",
            "precision@10",
            "recall@10",
            "recall@20",
            "recall@50",
            "mrr_highly_relevant",
            "success@10_highly_relevant",
        ],
        "diagnostic_metrics": [
            "legacy_primary_rank",
            "legacy_primary_hit@1",
            "legacy_primary_hit@10",
            "legacy_primary_mrr",
        ],
    }
    (args.artifacts_dir / "evaluator_config.json").write_text(
        json.dumps(evaluator_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    approved = sum(1 for row in query_reviews if row["review_status"] == "approved")
    edited = sum(1 for row in query_reviews if row["review_status"] == "edited")
    rejected = sum(1 for row in query_reviews if row["review_status"] == "rejected")
    total_pairs = len(relevance_queue_rows)
    avg_candidates = (
        sum(row["candidate_count"] for row in candidate_pool_rows) / len(candidate_pool_rows)
        if candidate_pool_rows
        else 0
    )
    if rejected > 0 or not relevance_queue_rows:
        status = "NEEDS_QUERY_REVIEW"
    else:
        status = "READY_FOR_RELEVANCE_REVIEW"

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_scope": BENCHMARK_SCOPE,
        "corpus_scope": corpus_scope,
        "query_audit": {
            "total": len(query_reviews),
            "approved": approved,
            "edited": edited,
            "rejected": rejected,
            "dev": sum(1 for row in query_reviews if row["split"] == "dev"),
            "test": sum(1 for row in query_reviews if row["split"] == "test"),
        },
        "candidate_pool": {
            "queries": len(candidate_pool_rows),
            "total_unique_pairs": total_pairs,
            "avg_candidates_per_query": round(avg_candidates, 2),
            "dense_only_candidates": overlap_stats.get("dense_only", 0),
            "bm25_only_candidates": overlap_stats.get("bm25_only", 0),
            "hybrid_only_candidates": overlap_stats.get("hybrid_only", 0),
        },
        "relevance_review": {
            "reviewed_judgments": 0,
            "pending_judgments": total_pairs,
            "queries_with_grade_3": 0,
            "queries_with_no_grade_gte_2": len(v3_items),
        },
        "sanity_check_query": "nalus-cs-v2-001",
        "status": status,
    }
    (args.artifacts_dir / "REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# Legal v2 Golden v3 — Graded Multi-Relevance",
        "",
        "## Query Audit",
        f"- total: {len(query_reviews)}",
        f"- approved: {approved}",
        f"- edited: {edited}",
        f"- rejected/replaced: {rejected}",
        f"- DEV: {report['query_audit']['dev']}",
        f"- TEST: {report['query_audit']['test']}",
        "",
        "## Candidate Pool",
        f"- queries: {len(candidate_pool_rows)}",
        f"- total unique query-document pairs: {total_pairs}",
        f"- avg candidates/query: {avg_candidates:.1f}" if candidate_pool_rows else "- avg candidates/query: n/a (pooling skipped)",
        f"- dense-only candidates: {overlap_stats.get('dense_only', 0)}",
        f"- BM25-only candidates: {overlap_stats.get('bm25_only', 0)}",
        f"- hybrid-only candidates: {overlap_stats.get('hybrid_only', 0)}",
        "",
        "## Relevance Model",
        "- grades: 0 NOT_RELEVANT, 1 PARTIALLY_RELEVANT, 2 RELEVANT, 3 HIGHLY_RELEVANT",
        "- binary relevance threshold: grade >= 2",
        "- unjudged handling: pooled nDCG treats unjudged as 0 with flag; unjudged != explicit grade 0",
        "",
        "## Evaluator",
        "Supported metrics: nDCG@5/10/20, Precision@5/10, Recall@10/20/50, MRR_highly_relevant, Success@10_highly_relevant, legacy primary diagnostics",
        "",
        f"## Final Status: **{status}**",
    ]
    (args.artifacts_dir / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
