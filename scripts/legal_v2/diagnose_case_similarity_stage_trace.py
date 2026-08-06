#!/usr/bin/env python3
"""Stage-trace diagnostic for one case-similarity golden query (read-only).

Does not change retrieval knobs, golden data, Qdrant, or BM25. Prints where the
expected ECLI appears after QuerySpec → dense → BM25 → RRF → document aggregation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.benchmark.case_similarity_golden import (  # noqa: E402
    DEFAULT_PILOT_DATASET,
    load_case_similarity_golden_jsonl,
)
from app.rag.legal_v2.identity import ecli_key, is_valid_ecli, normalize_ecli  # noqa: E402
from app.rag.legal_v2.query_spec import build_query_spec_v2  # noqa: E402
from app.rag.legal_v2.retriever import (  # noqa: E402
    LegalV2RetrieverConfig,
    build_live_legal_v2_retriever,
    _document_id,
    _retrieval_query,
)
from app.rag.retrieval.bge_m3_embedder import BgeM3Embedder  # noqa: E402
from app.rag.retrieval.production_profile import ProductionRetrievalConfig  # noqa: E402
from app.rag.legal_v2.indexing import LEGAL_V2_PROFILE  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-id", default="nalus-cs-pilot-004")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_PILOT_DATASET)
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://qdrant:6333"))
    parser.add_argument(
        "--qdrant-collection",
        default=os.getenv(
            "NALUS_LEGAL_V2_QDRANT_COLLECTION",
            "nalus_legal_paragraph_chunks_v2_pilot_600",
        ),
    )
    parser.add_argument(
        "--bm25-sidecar-path",
        type=Path,
        default=Path(
            os.getenv(
                "NALUS_LEGAL_V2_BM25_SIDECAR_PATH",
                "/app/storage/rag/bm25/nalus_legal_paragraph_bm25_v2_pilot_600.sqlite",
            )
        ),
    )
    parser.add_argument(
        "--bm25-index-id",
        default=os.getenv(
            "NALUS_LEGAL_V2_BM25_INDEX_ID",
            "nalus_legal_paragraph_bm25_v2_pilot_600",
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON dump path under artifacts/.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    items = {item.benchmark_id: item for item in load_case_similarity_golden_jsonl(args.benchmark)}
    item = items.get(args.benchmark_id)
    if item is None:
        raise SystemExit(f"benchmark id not found: {args.benchmark_id}")

    expected = normalize_ecli(item.expected_primary_ecli or "")
    if not expected or not is_valid_ecli(expected):
        raise SystemExit(f"benchmark row lacks verified ECLI: {args.benchmark_id}")
    expected_key = ecli_key(expected)

    from qdrant_client import QdrantClient  # type: ignore[import-not-found]

    query_spec = build_query_spec_v2(item.query)
    retrieval_query = _retrieval_query(query_spec)

    config = LegalV2RetrieverConfig(
        qdrant_collection=args.qdrant_collection,
        bm25_sidecar_path=args.bm25_sidecar_path,
        bm25_index_id=args.bm25_index_id,
        dense_candidate_chunks=80,
        bm25_candidate_chunks=80,
        fused_candidate_chunks=120,
        candidate_documents=40,
        model_path=os.getenv(
            "EMBEDDING_MODEL_NAME",
            "/root/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181",
        ),
    )
    embedder = BgeM3Embedder(
        ProductionRetrievalConfig(
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
    )
    client = QdrantClient(url=args.qdrant_url, timeout=120)
    retriever = build_live_legal_v2_retriever(client, embedder, config)
    result = retriever.retrieve(query_spec)

    dense_doc = _document_rank_map(result.dense_results)
    bm25_doc = _document_rank_map(result.bm25_results)
    fused_doc = _document_rank_map(result.fused_results)
    agg_doc = {
        ecli_key(doc.document_id) if is_valid_ecli(doc.document_id) else doc.document_id: {
            "rank": index,
            "document_id": doc.document_id,
            "score": doc.score,
            "rrf_score": doc.rrf_score,
            "dense_rank": doc.dense_rank,
            "bm25_rank": doc.bm25_rank,
            "chunk_ids": list(doc.chunk_ids),
        }
        for index, doc in enumerate(result.documents, start=1)
    }

    marker_check = _marker_presence(item.query, query_spec, retrieval_query)

    payload: dict[str, Any] = {
        "benchmark_id": item.benchmark_id,
        "expected_ecli": expected,
        "original_query": item.query,
        "retrieval_query_used": retrieval_query,
        "query_spec": {
            "retrieval_queries": list(query_spec.retrieval_queries),
            "legal_concepts": list(
                (query_spec.structured_query or {}).get("legal_concepts")
                or (query_spec.structured_query or {}).get("concepts")
                or []
            ),
            "hard_constraints": [_constraint(c) for c in query_spec.hard_constraints],
            "soft_constraints": [_constraint(c) for c in query_spec.soft_constraints],
            "negative_constraints": [_constraint(c) for c in query_spec.negative_constraints],
            "relations": [_obj(r) for r in query_spec.relations],
            "entities": [_obj(e) for e in query_spec.entities],
            "events": [_obj(e) for e in query_spec.events],
            "procedural_posture": list(query_spec.procedural_posture),
            "decision_outcome": list(query_spec.decision_outcome),
            "document_types": list(query_spec.document_types),
            "courts": list(query_spec.courts),
            "structured_query": dict(query_spec.structured_query or {}),
        },
        "marker_check": marker_check,
        "stage_presence": {
            "dense_chunk_pool": len(result.dense_results),
            "bm25_chunk_pool": len(result.bm25_results),
            "rrf_chunk_pool": len(result.fused_results),
            "aggregated_documents": len(result.documents),
            "expected_in_dense_docs": expected_key in dense_doc,
            "expected_dense_doc_rank": dense_doc.get(expected_key, {}).get("best_rank"),
            "expected_dense_best_chunk_rank": dense_doc.get(expected_key, {}).get("best_chunk_rank"),
            "expected_in_bm25_docs": expected_key in bm25_doc,
            "expected_bm25_doc_rank": bm25_doc.get(expected_key, {}).get("best_rank"),
            "expected_bm25_best_chunk_rank": bm25_doc.get(expected_key, {}).get("best_chunk_rank"),
            "expected_in_rrf_docs": expected_key in fused_doc,
            "expected_rrf_doc_rank": fused_doc.get(expected_key, {}).get("best_rank"),
            "expected_rrf_best_chunk_rank": fused_doc.get(expected_key, {}).get("best_chunk_rank"),
            "expected_in_aggregated_docs": expected_key in agg_doc,
            "expected_aggregated_doc_rank": (agg_doc.get(expected_key) or {}).get("rank"),
            "expected_aggregated_detail": agg_doc.get(expected_key),
        },
        "top20": {
            "dense_docs": _top_docs(dense_doc, 20),
            "bm25_docs": _top_docs(bm25_doc, 20),
            "rrf_docs": _top_docs(fused_doc, 20),
            "aggregated_docs": [
                {
                    "rank": i,
                    "document_id": doc.document_id,
                    "score": doc.score,
                    "rrf_score": doc.rrf_score,
                }
                for i, doc in enumerate(result.documents[:20], start=1)
            ],
        },
        "drop_point": _drop_point(
            dense_doc=dense_doc,
            bm25_doc=bm25_doc,
            fused_doc=fused_doc,
            agg_doc=agg_doc,
            expected_key=expected_key,
        ),
        "diagnostics": dict(result.diagnostics),
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    out = args.output or (
        PROJECT_ROOT
        / "artifacts"
        / "legal_v2"
        / "case_similarity_golden_v1_baseline"
        / f"diagnose_{args.benchmark_id}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\nwrote={out}", file=sys.stderr)
    return 0


def _document_rank_map(chunks: list[Any]) -> dict[str, dict[str, Any]]:
    by_doc: dict[str, dict[str, Any]] = {}
    first_seen_rank = 0
    seen_docs: set[str] = set()
    for chunk_rank, chunk in enumerate(chunks, start=1):
        document_id = _document_id(chunk)
        if not document_id:
            continue
        key = ecli_key(document_id) if is_valid_ecli(document_id) else document_id
        if key not in seen_docs:
            first_seen_rank += 1
            seen_docs.add(key)
            by_doc[key] = {
                "best_rank": first_seen_rank,
                "best_chunk_rank": chunk_rank,
                "document_id": document_id,
                "chunk_hits": [],
            }
        by_doc[key]["chunk_hits"].append(
            {
                "chunk_rank": chunk_rank,
                "chunk_id": chunk.id,
                "score": chunk.score,
            }
        )
        # keep earliest chunk rank
        if chunk_rank < by_doc[key]["best_chunk_rank"]:
            by_doc[key]["best_chunk_rank"] = chunk_rank
    return by_doc


def _top_docs(doc_map: dict[str, dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    rows = sorted(doc_map.values(), key=lambda row: int(row["best_rank"]))
    return [
        {
            "rank": row["best_rank"],
            "document_id": row["document_id"],
            "best_chunk_rank": row["best_chunk_rank"],
            "chunk_hit_count": len(row["chunk_hits"]),
        }
        for row in rows[:limit]
    ]


def _drop_point(
    *,
    dense_doc: dict[str, Any],
    bm25_doc: dict[str, Any],
    fused_doc: dict[str, Any],
    agg_doc: dict[str, Any],
    expected_key: str,
) -> str:
    in_dense = expected_key in dense_doc
    in_bm25 = expected_key in bm25_doc
    in_rrf = expected_key in fused_doc
    in_agg = expected_key in agg_doc
    if not in_dense and in_bm25 and not in_rrf:
        bm25_rank = int(bm25_doc[expected_key]["best_rank"])
        chunk_rank = int(bm25_doc[expected_key]["best_chunk_rank"])
        return (
            "present_in_bm25_only_but_dropped_by_rrf"
            f"(bm25_doc_rank={bm25_rank}, bm25_chunk_rank={chunk_rank})"
        )
    if not in_dense and not in_bm25:
        return "missing_from_dense_and_bm25_candidate_pools"
    if not in_dense and in_bm25:
        return "missing_from_dense_only_present_in_bm25"
    if in_dense and not in_bm25:
        return "missing_from_bm25_only_present_in_dense"
    if (in_dense or in_bm25) and not in_rrf:
        return "present_in_candidates_but_dropped_by_rrf"
    if in_rrf and not in_agg:
        return "present_after_rrf_but_dropped_by_document_aggregation"
    if in_agg:
        rank = int(agg_doc[expected_key]["rank"])
        if rank > 10:
            return f"survived_aggregation_but_outside_top10_rank_{rank}"
        return f"survived_all_stages_rank_{rank}"
    return "unknown"


def _marker_presence(original: str, query_spec: Any, retrieval_query: str) -> dict[str, Any]:
    haystacks = {
        "original_query": original.lower(),
        "retrieval_query_used": retrieval_query.lower(),
        "retrieval_queries_joined": " ".join(query_spec.retrieval_queries).lower(),
        "hard_soft_constraint_values": " ".join(
            str(c.normalized_value or c.value or "")
            for c in list(query_spec.hard_constraints) + list(query_spec.soft_constraints)
        ).lower(),
        "legal_concepts": json.dumps(
            (query_spec.structured_query or {}).get("legal_concepts")
            or (query_spec.structured_query or {}).get("concepts")
            or [],
            ensure_ascii=False,
        ).lower(),
    }
    markers = {
        "child_removal_ospod": ["odebr", "dět", "deti", "ospod", "sociál", "social", "péč", "pec"],
        "no_lawyer_representation": ["advokát", "advokat", "zastoup", "bez advok"],
        "reasoning_defects": ["odůvodněn", "oduvodnen", "vad", "chaotick", "náležit"],
        "formal_rejection": ["odmít", "odmit", "formál", "formal", "stížnost", "stiznost", "ústavn", "ustavn"],
    }
    report: dict[str, Any] = {}
    for marker_name, needles in markers.items():
        found_in: dict[str, bool] = {}
        for hay_name, hay in haystacks.items():
            found_in[hay_name] = any(n in hay for n in needles)
        report[marker_name] = {
            "found_in": found_in,
            "preserved_in_retrieval_query": found_in["retrieval_query_used"]
            or found_in["retrieval_queries_joined"],
            "present_as_constraint_or_concept": found_in["hard_soft_constraint_values"]
            or found_in["legal_concepts"],
        }
    return report


def _constraint(constraint: Any) -> dict[str, Any]:
    return {
        "category": str(getattr(constraint, "category", None) or getattr(constraint, "constraint_category", None)),
        "value": getattr(constraint, "value", None),
        "normalized_value": getattr(constraint, "normalized_value", None),
        "polarity": str(getattr(constraint, "polarity", None)),
        "attribute": getattr(constraint, "attribute", None),
    }


def _obj(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return {k: _obj(v) for k, v in vars(value).items() if not k.startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_obj(v) for v in value]
    if isinstance(value, dict):
        return {k: _obj(v) for k, v in value.items()}
    return value


if __name__ == "__main__":
    raise SystemExit(main())
