#!/usr/bin/env python3
"""Local A/B: dense-only vs hybrid BM25+RRF on Legal v2 full-corpus FAST A.

Does not rebuild BM25. Does not change production compose/env. No LLM / CE / ColBERT.

Modes:
  A) dense_only        = BGE-M3 + Qdrant
  B) hybrid_bm25_rrf   = BGE-M3 + BM25 sidecar + RRF (rrf_k from LEGAL_V2_PROFILE)

Court/year/type filters are generic metadata constraints applied to both
channels before fusion. They do not encode expected case numbers.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import os
import re
import sqlite3
import statistics
import sys
import time
import unicodedata
from datetime import datetime, timezone
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
from app.rag.legal_v2.benchmark.case_similarity_golden import (  # noqa: E402
    DEFAULT_PILOT_DATASET,
    load_case_similarity_golden_jsonl,
)
from app.rag.legal_v2.identity import is_valid_ecli, normalize_ecli  # noqa: E402
from app.rag.legal_v2.indexing import LEGAL_V2_PROFILE  # noqa: E402
from app.rag.legal_v2.query_spec import build_query_spec_v2  # noqa: E402
from app.rag.legal_v2.retrieve.retriever import (  # noqa: E402
    LegalV2RetrieverConfig,
    build_live_legal_v2_retriever,
)
from app.rag.legal_v2.retrieve.source_filters import (  # noqa: E402
    RetrievalSourceFilters,
    parse_retrieval_source_filters,
)
from app.rag.retrieval.bge_m3_embedder import BgeM3Embedder  # noqa: E402
from app.rag.retrieval.production_profile import ProductionRetrievalConfig  # noqa: E402

_EVAL_PATH = PROJECT_ROOT / "scripts" / "legal_v2" / "evaluate_case_similarity_golden_v1.py"
_eval_loader = importlib.machinery.SourceFileLoader(
    "evaluate_case_similarity_golden_v1",
    str(_EVAL_PATH),
)
_eval_mod = _eval_loader.load_module()
_list_indexed_document_ids = _eval_mod._list_indexed_document_ids
_stage1_docs_from_retrieval = _eval_mod._stage1_docs_from_retrieval

DEFAULT_COLLECTION = "nalus_legal_paragraph_chunks_v2_chunk_ab_v8_a_current_full"
DEFAULT_BM25_ID = "nalus_legal_paragraph_bm25_v2_chunk_ab_v8_a_current_full"
DEFAULT_BM25_PATH = Path(
    "/app/storage/rag/bm25/nalus_legal_paragraph_bm25_v2_chunk_ab_v8_a_current_full.sqlite"
)
HOST_BM25_FALLBACK = (
    PROJECT_ROOT.parent
    / "nalus-scraper"
    / "storage"
    / "rag"
    / "bm25"
    / "nalus_legal_paragraph_bm25_v2_chunk_ab_v8_a_current_full.sqlite"
)
DEFAULT_QUERY = (
    "Odpovědnost zaměstnance vůči zaměstnavateli za škodu způsobenou neodpracováním "
    "předem rozvržené směny při práci na DPP; zaměstnavatel požaduje ušlý zisk podle "
    "předem sjednané smluvní klauzule za absenci. Platnost předem sjednané sankce, "
    "smluvní pokuty nebo paušalizované náhrady škody v pracovněprávním vztahu; "
    "povinnost zaměstnavatele prokázat vznik skutečné škody nebo ušlého zisku, "
    "porušení pracovní povinnosti, zavinění zaměstnance a příčinnou souvislost."
)
_CASE_FOLD_RE = re.compile(r"[^a-z0-9]+")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://nalus-scraper-qdrant-1:6333"))
    parser.add_argument(
        "--qdrant-collection",
        default=os.getenv("NALUS_LEGAL_V2_FAST_QDRANT_COLLECTION", DEFAULT_COLLECTION),
    )
    parser.add_argument(
        "--bm25-sidecar-path",
        type=Path,
        default=Path(os.getenv("NALUS_LEGAL_V2_FAST_BM25_SIDECAR_PATH", str(DEFAULT_BM25_PATH))),
    )
    parser.add_argument(
        "--bm25-index-id",
        default=os.getenv("NALUS_LEGAL_V2_FAST_BM25_INDEX_ID", DEFAULT_BM25_ID),
    )
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--court", default="Ústavní soud")
    parser.add_argument("--target-case-number", default="III. ÚS 479/04")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-documents", type=int, default=50)
    parser.add_argument("--golden-top-k", type=int, default=20)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_PILOT_DATASET)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--skip-golden", action="store_true")
    parser.add_argument("--skip-query", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args(argv)


def _fold_case_number(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    without_marks = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return _CASE_FOLD_RE.sub("", without_marks.casefold())


def _resolve_bm25_path(path: Path) -> Path:
    if path.exists():
        return path
    if HOST_BM25_FALLBACK.exists():
        return HOST_BM25_FALLBACK
    return path


def verify_bm25_index(
    *,
    bm25_path: Path,
    expected_index_id: str,
    expected_collection: str,
    qdrant_client: Any | None,
) -> dict[str, Any]:
    if not bm25_path.exists():
        raise SystemExit(f"BM25 sidecar missing: {bm25_path}")
    payload: dict[str, Any] = {
        "path": str(bm25_path),
        "size_bytes": bm25_path.stat().st_size,
        "rebuilt": False,
    }
    with sqlite3.connect(f"file:{bm25_path.as_posix()}?mode=ro", uri=True) as connection:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        payload["tables"] = tables
        if "bm25_chunks" not in tables:
            payload["compatible"] = False
            payload["mismatch"] = "missing bm25_chunks table"
            return payload
        payload["quick_check"] = "skipped"
        payload["chunk_count"] = int(
            connection.execute("SELECT COUNT(*) FROM bm25_chunks").fetchone()[0]
        )
        sample_meta = connection.execute(
            """
            SELECT chunk_id, document_id, bm25_index_id, qdrant_collection,
                   json_extract(metadata, '$.court'),
                   json_extract(metadata, '$.ecli'),
                   json_extract(metadata, '$.case_reference')
            FROM bm25_chunks
            LIMIT 200
            """
        ).fetchall()
        payload["index_ids"] = connection.execute(
            "SELECT bm25_index_id FROM bm25_chunks LIMIT 1"
        ).fetchall()
        payload["qdrant_collections"] = connection.execute(
            "SELECT qdrant_collection FROM bm25_chunks LIMIT 1"
        ).fetchall()
        if sample_meta:
            payload["index_ids"] = [(sample_meta[0][2], payload["chunk_count"])]
            payload["qdrant_collections"] = [(sample_meta[0][3], payload["chunk_count"])]
        payload["sample_chunk_ids"] = [row[0] for row in sample_meta if row[0]]
        payload["sample_document_ids"] = [row[1] for row in sample_meta if row[1]]
        courts: dict[str, int] = {}
        for row in sample_meta:
            courts[str(row[4] or "")] = courts.get(str(row[4] or ""), 0) + 1
        payload["court_values"] = sorted(courts.items(), key=lambda item: -item[1])
        payload["court_values_sampled_rows"] = len(sample_meta)
        payload["null_document_id_sampled"] = sum(
            1 for row in sample_meta if not str(row[1] or "").strip()
        )
        payload["eval_target_rows"] = [
            row[1:5] + (1,)
            for row in sample_meta
            if "479/04" in str(row[6] or "") or "479.04" in str(row[5] or "")
        ]
        payload["document_count"] = None

    index_ids = {row[0] for row in payload["index_ids"]}
    collections = {row[0] for row in payload["qdrant_collections"]}
    mismatches: list[str] = []
    if payload.get("quick_check") not in {"ok", "skipped", None}:
        mismatches.append(f"sqlite quick_check={payload['quick_check']}")
    if expected_index_id not in index_ids:
        mismatches.append(
            f"bm25_index_id mismatch: {sorted(index_ids)!r} vs {expected_index_id}"
        )
    if expected_collection not in collections:
        mismatches.append(
            f"qdrant_collection mismatch: {sorted(collections)!r} vs {expected_collection}"
        )
    if payload["chunk_count"] < 100_000:
        mismatches.append(
            f"chunk_count {payload['chunk_count']} looks like a pilot/slice, not full corpus"
        )
    payload["mismatches"] = mismatches

    if qdrant_client is not None:
        info = qdrant_client.get_collection(expected_collection)
        qdrant_points = int(getattr(info, "points_count", 0) or 0)
        payload["qdrant_points_count"] = qdrant_points
        payload["chunk_count_delta_vs_qdrant"] = payload["chunk_count"] - qdrant_points
        payload["sample_chunk_ids_resolved_in_qdrant"] = None
        payload["sample_chunk_lookup_skipped"] = True
        if abs(payload["chunk_count"] - qdrant_points) > 25:
            mismatches.append(
                "BM25 chunk_count "
                f"{payload['chunk_count']} vs Qdrant points {qdrant_points}"
            )
    payload["compatible"] = not mismatches
    payload["mismatches"] = mismatches
    return payload


def _embedder_config(retriever_config: LegalV2RetrieverConfig) -> ProductionRetrievalConfig:
    return ProductionRetrievalConfig(
        profile=LEGAL_V2_PROFILE,
        qdrant_collection=retriever_config.qdrant_collection,
        bm25_sidecar_path=retriever_config.bm25_sidecar_path,
        bm25_index_id=retriever_config.bm25_index_id,
        model_path=retriever_config.model_path,
        local_files_only=True,
        trust_remote_code=False,
        device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        candidate_multiplier=1,
        min_candidate_count=1,
        max_candidate_count=max(
            retriever_config.dense_candidate_chunks,
            retriever_config.bm25_candidate_chunks,
        ),
        lexical_filter_enabled=False,
    )


def _best_passage(doc: Any) -> str:
    for passage in list(getattr(doc, "relevant_passages", None) or []):
        text = str(getattr(passage, "text", "") or "").strip()
        if text:
            return text[:500]
    return ""


def _document_case_number(doc: Any) -> str:
    meta = dict(getattr(doc, "metadata", None) or {})
    return str(
        meta.get("case_reference")
        or meta.get("case_number")
        or getattr(doc, "case_number", "")
        or ""
    )


def _rank_target(docs: list[Any], target_case_number: str) -> dict[str, Any] | None:
    wanted = _fold_case_number(target_case_number)
    if not wanted:
        return None
    for rank, doc in enumerate(docs, start=1):
        case_number = _document_case_number(doc)
        ecli = str(getattr(doc, "ecli", "") or "")
        haystack = _fold_case_number(f"{case_number} {ecli}")
        if wanted in haystack or haystack in wanted:
            return {
                "rank": rank,
                "ecli": ecli,
                "case_number": case_number,
                "score": getattr(doc, "score", None),
                "dense_rank": getattr(doc, "dense_rank", None),
                "bm25_rank": getattr(doc, "bm25_rank", None),
                "rrf_score": getattr(doc, "rrf_score", None),
            }
    return None


def _serialize_docs(docs: list[Any], *, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, doc in enumerate(list(docs)[:limit], start=1):
        rows.append(
            {
                "rank": rank,
                "ecli": getattr(doc, "ecli", None),
                "case_number": _document_case_number(doc),
                "court": (getattr(doc, "metadata", None) or {}).get("court"),
                "dense_rank": getattr(doc, "dense_rank", None),
                "bm25_rank": getattr(doc, "bm25_rank", None),
                "rrf_score": getattr(doc, "rrf_score", None),
                "final_score": getattr(doc, "score", None),
                "best_passage": _best_passage(doc),
            }
        )
    return rows


def run_query(
    *,
    retriever: Any,
    query: str,
    source_filters: RetrievalSourceFilters | None,
    top_k: int,
    target_case_number: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    retrieval = retriever.retrieve(
        build_query_spec_v2(query),
        source_filters=source_filters,
    )
    docs = _stage1_docs_from_retrieval(
        retrieval.documents,
        limit=max(top_k, 50),
        evidence_limit=3,
        prefer_chunk_evidence=True,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    target = _rank_target(docs, target_case_number)
    return {
        "latency_ms": round(elapsed_ms, 3),
        "diagnostics": retrieval.diagnostics,
        "target": target,
        "top": _serialize_docs(docs, limit=top_k),
        "result_count": len(docs),
    }


def _recall_at(ranked: list[str], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    hit = {item for item in ranked[:k] if item in relevant}
    return len(hit) / len(relevant)


def run_golden(
    *,
    retriever: Any,
    items: list[Any],
    index_doc_ids: set[str],
    top_k: int,
    source_filters: RetrievalSourceFilters | None = None,
) -> dict[str, Any]:
    results: list[CaseSimilarityQueryEvalResult] = []
    latencies: list[float] = []
    for item in items:
        primary_ecli = (
            normalize_ecli(item.expected_primary_ecli) if item.expected_primary_ecli else None
        )
        alt_eclis = [
            normalize_ecli(row.ecli)
            for row in item.accepted_alternative_rationales
            if row.ecli
        ]
        hn_eclis = [
            normalize_ecli(row.ecli) for row in item.hard_negative_rationales if row.ecli
        ]
        if not primary_ecli:
            continue
        started = time.perf_counter()
        retrieval = retriever.retrieve(
            build_query_spec_v2(item.query),
            source_filters=source_filters,
        )
        latencies.append((time.perf_counter() - started) * 1000.0)
        docs = _stage1_docs_from_retrieval(
            retrieval.documents,
            limit=top_k,
            evidence_limit=3,
            prefer_chunk_evidence=True,
        )
        ranked_eclis: list[str] = []
        retrieved_results: list[RetrievedDocumentScore] = []
        for doc in docs:
            if not doc.ecli or not is_valid_ecli(doc.ecli):
                continue
            ecli_n = normalize_ecli(doc.ecli)
            if ecli_n in ranked_eclis:
                continue
            ranked_eclis.append(ecli_n)
            retrieved_results.append(
                RetrievedDocumentScore(
                    rank=len(retrieved_results) + 1,
                    document_id=ecli_n,
                    ecli=ecli_n,
                    canonical_document_id=ecli_n,
                    score=doc.score,
                    fusion_score=doc.rrf_score,
                )
            )
        results.append(
            evaluate_ranked_documents(
                query_id=item.benchmark_id,
                query=item.query,
                query_style=item.query_style,
                difficulty=item.difficulty,
                expected_primary_document_id=primary_ecli,
                accepted_alternative_document_ids=alt_eclis,
                hard_negative_document_ids=hn_eclis,
                hard_negative_evaluable=item.hard_negative_evaluable,
                hard_negative_blocker=item.hard_negative_blocker,
                ranked_document_ids=ranked_eclis,
                retrieved_results=retrieved_results,
                corpus_compatible=primary_ecli in index_doc_ids,
                top_k=min(10, top_k),
                expected_primary_source_document_id=item.source_document_id,
                expected_primary_ecli=primary_ecli,
            )
        )
    metrics = aggregate_case_similarity_metrics(results, missing_hard_negative_document_count=0)
    recalls_10: list[float] = []
    recalls_20: list[float] = []
    per_query: list[dict[str, Any]] = []
    for row in results:
        relevant = {row.expected_primary_document_id, *row.accepted_alternative_document_ids}
        relevant = {normalize_ecli(item) for item in relevant if item}
        ranked = [normalize_ecli(item) for item in row.retrieved_document_ids]
        r10 = _recall_at(ranked, relevant, 10)
        r20 = _recall_at(ranked, relevant, 20)
        if r10 is not None:
            recalls_10.append(r10)
        if r20 is not None:
            recalls_20.append(r20)
        per_query.append(
            {
                "query_id": row.query_id,
                "primary_rank": row.primary_rank,
                "hit_at_1": row.hit_at_1,
                "hit_at_10": row.hit_at_10,
                "reciprocal_rank": row.reciprocal_rank,
                "failure_type": row.failure_type,
            }
        )
    latency_stats = {
        "count": len(latencies),
        "p50_ms": round(statistics.median(latencies), 3) if latencies else None,
        "p95_ms": round(_percentile(latencies, 0.95), 3) if latencies else None,
        "mean_ms": round(statistics.fmean(latencies), 3) if latencies else None,
    }
    return {
        "metrics": metrics.model_dump(),
        "recall_at_10": round(sum(recalls_10) / len(recalls_10), 6) if recalls_10 else None,
        "recall_at_20": round(sum(recalls_20) / len(recalls_20), 6) if recalls_20 else None,
        "latency": latency_stats,
        "per_query": per_query,
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def _print_mode(label: str, payload: dict[str, Any]) -> None:
    print(f"\n=== {label} ===")
    print(
        f"latency_ms={payload['latency_ms']} dense_ms="
        f"{payload['diagnostics'].get('dense_latency_ms')} bm25_ms="
        f"{payload['diagnostics'].get('bm25_latency_ms')} rrf_ms="
        f"{payload['diagnostics'].get('rrf_latency_ms')}"
    )
    target = payload.get("target")
    if target:
        print(
            "target "
            f"{target.get('case_number')} rank={target.get('rank')} "
            f"ecli={target.get('ecli')} dense_rank={target.get('dense_rank')} "
            f"bm25_rank={target.get('bm25_rank')} rrf={target.get('rrf_score')}"
        )
    else:
        print("target not in returned candidate documents")
    print(
        f"{'rank':<5}{'ecli':<42}{'case':<18}{'dense':<8}{'bm25':<8}{'rrf':<10}"
    )
    for row in payload["top"]:
        print(
            f"{row['rank']:<5}{str(row.get('ecli') or '')[:40]:<42}"
            f"{str(row.get('case_number') or '')[:16]:<18}"
            f"{str(row.get('dense_rank') or '-'):<8}"
            f"{str(row.get('bm25_rank') or '-'):<8}"
            f"{row.get('rrf_score') if row.get('rrf_score') is not None else '-':<10}"
        )
        passage = str(row.get("best_passage") or "").replace("\n", " ")
        if passage:
            print(f"      passage: {passage[:240]}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (
        PROJECT_ROOT
        / "artifacts"
        / "legal_v2"
        / "bm25_hybrid_ab"
        / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    bm25_path = _resolve_bm25_path(args.bm25_sidecar_path)

    try:
        from qdrant_client import QdrantClient
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "qdrant_client is required. Run this script inside the API image."
        ) from exc

    client = QdrantClient(url=args.qdrant_url, timeout=60)
    verification = verify_bm25_index(
        bm25_path=bm25_path,
        expected_index_id=args.bm25_index_id,
        expected_collection=args.qdrant_collection,
        qdrant_client=client,
    )
    (output_dir / "bm25_index_verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: verification[k] for k in (
        "path", "size_bytes", "chunk_count", "document_count",
        "qdrant_points_count", "compatible", "mismatches", "rebuilt",
        "index_ids", "qdrant_collections", "court_values",
        "eval_target_rows", "sample_chunk_ids_resolved_in_qdrant",
    ) if k in verification}, ensure_ascii=False, indent=2, default=str))
    if not verification["compatible"]:
        print("BM25 index mismatch; refusing to rebuild. Fix/report before any reindex.")
        return 2
    if args.verify_only:
        print("verify-only: index reused as-is.")
        return 0

    source_filters = parse_retrieval_source_filters(
        courts=[args.court] if args.court and args.court.lower() not in {"all", ""} else []
    )
    base_config = LegalV2RetrieverConfig(
        qdrant_collection=args.qdrant_collection,
        bm25_sidecar_path=bm25_path,
        bm25_index_id=args.bm25_index_id,
        dense_candidate_chunks=80,
        bm25_candidate_chunks=80,
        fused_candidate_chunks=120,
        candidate_documents=args.candidate_documents,
        model_path=os.getenv(
            "EMBEDDING_MODEL_NAME",
            "/root/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/"
            "5617a9f61b028005a4858fdac845db406aefb181",
        ),
        bm25_enabled=False,
    )
    embedder = BgeM3Embedder(_embedder_config(base_config))
    dense_retriever = build_live_legal_v2_retriever(client, embedder, base_config)
    from dataclasses import replace

    hybrid_config = replace(base_config, bm25_enabled=True)
    hybrid_retriever = build_live_legal_v2_retriever(client, embedder, hybrid_config)

    report: dict[str, Any] = {
        "run_id": run_id,
        "architecture": {
            "dense": "BGE-M3 + Qdrant",
            "lexical": "existing Bm25Sidecar inverted BM25 k1/b from LEGAL_V2_PROFILE",
            "fusion": "rrf_fuse by chunk.id",
            "rrf_k": LEGAL_V2_PROFILE.rrf_k,
            "bm25_k1": LEGAL_V2_PROFILE.bm25_k1,
            "bm25_b": LEGAL_V2_PROFILE.bm25_b,
            "aggregation": "group fused chunks by ECLI / document_id",
            "filters": "generic court/year/type metadata filters on both channels before RRF",
        },
        "index": verification,
        "query": args.query,
        "court_filter": args.court,
        "target_case_number": args.target_case_number,
        "rebuilt_bm25": False,
    }

    if not args.skip_query:
        dense_payload = run_query(
            retriever=dense_retriever,
            query=args.query,
            source_filters=source_filters,
            top_k=args.top_k,
            target_case_number=args.target_case_number,
        )
        hybrid_payload = run_query(
            retriever=hybrid_retriever,
            query=args.query,
            source_filters=source_filters,
            top_k=args.top_k,
            target_case_number=args.target_case_number,
        )
        report["dense_only"] = dense_payload
        report["hybrid_bm25_rrf"] = hybrid_payload
        _print_mode("DENSE_ONLY", dense_payload)
        _print_mode("HYBRID_BM25_RRF", hybrid_payload)

    if not args.skip_golden:
        items = load_case_similarity_golden_jsonl(args.benchmark)
        index_doc_ids = _list_indexed_document_ids(client, args.qdrant_collection)
        print(f"\nRunning golden ({len(items)} queries) dense_only then hybrid_bm25_rrf...")
        dense_golden = run_golden(
            retriever=dense_retriever,
            items=items,
            index_doc_ids=index_doc_ids,
            top_k=args.golden_top_k,
        )
        hybrid_golden = run_golden(
            retriever=hybrid_retriever,
            items=items,
            index_doc_ids=index_doc_ids,
            top_k=args.golden_top_k,
        )
        regressions: list[dict[str, Any]] = []
        dense_by_id = {row["query_id"]: row for row in dense_golden["per_query"]}
        for hybrid_row in hybrid_golden["per_query"]:
            before = dense_by_id.get(hybrid_row["query_id"]) or {}
            before_rank = before.get("primary_rank")
            after_rank = hybrid_row.get("primary_rank")
            before_hit1 = bool(before.get("hit_at_1"))
            after_hit1 = bool(hybrid_row.get("hit_at_1"))
            worse = False
            if before_hit1 and not after_hit1:
                worse = True
            if before_rank is not None and (after_rank is None or after_rank > before_rank):
                worse = True
            if worse:
                regressions.append(
                    {
                        "query_id": hybrid_row["query_id"],
                        "dense_primary_rank": before_rank,
                        "hybrid_primary_rank": after_rank,
                        "dense_hit_at_1": before_hit1,
                        "hybrid_hit_at_1": after_hit1,
                    }
                )
        report["golden"] = {
            "benchmark": str(args.benchmark),
            "human_review_status": "PENDING_HUMAN_REVIEW",
            "dense_only": dense_golden,
            "hybrid_bm25_rrf": hybrid_golden,
            "regressions": regressions,
        }
        d_m = dense_golden["metrics"]
        h_m = hybrid_golden["metrics"]
        print("\n=== GOLDEN COMPARISON ===")
        print(
            f"{'metric':<16}{'dense_only':<16}{'hybrid':<16}"
        )
        for key, extra_d, extra_h in (
            ("Hit@1", d_m.get("hit_at_1"), h_m.get("hit_at_1")),
            ("Hit@3", d_m.get("hit_at_3"), h_m.get("hit_at_3")),
            ("Hit@10", d_m.get("hit_at_10"), h_m.get("hit_at_10")),
            ("MRR", d_m.get("mrr"), h_m.get("mrr")),
            ("Recall@10", dense_golden.get("recall_at_10"), hybrid_golden.get("recall_at_10")),
            ("Recall@20", dense_golden.get("recall_at_20"), hybrid_golden.get("recall_at_20")),
            ("p50_ms", dense_golden["latency"].get("p50_ms"), hybrid_golden["latency"].get("p50_ms")),
            ("p95_ms", dense_golden["latency"].get("p95_ms"), hybrid_golden["latency"].get("p95_ms")),
        ):
            print(f"{key:<16}{str(extra_d):<16}{str(extra_h):<16}")
        print(f"query-level regressions: {len(regressions)}")
        for row in regressions:
            print(
                f"  {row['query_id']}: dense_rank={row['dense_primary_rank']} "
                f"hybrid_rank={row['hybrid_primary_rank']}"
            )

    report["elapsed_s"] = round(time.perf_counter() - started, 3)
    (output_dir / "ab_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {output_dir / 'ab_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
