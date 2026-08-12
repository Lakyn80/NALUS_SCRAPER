#!/usr/bin/env python3
"""Evaluate experimental Hybrid B (BGE-M3 + BM25) + ColBERT RRF on golden v1.

Uses reusable components only:
  dense + BM25 via LegalV2HybridRetriever
  ColBERT via async ColbertRetriever
  fusion via rrf_fuse (N lists)
  document aggregation via aggregate_legal_v2_documents
  metrics via case_similarity_eval helpers

No CE. Does not change FAST/CE canonical profiles. No production activation.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.benchmark.case_similarity_eval import (  # noqa: E402
    FAILURE_RETRIEVAL_ERROR,
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
from app.rag.legal_v2.retrieve.colbert import (  # noqa: E402
    DEFAULT_COLBERT_MODEL,
    DEFAULT_INDEX_NAME,
    ColbertConfig,
    ColbertRetriever,
    PyLateColbertBackend,
)
from app.rag.legal_v2.retrieve.colbert.mapping import load_mapping_jsonl  # noqa: E402
from app.rag.legal_v2.retrieve.colbert_hybrid import (  # noqa: E402
    EXPERIMENT_PROFILE_ID,
    retrieve_hybrid_plus_colbert,
)
from app.rag.legal_v2.retrieve.retrieval_profiles import (  # noqa: E402
    CE_CANONICAL_BM25_INDEX_ID,
    CE_CANONICAL_BM25_SIDECAR_PATH,
    CE_CANONICAL_QDRANT_COLLECTION,
)
from app.rag.legal_v2.retriever import (  # noqa: E402
    LegalV2RetrieverConfig,
    build_live_legal_v2_retriever,
)
from app.rag.retrieval.bge_m3_embedder import BgeM3Embedder  # noqa: E402
from app.rag.retrieval.production_profile import ProductionRetrievalConfig  # noqa: E402

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "legal_v2"
    / "chunking_ab_pilot_300_v1"
    / "colbert_v1"
    / "hybrid_eval"
)
DEFAULT_INDEX_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "legal_v2"
    / "chunking_ab_pilot_300_v1"
    / "colbert_v1"
    / "index"
)
DEFAULT_MAPPING = (
    PROJECT_ROOT
    / "artifacts"
    / "legal_v2"
    / "chunking_ab_pilot_300_v1"
    / "colbert_v1"
    / "colbert_chunk_mapping.jsonl"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "artifacts"
    / "legal_v2"
    / "chunking_ab_pilot_300_v1"
    / "colbert_v1"
    / "colbert_index_manifest.json"
)
DEFAULT_FAST_BASELINE = (
    PROJECT_ROOT
    / "artifacts"
    / "legal_v2"
    / "chunking_ab_pilot_300_v1"
    / "fast_ab_results"
    / "FAST_AB_COMPARISON.json"
)
DEFAULT_CE_BASELINE = (
    PROJECT_ROOT
    / "artifacts"
    / "legal_v2"
    / "chunking_ab_pilot_300_v1"
    / "ce_ab_results"
    / "CE_AB_COMPARISON.json"
)
DEFAULT_COLBERT_BASELINE = (
    PROJECT_ROOT
    / "artifacts"
    / "legal_v2"
    / "chunking_ab_pilot_300_v1"
    / "colbert_v1"
    / "eval"
    / "COLBERT_GOLDEN_RESULTS.json"
)
CRITICAL_QUERY_IDS = (
    "nalus-cs-pilot-002",
    "nalus-cs-pilot-004",
    "nalus-cs-pilot-003",
    "nalus-cs-pilot-005",
    "nalus-cs-pilot-007",
    "nalus-cs-pilot-008",
    "nalus-cs-pilot-009",
    "nalus-cs-pilot-012",
    "nalus-cs-pilot-017",
    "nalus-cs-pilot-018",
    "nalus-cs-pilot-019",
)
METRIC_KEYS = (
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "hit_at_10",
    "mrr",
    "mean_relevant_rank",
    "hard_negative_outrank_rate",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", type=Path, default=DEFAULT_PILOT_DATASET)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    p.add_argument("--qdrant-collection", default=CE_CANONICAL_QDRANT_COLLECTION)
    p.add_argument(
        "--bm25-sidecar-path",
        type=Path,
        default=Path(
            os.getenv(
                "NALUS_LEGAL_V2_BM25_SIDECAR_PATH",
                str(CE_CANONICAL_BM25_SIDECAR_PATH),
            )
        ),
    )
    p.add_argument("--bm25-index-id", default=CE_CANONICAL_BM25_INDEX_ID)
    p.add_argument("--dense-candidate-chunks", type=int, default=80)
    p.add_argument("--bm25-candidate-chunks", type=int, default=80)
    p.add_argument("--colbert-candidate-chunks", type=int, default=80)
    p.add_argument("--fused-candidate-chunks", type=int, default=120)
    p.add_argument("--candidate-documents", type=int, default=40)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_DIR)
    p.add_argument("--mapping-path", type=Path, default=DEFAULT_MAPPING)
    p.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--fast-baseline", type=Path, default=DEFAULT_FAST_BASELINE)
    p.add_argument("--ce-baseline", type=Path, default=DEFAULT_CE_BASELINE)
    p.add_argument("--colbert-baseline", type=Path, default=DEFAULT_COLBERT_BASELINE)
    p.add_argument("--model", default=DEFAULT_COLBERT_MODEL)
    p.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--embedding-device", default=os.getenv("EMBEDDING_DEVICE", "cpu"))
    p.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow HF model download if ColBERT weights are not cached.",
    )
    return p.parse_args(argv)


def _git_meta() -> dict[str, Any]:
    def _run(args: list[str]) -> str:
        try:
            return (
                subprocess.check_output(args, cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL)
                .decode("utf-8")
                .strip()
            )
        except Exception:  # noqa: BLE001
            return "unknown"

    return {
        "git_head": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "branch", "--show-current"]),
        "dirty": bool(_run(["git", "status", "--porcelain"])),
    }


def _build_source_to_ecli_mapping(items: list[Any]) -> dict[str, str | None]:
    mapping: dict[str, str | None] = {}
    for item in items:
        mapping[item.source_document_id] = (
            normalize_ecli(item.expected_primary_ecli) if item.expected_primary_ecli else None
        )
        for row in item.accepted_alternative_rationales:
            mapping[row.document_id] = normalize_ecli(row.ecli) if row.ecli else None
        for row in item.hard_negative_rationales:
            mapping[row.document_id] = normalize_ecli(row.ecli) if row.ecli else None
    return mapping


def _document_ids_from_mapping(mapping_path: Path) -> set[str]:
    mapping = load_mapping_jsonl(mapping_path)
    ids: set[str] = set()
    for row in mapping.rows.values():
        doc = str(row.document_id or "").strip()
        if not doc:
            continue
        ids.add(normalize_ecli(doc) if is_valid_ecli(doc) else doc)
    return ids


def _mean_relevant_rank(rows: list[CaseSimilarityQueryEvalResult]) -> float | None:
    ranks = [r.best_positive_rank for r in rows if r.best_positive_rank is not None]
    if not ranks:
        return None
    return float(mean(ranks))


def _metrics_bundle(results: list[CaseSimilarityQueryEvalResult]) -> dict[str, Any]:
    agg = aggregate_case_similarity_metrics(results)
    evaluable = [
        row
        for row in results
        if row.corpus_compatible
        and row.failure_type != FAILURE_RETRIEVAL_ERROR
        and not row.error
    ]
    return {
        "evaluable_queries": agg.evaluable_positive_retrieval_queries,
        "hit_at_1": agg.hit_at_1,
        "hit_at_3": agg.hit_at_3,
        "hit_at_5": agg.hit_at_5,
        "hit_at_10": agg.hit_at_10,
        "mrr": agg.mrr,
        "mean_relevant_rank": _mean_relevant_rank(evaluable),
        "hit_at_1_count": sum(1 for r in evaluable if r.hit_at_1),
        "hit_at_3_count": sum(1 for r in evaluable if r.hit_at_3),
        "hit_at_5_count": sum(1 for r in evaluable if r.hit_at_5),
        "hit_at_10_count": sum(1 for r in evaluable if r.hit_at_10),
        "no_positive_in_top_10": agg.no_positive_in_top_10,
        "hard_negative_outrank_count": agg.hard_negative_outrank_count,
        "hard_negative_outrank_rate": agg.hard_negative_outrank_rate,
        "hard_negative_outrank_query_ids": list(agg.hard_negative_outrank_query_ids),
        "retrieval_execution_failures": agg.retrieval_execution_failures,
        "accepted_alternative_wins": agg.accepted_alternative_wins,
    }


def _load_fast_metrics(path: Path, key: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") or {}
    if key not in metrics:
        raise SystemExit(f"baseline key {key!r} missing in {path}")
    return dict(metrics[key])


def _load_ce_b_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") or {}
    for key in ("B_ce", "B", "CE_B"):
        if key in metrics:
            return dict(metrics[key])
    raise SystemExit(f"CE B metrics missing in {path}")


def _load_pure_colbert_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload.get("metrics") or {})


def _rank_map_from_fast(path: Path, field: str) -> dict[str, int | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, int | None] = {}
    for row in payload.get("queries") or []:
        qid = str(row.get("query_id") or "")
        raw = row.get(field)
        out[qid] = None if raw in (None, ">10") else int(raw)
    return out


def _rank_map_from_ce(path: Path) -> dict[str, int | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, int | None] = {}
    for row in payload.get("queries") or []:
        qid = str(row.get("query_id") or "")
        raw = row.get("rank_b_ce")
        out[qid] = None if raw in (None, ">10") else int(raw)
    return out


def _rank_map_from_colbert(path: Path) -> dict[str, int | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, int | None] = {}
    for row in payload.get("queries") or []:
        qid = str(row.get("query_id") or "")
        raw = row.get("relevant_rank")
        out[qid] = None if raw is None else int(raw)
    return out


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(b) - float(a)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _rank_display(rank: int | None) -> str:
    if rank is None:
        return ">10"
    return str(rank)


def _pairwise_score(
    left: dict[str, Any],
    right: dict[str, Any],
) -> float:
    weights = (
        ("hit_at_1", 3.0, True),
        ("hit_at_10", 2.0, True),
        ("mrr", 2.5, True),
        ("hit_at_3", 1.5, True),
        ("hit_at_5", 1.0, True),
        ("mean_relevant_rank", 1.5, False),
        ("hard_negative_outrank_rate", 1.0, False),
    )
    score = 0.0
    for key, weight, higher_better in weights:
        lv = left.get(key)
        rv = right.get(key)
        if lv is None or rv is None:
            continue
        if math.isclose(float(lv), float(rv), abs_tol=1e-9):
            continue
        right_better = (float(rv) > float(lv)) if higher_better else (float(rv) < float(lv))
        score += weight if right_better else -weight
    return score


def _hybrid_verdict(
    *,
    hybrid: dict[str, Any],
    fast_a: dict[str, Any],
    fast_b: dict[str, Any],
) -> dict[str, Any]:
    """Decide IMPROVES / NO BENEFIT / REGRESSION vs FAST A and FAST B."""
    vs_a = _pairwise_score(fast_a, hybrid)
    vs_b = _pairwise_score(fast_b, hybrid)
    reasons: list[str] = []
    material_gain = False
    material_loss = False
    for key, higher_better, threshold in (
        ("hit_at_1", True, 0.049),
        ("hit_at_10", True, 0.049),
        ("mrr", True, 0.02),
        ("hit_at_3", True, 0.049),
        ("hit_at_5", True, 0.049),
        ("mean_relevant_rank", False, 0.25),
        ("hard_negative_outrank_rate", False, 0.04),
    ):
        for name, base in (("FAST_A", fast_a), ("FAST_B", fast_b)):
            lv = base.get(key)
            rv = hybrid.get(key)
            if lv is None or rv is None:
                continue
            delta = float(rv) - float(lv)
            improved = delta > threshold if higher_better else (-delta) > threshold
            regressed = (-delta) > threshold if higher_better else delta > threshold
            if improved:
                material_gain = True
                reasons.append(f"{key} vs {name}: +{delta:.4f}")
            if regressed:
                material_loss = True
                reasons.append(f"{key} vs {name}: REGRESSION {delta:.4f}")

    if material_loss and not material_gain:
        label = "REGRESSION"
    elif material_gain and not material_loss and (vs_a > 0.5 or vs_b > 0.5):
        label = "IMPROVES"
    elif abs(vs_a) < 1.0 and abs(vs_b) < 1.0 and not material_gain and not material_loss:
        label = "NO BENEFIT"
    elif material_gain and material_loss:
        # Mixed: prefer NO BENEFIT unless clearly net-positive on weighted score.
        label = "IMPROVES" if (vs_a >= 2.0 and vs_b >= 1.0) else "NO BENEFIT"
    elif vs_a < -1.0 or vs_b < -1.0:
        label = "REGRESSION"
    elif vs_a > 1.0 or vs_b > 1.0:
        label = "IMPROVES"
    else:
        label = "NO BENEFIT"

    return {
        "COLBERT_HYBRID_VERDICT": label,
        "vs_fast_a_score": vs_a,
        "vs_fast_b_score": vs_b,
        "reasons": reasons,
        "note": (
            "Experiment only (no CE). FAST canonical remains A; "
            "CE canonical remains B contextual."
        ),
    }


def _transition_flags(
    *,
    baseline_rank: int | None,
    hybrid_rank: int | None,
    baseline_hn: bool | None,
    hybrid_hn: bool | None,
) -> list[str]:
    flags: list[str] = []
    b_in = baseline_rank is not None and baseline_rank <= 10
    h_in = hybrid_rank is not None and hybrid_rank <= 10
    if not b_in and h_in:
        flags.append("entered_top10")
    if b_in and not h_in:
        flags.append("left_top10")
    if (baseline_rank is None or baseline_rank > 1) and hybrid_rank == 1:
        flags.append("hit1_gain")
    if baseline_rank == 1 and hybrid_rank != 1:
        flags.append("hit1_loss")
    if baseline_rank is not None and hybrid_rank is not None:
        if hybrid_rank < baseline_rank:
            flags.append("improvement")
        elif hybrid_rank > baseline_rank:
            flags.append("regression")
    elif baseline_rank is None and hybrid_rank is not None:
        flags.append("improvement")
    elif baseline_rank is not None and hybrid_rank is None:
        flags.append("regression")
    if hybrid_hn and not baseline_hn:
        flags.append("HN regression")
    return flags


def _embedder_config(config: LegalV2RetrieverConfig, *, device: str) -> ProductionRetrievalConfig:
    return ProductionRetrievalConfig(
        profile=LEGAL_V2_PROFILE,
        qdrant_collection=config.qdrant_collection,
        bm25_sidecar_path=config.bm25_sidecar_path,
        bm25_index_id=config.bm25_index_id,
        model_path=config.model_path,
        local_files_only=True,
        trust_remote_code=False,
        device=device,
        candidate_multiplier=1,
        min_candidate_count=1,
        max_candidate_count=max(
            config.dense_candidate_chunks,
            config.bm25_candidate_chunks,
        ),
        lexical_filter_enabled=False,
    )


async def async_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.index_path.exists():
        raise SystemExit(f"ColBERT index missing: {args.index_path}")
    if not args.mapping_path.exists():
        raise SystemExit(f"ColBERT mapping missing: {args.mapping_path}")
    if not args.bm25_sidecar_path.exists():
        raise SystemExit(f"BM25 sidecar missing: {args.bm25_sidecar_path}")
    manifest = {}
    if args.manifest_path.exists():
        manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
    if manifest and not bool(manifest.get("COLBERT_INDEX_READY")):
        raise SystemExit("COLBERT_INDEX_READY is false; refusing to score.")

    items = load_case_similarity_golden_jsonl(args.benchmark)
    if len(items) != 20:
        raise SystemExit(f"expected 20 golden rows, found {len(items)}")

    try:
        from qdrant_client import QdrantClient  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing qdrant_client. Run inside the Slice-4 GPU / API Docker image."
        ) from exc

    client = QdrantClient(url=args.qdrant_url, timeout=60)
    indexed_docs = _document_ids_from_mapping(args.mapping_path)
    source_to_ecli = _build_source_to_ecli_mapping(items)
    git = _git_meta()
    command = " ".join(
        ["python", "scripts/legal_v2/evaluate_colbert_hybrid_golden_v1.py", *sys.argv[1:]]
    )
    benchmark_sha = hashlib.sha256(args.benchmark.read_bytes()).hexdigest()

    retriever_config = LegalV2RetrieverConfig(
        qdrant_collection=args.qdrant_collection,
        bm25_sidecar_path=args.bm25_sidecar_path,
        bm25_index_id=args.bm25_index_id,
        dense_candidate_chunks=args.dense_candidate_chunks,
        bm25_candidate_chunks=args.bm25_candidate_chunks,
        fused_candidate_chunks=args.fused_candidate_chunks,
        candidate_documents=args.candidate_documents,
        model_path=os.getenv("EMBEDDING_MODEL_NAME", "/app/models/BAAI/bge-m3"),
    )
    embedder = BgeM3Embedder(
        _embedder_config(retriever_config, device=str(args.embedding_device))
    )
    hybrid_retriever = build_live_legal_v2_retriever(client, embedder, retriever_config)

    colbert_config = ColbertConfig(
        model_name=args.model,
        index_path=args.index_path,
        index_name=args.index_name,
        device=args.device,
        top_k=max(int(args.colbert_candidate_chunks), int(args.top_k)),
        batch_size=int(args.batch_size),
        concurrency_limit=1,
        mapping_path=args.mapping_path,
        allow_download=bool(args.allow_download),
    )
    backend = PyLateColbertBackend(colbert_config)
    colbert_retriever = ColbertRetriever(colbert_config, backend=backend)
    await backend.initialize()

    results: list[CaseSimilarityQueryEvalResult] = []
    query_details: list[dict[str, Any]] = []

    try:
        for item in items:
            primary_ecli = source_to_ecli.get(item.source_document_id)
            hn_eclis = [
                ecli
                for doc_id in item.hard_negative_document_ids
                if (ecli := source_to_ecli.get(doc_id))
            ]
            alt_eclis = [
                ecli
                for doc_id in item.accepted_alternative_document_ids
                if (ecli := source_to_ecli.get(doc_id))
            ]
            if not primary_ecli:
                row = evaluate_ranked_documents(
                    query_id=item.benchmark_id,
                    query=item.query,
                    query_style=item.query_style,
                    difficulty=item.difficulty,
                    expected_primary_document_id=item.source_document_id,
                    accepted_alternative_document_ids=alt_eclis,
                    hard_negative_document_ids=hn_eclis,
                    hard_negative_evaluable=item.hard_negative_evaluable,
                    hard_negative_blocker=item.hard_negative_blocker,
                    ranked_document_ids=[],
                    corpus_compatible=False,
                    failure_type="missing_verified_ecli",
                    error="primary judgment lacks verified ECLI",
                    top_k=args.top_k,
                    expected_primary_source_document_id=item.source_document_id,
                    expected_primary_ecli=None,
                )
                results.append(row)
                continue

            corpus_ok = primary_ecli in indexed_docs
            if not corpus_ok:
                row = evaluate_ranked_documents(
                    query_id=item.benchmark_id,
                    query=item.query,
                    query_style=item.query_style,
                    difficulty=item.difficulty,
                    expected_primary_document_id=primary_ecli,
                    accepted_alternative_document_ids=alt_eclis,
                    hard_negative_document_ids=hn_eclis,
                    hard_negative_evaluable=item.hard_negative_evaluable,
                    hard_negative_blocker=item.hard_negative_blocker,
                    ranked_document_ids=[],
                    corpus_compatible=False,
                    failure_type="expected_ecli_missing_from_index",
                    error="primary ECLI absent from ColBERT/B mapping",
                    top_k=args.top_k,
                    expected_primary_source_document_id=item.source_document_id,
                    expected_primary_ecli=primary_ecli,
                )
                results.append(row)
                continue

            try:
                query_spec = build_query_spec_v2(item.query)
                retrieval = await retrieve_hybrid_plus_colbert(
                    hybrid_retriever=hybrid_retriever,
                    colbert_retriever=colbert_retriever,
                    query_spec=query_spec,
                    colbert_candidate_chunks=int(args.colbert_candidate_chunks),
                    fused_candidate_chunks=int(args.fused_candidate_chunks),
                    candidate_documents=int(args.candidate_documents),
                    rrf_k=int(LEGAL_V2_PROFILE.rrf_k),
                )
                ranked_eclis: list[str] = []
                retrieved_results: list[RetrievedDocumentScore] = []
                for doc in retrieval.documents:
                    raw_id = str(doc.document_id or "")
                    meta = dict(doc.metadata or {})
                    ecli_raw = str(meta.get("ecli") or raw_id)
                    if not ecli_raw or not is_valid_ecli(ecli_raw):
                        continue
                    ecli_n = normalize_ecli(ecli_raw)
                    if ecli_n in ranked_eclis:
                        continue
                    ranked_eclis.append(ecli_n)
                    retrieved_results.append(
                        RetrievedDocumentScore(
                            rank=len(retrieved_results) + 1,
                            document_id=ecli_n,
                            ecli=ecli_n,
                            canonical_document_id=ecli_n,
                            source_document_id=None,
                            score=float(doc.score),
                            dense_score=None,
                            sparse_score=None,
                            fusion_score=float(doc.rrf_score)
                            if doc.rrf_score is not None
                            else None,
                            reranker_score=None,
                        )
                    )
                    if len(ranked_eclis) >= args.top_k:
                        break

                row = evaluate_ranked_documents(
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
                    corpus_compatible=True,
                    failure_type=None,
                    error=None,
                    top_k=args.top_k,
                    expected_primary_source_document_id=item.source_document_id,
                    expected_primary_ecli=primary_ecli,
                )
                results.append(row)
                query_details.append(
                    {
                        "query_id": item.benchmark_id,
                        "query_text": item.query,
                        "query_style": item.query_style,
                        "difficulty": item.difficulty,
                        "expected_document_id": primary_ecli,
                        "expected_source_document_id": item.source_document_id,
                        "relevant_rank": row.best_positive_rank,
                        "relevant_rank_display": _rank_display(row.best_positive_rank),
                        "hit_at_1": row.hit_at_1,
                        "hit_at_3": row.hit_at_3,
                        "hit_at_5": row.hit_at_5,
                        "hit_at_10": row.hit_at_10,
                        "reciprocal_rank": row.reciprocal_rank,
                        "hard_negative_evaluable": row.hard_negative_evaluable,
                        "hard_negative_before_positive": row.hard_negative_before_positive,
                        "hard_negative_ranks": list(row.hard_negative_ranks or []),
                        "failure_type": row.failure_type,
                        "error": row.error,
                        "diagnostics": dict(retrieval.diagnostics),
                        "top10": [
                            {
                                "rank": r.rank,
                                "document_id": r.document_id,
                                "score": r.score,
                            }
                            for r in retrieved_results
                        ],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                row = evaluate_ranked_documents(
                    query_id=item.benchmark_id,
                    query=item.query,
                    query_style=item.query_style,
                    difficulty=item.difficulty,
                    expected_primary_document_id=primary_ecli,
                    accepted_alternative_document_ids=alt_eclis,
                    hard_negative_document_ids=hn_eclis,
                    hard_negative_evaluable=item.hard_negative_evaluable,
                    hard_negative_blocker=item.hard_negative_blocker,
                    ranked_document_ids=[],
                    corpus_compatible=True,
                    failure_type=FAILURE_RETRIEVAL_ERROR,
                    error=str(exc),
                    top_k=args.top_k,
                    expected_primary_source_document_id=item.source_document_id,
                    expected_primary_ecli=primary_ecli,
                )
                results.append(row)
                query_details.append(
                    {
                        "query_id": item.benchmark_id,
                        "query_text": item.query,
                        "relevant_rank": None,
                        "relevant_rank_display": ">10",
                        "hit_at_1": False,
                        "hit_at_3": False,
                        "hit_at_5": False,
                        "hit_at_10": False,
                        "failure_type": FAILURE_RETRIEVAL_ERROR,
                        "error": str(exc),
                        "top10": [],
                    }
                )
    finally:
        await backend.close()

    metrics = _metrics_bundle(results)
    fast_a = _load_fast_metrics(args.fast_baseline, "A")
    fast_b = _load_fast_metrics(args.fast_baseline, "B")
    ce_b = _load_ce_b_metrics(args.ce_baseline)
    pure_colbert = _load_pure_colbert_metrics(args.colbert_baseline)

    rank_fast_a = _rank_map_from_fast(args.fast_baseline, "rank_a")
    rank_fast_b = _rank_map_from_fast(args.fast_baseline, "rank_b")
    rank_ce_b = _rank_map_from_ce(args.ce_baseline)
    rank_colbert = _rank_map_from_colbert(args.colbert_baseline)
    hybrid_rank = {
        q["query_id"]: q.get("relevant_rank") for q in query_details
    }
    hybrid_hn = {
        q["query_id"]: bool(q.get("hard_negative_before_positive")) for q in query_details
    }

    # FAST baselines do not expose HN-before in comparison JSON; only mark HN
    # regressions against pure ColBERT / self when hybrid introduces HN.
    transitions: list[dict[str, Any]] = []
    top10_transitions: list[dict[str, Any]] = []
    hit1_gains: list[str] = []
    hit1_losses: list[str] = []
    hn_regressions: list[str] = []

    for qid in sorted(set(rank_fast_a) | set(hybrid_rank)):
        h_rank = hybrid_rank.get(qid)
        detail = {
            "query_id": qid,
            "rank_fast_a": rank_fast_a.get(qid),
            "rank_fast_b": rank_fast_b.get(qid),
            "rank_pure_colbert": rank_colbert.get(qid),
            "rank_hybrid_b_colbert": h_rank,
            "rank_ce_b": rank_ce_b.get(qid),
            "flags_vs_fast_a": _transition_flags(
                baseline_rank=rank_fast_a.get(qid),
                hybrid_rank=h_rank,
                baseline_hn=None,
                hybrid_hn=hybrid_hn.get(qid),
            ),
            "flags_vs_fast_b": _transition_flags(
                baseline_rank=rank_fast_b.get(qid),
                hybrid_rank=h_rank,
                baseline_hn=None,
                hybrid_hn=hybrid_hn.get(qid),
            ),
            "flags_vs_pure_colbert": _transition_flags(
                baseline_rank=rank_colbert.get(qid),
                hybrid_rank=h_rank,
                baseline_hn=None,
                hybrid_hn=hybrid_hn.get(qid),
            ),
        }
        transitions.append(detail)
        flags = set(detail["flags_vs_fast_a"]) | set(detail["flags_vs_fast_b"])
        if "entered_top10" in flags or "left_top10" in flags:
            top10_transitions.append(detail)
        if "hit1_gain" in detail["flags_vs_fast_a"] or "hit1_gain" in detail["flags_vs_fast_b"]:
            hit1_gains.append(qid)
        if "hit1_loss" in detail["flags_vs_fast_a"] or "hit1_loss" in detail["flags_vs_fast_b"]:
            hit1_losses.append(qid)
        if hybrid_hn.get(qid) and qid in (pure_colbert.get("hard_negative_outrank_query_ids") or []):
            # retained HN
            pass
        if hybrid_hn.get(qid) and qid not in set(
            pure_colbert.get("hard_negative_outrank_query_ids") or []
        ):
            # New HN relative to pure ColBERT is weak; also compare aggregate rate later.
            if "HN regression" in detail["flags_vs_fast_a"]:
                hn_regressions.append(qid)

    # Prefer aggregate HN regression vs FAST A/B rates.
    for name, base in (("FAST_A", fast_a), ("FAST_B", fast_b)):
        base_hn = float(base.get("hard_negative_outrank_rate") or 0.0)
        hyb_hn = float(metrics.get("hard_negative_outrank_rate") or 0.0)
        if hyb_hn > base_hn + 1e-9:
            for qid in metrics.get("hard_negative_outrank_query_ids") or []:
                if qid not in hn_regressions:
                    hn_regressions.append(f"{qid} (vs {name})")

    verdict = _hybrid_verdict(hybrid=metrics, fast_a=fast_a, fast_b=fast_b)
    critical = {
        qid: next((q for q in query_details if q["query_id"] == qid), {})
        for qid in CRITICAL_QUERY_IDS
    }
    elapsed_s = time.perf_counter() - started

    payload = {
        "schema": "colbert_hybrid_golden_eval.v1",
        "benchmark": {
            "profile": EXPERIMENT_PROFILE_ID,
            "dataset": str(args.benchmark),
            "dataset_sha256": benchmark_sha,
            "query_count": len(items),
            "top_k_documents": args.top_k,
            "dense_candidate_chunks": args.dense_candidate_chunks,
            "bm25_candidate_chunks": args.bm25_candidate_chunks,
            "colbert_candidate_chunks": args.colbert_candidate_chunks,
            "fused_candidate_chunks": args.fused_candidate_chunks,
            "candidate_documents": args.candidate_documents,
            "rrf_k": int(LEGAL_V2_PROFILE.rrf_k),
            "document_dedupe": (
                "group_fused_chunks_by_document_id_best_chunk_rrf_plus_evidence_bonus"
            ),
            "qdrant_collection": args.qdrant_collection,
            "bm25_index_id": args.bm25_index_id,
            "bm25_sidecar_path": str(args.bm25_sidecar_path),
            "colbert_model": args.model,
            "colbert_index_name": args.index_name,
            "colbert_index_path": str(args.index_path),
            "device_colbert": args.device,
            "device_embedding": args.embedding_device,
            "cross_encoder": False,
            "command": command,
            "git": git,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": elapsed_s,
            "COLBERT_INDEX_READY": bool(manifest.get("COLBERT_INDEX_READY", True)),
        },
        "metrics": {
            "FAST_A": fast_a,
            "FAST_B": fast_b,
            "PURE_COLBERT": pure_colbert,
            "B_PLUS_COLBERT": metrics,
            "CE_B": ce_b,
        },
        "comparison_table": [
            {
                "metric": key,
                "FAST_A": fast_a.get(key),
                "FAST_B": fast_b.get(key),
                "PURE_COLBERT": pure_colbert.get(key),
                "B_PLUS_COLBERT": metrics.get(key),
                "CE_B": ce_b.get(key),
                "hybrid_minus_fast_a": _delta(fast_a.get(key), metrics.get(key)),
                "hybrid_minus_fast_b": _delta(fast_b.get(key), metrics.get(key)),
            }
            for key in METRIC_KEYS
        ],
        "queries": query_details,
        "transitions": transitions,
        "top10_transitions": top10_transitions,
        "hit1_gains": hit1_gains,
        "hit1_losses": hit1_losses,
        "hn_regressions": hn_regressions,
        "critical_queries": critical,
        "verdict": verdict,
        "raw_eval_results": [row.model_dump(mode="json") for row in results],
    }

    json_path = output_dir / "COLBERT_HYBRID_RESULTS.json"
    md_path = output_dir / "COLBERT_HYBRID_RESULTS.md"
    html_path = output_dir / "COLBERT_HYBRID_RESULTS.html"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    html_path.write_text(_render_html(payload), encoding="utf-8")

    print(f"WROTE {json_path}")
    print(f"WROTE {md_path}")
    print(f"WROTE {html_path}")
    print(f"COLBERT HYBRID VERDICT: {verdict['COLBERT_HYBRID_VERDICT']}")
    print(
        "METRICS "
        f"Hit@1={metrics.get('hit_at_1')} Hit@10={metrics.get('hit_at_10')} "
        f"MRR={metrics.get('mrr')} HN={metrics.get('hard_negative_outrank_rate')}"
    )
    return 0


def _render_markdown(payload: dict[str, Any]) -> str:
    v = payload["verdict"]
    lines = [
        "# ColBERT Hybrid Golden Evaluation",
        "",
        f"## COLBERT HYBRID VERDICT: {v['COLBERT_HYBRID_VERDICT']}",
        "",
        "- Pipeline: B contextual dense (BGE-M3) + BM25 + ColBERT → RRF → documents",
        "- Cross-encoder: **OFF**",
        f"- Queries: `{payload['benchmark']['query_count']}`",
        (
            f"- Depths: dense={payload['benchmark']['dense_candidate_chunks']}, "
            f"BM25={payload['benchmark']['bm25_candidate_chunks']}, "
            f"ColBERT={payload['benchmark']['colbert_candidate_chunks']}, "
            f"fused={payload['benchmark']['fused_candidate_chunks']}, "
            f"rrf_k={payload['benchmark']['rrf_k']}"
        ),
        "",
        "| Metric | FAST A | FAST B | Pure ColBERT | B+ColBERT | CE B |",
        "| ------ | -----: | -----: | -----------: | --------: | ---: |",
    ]
    for row in payload["comparison_table"]:
        lines.append(
            f"| {row['metric']} | {_fmt(row['FAST_A'])} | {_fmt(row['FAST_B'])} | "
            f"{_fmt(row['PURE_COLBERT'])} | {_fmt(row['B_PLUS_COLBERT'])} | "
            f"{_fmt(row['CE_B'])} |"
        )
    lines.extend(["", "## Critical queries", ""])
    for qid in CRITICAL_QUERY_IDS:
        q = payload["critical_queries"].get(qid) or {}
        t = next((x for x in payload["transitions"] if x["query_id"] == qid), {})
        lines.append(
            f"- `{qid}`: FAST_A=`{_rank_display(t.get('rank_fast_a'))}` "
            f"FAST_B=`{_rank_display(t.get('rank_fast_b'))}` "
            f"ColBERT=`{_rank_display(t.get('rank_pure_colbert'))}` "
            f"B+ColBERT=`{_rank_display(t.get('rank_hybrid_b_colbert'))}` "
            f"CE_B=`{_rank_display(t.get('rank_ce_b'))}`"
        )
        _ = q
    lines.extend(
        [
            "",
            "## TOP10 transitions vs FAST",
            "",
        ]
    )
    if not payload["top10_transitions"]:
        lines.append("- none")
    else:
        for t in payload["top10_transitions"]:
            lines.append(
                f"- `{t['query_id']}` A=`{_rank_display(t.get('rank_fast_a'))}` "
                f"B=`{_rank_display(t.get('rank_fast_b'))}` "
                f"hybrid=`{_rank_display(t.get('rank_hybrid_b_colbert'))}` "
                f"flagsA={t.get('flags_vs_fast_a')} flagsB={t.get('flags_vs_fast_b')}"
            )
    lines.extend(
        [
            "",
            f"- Hit@1 gains: {payload['hit1_gains'] or 'none'}",
            f"- Hit@1 losses: {payload['hit1_losses'] or 'none'}",
            f"- HN regressions: {payload['hn_regressions'] or 'none'}",
            "",
            f"- Note: {v['note']}",
            "",
        ]
    )
    return "\n".join(lines)


def _render_html(payload: dict[str, Any]) -> str:
    v = payload["verdict"]
    esc = html.escape
    rows = []
    for row in payload["comparison_table"]:
        rows.append(
            "<tr>"
            f"<td>{esc(row['metric'])}</td>"
            f"<td>{esc(_fmt(row['FAST_A']))}</td>"
            f"<td>{esc(_fmt(row['FAST_B']))}</td>"
            f"<td>{esc(_fmt(row['PURE_COLBERT']))}</td>"
            f"<td>{esc(_fmt(row['B_PLUS_COLBERT']))}</td>"
            f"<td>{esc(_fmt(row['CE_B']))}</td>"
            "</tr>"
        )
    crit = []
    for qid in CRITICAL_QUERY_IDS:
        t = next((x for x in payload["transitions"] if x["query_id"] == qid), {})
        crit.append(
            "<tr>"
            f"<td>{esc(qid)}</td>"
            f"<td>{esc(_rank_display(t.get('rank_fast_a')))}</td>"
            f"<td>{esc(_rank_display(t.get('rank_fast_b')))}</td>"
            f"<td>{esc(_rank_display(t.get('rank_pure_colbert')))}</td>"
            f"<td>{esc(_rank_display(t.get('rank_hybrid_b_colbert')))}</td>"
            f"<td>{esc(_rank_display(t.get('rank_ce_b')))}</td>"
            "</tr>"
        )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>ColBERT Hybrid Results</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#1a1a1a;background:#f7f7f5}}
h1,h2{{margin:0 0 12px}}
.verdict{{font-size:1.4rem;font-weight:700;padding:12px 16px;background:#fff;border-left:6px solid #0b6;margin-bottom:18px}}
table{{border-collapse:collapse;width:100%;background:#fff;margin:12px 0 24px}}
th,td{{border:1px solid #ddd;padding:8px;text-align:left}}
th{{background:#eee}}
.meta{{color:#555;margin-bottom:18px}}
</style></head><body>
<h1>ColBERT Hybrid Golden Evaluation</h1>
<div class="verdict">COLBERT HYBRID VERDICT: {esc(v['COLBERT_HYBRID_VERDICT'])}</div>
<p class="meta">B contextual · BGE-M3 + BM25 + ColBERT → RRF · CE OFF ·
dense={esc(str(payload['benchmark']['dense_candidate_chunks']))},
BM25={esc(str(payload['benchmark']['bm25_candidate_chunks']))},
ColBERT={esc(str(payload['benchmark']['colbert_candidate_chunks']))},
rrf_k={esc(str(payload['benchmark']['rrf_k']))}</p>
<h2>Metrics</h2>
<table><thead><tr>
<th>Metric</th><th>FAST A</th><th>FAST B</th><th>Pure ColBERT</th><th>B+ColBERT</th><th>CE B</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Critical queries</h2>
<table><thead><tr>
<th>Query</th><th>FAST A</th><th>FAST B</th><th>Pure ColBERT</th><th>B+ColBERT</th><th>CE B</th>
</tr></thead><tbody>{''.join(crit)}</tbody></table>
<p>Hit@1 gains: {esc(str(payload['hit1_gains'] or 'none'))}<br/>
Hit@1 losses: {esc(str(payload['hit1_losses'] or 'none'))}<br/>
HN regressions: {esc(str(payload['hn_regressions'] or 'none'))}</p>
<p>{esc(v['note'])}</p>
</body></html>
"""


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
