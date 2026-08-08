#!/usr/bin/env python3
"""Evaluate case-similarity golden v1 against LegalV2HybridRetriever (offline, no LLM).

Uses the authoritative legal_v2 hybrid path:
  build_query_spec_v2 → LegalV2HybridRetriever.retrieve → document aggregation.

Does not call DeepSeek / search-v2 verification. Does not tune retrieval knobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.benchmark.case_similarity_eval import (  # noqa: E402
    FAILURE_DOCUMENT_ID_MAPPING_ERROR,
    FAILURE_EXPECTED_DOCUMENT_MISSING,
    FAILURE_EXPECTED_ECLI_MISSING_FROM_INDEX,
    FAILURE_MISSING_VERIFIED_ECLI_IN_BENCHMARK,
    FAILURE_RETRIEVED_RESULT_MISSING_ECLI,
    FAILURE_RETRIEVAL_ERROR,
    CaseSimilarityQueryEvalResult,
    RetrievedDocumentScore,
    aggregate_case_similarity_metrics,
    corpus_presence_summary,
    dedupe_document_ids,
    evaluate_ranked_documents,
)
from app.rag.legal_v2.benchmark.case_similarity_golden import (  # noqa: E402
    DEFAULT_PILOT_DATASET,
    load_case_similarity_golden_jsonl,
)
from app.rag.legal_v2.identity import (  # noqa: E402
    IDENTITY_STATUS_BLOCKED_MISSING_ECLI,
    IDENTITY_STATUS_VERIFIED,
    is_valid_ecli,
    normalize_ecli,
)
from app.rag.legal_v2.indexing import LEGAL_V2_PROFILE  # noqa: E402
from app.rag.legal_v2.query_spec import build_query_spec_v2  # noqa: E402
from app.rag.legal_v2.retriever import (  # noqa: E402
    LegalV2RetrieverConfig,
    build_live_legal_v2_retriever,
)
from app.rag.retrieval.bge_m3_embedder import BgeM3Embedder  # noqa: E402
from app.rag.retrieval.production_profile import ProductionRetrievalConfig  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_PILOT_DATASET)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to artifacts/legal_v2/case_similarity_golden_v1_baseline/<run_id>/",
    )
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
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
                "storage/rag/bm25/nalus_legal_paragraph_bm25_v2_pilot_600.sqlite",
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
    parser.add_argument("--dense-candidate-chunks", type=int, default=80)
    parser.add_argument("--bm25-candidate-chunks", type=int, default=80)
    parser.add_argument("--fused-candidate-chunks", type=int, default=120)
    parser.add_argument("--candidate-documents", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--profile",
        choices=("fast", "fast_ce"),
        default="fast",
        help="fast = Stage 1 only; fast_ce = Stage 1 shortlist + Cross-Encoder rerank.",
    )
    parser.add_argument(
        "--ce-candidate-documents",
        type=int,
        default=int(os.getenv("NALUS_LEGAL_V2_CE_CANDIDATE_DOCUMENTS", "30")),
    )
    parser.add_argument(
        "--ce-passages-per-document",
        type=int,
        default=int(os.getenv("NALUS_LEGAL_V2_CE_PASSAGES_PER_DOCUMENT", "3")),
    )
    parser.add_argument(
        "--ce-passage-selector",
        default=os.getenv(
            "NALUS_LEGAL_V2_CE_PASSAGE_SELECTOR",
            "first_n_stage1_order_v1",
        ),
        help="Passage selector policy id (e.g. diversified_stage1_evidence_v1).",
    )
    parser.add_argument(
        "--ce-evidence-pool-limit",
        type=int,
        default=int(os.getenv("NALUS_LEGAL_V2_CE_EVIDENCE_POOL_LIMIT", "40")),
        help="Max Stage-1 evidence chunks retained per candidate for CE selection.",
    )
    parser.add_argument(
        "--ce-experiment-name",
        default=os.getenv("NALUS_LEGAL_V2_CE_EXPERIMENT_NAME", ""),
        help="Optional experiment label written into CE manifest (e.g. ce_bge_v2m3_p7_diverse_v1).",
    )
    parser.add_argument(
        "--ce-model",
        default=os.getenv("NALUS_LEGAL_V2_CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3"),
    )
    parser.add_argument(
        "--allow-scoring-with-missing-primaries",
        action="store_true",
        help="Run retrieval even when primary documents are missing from the index.",
    )
    parser.add_argument(
        "--compatibility-only",
        action="store_true",
        help="Write corpus compatibility artifacts and exit without retrieval scoring.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.profile == "fast_ce":
        experiment_label = str(args.ce_experiment_name or "").strip()
        default_root = experiment_label or "case_similarity_ce_v1"
    else:
        default_root = "case_similarity_golden_v1_baseline"
    output_dir = args.output_dir or (
        PROJECT_ROOT
        / "artifacts"
        / "legal_v2"
        / default_root
        / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_case_similarity_golden_jsonl(args.benchmark)
    if len(items) != 20:
        raise SystemExit(f"expected 20 benchmark rows, found {len(items)}")

    git_commit, dirty = _git_state()
    benchmark_sha = hashlib.sha256(args.benchmark.read_bytes()).hexdigest()
    command = " ".join(["python", "scripts/legal_v2/evaluate_case_similarity_golden_v1.py", *sys.argv[1:]])

    config_payload = {
        "run_id": run_id,
        "benchmark_path": str(args.benchmark),
        "benchmark_sha256": benchmark_sha,
        "code_commit": git_commit,
        "dirty_working_tree": dirty,
        "execution_command": command,
        "retrieval_entry_point": "app.rag.legal_v2.retrieve.retriever.LegalV2HybridRetriever.retrieve",
        "query_spec_builder": "app.rag.legal_v2.query_spec.build_query_spec_v2",
        "target_collection": args.qdrant_collection,
        "qdrant_url": args.qdrant_url,
        "bm25_sidecar_path": str(args.bm25_sidecar_path),
        "bm25_index_id": args.bm25_index_id,
        "embedding_model": os.getenv("EMBEDDING_MODEL_NAME", "/app/models/BAAI/bge-m3"),
        "embedding_provider": "sentence_transformer",
        "embedding_local_files_only": os.getenv("EMBEDDING_LOCAL_FILES_ONLY", "1"),
        "sparse_method": "Bm25Sidecar",
        "bm25_k1": LEGAL_V2_PROFILE.bm25_k1,
        "bm25_b": LEGAL_V2_PROFILE.bm25_b,
        "fusion": "rrf",
        "rrf_k": LEGAL_V2_PROFILE.rrf_k,
        "dense_candidate_chunks": args.dense_candidate_chunks,
        "bm25_candidate_chunks": args.bm25_candidate_chunks,
        "fused_candidate_chunks": args.fused_candidate_chunks,
        "candidate_documents": args.candidate_documents,
        "top_k_documents": args.top_k,
        "profile": args.profile,
        "reranker": (
            f"cross_encoder:{args.ce_model}" if args.profile == "fast_ce" else None
        ),
        "ce_candidate_documents": args.ce_candidate_documents if args.profile == "fast_ce" else None,
        "ce_passages_per_document": (
            args.ce_passages_per_document if args.profile == "fast_ce" else None
        ),
        "ce_passage_selector": (
            args.ce_passage_selector if args.profile == "fast_ce" else None
        ),
        "ce_evidence_pool_limit": (
            args.ce_evidence_pool_limit if args.profile == "fast_ce" else None
        ),
        "ce_experiment_name": (
            args.ce_experiment_name if args.profile == "fast_ce" else None
        ),
        "aggregation": "group_fused_chunks_by_document_id_best_chunk_rrf_plus_evidence_bonus",
        "feature_flags": {
            "NALUS_LEGAL_V2_SEARCH_ENABLED": os.getenv("NALUS_LEGAL_V2_SEARCH_ENABLED"),
            "NALUS_LEGAL_V2_QDRANT_COLLECTION": os.getenv("NALUS_LEGAL_V2_QDRANT_COLLECTION"),
            "NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED": os.getenv(
                "NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED"
            ),
        },
        "paid_provider_calls": False,
    }
    (output_dir / "retrieval_run_config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    try:
        from qdrant_client import QdrantClient  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'qdrant_client' in this Python environment. "
            "Run the evaluator inside the nalus-scraper-api Docker image "
            "(same setup as golden ECLI indexing), e.g.\n"
            "  docker run --rm --network nalus-scraper_default "
            "--volumes-from nalus-scraper-api-1 "
            "-v \"<repo>:/work\" -w /work "
            "-e EMBEDDING_MODEL_NAME=/root/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181 "
            "-e EMBEDDING_LOCAL_FILES_ONLY=1 "
            "-e NALUS_LEGAL_V2_QDRANT_COLLECTION=nalus_legal_paragraph_chunks_v2_pilot_600 "
            "-e NALUS_LEGAL_V2_BM25_INDEX_ID=nalus_legal_paragraph_bm25_v2_pilot_600 "
            "-e NALUS_LEGAL_V2_BM25_SIDECAR_PATH=/app/storage/rag/bm25/nalus_legal_paragraph_bm25_v2_pilot_600.sqlite "
            "nalus-scraper-api python scripts/legal_v2/evaluate_case_similarity_golden_v1.py "
            "--qdrant-url http://qdrant:6333"
        ) from exc

    client = QdrantClient(url=args.qdrant_url, timeout=60)
    index_doc_ids = _list_indexed_document_ids(client, args.qdrant_collection)
    source_to_ecli = _build_source_to_ecli_mapping(items)
    golden_to_index = {
        source_id: ecli if ecli and ecli in index_doc_ids else None
        for source_id, ecli in source_to_ecli.items()
    }
    # Compatibility audit uses indexed ECLIs, never doc-* as production IDs.
    compatibility = corpus_presence_summary(items=items, present_document_ids=index_doc_ids)
    compatibility["target_collection"] = args.qdrant_collection
    compatibility["indexed_document_count"] = len(index_doc_ids)
    compatibility["source_to_ecli_mapping"] = {
        source_id: ecli for source_id, ecli in source_to_ecli.items() if ecli
    }
    compatibility["blocked_missing_verified_ecli"] = sorted(
        source_id for source_id, ecli in source_to_ecli.items() if not ecli
    )
    compatibility["unmapped_golden_document_ids"] = sorted(
        {
            document_id
            for item in items
            for document_id in (
                [item.source_document_id]
                + list(item.accepted_alternative_document_ids)
                + list(item.hard_negative_document_ids)
            )
            if not source_to_ecli.get(document_id)
            or source_to_ecli.get(document_id) not in index_doc_ids
        }
    )
    (output_dir / "corpus_compatibility.json").write_text(
        json.dumps(compatibility, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "\n".join(
            [
                f"run_id={run_id}",
                f"collection={args.qdrant_collection}",
                f"indexed_documents={len(index_doc_ids)}",
                f"primary_present={compatibility['primary_documents_present']}",
                f"primary_missing={compatibility['primary_documents_missing']}",
                f"hard_negative_blocked_entries={compatibility['hard_negative_blocked_entries']}",
                f"output_dir={output_dir}",
            ]
        )
    )

    if args.compatibility_only:
        _write_blocked_report(
            output_dir=output_dir,
            config=config_payload,
            compatibility=compatibility,
            results=[],
            reason="compatibility_only",
            elapsed_s=time.perf_counter() - started,
        )
        return 0

    if compatibility["primary_documents_missing"] > 0 and not args.allow_scoring_with_missing_primaries:
        _write_blocked_report(
            output_dir=output_dir,
            config=config_payload,
            compatibility=compatibility,
            results=[],
            reason=(
                "Corpus/index incompatibility: one or more primary expected ECLIs are "
                "absent from the target collection (or missing verified ECLI in the "
                "benchmark). Scoring aborted to avoid fake misses. Index the reviewed "
                "judgments under their verified ECLI before running the first real baseline."
            ),
            elapsed_s=time.perf_counter() - started,
        )
        print("BLOCKED: primary ECLIs missing from target collection; scoring skipped.")
        return 2

    if not args.bm25_sidecar_path.exists():
        raise SystemExit(f"BM25 sidecar missing: {args.bm25_sidecar_path}")

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
    embedder = BgeM3Embedder(_embedder_config(retriever_config))
    retriever = build_live_legal_v2_retriever(client, embedder, retriever_config)

    ce_service = None
    if args.profile == "fast_ce":
        from app.rag.legal_v2.rerank.config import CrossEncoderConfig
        from app.rag.legal_v2.rerank.service import CrossEncoderRerankingService

        allow_download = os.getenv("NALUS_LEGAL_V2_CE_ALLOW_DOWNLOAD", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        ce_service = CrossEncoderRerankingService(
            CrossEncoderConfig(
                enabled=True,
                model_id=args.ce_model,
                candidate_documents=args.ce_candidate_documents,
                passages_per_document=args.ce_passages_per_document,
                batch_size=int(os.getenv("NALUS_LEGAL_V2_CE_BATCH_SIZE", "16")),
                device=os.getenv("NALUS_LEGAL_V2_CE_DEVICE", "auto"),
                max_length=int(os.getenv("NALUS_LEGAL_V2_CE_MAX_LENGTH", "512")),
                allow_download=allow_download,
                local_files_only=not allow_download,
                passage_selector=args.ce_passage_selector,
                evidence_pool_limit=args.ce_evidence_pool_limit,
            )
        )
        # Fail fast if model cannot load (no silent FAST fallback in CE benchmark).
        ce_service._get_provider().load()

    results: list[CaseSimilarityQueryEvalResult] = []
    for item in items:
        primary_ecli = normalize_ecli(item.expected_primary_ecli) if item.expected_primary_ecli else None
        alt_eclis = [
            normalize_ecli(row.ecli)
            for row in item.accepted_alternative_rationales
            if row.ecli
        ]
        hn_eclis = [
            normalize_ecli(row.ecli)
            for row in item.hard_negative_rationales
            if row.ecli
        ]

        if item.primary_identity_status == IDENTITY_STATUS_BLOCKED_MISSING_ECLI or not primary_ecli:
            results.append(
                evaluate_ranked_documents(
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
                    failure_type=FAILURE_MISSING_VERIFIED_ECLI_IN_BENCHMARK,
                    error="primary judgment lacks verified ECLI",
                    top_k=args.top_k,
                    expected_primary_source_document_id=item.source_document_id,
                    expected_primary_ecli=None,
                )
            )
            continue

        corpus_ok = primary_ecli in index_doc_ids
        if not corpus_ok:
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
                    ranked_document_ids=[],
                    corpus_compatible=False,
                    failure_type=FAILURE_EXPECTED_ECLI_MISSING_FROM_INDEX,
                    error="primary ECLI absent from target collection",
                    top_k=args.top_k,
                    expected_primary_source_document_id=item.source_document_id,
                    expected_primary_ecli=primary_ecli,
                )
            )
            continue

        try:
            query_spec = build_query_spec_v2(item.query)
            retrieval = retriever.retrieve(query_spec)
            if retrieval.diagnostics.get("collection") != args.qdrant_collection:
                raise RuntimeError(
                    "collection drift: "
                    f"{retrieval.diagnostics.get('collection')} != {args.qdrant_collection}"
                )
            stage1_docs = _stage1_docs_from_retrieval(
                retrieval.documents,
                limit=(
                    args.ce_candidate_documents
                    if args.profile == "fast_ce"
                    else args.top_k
                ),
                evidence_limit=(
                    args.ce_evidence_pool_limit if args.profile == "fast_ce" else 5
                ),
                prefer_chunk_evidence=args.profile == "fast_ce",
            )
            ranked_rows: list[tuple[str, float | None, float | None, float | None]] = []
            if args.profile == "fast_ce":
                assert ce_service is not None
                reranked = ce_service.rerank(item.query, stage1_docs, require_success=True)
                for row in reranked.documents:
                    ranked_rows.append(
                        (row.ecli, row.stage1_score, row.rrf_score, row.ce_score)
                    )
            else:
                for doc in stage1_docs:
                    ranked_rows.append(
                        (doc.ecli, doc.score, getattr(doc, "rrf_score", None), None)
                    )

            ranked_eclis: list[str] = []
            retrieved_results: list[RetrievedDocumentScore] = []
            missing_ecli = False
            for ecli, stage1_score, rrf_score, ce_score in ranked_rows:
                if not ecli or not is_valid_ecli(ecli):
                    missing_ecli = True
                    continue
                ecli_n = normalize_ecli(ecli)
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
                        score=stage1_score,
                        dense_score=None,
                        sparse_score=None,
                        fusion_score=rrf_score,
                        reranker_score=ce_score,
                    )
                )
                if len(ranked_eclis) >= args.top_k:
                    break
            failure = FAILURE_RETRIEVED_RESULT_MISSING_ECLI if missing_ecli and not ranked_eclis else None
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
                    corpus_compatible=True,
                    failure_type=failure,
                    error="retrieved documents missing ECLI identity" if failure else None,
                    top_k=args.top_k,
                    expected_primary_source_document_id=item.source_document_id,
                    expected_primary_ecli=primary_ecli,
                )
            )
        except Exception as exc:  # noqa: BLE001
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
                    ranked_document_ids=[],
                    corpus_compatible=corpus_ok,
                    failure_type=FAILURE_RETRIEVAL_ERROR,
                    error=f"{exc.__class__.__name__}: {exc}",
                    top_k=args.top_k,
                    expected_primary_source_document_id=item.source_document_id,
                    expected_primary_ecli=primary_ecli,
                )
            )

    metrics = aggregate_case_similarity_metrics(
        results,
        missing_hard_negative_document_count=compatibility["hard_negatives_missing"],
    )
    _write_results(output_dir, results)
    _write_report(
        output_dir=output_dir,
        config=config_payload,
        compatibility=compatibility,
        results=results,
        metrics=metrics,
        elapsed_s=time.perf_counter() - started,
        blocked_reason=None,
    )
    if args.profile == "fast_ce":
        ce_ready = ce_service.readiness() if ce_service is not None else {}
        provider = ce_service._get_provider() if ce_service is not None else None
        manifest = {
            "experiment_id": run_id,
            "experiment_name": str(args.ce_experiment_name or "").strip() or None,
            "profile": "fast_plus_ce_experiment",
            "git_commit": git_commit,
            "timestamp": run_id,
            "model_id": args.ce_model,
            "model_revision": getattr(provider, "model_revision", None),
            "dtype": getattr(provider, "dtype", None),
            "device": ce_ready.get("device") or os.getenv("NALUS_LEGAL_V2_CE_DEVICE", "auto"),
            "max_length": int(os.getenv("NALUS_LEGAL_V2_CE_MAX_LENGTH", "512")),
            "batch_size": int(os.getenv("NALUS_LEGAL_V2_CE_BATCH_SIZE", "16")),
            "candidate_documents": args.ce_candidate_documents,
            "passages_per_document": args.ce_passages_per_document,
            "passage_selector": args.ce_passage_selector,
            "evidence_pool_limit": args.ce_evidence_pool_limit,
            "document_aggregation": "max",
            "aggregation": "max",
            "stage1_collection": args.qdrant_collection,
            "bm25_index_id": args.bm25_index_id,
            "stage1_config_fingerprint": {
                "dense_candidate_chunks": args.dense_candidate_chunks,
                "bm25_candidate_chunks": args.bm25_candidate_chunks,
                "fused_candidate_chunks": args.fused_candidate_chunks,
                "candidate_documents": args.candidate_documents,
                "rrf_k": LEGAL_V2_PROFILE.rrf_k,
                "bm25_k1": LEGAL_V2_PROFILE.bm25_k1,
                "bm25_b": LEGAL_V2_PROFILE.bm25_b,
            },
            "benchmark_sha256": benchmark_sha,
            "metrics": {
                "hit_at_1": metrics.hit_at_1,
                "hit_at_10": metrics.hit_at_10,
                "mrr": metrics.mrr,
                "hn_outrank_rate": metrics.hard_negative_outrank_rate,
                "evaluable": metrics.evaluable_positive_retrieval_queries,
                "retrieval_failures": metrics.retrieval_execution_failures,
            },
        }
        (output_dir / "ce_experiment_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        "\n".join(
            [
                f"processed={len(results)}",
                f"evaluable={metrics.evaluable_positive_retrieval_queries}",
                f"retrieval_failures={metrics.retrieval_execution_failures}",
                f"hit_at_1={metrics.hit_at_1}",
                f"hit_at_10={metrics.hit_at_10}",
                f"mrr={metrics.mrr}",
                f"hn_blocked={metrics.hard_negative_blocked_query_count}",
                f"hn_outrank_rate={metrics.hard_negative_outrank_rate}",
                f"output_dir={output_dir}",
            ]
        )
    )
    return 0


@dataclass
class _EvalStage1Passage:
    text: str
    chunk_id: str
    section: str | None = None
    page: int | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_rank: int | None = None
    retrieval_channels: tuple[str, ...] = ()
    chunk_position: int | None = None


@dataclass
class _EvalStage1Doc:
    ecli: str
    rank: int
    score: float
    relevant_passages: list[_EvalStage1Passage]
    rrf_score: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    metadata: dict[str, Any] | None = None
    chunk_evidence: list[dict[str, Any]] | None = None


def _stage1_docs_from_retrieval(
    documents: list[Any],
    *,
    limit: int,
    evidence_limit: int = 5,
    prefer_chunk_evidence: bool = False,
) -> list[_EvalStage1Doc]:
    out: list[_EvalStage1Doc] = []
    for index, doc in enumerate(list(documents)[: max(0, limit)], start=1):
        raw_id = str(getattr(doc, "document_id", "") or "")
        meta = dict(getattr(doc, "metadata", None) or {})
        ecli_raw = str(meta.get("ecli") or raw_id)
        ecli = normalize_ecli(ecli_raw) if is_valid_ecli(ecli_raw) else ""
        if not ecli:
            continue
        chunk_evidence_raw = [
            dict(item)
            for item in list(getattr(doc, "chunk_evidence", None) or [])[
                : max(0, int(evidence_limit))
            ]
            if isinstance(item, dict)
        ]
        passages: list[_EvalStage1Passage] = []
        if prefer_chunk_evidence and chunk_evidence_raw:
            for item in chunk_evidence_raw:
                text = str(item.get("text") or "").strip()
                if not text:
                    continue
                passages.append(
                    _EvalStage1Passage(
                        text=text,
                        chunk_id=str(item.get("chunk_id") or f"p-{len(passages)}"),
                        section=item.get("section"),
                        page=item.get("page"),
                        dense_rank=item.get("dense_rank"),
                        bm25_rank=item.get("bm25_rank"),
                        rrf_rank=item.get("rrf_rank"),
                        retrieval_channels=tuple(item.get("retrieval_channels") or ()),
                        chunk_position=item.get("chunk_position"),
                    )
                )
        else:
            for paragraph in list(getattr(doc, "paragraphs", None) or [])[
                : max(0, int(evidence_limit))
            ]:
                text = str(
                    getattr(paragraph, "normalized_text", None)
                    or getattr(paragraph, "original_text", None)
                    or ""
                ).strip()
                if not text:
                    continue
                passages.append(
                    _EvalStage1Passage(
                        text=text,
                        chunk_id=str(
                            getattr(paragraph, "paragraph_id", "") or f"p-{len(passages)}"
                        ),
                    )
                )
        out.append(
            _EvalStage1Doc(
                ecli=ecli,
                rank=index,
                score=float(getattr(doc, "score", 0.0) or 0.0),
                relevant_passages=passages,
                rrf_score=getattr(doc, "rrf_score", None),
                dense_rank=getattr(doc, "dense_rank", None),
                bm25_rank=getattr(doc, "bm25_rank", None),
                metadata=meta,
                chunk_evidence=chunk_evidence_raw,
            )
        )
    return out


def _embedder_config(config: LegalV2RetrieverConfig) -> ProductionRetrievalConfig:
    return ProductionRetrievalConfig(
        profile=LEGAL_V2_PROFILE,
        qdrant_collection=config.qdrant_collection,
        bm25_sidecar_path=config.bm25_sidecar_path,
        bm25_index_id=config.bm25_index_id,
        model_path=config.model_path,
        local_files_only=True,
        trust_remote_code=False,
        device="cpu",
        candidate_multiplier=1,
        min_candidate_count=1,
        max_candidate_count=max(
            config.dense_candidate_chunks,
            config.bm25_candidate_chunks,
        ),
        lexical_filter_enabled=False,
    )


def _git_state() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=str(PROJECT_ROOT),
                text=True,
            ).strip()
        )
        return commit, dirty
    except Exception:  # noqa: BLE001
        return "unknown", True


def _list_indexed_document_ids(client: Any, collection: str) -> set[str]:
    """Return normalized production document IDs (ECLI preferred) from the collection."""
    ids: set[str] = set()
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=next_offset,
            with_payload=["document_id", "ecli", "canonical_document_id"],
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            for key in ("ecli", "canonical_document_id", "document_id"):
                value = str(payload.get(key) or "").strip()
                if value and is_valid_ecli(value):
                    ids.add(normalize_ecli(value))
                    break
            else:
                document_id = payload.get("document_id")
                if document_id:
                    ids.add(str(document_id))
        if next_offset is None:
            break
    return ids


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


def _write_results(output_dir: Path, results: list[CaseSimilarityQueryEvalResult]) -> None:
    path = output_dir / "case_similarity_retrieval_results.jsonl"
    lines = [row.model_dump_json() for row in results]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_blocked_report(
    *,
    output_dir: Path,
    config: dict[str, Any],
    compatibility: dict[str, Any],
    results: list[CaseSimilarityQueryEvalResult],
    reason: str,
    elapsed_s: float,
) -> None:
    metrics = aggregate_case_similarity_metrics(
        results,
        missing_hard_negative_document_count=compatibility.get("hard_negatives_missing", 0),
    )
    _write_results(output_dir, results)
    _write_report(
        output_dir=output_dir,
        config=config,
        compatibility=compatibility,
        results=results,
        metrics=metrics,
        elapsed_s=elapsed_s,
        blocked_reason=reason,
    )


def _write_report(
    *,
    output_dir: Path,
    config: dict[str, Any],
    compatibility: dict[str, Any],
    results: list[CaseSimilarityQueryEvalResult],
    metrics: Any,
    elapsed_s: float,
    blocked_reason: str | None,
) -> None:
    lines: list[str] = [
        "# Case Similarity Golden v1 — Retrieval Baseline Report",
        "",
        f"- timestamp_utc: `{config['run_id']}`",
        f"- code_commit: `{config['code_commit']}`",
        f"- dirty_working_tree: `{config['dirty_working_tree']}`",
        f"- execution_command: `{config['execution_command']}`",
        f"- benchmark_path: `{config['benchmark_path']}`",
        f"- benchmark_sha256: `{config['benchmark_sha256']}`",
        f"- retrieval_entry_point: `{config['retrieval_entry_point']}`",
        f"- target_collection: `{config['target_collection']}`",
        f"- embedding_model: `{config['embedding_model']}`",
        f"- sparse_method: `{config['sparse_method']}`",
        f"- fusion: `{config['fusion']}` (rrf_k={config['rrf_k']})",
        f"- reranker: `{config['reranker']}`",
        f"- aggregation: `{config['aggregation']}`",
        f"- elapsed_seconds: `{elapsed_s:.3f}`",
        "",
        "## Corpus compatibility",
        "",
        f"- indexed_document_count: `{compatibility.get('indexed_document_count')}`",
        f"- primary_present: `{compatibility['primary_documents_present']}`",
        f"- primary_missing: `{compatibility['primary_documents_missing']}`",
        f"- alternatives_present/missing: "
        f"`{compatibility['accepted_alternatives_present']}` / "
        f"`{compatibility['accepted_alternatives_missing']}`",
        f"- hard_negatives_present/missing: "
        f"`{compatibility['hard_negatives_present']}` / "
        f"`{compatibility['hard_negatives_missing']}`",
        f"- hard_negative_evaluable_entries: `{compatibility['hard_negative_evaluable_entries']}`",
        f"- hard_negative_blocked_entries: `{compatibility['hard_negative_blocked_entries']}`",
        "",
    ]
    if blocked_reason:
        lines.extend(
            [
                "## Baseline status",
                "",
                "**BLOCKED — no scored baseline produced.**",
                "",
                blocked_reason,
                "",
                "Conclusion: do not treat Hit@K / MRR as measured until the reviewed "
                "case-similarity documents are present in the authoritative legal_v2 index "
                "under resolvable document IDs.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Aggregate positive-retrieval metrics",
                "",
                f"- total_queries: `{metrics.total_queries}`",
                f"- evaluable_positive_retrieval_queries: `{metrics.evaluable_positive_retrieval_queries}`",
                f"- corpus_index_failures: `{metrics.corpus_index_failures}`",
                f"- retrieval_execution_failures: `{metrics.retrieval_execution_failures}`",
                f"- Hit@1: `{metrics.hit_at_1}`",
                f"- Hit@3: `{metrics.hit_at_3}`",
                f"- Hit@5: `{metrics.hit_at_5}`",
                f"- Hit@10: `{metrics.hit_at_10}`",
                f"- MRR: `{metrics.mrr}`",
                f"- primary-only Hit@1/3/5/10: "
                f"`{metrics.primary_only_hit_at_1}` / `{metrics.primary_only_hit_at_3}` / "
                f"`{metrics.primary_only_hit_at_5}` / `{metrics.primary_only_hit_at_10}`",
                f"- accepted_alternative_wins: `{metrics.accepted_alternative_wins}`",
                f"- no_positive_in_TOP_10: `{metrics.no_positive_in_top_10}`",
                "",
                "## Hard-negative metrics",
                "",
                f"- hard_negative_evaluable_query_count: `{metrics.hard_negative_evaluable_query_count}`",
                f"- hard_negative_blocked_query_count: `{metrics.hard_negative_blocked_query_count}`",
                f"- hard_negative_outrank_count: `{metrics.hard_negative_outrank_count}`",
                f"- hard_negative_outrank_rate (evaluable denominator only): "
                f"`{metrics.hard_negative_outrank_rate}`",
                f"- outrank_query_ids: `{', '.join(metrics.hard_negative_outrank_query_ids) or '—'}`",
                f"- missing_hard_negative_document_count: `{metrics.missing_hard_negative_document_count}`",
                "",
                "Blocked rows (including `nalus-cs-pilot-007`) are included in Hit@K/MRR but "
                "excluded from the hard-negative outrank denominator.",
                "",
            ]
        )

    lines.extend(["## Per-query table", "", "| query | primary | best+ | best+rank | primary_rank | H@1 | H@3 | H@5 | H@10 | RR | HN eval | HN before+ | failure |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|"])
    for row in results:
        hn_best = None
        for rank in row.hard_negative_ranks.values():
            if rank is None:
                continue
            hn_best = rank if hn_best is None else min(hn_best, rank)
        lines.append(
            "| "
            + " | ".join(
                [
                    row.query_id,
                    f"`{row.expected_primary_document_id}`",
                    f"`{row.best_positive_document_id}`" if row.best_positive_document_id else "—",
                    str(row.best_positive_rank) if row.best_positive_rank is not None else "—",
                    str(row.primary_rank) if row.primary_rank is not None else "—",
                    str(row.hit_at_1),
                    str(row.hit_at_3),
                    str(row.hit_at_5),
                    str(row.hit_at_10),
                    f"{row.reciprocal_rank:.4f}",
                    str(row.hard_negative_evaluable),
                    str(row.hard_negative_before_positive),
                    row.failure_type or "—",
                ]
            )
            + " |"
        )

    if results:
        lines.extend(["", "## TOP 10 for non-Hit@10 queries", ""])
        for row in results:
            if row.hit_at_10:
                continue
            lines.append(f"### {row.query_id}")
            lines.append("")
            if not row.retrieved_document_ids:
                lines.append("_No retrieved documents._")
            else:
                for index, document_id in enumerate(row.retrieved_document_ids, start=1):
                    lines.append(f"{index}. `{document_id}`")
            lines.append("")

    path = output_dir / "case_similarity_retrieval_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
