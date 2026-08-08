#!/usr/bin/env python3
"""Focused CE-7 passage-coverage diagnostics for critical CE-3 failures.

Reconstructs CE-3 (first_n, 3 passages) vs CE-7 (diversified, 7 passages) for
selected golden queries. Does not change Stage 1 ranking or golden labels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client import QdrantClient

from app.rag.legal_v2.benchmark.case_similarity_golden import (  # noqa: E402
    load_case_similarity_golden_jsonl,
)
from app.rag.legal_v2.identity import is_valid_ecli, normalize_ecli  # noqa: E402
from app.rag.legal_v2.indexing import LEGAL_V2_PROFILE  # noqa: E402
from app.rag.legal_v2.query_spec import build_query_spec_v2  # noqa: E402
from app.rag.legal_v2.rerank.config import CrossEncoderConfig  # noqa: E402
from app.rag.legal_v2.rerank.models import RerankCandidate  # noqa: E402
from app.rag.legal_v2.rerank.passage_selection import (  # noqa: E402
    build_candidates_from_stage1_docs,
    evidence_records_from_stage1_doc,
)
from app.rag.legal_v2.rerank.selectors.names import (  # noqa: E402
    DIVERSIFIED_STAGE1_EVIDENCE_V1,
    FIRST_N_STAGE1_ORDER_V1,
)
from app.rag.legal_v2.rerank.service import CrossEncoderRerankingService  # noqa: E402
from app.rag.legal_v2.retrieve.retriever import (  # noqa: E402
    LegalV2RetrieverConfig,
    build_live_legal_v2_retriever,
)
from app.rag.retrieval.bge_m3_embedder import BgeM3Embedder  # noqa: E402
from app.rag.retrieval.production_profile import ProductionRetrievalConfig  # noqa: E402


def _stage1_docs_from_retrieval(
    documents: list[Any],
    *,
    limit: int,
    evidence_limit: int = 5,
    prefer_chunk_evidence: bool = False,
) -> list[Any]:
    """Local mirror of eval helper — keep CE diagnostics self-contained."""
    from dataclasses import dataclass

    @dataclass
    class _Passage:
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
    class _Doc:
        ecli: str
        rank: int
        score: float
        relevant_passages: list[_Passage]
        rrf_score: float | None = None
        dense_rank: int | None = None
        bm25_rank: int | None = None
        metadata: dict[str, Any] | None = None
        chunk_evidence: list[dict[str, Any]] | None = None

    out: list[_Doc] = []
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
        passages: list[_Passage] = []
        if prefer_chunk_evidence and chunk_evidence_raw:
            for item in chunk_evidence_raw:
                text = str(item.get("text") or "").strip()
                if not text:
                    continue
                passages.append(
                    _Passage(
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
                    _Passage(
                        text=text,
                        chunk_id=str(
                            getattr(paragraph, "paragraph_id", "") or f"p-{len(passages)}"
                        ),
                    )
                )
        out.append(
            _Doc(
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


DEFAULT_BENCHMARK = (
    PROJECT_ROOT / "benchmarks" / "legal_v2" / "case_similarity_golden_v1_pilot.jsonl"
)
DEFAULT_QUERY_IDS = ("nalus-cs-pilot-004", "nalus-cs-pilot-016")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--query-ids",
        nargs="+",
        default=list(DEFAULT_QUERY_IDS),
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", "http://qdrant:6333"),
    )
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
    parser.add_argument("--candidate-documents", type=int, default=50)
    parser.add_argument("--ce-candidate-documents", type=int, default=30)
    parser.add_argument("--ce-evidence-pool-limit", type=int, default=40)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
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
        device="cpu",
        candidate_multiplier=1,
        min_candidate_count=1,
        max_candidate_count=max(
            config.dense_candidate_chunks,
            config.bm25_candidate_chunks,
        ),
        lexical_filter_enabled=False,
    )


def _preview(text: str, limit: int = 220) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "…"


def _passage_rows(passages: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for p in passages:
        rows.append(
            {
                "chunk_id": p.chunk_id,
                "selection_slot": p.selection_slot,
                "selection_reason": p.selection_reason,
                "dense_rank": p.dense_rank,
                "bm25_rank": p.bm25_rank,
                "rrf_rank": p.rrf_rank,
                "retrieval_channels": list(p.retrieval_channels or ()),
                "chunk_position": p.chunk_position,
                "section": p.section,
                "text_preview": _preview(p.text),
            }
        )
    return rows


def _classify_new_evidence(ce3_ids: set[str], ce7_passages: list[Any]) -> list[dict[str, Any]]:
    out = []
    for p in ce7_passages:
        if p.chunk_id in ce3_ids:
            continue
        channels = list(p.retrieval_channels or ())
        if p.selection_reason == "diversity_support" or str(p.selection_reason or "").startswith(
            "fallback_after_diversity"
        ):
            source = "new_diversity_evidence"
        elif "dense" in channels and p.dense_rank is not None and (
            p.rrf_rank is None or (p.dense_rank <= (p.rrf_rank or 10**9))
        ):
            source = "new_dense_evidence"
        elif "bm25" in channels and p.bm25_rank is not None:
            source = "new_bm25_evidence"
        elif "rrf" in channels:
            source = "new_rrf_evidence"
        else:
            source = "new_fallback_evidence"
        out.append(
            {
                "chunk_id": p.chunk_id,
                "source_class": source,
                "selection_reason": p.selection_reason,
                "dense_rank": p.dense_rank,
                "bm25_rank": p.bm25_rank,
                "rrf_rank": p.rrf_rank,
            }
        )
    return out


def _score_docs(
    service: CrossEncoderRerankingService,
    query: str,
    docs: list[Any],
) -> Any:
    return service.rerank(query, docs, require_success=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (
        PROJECT_ROOT
        / "artifacts"
        / "legal_v2"
        / "ce_bge_v2m3_p7_diverse_v1"
        / f"{run_id}_diagnostics_004_016"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    items = {
        item.benchmark_id: item
        for item in load_case_similarity_golden_jsonl(args.benchmark)
        if item.benchmark_id in set(args.query_ids)
    }
    missing = [qid for qid in args.query_ids if qid not in items]
    if missing:
        raise SystemExit(f"missing benchmark ids: {missing}")

    retriever_config = LegalV2RetrieverConfig(
        qdrant_collection=args.qdrant_collection,
        bm25_sidecar_path=args.bm25_sidecar_path,
        bm25_index_id=args.bm25_index_id,
        dense_candidate_chunks=80,
        bm25_candidate_chunks=80,
        fused_candidate_chunks=120,
        candidate_documents=args.candidate_documents,
        model_path=os.getenv("EMBEDDING_MODEL_NAME", "/app/models/BAAI/bge-m3"),
    )
    client = QdrantClient(url=args.qdrant_url, prefer_grpc=False)
    embedder = BgeM3Embedder(_embedder_config(retriever_config))
    retriever = build_live_legal_v2_retriever(client, embedder, retriever_config)

    allow_download = os.getenv("NALUS_LEGAL_V2_CE_ALLOW_DOWNLOAD", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    ce3_service = CrossEncoderRerankingService(
        CrossEncoderConfig(
            enabled=True,
            candidate_documents=args.ce_candidate_documents,
            passages_per_document=3,
            passage_selector=FIRST_N_STAGE1_ORDER_V1,
            evidence_pool_limit=args.ce_evidence_pool_limit,
            allow_download=allow_download,
            local_files_only=not allow_download,
            batch_size=int(os.getenv("NALUS_LEGAL_V2_CE_BATCH_SIZE", "16")),
            device=os.getenv("NALUS_LEGAL_V2_CE_DEVICE", "auto"),
            max_length=int(os.getenv("NALUS_LEGAL_V2_CE_MAX_LENGTH", "512")),
        )
    )
    ce7_service = CrossEncoderRerankingService(
        CrossEncoderConfig(
            enabled=True,
            candidate_documents=args.ce_candidate_documents,
            passages_per_document=7,
            passage_selector=DIVERSIFIED_STAGE1_EVIDENCE_V1,
            evidence_pool_limit=args.ce_evidence_pool_limit,
            allow_download=allow_download,
            local_files_only=not allow_download,
            batch_size=int(os.getenv("NALUS_LEGAL_V2_CE_BATCH_SIZE", "16")),
            device=os.getenv("NALUS_LEGAL_V2_CE_DEVICE", "auto"),
            max_length=int(os.getenv("NALUS_LEGAL_V2_CE_MAX_LENGTH", "512")),
        )
    )
    ce3_service._get_provider().load()
    # Reuse same loaded provider for CE-7 (identical model config).
    ce7_service = CrossEncoderRerankingService(
        ce7_service.config,
        provider=ce3_service._get_provider(),
    )

    reports: list[dict[str, Any]] = []
    for query_id in args.query_ids:
        item = items[query_id]
        expected = (
            normalize_ecli(item.expected_primary_ecli)
            if item.expected_primary_ecli and is_valid_ecli(item.expected_primary_ecli)
            else None
        )
        query_spec = build_query_spec_v2(item.query)
        retrieval = retriever.retrieve(query_spec)
        stage1_docs = _stage1_docs_from_retrieval(
            retrieval.documents,
            limit=args.ce_candidate_documents,
            evidence_limit=args.ce_evidence_pool_limit,
            prefer_chunk_evidence=True,
        )
        stage1_rank = {
            doc.ecli: doc.rank for doc in stage1_docs if is_valid_ecli(doc.ecli)
        }

        ce3 = _score_docs(ce3_service, item.query, stage1_docs)
        ce7 = _score_docs(ce7_service, item.query, stage1_docs)
        ce3_rank = {d.ecli: d.ce_rank for d in ce3.documents}
        ce7_rank = {d.ecli: d.ce_rank for d in ce7.documents}
        ce7_by_ecli = {d.ecli: d for d in ce7.documents}
        ce3_by_ecli = {d.ecli: d for d in ce3.documents}

        focus_eclis: list[str] = []
        if expected:
            focus_eclis.append(expected)
        for d in ce7.documents[:3]:
            if d.ecli not in focus_eclis:
                focus_eclis.append(d.ecli)
        # Always include CE-3 top3 competitors for audit.
        for d in ce3.documents[:3]:
            if d.ecli not in focus_eclis:
                focus_eclis.append(d.ecli)

        focus_reports = []
        for ecli in focus_eclis:
            source_doc = next((d for d in stage1_docs if d.ecli == ecli), None)
            if source_doc is None:
                continue
            evidence = evidence_records_from_stage1_doc(source_doc)
            skeleton = RerankCandidate(
                ecli=ecli,
                stage1_rank=source_doc.rank,
                stage1_score=source_doc.score,
                passages=(),
                evidence_pool=evidence,
            )
            ce3_cand, _ = build_candidates_from_stage1_docs(
                [source_doc],
                max_documents=1,
                max_passages=3,
                passage_selector_name=FIRST_N_STAGE1_ORDER_V1,
            )
            ce7_cand, _ = build_candidates_from_stage1_docs(
                [source_doc],
                max_documents=1,
                max_passages=7,
                passage_selector_name=DIVERSIFIED_STAGE1_EVIDENCE_V1,
            )
            ce3_passages = list(ce3_cand[0].passages) if ce3_cand else []
            ce7_passages = list(ce7_cand[0].passages) if ce7_cand else []
            ce3_ids = {p.chunk_id for p in ce3_passages}
            ce7_doc = ce7_by_ecli.get(ecli)
            ce3_doc = ce3_by_ecli.get(ecli)
            best_ce7 = None
            if ce7_doc and ce7_doc.passage_scores:
                best_ce7 = max(ce7_doc.passage_scores, key=lambda s: s.score)
            focus_reports.append(
                {
                    "ecli": ecli,
                    "is_expected_primary": ecli == expected,
                    "stage1_rank": stage1_rank.get(ecli),
                    "available_evidence_chunks": [
                        {
                            "chunk_id": row.chunk_id,
                            "dense_rank": row.dense_rank,
                            "bm25_rank": row.bm25_rank,
                            "rrf_rank": row.rrf_rank,
                            "retrieval_channels": list(row.retrieval_channels or ()),
                            "chunk_position": row.chunk_position,
                            "text_preview": _preview(row.text),
                        }
                        for row in evidence
                    ],
                    "ce3_selected": _passage_rows(ce3_passages),
                    "ce7_selected": _passage_rows(ce7_passages),
                    "newly_added_ce7": _classify_new_evidence(ce3_ids, ce7_passages),
                    "ce3_document_max_score": None if ce3_doc is None else ce3_doc.ce_score,
                    "ce7_document_max_score": None if ce7_doc is None else ce7_doc.ce_score,
                    "ce3_final_rank": ce3_rank.get(ecli),
                    "ce7_final_rank": ce7_rank.get(ecli),
                    "ce7_passage_scores": [
                        {
                            "chunk_id": s.chunk_id,
                            "score": s.score,
                            "passage_index": s.passage_index,
                            "present_in_ce3": s.chunk_id in ce3_ids,
                        }
                        for s in (ce7_doc.passage_scores if ce7_doc else ())
                    ],
                    "best_ce7_passage": None
                    if best_ce7 is None
                    else {
                        "chunk_id": best_ce7.chunk_id,
                        "score": best_ce7.score,
                        "was_present_in_ce3": best_ce7.chunk_id in ce3_ids,
                    },
                    "requested_vs_selected": {
                        "ce3_requested": 3,
                        "ce3_selected": len(ce3_passages),
                        "ce7_requested": 7,
                        "ce7_selected": len(ce7_passages),
                    },
                    "skeleton_stage1_rank_unused_by_selector": skeleton.stage1_rank,
                }
            )

        reports.append(
            {
                "benchmark_id": query_id,
                "expected_primary_ecli": expected,
                "query_preview": _preview(item.query, 160),
                "stage1_top10": [
                    {"rank": d.rank, "ecli": d.ecli, "score": d.score}
                    for d in stage1_docs[:10]
                ],
                "ce3_top10": [
                    {"rank": d.ce_rank, "ecli": d.ecli, "ce_score": d.ce_score}
                    for d in ce3.documents[:10]
                ],
                "ce7_top10": [
                    {"rank": d.ce_rank, "ecli": d.ecli, "ce_score": d.ce_score}
                    for d in ce7.documents[:10]
                ],
                "expected_ranks": {
                    "stage1": stage1_rank.get(expected) if expected else None,
                    "ce3": ce3_rank.get(expected) if expected else None,
                    "ce7": ce7_rank.get(expected) if expected else None,
                },
                "focus_documents": focus_reports,
                "ce3_diagnostics": ce3.diagnostics.as_dict(),
                "ce7_diagnostics": ce7.diagnostics.as_dict(),
            }
        )

    payload = {
        "experiment": "ce_bge_v2m3_p7_diverse_v1",
        "run_id": run_id,
        "elapsed_s": time.perf_counter() - started,
        "frozen_reference": {
            "model": "BAAI/bge-reranker-v2-m3",
            "candidate_documents": args.ce_candidate_documents,
            "ce3_passages_per_document": 3,
            "ce3_selector": FIRST_N_STAGE1_ORDER_V1,
            "ce7_passages_per_document": 7,
            "ce7_selector": DIVERSIFIED_STAGE1_EVIDENCE_V1,
            "document_aggregation": "max",
            "evidence_pool_limit": args.ce_evidence_pool_limit,
        },
        "queries": reports,
    }
    (output_dir / "ce7_passage_coverage_diagnostics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    md_lines = [
        "# CE-7 passage coverage diagnostics",
        "",
        f"run_id: `{run_id}`",
        "",
    ]
    for report in reports:
        md_lines.extend(
            [
                f"## {report['benchmark_id']}",
                "",
                f"- expected: `{report['expected_primary_ecli']}`",
                f"- Stage1 rank: `{report['expected_ranks']['stage1']}`",
                f"- CE-3 rank: `{report['expected_ranks']['ce3']}`",
                f"- CE-7 rank: `{report['expected_ranks']['ce7']}`",
                "",
            ]
        )
        for focus in report["focus_documents"]:
            if not focus["is_expected_primary"]:
                continue
            md_lines.append("### Expected primary evidence")
            md_lines.append("")
            md_lines.append("CE-3 selected:")
            for row in focus["ce3_selected"]:
                md_lines.append(
                    f"- `{row['chunk_id']}` slot={row['selection_slot']} "
                    f"rrf={row['rrf_rank']} dense={row['dense_rank']} bm25={row['bm25_rank']}"
                )
            md_lines.append("")
            md_lines.append("CE-7 selected:")
            for row in focus["ce7_selected"]:
                md_lines.append(
                    f"- `{row['chunk_id']}` slot={row['selection_slot']} "
                    f"reason={row['selection_reason']} "
                    f"rrf={row['rrf_rank']} dense={row['dense_rank']} bm25={row['bm25_rank']}"
                )
            md_lines.append("")
            best = focus.get("best_ce7_passage") or {}
            md_lines.append(
                f"Best CE-7 passage: `{best.get('chunk_id')}` "
                f"score={best.get('score')} present_in_ce3={best.get('was_present_in_ce3')}"
            )
            md_lines.append("")
    (output_dir / "ce7_passage_coverage_diagnostics.md").write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )
    print(f"output_dir={output_dir}")
    for report in reports:
        print(
            f"{report['benchmark_id']}: stage1={report['expected_ranks']['stage1']} "
            f"ce3={report['expected_ranks']['ce3']} ce7={report['expected_ranks']['ce7']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
