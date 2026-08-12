#!/usr/bin/env python3
"""Evaluate pure ColBERT retrieval on the case-similarity golden v1 pilot.

Uses the existing ColBERT async module only:
  query → ColbertRetriever.retrieve → document-level dedupe → shared metrics.

No BM25, RRF, BGE-M3, CE, or hybrid fusion. Does not rebuild the ColBERT index.
Does not modify FAST/CE profiles.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import json
import math
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
    dedupe_document_ids,
    evaluate_ranked_documents,
)
from app.rag.legal_v2.benchmark.case_similarity_golden import (  # noqa: E402
    DEFAULT_PILOT_DATASET,
    load_case_similarity_golden_jsonl,
)
from app.rag.legal_v2.identity import normalize_ecli  # noqa: E402
from app.rag.legal_v2.retrieve.colbert import (  # noqa: E402
    DEFAULT_COLBERT_MODEL,
    DEFAULT_INDEX_NAME,
    ColbertConfig,
    ColbertRetriever,
    PyLateColbertBackend,
)
from app.rag.legal_v2.retrieve.colbert.mapping import load_mapping_jsonl  # noqa: E402

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "legal_v2"
    / "chunking_ab_pilot_300_v1"
    / "colbert_v1"
    / "eval"
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
CRITICAL_QUERY_IDS = ("nalus-cs-pilot-002", "nalus-cs-pilot-004")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", type=Path, default=DEFAULT_PILOT_DATASET)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_DIR)
    p.add_argument("--mapping-path", type=Path, default=DEFAULT_MAPPING)
    p.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--fast-baseline", type=Path, default=DEFAULT_FAST_BASELINE)
    p.add_argument("--ce-baseline", type=Path, default=DEFAULT_CE_BASELINE)
    p.add_argument("--model", default=DEFAULT_COLBERT_MODEL)
    p.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    p.add_argument("--device", default="cuda")
    p.add_argument("--top-k", type=int, default=10, help="Document-level TOP-K for metrics.")
    p.add_argument(
        "--chunk-pool",
        type=int,
        default=50,
        help="Chunk-level ColBERT pool before document dedupe.",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow HF model download if weights are not cached.",
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
    return {row.document_id for row in mapping.rows.values() if row.document_id}


def _aggregate_document_hits(
    hits: list[Any],
    *,
    top_k: int,
) -> tuple[list[str], list[RetrievedDocumentScore], list[dict[str, Any]]]:
    """Collapse chunk hits to first-seen document order (same as hybrid eval)."""
    ordered_docs: list[str] = []
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in hits:
        doc_id = str(getattr(hit, "document_id", "") or "").strip()
        if not doc_id:
            continue
        key = doc_id.upper()
        evidence.append(
            {
                "rank": int(getattr(hit, "rank", 0) or 0),
                "document_id": doc_id,
                "chunk_id": str(getattr(hit, "chunk_id", "") or ""),
                "score": float(getattr(hit, "score", 0.0) or 0.0),
                "text": str(getattr(hit, "text", "") or ""),
                "metadata": dict(getattr(hit, "metadata", {}) or {}),
            }
        )
        if key in seen:
            continue
        seen.add(key)
        ordered_docs.append(doc_id)
        if len(ordered_docs) >= top_k:
            break
    ranked = dedupe_document_ids(ordered_docs)[:top_k]
    results = [
        RetrievedDocumentScore(
            rank=index,
            document_id=doc_id,
            ecli=doc_id if doc_id.upper().startswith("ECLI:") else None,
            canonical_document_id=doc_id if doc_id.upper().startswith("ECLI:") else None,
            score=next(
                (float(row["score"]) for row in evidence if row["document_id"] == doc_id),
                None,
            ),
        )
        for index, doc_id in enumerate(ranked, start=1)
    ]
    return ranked, results, evidence


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


def _load_baseline_metrics(path: Path, key: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") or {}
    if key not in metrics:
        raise SystemExit(f"baseline key {key!r} missing in {path}")
    return dict(metrics[key])


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(b) - float(a)


def _pairwise_verdict(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    """Weighted comparison: higher Hits/MRR better; lower mean rank / HN better."""
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
    reasons: list[str] = []
    for key, weight, higher_better in weights:
        lv = left.get(key)
        rv = right.get(key)
        if lv is None or rv is None:
            continue
        if math.isclose(float(lv), float(rv), abs_tol=1e-9):
            continue
        right_better = (float(rv) > float(lv)) if higher_better else (float(rv) < float(lv))
        if right_better:
            score += weight
            reasons.append(f"{key}: {right_name} better ({float(lv):.4f} → {float(rv):.4f})")
        else:
            score -= weight
            reasons.append(f"{key}: {left_name} better ({float(lv):.4f} → {float(rv):.4f})")
    if abs(score) < 1e-9:
        winner = "TIE"
    elif score > 0:
        winner = right_name
    else:
        winner = left_name
    return {
        "winner": winner,
        "score": score,
        "reasons": reasons,
        "label": (
            f"{winner} WINS"
            if winner in {left_name, right_name}
            else "TIE"
        ),
    }


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


async def async_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.index_path.exists():
        raise SystemExit(f"ColBERT index missing: {args.index_path}")
    if not args.mapping_path.exists():
        raise SystemExit(f"ColBERT mapping missing: {args.mapping_path}")
    manifest = {}
    if args.manifest_path.exists():
        manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
    if manifest and not bool(manifest.get("COLBERT_INDEX_READY")):
        raise SystemExit("COLBERT_INDEX_READY is false; refusing to score.")

    items = load_case_similarity_golden_jsonl(args.benchmark)
    if len(items) != 20:
        raise SystemExit(f"expected 20 golden rows, found {len(items)}")

    indexed_docs = _document_ids_from_mapping(args.mapping_path)
    source_to_ecli = _build_source_to_ecli_mapping(items)
    git = _git_meta()
    command = " ".join(
        ["python", "scripts/legal_v2/evaluate_colbert_golden_v1.py", *sys.argv[1:]]
    )
    benchmark_sha = hashlib.sha256(args.benchmark.read_bytes()).hexdigest()

    config = ColbertConfig(
        model_name=args.model,
        index_path=args.index_path,
        index_name=args.index_name,
        device=args.device,
        top_k=max(int(args.chunk_pool), int(args.top_k)),
        batch_size=int(args.batch_size),
        concurrency_limit=1,
        mapping_path=args.mapping_path,
        allow_download=bool(args.allow_download),
    )
    backend = PyLateColbertBackend(config)
    retriever = ColbertRetriever(config, backend=backend)
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
                    accepted_alternative_document_ids=list(item.accepted_alternative_document_ids),
                    hard_negative_document_ids=list(item.hard_negative_document_ids),
                    hard_negative_evaluable=item.hard_negative_evaluable,
                    hard_negative_blocker=item.hard_negative_blocker,
                    ranked_document_ids=[],
                    corpus_compatible=False,
                    failure_type="missing_verified_ecli_in_benchmark",
                    error="primary ECLI missing from golden mapping",
                    top_k=args.top_k,
                    expected_primary_source_document_id=item.source_document_id,
                    expected_primary_ecli=item.expected_primary_ecli,
                )
                results.append(row)
                query_details.append(
                    {
                        "query_id": item.benchmark_id,
                        "query_text": item.query,
                        "expected_document_id": item.expected_primary_ecli,
                        "error": row.error,
                        "top10": [],
                        "chunk_evidence": [],
                    }
                )
                continue

            try:
                retrieval = await retriever.retrieve(
                    item.query,
                    top_k=int(args.chunk_pool),
                )
                ranked_ids, retrieved_results, chunk_evidence = _aggregate_document_hits(
                    list(retrieval.hits),
                    top_k=int(args.top_k),
                )
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
                    ranked_document_ids=ranked_ids,
                    retrieved_results=retrieved_results,
                    corpus_compatible=primary_ecli in indexed_docs,
                    top_k=args.top_k,
                    expected_primary_source_document_id=item.source_document_id,
                    expected_primary_ecli=primary_ecli,
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
                    failure_type=FAILURE_RETRIEVAL_ERROR,
                    error=str(exc),
                    top_k=args.top_k,
                    expected_primary_source_document_id=item.source_document_id,
                    expected_primary_ecli=primary_ecli,
                )
                chunk_evidence = []
                retrieval = None

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
                    "hard_negative_ranks": row.hard_negative_ranks,
                    "failure_type": row.failure_type,
                    "error": row.error,
                    "diagnostics": (
                        dict(retrieval.diagnostics) if retrieval is not None else {}
                    ),
                    "top10": [
                        {
                            "rank": doc.rank,
                            "document_id": doc.document_id,
                            "score": doc.score,
                        }
                        for doc in row.retrieved_results
                    ],
                    "chunk_evidence_top": [
                        {
                            "rank": ev["rank"],
                            "document_id": ev["document_id"],
                            "chunk_id": ev["chunk_id"],
                            "score": ev["score"],
                            "text_preview": ev["text"][:180].replace("\n", " "),
                        }
                        for ev in chunk_evidence[:10]
                    ],
                }
            )
            print(
                f"DONE {item.benchmark_id} rank={_rank_display(row.best_positive_rank)} "
                f"hit10={row.hit_at_10}"
            )
    finally:
        await backend.close()

    metrics = _metrics_bundle(results)
    fast_a = _load_baseline_metrics(args.fast_baseline, "A")
    ce_b = _load_baseline_metrics(args.ce_baseline, "B_ce")
    vs_fast = _pairwise_verdict(fast_a, metrics, left_name="FAST", right_name="COLBERT")
    vs_ce = _pairwise_verdict(ce_b, metrics, left_name="CE", right_name="COLBERT")

    critical = {
        qid: next((q for q in query_details if q["query_id"] == qid), None)
        for qid in CRITICAL_QUERY_IDS
    }
    failures = [
        q
        for q in query_details
        if (not q.get("hit_at_10")) or q.get("hard_negative_before_positive") or q.get("error")
    ]

    elapsed_s = time.perf_counter() - started
    payload = {
        "schema": "colbert_golden_eval.v1",
        "benchmark": {
            "profile": "colbert_pure",
            "dataset": str(args.benchmark.as_posix()),
            "dataset_sha256": benchmark_sha,
            "query_count": len(items),
            "top_k_documents": int(args.top_k),
            "chunk_pool": int(args.chunk_pool),
            "model": args.model,
            "library": "pylate",
            "library_version": str(manifest.get("library_version") or "1.6.0"),
            "index_name": args.index_name,
            "index_path": str(args.index_path),
            "mapping_path": str(args.mapping_path),
            "source_corpus": manifest.get(
                "source_corpus",
                "nalus_legal_paragraph_chunks_v2_chunk_ab_v8_b_contextual_300",
            ),
            "source_chunk_count": int(
                (manifest.get("integrity") or {}).get("indexed_chunks")
                or len(load_mapping_jsonl(args.mapping_path))
            ),
            "device": backend.device,
            "requested_device": args.device,
            "COLBERT_INDEX_READY": bool(manifest.get("COLBERT_INDEX_READY", True)),
            "command": command,
            "git": git,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": elapsed_s,
            "bm25": False,
            "rrf": False,
            "bge_m3_retrieval": False,
            "cross_encoder": False,
        },
        "metrics": metrics,
        "comparison": {
            "FAST_A": fast_a,
            "CE_B": ce_b,
            "COLBERT": metrics,
            "deltas": {
                "colbert_minus_fast": {
                    "hit_at_1": _delta(fast_a.get("hit_at_1"), metrics.get("hit_at_1")),
                    "hit_at_3": _delta(fast_a.get("hit_at_3"), metrics.get("hit_at_3")),
                    "hit_at_5": _delta(fast_a.get("hit_at_5"), metrics.get("hit_at_5")),
                    "hit_at_10": _delta(fast_a.get("hit_at_10"), metrics.get("hit_at_10")),
                    "mrr": _delta(fast_a.get("mrr"), metrics.get("mrr")),
                    "mean_relevant_rank": _delta(
                        fast_a.get("mean_relevant_rank"),
                        metrics.get("mean_relevant_rank"),
                    ),
                    "hard_negative_outrank_rate": _delta(
                        fast_a.get("hard_negative_outrank_rate"),
                        metrics.get("hard_negative_outrank_rate"),
                    ),
                },
                "colbert_minus_ce": {
                    "hit_at_1": _delta(ce_b.get("hit_at_1"), metrics.get("hit_at_1")),
                    "hit_at_3": _delta(ce_b.get("hit_at_3"), metrics.get("hit_at_3")),
                    "hit_at_5": _delta(ce_b.get("hit_at_5"), metrics.get("hit_at_5")),
                    "hit_at_10": _delta(ce_b.get("hit_at_10"), metrics.get("hit_at_10")),
                    "mrr": _delta(ce_b.get("mrr"), metrics.get("mrr")),
                    "mean_relevant_rank": _delta(
                        ce_b.get("mean_relevant_rank"),
                        metrics.get("mean_relevant_rank"),
                    ),
                    "hard_negative_outrank_rate": _delta(
                        ce_b.get("hard_negative_outrank_rate"),
                        metrics.get("hard_negative_outrank_rate"),
                    ),
                },
            },
            "table": [
                {
                    "metric": key,
                    "FAST_A": fast_a.get(key),
                    "CE_B": ce_b.get(key),
                    "COLBERT": metrics.get(key),
                    "COLBERT_vs_FAST": _delta(fast_a.get(key), metrics.get(key)),
                    "COLBERT_vs_CE": _delta(ce_b.get(key), metrics.get(key)),
                }
                for key in (
                    "hit_at_1",
                    "hit_at_3",
                    "hit_at_5",
                    "hit_at_10",
                    "mrr",
                    "mean_relevant_rank",
                    "hard_negative_outrank_rate",
                )
            ],
        },
        "queries": query_details,
        "critical_queries": critical,
        "failures": failures,
        "verdict": {
            "COLBERT_GOLDEN_VERDICT": (
                f"Hit@1={_fmt(metrics.get('hit_at_1'))} "
                f"Hit@10={_fmt(metrics.get('hit_at_10'))} "
                f"MRR={_fmt(metrics.get('mrr'))} "
                f"HN={_fmt(metrics.get('hard_negative_outrank_rate'))}"
            ),
            "COLBERT_VS_FAST": vs_fast,
            "COLBERT_VS_CE": vs_ce,
            "note": (
                "Evaluation only. FAST/CE canonical profiles remain unchanged. "
                "No ColBERT+CE and no production activation in this step."
            ),
        },
        "raw_eval_results": [row.model_dump(mode="json") for row in results],
    }

    json_path = output_dir / "COLBERT_GOLDEN_RESULTS.json"
    md_path = output_dir / "COLBERT_GOLDEN_RESULTS.md"
    html_path = output_dir / "COLBERT_GOLDEN_RESULTS.html"
    jsonl_path = output_dir / "case_similarity_retrieval_results.jsonl"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    jsonl_path.write_text(
        "\n".join(row.model_dump_json() for row in results) + ("\n" if results else ""),
        encoding="utf-8",
    )
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    html_path.write_text(_render_html(payload), encoding="utf-8")

    print(f"WROTE {json_path}")
    print(f"WROTE {md_path}")
    print(f"WROTE {html_path}")
    print(
        "METRICS "
        f"Hit@1={metrics.get('hit_at_1')} Hit@3={metrics.get('hit_at_3')} "
        f"Hit@5={metrics.get('hit_at_5')} Hit@10={metrics.get('hit_at_10')} "
        f"MRR={metrics.get('mrr')} HN={metrics.get('hard_negative_outrank_rate')}"
    )
    print(f"COLBERT_VS_FAST {vs_fast['label']}")
    print(f"COLBERT_VS_CE {vs_ce['label']}")
    return 0


def _render_markdown(payload: dict[str, Any]) -> str:
    m = payload["metrics"]
    v = payload["verdict"]
    lines = [
        "# ColBERT Golden Evaluation",
        "",
        "## Executive summary",
        "",
        f"- Profile: pure ColBERT (no BM25 / RRF / BGE-M3 / CE)",
        f"- Queries: `{payload['benchmark']['query_count']}`",
        f"- Model: `{payload['benchmark']['model']}`",
        f"- Index: `{payload['benchmark']['index_name']}`",
        f"- Device: `{payload['benchmark']['device']}`",
        f"- Hit@1 / Hit@10 / MRR: "
        f"`{_fmt(m.get('hit_at_1'))}` / `{_fmt(m.get('hit_at_10'))}` / `{_fmt(m.get('mrr'))}`",
        f"- COLBERT VS FAST: **{v['COLBERT_VS_FAST']['label']}**",
        f"- COLBERT VS CE: **{v['COLBERT_VS_CE']['label']}**",
        "",
        "## Metrics",
        "",
        f"| Metric | Value |",
        f"|---|---:|",
        f"| Hit@1 | {_fmt(m.get('hit_at_1'))} |",
        f"| Hit@3 | {_fmt(m.get('hit_at_3'))} |",
        f"| Hit@5 | {_fmt(m.get('hit_at_5'))} |",
        f"| Hit@10 | {_fmt(m.get('hit_at_10'))} |",
        f"| MRR | {_fmt(m.get('mrr'))} |",
        f"| Mean relevant rank | {_fmt(m.get('mean_relevant_rank'))} |",
        f"| HN outrank rate | {_fmt(m.get('hard_negative_outrank_rate'))} |",
        "",
        "## FAST A vs CE B vs ColBERT",
        "",
        "| Metric | FAST A | CE B | ColBERT | ColBERT vs FAST | ColBERT vs CE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["comparison"]["table"]:
        lines.append(
            f"| {row['metric']} | {_fmt(row['FAST_A'])} | {_fmt(row['CE_B'])} | "
            f"{_fmt(row['COLBERT'])} | {_fmt(row['COLBERT_vs_FAST'])} | "
            f"{_fmt(row['COLBERT_vs_CE'])} |"
        )
    lines.extend(["", "## Critical queries", ""])
    for qid in CRITICAL_QUERY_IDS:
        q = payload["critical_queries"].get(qid) or {}
        lines.append(
            f"- `{qid}`: relevant_rank=`{_rank_display(q.get('relevant_rank'))}` "
            f"Hit@1=`{q.get('hit_at_1')}` Hit@10=`{q.get('hit_at_10')}` "
            f"RR=`{_fmt(q.get('reciprocal_rank'))}`"
        )
    lines.extend(["", "## Query-by-query", ""])
    for q in payload["queries"]:
        lines.append(
            f"- `{q['query_id']}` rank=`{q.get('relevant_rank_display')}` "
            f"Hit@1/3/5/10=`{q.get('hit_at_1')}/{q.get('hit_at_3')}/"
            f"{q.get('hit_at_5')}/{q.get('hit_at_10')}` "
            f"HN_before=`{q.get('hard_negative_before_positive')}`"
        )
    lines.extend(["", "## Failures / weak rows", ""])
    if not payload["failures"]:
        lines.append("- none")
    else:
        for q in payload["failures"]:
            lines.append(
                f"- `{q['query_id']}` rank=`{q.get('relevant_rank_display')}` "
                f"failure=`{q.get('failure_type')}` error=`{q.get('error')}`"
            )
    lines.extend(
        [
            "",
            "## Verdicts",
            "",
            f"- COLBERT GOLDEN: {v['COLBERT_GOLDEN_VERDICT']}",
            f"- COLBERT VS FAST: **{v['COLBERT_VS_FAST']['label']}**",
            f"- COLBERT VS CE: **{v['COLBERT_VS_CE']['label']}**",
            f"- Note: {v['note']}",
            "",
        ]
    )
    return "\n".join(lines)


def _render_html(payload: dict[str, Any]) -> str:
    m = payload["metrics"]
    v = payload["verdict"]
    esc = html.escape

    def card(title: str, value: str) -> str:
        return (
            f'<div class="card"><div class="k">{esc(title)}</div>'
            f'<div class="v">{esc(value)}</div></div>'
        )

    table_rows = []
    for row in payload["comparison"]["table"]:
        table_rows.append(
            "<tr>"
            f"<td>{esc(row['metric'])}</td>"
            f"<td>{esc(_fmt(row['FAST_A']))}</td>"
            f"<td>{esc(_fmt(row['CE_B']))}</td>"
            f"<td>{esc(_fmt(row['COLBERT']))}</td>"
            f"<td>{esc(_fmt(row['COLBERT_vs_FAST']))}</td>"
            f"<td>{esc(_fmt(row['COLBERT_vs_CE']))}</td>"
            "</tr>"
        )

    critical_html = []
    for qid in CRITICAL_QUERY_IDS:
        q = payload["critical_queries"].get(qid) or {}
        critical_html.append(
            "<tr>"
            f"<td>{esc(qid)}</td>"
            f"<td>{esc(_rank_display(q.get('relevant_rank')))}</td>"
            f"<td>{esc(str(q.get('hit_at_1')))}</td>"
            f"<td>{esc(str(q.get('hit_at_10')))}</td>"
            f"<td>{esc(_fmt(q.get('reciprocal_rank')))}</td>"
            "</tr>"
        )

    query_blocks = []
    for q in payload["queries"]:
        top = "".join(
            "<tr>"
            f"<td>{esc(str(h['rank']))}</td>"
            f"<td>{esc(str(h['document_id']))}</td>"
            f"<td>{esc(_fmt(h.get('score'), 3))}</td>"
            "</tr>"
            for h in q.get("top10") or []
        )
        query_blocks.append(
            f"""
<section class="q">
  <h3>{esc(q['query_id'])} · rank {_rank_display(q.get('relevant_rank'))}</h3>
  <p class="qt">{esc((q.get('query_text') or '')[:280])}</p>
  <p>Hit@1/3/5/10 =
     {esc(str(q.get('hit_at_1')))}/
     {esc(str(q.get('hit_at_3')))}/
     {esc(str(q.get('hit_at_5')))}/
     {esc(str(q.get('hit_at_10')))}
     · HN before positive = {esc(str(q.get('hard_negative_before_positive')))}
  </p>
  <table><thead><tr><th>Rank</th><th>Document</th><th>Score</th></tr></thead>
  <tbody>{top}</tbody></table>
</section>
"""
        )

    fail_rows = []
    for q in payload["failures"]:
        fail_rows.append(
            "<tr>"
            f"<td>{esc(q['query_id'])}</td>"
            f"<td>{esc(_rank_display(q.get('relevant_rank')))}</td>"
            f"<td>{esc(str(q.get('failure_type')))}</td>"
            f"<td>{esc(str(q.get('error') or ''))}</td>"
            "</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>ColBERT Golden Results</title>
<style>
body {{ font-family: Segoe UI, sans-serif; margin: 24px; color: #0f172a; background: #f8fafc; }}
h1,h2,h3 {{ margin: 0 0 10px; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(140px,1fr)); gap: 12px; margin: 16px 0 24px; }}
.card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; }}
.card .k {{ font-size: 12px; color: #64748b; text-transform: uppercase; }}
.card .v {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
.verdict {{ background: #0f766e; color: #fff; border-radius: 12px; padding: 16px 18px; margin-bottom: 18px; }}
.verdict.ce {{ background: #1d4ed8; }}
table {{ width: 100%; border-collapse: collapse; background: #fff; margin: 10px 0 22px; }}
th, td {{ border: 1px solid #e2e8f0; padding: 8px 10px; text-align: left; font-size: 14px; }}
th {{ background: #f1f5f9; }}
.q {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; margin-bottom: 12px; }}
.qt {{ color: #334155; }}
.meta {{ color: #64748b; font-size: 13px; margin-bottom: 18px; }}
</style></head><body>
<div class="verdict">
  <h1>COLBERT GOLDEN VERDICT</h1>
  <p>{esc(v['COLBERT_GOLDEN_VERDICT'])}</p>
  <p><strong>vs FAST:</strong> {esc(v['COLBERT_VS_FAST']['label'])}
     &nbsp;|&nbsp; <strong>vs CE:</strong> {esc(v['COLBERT_VS_CE']['label'])}</p>
</div>
<p class="meta">
  model={esc(payload['benchmark']['model'])} ·
  index={esc(payload['benchmark']['index_name'])} ·
  device={esc(str(payload['benchmark']['device']))} ·
  queries={esc(str(payload['benchmark']['query_count']))} ·
  git={esc(str(payload['benchmark']['git'].get('git_head',''))[:12])}
</p>
<div class="cards">
  {card('Hit@1', _fmt(m.get('hit_at_1')))}
  {card('Hit@3', _fmt(m.get('hit_at_3')))}
  {card('Hit@5', _fmt(m.get('hit_at_5')))}
  {card('Hit@10', _fmt(m.get('hit_at_10')))}
  {card('MRR', _fmt(m.get('mrr')))}
  {card('Mean rank', _fmt(m.get('mean_relevant_rank')))}
  {card('HN outrank', _fmt(m.get('hard_negative_outrank_rate')))}
</div>
<h2>FAST A vs CE B vs ColBERT</h2>
<table>
<thead><tr><th>Metric</th><th>FAST A</th><th>CE B</th><th>ColBERT</th><th>vs FAST</th><th>vs CE</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody>
</table>
<h2>Critical queries (002 / 004)</h2>
<table>
<thead><tr><th>Query</th><th>Relevant rank</th><th>Hit@1</th><th>Hit@10</th><th>RR</th></tr></thead>
<tbody>{''.join(critical_html)}</tbody>
</table>
<h2>Failures / weak rows</h2>
<table>
<thead><tr><th>Query</th><th>Rank</th><th>Failure</th><th>Error</th></tr></thead>
<tbody>{''.join(fail_rows) if fail_rows else '<tr><td colspan="4">none</td></tr>'}</tbody>
</table>
<h2>Query-by-query TOP10</h2>
{''.join(query_blocks)}
</body></html>
"""


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
