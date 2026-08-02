"""Bounded 16-query semantic smoke with the selected hybrid thinking policy.

Limits:
- 16 stratified diagnostic/tuning queries
- max 2 concurrent queries
- QuerySpec thinking enabled, timeout 120s, max 2 provider attempts
- max 8 candidates into fast non-thinking verifier per query
- max 2 thinking-fallback verifier calls per query
- no full 64-query benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.eval_budget import (  # noqa: E402
    BudgetExhaustedError,
    BudgetLimits,
    BudgetOperation,
    BudgetStopReason,
    EvalBudgetTracker,
    bind_budget_tracker,
    budget_operation_context,
    build_evaluation_fingerprint,
    checksum_text,
    fingerprints_compatible,
)
from app.rag.legal_v2.interpreter import DeepSeekQuerySpecProvider  # noqa: E402
from app.rag.legal_v2.pipeline import search_legal_v2  # noqa: E402
from app.rag.legal_v2.retriever import (  # noqa: E402
    LegalV2RetrieverConfig,
    build_live_legal_v2_retriever,
)
from app.rag.legal_v2.verifier import DeepSeekSemanticVerifierProvider  # noqa: E402
from app.rag.llm.config import effective_llm_config_from_env  # noqa: E402
from app.rag.llm.deepseek_pricing import PRICING_TABLE_VERSION  # noqa: E402
from app.rag.llm.providers.deepseek import DeepSeekThinkingMode  # noqa: E402
from app.rag.retrieval.bge_m3_embedder import BgeM3Embedder  # noqa: E402
from app.rag.retrieval.production_profile import ProductionRetrievalConfig  # noqa: E402

PROMPT_INJECTION_MARKERS = ("ignore previous", "ignore all previous", "system prompt")
BUDGET_STOP_REASONS = {reason.value for reason in BudgetStopReason}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid thinking Legal v2 16-query smoke.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/legal_v2/pilot_600_20260731/universal_quality/reviewed_benchmark_v2.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/legal_v2/pilot_600_20260731/universal_quality/thinking_ab_test",
    )
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://qdrant:6333"))
    parser.add_argument(
        "--qdrant-collection",
        default="nalus_legal_paragraph_chunks_v2_pilot_600",
    )
    parser.add_argument(
        "--bm25-sidecar-path",
        type=Path,
        default=Path("/app/storage/rag/bm25/nalus_legal_paragraph_bm25_v2_pilot_600.sqlite"),
    )
    parser.add_argument("--bm25-index-id", default="nalus_legal_paragraph_bm25_v2_pilot_600")
    parser.add_argument("--query-limit", type=int, default=16)
    parser.add_argument(
        "--query-ids",
        default="",
        help="Comma-separated benchmark query IDs to run (overrides stratified selection).",
    )
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument(
        "--json-name",
        default="hybrid_smoke_16.json",
        help="Output JSON filename under --output-dir.",
    )
    parser.add_argument(
        "--markdown-name",
        default="hybrid_smoke_16.md",
        help="Output Markdown filename under --output-dir.",
    )
    parser.add_argument(
        "--resume-json",
        type=Path,
        default=None,
        help="Resume from prior smoke/eval JSON; keep completed rows and re-run missing/failed QuerySpec rows.",
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable repeated QuerySpec early-stop (recommended for full non-holdout eval).",
    )
    parser.add_argument("--max-cost-usd", type=float, default=None)
    parser.add_argument("--max-provider-calls", type=int, default=None)
    parser.add_argument("--max-queryspec-calls", type=int, default=None)
    parser.add_argument("--max-fast-verifier-calls", type=int, default=None)
    parser.add_argument("--max-thinking-fallback-calls", type=int, default=None)
    parser.add_argument(
        "--dump-full-documents",
        action="store_true",
        help="Include full reconstructed judgment text for each Stage-B candidate and write per-query review Markdown.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key or api_key == "your-api-key-here":
        raise SystemExit("LLM_API_KEY is not configured")

    os.environ.setdefault("NALUS_LEGAL_V2_QUERYSPEC_TIMEOUT_SECONDS", "120")
    os.environ.setdefault("NALUS_LEGAL_V2_VERIFIER_TIMEOUT_SECONDS", "30")
    os.environ.setdefault("NALUS_LEGAL_V2_VERIFIER_THINKING_TIMEOUT_SECONDS", "120")
    os.environ.setdefault("NALUS_LEGAL_V2_VERIFIER_THINKING_FALLBACK", "1")
    os.environ.setdefault("NALUS_LEGAL_V2_VERIFIER_THINKING_FALLBACK_MAX_PER_QUERY", "2")
    os.environ.setdefault("NALUS_LEGAL_V2_VERIFIER_MAX_CANDIDATES_PER_QUERY", "8")
    # Bounded smoke keeps provider retries off for deterministic A/B; full eval allows transient network recovery.
    if args.query_limit > 16:
        os.environ.setdefault("LLM_RETRY", "2")
    else:
        os.environ["LLM_RETRY"] = "0"

    benchmark_text = args.benchmark.read_text(encoding="utf-8")
    benchmark = json.loads(benchmark_text)
    query_ids = [
        item.strip()
        for item in str(args.query_ids or "").split(",")
        if item.strip()
    ]
    if query_ids:
        selected = _select_rows_by_ids(list(benchmark.get("items") or []), query_ids)
        if len(selected) != len(query_ids):
            found = {str(row.get("id")) for row in selected}
            missing = [query_id for query_id in query_ids if query_id not in found]
            raise SystemExit(f"Missing benchmark query IDs: {', '.join(missing)}")
        args.query_limit = len(selected)
    else:
        selected = _select_smoke_rows(list(benchmark.get("items") or []), args.query_limit)
        if len(selected) < args.query_limit:
            raise SystemExit(f"Only {len(selected)} diagnostic/tuning smoke rows available.")

    configured_model = effective_llm_config_from_env().deepseek_model
    budget_limits = BudgetLimits(
        max_cost_usd=args.max_cost_usd,
        max_provider_calls=args.max_provider_calls,
        max_queryspec_calls=args.max_queryspec_calls,
        max_fast_verifier_calls=args.max_fast_verifier_calls,
        max_thinking_fallback_calls=args.max_thinking_fallback_calls,
    )
    index_identity = {
        "qdrant_collection": args.qdrant_collection,
        "bm25_index_id": args.bm25_index_id,
        "bm25_sidecar_path": str(args.bm25_sidecar_path),
    }
    policy_path = (
        PROJECT_ROOT
        / "artifacts/legal_v2/pilot_600_20260731/universal_quality/thinking_ab_test/mode_policy.json"
    )
    policy_fingerprint = (
        checksum_text(policy_path.read_text(encoding="utf-8"))
        if policy_path.exists()
        else "mode_policy_missing"
    )
    evaluation_fingerprint = build_evaluation_fingerprint(
        benchmark_checksum=checksum_text(benchmark_text),
        runtime_policy_fingerprint=policy_fingerprint,
        model_identity=configured_model,
        pricing_table_version=PRICING_TABLE_VERSION,
        budget_limits=budget_limits,
        index_identity=index_identity,
    )
    kept_rows, pending_rows = _partition_resume_rows(
        selected,
        args.resume_json,
        evaluation_fingerprint=evaluation_fingerprint,
    )
    disable_qs_early_stop = bool(args.no_early_stop or args.query_limit > 16)

    from qdrant_client import QdrantClient  # type: ignore[import-not-found]

    config = LegalV2RetrieverConfig(
        qdrant_collection=args.qdrant_collection,
        bm25_sidecar_path=Path(args.bm25_sidecar_path),
        bm25_index_id=args.bm25_index_id,
        dense_candidate_chunks=80,
        bm25_candidate_chunks=80,
        fused_candidate_chunks=120,
        candidate_documents=40,
        returned_verified_documents=8,
        evidence_windows_per_constraint=3,
        model_path=os.getenv("EMBEDDING_MODEL_NAME", "/app/models/BAAI/bge-m3"),
    )
    client = QdrantClient(url=args.qdrant_url, timeout=30)
    retriever = build_live_legal_v2_retriever(
        client,
        BgeM3Embedder(_embedder_config(config)),
        config,
    )
    query_api_key = api_key
    retrieval_lock = threading.Lock()
    budget_tracker = EvalBudgetTracker(
        limits=budget_limits,
        configured_model=configured_model,
        pricing_table_version=PRICING_TABLE_VERSION,
    )
    started = time.perf_counter()
    rows: list[dict[str, Any]] = list(kept_rows)
    stop_reason: str | None = None
    json_path = args.output_dir / args.json_name
    md_path = args.output_dir / args.markdown_name

    def _checkpoint(current_stop: str | None) -> dict[str, Any]:
        return _write_checkpoint(
            output_dir=args.output_dir,
            json_path=json_path,
            md_path=md_path,
            rows=rows,
            query_limit=args.query_limit,
            stop_reason=current_stop,
            started=started,
            resume_kept=len(kept_rows),
            pending_total=len(pending_rows),
            budget_tracker=budget_tracker,
            evaluation_fingerprint=evaluation_fingerprint,
            budget_limits=budget_limits,
        )

    with bind_budget_tracker(budget_tracker):
        with ThreadPoolExecutor(max_workers=max(1, min(2, args.max_workers))) as pool:
            futures = {
                pool.submit(
                    _run_one,
                    row=row,
                    retriever=retriever,
                    api_key=query_api_key,
                    config=config,
                    expected_collection=args.qdrant_collection,
                    expected_bm25=args.bm25_index_id,
                    retrieval_lock=retrieval_lock,
                    budget_tracker=budget_tracker,
                    dump_full_documents=bool(args.dump_full_documents),
                    review_dir=(args.output_dir / "document_reviews") if args.dump_full_documents else None,
                ): row
                for row in pending_rows
            }
            for future in as_completed(futures):
                row_result = future.result()
                rows.append(row_result)
                if budget_tracker.stop_reason and stop_reason is None:
                    stop_reason = budget_tracker.stop_reason
                _checkpoint(stop_reason)
                if row_result.get("prompt_injection_success"):
                    stop_reason = "prompt_injection_success"
                    for pending in futures:
                        pending.cancel()
                    break
                if row_result.get("wrong_index_identity"):
                    stop_reason = "wrong_index_identity"
                    for pending in futures:
                        pending.cancel()
                    break
                if stop_reason in BUDGET_STOP_REASONS:
                    for pending in futures:
                        pending.cancel()
                    break
                if disable_qs_early_stop:
                    continue
                structural_qs_failures = sum(
                    1
                    for item in rows
                    if not item.get("queryspec_schema_valid")
                    and (
                        item.get("status") == "query_interpretation_error"
                        or item.get("interpretation_status") == "failed"
                    )
                    and not _looks_like_transient_network_failure(item)
                )
                if structural_qs_failures >= 5:
                    stop_reason = "repeated_queryspec_schema_failures"
                    for pending in futures:
                        pending.cancel()
                    break

    if budget_tracker.stop_reason and stop_reason is None:
        stop_reason = budget_tracker.stop_reason
    artifact = _checkpoint(stop_reason)
    summary = artifact["summary"]
    print(
        json.dumps(
            {
                "smoke_gate_passed": summary["smoke_gate_passed"],
                "budget_limited": summary.get("budget_limited"),
                "summary": summary,
                "resume_kept": len(kept_rows),
                "rerun_count": len(pending_rows),
            },
            indent=2,
        )
    )
    if stop_reason in BUDGET_STOP_REASONS:
        return 0
    return 0 if summary["smoke_gate_passed"] else 2


def _run_one(
    *,
    row: dict[str, Any],
    retriever: Any,
    api_key: str,
    config: LegalV2RetrieverConfig,
    expected_collection: str,
    expected_bm25: str,
    retrieval_lock: threading.Lock,
    budget_tracker: EvalBudgetTracker | None = None,
    dump_full_documents: bool = False,
    review_dir: Path | None = None,
) -> dict[str, Any]:
    query = str(row.get("query") or "")
    query_id = str(row.get("id") or "")
    with bind_budget_tracker(budget_tracker):
        with budget_operation_context(BudgetOperation.OTHER, query_id=query_id or None):
            return _run_one_inner(
                row=row,
                query=query,
                query_id=query_id,
                retriever=retriever,
                api_key=api_key,
                config=config,
                expected_collection=expected_collection,
                expected_bm25=expected_bm25,
                retrieval_lock=retrieval_lock,
                budget_tracker=budget_tracker,
                dump_full_documents=dump_full_documents,
                review_dir=review_dir,
            )


def _run_one_inner(
    *,
    row: dict[str, Any],
    query: str,
    query_id: str,
    retriever: Any,
    api_key: str,
    config: LegalV2RetrieverConfig,
    expected_collection: str,
    expected_bm25: str,
    retrieval_lock: threading.Lock,
    budget_tracker: EvalBudgetTracker | None,
    dump_full_documents: bool = False,
    review_dir: Path | None = None,
) -> dict[str, Any]:
    if budget_tracker is not None and budget_tracker.is_stopped:
        return {
            "id": row.get("id"),
            "clarification_expected": bool(row.get("clarification_expected")),
            "status": "budget_skipped",
            "interpretation_status": "skipped",
            "queryspec_schema_valid": True,
            "queryspec_calls": 0,
            "fast_verifier_calls": 0,
            "thinking_fallback_calls": 0,
            "fast_verifier_results": [],
            "thinking_verifier_results": [],
            "candidate_documents": [],
            "false_approvals": 0,
            "false_rejections": 0,
            "prompt_injection_success": 0,
            "wrong_index_identity": 0,
            "total_latency_ms": 0.0,
            "stop_reason": budget_tracker.stop_reason,
            "verified_count": 0,
            "rejected_count": 0,
            "budget_skipped": True,
            "provider": {},
        }
    query_provider = DeepSeekQuerySpecProvider(
        api_key,
        thinking=DeepSeekThinkingMode.ENABLED,
        timeout_seconds=120,
        max_tokens=8000,
    )
    fast_verifier = DeepSeekSemanticVerifierProvider(
        api_key,
        thinking=DeepSeekThinkingMode.DISABLED,
        timeout_seconds=30,
        max_tokens=1024,
    )
    thinking_verifier = DeepSeekSemanticVerifierProvider(
        api_key,
        thinking=DeepSeekThinkingMode.ENABLED,
        timeout_seconds=120,
        max_tokens=8000,
    )
    started = time.perf_counter()
    try:
        with retrieval_lock:
            result = search_legal_v2(
                query=query,
                retriever=retriever,
                verifier=fast_verifier,
                thinking_verifier=thinking_verifier,
                config=config,
                query_provider=query_provider,
                debug=True,
                include_full_document_text=dump_full_documents,
            )
    except BudgetExhaustedError as exc:
        if budget_tracker is not None:
            budget_tracker.mark_query_completed(query_id)
        return {
            "id": row.get("id"),
            "clarification_expected": bool(row.get("clarification_expected")),
            "status": "budget_stopped",
            "interpretation_status": "stopped",
            "queryspec_schema_valid": True,
            "queryspec_calls": 0,
            "fast_verifier_calls": 0,
            "thinking_fallback_calls": 0,
            "fast_verifier_results": [],
            "thinking_verifier_results": [],
            "candidate_documents": [],
            "false_approvals": 0,
            "false_rejections": 0,
            "prompt_injection_success": 0,
            "wrong_index_identity": 0,
            "total_latency_ms": (time.perf_counter() - started) * 1000,
            "stop_reason": exc.stop_reason,
            "verified_count": 0,
            "rejected_count": 0,
            "budget_stopped": True,
            "provider": {"error": exc.stop_reason},
        }
    if budget_tracker is not None:
        budget_tracker.mark_query_completed(query_id)
    total_latency_ms = (time.perf_counter() - started) * 1000
    gold = set(_gold_ids(row))
    hard_negatives = set(_hard_negative_ids(row))
    related_only = {
        str(item).strip()
        for item in row.get("related_only_document_ids") or []
        if str(item).strip()
    }
    thinking_calls = int((result.provider or {}).get("thinking_fallback_calls") or 0)
    provider = dict(result.provider or {})
    all_docs = list(result.verified_documents) + list(result.rejected_documents)
    fast_results = []
    thinking_results = []
    candidate_documents = []
    for doc_index, document in enumerate(all_docs, start=1):
        diagnostics = dict(document.verifier_diagnostics or {})
        constraint_results = list(document.constraint_results or [])
        entry = {
            "schema_valid": not bool(diagnostics.get("failed_closed")),
            "evidence_id_valid": not bool(diagnostics.get("failed_closed")),
            "classification": document.relevance_classification,
            "status": document.status,
            "failed_closed_reason": diagnostics.get("reason") if diagnostics.get("failed_closed") else None,
        }
        if diagnostics.get("thinking_fallback_used"):
            thinking_results.append(entry)
        else:
            fast_results.append(entry)
        benchmark_label = "other"
        if document.document_id in gold:
            benchmark_label = "gold"
        elif document.document_id in hard_negatives:
            benchmark_label = "hard_negative"
        elif document.document_id in related_only:
            benchmark_label = "related_only"
        candidate_documents.append(
            {
                "document_id": document.document_id,
                "ecli": diagnostics.get("ecli") or document.document_id,
                "candidate_rank": diagnostics.get("candidate_rank") or doc_index,
                "dense_rank": document.dense_rank,
                "bm25_rank": document.bm25_rank,
                "rrf_score": document.rrf_score,
                "fast_classification": diagnostics.get("fast_classification"),
                "fast_decision": diagnostics.get("fast_decision"),
                "thinking_classification": (
                    document.relevance_classification
                    if diagnostics.get("thinking_fallback_used")
                    else None
                ),
                "thinking_decision": (
                    diagnostics.get("classification")
                    if diagnostics.get("thinking_fallback_used")
                    else None
                ),
                "final_decision": document.status,
                "relevance_classification": document.relevance_classification,
                "thinking_fallback_used": bool(diagnostics.get("thinking_fallback_used")),
                "thinking_promotion_reason": diagnostics.get("thinking_promotion_reason"),
                "thinking_promotion_applied": diagnostics.get("thinking_promotion_applied"),
                "thinking_promotion_rejected": diagnostics.get("thinking_promotion_rejected"),
                "final_rejection_code": diagnostics.get("final_rejection_code"),
                "constraint_status_summary": diagnostics.get("constraint_status_summary"),
                "constraint_results": [
                    {
                        "constraint_id": item.get("constraint_id"),
                        "status": item.get("status"),
                        "evidence_paragraph_ids": item.get("evidence_paragraph_ids"),
                        "source_of_claim": item.get("source_of_claim"),
                    }
                    for item in constraint_results
                    if isinstance(item, dict)
                ],
                "evidence_ids_by_constraint": {
                    str(item.get("constraint_id")): list(item.get("evidence_paragraph_ids") or [])
                    for item in constraint_results
                    if isinstance(item, dict) and item.get("constraint_id")
                },
                "source_of_claim_by_constraint": {
                    str(item.get("constraint_id")): item.get("source_of_claim")
                    for item in constraint_results
                    if isinstance(item, dict) and item.get("constraint_id")
                },
                "benchmark_label": benchmark_label,
                "document_paragraph_count": diagnostics.get("document_paragraph_count"),
                "document_text_char_count": diagnostics.get("document_text_char_count"),
                "document_text": diagnostics.get("document_text") if dump_full_documents else None,
                "document_paragraphs": (
                    diagnostics.get("document_paragraphs") if dump_full_documents else None
                ),
            }
        )
    false_approvals = sum(
        1
        for document in result.verified_documents
        if document.document_id in hard_negatives
    )
    false_rejections = 0
    if gold and not any(document.document_id in gold for document in result.verified_documents):
        if result.interpretation_status != "failed" and not bool(row.get("clarification_expected")):
            false_rejections = 0
    injection = any(
        marker in query.casefold() for marker in PROMPT_INJECTION_MARKERS
    ) and result.status == "verified_match"
    wrong_index = 0
    index_meta = dict(result.index or {})
    if index_meta.get("collection") not in {None, expected_collection}:
        wrong_index = 1
    if index_meta.get("bm25_index_id") not in {None, expected_bm25}:
        wrong_index = 1
    stop_reason = None
    if injection:
        stop_reason = "prompt_injection_success"
    if wrong_index:
        stop_reason = "wrong_index_identity"
    if budget_tracker is not None and budget_tracker.stop_reason:
        stop_reason = budget_tracker.stop_reason
    row_result = {
        "id": row.get("id"),
        "clarification_expected": bool(row.get("clarification_expected")),
        "status": result.status,
        "interpretation_status": result.interpretation_status,
        "queryspec_schema_valid": result.query_spec_summary is not None,
        "queryspec_calls": 1,
        "fast_verifier_calls": len(fast_results),
        "thinking_fallback_calls": thinking_calls,
        "fast_verifier_results": fast_results,
        "thinking_verifier_results": thinking_results,
        "candidate_documents": candidate_documents,
        "false_approvals": false_approvals,
        "false_rejections": false_rejections,
        "prompt_injection_success": int(injection),
        "wrong_index_identity": wrong_index,
        "total_latency_ms": total_latency_ms,
        "stop_reason": stop_reason,
        "verified_count": len(result.verified_documents),
        "rejected_count": len(result.rejected_documents),
        "retrieval_error_type": (result.diagnostics or {}).get("error_type"),
        "interpretation_reason": provider.get("reason"),
        "interpretation_error": provider.get("error"),
        "provider": {
            "query_interpreter": provider.get("query_interpreter"),
            "thinking_fallback_calls": thinking_calls,
            "reason": provider.get("reason"),
            "error": provider.get("error"),
        },
        "query": query,
        "full_documents_dumped": bool(dump_full_documents),
    }
    if dump_full_documents and review_dir is not None:
        review_path = _write_query_document_review(
            review_dir=review_dir,
            query_id=query_id or "unknown",
            query=query,
            candidate_documents=candidate_documents,
        )
        row_result["document_review_path"] = str(review_path)
    return row_result


def _write_query_document_review(
    *,
    review_dir: Path,
    query_id: str,
    query: str,
    candidate_documents: list[dict[str, Any]],
) -> Path:
    review_dir.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in query_id)
    path = review_dir / f"{safe_id}_full_documents.md"
    lines: list[str] = [
        f"# Document review: `{query_id}`",
        "",
        f"- Query: {query}",
        f"- Candidates: `{len(candidate_documents)}`",
        "",
    ]
    for item in candidate_documents:
        lines.extend(
            [
                f"## Rank {item.get('candidate_rank')} — `{item.get('ecli') or item.get('document_id')}`",
                "",
                f"- document_id: `{item.get('document_id')}`",
                f"- benchmark_label: `{item.get('benchmark_label')}`",
                f"- final_decision: `{item.get('final_decision')}`",
                f"- relevance_classification: `{item.get('relevance_classification')}`",
                f"- fast_decision/class: `{item.get('fast_decision')}` / `{item.get('fast_classification')}`",
                f"- thinking_used: `{item.get('thinking_fallback_used')}`",
                f"- paragraph_count: `{item.get('document_paragraph_count')}`",
                f"- char_count: `{item.get('document_text_char_count')}`",
                "",
                "### Full document text (pipeline reconstruction)",
                "",
                "```text",
                str(item.get("document_text") or ""),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path



def _partition_resume_rows(
    selected: list[dict[str, Any]],
    resume_json: Path | None,
    *,
    evaluation_fingerprint: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if resume_json is None or not resume_json.exists():
        return [], selected
    prior = json.loads(resume_json.read_text(encoding="utf-8"))
    prior_fp = dict(prior.get("evaluation_fingerprint") or {})
    if not fingerprints_compatible(evaluation_fingerprint, prior_fp):
        return [], selected
    prior_by_id = {
        str(item.get("id")): item
        for item in list(prior.get("rows") or [])
        if item.get("id") is not None
    }
    kept: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for row in selected:
        row_id = str(row.get("id") or "")
        previous = prior_by_id.get(row_id)
        if previous is None:
            pending.append(row)
            continue
        if previous.get("budget_skipped") or previous.get("budget_stopped"):
            pending.append(row)
            continue
        if _row_is_resume_keep(previous):
            kept.append(previous)
        else:
            pending.append(row)
    return kept, pending


def _row_is_resume_keep(row: dict[str, Any]) -> bool:
    if _looks_like_transient_network_failure(row):
        return False
    if _row_has_verifier_schema_failure(row):
        return False
    if row.get("status") == "query_interpretation_error":
        return False
    if row.get("interpretation_status") == "failed":
        return False
    return bool(row.get("queryspec_schema_valid")) or row.get("status") in {
        "unverifiable_query",
        "verified_match",
        "no_verified_results",
    }


def _row_has_verifier_schema_failure(row: dict[str, Any]) -> bool:
    verifier_rows = [
        *(row.get("fast_verifier_results") or []),
        *(row.get("thinking_verifier_results") or []),
        *(row.get("thinking_fallback_results") or []),
    ]
    return any(not bool(item.get("schema_valid")) for item in verifier_rows)


def _looks_like_transient_network_failure(row: dict[str, Any]) -> bool:
    verifier_reasons = " ".join(
        str(item.get("failed_closed_reason") or "")
        for item in [
            *(row.get("fast_verifier_results") or []),
            *(row.get("thinking_verifier_results") or []),
            *(row.get("thinking_fallback_results") or []),
        ]
    )
    haystack = " ".join(
        [
            str(row.get("status") or ""),
            str(row.get("interpretation_status") or ""),
            str(row.get("interpretation_reason") or ""),
            str(row.get("interpretation_error") or ""),
            str((row.get("provider") or {}).get("reason") or ""),
            str((row.get("provider") or {}).get("error") or ""),
            verifier_reasons,
        ]
    ).lower()
    return (
        "network" in haystack
        or "name or service not known" in haystack
        or "disconnected" in haystack
        or "empty_message_content" in haystack
        or "timeout" in haystack
    )

def _verifier_result_ok_for_smoke(entry: dict[str, Any]) -> bool:
    if bool(entry.get("schema_valid")):
        return True
    reason = str(entry.get("failed_closed_reason") or "").lower()
    # Provider empty/timeout failures fail-closed safely; do not fail the smoke schema gate.
    return any(
        token in reason
        for token in (
            "empty_message_content",
            "timeout",
            "network_error",
            "name or service not known",
            "disconnected",
        )
    )


def _write_checkpoint(
    *,
    output_dir: Path,
    json_path: Path,
    md_path: Path,
    rows: list[dict[str, Any]],
    query_limit: int,
    stop_reason: str | None,
    started: float,
    resume_kept: int,
    pending_total: int,
    budget_tracker: EvalBudgetTracker,
    evaluation_fingerprint: dict[str, Any],
    budget_limits: BudgetLimits,
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda item: str(item.get("id") or ""))
    quality_rows = [
        item
        for item in ordered
        if not item.get("budget_skipped") and not item.get("budget_stopped")
    ]
    total_latencies = [
        float(item["total_latency_ms"]) for item in quality_rows if item.get("total_latency_ms") is not None
    ]
    queryspec_schema_ok = all(bool(item.get("queryspec_schema_valid")) for item in quality_rows) if quality_rows else True
    fast_schema_ok = all(
        all(_verifier_result_ok_for_smoke(vr) for vr in item.get("fast_verifier_results") or [])
        for item in quality_rows
    )
    thinking_schema_ok = all(
        all(_verifier_result_ok_for_smoke(vr) for vr in item.get("thinking_verifier_results") or [])
        for item in quality_rows
    )
    evidence_ok = all(
        all(_verifier_result_ok_for_smoke(vr) for vr in (item.get("fast_verifier_results") or []))
        and all(_verifier_result_ok_for_smoke(vr) for vr in (item.get("thinking_verifier_results") or []))
        for item in quality_rows
    )
    prompt_injection = sum(int(item.get("prompt_injection_success") or 0) for item in ordered)
    wrong_index = sum(int(item.get("wrong_index_identity") or 0) for item in ordered)
    retrieval_errors = sum(1 for item in quality_rows if item.get("status") == "retrieval_error")
    expected_verifier_rows = [
        item
        for item in quality_rows
        if item.get("status") not in {"unverifiable_query", "query_interpretation_error"}
        and not item.get("clarification_expected")
    ]
    verifier_calls_total = sum(int(item.get("fast_verifier_calls") or 0) for item in quality_rows) + sum(
        int(item.get("thinking_fallback_calls") or 0) for item in quality_rows
    )
    budget_limited = stop_reason in BUDGET_STOP_REASONS
    quality_ok = (
        queryspec_schema_ok
        and fast_schema_ok
        and thinking_schema_ok
        and evidence_ok
        and prompt_injection == 0
        and wrong_index == 0
        and retrieval_errors == 0
        and (
            verifier_calls_total > 0
            or not expected_verifier_rows
            or budget_limited
        )
        and all(
            int(item.get("fast_verifier_calls") or 0) + int(item.get("thinking_fallback_calls") or 0) > 0
            for item in expected_verifier_rows
        )
    )
    smoke_passed = quality_ok and (
        (stop_reason is None and len(ordered) == query_limit)
        or budget_limited
    )
    budget_summary = budget_tracker.summary()
    summary = {
        **budget_summary,
        "queries_completed": len(quality_rows),
        "queries_selected": query_limit,
        "queryspec_calls": sum(int(item.get("queryspec_calls") or 0) for item in quality_rows),
        "fast_verifier_calls": sum(int(item.get("fast_verifier_calls") or 0) for item in quality_rows),
        "thinking_fallback_calls": sum(int(item.get("thinking_fallback_calls") or 0) for item in quality_rows),
        "retrieval_errors": retrieval_errors,
        "average_total_latency_ms": _mean(total_latencies),
        "p50_total_latency_ms": _percentile(total_latencies, 50),
        "p95_total_latency_ms": _percentile(total_latencies, 95),
        "queryspec_schema_success": queryspec_schema_ok,
        "fast_verifier_schema_success": fast_schema_ok,
        "thinking_fallback_schema_success": thinking_schema_ok,
        "evidence_id_success": evidence_ok,
        "false_approvals": sum(int(item.get("false_approvals") or 0) for item in quality_rows),
        "false_rejections": sum(int(item.get("false_rejections") or 0) for item in quality_rows),
        "prompt_injection_success": prompt_injection,
        "wrong_index_identity": wrong_index,
        "stop_reason": stop_reason or budget_summary.get("stop_reason"),
        "budget_limited": budget_limited,
        "smoke_gate_passed": smoke_passed,
        "elapsed_seconds": time.perf_counter() - started,
        "resume_kept": resume_kept,
        "pending_total": pending_total,
        "interpretation_failures": sum(
            1 for item in quality_rows if item.get("status") == "query_interpretation_error"
        ),
        "transient_network_failures": sum(
            1 for item in quality_rows if _looks_like_transient_network_failure(item)
        ),
        "budget_limits": budget_limits.to_dict(),
        "pricing_table_version": PRICING_TABLE_VERSION,
    }
    artifact = {
        "schema": "legal_v2_thinking_hybrid_smoke_v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "policy": "mode_policy.json",
        "evaluation_fingerprint": evaluation_fingerprint,
        "summary": summary,
        "rows": ordered,
        "secrets_logged": False,
        "prompts_logged": False,
        "reasoning_content_persisted": False,
        "stage_a_modified": False,
        "full_64_evaluation_run": False,
        "nonholdout_full_evaluation": query_limit >= 59,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                f"# Hybrid thinking {query_limit}-query evaluation",
                "",
                f"- Generated: `{artifact['generated_at']}`",
                f"- Smoke gate passed: `{smoke_passed}`",
                f"- Budget limited: `{budget_limited}`",
                f"- Queries completed: `{summary['queries_completed']}`",
                f"- QuerySpec calls: `{summary['queryspec_calls']}`",
                f"- Fast verifier calls: `{summary['fast_verifier_calls']}`",
                f"- Thinking fallback calls: `{summary['thinking_fallback_calls']}`",
                f"- Interpretation failures: `{summary['interpretation_failures']}`",
                f"- Pricing table: `{PRICING_TABLE_VERSION}`",
                f"- Actual cost USD: `{summary['actual_cost_usd']}`",
                f"- Configured cost budget USD: `{summary['configured_cost_budget_usd']}`",
                f"- Budget remaining USD: `{summary['budget_remaining_usd']}`",
                f"- Stop reason: `{stop_reason}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return artifact


def _select_rows_by_ids(rows: list[dict[str, Any]], query_ids: list[str]) -> list[dict[str, Any]]:
    wanted = {query_id: index for index, query_id in enumerate(query_ids)}
    selected: list[dict[str, Any] | None] = [None] * len(query_ids)
    for row in rows:
        row_id = str(row.get("id") or "").strip()
        if row_id in wanted:
            selected[wanted[row_id]] = row
    return [row for row in selected if row is not None]


def _select_smoke_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    preferred = [
        row
        for row in rows
        if str(row.get("benchmark_split") or "") in {"diagnostic", "tuning"}
        and not bool(row.get("holdout"))
    ]
    if len(preferred) < limit:
        preferred = [row for row in rows if str(row.get("benchmark_split") or "") != "holdout"]
    # Stratify lightly by domain then take first N.
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for row in preferred:
        domain = str(row.get("legal_domain") or "unknown")
        by_domain.setdefault(domain, []).append(row)
    selected: list[dict[str, Any]] = []
    while len(selected) < limit and any(by_domain.values()):
        for domain in sorted(by_domain):
            bucket = by_domain[domain]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            if len(selected) >= limit:
                break
    return selected[:limit]


def _gold_ids(row: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in ("strongly_relevant_document_ids", "materially_relevant_document_ids", "relevant_document_ids"):
        for item in row.get(key) or []:
            text = str(item).strip()
            if text:
                values.add(text)
    return values


def _hard_negative_ids(row: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in ("explicit_hard_negative_document_ids", "hard_negative_document_ids"):
        for item in row.get(key) or []:
            text = str(item).strip()
            if text:
                values.add(text)
    return values


def _embedder_config(config: LegalV2RetrieverConfig) -> ProductionRetrievalConfig:
    from app.rag.legal_v2.indexing import LEGAL_V2_PROFILE

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


def _mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


if __name__ == "__main__":
    raise SystemExit(main())
