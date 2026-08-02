from __future__ import annotations

import argparse
import hashlib
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

from app.rag.legal_v2.evidence import select_evidence_windows  # noqa: E402
from app.rag.legal_v2.interpreter import (  # noqa: E402
    DeepSeekQuerySpecProvider,
    QuerySpecProvider,
    interpret_query_spec_v2,
)
from app.rag.legal_v2.query_spec import QuerySpecV2  # noqa: E402
from app.rag.legal_v2.retriever import (  # noqa: E402
    LegalV2RetrieverConfig,
    build_live_legal_v2_retriever,
)
from app.rag.legal_v2.verifier import (  # noqa: E402
    CandidateDocumentForVerification,
    DeepSeekSemanticVerifierProvider,
    EvidenceWindowForConstraint,
    SemanticVerifierProvider,
    deterministic_verification_gate,
    run_semantic_verifier,
)
from app.rag.retrieval.bge_m3_embedder import BgeM3Embedder  # noqa: E402
from app.rag.retrieval.production_profile import ProductionRetrievalConfig  # noqa: E402


PROMPT_SCHEMA_VERSION = "legal_v2_semantic_verifier_v4_adapter_fix"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded DeepSeek semantic smoke for Legal Retrieval v2."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/legal_v2/pilot_600_20260731/universal_quality/reviewed_benchmark.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/legal_v2/pilot_600_20260731/universal_quality",
    )
    parser.add_argument("--qdrant-url", default="http://qdrant:6333")
    parser.add_argument(
        "--qdrant-collection",
        default="nalus_legal_paragraph_chunks_v2_pilot_600",
    )
    parser.add_argument(
        "--bm25-sidecar-path",
        type=Path,
        default=Path(
            "/app/storage/rag/bm25/nalus_legal_paragraph_bm25_v2_pilot_600.sqlite"
        ),
    )
    parser.add_argument(
        "--bm25-index-id",
        default="nalus_legal_paragraph_bm25_v2_pilot_600",
    )
    parser.add_argument("--query-limit", type=int, default=16)
    parser.add_argument("--candidate-documents", type=int, default=8)
    parser.add_argument("--semantic-candidates-per-query", type=int, default=2)
    parser.add_argument("--max-structural-failures", type=int, default=3)
    parser.add_argument("--json-name", default="deepseek_semantic_smoke.json")
    parser.add_argument("--markdown-name", default="deepseek_semantic_smoke.md")
    parser.add_argument(
        "--cache-name",
        default="deepseek_semantic_eval_cache.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key or api_key == "your-api-key-here":
        return _write_blocked(args, "LLM_API_KEY is not configured.")

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    selected = _select_smoke_rows(list(benchmark.get("items") or []), args.query_limit)
    if len(selected) < args.query_limit:
        return _write_blocked(
            args,
            f"Only {len(selected)} diagnostic/tuning smoke rows available.",
        )

    from qdrant_client import QdrantClient  # type: ignore[import-not-found]

    config = LegalV2RetrieverConfig(
        qdrant_collection=args.qdrant_collection,
        bm25_sidecar_path=args.bm25_sidecar_path,
        bm25_index_id=args.bm25_index_id,
        dense_candidate_chunks=80,
        bm25_candidate_chunks=80,
        fused_candidate_chunks=120,
        candidate_documents=args.candidate_documents,
        returned_verified_documents=args.semantic_candidates_per_query,
        model_path=os.getenv("EMBEDDING_MODEL_NAME", "/app/models/BAAI/bge-m3"),
    )
    client = QdrantClient(url=args.qdrant_url, timeout=30)
    retriever = build_live_legal_v2_retriever(
        client,
        BgeM3Embedder(_embedder_config(config)),
        config,
    )
    cache_path = args.output_dir / args.cache_name
    cache = _load_cache(cache_path)
    query_provider = CountingQueryProvider(DeepSeekQuerySpecProvider(api_key))
    verifier = CachedCountingVerifier(
        DeepSeekSemanticVerifierProvider(api_key),
        cache=cache,
    )

    rows: list[dict[str, Any]] = []
    structural_failures = 0
    verified_hard_negative_leakage = 0
    unverified_returned_as_verified = 0
    wrong_index_identity = 0
    provider_errors = 0
    latency_values: list[float] = []
    stopped_reason: str | None = None

    for row in selected:
        query = str(row.get("query") or "")
        row_result: dict[str, Any] = {
            "id": row.get("id"),
            "intent_id": row.get("intent_id"),
            "domain": row.get("legal_domain"),
            "split": row.get("benchmark_split"),
            "style": row.get("query_style"),
            "clarification_expected": bool(row.get("clarification_expected")),
            "zero_result_expected": bool(row.get("zero_result_expected")),
            "gold": _gold_ids(row),
            "hard_negative_ids": sorted(_hard_negative_ids(row)),
            "query_spec": None,
            "top_candidate_documents": [],
            "verifier_results": [],
            "provider_error": None,
        }
        interpretation = interpret_query_spec_v2(
            query,
            provider=query_provider,
            allow_deterministic_fallback=False,
        )
        row_result["interpretation_status"] = interpretation.status
        row_result["query_provider"] = interpretation.provider_name
        row_result["query_provider_latency_ms"] = interpretation.latency_ms
        if interpretation.query_spec is None:
            structural_failures += 1
            provider_errors += 1
            row_result["provider_error"] = interpretation.reason
            rows.append(row_result)
            if structural_failures >= args.max_structural_failures:
                stopped_reason = "repeated_query_interpreter_structural_failures"
                break
            continue

        query_spec = interpretation.query_spec
        row_result["query_spec"] = _safe_query_spec(query_spec)
        if row_result["clarification_expected"] or row_result["zero_result_expected"]:
            rows.append(row_result)
            continue

        try:
            retrieval = retriever.retrieve(query_spec)
        except Exception as exc:  # noqa: BLE001
            structural_failures += 1
            row_result["provider_error"] = f"retrieval_error:{exc.__class__.__name__}"
            rows.append(row_result)
            if structural_failures >= args.max_structural_failures:
                stopped_reason = "repeated_retrieval_failures"
                break
            continue

        if (
            retrieval.diagnostics.get("collection") != args.qdrant_collection
            or retrieval.diagnostics.get("bm25_index_id") != args.bm25_index_id
        ):
            wrong_index_identity += 1
        candidates = retrieval.documents[: args.semantic_candidates_per_query]
        row_result["top_candidate_documents"] = [
            {
                "document_id": candidate.document_id,
                "score": candidate.score,
                "dense_rank": candidate.dense_rank,
                "bm25_rank": candidate.bm25_rank,
                "rrf_score": candidate.rrf_score,
            }
            for candidate in retrieval.documents[: args.candidate_documents]
        ]
        for candidate in candidates:
            windows = select_evidence_windows(
                query_spec=query_spec,
                candidate=candidate,
                max_windows_per_constraint=config.evidence_windows_per_constraint,
            )
            verifier_result = run_semantic_verifier(
                provider=verifier,
                query_spec=query_spec,
                candidate_document=CandidateDocumentForVerification(
                    document_id=candidate.document_id,
                    metadata=candidate.metadata,
                    paragraphs=candidate.paragraphs,
                ),
                evidence_windows=windows,
                timeout_seconds=float(
                    os.getenv("NALUS_LEGAL_V2_VERIFIER_TIMEOUT_SECONDS", "20")
                ),
            )
            final_decision = deterministic_verification_gate(
                query_spec=query_spec,
                verifier_result=verifier_result,
            )
            latency_values.append(verifier_result.latency_ms)
            diagnostics = dict(verifier_result.raw_diagnostics or {})
            failed_closed = bool(diagnostics.get("failed_closed"))
            if failed_closed:
                structural_failures += 1
                provider_errors += 1
            classification = str(diagnostics.get("classification") or "unknown")
            hard_negative = candidate.document_id in _hard_negative_ids(row)
            if final_decision.value == "verified_match" and hard_negative:
                verified_hard_negative_leakage += 1
            if final_decision.value == "verified_match" and candidate.document_id not in _gold_ids(row):
                unverified_returned_as_verified += 1
            row_result["verifier_results"].append(
                {
                    "document_id": candidate.document_id,
                    "classification": classification,
                    "provider_decision": verifier_result.decision.value,
                    "final_decision": final_decision.value,
                    "failed_closed": failed_closed,
                    "failed_closed_reason": diagnostics.get("reason") if failed_closed else None,
                    "latency_ms": verifier_result.latency_ms,
                    "constraint_result_count": len(verifier_result.constraint_results),
                    "evidence_validation_result": "fail_closed"
                    if failed_closed
                    else "accepted_structurally",
                    "evidence_passages": [
                        {
                            "constraint_id": window.constraint_id,
                            "paragraph_ids": window.paragraph_ids,
                            "quote": _bounded(window.text),
                            "source_of_claim": window.source_of_claim,
                        }
                        for window in windows
                    ],
                    "is_gold": candidate.document_id in _gold_ids(row),
                    "is_hard_negative": hard_negative,
                }
            )
            if structural_failures >= args.max_structural_failures:
                stopped_reason = "repeated_verifier_structural_failures"
                break
        rows.append(row_result)
        if stopped_reason is not None:
            break

    _write_cache(cache_path, cache)
    summary = {
        "schema": "legal_v2_deepseek_semantic_smoke_v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "blocked" if stopped_reason else "pass",
        "stopped_reason": stopped_reason,
        "query_count_requested": args.query_limit,
        "query_count_started": len(rows),
        "query_provider_calls": query_provider.calls,
        "semantic_verifier_provider_calls": verifier.calls,
        "semantic_verifier_cache_hits": verifier.cache_hits,
        "semantic_verifier_empty_content_retries": verifier.empty_content_retries,
        "total_provider_calls": query_provider.calls + verifier.calls,
        "structural_failures": structural_failures,
        "provider_errors": provider_errors,
        "wrong_index_identity": wrong_index_identity,
        "verified_hard_negative_leakage": verified_hard_negative_leakage,
        "unverified_candidates_returned_as_verified": unverified_returned_as_verified,
        "average_verifier_latency_ms": _average(latency_values),
        "elapsed_seconds": time.perf_counter() - started,
        "prompts_logged": False,
        "raw_provider_responses_logged": False,
        "secrets_logged": False,
        "holdout_used": False,
    }
    summary["structural_passed"] = (
        stopped_reason is None
        and structural_failures == 0
        and wrong_index_identity == 0
        and provider_errors == 0
    )
    payload = {"summary": summary, "rows": rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / args.json_name
    markdown_path = args.output_dir / args.markdown_name
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    print(json_path)
    return 0 if summary["structural_passed"] else 2


class CountingQueryProvider:
    provider_name = "deepseek_query_spec_v2_counted"

    def __init__(self, inner: QuerySpecProvider) -> None:
        self.inner = inner
        self.calls = 0
        self._cache: dict[str, dict[str, Any] | str] = {}
        self.model = getattr(inner, "model", None)

    def interpret(
        self,
        original_query: str,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any] | str:
        key = _hash({"query": original_query, "schema": "legal_query_spec_v2"})
        if key in self._cache:
            return self._cache[key]
        self.calls += 1
        payload = self.inner.interpret(
            original_query,
            timeout_seconds=timeout_seconds,
        )
        self._cache[key] = payload
        return payload


class CachedCountingVerifier:
    provider_name = "deepseek_semantic_verifier_v2_counted"

    def __init__(
        self,
        inner: SemanticVerifierProvider,
        *,
        cache: dict[str, Any],
    ) -> None:
        self.inner = inner
        self.cache = cache
        self.calls = 0
        self.cache_hits = 0
        self.model = getattr(inner, "model", None)

    def verify(
        self,
        *,
        query_spec: QuerySpecV2,
        candidate_document: CandidateDocumentForVerification,
        evidence_windows: list[EvidenceWindowForConstraint],
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        key = _verifier_cache_key(query_spec, candidate_document, evidence_windows)
        if key in self.cache:
            self.cache_hits += 1
            return dict(self.cache[key])
        self.calls += 1
        payload = self.inner.verify(
            query_spec=query_spec,
            candidate_document=candidate_document,
            evidence_windows=evidence_windows,
            timeout_seconds=timeout_seconds,
        )
        self.cache[key] = payload
        return payload

    @property
    def empty_content_retries(self) -> int:
        return int(getattr(self.inner, "empty_content_retries", 0))


def _select_smoke_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    wanted_domains = [
        "international_child_removal",
        "domestic_custody",
        "citizenship",
        "administrative_decision_review",
        "civil_procedure",
        "criminal_procedure",
        "constitutional_admissibility",
        "fair_trial_rights",
        "omitted_evidence",
        "property_disputes",
        "contractual_disputes",
        "damages",
        "service_of_documents",
        "maintenance",
    ]
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for domain in wanted_domains:
        for row in rows:
            if row.get("benchmark_split") not in {"diagnostic", "tuning"}:
                continue
            if row.get("legal_domain") != domain:
                continue
            if not _gold_ids(row):
                continue
            if str(row.get("id")) in seen_ids:
                continue
            selected.append(row)
            seen_ids.add(str(row.get("id")))
            break
    for style in ("ambiguous", "unsupported_zero_result"):
        for row in rows:
            if row.get("benchmark_split") not in {"diagnostic", "tuning"}:
                continue
            if row.get("query_style") != style:
                continue
            if str(row.get("id")) in seen_ids:
                continue
            selected.append(row)
            seen_ids.add(str(row.get("id")))
            if sum(1 for item in selected if item.get("query_style") == style) >= 1:
                break
    for row in rows:
        if len(selected) >= limit:
            break
        if row.get("benchmark_split") not in {"diagnostic", "tuning"}:
            continue
        if str(row.get("id")) in seen_ids:
            continue
        selected.append(row)
        seen_ids.add(str(row.get("id")))
    return selected[:limit]


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
        device="cpu",
        candidate_multiplier=1,
        min_candidate_count=1,
        max_candidate_count=max(
            config.dense_candidate_chunks,
            config.bm25_candidate_chunks,
        ),
        lexical_filter_enabled=False,
    )


def _safe_query_spec(spec: QuerySpecV2) -> dict[str, Any]:
    return {
        "intent": spec.intent.value,
        "retrieval_queries": spec.retrieval_queries[:5],
        "hard_constraint_count": len(spec.hard_constraints),
        "soft_constraint_count": len(spec.soft_constraints),
        "negative_constraint_count": len(spec.negative_constraints),
        "ambiguities": spec.ambiguities,
        "requires_verification": spec.requires_verification,
        "structured_query": spec.structured_query,
    }


def _gold_ids(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in (
        "strongly_relevant_document_ids",
        "materially_relevant_document_ids",
        "partial_match_document_ids",
    ):
        values.extend(str(item) for item in row.get(key) or [])
    return _dedupe(values)


def _hard_negative_ids(row: dict[str, Any]) -> set[str]:
    values: list[str] = []
    for key in ("explicit_hard_negative_document_ids", "related_only_document_ids"):
        values.extend(str(item) for item in row.get(key) or [])
    return set(_dedupe(values))


def _verifier_cache_key(
    query_spec: QuerySpecV2,
    candidate_document: CandidateDocumentForVerification,
    evidence_windows: list[EvidenceWindowForConstraint],
) -> str:
    evidence_checksum = hashlib.sha256(
        json.dumps(
            [
                {
                    "constraint_id": window.constraint_id,
                    "paragraph_ids": window.paragraph_ids,
                    "text": window.text,
                }
                for window in evidence_windows
            ],
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return _hash(
        {
            "query_spec": query_spec.to_dict(),
            "document_id": candidate_document.document_id,
            "evidence_checksum": evidence_checksum,
            "prompt_schema_version": PROMPT_SCHEMA_VERSION,
        }
    )


def _hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _bounded(text: str, limit: int = 500) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def _load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# DeepSeek semantic smoke",
        "",
        f"- Status: `{summary['status']}`",
        f"- Structural passed: `{summary.get('structural_passed')}`",
        f"- Stopped reason: `{summary.get('stopped_reason') or summary.get('reason')}`",
        f"- Query count started: `{summary.get('query_count_started', 0)}`",
        f"- Query provider calls: `{summary.get('query_provider_calls', 0)}`",
        "- Semantic verifier provider calls: "
        f"`{summary.get('semantic_verifier_provider_calls', 0)}`",
        f"- Semantic verifier cache hits: `{summary.get('semantic_verifier_cache_hits', 0)}`",
        "- Semantic verifier empty-content retries: "
        f"`{summary.get('semantic_verifier_empty_content_retries', 0)}`",
        f"- Total provider calls: `{summary.get('total_provider_calls', 0)}`",
        f"- Structural failures: `{summary.get('structural_failures', 0)}`",
        f"- Provider errors: `{summary.get('provider_errors', 0)}`",
        f"- Wrong index identity: `{summary.get('wrong_index_identity', 0)}`",
        "- Verified hard-negative leakage: "
        f"`{summary.get('verified_hard_negative_leakage', 0)}`",
        "- Unverified candidates returned as verified: "
        f"`{summary.get('unverified_candidates_returned_as_verified', 0)}`",
        "- Average verifier latency ms: "
        f"`{summary.get('average_verifier_latency_ms', 0.0)}`",
        f"- Prompts logged: `{summary.get('prompts_logged', False)}`",
        "- Raw provider responses logged: "
        f"`{summary.get('raw_provider_responses_logged', False)}`",
        f"- Secrets logged: `{summary.get('secrets_logged', False)}`",
        f"- Holdout used: `{summary.get('holdout_used', False)}`",
        "",
        "## Query Results",
        "",
    ]
    for row in payload["rows"]:
        lines.append(
            f"- `{row.get('id')}` `{row.get('domain')}` "
            f"interpretation=`{row.get('interpretation_status')}` "
            f"verified={[item['document_id'] for item in row.get('verifier_results', []) if item['final_decision'] == 'verified_match']}"
        )
    return "\n".join(lines) + "\n"


def _write_blocked(args: argparse.Namespace, reason: str) -> int:
    payload = {
        "summary": {
            "schema": "legal_v2_deepseek_semantic_smoke_v1",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": "blocked",
            "structural_passed": False,
            "reason": reason,
            "query_provider_calls": 0,
            "semantic_verifier_provider_calls": 0,
            "total_provider_calls": 0,
        },
        "rows": [],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / args.json_name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / args.markdown_name).write_text(_markdown(payload), encoding="utf-8")
    print(reason)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
