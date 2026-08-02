"""Forensic dump for known hybrid-smoke false-approval query IDs.

Runs only the listed benchmark rows and writes one card per false-approved
hard-negative document (classification, evidence, hard-constraint statuses).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.interpreter import DeepSeekQuerySpecProvider  # noqa: E402
from app.rag.legal_v2.pipeline import search_legal_v2  # noqa: E402
from app.rag.legal_v2.retriever import (  # noqa: E402
    LegalV2RetrieverConfig,
    build_live_legal_v2_retriever,
)
from app.rag.legal_v2.verifier import DeepSeekSemanticVerifierProvider  # noqa: E402
from app.rag.llm.providers.deepseek import DeepSeekThinkingMode  # noqa: E402
from app.rag.retrieval.bge_m3_embedder import BgeM3Embedder  # noqa: E402
from app.rag.retrieval.production_profile import ProductionRetrievalConfig  # noqa: E402
from app.rag.legal_v2.indexing import LEGAL_V2_PROFILE  # noqa: E402

TARGET_IDS = ("uq_028", "uq_031", "uq_037")


def main() -> int:
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key or api_key == "your-api-key-here":
        raise SystemExit("LLM_API_KEY is not configured")

    os.environ.setdefault("NALUS_LEGAL_V2_QUERYSPEC_TIMEOUT_SECONDS", "120")
    os.environ.setdefault("NALUS_LEGAL_V2_VERIFIER_TIMEOUT_SECONDS", "30")
    os.environ.setdefault("NALUS_LEGAL_V2_VERIFIER_THINKING_TIMEOUT_SECONDS", "120")
    os.environ.setdefault("NALUS_LEGAL_V2_VERIFIER_THINKING_FALLBACK", "1")
    os.environ.setdefault("NALUS_LEGAL_V2_VERIFIER_THINKING_FALLBACK_MAX_PER_QUERY", "2")
    os.environ.setdefault("NALUS_LEGAL_V2_VERIFIER_MAX_CANDIDATES_PER_QUERY", "8")
    os.environ["LLM_RETRY"] = "0"

    benchmark_path = (
        PROJECT_ROOT
        / "artifacts/legal_v2/pilot_600_20260731/universal_quality/reviewed_benchmark_v2.json"
    )
    output_dir = (
        PROJECT_ROOT
        / "artifacts/legal_v2/pilot_600_20260731/universal_quality/thinking_ab_test"
    )
    items = {
        str(item.get("id")): item
        for item in json.loads(benchmark_path.read_text(encoding="utf-8")).get("items") or []
    }
    rows = [items[qid] for qid in TARGET_IDS if qid in items]
    if len(rows) != len(TARGET_IDS):
        missing = [qid for qid in TARGET_IDS if qid not in items]
        raise SystemExit(f"Missing benchmark rows: {missing}")

    from qdrant_client import QdrantClient  # type: ignore[import-not-found]

    config = LegalV2RetrieverConfig(
        qdrant_collection="nalus_legal_paragraph_chunks_v2_pilot_600",
        bm25_sidecar_path=Path(
            "/app/storage/rag/bm25/nalus_legal_paragraph_bm25_v2_pilot_600.sqlite"
        ),
        bm25_index_id="nalus_legal_paragraph_bm25_v2_pilot_600",
        dense_candidate_chunks=80,
        bm25_candidate_chunks=80,
        fused_candidate_chunks=120,
        candidate_documents=40,
        returned_verified_documents=8,
        evidence_windows_per_constraint=3,
        model_path=os.getenv(
            "EMBEDDING_MODEL_NAME",
            "/root/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181",
        ),
    )
    # Prefer configured path when present; otherwise fall back to the local HF snapshot.
    model_path = Path(config.model_path)
    if not model_path.exists():
        hf_fallback = Path(
            "/root/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
        )
        if hf_fallback.exists():
            config = LegalV2RetrieverConfig(
                **{
                    **config.__dict__,
                    "model_path": str(hf_fallback),
                }
            )
        else:
            raise SystemExit(f"BGE-M3 model path missing: {model_path}")
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://qdrant:6333"), timeout=30)
    embedder_config = ProductionRetrievalConfig(
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
    retriever = build_live_legal_v2_retriever(
        client,
        BgeM3Embedder(embedder_config),
        config,
    )

    cards: list[dict[str, Any]] = []
    for row in rows:
        cards.append(_forensic_one(row=row, retriever=retriever, api_key=api_key, config=config))

    artifact = {
        "schema": "legal_v2_false_approval_forensics_v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_ids": list(TARGET_IDS),
        "cards": cards,
        "secrets_logged": False,
        "prompts_logged": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "false_approval_forensics_3.json"
    md_path = output_dir / "false_approval_forensics_3.md"
    json_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(artifact), encoding="utf-8")
    print(json.dumps({"written": str(json_path), "false_approval_docs": sum(len(c.get("false_approvals") or []) for c in cards)}, indent=2))
    return 0


def _forensic_one(*, row: dict[str, Any], retriever: Any, api_key: str, config: LegalV2RetrieverConfig) -> dict[str, Any]:
    query = str(row.get("query") or "")
    hard_negatives = {
        str(item).strip()
        for key in ("explicit_hard_negative_document_ids", "hard_negative_document_ids")
        for item in row.get(key) or []
        if str(item).strip()
    }
    gold = {
        str(item).strip()
        for key in (
            "strongly_relevant_document_ids",
            "materially_relevant_document_ids",
            "relevant_document_ids",
        )
        for item in row.get(key) or []
        if str(item).strip()
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
    result = search_legal_v2(
        query=query,
        retriever=retriever,
        verifier=fast_verifier,
        thinking_verifier=thinking_verifier,
        config=config,
        query_provider=query_provider,
        debug=True,
    )
    if result.status == "retrieval_error":
        print(
            json.dumps(
                {
                    "id": row.get("id"),
                    "retrieval_error": True,
                    "diagnostics": result.diagnostics,
                    "provider": result.provider,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    hard_constraint_summary = []
    if result.query_spec_summary:
        # summary may not include constraints; pull from diagnostics if present
        hard_constraint_summary = list(
            (result.diagnostics or {}).get("hard_constraints")
            or (result.query_spec_summary or {}).get("hard_constraints")
            or []
        )

    false_approvals: list[dict[str, Any]] = []
    verified_cards: list[dict[str, Any]] = []
    for document in result.verified_documents:
        card = _document_card(document)
        verified_cards.append(card)
        if document.document_id in hard_negatives:
            false_approvals.append(
                {
                    **card,
                    "hard_negative": True,
                    "in_gold": document.document_id in gold,
                    "benchmark_related_only": document.document_id
                    in {
                        str(item).strip()
                        for item in row.get("related_only_document_ids") or []
                        if str(item).strip()
                    },
                }
            )

    rejected_hard_negatives = [
        _document_card(document)
        for document in result.rejected_documents
        if document.document_id in hard_negatives
    ]

    return {
        "id": row.get("id"),
        "query": query,
        "legal_domain": row.get("legal_domain"),
        "normalized_legal_intent": row.get("normalized_legal_intent"),
        "mandatory_legal_concepts": list(row.get("mandatory_legal_concepts") or []),
        "mandatory_jurisdictions": list(row.get("mandatory_jurisdictions") or []),
        "excluded_concepts": list(row.get("excluded_concepts") or []),
        "hard_negative_document_ids": sorted(hard_negatives),
        "gold_document_ids": sorted(gold),
        "pipeline_status": result.status,
        "interpretation_status": result.interpretation_status,
        "interpretation_reason": (result.provider or {}).get("reason"),
        "retrieval_diagnostics": dict(result.diagnostics or {}),
        "query_spec_summary": result.query_spec_summary,
        "hard_constraints_from_diagnostics": hard_constraint_summary,
        "thinking_fallback_calls": int((result.provider or {}).get("thinking_fallback_calls") or 0),
        "verified_count": len(result.verified_documents),
        "rejected_count": len(result.rejected_documents),
        "verified_documents": verified_cards,
        "false_approvals": false_approvals,
        "rejected_hard_negatives": rejected_hard_negatives,
        "root_cause_hypothesis": _hypothesis(false_approvals, row),
    }


def _document_card(document: Any) -> dict[str, Any]:
    diagnostics = dict(document.verifier_diagnostics or {})
    constraint_results = []
    for item in list(diagnostics.get("constraint_results") or []):
        if isinstance(item, dict):
            constraint_results.append(item)
    # Prefer structured fields from the verified document itself.
    evidence = list(document.evidence or [])
    return {
        "document_id": document.document_id,
        "status": document.status,
        "relevance_classification": document.relevance_classification,
        "decision": getattr(document, "decision", None) or diagnostics.get("decision"),
        "thinking_fallback_used": bool(diagnostics.get("thinking_fallback_used")),
        "confidence": diagnostics.get("confidence"),
        "jurisdiction_match": diagnostics.get("jurisdiction_match"),
        "mandatory_concepts_supported": list(diagnostics.get("mandatory_concepts_supported") or []),
        "mandatory_concepts_missing": list(diagnostics.get("mandatory_concepts_missing") or []),
        "contradictory_facts": list(diagnostics.get("contradictory_facts") or []),
        "evidence_references": list(diagnostics.get("evidence_references") or []),
        "evidence": evidence,
        "constraint_results_summary": [
            {
                "constraint_id": item.get("constraint_id") if isinstance(item, dict) else getattr(item, "constraint_id", None),
                "status": (
                    item.get("status")
                    if isinstance(item, dict)
                    else (item.status.value if hasattr(getattr(item, "status", None), "value") else str(getattr(item, "status", "")))
                ),
                "evidence_paragraph_ids": (
                    list(item.get("evidence_paragraph_ids") or [])
                    if isinstance(item, dict)
                    else list(getattr(item, "evidence_paragraph_ids", None) or [])
                ),
                "source_of_claim": (
                    item.get("source_of_claim")
                    if isinstance(item, dict)
                    else getattr(item, "source_of_claim", None)
                ),
                "reason": (
                    item.get("reason")
                    if isinstance(item, dict)
                    else getattr(item, "reason", None)
                ),
            }
            for item in list(getattr(document, "constraint_results", None) or [])
        ]
        or constraint_results,
        "failed_closed": bool(diagnostics.get("failed_closed")),
        "failed_closed_reason": diagnostics.get("reason") if diagnostics.get("failed_closed") else None,
    }


def _hypothesis(false_approvals: list[dict[str, Any]], row: dict[str, Any]) -> str:
    if not false_approvals:
        return "no_false_approval_reproduced"
    thinking = any(item.get("thinking_fallback_used") for item in false_approvals)
    exactish = any(
        str(item.get("relevance_classification") or "") in {"exact_match", "strong_match"}
        for item in false_approvals
    )
    related = any(item.get("benchmark_related_only") for item in false_approvals)
    excluded = list(row.get("excluded_concepts") or [])
    if thinking and exactish:
        return (
            "thinking_fallback_overpromotion: hard-negative promoted to verified via thinking "
            f"path with exact/strong classification; excluded_concepts={excluded}"
        )
    if exactish and related:
        return (
            "lexical_overlap_related_only_approved: benchmark related_only/hard-negative "
            "approved as exact/strong; likely missing excluded-concept or intent gate"
        )
    if exactish:
        return "fast_or_gate_overapproval: hard-negative verified with exact/strong classification"
    return "unexpected_false_approval_shape"


def _markdown(artifact: dict[str, Any]) -> str:
    lines = [
        "# False-approval forensics (3 queries)",
        "",
        f"- Generated: `{artifact['generated_at']}`",
        f"- Targets: `{', '.join(artifact['target_ids'])}`",
        "",
    ]
    for card in artifact.get("cards") or []:
        lines.extend(
            [
                f"## {card.get('id')}",
                "",
                f"- Query: `{card.get('query')}`",
                f"- Domain: `{card.get('legal_domain')}`",
                f"- Intent: `{card.get('normalized_legal_intent')}`",
                f"- Pipeline status: `{card.get('pipeline_status')}`",
                f"- Thinking fallback calls: `{card.get('thinking_fallback_calls')}`",
                f"- Hypothesis: `{card.get('root_cause_hypothesis')}`",
                f"- Hard negatives: `{', '.join(card.get('hard_negative_document_ids') or [])}`",
                "",
            ]
        )
        fas = card.get("false_approvals") or []
        if not fas:
            lines.append("- No false approval reproduced on this run.")
            lines.append("")
            continue
        for item in fas:
            lines.extend(
                [
                    f"### FA document `{item.get('document_id')}`",
                    "",
                    f"- classification: `{item.get('relevance_classification')}`",
                    f"- status: `{item.get('status')}`",
                    f"- thinking_fallback_used: `{item.get('thinking_fallback_used')}`",
                    f"- confidence: `{item.get('confidence')}`",
                    f"- jurisdiction_match: `{item.get('jurisdiction_match')}`",
                    f"- related_only in benchmark: `{item.get('benchmark_related_only')}`",
                    f"- mandatory supported: `{item.get('mandatory_concepts_supported')}`",
                    f"- mandatory missing: `{item.get('mandatory_concepts_missing')}`",
                    f"- evidence_references: `{item.get('evidence_references')}`",
                    f"- constraint_results: `{json.dumps(item.get('constraint_results_summary') or [], ensure_ascii=False)}`",
                    "",
                ]
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
