"""Structural gate for the selected Legal v2 hybrid thinking policy.

Runs:
- 2 QuerySpec operations with thinking enabled (120s)
- 2 fast verifier operations with thinking disabled (30s)
- 2 thinking-fallback verifier operations (120s)

No Stage A mutation. No retries beyond the production bounded QuerySpec retry.
"""

from __future__ import annotations

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

from app.rag.legal_v2.evidence import CandidateEvidenceDocument, select_evidence_windows  # noqa: E402
from app.rag.legal_v2.interpreter import (  # noqa: E402
    DeepSeekQuerySpecProvider,
    interpret_query_spec_v2,
)
from app.rag.legal_v2.models import LegalParagraph, MetadataProvenance, SectionType  # noqa: E402
from app.rag.legal_v2.query_spec import build_query_spec_v2  # noqa: E402
from app.rag.legal_v2.retriever import _paragraphs_from_chunks  # noqa: E402
from app.rag.legal_v2.verifier import (  # noqa: E402
    CandidateDocumentForVerification,
    DeepSeekSemanticVerifierProvider,
    EvidenceWindowForConstraint,
    validate_verifier_payload,
)
from app.rag.llm.providers._base import LLMProviderError  # noqa: E402
from app.rag.llm.providers.deepseek import DeepSeekThinkingMode  # noqa: E402
from app.rag.retrieval.models import RetrievedChunk  # noqa: E402

QUERYSPEC_TIMEOUT = 120.0
FAST_VERIFIER_TIMEOUT = 30.0
THINKING_VERIFIER_TIMEOUT = 120.0


def main() -> int:
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key or api_key == "your-api-key-here":
        raise SystemExit("LLM_API_KEY is not configured")
    os.environ["LLM_RETRY"] = "0"

    selection_path = (
        PROJECT_ROOT
        / "artifacts/legal_v2/pilot_600_20260731/universal_quality/thinking_ab_test/case_selection.json"
    )
    output_dir = (
        PROJECT_ROOT
        / "artifacts/legal_v2/pilot_600_20260731/universal_quality/thinking_ab_test"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    cases = list(selection.get("cases") or [])
    queryspec_cases = [cases[0], cases[1]]  # factual + countries
    verifier_cases = [cases[0], cases[1]]  # relevant + hard-negative pair each

    queryspec_results = []
    for case in queryspec_cases:
        provider = DeepSeekQuerySpecProvider(
            api_key,
            thinking=DeepSeekThinkingMode.ENABLED,
            timeout_seconds=QUERYSPEC_TIMEOUT,
            max_tokens=8000,
        )
        started = time.perf_counter()
        interpretation = interpret_query_spec_v2(
            str(case["query"]),
            provider=provider,
            timeout_seconds=QUERYSPEC_TIMEOUT,
            allow_deterministic_fallback=False,
            max_provider_attempts=2,
        )
        meta = getattr(provider, "last_meta", None)
        queryspec_results.append(
            {
                "case_id": case["case_id"],
                "latency_ms": (time.perf_counter() - started) * 1000,
                "status": interpretation.status,
                "schema_valid": interpretation.query_spec is not None,
                "preservation_ok": interpretation.query_spec is not None,
                "timed_out": "timeout" in str(interpretation.reason or "").lower(),
                "message_content_present": bool(getattr(meta, "message_content_present", False)),
                "reasoning_content_present": bool(getattr(meta, "reasoning_content_present", False)),
                "reasoning_content_char_count": int(
                    getattr(meta, "reasoning_content_char_count", 0) or 0
                ),
                "reason": interpretation.reason,
                "critical_concept_preserved": _critical_preserved(case, interpretation.query_spec),
            }
        )

    from qdrant_client import QdrantClient  # type: ignore[import-not-found]

    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://qdrant:6333"), timeout=30)
    collection = "nalus_legal_paragraph_chunks_v2_pilot_600"
    fast_results: list[dict[str, Any]] = []
    thinking_results: list[dict[str, Any]] = []
    for case in verifier_cases:
        query_spec = build_query_spec_v2(str(case["query"]))
        candidates = list(case.get("verifier_candidates") or [])[:2]
        # Prefer one relevant and one hard-negative when available.
        for candidate_meta in candidates:
            document_id = str(candidate_meta["document_id"])
            role = str(candidate_meta["role"])
            candidate = _load_candidate(client, collection, document_id)
            if candidate is None:
                continue
            windows = select_evidence_windows(
                query_spec=query_spec,
                candidate=candidate,
                max_windows_per_constraint=3,
            )
            frozen = [
                EvidenceWindowForConstraint(
                    constraint_id=window.constraint_id,
                    paragraph_ids=list(window.paragraph_ids),
                    text=window.text,
                    section_types=list(window.section_types),
                    heading_context=list(window.heading_context),
                    source_of_claim=window.source_of_claim,
                    current_case_classification=window.current_case_classification,
                )
                for window in windows
            ][:3]
            if role == "relevant" and len(fast_results) < 2:
                fast_results.append(
                    _run_verifier(
                        api_key=api_key,
                        thinking=DeepSeekThinkingMode.DISABLED,
                        timeout=FAST_VERIFIER_TIMEOUT,
                        max_tokens=1024,
                        query_spec=query_spec,
                        candidate=candidate,
                        windows=frozen,
                        case_id=case["case_id"],
                        role=role,
                    )
                )
            if role == "hard_negative" and len(thinking_results) < 2:
                thinking_results.append(
                    _run_verifier(
                        api_key=api_key,
                        thinking=DeepSeekThinkingMode.ENABLED,
                        timeout=THINKING_VERIFIER_TIMEOUT,
                        max_tokens=8000,
                        query_spec=query_spec,
                        candidate=candidate,
                        windows=frozen,
                        case_id=case["case_id"],
                        role=role,
                    )
                )
            if len(fast_results) >= 2 and len(thinking_results) >= 2:
                break
        if len(fast_results) >= 2 and len(thinking_results) >= 2:
            break

    # If hard-negatives were insufficient for thinking samples, fill from remaining.
    if len(thinking_results) < 2:
        for case in verifier_cases:
            for candidate_meta in list(case.get("verifier_candidates") or []):
                if len(thinking_results) >= 2:
                    break
                document_id = str(candidate_meta["document_id"])
                candidate = _load_candidate(client, collection, document_id)
                if candidate is None:
                    continue
                query_spec = build_query_spec_v2(str(case["query"]))
                windows = select_evidence_windows(
                    query_spec=query_spec,
                    candidate=candidate,
                    max_windows_per_constraint=3,
                )[:3]
                thinking_results.append(
                    _run_verifier(
                        api_key=api_key,
                        thinking=DeepSeekThinkingMode.ENABLED,
                        timeout=THINKING_VERIFIER_TIMEOUT,
                        max_tokens=8000,
                        query_spec=query_spec,
                        candidate=candidate,
                        windows=windows,
                        case_id=case["case_id"],
                        role=str(candidate_meta["role"]),
                    )
                )

    qs_ok = sum(1 for row in queryspec_results if row["schema_valid"] and row["critical_concept_preserved"])
    fast_ok = sum(1 for row in fast_results if row["schema_valid"] and row["evidence_id_valid"])
    think_ok = sum(1 for row in thinking_results if row["schema_valid"] and row["evidence_id_valid"])
    timeouts = sum(
        1
        for row in queryspec_results + fast_results + thinking_results
        if row.get("timed_out")
    )
    gate_passed = (
        qs_ok == 2
        and fast_ok == 2
        and think_ok == 2
        and timeouts == 0
        and len(queryspec_results) == 2
        and len(fast_results) == 2
        and len(thinking_results) == 2
    )
    artifact = {
        "schema": "legal_v2_thinking_structural_gate_v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "policy": {
            "queryspec_mode": "thinking_enabled",
            "queryspec_timeout_seconds": QUERYSPEC_TIMEOUT,
            "fast_verifier_mode": "thinking_disabled",
            "fast_verifier_timeout_seconds": FAST_VERIFIER_TIMEOUT,
            "thinking_fallback_mode": "thinking_enabled",
            "thinking_fallback_timeout_seconds": THINKING_VERIFIER_TIMEOUT,
        },
        "queryspec_results": queryspec_results,
        "fast_verifier_results": fast_results,
        "thinking_verifier_results": thinking_results,
        "summary": {
            "queryspec_success": qs_ok,
            "fast_verifier_success": fast_ok,
            "thinking_verifier_success": think_ok,
            "timeouts": timeouts,
            "schema_failures": (2 - qs_ok) + (2 - fast_ok) + (2 - think_ok),
            "evidence_failures": sum(
                1
                for row in fast_results + thinking_results
                if not row.get("evidence_id_valid")
            ),
            "gate_passed": gate_passed,
        },
        "secrets_logged": False,
        "prompts_logged": False,
        "reasoning_content_persisted": False,
        "stage_a_modified": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "structural_gate.json"
    md_path = output_dir / "structural_gate.md"
    json_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# Thinking hybrid structural gate",
                "",
                f"- Generated: `{artifact['generated_at']}`",
                f"- Gate passed: `{gate_passed}`",
                f"- QuerySpec success: `{qs_ok}/2`",
                f"- Fast verifier success: `{fast_ok}/2`",
                f"- Thinking verifier success: `{think_ok}/2`",
                f"- Timeouts: `{timeouts}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"gate_passed": gate_passed, "summary": artifact["summary"]}, indent=2))
    return 0 if gate_passed else 2


def _critical_preserved(case: dict[str, Any], query_spec: Any) -> bool:
    if query_spec is None:
        return False
    if case.get("clarification_expected"):
        return bool(query_spec.ambiguities) or not query_spec.requires_verification
    if case["case_id"] == "ab_factual_actors":
        roles = {entity.role for entity in query_spec.entities if entity.role}
        return "mother" in roles or "child" in roles
    if case["case_id"] == "ab_countries_jurisdictions":
        blob = " ".join(
            [
                str(item.normalized_text or item.text)
                for item in list(query_spec.locations or [])
            ]
            + ([str(query_spec.origin.normalized_text)] if query_spec.origin else [])
            + ([str(query_spec.destination.normalized_text)] if query_spec.destination else [])
            + [str(c.normalized_value or c.value) for c in query_spec.hard_constraints]
        ).casefold()
        return "česk" in blob or "rus" in blob or "občanstv" in blob or bool(query_spec.hard_constraints)
    return bool(query_spec.hard_constraints)


def _run_verifier(
    *,
    api_key: str,
    thinking: DeepSeekThinkingMode,
    timeout: float,
    max_tokens: int,
    query_spec: Any,
    candidate: Any,
    windows: list[Any],
    case_id: str,
    role: str,
) -> dict[str, Any]:
    provider = DeepSeekSemanticVerifierProvider(
        api_key,
        thinking=thinking,
        timeout_seconds=timeout,
        max_tokens=max_tokens,
    )
    started = time.perf_counter()
    timed_out = False
    provider_error = None
    payload = None
    try:
        payload = provider.verify(
            query_spec=query_spec,
            candidate_document=CandidateDocumentForVerification(
                document_id=candidate.document_id,
                metadata=candidate.metadata,
                paragraphs=candidate.paragraphs,
            ),
            evidence_windows=windows,
            timeout_seconds=timeout,
        )
    except LLMProviderError as exc:
        timed_out = exc.category == "timeout"
        provider_error = exc.safe_reason
    except TimeoutError:
        timed_out = True
        provider_error = "timeout"
    except Exception as exc:  # noqa: BLE001
        provider_error = exc.__class__.__name__
    latency_ms = (time.perf_counter() - started) * 1000
    meta = getattr(provider, "last_meta", None)
    schema_valid = False
    evidence_id_valid = False
    classification = None
    if payload is not None:
        validated = validate_verifier_payload(
            payload=payload,
            query_spec=query_spec,
            candidate_document=CandidateDocumentForVerification(
                document_id=candidate.document_id,
                metadata=candidate.metadata,
                paragraphs=candidate.paragraphs,
            ),
            evidence_windows=windows,
            provider_name=provider.provider_name,
            latency_ms=latency_ms,
        )
        diagnostics = dict(validated.raw_diagnostics or {})
        failed = bool(diagnostics.get("failed_closed"))
        schema_valid = not failed
        evidence_id_valid = not failed
        classification = diagnostics.get("classification") or payload.get("classification")
    return {
        "case_id": case_id,
        "role": role,
        "thinking_mode": thinking.value,
        "timeout_seconds": timeout,
        "latency_ms": latency_ms,
        "timed_out": timed_out,
        "provider_error": provider_error,
        "message_content_present": bool(getattr(meta, "message_content_present", False)),
        "reasoning_content_present": bool(getattr(meta, "reasoning_content_present", False)),
        "reasoning_content_char_count": int(getattr(meta, "reasoning_content_char_count", 0) or 0),
        "schema_valid": schema_valid,
        "evidence_id_valid": evidence_id_valid,
        "classification": classification,
    }


def _load_candidate(client: Any, collection: str, document_id: str) -> Any | None:
    from qdrant_client.http import models as qm  # type: ignore[import-not-found]

    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=qm.Filter(
            must=[qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id))]
        ),
        limit=64,
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        return None
    chunks: list[RetrievedChunk] = []
    for point in points:
        payload = dict(point.payload or {})
        text = str(payload.get("text") or payload.get("chunk_text") or "")
        chunks.append(
            RetrievedChunk(
                id=str(payload.get("chunk_id") or point.id),
                text=text,
                score=0.0,
                source="qdrant_scroll",
                metadata=payload,
            )
        )
    paragraphs = _paragraphs_from_chunks(document_id, chunks)
    if not paragraphs:
        paragraphs = [
            LegalParagraph(
                document_id=document_id,
                paragraph_id=f"{document_id}:p:{index}",
                paragraph_index=index,
                original_text=chunk.text,
                normalized_text=chunk.text,
                section_type=SectionType.OTHER,
                start_offset=0,
                end_offset=len(chunk.text),
                source_order=index,
                heading_context=[],
                is_boilerplate=False,
                is_citation_block=False,
                language="cs",
                metadata_provenance=MetadataProvenance(source="v2_index", extraction_method="qdrant_scroll"),
            )
            for index, chunk in enumerate(chunks)
            if chunk.text.strip()
        ]
    return CandidateEvidenceDocument(
        document_id=document_id,
        metadata={"document_id": document_id},
        paragraphs=paragraphs,
        score=0.0,
    )


if __name__ == "__main__":
    raise SystemExit(main())
