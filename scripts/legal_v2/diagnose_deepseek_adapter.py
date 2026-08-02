from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.interpreter import _query_spec_prompt  # noqa: E402
from app.rag.legal_v2.models import LegalParagraph, MetadataProvenance, SectionType  # noqa: E402
from app.rag.legal_v2.query_spec import QuerySpecV2, build_query_spec_v2  # noqa: E402
from app.rag.legal_v2.structured_output import extract_json_object  # noqa: E402
from app.rag.legal_v2.verifier import (  # noqa: E402
    CandidateDocumentForVerification,
    EvidenceWindowForConstraint,
    _semantic_payload_error,
    _verifier_prompt,
)
from app.rag.llm.config import effective_llm_config_from_env  # noqa: E402

_ENDPOINT = "https://api.deepseek.com/chat/completions"
_CASE_RE = re.compile(r"\b(?:[IVXLCDM]+\.\s*)?ÚS\s*\d+/\d+\b", re.IGNORECASE)
_ECLI_RE = re.compile(r"ECLI:[A-Z]{2}:[A-Z]+:[0-9]{4}:[A-Z0-9.:]+", re.IGNORECASE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Redacted Legal v2 DeepSeek response-shape diagnostic.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/legal_v2/pilot_600_20260731/universal_quality/deepseek_adapter_fix",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key or api_key == "your-api-key-here":
        raise SystemExit("LLM_API_KEY is not configured")
    config = effective_llm_config_from_env()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    query = "lhůta pro podání ústavní stížnosti"
    spec = build_query_spec_v2(query)
    records = [
        _call(
            api_key=api_key,
            operation="query_spec",
            model=config.deepseek_model,
            max_tokens=config.legal_v2_max_tokens,
            prompt=_query_spec_prompt(query),
            schema_validator=lambda payload: _validate_query_spec(query, payload),
        ),
        _call(
            api_key=api_key,
            operation="semantic_verifier",
            model=config.deepseek_model,
            max_tokens=config.legal_v2_max_tokens,
            prompt=_verifier_prompt(
                query_spec=spec,
                candidate_document=_diagnostic_candidate(),
                evidence_windows=_diagnostic_evidence(spec),
            ),
            schema_validator=lambda payload: _validate_verifier(spec, payload),
        ),
    ]
    artifact = {
        "schema": "legal_v2_deepseek_redacted_response_shape_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_provider_calls": len(records),
        "elapsed_seconds": time.perf_counter() - started,
        "minimal_json_mode_comparison": {
            "same_endpoint": True,
            "same_model": True,
            "same_temperature": True,
            "same_response_format": {"type": "json_object"},
            "successful_minimal_prompt_length": 45,
            "successful_minimal_max_tokens": 128,
            "legal_v2_max_tokens": config.legal_v2_max_tokens,
            "primary_difference": "Legal v2 prompts are much longer and request large structured objects.",
        },
        "records": records,
        "secrets_logged": False,
        "prompts_logged": False,
        "complete_provider_outputs_logged": False,
    }
    json_path = args.output_dir / "redacted_response_shape.json"
    md_path = args.output_dir / "redacted_response_shape.md"
    json_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(artifact), encoding="utf-8")
    print(json.dumps({"status": "ok", "calls": len(records), "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False))
    return 0


def _call(
    *,
    api_key: str,
    operation: str,
    model: str,
    max_tokens: int,
    prompt: str,
    schema_validator: Any,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    started = time.perf_counter()
    response = httpx.post(
        _ENDPOINT,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=effective_llm_config_from_env().timeout_seconds,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    body: Any
    try:
        body = response.json()
    except ValueError:
        body = {}
    choices = body.get("choices") if isinstance(body, dict) else None
    first = choices[0] if isinstance(choices, list) and choices else {}
    message = first.get("message") if isinstance(first, dict) else {}
    content = message.get("content") if isinstance(message, dict) else None
    content_text = content if isinstance(content, str) else ""
    extraction = extract_json_object(content_text)
    schema_error = schema_validator(extraction.payload) if extraction.payload is not None else None
    usage = body.get("usage") if isinstance(body, dict) else None
    completion_details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
    reasoning_tokens = (
        completion_details.get("reasoning_tokens")
        if isinstance(completion_details, dict)
        else None
    )
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": operation,
        "configured_model": model,
        "http_status": response.status_code,
        "provider_request_id": _request_id(response.headers),
        "top_level_response_type": type(body).__name__,
        "top_level_response_keys": sorted(body.keys()) if isinstance(body, dict) else [],
        "choice_count": len(choices) if isinstance(choices, list) else 0,
        "choice_keys": sorted(first.keys()) if isinstance(first, dict) else [],
        "message_keys": sorted(message.keys()) if isinstance(message, dict) else [],
        "finish_reason": first.get("finish_reason") if isinstance(first, dict) else None,
        "message_content_exists": "content" in message if isinstance(message, dict) else False,
        "message_content_type": type(content).__name__ if content is not None else None,
        "tool_calls_exists": "tool_calls" in message if isinstance(message, dict) else False,
        "refusal_exists": "refusal" in message if isinstance(message, dict) else False,
        "reasoning_content_exists": "reasoning_content" in message if isinstance(message, dict) else False,
        "reasoning_content_length": len(str(message.get("reasoning_content") or "")) if isinstance(message, dict) else 0,
        "response_format_requested": payload["response_format"],
        "raw_extracted_content_empty": not bool(content_text.strip()),
        "raw_extracted_content_char_count": len(content_text),
        "raw_extracted_content_sha256": hashlib.sha256(content_text.encode("utf-8")).hexdigest(),
        "raw_extracted_content_start_fragment": _fragment(content_text[:300]),
        "raw_extracted_content_end_fragment": _fragment(content_text[-300:]),
        "markdown_code_fences_present": "```" in content_text,
        "json_object_located": extraction.payload is not None,
        "json_parses": extraction.payload is not None,
        "extraction": extraction.diagnostics.to_dict(),
        "schema_error": schema_error,
        "schema_error_paths": _schema_paths(schema_error),
        "usage": _safe_usage(usage),
        "reasoning_tokens": reasoning_tokens,
        "truncation_indicators": {
            "finish_reason_length": first.get("finish_reason") == "length" if isinstance(first, dict) else False,
            "completion_tokens_equals_max_tokens": isinstance(usage, dict) and usage.get("completion_tokens") == max_tokens,
            "content_empty_with_reasoning": not bool(content_text.strip()) and bool(message.get("reasoning_content")) if isinstance(message, dict) else False,
        },
        "request_parameters": {
            "endpoint": _ENDPOINT,
            "model": model,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "response_format": payload["response_format"],
            "message_count": 1,
            "prompt_length": len(prompt),
        },
        "latency_ms": latency_ms,
    }


def _validate_query_spec(query: str, payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    try:
        spec = QuerySpecV2.from_dict(payload)
    except Exception as exc:  # noqa: BLE001
        return f"schema:{exc.__class__.__name__}:{_safe(str(exc), 240)}"
    if spec.original_query != query:
        return "schema:original_query_changed"
    if not spec.retrieval_queries:
        return "schema:retrieval_queries_missing"
    return None


def _validate_verifier(spec: QuerySpecV2, payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    candidate = _diagnostic_candidate()
    windows = _diagnostic_evidence(spec)
    if str(payload.get("document_id") or "") != candidate.document_id:
        return "schema:document_id_mismatch"
    classification = str(payload.get("classification") or "insufficient_evidence")
    semantic_error = _semantic_payload_error(
        payload=payload,
        classification=classification,  # type: ignore[arg-type]
        evidence_windows=windows,
    )
    if semantic_error:
        return "schema:" + semantic_error
    return None


def _diagnostic_candidate() -> CandidateDocumentForVerification:
    paragraph = LegalParagraph(
        document_id="DIAGNOSTIC-DOC",
        paragraph_id="DIAGNOSTIC-DOC:p:00001",
        paragraph_index=1,
        original_text="Ústavní soud posuzoval lhůtu pro podání ústavní stížnosti a odmítl opožděný návrh.",
        normalized_text="Ústavní soud posuzoval lhůtu pro podání ústavní stížnosti a odmítl opožděný návrh.",
        section_type=SectionType.COURT_REASONING,
        start_offset=0,
        end_offset=89,
        source_order=1,
        heading_context=[],
        is_boilerplate=False,
        is_citation_block=False,
        language="cs",
        metadata_provenance=MetadataProvenance(source="diagnostic", extraction_method="synthetic"),
    )
    return CandidateDocumentForVerification(
        document_id="DIAGNOSTIC-DOC",
        metadata={"court_name": "Ústavní soud"},
        paragraphs=[paragraph],
    )


def _diagnostic_evidence(spec: QuerySpecV2) -> list[EvidenceWindowForConstraint]:
    candidate = _diagnostic_candidate()
    constraint_id = (
        spec.hard_constraints[0].constraint_id
        if spec.hard_constraints
        else "diagnostic_constraint"
    )
    paragraph = candidate.paragraphs[0]
    return [
        EvidenceWindowForConstraint(
            constraint_id=constraint_id,
            paragraph_ids=[paragraph.paragraph_id],
            text=paragraph.normalized_text,
            section_types=[paragraph.section_type],
            source_of_claim="court_finding",
            current_case_classification="current_case",
        )
    ]


def _fragment(value: str) -> str:
    return _safe(value.replace("\r", "\\r").replace("\n", "\\n"), 300)


def _safe(value: str, limit: int) -> str:
    value = _ECLI_RE.sub("[REDACTED_ECLI]", value)
    value = _CASE_RE.sub("[REDACTED_CASE_ID]", value)
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _schema_paths(schema_error: str | None) -> list[str]:
    if not schema_error:
        return []
    if schema_error.startswith("schema:"):
        return [schema_error.split(":", 2)[1]]
    return [schema_error]


def _safe_usage(usage: Any) -> dict[str, Any]:
    if not isinstance(usage, dict):
        return {}
    allowed = {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
    }
    return {key: usage.get(key) for key in sorted(allowed) if key in usage}


def _request_id(headers: httpx.Headers) -> str | None:
    for key in ("x-request-id", "x-ds-request-id", "request-id"):
        value = headers.get(key)
        if value:
            return _safe(value, 120)
    return None


def _markdown(artifact: dict[str, Any]) -> str:
    lines = ["# DeepSeek adapter redacted response shape", ""]
    lines.append(f"- Provider calls: `{artifact['total_provider_calls']}`")
    lines.append("- Secrets logged: `False`")
    lines.append("- Prompts logged: `False`")
    lines.append("- Complete provider outputs logged: `False`")
    lines.append("")
    for record in artifact["records"]:
        lines.extend(
            [
                f"## {record['operation']}",
                f"- HTTP status: `{record['http_status']}`",
                f"- Finish reason: `{record['finish_reason']}`",
                f"- Content chars: `{record['raw_extracted_content_char_count']}`",
                f"- Reasoning content chars: `{record['reasoning_content_length']}`",
                f"- JSON parses: `{record['json_parses']}`",
                f"- Extraction method: `{record['extraction']['extraction_method']}`",
                f"- Extraction error: `{record['extraction']['error']}`",
                f"- Schema error: `{record['schema_error']}`",
                f"- Truncated by length: `{record['truncation_indicators']['finish_reason_length']}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
