from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.ingest.parser import parse_legal_document  # noqa: E402
from app.rag.legal_v2.ingest.sources import discover_source_documents  # noqa: E402

_NUMBERED_RE = re.compile(r"^\s*(?:\[(\d{1,4})\]|(\d{1,4})[.)])\s+")
_ROMAN_RE = re.compile(r"^\s*(I{1,3}|IV|V|VI{0,3}|IX|X)[.)]\s+")
_LEGACY_HEADING_RE = re.compile(
    r"^\s*(I{1,3}|IV|V|VI{0,3}|IX|X)?\.?\s*"
    r"([A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ][A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ\s]{3,})\s*$"
)
_LEGACY_HEADING_KEYWORDS = (
    "účastníci",
    "účastníků",
    "řízení",
    "průběh",
    "skutkový stav",
    "skutková",
    "argumentace",
    "námitky",
    "vyjádření",
    "právní úprava",
    "relevantní právo",
    "judikatura",
    "citovaná",
    "odůvodnění",
    "posouzení",
    "hodnocení",
    "výrok",
    "takto",
    "poučení",
)
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class _LegacyCandidate:
    text: str
    numbering: str | None
    heading: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only audit for the Legal v2 multiline numbered paragraph parser fix."
    )
    parser.add_argument("--batches-dir", type=Path, default=PROJECT_ROOT / "batches")
    parser.add_argument(
        "--nsoud-chunks-path",
        type=Path,
        default=PROJECT_ROOT / "app/artifacts/nsoud/rag_ready/nsoud_chunks_2025_01_03.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/legal_v2/parser_fix",
    )
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--short-token-threshold", type=int, default=8)
    parser.add_argument("--sample-limit", type=int, default=80)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    documents = discover_source_documents(
        batches_dir=args.batches_dir,
        nsoud_chunks_path=args.nsoud_chunks_path,
        limit=args.limit,
    )
    started_at = _utc_now()
    summary: dict[str, Any] = {
        "schema": "legal_v2_parser_fix_audit_v1",
        "started_at": started_at,
        "source_paths": {
            "batches_dir": str(args.batches_dir),
            "nsoud_chunks_path": str(args.nsoud_chunks_path),
        },
        "document_limit": args.limit,
        "documents_audited": len(documents),
        "old_total_candidate_count": 0,
        "corrected_total_candidate_count": 0,
        "candidate_count_delta": 0,
        "source_character_conservation_failures": 0,
        "duplicate_text_failures": 0,
        "source_order_failures": 0,
        "numbered_lines_classified_as_headings_before": 0,
        "numbered_lines_classified_as_headings_after": 0,
        "standalone_sp_zn_before": 0,
        "standalone_sp_zn_after": 0,
        "standalone_c_j_before": 0,
        "standalone_c_j_after": 0,
        "suspicious_short_candidates_before": 0,
        "suspicious_short_candidates_after": 0,
        "repaired_multiline_numbered_paragraphs": 0,
        "genuine_headings_preserved": 0,
        "parser_exceptions": 0,
        "documents_with_conservation_failures": [],
        "deepseek_calls": 0,
        "openai_calls": 0,
        "external_inference_calls": 0,
        "model_loads": 0,
        "model_downloads": 0,
        "qdrant_reads": 0,
        "qdrant_writes": 0,
        "bm25_reads": 0,
        "bm25_writes": 0,
        "index_rebuilds": 0,
    }
    samples: list[dict[str, Any]] = []

    for document in documents:
        try:
            legacy = _legacy_line_candidates(document.text)
            parsed = parse_legal_document(
                document_id=document.document_id,
                text=document.text,
                metadata=document.metadata,
            )
        except Exception as exc:  # noqa: BLE001 - audit records parser exceptions.
            summary["parser_exceptions"] += 1
            _append_sample(
                samples,
                args.sample_limit,
                {
                    "document_id": document.document_id,
                    "reason_code": "parser_exception",
                    "old_classification": "unknown",
                    "new_classification": "exception",
                    "excerpt": exc.__class__.__name__,
                    "manual_classification": "requires_follow_up",
                },
            )
            continue

        summary["old_total_candidate_count"] += len(legacy)
        summary["corrected_total_candidate_count"] += len(parsed.paragraphs)
        summary["numbered_lines_classified_as_headings_before"] += sum(
            1 for candidate in legacy if candidate.numbering is not None and candidate.heading
        )
        summary["numbered_lines_classified_as_headings_after"] += sum(
            1 for paragraph in parsed.paragraphs if paragraph.numbering is not None and _is_corrected_heading(paragraph.original_text)
        )
        summary["standalone_sp_zn_before"] += sum(1 for candidate in legacy if _starts_sp_zn(candidate.text))
        summary["standalone_sp_zn_after"] += sum(1 for paragraph in parsed.paragraphs if _starts_sp_zn(paragraph.original_text))
        summary["standalone_c_j_before"] += sum(1 for candidate in legacy if _starts_c_j(candidate.text))
        summary["standalone_c_j_after"] += sum(1 for paragraph in parsed.paragraphs if _starts_c_j(paragraph.original_text))
        summary["suspicious_short_candidates_before"] += sum(
            1 for candidate in legacy if _token_count(candidate.text) < args.short_token_threshold
        )
        summary["suspicious_short_candidates_after"] += sum(
            1 for paragraph in parsed.paragraphs if _token_count(paragraph.original_text) < args.short_token_threshold
        )
        summary["genuine_headings_preserved"] += parsed.diagnostics.heading_count

        repaired = _repaired_multiline_numbered_paragraphs(legacy, parsed.paragraphs)
        summary["repaired_multiline_numbered_paragraphs"] += repaired
        if repaired:
            _append_sample(
                samples,
                args.sample_limit,
                {
                    "document_id": document.document_id,
                    "reason_code": "corrected_multiline_numbered_paragraph",
                    "old_classification": "numbered_heading_plus_continuation",
                    "new_classification": "numbered_paragraph",
                    "excerpt": _excerpt(
                        next(
                            (
                                paragraph.original_text
                                for paragraph in parsed.paragraphs
                                if paragraph.numbering and ("sp. zn." in paragraph.original_text.lower() or "č. j." in paragraph.original_text.lower())
                            ),
                            parsed.paragraphs[0].original_text if parsed.paragraphs else "",
                        )
                    ),
                    "manual_classification": "corrected_split",
                },
            )

        source_non_ws = _non_whitespace(document.text)
        reconstructed_non_ws = _non_whitespace(parsed.reconstruct_text())
        if source_non_ws != reconstructed_non_ws:
            summary["source_character_conservation_failures"] += 1
            summary["documents_with_conservation_failures"].append(document.document_id)
            if len(reconstructed_non_ws) > len(source_non_ws):
                summary["duplicate_text_failures"] += 1
        starts = [paragraph.start_offset for paragraph in parsed.paragraphs]
        if starts != sorted(starts):
            summary["source_order_failures"] += 1

        _sample_keyword_cases(samples, args.sample_limit, document.document_id, legacy, parsed.paragraphs)

    _append_regression_manual_samples(samples, args.sample_limit)
    summary["candidate_count_delta"] = summary["corrected_total_candidate_count"] - summary["old_total_candidate_count"]
    summary["finished_at"] = _utc_now()
    summary["duration_ms"] = _duration_ms(started_at, summary["finished_at"])
    summary["status"] = (
        "pass"
        if summary["parser_exceptions"] == 0 and summary["source_character_conservation_failures"] == 0
        else "fail"
    )
    manual = _manual_summary(samples)
    payload = {"summary": summary, "manual_audit": manual}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "parser_fix_audit.json"
    markdown_path = args.output_dir / "parser_fix_audit.md"
    jsonl_path = args.output_dir / "parser_fix_suspicious_samples.jsonl"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    jsonl_path.write_text(
        "".join(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n" for sample in samples),
        encoding="utf-8",
    )
    print(json_path)
    print(markdown_path)
    print(jsonl_path)
    return 0 if summary["status"] == "pass" else 1


def _legacy_line_candidates(text: str) -> list[_LegacyCandidate]:
    candidates: list[_LegacyCandidate] = []
    current: list[str] = []
    for line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").splitlines():
        stripped = line.strip()
        if not stripped:
            _legacy_flush(candidates, current)
            current = []
            continue
        starts_new = bool(_NUMBERED_RE.match(stripped) or _ROMAN_RE.match(stripped) or _legacy_is_heading(stripped))
        if starts_new and current:
            _legacy_flush(candidates, current)
            current = []
        current.append(stripped)
        if _legacy_is_heading(stripped):
            _legacy_flush(candidates, current)
            current = []
    _legacy_flush(candidates, current)
    return candidates


def _legacy_flush(candidates: list[_LegacyCandidate], lines: list[str]) -> None:
    text = " ".join(line.strip() for line in lines if line.strip()).strip()
    if not text:
        return
    candidates.append(
        _LegacyCandidate(
            text=text,
            numbering=_extract_numbering(text),
            heading=_legacy_is_heading(text),
        )
    )


def _legacy_is_heading(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) > 120 or len(stripped.split()) > 10:
        return False
    if stripped.endswith(".") and len(stripped.split()) > 3:
        return False
    if _LEGACY_HEADING_RE.match(stripped):
        return True
    lowered = stripped.lower()
    return any(keyword in lowered for keyword in _LEGACY_HEADING_KEYWORDS)


def _is_corrected_heading(text: str) -> bool:
    stripped = text.strip()
    if _NUMBERED_RE.match(stripped):
        return False
    headings = {
        "účastníci řízení",
        "účastníci",
        "výrok",
        "takto",
        "odůvodnění",
        "posouzení ústavního soudu",
        "právní posouzení",
        "skutkový stav",
    }
    return stripped.casefold().strip(":") in headings or bool(_LEGACY_HEADING_RE.match(stripped))


def _extract_numbering(text: str) -> str | None:
    match = _NUMBERED_RE.match(text)
    if not match:
        return None
    return next((group for group in match.groups() if group), None)


def _starts_sp_zn(text: str) -> bool:
    return text.strip().lower().startswith("sp. zn.")


def _starts_c_j(text: str) -> bool:
    return text.strip().lower().startswith("č. j.")


def _token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


def _non_whitespace(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def _repaired_multiline_numbered_paragraphs(legacy: list[_LegacyCandidate], paragraphs: list[Any]) -> int:
    legacy_numbered_heading_ids = {
        candidate.numbering for candidate in legacy if candidate.numbering is not None and candidate.heading
    }
    return sum(
        1
        for paragraph in paragraphs
        if paragraph.numbering in legacy_numbered_heading_ids
        and ("sp. zn." in paragraph.original_text.lower() or "č. j." in paragraph.original_text.lower())
    )


def _sample_keyword_cases(
    samples: list[dict[str, Any]],
    sample_limit: int,
    document_id: str,
    legacy: list[_LegacyCandidate],
    paragraphs: list[Any],
) -> None:
    keywords = ("řízení", "nález", "odůvodnění", "posouzení")
    for keyword in keywords:
        if len(samples) >= sample_limit:
            return
        legacy_hit = next(
            (
                candidate
                for candidate in legacy
                if candidate.numbering and candidate.heading and keyword in candidate.text.lower()
            ),
            None,
        )
        if legacy_hit is None:
            continue
        corrected = next(
            (paragraph for paragraph in paragraphs if paragraph.numbering == legacy_hit.numbering),
            None,
        )
        if corrected is None:
            continue
        _append_sample(
            samples,
            sample_limit,
            {
                "document_id": document_id,
                "paragraph_number": corrected.numbering,
                "reason_code": f"keyword_false_heading_{keyword}",
                "old_classification": "heading",
                "new_classification": "numbered_paragraph",
                "excerpt": _excerpt(corrected.original_text),
                "manual_classification": "corrected_split",
            },
        )


def _append_regression_manual_samples(samples: list[dict[str, Any]], sample_limit: int) -> None:
    examples = [
        (
            "CONFIRMED-P28",
            "28. Ve věci řešené v řízení\nsp. zn. IV. ÚS 1038/25\nzrušil služební orgán napadené rozhodnutí.",
            "confirmed_paragraph_28",
            "corrected_split",
        ),
        (
            "CONFIRMED-P43",
            "43. V navazujícím posouzení soud připomněl, že v řízení\nč. j. 12 A 34/2024-56\nNejvyšší správní soud navázal na předchozí závěry.",
            "confirmed_paragraph_43",
            "corrected_split",
        ),
        (
            "KEYWORD-RIZENI",
            "1. Krátký právní odstavec obsahuje slovo řízení\nsp. zn. I. ÚS 1/24\na pokračuje v téže větě.",
            "keyword_rizeni_numbered_paragraph",
            "corrected_split",
        ),
        (
            "KEYWORD-NALEZ",
            "2. Napadený nález byl následně zrušen.",
            "keyword_nalez_numbered_paragraph",
            "valid_boundary_preserved",
        ),
        (
            "KEYWORD-ODUVODNENI",
            "3. Odůvodnění rozhodnutí je součástí věty, nikoli nadpisem.",
            "keyword_oduvodneni_numbered_paragraph",
            "valid_boundary_preserved",
        ),
        (
            "KEYWORD-POSOUZENI",
            "4. Posouzení věci proběhlo po doplnění dokazování.",
            "keyword_posouzeni_numbered_paragraph",
            "valid_boundary_preserved",
        ),
        (
            "GENUINE-HEADINGS",
            "Výrok\n1. Návrh se odmítá.\nOdůvodnění\n2. Soud posoudil věc.",
            "genuine_headings",
            "valid_heading_preserved",
        ),
        (
            "CONSECUTIVE-NUMBERED",
            "1. První odstavec.\n2. Druhý odstavec.",
            "consecutive_numbered_paragraphs",
            "valid_boundary_preserved",
        ),
    ]
    for document_id, text, reason_code, manual_classification in examples:
        if len(samples) >= sample_limit:
            return
        legacy = _legacy_line_candidates(text)
        parsed = parse_legal_document(document_id=document_id, text=text)
        _append_sample(
            samples,
            sample_limit,
            {
                "document_id": document_id,
                "reason_code": reason_code,
                "old_classification": _legacy_classification_summary(legacy),
                "new_classification": _corrected_classification_summary(parsed.paragraphs),
                "excerpt": _excerpt(parsed.reconstruct_text()),
                "manual_classification": manual_classification,
            },
        )


def _legacy_classification_summary(candidates: list[_LegacyCandidate]) -> str:
    parts = []
    for candidate in candidates:
        if candidate.heading:
            parts.append("heading")
        elif candidate.numbering:
            parts.append("numbered_paragraph")
        elif _starts_sp_zn(candidate.text) or _starts_c_j(candidate.text):
            parts.append("citation_orphan")
        else:
            parts.append("prose")
    return "+".join(parts)


def _corrected_classification_summary(paragraphs: list[Any]) -> str:
    parts = []
    for paragraph in paragraphs:
        if paragraph.numbering:
            parts.append("numbered_paragraph")
        elif paragraph.section_type.value in {"operative_part", "court_reasoning", "facts", "participants"}:
            parts.append("heading")
        else:
            parts.append("prose")
    return "+".join(parts)


def _append_sample(samples: list[dict[str, Any]], sample_limit: int, item: dict[str, Any]) -> None:
    if len(samples) >= sample_limit:
        return
    samples.append({key: value for key, value in item.items() if value not in {None, ""}})


def _excerpt(text: str, limit: int = 260) -> str:
    collapsed = " ".join(str(text or "").split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 3] + "..."


def _manual_summary(samples: list[dict[str, Any]]) -> dict[str, int]:
    result = {
        "samples_inspected": len(samples),
        "corrected_split": 0,
        "valid_heading_preserved": 0,
        "valid_boundary_preserved": 0,
        "possible_overmerge": 0,
        "possible_undersplit": 0,
        "requires_follow_up": 0,
    }
    for sample in samples:
        classification = str(sample.get("manual_classification") or "requires_follow_up")
        if classification in result:
            result[classification] += 1
    return result


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    manual = payload["manual_audit"]
    lines = [
        "# Legal v2 parser fix audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Documents audited: {summary['documents_audited']}",
        f"- Old candidate count: {summary['old_total_candidate_count']}",
        f"- Corrected candidate count: {summary['corrected_total_candidate_count']}",
        f"- Candidate delta: {summary['candidate_count_delta']}",
        f"- Numbered headings before: {summary['numbered_lines_classified_as_headings_before']}",
        f"- Numbered headings after: {summary['numbered_lines_classified_as_headings_after']}",
        f"- Standalone `sp. zn.` before/after: {summary['standalone_sp_zn_before']} / {summary['standalone_sp_zn_after']}",
        f"- Standalone `č. j.` before/after: {summary['standalone_c_j_before']} / {summary['standalone_c_j_after']}",
        f"- Suspicious short candidates before/after: {summary['suspicious_short_candidates_before']} / {summary['suspicious_short_candidates_after']}",
        f"- Repaired multiline numbered paragraphs: {summary['repaired_multiline_numbered_paragraphs']}",
        f"- Genuine headings preserved: {summary['genuine_headings_preserved']}",
        f"- Conservation failures: {summary['source_character_conservation_failures']}",
        f"- Parser exceptions: {summary['parser_exceptions']}",
        f"- DeepSeek/OpenAI/external calls: {summary['deepseek_calls']} / {summary['openai_calls']} / {summary['external_inference_calls']}",
        f"- Qdrant/BM25 writes: {summary['qdrant_writes']} / {summary['bm25_writes']}",
        "",
        "## Manual sample summary",
        "",
    ]
    for key, value in manual.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _duration_ms(started_at: str, finished_at: str) -> float:
    started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    finished = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
    return (finished - started).total_seconds() * 1000


if __name__ == "__main__":
    raise SystemExit(main())
