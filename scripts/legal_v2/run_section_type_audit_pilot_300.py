#!/usr/bin/env python3
"""Targeted SectionType audit over chunking_ab_pilot_300_v1.

Does NOT embed, does NOT build indexes, does NOT modify the parser.
Uses the same immutable 300-ECLI inventory and Stage1 full-text API.

Produces:
- section label distributions
- heuristic suspicion rates (esp. header-vs-reasoning)
- short structural chunk stats
- unreviewed boundary sample (~100-200 flags)
- materiality verdict for whether Slice 4 should wait on a parser fix
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.rag.legal_v2.models import SectionType
from app.rag.legal_v2.parser import parse_legal_document

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = (
    PROJECT_ROOT / "benchmarks" / "legal_v2" / "chunking_ab_pilot_300_v1" / "inventory_manifest.json"
)
DEFAULT_ARTIFACTS_INVENTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "legal_v2"
    / "chunking_ab_pilot_300_v1"
    / "inventory_manifest.json"
)
DEFAULT_OUT = (
    PROJECT_ROOT / "artifacts" / "legal_v2" / "chunking_ab_pilot_300_v1" / "section_type_audit"
)
DEFAULT_API = "http://127.0.0.1:8029"
NATIVE_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_NUMBERED_START_RE = re.compile(r"^\s*\d{1,3}[.)]\s+")
_ROMAN_OPERATIVE_RE = re.compile(
    r"^\s*(?:I{1,3}|IV|V|VI{0,3}|IX|X)\.\s+.+\b(se\s+(?:potvrzuje|odmítá|zamítá|zrušuje)|"
    r"se\s+vyhovuje|se\s+nepřiznává)\b",
    re.IGNORECASE,
)
_REASONING_CUES_RE = re.compile(
    r"\b(předně\s+posoudil|dospěl\s+k\s+závěru|ústavní\s+soud\s+uvádí|"
    r"ústavní\s+soud\s+konstatuje|odmítl\s+podle|jako\s+návrh\s+zjevně|"
    r"soud\s+shledal|právní\s+posouzení|vzhledem\s+ke\s+shora)\b",
    re.IGNORECASE,
)
_HEADER_LOOKS_RE = re.compile(
    r"^(?:Česká republika|Ústavní soud|NÁLEZ|USNESENÍ|Jménem republiky|"
    r"Vrchní soud|Nejvyšší soud)\b",
    re.IGNORECASE,
)
_SIMPLE_HEADING_RE = re.compile(
    r"^(?:Výrok|Odůvodnění:?|Poučení:?|I+|II+|III+|IV+|V+)\.?$",
    re.IGNORECASE,
)
_APPEAL_CUES_RE = re.compile(
    r"\b(podal\s+odvolání|namítá|domáhal\s+se|stěžovatel\s+uvádí|"
    r"v\s+odvolání\s+namít)\b",
    re.IGNORECASE,
)
_SUSPECT_LABELS = {
    SectionType.HEADER.value,
    SectionType.PARTICIPANTS.value,
    SectionType.CITED_CASE.value,
    SectionType.PROCEDURAL_HISTORY.value,
    SectionType.LEGAL_FRAMEWORK.value,
    SectionType.COURT_REASONING.value,
}


def _request_json(url: str, timeout: float = 180.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_full_text(api: str, ecli: str) -> str:
    url = f"{api.rstrip('/')}/api/rag/documents/{urllib.parse.quote(ecli, safe='')}"
    payload = _request_json(url, timeout=180.0)
    text = payload.get("full_text") or ""
    if not str(text).strip():
        raise RuntimeError(f"empty full_text for {ecli}")
    return str(text)


def _native_tokens(text: str) -> int:
    return len(NATIVE_TOKEN_RE.findall(text))


def _flag_paragraph(
    *,
    section: str,
    text: str,
    is_heading: bool,
    token_count: int,
    prev_section: str | None,
) -> list[str]:
    flags: list[str] = []
    if section == SectionType.HEADER.value:
        if _NUMBERED_START_RE.match(text) and _REASONING_CUES_RE.search(text):
            flags.append("header_numbered_reasoning_cues")
        elif _REASONING_CUES_RE.search(text) and token_count >= 40:
            flags.append("header_substantive_reasoning_cues")
        elif token_count >= 80 and not _HEADER_LOOKS_RE.match(text) and not is_heading:
            flags.append("header_long_nonheader_prose")
    if section == SectionType.CITED_CASE.value and _ROMAN_OPERATIVE_RE.search(text):
        flags.append("cited_case_looks_like_operative")
    if section == SectionType.PARTICIPANTS.value and _APPEAL_CUES_RE.search(text) and token_count >= 40:
        flags.append("participants_looks_like_party_arguments")
    if section == SectionType.LEGAL_FRAMEWORK.value and _REASONING_CUES_RE.search(text):
        flags.append("legal_framework_with_reasoning_cues")
    if section == SectionType.PROCEDURAL_HISTORY.value and _REASONING_CUES_RE.search(text):
        flags.append("procedural_history_with_reasoning_cues")
    if section == SectionType.OPERATIVE_PART.value and _SIMPLE_HEADING_RE.match(text.strip()):
        flags.append("operative_heading_only_tiny")
    if token_count <= 10 and (
        _SIMPLE_HEADING_RE.match(text.strip())
        or text.strip().rstrip(":").lower() in {"výrok", "odůvodnění", "poučení"}
    ):
        flags.append("tiny_structural_heading_chunk_candidate")
    if (
        prev_section == SectionType.COURT_REASONING.value
        and section == SectionType.HEADER.value
        and _NUMBERED_START_RE.match(text)
    ):
        flags.append("reasoning_to_header_flip_on_numbered")
    if (
        prev_section
        and prev_section != section
        and _NUMBERED_START_RE.match(text)
        and section in _SUSPECT_LABELS
        and prev_section in _SUSPECT_LABELS
        and token_count >= 30
    ):
        flags.append("suspect_label_flip_in_numbered_sequence")
    return flags


def _materiality_verdict(stats: dict[str, Any]) -> dict[str, Any]:
    """Decide if SectionType errors are material enough to block Slice 4."""
    docs = max(int(stats["documents"]), 1)
    header_bad = int(stats["flag_counts"].get("header_numbered_reasoning_cues", 0)) + int(
        stats["flag_counts"].get("header_substantive_reasoning_cues", 0)
    ) + int(stats["flag_counts"].get("header_long_nonheader_prose", 0))
    tiny = int(stats["flag_counts"].get("tiny_structural_heading_chunk_candidate", 0))
    docs_with_header_bad = int(stats["documents_with_header_suspicion"])
    docs_with_any = int(stats["documents_with_any_suspicion"])
    header_share = float(stats["section_paragraph_share"].get("header", 0.0))

    reasons: list[str] = []
    warnings: list[str] = []
    material = False
    if docs_with_header_bad / docs >= 0.15 or header_bad >= 40:
        material = True
        reasons.append(
            f"header_mislabel_signal docs={docs_with_header_bad}/{docs} flags={header_bad}"
        )
    if header_share >= 0.25 and docs_with_header_bad / docs >= 0.10:
        material = True
        reasons.append(f"header_paragraph_share={header_share:.3f} with mislabel signal")
    # Tiny "Výrok"/"Odůvodnění" paragraphs are usually correct headings; they are a
    # chunker attach/merge concern, not a SectionType mislabel by themselves.
    if tiny >= 80:
        warnings.append(
            f"tiny_structural_heading_candidates={tiny} "
            "(chunker should attach headings; not SectionType material alone)"
        )
    if docs_with_any / docs >= 0.40 and material:
        reasons.append(f"broad_suspicion_coverage docs={docs_with_any}/{docs}")

    if material:
        verdict = "SECTION_TYPE_MATERIAL_REGRESSION"
        recommendation = (
            "Do NOT start Slice 4 embeddings yet. Fix deterministic SectionType "
            "classification (sticky headings / stop keyword-to-header traps), "
            "then regenerate A300/B300 chunk QA."
        )
    elif docs_with_header_bad / docs <= 0.02 and header_share < 0.08 and header_bad <= 5:
        verdict = "SECTION_TYPE_OK_FOR_CHUNKING_AB"
        recommendation = (
            "SectionType header traps cleared. Regenerate A/B chunk QA, then Slice 4 "
            "may proceed. Remaining tiny heading paragraphs are a chunker attach issue."
        )
    elif docs_with_any / docs >= 0.20:
        verdict = "SECTION_TYPE_NEEDS_MANUAL_REVIEW"
        recommendation = (
            "Suspicion present but below hard materiality thresholds. "
            "Review the unreviewed sample before Slice 4."
        )
    else:
        verdict = "SECTION_TYPE_OK_FOR_CHUNKING_AB"
        recommendation = (
            "No strong material SectionType signal; Slice 4 may proceed after "
            "quick manual spot-check of the review sample."
        )
    return {
        "verdict": verdict,
        "material_for_chunking": material,
        "reasons": reasons,
        "warnings": warnings,
        "recommendation": recommendation,
        "block_slice4": material,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--expected-count", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--review-target", type=int, default=180)
    args = parser.parse_args()

    inv_path = args.inventory
    if inv_path is None:
        inv_path = DEFAULT_INVENTORY if DEFAULT_INVENTORY.exists() else DEFAULT_ARTIFACTS_INVENTORY
    inventory = json.loads(inv_path.read_text(encoding="utf-8"))
    ordered = list(inventory.get("ordered_eclis") or [])
    meta_by_ecli = {
        str(d.get("ecli")): d for d in (inventory.get("documents") or []) if d.get("ecli")
    }
    if inventory.get("inventory_id") != "chunking_ab_pilot_300_v1":
        raise SystemExit(f"unexpected inventory_id={inventory.get('inventory_id')}")
    if len(ordered) != args.expected_count and not args.limit:
        raise SystemExit(f"ABORT: inventory count {len(ordered)} != {args.expected_count}")
    if args.limit and args.limit > 0:
        ordered = ordered[: args.limit]

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    section_para_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    tiny_structural = 0
    docs_with_header_suspicion = 0
    docs_with_any_suspicion = 0
    failures: list[dict[str, Any]] = []
    review_candidates: list[dict[str, Any]] = []
    per_doc: list[dict[str, Any]] = []
    total_paragraphs = 0
    token_hist_header: list[int] = []
    transitions: Counter[str] = Counter()

    for index, ecli in enumerate(ordered, start=1):
        print(f"[{index}/{len(ordered)}] {ecli}", flush=True)
        try:
            text = _fetch_full_text(args.api, ecli)
            meta = meta_by_ecli.get(ecli) or {}
            court = str(meta.get("court") or "")
            parsed = parse_legal_document(
                document_id=ecli,
                text=text,
                metadata={"court": court, "ecli": ecli},
            )
            doc_flags: Counter[str] = Counter()
            doc_sections: Counter[str] = Counter()
            prev_section: str | None = None
            header_suspect_here = False
            any_suspect_here = False
            for para in parsed.paragraphs:
                section = para.section_type.value
                tokens = _native_tokens(para.normalized_text)
                total_paragraphs += 1
                section_para_counts[section] += 1
                doc_sections[section] += 1
                if prev_section and prev_section != section:
                    transitions[f"{prev_section}->{section}"] += 1
                if section == SectionType.HEADER.value:
                    token_hist_header.append(tokens)
                # heading_context non-empty + short text often means heading-ish
                is_heading = bool(para.heading_context) and tokens <= 12
                flags = _flag_paragraph(
                    section=section,
                    text=para.normalized_text,
                    is_heading=is_heading or tokens <= 8,
                    token_count=tokens,
                    prev_section=prev_section,
                )
                for flag in flags:
                    flag_counts[flag] += 1
                    doc_flags[flag] += 1
                if "tiny_structural_heading_chunk_candidate" in flags:
                    tiny_structural += 1
                if any(f.startswith("header_") for f in flags):
                    header_suspect_here = True
                if flags:
                    any_suspect_here = True
                    review_candidates.append(
                        {
                            "ecli": ecli,
                            "court": court or None,
                            "paragraph_index": para.paragraph_index,
                            "section_type": section,
                            "token_count_native": tokens,
                            "flags": flags,
                            "numbering": para.numbering,
                            "heading_context": list(para.heading_context)[:3],
                            "text_preview": para.normalized_text[:320],
                            "prev_section": prev_section,
                            "length_bucket": meta.get("length_bucket"),
                            "selection_reason": meta.get("selection_reason"),
                        }
                    )
                prev_section = section

            if header_suspect_here:
                docs_with_header_suspicion += 1
            if any_suspect_here:
                docs_with_any_suspicion += 1
            per_doc.append(
                {
                    "ecli": ecli,
                    "court": court or None,
                    "paragraph_count": len(parsed.paragraphs),
                    "section_counts": dict(doc_sections),
                    "flag_counts": dict(doc_flags),
                    "parser_section_counts": dict(parsed.diagnostics.section_counts),
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"ecli": ecli, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  FAIL {exc}", flush=True)

    if failures:
        (out_dir / "section_type_audit_failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise SystemExit(f"ABORT: {len(failures)} document failures")

    # Prefer diverse review sample: header issues first, then other flags.
    priority = {
        "header_numbered_reasoning_cues": 0,
        "header_substantive_reasoning_cues": 1,
        "header_long_nonheader_prose": 2,
        "reasoning_to_header_flip_on_numbered": 3,
        "cited_case_looks_like_operative": 4,
        "participants_looks_like_party_arguments": 5,
        "tiny_structural_heading_chunk_candidate": 6,
    }

    def _rank(item: dict[str, Any]) -> tuple[int, str, int]:
        best = min(priority.get(f, 50) for f in item["flags"])
        return (best, item["ecli"], int(item["paragraph_index"]))

    review_candidates.sort(key=_rank)
    # Cap per document to keep breadth.
    selected: list[dict[str, Any]] = []
    per_ecli: Counter[str] = Counter()
    for item in review_candidates:
        if per_ecli[item["ecli"]] >= 4:
            continue
        selected.append(item)
        per_ecli[item["ecli"]] += 1
        if len(selected) >= args.review_target:
            break
    # If still short, fill without per-doc cap.
    if len(selected) < min(args.review_target, len(review_candidates)):
        seen = {(x["ecli"], x["paragraph_index"]) for x in selected}
        for item in review_candidates:
            key = (item["ecli"], item["paragraph_index"])
            if key in seen:
                continue
            selected.append(item)
            if len(selected) >= args.review_target:
                break

    section_share = {
        k: (v / max(total_paragraphs, 1)) for k, v in sorted(section_para_counts.items())
    }
    stats = {
        "documents": len(ordered),
        "total_paragraphs": total_paragraphs,
        "section_paragraph_counts": dict(section_para_counts),
        "section_paragraph_share": section_share,
        "flag_counts": dict(flag_counts),
        "tiny_structural_heading_candidates": tiny_structural,
        "documents_with_header_suspicion": docs_with_header_suspicion,
        "documents_with_any_suspicion": docs_with_any_suspicion,
        "top_section_transitions": dict(transitions.most_common(30)),
        "header_token_stats": {
            "count": len(token_hist_header),
            "median": (
                sorted(token_hist_header)[len(token_hist_header) // 2]
                if token_hist_header
                else 0
            ),
            "p90": (
                sorted(token_hist_header)[int(0.9 * (len(token_hist_header) - 1))]
                if token_hist_header
                else 0
            ),
            "max": max(token_hist_header) if token_hist_header else 0,
        },
    }
    verdict = _materiality_verdict(stats)

    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inventory_id": inventory["inventory_id"],
        "inventory_hash_sha256": inventory["inventory_hash_sha256"],
        "inventory_path": str(inv_path.as_posix()),
        "a_implementation_note": "SectionType comes from parse_legal_document; A/B both consume it",
        "no_embeddings_generated": True,
        "no_experimental_indexes_built": True,
        "slice4_not_started": True,
        "stats": stats,
        "verdict": verdict,
        "review_sample_count": len(selected),
        "review_candidate_pool": len(review_candidates),
        "documents": per_doc,
    }
    (out_dir / "section_type_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "section_type_review_unreviewed.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = [
        "# SectionType audit (chunking_ab_pilot_300_v1)",
        "",
        f"- verdict: `{verdict['verdict']}`",
        f"- material_for_chunking: `{verdict['material_for_chunking']}`",
        f"- block_slice4: `{verdict['block_slice4']}`",
        f"- inventory_hash: `{inventory['inventory_hash_sha256']}`",
        f"- documents: {len(ordered)}",
        f"- paragraphs: {total_paragraphs}",
        f"- docs with header suspicion: {docs_with_header_suspicion}/{len(ordered)}",
        f"- docs with any suspicion: {docs_with_any_suspicion}/{len(ordered)}",
        f"- tiny structural heading candidates: {tiny_structural}",
        f"- review sample: {len(selected)} / pool {len(review_candidates)}",
        "",
        "## Recommendation",
        "",
        verdict["recommendation"],
        "",
        "## Reasons",
        "",
    ]
    for reason in verdict["reasons"] or ["(none)"]:
        md.append(f"- {reason}")
    md.extend(["", "## Section paragraph share", ""])
    for key, value in section_share.items():
        md.append(f"- `{key}`: {value:.4f} ({section_para_counts[key]})")
    md.extend(["", "## Flag counts", ""])
    for key, value in sorted(flag_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        md.append(f"- `{key}`: {value}")
    md.extend(
        [
            "",
            "## Review sample (UNREVIEWED)",
            "",
            "Heuristic flags only — not gold labels.",
            "",
        ]
    )
    for item in selected[:80]:
        md.append(
            f"### {item['ecli']} ¶{item['paragraph_index']} `{item['section_type']}`"
        )
        md.append(f"- flags: {', '.join(item['flags'])}")
        md.append(f"- tokens: {item['token_count_native']} | prev: {item.get('prev_section')}")
        preview = item["text_preview"].replace("\n", " ")
        md.append(f"- preview: {preview[:240]}")
        md.append("")
    (out_dir / "section_type_audit_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"WROTE {out_dir / 'section_type_audit_summary.json'}", flush=True)
    print(f"VERDICT {verdict['verdict']} block_slice4={verdict['block_slice4']}", flush=True)
    return 0 if not verdict["block_slice4"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
