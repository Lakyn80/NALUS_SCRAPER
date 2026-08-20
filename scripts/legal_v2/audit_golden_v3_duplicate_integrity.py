#!/usr/bin/env python3
"""Audit Golden v3 relevance_review_queue for ECLI / case-reference duplicate integrity.

Evidence-based classification only. Does NOT auto-merge (except reporting that 015
was already resolved by human confirmation). Does NOT touch Qdrant.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_QUEUE = (
    PROJECT_ROOT / "artifacts" / "legal_v2" / "golden_v3_graded" / "relevance_review_queue.jsonl"
)
DEFAULT_OUT_JSONL = (
    PROJECT_ROOT / "artifacts" / "legal_v2" / "golden_v3_graded" / "duplicate_integrity_audit.jsonl"
)
DEFAULT_OUT_MD = (
    PROJECT_ROOT / "artifacts" / "legal_v2" / "golden_v3_graded" / "DUPLICATE_INTEGRITY_AUDIT.md"
)

_ROMAN = {"I": "1", "II": "2", "III": "3", "IV": "4"}
_CASE_SUFFIX_RE = re.compile(r"\s*#\d+\s*$")
_WS_RE = re.compile(r"\s+")
_SINGLE_ORD_RE = re.compile(r"\.\d$")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    path.write_text(text + ("\n" if rows else ""), encoding="utf-8")


def normalize_ws(value: Any) -> str:
    return _WS_RE.sub(" ", str(value or "").strip())


def normalize_text_for_sim(value: Any) -> str:
    s = normalize_ws(value).lower()
    s = re.sub(r"[^\w\s§áčďéěíňóřšťúůýž]", "", s, flags=re.UNICODE)
    return _WS_RE.sub(" ", s).strip()


def normalize_case_reference(value: Any) -> str:
    s = normalize_ws(value)
    s = _CASE_SUFFIX_RE.sub("", s)
    s = s.replace("ÚS", "US")
    s = re.sub(r"\s+", "", s).upper()

    def rom(m: re.Match[str]) -> str:
        return _ROMAN.get(m.group(1), m.group(1)) + "."

    s = re.sub(r"^(IV|III|II|I)\.", rom, s)
    return s


def court_from_row(row: dict[str, Any]) -> str:
    court = normalize_ws(row.get("court")).lower()
    if court:
        return court
    ecli = str(row.get("ecli") or row.get("document_id") or "").upper()
    m = re.match(r"ECLI:([A-Z]+):([A-Z]+):", ecli)
    if m:
        return f"{m.group(1)}:{m.group(2)}".lower()
    return ""


def ecli_core(value: Any) -> str:
    return str(value or "").strip().upper().replace("ÚS", "US")


def strip_single_trailing_ordinal(ecli: str) -> str:
    """Strip only a single-digit trailing .N (document ordinal), never year .97."""
    return _SINGLE_ORD_RE.sub("", ecli)


def is_ecli_representation_variant(a: str, b: str) -> bool:
    """True when one ECLI is the other + trailing single-digit .N, or one lacks final .1."""
    a = ecli_core(a)
    b = ecli_core(b)
    if not a or not b or a == b:
        return False
    a_base = strip_single_trailing_ordinal(a)
    b_base = strip_single_trailing_ordinal(b)
    if a_base == b_base and a_base:
        a_has = bool(_SINGLE_ORD_RE.search(a))
        b_has = bool(_SINGLE_ORD_RE.search(b))
        if a_has != b_has:
            return True
        return False
    if a.startswith(b + ".") and re.fullmatch(r"\d", a[len(b) + 1 :]):
        return True
    if b.startswith(a + ".") and re.fullmatch(r"\d", b[len(a) + 1 :]):
        return True
    return False


def content_similarity(row_a: dict[str, Any], row_b: dict[str, Any]) -> dict[str, Any]:
    texts_a = [
        normalize_text_for_sim(row_a.get("candidate_summary")),
        normalize_text_for_sim(row_a.get("reasoning_excerpt")),
    ]
    texts_b = [
        normalize_text_for_sim(row_b.get("candidate_summary")),
        normalize_text_for_sim(row_b.get("reasoning_excerpt")),
    ]
    best_prefix_equal = False
    best_ratio = 0.0
    best_pair = ("", "")
    for ta in texts_a:
        for tb in texts_b:
            if not ta or not tb:
                continue
            if ta[:120] == tb[:120] and len(ta[:120]) >= 40:
                best_prefix_equal = True
            ratio = SequenceMatcher(None, ta[:400], tb[:400]).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_pair = (ta[:80], tb[:80])
    strong = best_prefix_equal or best_ratio >= 0.92
    weak = 0.0 < best_ratio < 0.92 and not best_prefix_equal
    missing = best_ratio == 0.0 and not best_prefix_equal
    return {
        "prefix120_equal": best_prefix_equal,
        "seq_ratio_first400": round(best_ratio, 4),
        "strong_content_match": strong,
        "weak_or_missing_content": weak or missing,
        "sample_a": best_pair[0],
        "sample_b": best_pair[1],
    }


def prefer_canonical(doc_a: str, doc_b: str) -> str:
    """Prefer the ECLI that includes a trailing single-digit ordinal (more specific)."""
    a = ecli_core(doc_a)
    b = ecli_core(doc_b)
    a_ord = bool(_SINGLE_ORD_RE.search(a))
    b_ord = bool(_SINGLE_ORD_RE.search(b))
    if a_ord and not b_ord:
        return doc_a
    if b_ord and not a_ord:
        return doc_b
    return doc_a if len(a) >= len(b) else doc_b


def has_human_judgment(row: dict[str, Any]) -> bool:
    if row.get("review_status") == "human_reviewed":
        return True
    if row.get("human_status") in ("confirmed", "changed"):
        return True
    if row.get("human_grade") is not None:
        return True
    notes = str(row.get("reviewer_notes") or "")
    return notes.startswith("HUMAN_AUDIT_BATCH")


def classify_pair(row_a: dict[str, Any], row_b: dict[str, Any]) -> dict[str, Any] | None:
    doc_a = str(row_a.get("document_id") or "")
    doc_b = str(row_b.get("document_id") or "")
    if not doc_a or not doc_b or doc_a == doc_b:
        return None

    court_a = court_from_row(row_a)
    court_b = court_from_row(row_b)
    date_a = normalize_ws(row_a.get("decision_date"))
    date_b = normalize_ws(row_b.get("decision_date"))
    case_a = normalize_case_reference(row_a.get("case_reference"))
    case_b = normalize_case_reference(row_b.get("case_reference"))

    same_court = bool(court_a) and court_a == court_b
    same_date = bool(date_a) and date_a == date_b
    same_case = bool(case_a) and case_a == case_b
    ecli_variant = is_ecli_representation_variant(
        str(row_a.get("ecli") or doc_a), str(row_b.get("ecli") or doc_b)
    )
    content = content_similarity(row_a, row_b)

    if not (same_court and same_date and same_case):
        if not (same_court and same_date and ecli_variant):
            return None

    if same_court and same_date and same_case and ecli_variant and content["strong_content_match"]:
        classification = "CONFIRMED_DUPLICATE"
    elif same_court and same_date and same_case and (
        content["seq_ratio_first400"] > 0
        and content["seq_ratio_first400"] < 0.75
        and not content["prefix120_equal"]
    ):
        classification = "DISTINCT_DECISIONS"
    elif same_court and same_date and same_case:
        classification = "NEEDS_HUMAN_CHECK"
    elif same_court and same_date and ecli_variant and content["strong_content_match"]:
        classification = "CONFIRMED_DUPLICATE"
    elif same_court and same_date and ecli_variant:
        classification = "NEEDS_HUMAN_CHECK"
    else:
        return None

    canonical = prefer_canonical(doc_a, doc_b) if classification == "CONFIRMED_DUPLICATE" else None
    return {
        "classification": classification,
        "document_ids": sorted([doc_a, doc_b]),
        "eclis": sorted(
            [
                str(row_a.get("ecli") or doc_a),
                str(row_b.get("ecli") or doc_b),
            ]
        ),
        "court": court_a,
        "decision_date": date_a,
        "case_reference_normalized": case_a or case_b,
        "case_references_raw": [
            str(row_a.get("case_reference") or ""),
            str(row_b.get("case_reference") or ""),
        ],
        "ecli_representation_variant": ecli_variant,
        "content_evidence": content,
        "canonical_recommendation": canonical,
        "already_resolved_example_015": sorted([doc_a, doc_b])
        == sorted(["ECLI:CZ:US:1999:4.US.23.99", "ECLI:CZ:US:1999:4.US.23.99.1"]),
    }


def audit(queue_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    total_rows = len(queue_rows)
    unique_pairs = len({(r.get("query_id"), r.get("document_id")) for r in queue_rows})

    by_doc: dict[str, dict[str, Any]] = {}
    queries_by_doc: dict[str, set[str]] = defaultdict(set)
    human_by_doc: dict[str, bool] = defaultdict(bool)
    for r in queue_rows:
        did = str(r.get("document_id") or "")
        if not did:
            continue
        queries_by_doc[did].add(str(r.get("query_id") or ""))
        human_by_doc[did] = human_by_doc[did] or has_human_judgment(r)
        if did not in by_doc:
            by_doc[did] = dict(r)
        else:
            for k, v in r.items():
                if by_doc[did].get(k) in (None, "", []) and v not in (None, "", []):
                    by_doc[did][k] = v

    groups: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for did, row in by_doc.items():
        court = court_from_row(row)
        case_n = normalize_case_reference(row.get("case_reference"))
        date = normalize_ws(row.get("decision_date"))
        if court and case_n and date:
            groups[(court, case_n, date)].add(did)

    multi_groups = {k: sorted(v) for k, v in groups.items() if len(v) > 1}

    findings: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for (court, case_n, date), docs in sorted(multi_groups.items()):
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                a, b = docs[i], docs[j]
                key = tuple(sorted([a, b]))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                result = classify_pair(by_doc[a], by_doc[b])
                if result is None:
                    continue
                qids = sorted((queries_by_doc[a] | queries_by_doc[b]) - {""})
                result["group_key"] = {
                    "court": court,
                    "case_reference_normalized": case_n,
                    "decision_date": date,
                }
                result["group_document_ids"] = docs
                result["affected_query_ids"] = qids
                result["human_judgment_on_either"] = bool(human_by_doc[a] or human_by_doc[b])
                result["human_judgment_by_document_id"] = {
                    a: bool(human_by_doc[a]),
                    b: bool(human_by_doc[b]),
                }
                findings.append(result)

    by_court_date: dict[tuple[str, str], list[str]] = defaultdict(list)
    for did, row in by_doc.items():
        court = court_from_row(row)
        date = normalize_ws(row.get("decision_date"))
        if court and date:
            by_court_date[(court, date)].append(did)

    for court, date in by_court_date:
        docs = by_court_date[(court, date)]
        if len(docs) < 2:
            continue
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                a, b = docs[i], docs[j]
                key = tuple(sorted([a, b]))
                if key in seen_pairs:
                    continue
                if not is_ecli_representation_variant(a, b):
                    continue
                seen_pairs.add(key)
                result = classify_pair(by_doc[a], by_doc[b])
                if result is None:
                    continue
                qids = sorted((queries_by_doc[a] | queries_by_doc[b]) - {""})
                result["group_key"] = {
                    "court": court,
                    "case_reference_normalized": result.get("case_reference_normalized"),
                    "decision_date": date,
                }
                result["group_document_ids"] = sorted(set(docs))
                result["affected_query_ids"] = qids
                result["human_judgment_on_either"] = bool(human_by_doc[a] or human_by_doc[b])
                result["human_judgment_by_document_id"] = {
                    a: bool(human_by_doc[a]),
                    b: bool(human_by_doc[b]),
                }
                findings.append(result)

    uniq: dict[tuple[str, str], dict[str, Any]] = {}
    rank = {"CONFIRMED_DUPLICATE": 3, "NEEDS_HUMAN_CHECK": 2, "DISTINCT_DECISIONS": 1}
    for f in findings:
        key = tuple(sorted(f["document_ids"]))
        prev = uniq.get(key)
        if prev is None or rank.get(f["classification"], 0) > rank.get(prev["classification"], 0):
            uniq[key] = f

    findings = sorted(
        uniq.values(),
        key=lambda x: (
            {"CONFIRMED_DUPLICATE": 0, "NEEDS_HUMAN_CHECK": 1, "DISTINCT_DECISIONS": 2}.get(
                x["classification"], 9
            ),
            x["document_ids"][0],
            x["document_ids"][1],
        ),
    )

    confirmed = [f for f in findings if f["classification"] == "CONFIRMED_DUPLICATE"]
    needs = [f for f in findings if f["classification"] == "NEEDS_HUMAN_CHECK"]
    distinct = [f for f in findings if f["classification"] == "DISTINCT_DECISIONS"]

    collapse_map: dict[str, str] = {}
    for f in confirmed:
        ids = f["document_ids"]
        canon = f.get("canonical_recommendation") or prefer_canonical(ids[0], ids[1])
        for did in ids:
            if did != canon:
                collapse_map[did] = canon

    hypo_all = {
        (
            r.get("query_id"),
            collapse_map.get(str(r.get("document_id") or ""), str(r.get("document_id") or "")),
        )
        for r in queue_rows
    }
    hypo_excl = {
        (
            r.get("query_id"),
            collapse_map.get(str(r.get("document_id") or ""), str(r.get("document_id") or "")),
        )
        for r in queue_rows
        if r.get("exclude_from_qrels") is not True
    }

    summary = {
        "total_queue_rows_before": total_rows,
        "unique_query_document_pairs": unique_pairs,
        "multi_doc_identity_groups": len(multi_groups),
        "confirmed_duplicate_pairs": len(confirmed),
        "needs_human_check_pairs": len(needs),
        "distinct_decisions_pairs": len(distinct),
        "findings_total": len(findings),
        "hypothetical_unique_query_document_after_collapse": len(hypo_all),
        "hypothetical_unique_excluding_exclude_from_qrels": len(hypo_excl),
        "collapse_map_size": len(collapse_map),
        "note_015_already_resolved": True,
    }
    return findings, summary


def render_md(findings: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    confirmed = [f for f in findings if f["classification"] == "CONFIRMED_DUPLICATE"]
    needs = [f for f in findings if f["classification"] == "NEEDS_HUMAN_CHECK"]
    distinct = [f for f in findings if f["classification"] == "DISTINCT_DECISIONS"]

    lines: list[str] = []
    lines.append("# Duplicate Integrity Audit — Golden v3 relevance queue")
    lines.append("")
    lines.append("Evidence-based scan of **all** `relevance_review_queue.jsonl` rows.")
    lines.append("No auto-merge except query **015** (already human-confirmed). Qdrant untouched.")
    lines.append("")
    lines.append("## Summary counts")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("| --- | ---: |")
    for k, label in [
        ("total_queue_rows_before", "Total queue rows"),
        ("unique_query_document_pairs", "Unique (query_id, document_id)"),
        ("multi_doc_identity_groups", "Identity groups with ≥2 document_ids"),
        ("confirmed_duplicate_pairs", "CONFIRMED_DUPLICATE pairs"),
        ("needs_human_check_pairs", "NEEDS_HUMAN_CHECK pairs"),
        ("distinct_decisions_pairs", "DISTINCT_DECISIONS pairs"),
        (
            "hypothetical_unique_query_document_after_collapse",
            "Hypothetical unique (q,doc) if confirmed collapsed",
        ),
    ]:
        lines.append(f"| {label} | {summary.get(k, 0)} |")
    lines.append("")
    lines.append(
        f"Collapse map size (alias→canonical recommendations): **{summary.get('collapse_map_size', 0)}**."
    )
    lines.append("")
    lines.append(
        "### Note: 015 already resolved\n\n"
        "`ECLI:CZ:US:1999:4.US.23.99` → canonical `ECLI:CZ:US:1999:4.US.23.99.1` "
        "was merged under **HUMAN_AUDIT_BATCH_3_MERGE_015** "
        "(`dedup_status=merged_into_canonical`, `exclude_from_qrels=true`). "
        "It appears below as a CONFIRMED_DUPLICATE example if still present in the queue."
    )
    lines.append("")
    lines.append("## CONFIRMED_DUPLICATE")
    lines.append("")
    if not confirmed:
        lines.append("_None._")
    else:
        lines.append("| Canonical (rec.) | Alias | Case / date | Queries | Human? | 015? |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for f in confirmed:
            ids = f["document_ids"]
            canon = f.get("canonical_recommendation") or ""
            alias = [x for x in ids if x != canon]
            alias_s = alias[0] if alias else ids[1]
            case = f.get("case_reference_normalized") or ""
            date = f.get("decision_date") or ""
            q = ", ".join(f.get("affected_query_ids") or [])
            human = "yes" if f.get("human_judgment_on_either") else "no"
            is015 = "yes" if f.get("already_resolved_example_015") else ""
            lines.append(
                f"| `{canon}` | `{alias_s}` | {case} / {date} | {q} | {human} | {is015} |"
            )
    lines.append("")
    lines.append("## NEEDS_HUMAN_CHECK")
    lines.append("")
    if not needs:
        lines.append("_None._")
    else:
        lines.append(
            "| Doc A | Doc B | Case / date | ECLI variant? | Content ratio | Queries | Human? |"
        )
        lines.append("| --- | --- | --- | --- | ---: | --- | --- |")
        for f in needs:
            a, b = f["document_ids"]
            case = f.get("case_reference_normalized") or ""
            date = f.get("decision_date") or ""
            var = "yes" if f.get("ecli_representation_variant") else "no"
            ratio = (f.get("content_evidence") or {}).get("seq_ratio_first400", "")
            q = ", ".join(f.get("affected_query_ids") or [])
            human = "yes" if f.get("human_judgment_on_either") else "no"
            lines.append(
                f"| `{a}` | `{b}` | {case} / {date} | {var} | {ratio} | {q} | {human} |"
            )
    lines.append("")
    lines.append("## DISTINCT_DECISIONS (same case/date/court, different content)")
    lines.append("")
    lines.append(f"Count: **{len(distinct)}** (see JSONL for evidence).")
    lines.append("")
    lines.append("## Method (short)")
    lines.append("")
    lines.append(
        "- Strong CONFIRMED_DUPLICATE only when same court + decision_date + "
        "case_reference (optional `#N` stripped) **and** ECLI representation variant "
        "(trailing single-digit `.N` / missing `.1` with matching stem) **and** strong content "
        "(prefix120 equal or SequenceMatcher ≥ 0.92 on first 400 chars)."
    )
    lines.append(
        "- Same identifiers with clearly different content → DISTINCT_DECISIONS."
    )
    lines.append(
        "- Same identifiers with weak/missing content → NEEDS_HUMAN_CHECK."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    p.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT_JSONL)
    p.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_jsonl(args.queue)
    findings, summary = audit(rows)
    for i, f in enumerate(findings, start=1):
        f["finding_id"] = f"dup-{i:04d}"
    write_jsonl(args.out_jsonl, findings)
    args.out_md.write_text(render_md(findings, summary), encoding="utf-8")
    out = {
        "summary": summary,
        "confirmed_duplicate_groups": [
            {
                "document_ids": f["document_ids"],
                "canonical_recommendation": f.get("canonical_recommendation"),
                "affected_query_ids": f.get("affected_query_ids"),
                "already_resolved_example_015": f.get("already_resolved_example_015"),
            }
            for f in findings
            if f["classification"] == "CONFIRMED_DUPLICATE"
        ],
        "needs_human_check_count": summary["needs_human_check_pairs"],
        "out_jsonl": str(args.out_jsonl),
        "out_md": str(args.out_md),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
