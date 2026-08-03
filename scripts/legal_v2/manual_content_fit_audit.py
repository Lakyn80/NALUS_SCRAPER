#!/usr/bin/env python3
"""Offline content-fit audit: verified docs vs query + benchmark labels."""

from __future__ import annotations

import json
import re
from pathlib import Path

EVAL = Path(
    "artifacts/legal_v2/pilot_600_20260731/universal_quality/thinking_ab_test/hybrid_eval_59_post_fix.json"
)
BENCH = Path(
    "artifacts/legal_v2/pilot_600_20260731/universal_quality/reviewed_benchmark_v2.json"
)
REVIEWS = Path(
    "artifacts/legal_v2/pilot_600_20260731/universal_quality/thinking_ab_test/document_reviews"
)
OUT = Path(
    "artifacts/legal_v2/pilot_600_20260731/universal_quality/thinking_ab_test/manual_content_fit_audit_20260803.json"
)


def _bench_items(payload: dict) -> dict[str, dict]:
    for key in ("items", "queries", "rows"):
        value = payload.get(key)
        if isinstance(value, list) and value and isinstance(value[0], dict) and "id" in value[0]:
            return {str(item["id"]): item for item in value}
    return {}


def _label(doc_id: str, item: dict, fallback: str | None) -> str:
    if doc_id in set(item.get("strongly_relevant_document_ids") or []):
        return "strongly_relevant"
    if doc_id in set(item.get("materially_relevant_document_ids") or []):
        return "materially_relevant"
    if doc_id in set(item.get("explicit_hard_negative_document_ids") or []):
        return "hard_negative"
    if doc_id in set(item.get("related_only_document_ids") or []):
        return "related_only"
    return fallback or "other"


def _extract_doc_text(review_md: str, document_id: str) -> str:
    # Split on rank headers; find block containing document_id
    parts = re.split(r"\n## Rank ", review_md)
    for part in parts[1:]:
        if document_id not in part:
            continue
        m = re.search(r"```text\n(.*?)```", part, flags=re.S)
        return (m.group(1) if m else part)[:12000]
    return ""


def _snippet(text: str, terms: list[str], width: int = 180) -> list[str]:
    low = text.casefold()
    hits: list[str] = []
    for term in terms:
        t = term.casefold().strip()
        if len(t) < 4:
            continue
        idx = low.find(t)
        if idx < 0:
            continue
        start = max(0, idx - 60)
        end = min(len(text), idx + len(term) + 120)
        hits.append(text[start:end].replace("\n", " "))
        if len(hits) >= 4:
            break
    return hits


def main() -> None:
    eval_data = json.loads(EVAL.read_text(encoding="utf-8"))
    bench = _bench_items(json.loads(BENCH.read_text(encoding="utf-8")))
    rows = [row for row in eval_data["rows"] if row.get("status") == "verified_match"]
    report: list[dict] = []
    for row in sorted(rows, key=lambda item: str(item.get("id") or "")):
        qid = str(row["id"])
        item = bench.get(qid) or {}
        query = str(row.get("query") or item.get("query") or "")
        review_path = REVIEWS / f"{qid}_full_documents.md"
        review_md = review_path.read_text(encoding="utf-8") if review_path.exists() else ""
        terms = []
        for bucket in (
            item.get("mandatory_facts") or [],
            item.get("mandatory_legal_concepts") or [],
            item.get("mandatory_jurisdictions") or [],
        ):
            terms.extend(str(x) for x in bucket)
        # also loose tokens from query
        terms.extend([tok for tok in re.findall(r"[A-Za-zÁ-ž0-9§/.-]{4,}", query)])
        verified = []
        for cand in row.get("candidate_documents") or []:
            if cand.get("final_decision") != "verified_match":
                continue
            doc_id = str(cand.get("document_id") or "")
            text = _extract_doc_text(review_md, doc_id)
            label = _label(doc_id, item, cand.get("benchmark_label"))
            verified.append(
                {
                    "document_id": doc_id,
                    "benchmark_label": label,
                    "relevance_classification": cand.get("relevance_classification"),
                    "char_count": len(text),
                    "term_hits": _snippet(text, terms),
                    "opening": text[:350].replace("\n", " "),
                }
            )
        report.append(
            {
                "id": qid,
                "query": query,
                "mandatory_facts": item.get("mandatory_facts"),
                "mandatory_legal_concepts": item.get("mandatory_legal_concepts"),
                "verified_count": len(verified),
                "verified_documents": verified,
                "hard_negative_verified": [
                    v["document_id"] for v in verified if v["benchmark_label"] == "hard_negative"
                ],
                "strong_or_material_verified": [
                    v["document_id"]
                    for v in verified
                    if v["benchmark_label"] in {"strongly_relevant", "materially_relevant", "gold"}
                ],
            }
        )

    summary = {
        "verified_queries": len(report),
        "total_verified_docs": sum(item["verified_count"] for item in report),
        "queries_with_hard_negative_verified": [
            item["id"] for item in report if item["hard_negative_verified"]
        ],
        "queries_without_strong_or_material": [
            item["id"]
            for item in report
            if not item["strong_or_material_verified"] and item["verified_count"]
        ],
        "rows": report,
    }
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "rows"}, ensure_ascii=False, indent=2))
    for item in report:
        print("\n===", item["id"], "===")
        print("Q:", item["query"][:160])
        print("concepts:", item["mandatory_legal_concepts"])
        print("facts:", item["mandatory_facts"])
        for doc in item["verified_documents"]:
            print(
                "-",
                doc["document_id"],
                "|",
                doc["benchmark_label"],
                "|",
                doc["relevance_classification"],
                "| chars",
                doc["char_count"],
            )
            if doc["term_hits"]:
                print("  hit:", doc["term_hits"][0][:160])
            else:
                print("  opening:", doc["opening"][:160])


if __name__ == "__main__":
    main()
