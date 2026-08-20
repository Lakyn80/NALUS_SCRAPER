#!/usr/bin/env python3
"""Approve Golden v3 query wording (curated natural rewrites)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from app.rag.legal_v2.benchmark.case_similarity_golden_v3 import (
    CaseSimilarityGoldenV3Item,
    load_case_similarity_golden_v3_jsonl,
    write_case_similarity_golden_v3_jsonl,
)

QR = Path("artifacts/legal_v2/golden_v3_graded/query_review.jsonl")
BENCH = Path("benchmarks/legal_v2/case_similarity_golden_v3_graded.jsonl")


def main() -> None:
    rows = [json.loads(line) for line in QR.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        row["review_status"] = "approved"
        row["reviewer_notes"] = (
            "Query wording approved for graded v3 batch relevance review "
            "(curated natural rewrite)."
        )
    QR.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(dict(Counter(row["review_status"] for row in rows)))

    items = load_case_similarity_golden_v3_jsonl(BENCH)
    updated = []
    for item in items:
        payload = item.model_dump()
        payload["query_review_status"] = "approved"
        payload["query_review_notes"] = "Query wording approved for graded v3 batch relevance review."
        updated.append(CaseSimilarityGoldenV3Item.model_validate(payload))
    write_case_similarity_golden_v3_jsonl(BENCH, updated)
    print("benchmark approved", len(updated))


if __name__ == "__main__":
    main()
