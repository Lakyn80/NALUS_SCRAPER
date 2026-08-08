#!/usr/bin/env python3
"""Export UNREVIEWED confusable-case review rows from FAST vs CE run dirs.

Does not invent legal relevance labels. Manual review required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CONFUSABLE_QUERIES = [
    {
        "theme": "lease_vs_employment_termination",
        "query_hint": "výpověď z nájmu",
    },
    {
        "theme": "custody_merits_vs_procedural_rejection",
        "query_hint": "odmítnutí ústavní stížnosti",
    },
    {
        "theme": "damages_vs_limitation",
        "query_hint": "promlčení",
    },
    {
        "theme": "contract_validity_vs_costs",
        "query_hint": "náklady řízení",
    },
    {
        "theme": "guilt_vs_admissibility",
        "query_hint": "přípustnost",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast-dir", type=Path, required=True)
    parser.add_argument("--ce-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_results(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "case_similarity_retrieval_results.jsonl"
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row.get("query_id") or row.get("benchmark_id") or "")] = row
    return rows


def main() -> int:
    args = parse_args()
    fast = _load_results(args.fast_dir)
    ce = _load_results(args.ce_dir)
    export_rows: list[dict[str, Any]] = []
    for query_id in sorted(set(fast) | set(ce)):
        f = fast.get(query_id) or {}
        c = ce.get(query_id) or {}
        query = str(f.get("query") or c.get("query") or "")
        theme = "unspecified"
        for item in CONFUSABLE_QUERIES:
            if item["query_hint"].casefold() in query.casefold():
                theme = item["theme"]
                break
        export_rows.append(
            {
                "query_id": query_id,
                "query": query,
                "theme": theme,
                "relevance": "UNREVIEWED",
                "fast_top1_ecli": _top1(f),
                "ce_top1_ecli": _top1(c),
                "fast_primary_rank": f.get("primary_rank") or f.get("best_positive_rank"),
                "ce_primary_rank": c.get("primary_rank") or c.get("best_positive_rank"),
                "fast_top": _top_n(f, 5),
                "ce_top": _top_n(c, 5),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(export_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(export_rows)} UNREVIEWED rows -> {args.output}")
    return 0


def _top1(row: dict[str, Any]) -> str | None:
    tops = _top_n(row, 1)
    return tops[0]["ecli"] if tops else None


def _top_n(row: dict[str, Any], n: int) -> list[dict[str, Any]]:
    retrieved = row.get("retrieved_results") or row.get("ranked_results") or []
    out: list[dict[str, Any]] = []
    for item in retrieved[:n]:
        out.append(
            {
                "rank": item.get("rank"),
                "ecli": item.get("ecli") or item.get("document_id"),
                "stage1_score": item.get("score"),
                "ce_score": item.get("reranker_score"),
            }
        )
    return out


if __name__ == "__main__":
    raise SystemExit(main())
