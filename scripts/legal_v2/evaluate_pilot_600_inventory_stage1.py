#!/usr/bin/env python3
"""Evaluate pilot_600 inventory search queries against Stage 1 HTTP API.

For each query in pilot_600_search_queries.jsonl, call
POST /api/rag/legal-v2/case-similarity/search and measure whether the expected
ECLI appears in Top-K.

Heavy: ~1244 queries. Prefer running locally in PowerShell.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUERIES = (
    PROJECT_ROOT
    / "artifacts"
    / "legal_v2"
    / "pilot_600_judgment_inventory"
    / "pilot_600_search_queries.jsonl"
)
DEFAULT_OUT = PROJECT_ROOT / "artifacts" / "legal_v2" / "pilot_600_inventory_stage1_eval"


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str, *, timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _load_queries(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _rank_of_ecli(results: list[dict[str, Any]], expected_ecli: str) -> int | None:
    expected = expected_ecli.strip().upper()
    for index, item in enumerate(results, start=1):
        ecli = str(item.get("ecli") or item.get("canonical_document_id") or item.get("document_id") or "")
        if ecli.strip().upper() == expected:
            return index
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default="http://localhost:8029")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=10, help="Top-K window for Hit@K")
    parser.add_argument("--max-queries", type=int, default=0, help="0 = all queries")
    parser.add_argument("--query-index", type=int, default=0, help="0=all, 1=short only, 2=long only")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--sleep-ms", type=int, default=0)
    args = parser.parse_args()

    api_base = args.api_base.rstrip("/")
    ready_url = f"{api_base}/api/rag/legal-v2/case-similarity/ready"
    search_url = f"{api_base}/api/rag/legal-v2/case-similarity/search"

    ready = _get_json(ready_url, timeout_s=min(30.0, args.timeout_s))
    if not ready.get("ready"):
        raise SystemExit(f"Stage 1 API not ready: {ready}")

    rows = _load_queries(args.queries)
    if args.query_index in {1, 2}:
        rows = [row for row in rows if int(row.get("query_index") or 0) == args.query_index]
    if args.max_queries and args.max_queries > 0:
        rows = rows[: args.max_queries]
    if not rows:
        raise SystemExit("No queries to evaluate")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (DEFAULT_OUT / run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "inventory_stage1_results.jsonl"
    summary_path = output_dir / "inventory_stage1_summary.json"
    report_path = output_dir / "inventory_stage1_report.md"

    hit_at = {1: 0, 3: 0, 5: 0, 10: 0}
    if args.limit not in hit_at:
        hit_at[args.limit] = 0
    mrr_sum = 0.0
    errors = 0
    missed: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    topic_hits = Counter()
    topic_total = Counter()

    started = time.perf_counter()
    with results_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            query = str(row["query"])
            expected = str(row["ecli"])
            query_started = time.perf_counter()
            record: dict[str, Any] = {
                "query_id": row.get("query_id"),
                "ecli": expected,
                "case_number": row.get("case_number"),
                "query_index": row.get("query_index"),
                "query": query,
                "topic_tags": row.get("topic_tags") or [],
            }
            try:
                response = _post_json(
                    search_url,
                    {"query": query, "limit": args.limit},
                    timeout_s=args.timeout_s,
                )
                latency_ms = (time.perf_counter() - query_started) * 1000.0
                latencies_ms.append(latency_ms)
                results = list(response.get("results") or [])
                rank = _rank_of_ecli(results, expected)
                record.update(
                    {
                        "ok": True,
                        "rank": rank,
                        "hit_at_limit": rank is not None,
                        "result_count": len(results),
                        "latency_ms": round(latency_ms, 1),
                        "top_eclis": [
                            str(item.get("ecli") or item.get("document_id") or "")
                            for item in results[: min(5, len(results))]
                        ],
                    }
                )
                if rank is not None:
                    mrr_sum += 1.0 / float(rank)
                    for k in list(hit_at):
                        if rank <= k:
                            hit_at[k] += 1
                else:
                    missed.append(
                        {
                            "query_id": record["query_id"],
                            "ecli": expected,
                            "case_number": row.get("case_number"),
                            "query": query,
                            "top_eclis": record["top_eclis"],
                        }
                    )
                tags = list(row.get("topic_tags") or []) or ["(untagged)"]
                for tag in tags[:3]:
                    topic_total[tag] += 1
                    if rank is not None and rank <= args.limit:
                        topic_hits[tag] += 1
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
                errors += 1
                record.update(
                    {
                        "ok": False,
                        "rank": None,
                        "hit_at_limit": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:300],
                    }
                )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if index % 25 == 0 or index == len(rows):
                print(f"progress={index}/{len(rows)} errors={errors}")
            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)

    total = len(rows)
    scored = total - errors
    elapsed_s = time.perf_counter() - started
    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": run_id,
        "api_base": api_base,
        "queries_path": str(args.queries),
        "query_count": total,
        "scored_count": scored,
        "error_count": errors,
        "limit": args.limit,
        "query_index_filter": args.query_index,
        "hit_at_1": (hit_at.get(1, 0) / scored) if scored else None,
        "hit_at_3": (hit_at.get(3, 0) / scored) if scored else None,
        "hit_at_5": (hit_at.get(5, 0) / scored) if scored else None,
        "hit_at_10": (hit_at.get(10, 0) / scored) if scored else None,
        "mrr": (mrr_sum / scored) if scored else None,
        "hit_counts": {str(k): hit_at[k] for k in sorted(hit_at)},
        "avg_latency_ms": (sum(latencies_ms) / len(latencies_ms)) if latencies_ms else None,
        "elapsed_s": round(elapsed_s, 1),
        "missed_count": len(missed),
        "topic_hit_at_limit": {
            tag: {
                "hit": topic_hits[tag],
                "total": topic_total[tag],
                "rate": round(topic_hits[tag] / topic_total[tag], 4) if topic_total[tag] else None,
            }
            for tag in sorted(topic_total, key=lambda t: (-topic_total[t], t))[:25]
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Pilot 600 inventory → Stage 1 eval",
        "",
        f"- Run: `{run_id}`",
        f"- API: `{api_base}`",
        f"- Queries: **{total}** (scored {scored}, errors {errors})",
        f"- Limit: Top-{args.limit}",
        "",
        "## Metrics",
        "",
        f"- Hit@1: `{summary['hit_at_1']}`",
        f"- Hit@3: `{summary['hit_at_3']}`",
        f"- Hit@5: `{summary['hit_at_5']}`",
        f"- Hit@10: `{summary['hit_at_10']}`",
        f"- MRR: `{summary['mrr']}`",
        f"- Avg latency ms: `{summary['avg_latency_ms']}`",
        f"- Elapsed s: `{summary['elapsed_s']}`",
        "",
        "## Missed (sample)",
        "",
    ]
    for item in missed[:30]:
        lines.append(f"- `{item.get('case_number') or item['ecli']}` :: {item['query'][:120]}")
        lines.append(f"  - top: {', '.join(item.get('top_eclis') or [])}")
    if len(missed) > 30:
        lines.append(f"- … and {len(missed) - 30} more")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({k: summary[k] for k in ("query_count", "scored_count", "error_count", "hit_at_1", "hit_at_10", "mrr", "elapsed_s")}, ensure_ascii=False, indent=2))
    print(f"summary={summary_path}")
    print(f"report={report_path}")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
