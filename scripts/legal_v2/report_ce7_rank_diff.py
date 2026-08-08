#!/usr/bin/env python3
"""Build FAST / CE-3 / CE-7 rank-diff and passage-coverage summary artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast-dir", type=Path, required=True)
    parser.add_argument("--ce3-dir", type=Path, required=True)
    parser.add_argument("--ce7-dir", type=Path, required=True)
    parser.add_argument("--diagnostics-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_results(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "case_similarity_retrieval_results.jsonl"
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qid = row.get("query_id") or row.get("benchmark_id")
        if not qid:
            continue
        ranked = []
        for item in row.get("retrieved_results") or row.get("ranked_results") or []:
            ecli = item.get("ecli") or item.get("document_id")
            if ecli:
                ranked.append(ecli)
        if not ranked:
            ranked = list(row.get("retrieved_eclis") or row.get("ranked_document_ids") or [])
        expected = row.get("expected_primary_ecli") or row.get(
            "expected_primary_document_id"
        )
        rank = row.get("primary_rank")
        if rank is None and expected and expected in ranked:
            rank = ranked.index(expected) + 1
        out[qid] = {
            "ranked": ranked,
            "expected": expected,
            "rank": rank,
            "hit_at_1": bool(row.get("hit_at_1")),
            "hit_at_10": bool(row.get("hit_at_10")),
            "mrr": row.get("reciprocal_rank") or row.get("mrr"),
            "hard_negative_outrank": bool(
                row.get("hard_negative_before_positive")
                or row.get("hard_negative_outrank")
            ),
        }
    return out


def _load_metrics(run_dir: Path) -> dict[str, Any]:
    report = run_dir / "case_similarity_retrieval_report.md"
    manifest = run_dir / "ce_experiment_manifest.json"
    metrics: dict[str, Any] = {}
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        metrics.update(payload.get("metrics") or {})
        metrics["manifest"] = payload
    # Prefer machine JSON if present beside report.
    for name in (
        "case_similarity_retrieval_metrics.json",
        "metrics.json",
        "summary.json",
    ):
        path = run_dir / name
        if path.exists():
            metrics.update(json.loads(path.read_text(encoding="utf-8")))
            break
    if report.exists() and "hit_at_1" not in metrics:
        text = report.read_text(encoding="utf-8")
        for key in ("hit_at_1", "hit_at_10", "mrr", "hard_negative_outrank_rate"):
            token = f"{key}="
            if token in text:
                # best-effort parse from printed summary lines
                for line in text.splitlines():
                    if line.startswith(token) or f"| {key} |" in line:
                        pass
    return metrics


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fast = _load_results(args.fast_dir)
    ce3 = _load_results(args.ce3_dir)
    ce7 = _load_results(args.ce7_dir)
    query_ids = sorted(set(fast) | set(ce3) | set(ce7))

    rows = []
    material = []
    for qid in query_ids:
        f_rank = (fast.get(qid) or {}).get("rank")
        c3_rank = (ce3.get(qid) or {}).get("rank")
        c7_rank = (ce7.get(qid) or {}).get("rank")
        delta = None
        if isinstance(c3_rank, int) and isinstance(c7_rank, int):
            delta = c3_rank - c7_rank  # positive => improved (lower rank number)
        row = {
            "query_id": qid,
            "fast_rank": f_rank,
            "ce3_rank": c3_rank,
            "ce7_rank": c7_rank,
            "ce3_to_ce7_delta": delta,
            "fast_hit1": (fast.get(qid) or {}).get("hit_at_1"),
            "ce3_hit1": (ce3.get(qid) or {}).get("hit_at_1"),
            "ce7_hit1": (ce7.get(qid) or {}).get("hit_at_1"),
            "fast_hit10": (fast.get(qid) or {}).get("hit_at_10"),
            "ce3_hit10": (ce3.get(qid) or {}).get("hit_at_10"),
            "ce7_hit10": (ce7.get(qid) or {}).get("hit_at_10"),
            "ce3_hn": (ce3.get(qid) or {}).get("hard_negative_outrank"),
            "ce7_hn": (ce7.get(qid) or {}).get("hard_negative_outrank"),
        }
        rows.append(row)
        interesting = qid in {"nalus-cs-pilot-004", "nalus-cs-pilot-016"}
        interesting = interesting or bool(row["ce3_hit1"]) != bool(row["ce7_hit1"])
        interesting = interesting or bool(row["ce3_hit10"]) != bool(row["ce7_hit10"])
        interesting = interesting or bool(row["ce7_hn"]) or bool(row["ce3_hn"])
        interesting = interesting or (
            bool(row["fast_hit1"]) != bool(row["ce7_hit1"])
        )
        if interesting:
            material.append(row)

    ce7_manifest = _load_metrics(args.ce7_dir)
    ce3_manifest = _load_metrics(args.ce3_dir)
    fast_metrics = _load_metrics(args.fast_dir)

    diagnostics = None
    if args.diagnostics_json and args.diagnostics_json.exists():
        diagnostics = json.loads(args.diagnostics_json.read_text(encoding="utf-8"))

    # Passage coverage stats from CE-7 per-query diagnostics if present in results.
    selected_counts: list[int] = []
    if diagnostics:
        for query in diagnostics.get("queries") or []:
            for focus in query.get("focus_documents") or []:
                if focus.get("is_expected_primary"):
                    selected_counts.append(
                        int((focus.get("requested_vs_selected") or {}).get("ce7_selected") or 0)
                    )

    summary = {
        "fast_dir": str(args.fast_dir),
        "ce3_dir": str(args.ce3_dir),
        "ce7_dir": str(args.ce7_dir),
        "metrics": {
            "fast": fast_metrics,
            "ce3": ce3_manifest,
            "ce7": ce7_manifest,
        },
        "rank_table": rows,
        "material_movements": material,
        "passage_coverage": {
            "expected_primary_selected_counts": selected_counts,
            "mean_selected_for_expected": (
                statistics.fmean(selected_counts) if selected_counts else None
            ),
        },
    }
    (args.output_dir / "ce7_rank_diff.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    md = [
        "# CE-7 rank diff (FAST / CE-3 / CE-7)",
        "",
        "| Query | FAST rank | CE-3 rank | CE-7 rank | CE-3→7 delta |",
        "| ----- | --------: | --------: | --------: | -----------: |",
    ]
    for row in rows:
        md.append(
            f"| {row['query_id']} | {row['fast_rank']} | {row['ce3_rank']} | "
            f"{row['ce7_rank']} | {row['ce3_to_ce7_delta']} |"
        )
    md.extend(["", "## Material movements", ""])
    for row in material:
        md.append(
            f"- `{row['query_id']}` FAST={row['fast_rank']} CE3={row['ce3_rank']} "
            f"CE7={row['ce7_rank']} hit10(ce3/ce7)={row['ce3_hit10']}/{row['ce7_hit10']} "
            f"hn(ce3/ce7)={row['ce3_hn']}/{row['ce7_hn']}"
        )
    (args.output_dir / "ce7_rank_diff.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"output_dir={args.output_dir}")
    print(f"material_movements={len(material)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
