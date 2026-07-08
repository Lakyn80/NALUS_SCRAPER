#!/usr/bin/env python3
"""Merge client long-form benchmark runs into ranking.json and combined_report.md."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

BASE = Path(__file__).resolve().parent
RAG_EVAL = BASE.parent
DATASET_PATH = RAG_EVAL / "nalus_client_longform_eval_v1.json"

RUNS = [
    ("multilingual_e5_small__dense", BASE / "out_dense_e5_small"),
    ("bm25__bm25", BASE / "out_bm25"),
    ("multilingual_e5_small__dense_plus_bm25", BASE / "out_hybrid_e5_small"),
    ("multilingual_e5_base__dense_plus_bm25", BASE / "out_hybrid_e5_base"),
    ("multilingual_e5_large__dense_plus_bm25", BASE / "out_hybrid_e5_large"),
    ("paraphrase_multilingual_mpnet_base_v2__dense_plus_bm25", BASE / "out_hybrid_mpnet"),
    ("bge_m3__dense_plus_bm25", BASE / "out_hybrid_bge_m3"),
]


def _load_ranking(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _row_from_ranking(ranking: dict, config_id: str, source_label: str) -> dict | None:
    for item in ranking.get("ranking", []):
        if item.get("config_id") == config_id:
            metrics = item.get("metrics", {})
            return {
                "config_id": config_id,
                "model_code": item.get("model_code"),
                "retrieval_mode": item.get("retrieval_mode") or config_id.split("__")[-1],
                "collection_name": item.get("collection_name"),
                "hit_rate": metrics.get("hit_rate"),
                "evidence_marker_coverage": metrics.get("evidence_marker_coverage"),
                "mrr": metrics.get("mrr"),
                "recall_at_k": metrics.get("recall_at_k"),
                "forbidden_marker_rate": metrics.get("forbidden_marker_rate"),
                "missing_expected_marker_count": metrics.get("missing_expected_marker_count"),
                "false_positive_count": metrics.get("false_positive_count"),
                "average_latency_ms": metrics.get("average_latency_ms"),
                "status": "completed",
                "source_run_id": ranking.get("run_id"),
                "source_label": source_label,
            }
    winner = ranking.get("winner") or {}
    if winner.get("config_id") == config_id:
        metrics = winner.get("metrics", {})
        return {
            "config_id": config_id,
            "model_code": winner.get("model_code"),
            "retrieval_mode": config_id.split("__")[-1],
            "collection_name": winner.get("collection_name"),
            "hit_rate": metrics.get("hit_rate"),
            "evidence_marker_coverage": metrics.get("evidence_marker_coverage"),
            "mrr": metrics.get("mrr"),
            "recall_at_k": metrics.get("recall_at_k"),
            "forbidden_marker_rate": metrics.get("forbidden_marker_rate"),
            "missing_expected_marker_count": metrics.get("missing_expected_marker_count"),
            "false_positive_count": metrics.get("false_positive_count"),
            "average_latency_ms": metrics.get("average_latency_ms"),
            "status": "completed",
            "source_run_id": ranking.get("run_id"),
            "source_label": source_label,
        }
    return None


def _dataset_stats() -> dict[str, int]:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    db_path = RAG_EVAL / "nalus_chunks.sqlite"
    with sqlite3.connect(db_path) as conn:
        chunk_count = conn.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
    return {"questions": len(dataset.get("cases", [])), "chunks": int(chunk_count)}


def merge_all() -> dict:
    completed: list[dict] = []
    failures: list[dict] = []

    for config_id, out_dir in RUNS:
        ranking = _load_ranking(out_dir / "ranking.json")
        if ranking is None:
            failures.append(
                {
                    "config_id": config_id,
                    "status": "NOT_RUN",
                    "error": f"Missing {out_dir / 'ranking.json'}",
                }
            )
            continue
        row = _row_from_ranking(ranking, config_id, out_dir.name)
        if row:
            completed.append(row)
            continue
        failed = next(
            (f for f in ranking.get("failed_models", []) if f.get("config_id") == config_id),
            None,
        )
        if failed:
            failures.append(
                {
                    "config_id": config_id,
                    "status": failed.get("status", "FAILED"),
                    "error": failed.get("error"),
                }
            )
        else:
            failures.append(
                {
                    "config_id": config_id,
                    "status": "FAILED",
                    "error": "No ranking row for expected config_id",
                }
            )

    completed.sort(
        key=lambda row: (
            -(row["hit_rate"] or 0),
            -(row["evidence_marker_coverage"] or 0),
            -(row["mrr"] or 0),
        )
    )
    expected = [config_id for config_id, _ in RUNS]
    missing = [c for c in expected if c not in {r["config_id"] for r in completed}]
    stats = _dataset_stats()
    winner = completed[0] if completed else None

    return {
        "benchmark_complete": len(missing) == 0 and not failures,
        "dataset_id": "nalus-client-longform-v1",
        "dataset_questions": stats["questions"],
        "dataset_chunks": stats["chunks"],
        "expected_candidate_count": len(expected),
        "completed_candidate_count": len(completed),
        "missing_candidates": missing,
        "winner": winner,
        "ranking": completed,
        "failures": failures,
    }


def write_outputs(payload: dict) -> tuple[Path, Path]:
    json_path = BASE / "ranking.json"
    md_path = BASE / "combined_report.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# NALUS Client Long-Form RAG Eval — Combined Report",
        "",
        f"- Benchmark complete: **{payload['benchmark_complete']}**",
        f"- Dataset: **{payload['dataset_questions']} client-style questions**, **{payload['dataset_chunks']} chunks**",
        f"- Completed candidates: **{payload['completed_candidate_count']} / {payload['expected_candidate_count']}**",
        "",
        "## Interpretation",
        "",
        "This benchmark tests semantic retrieval from long client narratives, not keyword or ECLI lookup.",
        "Different ECLI from dataset scope may still be legally relevant (`alternate_relevant`).",
        "",
        "## Benchmark metrics ranking",
        "",
        "| rank | config_id | hit_rate | recall_at_k | mrr | coverage | false_pos | latency_ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(payload["ranking"], start=1):
        lines.append(
            f"| {index} | `{row['config_id']}` | {row['hit_rate']} | {row['recall_at_k']} | "
            f"{row['mrr']} | {row['evidence_marker_coverage']} | {row['false_positive_count']} | "
            f"{row.get('average_latency_ms')} |"
        )

    winner = payload.get("winner")
    if winner:
        lines.extend(
            [
                "",
                "## Winner",
                "",
                f"- `{winner['config_id']}`",
                f"- hit_rate: `{winner['hit_rate']}`",
                f"- mrr: `{winner['mrr']}`",
                f"- evidence_marker_coverage: `{winner['evidence_marker_coverage']}`",
            ]
        )

    if payload.get("failures"):
        lines.extend(["", "## Failures", ""])
        for row in payload["failures"]:
            lines.append(f"- `{row['config_id']}` [{row['status']}]: {row.get('error') or 'n/a'}")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> None:
    payload = merge_all()
    json_path, md_path = write_outputs(payload)
    print(f"Complete: {payload['benchmark_complete']} ({payload['completed_candidate_count']}/{payload['expected_candidate_count']})")
    if payload.get("winner"):
        print(f"Winner: {payload['winner']['config_id']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
