#!/usr/bin/env python3
"""Merge all NALUS RAG eval runs into one final comparison report."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

BASE = Path(__file__).resolve().parent

# Completed from initial BM25 batch — do not take failed entries from out_bm25 for these.
COMPLETED_FROM_BM25 = frozenset({"bm25__bm25", "multilingual_e5_small__dense_plus_bm25"})

RERUN_DIRS = {
    "multilingual_e5_base__dense_plus_bm25": BASE / "out_rerun_e5_base",
    "paraphrase_multilingual_mpnet_base_v2__dense_plus_bm25": BASE / "out_rerun_mpnet",
    "multilingual_e5_large__dense_plus_bm25": BASE / "out_rerun_e5_large",
    "bge_m3__dense_plus_bm25": BASE / "out_rerun_bge_m3",
}

ENV_FIXES = [
    "torch upgraded from 2.5.1+cpu to 2.6.0+cpu for bge_m3 (CVE-2025-32434)",
    "failed hybrid candidates rerun one-by-one (separate artifact_dir per model)",
    "BM25 package 0.2.0 installed from /tmp copy (writable, not read-only mount)",
]


def _load_ranking(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_rows(ranking_payload: dict, *, source_label: str) -> list[dict]:
    rows: list[dict] = []
    for item in ranking_payload.get("ranking", []):
        metrics = item.get("metrics", {})
        config_id = item.get("config_id", "")
        rows.append(
            {
                "config_id": config_id,
                "model_code": item.get("model_code"),
                "retrieval_mode": item.get("retrieval_mode") or config_id.split("__")[-1],
                "hit_rate": metrics.get("hit_rate"),
                "evidence_marker_coverage": metrics.get("evidence_marker_coverage"),
                "mrr": metrics.get("mrr"),
                "recall_at_k": metrics.get("recall_at_k"),
                "status": "completed",
                "error": None,
                "source_run_id": ranking_payload.get("run_id"),
                "source_label": source_label,
            }
        )
    return rows


def _failed_rows(ranking_payload: dict, *, source_label: str) -> list[dict]:
    rows: list[dict] = []
    for item in ranking_payload.get("failed_models", []):
        config_id = item.get("config_id") or f"{item.get('model_code')}__unknown"
        rows.append(
            {
                "config_id": config_id,
                "model_code": item.get("model_code"),
                "retrieval_mode": config_id.split("__")[-1] if "__" in config_id else "unknown",
                "hit_rate": None,
                "evidence_marker_coverage": None,
                "mrr": None,
                "recall_at_k": None,
                "status": item.get("status", "FAILED"),
                "error": item.get("error"),
                "source_run_id": ranking_payload.get("run_id"),
                "source_label": source_label,
            }
        )
    return rows


def _dataset_stats() -> dict[str, int]:
    db_path = BASE / "nalus_chunks.sqlite"
    eval_path = BASE / "nalus_eval.json"
    question_count = len(json.loads(eval_path.read_text(encoding="utf-8")).get("cases", []))
    with sqlite3.connect(db_path) as conn:
        chunk_count = conn.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
    return {"questions": question_count, "chunks": int(chunk_count)}


def merge_all() -> dict:
    by_config: dict[str, dict] = {}
    failures: list[dict] = []
    sources_used: list[str] = []
    reruns_applied: list[str] = []

    dense = _load_ranking(BASE / "out_dense_baseline" / "ranking.json")
    if dense:
        sources_used.append(f"dense_baseline:{dense.get('run_id')}")
        for row in _candidate_rows(dense, source_label="dense_baseline"):
            by_config[row["config_id"]] = row
        for row in _failed_rows(dense, source_label="dense_baseline"):
            failures.append(row)

    bm25 = _load_ranking(BASE / "out_bm25" / "ranking.json")
    if bm25:
        sources_used.append(f"bm25_batch:{bm25.get('run_id')}")
        for row in _candidate_rows(bm25, source_label="bm25_batch"):
            if row["config_id"] in COMPLETED_FROM_BM25:
                by_config[row["config_id"]] = row
        for row in _failed_rows(bm25, source_label="bm25_batch"):
            if row["config_id"] not in RERUN_DIRS:
                failures.append(row)

    for config_id, rerun_dir in RERUN_DIRS.items():
        rerun = _load_ranking(rerun_dir / "ranking.json")
        if rerun is None:
            failures.append(
                {
                    "config_id": config_id,
                    "model_code": config_id.split("__")[0],
                    "retrieval_mode": "dense_plus_bm25",
                    "hit_rate": None,
                    "evidence_marker_coverage": None,
                    "mrr": None,
                    "recall_at_k": None,
                    "status": "NOT_RUN",
                    "error": f"Missing {rerun_dir / 'ranking.json'}",
                    "source_run_id": None,
                    "source_label": "rerun",
                }
            )
            continue
        reruns_applied.append(f"{config_id} -> {rerun.get('run_id')}")
        sources_used.append(f"rerun:{rerun.get('run_id')} ({config_id})")
        completed = _candidate_rows(rerun, source_label="rerun")
        if completed:
            by_config[config_id] = completed[0]
        else:
            for row in _failed_rows(rerun, source_label="rerun"):
                failures.append(row)

    completed_rows = list(by_config.values())
    completed_rows.sort(
        key=lambda row: (
            -(row["hit_rate"] or 0),
            -(row["evidence_marker_coverage"] or 0),
            -(row["mrr"] or 0),
        )
    )

    expected_configs = [
        "bm25__bm25",
        "multilingual_e5_small__dense_plus_bm25",
        "multilingual_e5_base__dense_plus_bm25",
        "paraphrase_multilingual_mpnet_base_v2__dense_plus_bm25",
        "multilingual_e5_large__dense_plus_bm25",
        "bge_m3__dense_plus_bm25",
        "multilingual_e5_small__dense",
    ]
    missing = [c for c in expected_configs if c not in by_config]

    stats = _dataset_stats()
    winner = completed_rows[0] if completed_rows else None
    bm25_row = by_config.get("bm25__bm25")
    best_dense_hybrid = None
    for row in completed_rows:
        if row["config_id"] != "bm25__bm25" and row["retrieval_mode"] in {"dense", "dense_plus_bm25"}:
            best_dense_hybrid = row
            break

    benchmark_complete = len(missing) == 0 and not any(
        f["status"] in {"FAILED", "NOT_RUN"} for f in failures if f["config_id"] in RERUN_DIRS
    )

    return {
        "benchmark_complete": benchmark_complete,
        "dataset_questions": stats["questions"],
        "dataset_chunks": stats["chunks"],
        "expected_candidate_count": len(expected_configs),
        "completed_candidate_count": len(completed_rows),
        "missing_candidates": missing,
        "environment_fixes": ENV_FIXES,
        "sources_used": sources_used,
        "reruns_applied": reruns_applied,
        "winner": winner,
        "bm25_beats_best_dense_or_hybrid": (
            bm25_row is not None
            and best_dense_hybrid is not None
            and (bm25_row["hit_rate"] or 0) >= (best_dense_hybrid["hit_rate"] or 0)
            and (bm25_row["mrr"] or 0) >= (best_dense_hybrid["mrr"] or 0)
        )
        if bm25_row and best_dense_hybrid
        else None,
        "ranking": completed_rows,
        "failures": failures,
    }


def write_report(payload: dict) -> tuple[Path, Path]:
    output_dir = BASE / "out_combined"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "combined_ranking.json"
    md_path = output_dir / "combined_report.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# NALUS RAG Eval — Final Combined Comparison",
        "",
        f"- Benchmark complete: **{payload['benchmark_complete']}**",
        f"- Dataset: **{payload['dataset_questions']} questions**, **{payload['dataset_chunks']} chunks**",
        f"- Completed candidates: **{payload['completed_candidate_count']} / {payload['expected_candidate_count']}**",
        "",
        "## Environment fixes applied",
        "",
    ]
    for fix in payload["environment_fixes"]:
        lines.append(f"- {fix}")
    if payload["reruns_applied"]:
        lines.extend(["", "## Reruns applied", ""])
        for item in payload["reruns_applied"]:
            lines.append(f"- {item}")

    if payload.get("missing_candidates"):
        lines.extend(["", "## Missing candidates", ""])
        for item in payload["missing_candidates"]:
            lines.append(f"- `{item}`")

    winner = payload.get("winner")
    if winner:
        lines.extend(
            [
                "",
                "## Overall winner (among completed)",
                "",
                f"- `{winner['config_id']}`",
                f"- hit_rate: `{winner['hit_rate']}`",
                f"- evidence_marker_coverage: `{winner['evidence_marker_coverage']}`",
                f"- mrr: `{winner['mrr']}`",
                f"- source: `{winner['source_label']}` / `{winner['source_run_id']}`",
            ]
        )

    beats = payload.get("bm25_beats_best_dense_or_hybrid")
    if beats is not None:
        lines.extend(["", f"## BM25 vs best dense/hybrid: **{'yes' if beats else 'no'}**", ""])

    lines.extend(["", "## Final ranking (completed only)", ""])
    for index, row in enumerate(payload["ranking"], start=1):
        lines.append(
            f"{index}. `{row['config_id']}` "
            f"(hit_rate={row['hit_rate']}, coverage={row['evidence_marker_coverage']}, "
            f"mrr={row['mrr']}, source={row['source_label']})"
        )

    if payload.get("failures"):
        lines.extend(["", "## Failures / not run", ""])
        for row in payload["failures"]:
            lines.append(
                f"- `{row['config_id']}` [{row['status']}]: {row.get('error') or 'n/a'}"
            )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> None:
    payload = merge_all()
    json_path, md_path = write_report(payload)
    winner = payload.get("winner")
    if winner:
        print(f"Winner: {winner['config_id']} (hit_rate={winner['hit_rate']})")
    print(f"Complete: {payload['benchmark_complete']} ({payload['completed_candidate_count']}/{payload['expected_candidate_count']})")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
