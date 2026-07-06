#!/usr/bin/env python3
"""Merge dense baseline + BM25/hybrid benchmark rankings into one comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_ranking(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing ranking file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_rows(ranking_payload: dict) -> list[dict]:
    rows: list[dict] = []
    for item in ranking_payload.get("ranking", []):
        metrics = item.get("metrics", {})
        rows.append(
            {
                "config_id": item.get("config_id"),
                "model_code": item.get("model_code"),
                "retrieval_mode": item.get("retrieval_mode") or item.get("config_id", "").split("__")[-1],
                "hit_rate": metrics.get("hit_rate"),
                "evidence_marker_coverage": metrics.get("evidence_marker_coverage"),
                "mrr": metrics.get("mrr"),
                "recall_at_k": metrics.get("recall_at_k"),
                "source_run_id": ranking_payload.get("run_id"),
            }
        )
    return rows


def merge_rankings(*, dense_path: Path, bm25_path: Path, output_dir: Path) -> dict:
    dense = _load_ranking(dense_path)
    bm25 = _load_ranking(bm25_path)

    combined_rows = _candidate_rows(dense) + _candidate_rows(bm25)
    combined_rows.sort(
        key=lambda row: (
            -(row["hit_rate"] or 0),
            -(row["evidence_marker_coverage"] or 0),
            -(row["mrr"] or 0),
        )
    )

    winner = combined_rows[0] if combined_rows else None
    payload = {
        "dense_baseline_run_id": dense.get("run_id"),
        "bm25_hybrid_run_id": bm25.get("run_id"),
        "candidate_count": len(combined_rows),
        "winner": winner,
        "ranking": combined_rows,
        "dense_failed_models": dense.get("failed_models", []),
        "bm25_failed_models": bm25.get("failed_models", []),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "combined_ranking.json"
    md_path = output_dir / "combined_report.md"

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# NALUS RAG Eval Combined Comparison",
        "",
        f"- Dense baseline run: `{dense.get('run_id')}`",
        f"- BM25/hybrid run: `{bm25.get('run_id')}`",
        f"- Combined candidates: `{len(combined_rows)}`",
        "",
    ]
    if winner:
        lines.extend(
            [
                "## Overall winner",
                "",
                f"- `{winner['config_id']}`",
                f"- hit_rate: `{winner['hit_rate']}`",
                f"- evidence_marker_coverage: `{winner['evidence_marker_coverage']}`",
                f"- mrr: `{winner['mrr']}`",
                "",
            ]
        )

    lines.extend(["## Ranking", ""])
    for index, row in enumerate(combined_rows, start=1):
        lines.append(
            f"{index}. `{row['config_id']}` "
            f"(hit_rate={row['hit_rate']}, coverage={row['evidence_marker_coverage']}, "
            f"mrr={row['mrr']}, run={row['source_run_id']})"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dense-ranking",
        type=Path,
        default=Path("artifacts/rag_eval/out_dense_baseline/ranking.json"),
    )
    parser.add_argument(
        "--bm25-ranking",
        type=Path,
        default=Path("artifacts/rag_eval/out_bm25/ranking.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/rag_eval/out_combined"),
    )
    args = parser.parse_args()
    result = merge_rankings(
        dense_path=args.dense_ranking,
        bm25_path=args.bm25_ranking,
        output_dir=args.output_dir,
    )
    winner = result.get("winner")
    if winner:
        print(f"Winner: {winner['config_id']} (hit_rate={winner['hit_rate']})")
    print(f"Wrote {args.output_dir / 'combined_ranking.json'}")


if __name__ == "__main__":
    main()
