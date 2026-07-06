#!/usr/bin/env python3
"""Run legal-quality evaluation on the BGE-M3 winner QA export."""

from __future__ import annotations

import json
from pathlib import Path

from legal_quality_eval import evaluate_winner_export, render_legal_quality_report

BASE = Path(__file__).resolve().parent
WINNER_QA_PATH = BASE / "out_combined" / "winner_bge_m3_qa.json"
DATASET_PATH = BASE / "nalus_eval.json"
OUT_JSON = BASE / "winner_bge_m3_legal_eval.json"
OUT_MD = BASE / "legal_quality_report.md"


def main() -> None:
    payload = evaluate_winner_export(
        winner_qa_path=WINNER_QA_PATH,
        dataset_path=DATASET_PATH,
    )
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_legal_quality_report(payload), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
