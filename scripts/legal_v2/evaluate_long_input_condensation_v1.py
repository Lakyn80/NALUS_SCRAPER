#!/usr/bin/env python3
"""Evaluate extractive SearchBrief quality for long-input fixtures (no Stage 1 retrieval)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.query_input.config import LongInputConfig  # noqa: E402
from app.rag.legal_v2.query_input.extractive import build_extractive_search_brief  # noqa: E402

SIGNAL_PATTERNS = {
    "constitutional_complaint": ("ústavní stížnost", "stížnost"),
    "mandatory_lawyer_representation": ("advokát", "zastoupen"),
    "formal_defects": ("vad", "formál"),
    "limitation": ("promlč", "prekluz"),
    "costs": ("náklad",),
    "child_custody_merits": ("péč", "svěření"),
    "damages_amount": ("škod", "ušlý"),
    "contract_validity": ("platnost smlouvy", "smlouv"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "legal_v2" / "long_input_condensation_v1.jsonl",
    )
    args = parser.parse_args()
    config = LongInputConfig(enabled=True)
    rows = [
        json.loads(line)
        for line in args.fixture.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    passed = 0
    for row in rows:
        brief = build_extractive_search_brief(row["raw_input"], config=config)
        text = brief.brief_text.lower()
        ok = True
        for signal in row.get("required_signals") or []:
            needles = SIGNAL_PATTERNS.get(signal, (signal,))
            if not any(n in text for n in needles):
                ok = False
                print(f"FAIL {row['id']} missing required={signal} brief={brief.brief_text[:180]}")
        for signal in row.get("forbidden_dominant_signals") or []:
            needles = SIGNAL_PATTERNS.get(signal, (signal,))
            if any(n in text for n in needles) and not (
                "nehled" in text
                or "nejde" in text
                or "neřeš" in text
                or "vad" in text
                or "náklad" in text
                or "promlč" in text
                or brief.negative_focus
            ):
                ok = False
                print(f"FAIL {row['id']} forbidden_dominant={signal}")
        if ok:
            passed += 1
            print(f"PASS {row['id']} brief_len={len(brief.brief_text)}")
    print(f"summary passed={passed}/{len(rows)}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
