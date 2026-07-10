"""Apply human-verified gold source annotations to legal Q&A datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.eval.legal_qa_benchmark import validate_dataset_item  # noqa: E402

# Verified from frozen baseline runs (2026-07-09); rank-1 hit@1 with stable ECLI.
USOUD_GOLD_ECLI: dict[str, str] = {
    "usoud-qa-001": "ECLI:CZ:US:2026:3.US.3031.24.1",
    "usoud-qa-002": "ECLI:CZ:US:2026:4.US.2338.25.1",
    "usoud-qa-003": "ECLI:CZ:US:2023:1.US.3171.22.1",
    "usoud-qa-004": "ECLI:CZ:US:2023:1.US.631.23.1",
    "usoud-qa-007": "ECLI:CZ:US:2026:2.US.927.26.1",
    "usoud-qa-009": "ECLI:CZ:US:2026:1.US.2699.25.1",
    "usoud-qa-010": "ECLI:CZ:US:2023:3.US.714.23.1",
    "usoud-qa-011": "ECLI:CZ:US:2026:4.US.1079.26.1",
    "usoud-qa-012": "ECLI:CZ:US:2026:4.US.1065.26.1",
    "usoud-qa-015": "ECLI:CZ:US:2026:2.US.3645.25.1",
}

NSOUD_GOLD_ECLI: dict[str, str] = {
    "nsoud-qa-003": "ECLI:CZ:NS:2025:21.CDO.372.2024.1",
    "nsoud-qa-004": "ECLI:CZ:NS:2024:8.TDO.760.2024.1",
    "nsoud-qa-007": "ECLI:CZ:NS:2025:5.TDO.1086.2024.1",
    "nsoud-qa-010": "ECLI:CZ:NS:2025:29.NSCR.1.2025.1",
}

# Corpus routing verified in mixed baseline (corpus_hit@3); no document-level gold yet.
MIXED_CORPUS_VERIFIED_IDS = frozenset(
    {
        "mixed-qa-001",
        "mixed-qa-002",
        "mixed-qa-003",
        "mixed-qa-005",
        "mixed-qa-006",
        "mixed-qa-007",
        "mixed-qa-008",
        "mixed-qa-009",
    }
)

DATASET_PATHS = {
    "usoud": PROJECT_ROOT / "artifacts/rag_eval/legal_qa/datasets/usoud_qa_v1.jsonl",
    "nsoud": PROJECT_ROOT / "artifacts/rag_eval/legal_qa/datasets/nsoud_qa_v1.jsonl",
    "mixed": PROJECT_ROOT / "artifacts/rag_eval/legal_qa/datasets/mixed_qa_v1.jsonl",
}


def _gold_constraints(*, ecli: str | None = None) -> dict[str, Any]:
    return {
        "court": None,
        "source": None,
        "case_reference": None,
        "source_document_id": ecli,
        "decision_date": None,
    }


def annotate_item(item: dict[str, Any]) -> dict[str, Any]:
    item_id = str(item["id"])
    corpus = str(item["corpus"])

    if item_id in USOUD_GOLD_ECLI:
        item["source_pending"] = False
        item["expected_source_constraints"] = _gold_constraints(ecli=USOUD_GOLD_ECLI[item_id])
        return item

    if item_id in NSOUD_GOLD_ECLI:
        item["source_pending"] = False
        item["expected_source_constraints"] = _gold_constraints(ecli=NSOUD_GOLD_ECLI[item_id])
        return item

    if corpus == "mixed" and item_id in MIXED_CORPUS_VERIFIED_IDS:
        item["source_pending"] = False
        return item

    return item


def annotate_dataset(path: Path) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = annotate_item(json.loads(line))
        validate_dataset_item(item)
        updated.append(item)
    return updated


def write_dataset(path: Path, items: list[dict[str, Any]]) -> None:
    lines = [json.dumps(item, ensure_ascii=False) for item in items]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply gold source annotations to legal Q&A datasets.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary = {
        "usoud_gold": len(USOUD_GOLD_ECLI),
        "nsoud_gold": len(NSOUD_GOLD_ECLI),
        "mixed_corpus_verified": len(MIXED_CORPUS_VERIFIED_IDS),
        "total_gold_touched": len(USOUD_GOLD_ECLI) + len(NSOUD_GOLD_ECLI) + len(MIXED_CORPUS_VERIFIED_IDS),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.dry_run:
        return 0

    for path in DATASET_PATHS.values():
        write_dataset(path, annotate_dataset(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
