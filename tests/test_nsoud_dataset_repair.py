from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.apply_gold_source_annotations import annotate_dataset


DATASET_PATH = Path("artifacts/rag_eval/legal_qa/datasets/nsoud_qa_v1.jsonl")


def _load_raw_items() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _items_by_id() -> dict[str, dict[str, Any]]:
    return {str(item["id"]): item for item in _load_raw_items()}


def test_repaired_nsoud_items_match_evidence_backed_decisions() -> None:
    items = _items_by_id()

    assert items["nsoud-qa-003"]["expected_keywords"] == [
        "přípustnost",
        "dovolání",
        "občanský",
    ]

    item_004 = items["nsoud-qa-004"]
    assert item_004["question"] == (
        "Jaké právní otázky mohou podle § 237 o. s. ř. založit přípustnost dovolání?"
    )
    assert item_004["expected_source_constraints"]["source_document_id"] == (
        "ECLI:CZ:NS:2025:33.CDO.79.2024.1"
    )
    assert item_004["expected_keywords"] == [
        "hmotného",
        "procesního",
        "ustálené rozhodovací praxe",
    ]

    item_007 = items["nsoud-qa-007"]
    assert item_007["expected_source_constraints"]["source_document_id"] == (
        "ECLI:CZ:NS:2025:5.TDO.1086.2024.1"
    )
    assert len(item_007["expected_answer_points"]) == 3
    assert "§ 265b odst. 1 písm. g)" in item_007["expected_answer_points"][1]

    item_010 = items["nsoud-qa-010"]
    assert item_010["question"] == (
        "Je dovolání přípustné proti rozhodnutí, jímž odvolací soud odmítl odvolání?"
    )
    assert item_010["expected_keywords"] == [
        "odmítl odvolání",
        "objektivně přípustné",
        "žaloba pro zmatečnost",
    ]


def test_gold_annotation_reapplication_is_idempotent_for_nsoud_dataset() -> None:
    raw_items = _load_raw_items()

    assert annotate_dataset(DATASET_PATH) == raw_items


def test_repair_does_not_add_unverified_provenance_fields() -> None:
    items = _items_by_id()
    expected_ecli = {
        "nsoud-qa-003": "ECLI:CZ:NS:2025:21.CDO.372.2024.1",
        "nsoud-qa-004": "ECLI:CZ:NS:2025:33.CDO.79.2024.1",
        "nsoud-qa-007": "ECLI:CZ:NS:2025:5.TDO.1086.2024.1",
        "nsoud-qa-010": "ECLI:CZ:NS:2025:29.NSCR.1.2025.1",
    }

    for item_id, ecli in expected_ecli.items():
        item = items[item_id]
        constraints = item["expected_source_constraints"]
        assert item["source_pending"] is False
        assert constraints == {
            "court": None,
            "source": None,
            "case_reference": None,
            "source_document_id": ecli,
            "decision_date": None,
        }
