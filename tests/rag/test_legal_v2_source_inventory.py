from __future__ import annotations

import json

from app.rag.legal_v2.source_inventory import build_source_inventory
from app.rag.legal_v2.sources import DecisionDateRange, parse_decision_date


def test_source_inventory_counts_missing_text_ids_and_duplicates(tmp_path) -> None:
    batches = tmp_path / "batches"
    batches.mkdir()
    (batches / "year_2026.json").write_text(
        json.dumps(
            [
                {"ecli": "ECLI:CZ:US:2026:1", "full_text": "text", "decision_date": "1. 1. 2026"},
                {"ecli": "ECLI:CZ:US:2026:1", "full_text": "duplicate", "decision_date": "2. 1. 2026"},
                {"case_reference": "III. US 1/26", "full_text": "", "decision_date": "2026"},
                {"full_text": "missing id"},
            ]
        ),
        encoding="utf-8",
    )
    nsoud = tmp_path / "nsoud.jsonl"
    nsoud.write_text(
        "\n".join(
            [
                json.dumps({"ecli": "ECLI:CZ:NS:2025:1", "text": "chunk", "decision_date": "2025-01-01"}),
                json.dumps({"ecli": "", "text": "missing id"}),
                json.dumps({"ecli": "ECLI:CZ:NS:2025:2", "text": ""}),
            ]
        ),
        encoding="utf-8",
    )

    report = build_source_inventory(batches_dir=batches, nsoud_chunks_path=nsoud)
    payload = report.to_dict()

    assert payload["total_discovered_source_documents"] == 2
    assert payload["documents_missing_stable_document_identifiers"] == 2
    assert payload["documents_missing_complete_text"] == 2
    assert payload["duplicate_source_document_identifiers"] == 1
    assert payload["document_count_per_adapter"] == {"constitutional": 1, "supreme": 1}


def test_source_inventory_counts_exact_decision_date_range(tmp_path) -> None:
    batches = tmp_path / "batches"
    batches.mkdir()
    (batches / "year_2026.json").write_text(
        json.dumps(
            [
                {"ecli": "ECLI:CZ:US:2020:OLD", "full_text": "text", "decision_date": "30. 7. 2020"},
                {"ecli": "ECLI:CZ:US:2020:IN", "full_text": "text", "decision_date": "2020-07-31"},
                {"ecli": "ECLI:CZ:US:2026:IN", "full_text": "text", "decision_date": "31. 7. 2026"},
                {"ecli": "ECLI:CZ:US:2026:NEW", "full_text": "text", "decision_date": "2026-08-01"},
                {"ecli": "ECLI:CZ:US:MISSING", "full_text": "text"},
            ]
        ),
        encoding="utf-8",
    )
    nsoud = tmp_path / "nsoud.jsonl"
    nsoud.write_text("", encoding="utf-8")

    report = build_source_inventory(
        batches_dir=batches,
        nsoud_chunks_path=nsoud,
        decision_date_range=DecisionDateRange(
            date_from=parse_decision_date("2020-07-31"),
            date_to=parse_decision_date("2026-07-31"),
        ),
    )
    source = report.to_dict()["sources"][0]["decision_date_range"]

    assert source["document_count_in_range"] == 2
    assert source["document_count_out_of_range"] == 2
    assert source["document_count_missing_or_invalid_decision_date"] == 1
