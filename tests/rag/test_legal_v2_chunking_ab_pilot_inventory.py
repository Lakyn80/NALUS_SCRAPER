"""Unit tests for chunking A/B pilot inventory builder (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.legal_v2.build_chunking_ab_pilot_300_inventory import (
    TARGET_COUNT,
    build_inventory,
)


def _write_identity(path: Path, mapping: dict[str, str]) -> None:
    docs = [
        {
            "source_document_id": doc_id,
            "ecli": ecli,
            "identity_status": "verified",
            "canonical_document_id": ecli,
        }
        for doc_id, ecli in mapping.items()
    ]
    path.write_text(json.dumps({"documents": docs}), encoding="utf-8")


def _write_golden(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_pilot(path: Path, eclis: list[str]) -> None:
    docs = []
    for index, ecli in enumerate(eclis):
        docs.append(
            {
                "ecli": ecli,
                "canonical_document_id": ecli,
                "document_type": "Usnesení" if index % 2 == 0 else "Nález",
                "chunk_count": 5 + (index % 60),
                "case_number": f"X.{index}",
                "court": "test",
                "decision_date": "2024-01-01",
            }
        )
    path.write_text(json.dumps({"documents": docs}), encoding="utf-8")


def test_build_inventory_includes_mandatory_and_is_deterministic(tmp_path: Path) -> None:
    primary = "ECLI:CZ:US:2024:1.US.1.1"
    hn = "ECLI:CZ:US:2024:2.US.2.2"
    alt = "ECLI:CZ:US:2024:3.US.3.3"
    pool = [primary, hn, alt] + [f"ECLI:CZ:US:2024:9.US.{i}.1" for i in range(1, 320)]
    golden = tmp_path / "golden.jsonl"
    identity = tmp_path / "identity.json"
    pilot = tmp_path / "pilot.json"
    _write_golden(
        golden,
        [
            {
                "benchmark_id": "q1",
                "expected_primary_ecli": primary,
                "accepted_alternative_document_ids": ["doc-alt"],
                "hard_negative_document_ids": ["doc-hn"],
            }
        ],
    )
    _write_identity(identity, {"doc-alt": alt, "doc-hn": hn})
    _write_pilot(pilot, pool)

    first = build_inventory(
        golden_path=golden,
        identity_path=identity,
        pilot_inventory_path=pilot,
        target_count=TARGET_COUNT,
        seed=20260809,
    )
    second = build_inventory(
        golden_path=golden,
        identity_path=identity,
        pilot_inventory_path=pilot,
        target_count=TARGET_COUNT,
        seed=20260809,
    )
    assert first["document_count"] == TARGET_COUNT
    assert first["inventory_hash_sha256"] == second["inventory_hash_sha256"]
    assert first["ordered_eclis"] == second["ordered_eclis"]
    assert primary in first["ordered_eclis"]
    assert hn in first["ordered_eclis"]
    assert alt in first["ordered_eclis"]
    assert first["golden_query_evaluability"]["evaluable_count"] == 1


def test_build_inventory_aborts_when_primary_missing(tmp_path: Path) -> None:
    pool = [f"ECLI:CZ:US:2024:9.US.{i}.1" for i in range(1, 320)]
    golden = tmp_path / "golden.jsonl"
    identity = tmp_path / "identity.json"
    pilot = tmp_path / "pilot.json"
    _write_golden(
        golden,
        [{"benchmark_id": "q1", "expected_primary_ecli": "ECLI:CZ:US:1999:MISSING.1"}],
    )
    _write_identity(identity, {})
    _write_pilot(pilot, pool)
    with pytest.raises(RuntimeError, match="missing from pilot inventory"):
        build_inventory(
            golden_path=golden,
            identity_path=identity,
            pilot_inventory_path=pilot,
            target_count=TARGET_COUNT,
            seed=1,
        )
