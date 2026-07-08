from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

import scripts.build_usoud_bge_m3_candidate as builder


def _args(**overrides):
    defaults = {
        "mode": "smoke",
        "limit": 20,
        "collection_name": "nalus_us_bge_m3_smoke_test",
        "source_batch": Path("batches/year_2026_20260708_124949.json"),
        "source_manifest": None,
        "output_dir": Path("artifacts/nalus_update/test_smoke"),
        "dry_run": True,
        "execute": False,
        "recreate_smoke_collection": False,
        "no_alias_update": True,
        "top_k_smoke_test": 5,
        "qdrant_url": "http://qdrant:6333",
        "chunk_size": 1400,
        "embedding_batch_size": 8,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_refuses_production_collection_names() -> None:
    for name in ("nalus", "nalus_live", "nalus_stable_20260326", "nalus_stable_future"):
        with pytest.raises(builder.SafetyError):
            builder.validate_collection_name(name, execute=True)


def test_execute_requires_stage1_collection_marker() -> None:
    with pytest.raises(builder.SafetyError, match="smoke, tmp, pilot"):
        builder.validate_collection_name("nalus_us_bge_m3_full_20260708", execute=True)


def test_smoke_limit_above_100_is_refused() -> None:
    with pytest.raises(builder.SafetyError, match="above 100"):
        builder.validate_smoke_limit(101)


def test_recreate_collection_requires_smoke_or_tmp_name() -> None:
    args = _args(collection_name="nalus_us_bge_m3_pilot_test", recreate_smoke_collection=True)
    with pytest.raises(builder.SafetyError, match="smoke.*tmp"):
        builder.validate_args(args)


def test_no_alias_update_guard_refuses_disabled_guard() -> None:
    args = _args(no_alias_update=False)
    with pytest.raises(builder.SafetyError, match="Alias updates"):
        builder.validate_args(args)


def test_vector_dimension_guard_requires_1024() -> None:
    builder.validate_vector_dimension([[0.0] * 1024])
    with pytest.raises(builder.SafetyError, match="expected 1024, got 3"):
        builder.validate_vector_dimension([[0.1, 0.2, 0.3]])


def test_deduplicates_by_stable_identity_preferring_fuller_text() -> None:
    short = builder.SourceRecord(
        identity="ECLI:CZ:US:2026:1",
        source_document_id="ECLI:CZ:US:2026:1",
        case_reference="I.ÚS 1/26",
        ecli="ECLI:CZ:US:2026:1",
        decision_date="1. 1. 2026",
        detail_url=None,
        text_url=None,
        full_text="kratky text",
        origin_file="a.json",
        raw={},
    )
    long = builder.SourceRecord(
        identity=short.identity,
        source_document_id=short.source_document_id,
        case_reference=short.case_reference,
        ecli=short.ecli,
        decision_date=short.decision_date,
        detail_url="https://example.test/detail",
        text_url=None,
        full_text="delší text s větším obsahem",
        origin_file="b.json",
        raw={},
    )

    assert builder.deduplicate_records([short, long]) == [long]


def test_chunking_is_deterministic_and_does_not_split_words() -> None:
    text = "První odstavec obsahuje několik slov.\n\n" + " ".join(f"slovo{i}" for i in range(80))
    first = builder.split_text_into_chunks(text, chunk_size=120, overlap_words=5)
    second = builder.split_text_into_chunks(text, chunk_size=120, overlap_words=5)

    assert first == second
    assert len(first) > 1
    assert all(not chunk.startswith(" ") and not chunk.endswith(" ") for chunk in first)
    assert all(" " in chunk or "\n" in chunk for chunk in first)


def test_dry_run_does_not_create_qdrant_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "batch.json"
    source.write_text(
        json.dumps(
            [
                {
                    "result_id": 1,
                    "case_reference": "I.ÚS 1/26",
                    "ecli": "ECLI:CZ:US:2026:1",
                    "decision_date": "1. 1. 2026",
                    "detail_url": "https://example.test/detail",
                    "text_url": "https://example.test/text",
                    "full_text": "Ústavní soud rozhodl o právu na spravedlivý proces. " * 20,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def fail_qdrant(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise AssertionError("dry-run must not create a Qdrant client")

    monkeypatch.setattr(builder, "_qdrant_client", fail_qdrant)
    monkeypatch.setattr(builder, "REPORT_PATH", tmp_path / "report.md")
    args = _args(source_batch=source, output_dir=tmp_path / "out")

    summary = builder.run_dry_run(args)

    assert summary["selected_record_count"] == 1
    assert summary["generated_chunk_count"] >= 1
    assert (tmp_path / "out" / "dry_run_summary.json").exists()
