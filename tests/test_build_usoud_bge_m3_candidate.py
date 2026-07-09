from __future__ import annotations

import json
from datetime import date
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
        "recreate_pilot_collection": False,
        "recreate_full_collection": False,
        "resume_full_collection": False,
        "no_alias_update": True,
        "top_k_smoke_test": 5,
        "qdrant_url": "http://qdrant:6333",
        "chunk_size": 1400,
        "embedding_batch_size": 8,
        "full_record_batch_size": 50,
        "years_back": None,
        "decision_date_from": None,
        "decision_date_to": None,
        "ingest_slice": "",
        "append_full_slice": False,
        "newest_first": False,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_refuses_production_collection_names() -> None:
    for name in ("nalus", "nalus_live", "nalus_stable_20260326", "nalus_stable_future"):
        with pytest.raises(builder.SafetyError):
            builder.validate_collection_name(name, execute=True)


def test_full_mode_refuses_production_collection_names() -> None:
    for name in ("nalus_live", "nalus_stable_20260326"):
        with pytest.raises(builder.SafetyError):
            builder.validate_collection_name(name, execute=True, mode="full")


def test_execute_requires_smoke_collection_marker() -> None:
    with pytest.raises(builder.SafetyError, match="smoke, tmp, pilot"):
        builder.validate_collection_name("nalus_us_bge_m3_full_20260708", execute=True, mode="smoke")


def test_full_mode_requires_full_collection_marker() -> None:
    with pytest.raises(builder.SafetyError, match="full, tmp, mvp"):
        builder.validate_collection_name("nalus_us_bge_m3_smoke_20260708", execute=True, mode="full")

    builder.validate_collection_name("nalus_us_bge_m3_full_20260708", execute=True, mode="full")
    builder.validate_collection_name("nalus_us_bge_m3_mvp_5y_20260708", execute=True, mode="full")


def test_years_back_filter_keeps_only_recent_decisions() -> None:
    old = _sample_record("ECLI:CZ:US:2020:1")
    old = builder.SourceRecord(
        identity=old.identity,
        source_document_id=old.source_document_id,
        case_reference=old.case_reference,
        ecli=old.ecli,
        decision_date="1. 1. 2020",
        detail_url=old.detail_url,
        text_url=old.text_url,
        full_text=old.full_text,
        origin_file=old.origin_file,
        raw=old.raw,
    )
    recent = _sample_record("ECLI:CZ:US:2026:1")
    date_filter = builder.DecisionDateFilter(
        date_from=builder._utc_today().replace(year=builder._utc_today().year - 5),
        date_to=builder._utc_today(),
        years_back=5,
        ingest_slice="mvp_5y",
    )
    filtered, stats = builder.filter_records_by_decision_date([old, recent], date_filter)

    assert recent in filtered
    assert old not in filtered
    assert stats["date_out_of_range_record_count"] == 1


def test_parse_decision_date_supports_czech_and_iso_formats() -> None:
    assert builder.parse_decision_date("1. 1. 2026") == date(2026, 1, 1)
    assert builder.parse_decision_date("2026-01-15") == date(2026, 1, 15)
    assert builder.parse_decision_date("invalid") is None


def test_years_back_is_allowed_only_in_full_mode() -> None:
    args = _args(mode="smoke", years_back=5)
    with pytest.raises(builder.SafetyError, match="only in full mode"):
        builder.validate_decision_date_args(args)


def test_smoke_limit_above_100_is_refused() -> None:
    with pytest.raises(builder.SafetyError, match="above 100"):
        builder.validate_smoke_limit(101)


def test_pilot_limit_above_1000_is_refused() -> None:
    with pytest.raises(builder.SafetyError, match="above 1000"):
        builder.validate_pilot_limit(1001)


def test_full_mode_accepts_limit_zero_for_entire_corpus() -> None:
    builder.validate_full_limit(0)
    builder.validate_full_limit(1000)


def test_full_mode_rejects_negative_limit() -> None:
    with pytest.raises(builder.SafetyError, match="0 \\(all deduplicated records\\)"):
        builder.validate_full_limit(-1)


def test_pilot_mode_accepts_guarded_500_decision_run() -> None:
    args = _args(
        mode="pilot",
        limit=500,
        collection_name="nalus_us_bge_m3_pilot_20260708",
        recreate_pilot_collection=True,
    )

    builder.validate_args(args)


def test_full_mode_requires_manifest_not_batch() -> None:
    args = _args(
        mode="full",
        limit=0,
        collection_name="nalus_us_bge_m3_full_20260708",
        source_batch=Path("batches/year_2026_20260708_124949.json"),
        source_manifest=None,
    )

    with pytest.raises(builder.SafetyError, match="--source-manifest"):
        builder.validate_args(args)


def test_full_mode_accepts_guarded_manifest_run() -> None:
    args = _args(
        mode="full",
        limit=0,
        collection_name="nalus_us_bge_m3_full_20260708",
        source_batch=None,
        source_manifest=Path("batches/manifest.json"),
        recreate_full_collection=True,
        execute=True,
        dry_run=False,
    )

    builder.validate_args(args)


def test_pilot_mode_requires_pilot_or_tmp_collection_name() -> None:
    args = _args(
        mode="pilot",
        limit=500,
        collection_name="nalus_us_bge_m3_smoke_20260708",
        execute=True,
        dry_run=False,
    )

    with pytest.raises(builder.SafetyError, match="pilot, tmp"):
        builder.validate_args(args)


def test_recreate_collection_requires_smoke_or_tmp_name() -> None:
    args = _args(collection_name="nalus_us_bge_m3_pilot_test", recreate_smoke_collection=True)
    with pytest.raises(builder.SafetyError, match="smoke.*tmp"):
        builder.validate_args(args)


def test_recreate_full_collection_requires_full_or_tmp_name() -> None:
    args = _args(
        mode="full",
        limit=0,
        collection_name="nalus_us_bge_m3_pilot_20260708",
        source_batch=None,
        source_manifest=Path("batches/manifest.json"),
        recreate_full_collection=True,
    )
    with pytest.raises(builder.SafetyError, match="full.*tmp"):
        builder.validate_args(args)


def test_no_alias_update_guard_refuses_disabled_guard() -> None:
    args = _args(no_alias_update=False)
    with pytest.raises(builder.SafetyError, match="Alias updates"):
        builder.validate_args(args)


def test_vector_dimension_guard_requires_1024() -> None:
    builder.validate_vector_dimension([[0.0] * 1024])
    with pytest.raises(builder.SafetyError, match="expected 1024, got 3"):
        builder.validate_vector_dimension([[0.1, 0.2, 0.3]])


def test_compare_production_safety_detects_alias_changes() -> None:
    before = {
        "aliases": [{"alias_name": "nalus_live", "collection_name": "nalus_stable_20260326"}],
        "nalus_live_point_count": 784812,
        "nalus_stable_20260326_point_count": 784812,
        "nalus_live_target": "nalus_stable_20260326",
    }
    after = {
        "aliases": [{"alias_name": "nalus_live", "collection_name": "nalus_stable_future"}],
        "nalus_live_point_count": 784812,
        "nalus_stable_20260326_point_count": 784812,
        "nalus_live_target": "nalus_stable_future",
    }

    delta = builder.compare_production_safety(before, after)

    assert delta["aliases_changed"] is True
    assert delta["production_touched"] is True
    assert delta["nalus_live_changed"] is False
    assert delta["nalus_stable_changed"] is False


def test_compare_production_safety_detects_point_count_changes() -> None:
    before = {
        "aliases": [{"alias_name": "nalus_live", "collection_name": "nalus_stable_20260326"}],
        "nalus_live_point_count": 784812,
        "nalus_stable_20260326_point_count": 784812,
        "nalus_live_target": "nalus_stable_20260326",
    }
    after = {
        "aliases": before["aliases"],
        "nalus_live_point_count": 784813,
        "nalus_stable_20260326_point_count": 784812,
        "nalus_live_target": "nalus_stable_20260326",
    }

    delta = builder.compare_production_safety(before, after)

    assert delta["nalus_live_changed"] is True
    assert delta["production_touched"] is True
    assert delta["aliases_changed"] is False


def test_report_path_for_full_mode() -> None:
    args = _args(mode="full", collection_name="nalus_us_bge_m3_full_20260708")

    assert builder._report_path(args) == builder.STAGE3_FULL_REPORT_PATH


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


def test_dry_run_reads_qdrant_state_but_does_not_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    class Alias:
        alias_name = "nalus_live"
        collection_name = "nalus_stable_20260326"

    class Aliases:
        aliases = [Alias()]

    class FakeClient:
        def get_aliases(self):
            return Aliases()

        def count(self, collection_name: str):  # noqa: ANN001
            class Count:
                count = 784812 if collection_name in {"nalus_live", "nalus_stable_20260326"} else 0

            return Count()

    def fake_qdrant(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        return FakeClient()

    def fail_write(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise AssertionError("dry-run must not prepare collections or write points")

    monkeypatch.setattr(builder, "_qdrant_client", fake_qdrant)
    monkeypatch.setattr(builder, "_prepare_candidate_collection", fail_write)
    monkeypatch.setattr(builder, "_upsert_chunks", fail_write)
    monkeypatch.setattr(builder, "REPORT_PATH", tmp_path / "report.md")
    args = _args(source_batch=source, output_dir=tmp_path / "out")

    summary = builder.run_dry_run(args)

    assert summary["selected_record_count"] == 1
    assert summary["generated_chunk_count"] >= 1
    assert summary["qdrant_write_occurred"] is False
    assert summary["final_status"] == "PASS"
    assert summary["qdrant"]["nalus_live_before"] == 784812
    assert summary["qdrant"]["nalus_live_target_before"] == "nalus_stable_20260326"
    assert (tmp_path / "out" / "dry_run_summary.json").exists()


def test_count_chunks_for_records_matches_chunk_records() -> None:
    record = builder.SourceRecord(
        identity="ECLI:CZ:US:2026:1",
        source_document_id="ECLI:CZ:US:2026:1",
        case_reference="I.ÚS 1/26",
        ecli="ECLI:CZ:US:2026:1",
        decision_date="1. 1. 2026",
        detail_url=None,
        text_url=None,
        full_text="Ústavní soud rozhodl o právu na spravedlivý proces. " * 40,
        origin_file="a.json",
        raw={},
    )
    records = [record]

    counted = builder.count_chunks_for_records(records, chunk_size=120, overlap_words=5)
    chunks = builder.chunk_records(records, collection_name="nalus_us_bge_m3_full_test", chunk_size=120)

    assert counted == len(chunks)


def test_full_dry_run_writes_production_snapshot_and_report_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "batch.json"
    source.write_text(
        json.dumps(
            [
                {
                    "result_id": 1,
                    "case_reference": "I.ÚS 1/26",
                    "ecli": "ECLI:CZ:US:2026:1",
                    "decision_date": "1. 1. 2026",
                    "full_text": "Ústavní soud rozhodl o právu na spravedlivý proces. " * 20,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"batches": [{"file": source.name}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    class Alias:
        alias_name = "nalus_live"
        collection_name = "nalus_stable_20260326"

    class Aliases:
        aliases = [Alias()]

    class FakeClient:
        def get_aliases(self):
            return Aliases()

        def count(self, collection_name: str):  # noqa: ANN001
            class Count:
                count = 784812 if collection_name in {"nalus_live", "nalus_stable_20260326"} else 0

            return Count()

    monkeypatch.setattr(builder, "_qdrant_client", lambda *args, **kwargs: FakeClient())
    monkeypatch.setattr(builder, "STAGE3_FULL_REPORT_PATH", tmp_path / "stage3_report.md")
    args = _args(
        mode="full",
        limit=0,
        collection_name="nalus_us_bge_m3_full_test",
        source_batch=None,
        source_manifest=manifest,
        output_dir=tmp_path / "full_out",
    )

    summary = builder.run_dry_run(args)

    assert summary["final_status"] == "PASS"
    assert summary["source_file_count"] == 1
    assert (tmp_path / "full_out" / "dry_run_summary.json").exists()
    assert (tmp_path / "full_out" / "production_safety_snapshot_before.json").exists()
    assert (tmp_path / "stage3_report.md").exists()


def test_recreate_and_resume_flags_conflict() -> None:
    args = _args(
        mode="full",
        limit=0,
        collection_name="nalus_us_bge_m3_full_20260708",
        source_batch=None,
        source_manifest=Path("batches/manifest.json"),
        execute=True,
        dry_run=False,
        recreate_full_collection=True,
        resume_full_collection=True,
    )

    with pytest.raises(builder.SafetyError, match="cannot be used together"):
        builder.validate_args(args)


def test_resolve_full_execute_resume_starts_fresh_checkpoint(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"batches": []}), encoding="utf-8")
    args = _args(
        mode="full",
        limit=0,
        collection_name="nalus_us_bge_m3_full_test",
        source_batch=None,
        source_manifest=manifest,
        output_dir=tmp_path / "out",
        execute=True,
        dry_run=False,
    )
    selected: list[builder.SourceRecord] = []
    date_filter = builder.resolve_decision_date_filter(args)

    state = builder.resolve_full_execute_resume(
        args,
        selected=selected,
        date_filter=date_filter,
        collection_before=None,
        expected_chunk_count=0,
    )

    assert state["resume_mode"] is False
    assert state["next_record_index"] == 0
    assert (tmp_path / "out" / builder.EXECUTE_CHECKPOINT_FILENAME).exists()


def test_resolve_full_execute_resume_continues_from_checkpoint(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"batches": []}), encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    args = _args(
        mode="full",
        limit=0,
        collection_name="nalus_us_bge_m3_full_test",
        source_batch=None,
        source_manifest=manifest,
        output_dir=output_dir,
        execute=True,
        dry_run=False,
    )
    date_filter = builder.resolve_decision_date_filter(args)
    checkpoint = {
        "status": "in_progress",
        "builder_version": builder.BUILDER_VERSION,
        "collection_name": "nalus_us_bge_m3_full_test",
        "source_manifest": str(manifest),
        "limit": 0,
        "chunk_size": 1400,
        "full_record_batch_size": 50,
        "embedding_batch_size": 8,
        "ingest_slice": date_filter.ingest_slice,
        "decision_date_filter": date_filter.as_summary(),
        "total_records": 10,
        "expected_chunk_count": 100,
        "collection_point_count_at_start": 0,
        "next_record_index": 5,
        "next_seq_id": 42,
        "inserted_point_count": 41,
        "last_record_identity": "ECLI:CZ:US:2026:1",
        "updated_at": "2026-07-08T18:00:00Z",
    }
    (output_dir / builder.EXECUTE_CHECKPOINT_FILENAME).write_text(
        json.dumps(checkpoint),
        encoding="utf-8",
    )

    state = builder.resolve_full_execute_resume(
        args,
        selected=[_sample_record(f"ECLI:CZ:US:2026:{index}") for index in range(10)],
        date_filter=date_filter,
        collection_before=41,
        expected_chunk_count=100,
    )

    assert state["resume_mode"] is True
    assert state["next_record_index"] == 5
    assert state["next_seq_id"] == 42
    assert state["inserted_point_count"] == 41


def test_resolve_full_execute_resume_refuses_orphan_collection(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"batches": []}), encoding="utf-8")
    args = _args(
        mode="full",
        limit=0,
        collection_name="nalus_us_bge_m3_full_test",
        source_batch=None,
        source_manifest=manifest,
        output_dir=tmp_path / "out",
        execute=True,
        dry_run=False,
    )

    with pytest.raises(builder.SafetyError, match="execute_checkpoint.json"):
        builder.resolve_full_execute_resume(
            args,
            selected=[],
            date_filter=builder.resolve_decision_date_filter(args),
            collection_before=1200,
            expected_chunk_count=0,
        )


def test_order_records_by_decision_date_newest_first() -> None:
    old = builder.SourceRecord(
        identity="ECLI:CZ:US:2021:1",
        source_document_id="ECLI:CZ:US:2021:1",
        case_reference=None,
        ecli="ECLI:CZ:US:2021:1",
        decision_date="1. 1. 2021",
        detail_url=None,
        text_url=None,
        full_text="text",
        origin_file="a.json",
        raw={},
    )
    mid = builder.SourceRecord(
        identity="ECLI:CZ:US:2024:1",
        source_document_id="ECLI:CZ:US:2024:1",
        case_reference=None,
        ecli="ECLI:CZ:US:2024:1",
        decision_date="2024-06-15",
        detail_url=None,
        text_url=None,
        full_text="text",
        origin_file="b.json",
        raw={},
    )
    recent = builder.SourceRecord(
        identity="ECLI:CZ:US:2026:1",
        source_document_id="ECLI:CZ:US:2026:1",
        case_reference=None,
        ecli="ECLI:CZ:US:2026:1",
        decision_date="9. 7. 2026",
        detail_url=None,
        text_url=None,
        full_text="text",
        origin_file="c.json",
        raw={},
    )

    ordered = builder.order_records_by_decision_date([old, recent, mid], newest_first=True)

    assert [record.identity for record in ordered] == [
        "ECLI:CZ:US:2026:1",
        "ECLI:CZ:US:2024:1",
        "ECLI:CZ:US:2021:1",
    ]


def test_newest_first_requires_positive_limit() -> None:
    args = _args(
        mode="full",
        limit=0,
        collection_name="nalus_us_bge_m3_mvp_recent_3h_20260709",
        source_batch=None,
        source_manifest=Path("batches/manifest.json"),
        newest_first=True,
    )

    with pytest.raises(builder.SafetyError, match="positive --limit"):
        builder.validate_newest_first_args(args)


def test_newest_first_is_allowed_only_in_full_mode() -> None:
    args = _args(mode="smoke", newest_first=True, limit=100)
    with pytest.raises(builder.SafetyError, match="only in full mode"):
        builder.validate_decision_date_args(args)


def _sample_record(identity: str) -> builder.SourceRecord:
    return builder.SourceRecord(
        identity=identity,
        source_document_id=identity,
        case_reference="I.ÚS 1/26",
        ecli=identity,
        decision_date="1. 1. 2026",
        detail_url=None,
        text_url=None,
        full_text="Ústavní soud rozhodl o právu na spravedlivý proces.",
        origin_file="a.json",
        raw={},
    )
