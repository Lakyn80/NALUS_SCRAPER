"""Unit tests for court staging identity, completeness, and path guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.court_staging.completeness import MonthCompleteness, finalize_month_status
from app.court_staging.identity import ChangeKind, classify_content_change, enrich_record_identity, resolve_canonical_id
from app.court_staging.jsonl_store import load_canonical_index, rewrite_jsonl_upsert
from app.court_staging.paths import assert_safe_staging_path, ensure_staging_tree


def test_canonical_id_prefers_ecli():
    cid, klass = resolve_canonical_id(
        {"ecli": "ecli:cz:ns:2024:123", "case_number": "21 Cdo 1/2024", "source": "nsoud"}
    )
    assert cid.startswith("ECLI:CZ:NS:2024:")
    assert klass == "ecli"


def test_content_hash_change_is_updated_not_new(tmp_path: Path):
    path = tmp_path / "docs.jsonl"
    known: dict[str, str] = {}
    r1 = enrich_record_identity(
        {
            "source": "nsoud",
            "ecli": "ECLI:CZ:NS:2024:1",
            "url": "https://example.test/1",
            "full_text": "text A",
        },
        source="nsoud",
    )
    assert rewrite_jsonl_upsert(path, r1, known=known) is ChangeKind.NEW
    r2 = dict(r1)
    r2["full_text"] = "text B"
    assert rewrite_jsonl_upsert(path, r2, known=known) is ChangeKind.UPDATED
    assert rewrite_jsonl_upsert(path, r2, known=known) is ChangeKind.UNCHANGED
    rows = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1
    assert "text B" in rows[0]


def test_classify_content_change():
    known = {"ECLI:CZ:NS:2024:1": "aaa"}
    assert classify_content_change(canonical_id="ECLI:CZ:NS:2024:2", content_hash="x", known=known) is ChangeKind.NEW
    assert (
        classify_content_change(canonical_id="ECLI:CZ:NS:2024:1", content_hash="aaa", known=known)
        is ChangeKind.UNCHANGED
    )
    assert (
        classify_content_change(canonical_id="ECLI:CZ:NS:2024:1", content_hash="bbb", known=known)
        is ChangeKind.UPDATED
    )


def test_completeness_ok_requires_full_accounting():
    stats = MonthCompleteness(
        site_total_results=10,
        discovered_entries=10,
        unique_source_ids=10,
        fetched_ok=9,
        failed=0,
        skipped_classified=0,
    )
    finalize_month_status(stats)
    assert stats.status == "partial"

    stats2 = MonthCompleteness(
        site_total_results=10,
        discovered_entries=10,
        unique_source_ids=10,
        fetched_ok=8,
        failed=2,
        skipped_classified=0,
        failure_reasons={"detail_fetch_or_parse": 2},
    )
    finalize_month_status(stats2)
    assert stats2.status == "partial"

    stats3 = MonthCompleteness(
        site_total_results=10,
        discovered_entries=10,
        unique_source_ids=10,
        fetched_ok=10,
        failed=0,
        skipped_classified=0,
    )
    finalize_month_status(stats3)
    assert stats3.status == "ok"


def test_path_guard_rejects_batches(tmp_path: Path):
    staging = ensure_staging_tree(tmp_path / "court_staging")
    ok = assert_safe_staging_path(staging / "ns" / "historical" / "x.jsonl", staging_root=staging)
    assert ok.exists() or True
    batches = tmp_path / "batches" / "year_2024.json"
    batches.parent.mkdir(parents=True)
    batches.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        assert_safe_staging_path(batches, staging_root=staging)


def test_load_canonical_index_roundtrip(tmp_path: Path):
    path = tmp_path / "a.jsonl"
    known: dict[str, str] = {}
    rewrite_jsonl_upsert(
        path,
        {
            "source": "nssoud",
            "ecli": "ECLI:CZ:NSS:2020:1",
            "url": "https://example.test/nss/1",
            "full_text": "rozhodnutí",
        },
        known=known,
        source="nssoud",
    )
    loaded = load_canonical_index([path])
    assert "ECLI:CZ:NSS:2020:1" in loaded
