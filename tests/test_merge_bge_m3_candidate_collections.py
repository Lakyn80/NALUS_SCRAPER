from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import scripts.build_bm25_sidecar_from_qdrant as bm25_export
import scripts.merge_bge_m3_candidate_collections as merge


def _payload(document_id: str, chunk_index: int, *, decision_date: str, text: str) -> dict:
    return {
        "document_id": document_id,
        "source_document_id": document_id,
        "chunk_index": chunk_index,
        "decision_date": decision_date,
        "text": text,
        "embedding_model": "BAAI/bge-m3",
        "embedding_dimension": 1024,
        "chunk_id": chunk_index + 1,
        "qdrant_collection": "source",
        "retrieval_profile": "nalus_bge_m3_dense_bm25_rrf_v1",
        "bm25_index_id": "nalus_bge_m3_dense_bm25_rrf_v1",
    }


def _source_point(
    collection: str,
    document_id: str,
    chunk_index: int,
    *,
    decision_date: str,
    text: str,
    priority: int,
) -> merge.SourcePoint:
    payload = _payload(document_id, chunk_index, decision_date=decision_date, text=text)
    return merge.SourcePoint(
        source_collection=collection,
        point_id=f"{collection}:{document_id}:{chunk_index}",
        vector=[0.1] * 1024,
        payload=payload,
        source_priority=priority,
    )


def test_merge_prefers_later_source_on_duplicate() -> None:
    points = [
        _source_point("mvp_5y", "ECLI:CZ:US:2026:1", 0, decision_date="1. 1. 2026", text="old", priority=0),
        _source_point(
            "mvp_recent_3h",
            "ECLI:CZ:US:2026:1",
            0,
            decision_date="1. 1. 2026",
            text="new",
            priority=1,
        ),
        _source_point("mvp_5y", "ECLI:CZ:US:2025:2", 0, decision_date="1. 1. 2025", text="only-5y", priority=0),
    ]

    merged, stats = merge.merge_source_points(points, target_collection="nalus_us_bge_m3_rag_combined_20260709")

    assert stats["deduplicated_point_count"] == 2
    assert merged[0].payload["text"] == "new"
    assert merged[0].payload["chunk_id"] == 1
    assert merged[0].payload["qdrant_collection"] == "nalus_us_bge_m3_rag_combined_20260709"
    assert merged[0].point_id == merge.point_id_for("nalus_us_bge_m3_rag_combined_20260709", "ECLI:CZ:US:2026:1", 0)
    assert merged[1].payload["text"] == "only-5y"
    assert merged[1].payload["chunk_id"] == 2


def test_merge_orders_newest_decision_date_first() -> None:
    points = [
        _source_point("mvp_5y", "ECLI:CZ:US:2024:1", 0, decision_date="1. 1. 2024", text="a", priority=0),
        _source_point("mvp_recent_3h", "ECLI:CZ:US:2026:2", 0, decision_date="9. 7. 2026", text="b", priority=1),
    ]

    merged, _ = merge.merge_source_points(points, target_collection="nalus_us_bge_m3_rag_combined_20260709")

    assert merged[0].payload["document_id"] == "ECLI:CZ:US:2026:2"
    assert merged[1].payload["document_id"] == "ECLI:CZ:US:2024:1"


def test_refuses_protected_target_collection() -> None:
    with pytest.raises(merge.SafetyError, match="protected"):
        merge.validate_collection_name("nalus_live", is_target=True)


def test_bm25_sqlite_schema_is_loadable(tmp_path: Path) -> None:
    rows = [
        _payload("ECLI:CZ:US:2026:1", 0, decision_date="1. 1. 2026", text="spravedlivý proces"),
        _payload("ECLI:CZ:US:2025:2", 1, decision_date="1. 1. 2025", text="odůvodnění rozhodnutí"),
    ]
    for index, row in enumerate(rows, start=1):
        row["chunk_id"] = index

    sqlite_path = tmp_path / "bm25.sqlite"
    count = bm25_export.write_bm25_sqlite(rows, sqlite_path)
    assert count == 2

    with sqlite3.connect(sqlite_path) as connection:
        table_names = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "bm25_chunks" in table_names
        loaded = connection.execute("SELECT chunk_id, text FROM bm25_chunks ORDER BY chunk_id").fetchall()

    assert loaded == [("1", "spravedlivý proces"), ("2", "odůvodnění rozhodnutí")]

    from app.rag.retrieval.bm25_sidecar import Bm25Sidecar

    sidecar = Bm25Sidecar(
        sqlite_path,
        k1=1.5,
        b=0.75,
        index_id="nalus_bge_m3_dense_bm25_rrf_v1",
    )
    sidecar.assert_ready()
    hits = sidecar.search("spravedlivý proces", top_k=1)
    assert hits[0].id == "1"
