from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.rag.retrieval.bm25_sidecar import Bm25Sidecar
from scripts.build_bm25_sidecar_from_qdrant import write_bm25_sqlite
from scripts.repair_nsoud_bm25_sidecar_provenance import (
    DEFAULT_COLLECTION,
    RepairError,
    inspect_sidecar,
    run_repair,
)


def _create_legacy_sidecar(path: Path, rows: list[dict[str, object]]) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE bm25_chunks (
                chunk_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                document_id TEXT,
                source_document_id TEXT,
                decision_date TEXT,
                chunk_index INTEGER,
                qdrant_collection TEXT,
                retrieval_profile TEXT,
                bm25_index_id TEXT
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO bm25_chunks (
                chunk_id,
                text,
                document_id,
                source_document_id,
                decision_date,
                chunk_index,
                qdrant_collection,
                retrieval_profile,
                bm25_index_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(row["chunk_id"]),
                    str(row["text"]),
                    str(row.get("document_id") or ""),
                    str(row.get("source_document_id") or ""),
                    str(row.get("decision_date") or ""),
                    int(row.get("chunk_index", -1)),
                    str(row.get("qdrant_collection") or ""),
                    str(row.get("retrieval_profile") or ""),
                    str(row.get("bm25_index_id") or ""),
                )
                for row in rows
            ],
        )
        connection.commit()


class _FakePoint:
    def __init__(self, payload: dict[str, object], point_id: str) -> None:
        self.payload = payload
        self.id = point_id


class _FakeQdrantClient:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self._payloads = payloads

    def scroll(
        self,
        *,
        collection_name: str,
        limit: int,
        offset: int | None,
        with_payload: bool,
        with_vectors: bool,
    ) -> tuple[list[_FakePoint], int | None]:
        del collection_name, with_payload, with_vectors
        start = int(offset or 0)
        end = start + limit
        batch = [
            _FakePoint(payload, point_id=f"point-{index}")
            for index, payload in enumerate(self._payloads[start:end], start=start)
        ]
        next_offset = end if end < len(self._payloads) else None
        return batch, next_offset


def _payload(
    chunk_id: str,
    *,
    document_id: str,
    source_document_id: str | None = None,
    case_number: str | None = None,
    source: str = "nsoud",
    text: str,
    chunk_index: int,
) -> dict[str, object]:
    return {
        "chunk_id": int(chunk_id),
        "text": text,
        "document_id": document_id,
        "source": source,
        "chunk_index": chunk_index,
        "chunk_metadata": {
            "source_document_id": source_document_id or document_id,
            "case_number": case_number or "",
        },
        "qdrant_collection": DEFAULT_COLLECTION,
        "retrieval_profile": "nalus_bge_m3_dense_bm25_rrf_v1",
        "bm25_index_id": "nalus_bge_m3_dense_bm25_rrf_v1",
    }


def test_dry_run_does_not_modify_original_sidecar(tmp_path: Path) -> None:
    sidecar = tmp_path / "legacy.sqlite"
    _create_legacy_sidecar(
        sidecar,
        [
            {
                "chunk_id": "735",
                "text": "dovolací důvod podle § 265b tr. ř.",
                "chunk_index": -1,
            }
        ],
    )
    before_bytes = sidecar.read_bytes()
    args = SimpleNamespace(
        sidecar_path=sidecar,
        collection_name=DEFAULT_COLLECTION,
        qdrant_url="http://qdrant:6333",
        output_sidecar_path=tmp_path / "repaired.sqlite",
        backup=False,
        dry_run=True,
        execute=False,
    )

    summary = run_repair(
        args,
        client=_FakeQdrantClient(
            [
                _payload(
                    "735",
                    document_id="ECLI:CZ:NS:2025:5.TDO.1086.2024.1",
                    case_number="5 Tdo 1086/2024",
                    text="dovolací důvod podle § 265b tr. ř.",
                    chunk_index=9,
                )
            ]
        ),
    )

    assert summary["mode"] == "dry_run"
    assert summary["audit_after"] is None
    assert sidecar.read_bytes() == before_bytes
    assert not args.output_sidecar_path.exists()


def test_execute_preserves_row_count_and_enriches_provenance(tmp_path: Path) -> None:
    sidecar = tmp_path / "legacy.sqlite"
    repaired = tmp_path / "repaired.sqlite"
    _create_legacy_sidecar(
        sidecar,
        [
            {
                "chunk_id": "735",
                "text": "dovolací důvod podle § 265b tr. ř.",
                "chunk_index": -1,
            },
            {
                "chunk_id": "991",
                "text": "dovolání lze podat podle § 265b tr. ř.",
                "chunk_index": -1,
            },
        ],
    )
    args = SimpleNamespace(
        sidecar_path=sidecar,
        collection_name=DEFAULT_COLLECTION,
        qdrant_url="http://qdrant:6333",
        output_sidecar_path=repaired,
        backup=False,
        dry_run=False,
        execute=True,
    )

    summary = run_repair(
        args,
        client=_FakeQdrantClient(
            [
                _payload(
                    "735",
                    document_id="ECLI:CZ:NS:2025:5.TDO.1086.2024.1",
                    case_number="5 Tdo 1086/2024",
                    text="dovolací důvod podle § 265b tr. ř.",
                    chunk_index=9,
                ),
                _payload(
                    "991",
                    document_id="ECLI:CZ:NS:2025:4.TDO.1137.2024.1",
                    case_number="4 Tdo 1137/2024",
                    text="dovolání lze podat podle § 265b tr. ř.",
                    chunk_index=23,
                ),
            ]
        ),
    )

    assert summary["audit_before"]["row_count"] == 2
    assert summary["audit_after"]["row_count"] == 2
    after = inspect_sidecar(repaired)
    assert after.blank_or_null_counts["document_id"] == 0
    assert after.blank_or_null_counts["source_document_id"] == 0
    assert after.blank_or_null_counts["ecli"] == 0
    assert after.blank_or_null_counts["case_number"] == 0
    assert after.blank_or_null_counts["source"] == 0


def test_ambiguous_qdrant_mapping_fails_clearly(tmp_path: Path) -> None:
    sidecar = tmp_path / "legacy.sqlite"
    _create_legacy_sidecar(
        sidecar,
        [{"chunk_id": "735", "text": "dovolací důvod", "chunk_index": -1}],
    )
    args = SimpleNamespace(
        sidecar_path=sidecar,
        collection_name=DEFAULT_COLLECTION,
        qdrant_url="http://qdrant:6333",
        output_sidecar_path=tmp_path / "repaired.sqlite",
        backup=False,
        dry_run=True,
        execute=False,
    )

    with pytest.raises(RepairError, match="Ambiguous Qdrant mapping"):
        run_repair(
            args,
            client=_FakeQdrantClient(
                [
                    _payload(
                        "735",
                        document_id="ECLI:CZ:NS:2025:5.TDO.1086.2024.1",
                        text="dovolací důvod",
                        chunk_index=9,
                    ),
                    _payload(
                        "735",
                        document_id="ECLI:CZ:NS:2025:5.TDO.1086.2024.1",
                        text="dovolací důvod",
                        chunk_index=9,
                    ),
                ]
            ),
        )


def test_bm25_metadata_and_scores_are_preserved_after_repair(tmp_path: Path) -> None:
    sidecar = tmp_path / "legacy.sqlite"
    repaired = tmp_path / "repaired.sqlite"
    rows = [
        {"chunk_id": "735", "text": "dovolací důvod podle 265b", "chunk_index": -1},
        {"chunk_id": "991", "text": "jiné dovolání podle 265b", "chunk_index": -1},
    ]
    _create_legacy_sidecar(sidecar, rows)
    args = SimpleNamespace(
        sidecar_path=sidecar,
        collection_name=DEFAULT_COLLECTION,
        qdrant_url="http://qdrant:6333",
        output_sidecar_path=repaired,
        backup=False,
        dry_run=False,
        execute=True,
    )
    client = _FakeQdrantClient(
        [
            _payload(
                "735",
                document_id="ECLI:CZ:NS:2025:5.TDO.1086.2024.1",
                source_document_id="ECLI:CZ:NS:2025:5.TDO.1086.2024.1",
                case_number="5 Tdo 1086/2024",
                text="DIFFERENT QDRANT TEXT SHOULD NOT REPLACE SIDECAR",
                chunk_index=9,
            ),
            _payload(
                "991",
                document_id="ECLI:CZ:NS:2025:4.TDO.1137.2024.1",
                source_document_id="ECLI:CZ:NS:2025:4.TDO.1137.2024.1",
                case_number="4 Tdo 1137/2024",
                text="ANOTHER DIFFERENT TEXT",
                chunk_index=23,
            ),
        ]
    )
    run_repair(args, client=client)

    original_sidecar = Bm25Sidecar(
        sidecar,
        k1=1.5,
        b=0.75,
        index_id="nalus_bge_m3_dense_bm25_rrf_v1",
    )
    repaired_sidecar = Bm25Sidecar(
        repaired,
        k1=1.5,
        b=0.75,
        index_id="nalus_bge_m3_dense_bm25_rrf_v1",
    )
    original_hits = original_sidecar.search("dovolací důvod 265b", top_k=2)
    repaired_hits = repaired_sidecar.search("dovolací důvod 265b", top_k=2)

    assert [hit.id for hit in original_hits] == [hit.id for hit in repaired_hits]
    assert [hit.score for hit in original_hits] == pytest.approx([hit.score for hit in repaired_hits])
    assert repaired_hits[0].metadata["document_id"] == "ECLI:CZ:NS:2025:5.TDO.1086.2024.1"
    assert repaired_hits[0].metadata["source_document_id"] == "ECLI:CZ:NS:2025:5.TDO.1086.2024.1"
    assert repaired_hits[0].metadata["ecli"] == "ECLI:CZ:NS:2025:5.TDO.1086.2024.1"
    assert repaired_hits[0].metadata["case_reference"] == "5 Tdo 1086/2024"
    assert repaired_hits[0].text == "dovolací důvod podle 265b"


def test_repaired_sidecar_supports_ecli_lookup_for_nsoud_qa_007(tmp_path: Path) -> None:
    repaired = tmp_path / "repaired.sqlite"
    write_bm25_sqlite(
        [
            {
                "chunk_id": "735",
                "text": "dovolací důvod podle § 265b tr. ř.",
                "document_id": "ECLI:CZ:NS:2025:5.TDO.1086.2024.1",
                "source_document_id": "ECLI:CZ:NS:2025:5.TDO.1086.2024.1",
                "chunk_index": 9,
                "source": "nsoud",
                "chunk_metadata": {"case_number": "5 Tdo 1086/2024"},
            }
        ],
        repaired,
    )

    with sqlite3.connect(repaired) as connection:
        count = connection.execute(
            """
            SELECT COUNT(*)
            FROM bm25_chunks
            WHERE document_id = ? OR source_document_id = ? OR ecli = ?
            """,
            (
                "ECLI:CZ:NS:2025:5.TDO.1086.2024.1",
                "ECLI:CZ:NS:2025:5.TDO.1086.2024.1",
                "ECLI:CZ:NS:2025:5.TDO.1086.2024.1",
            ),
        ).fetchone()[0]

    assert count == 1
