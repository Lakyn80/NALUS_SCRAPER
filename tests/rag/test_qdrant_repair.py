from __future__ import annotations

from types import SimpleNamespace

from app.rag.ingest.qdrant_ingest import POINT_ID_SCHEME, point_id_from_original_id
from app.rag.ingest.qdrant_repair import inspect_duplicates, repair_collection


def _make_point(
    point_id: str,
    original_id: str,
    *,
    text: str = "Text rozhodnutí.",
    with_scheme: bool = False,
) -> SimpleNamespace:
    payload = {
        "original_id": original_id,
        "text": text,
        "case_reference": "III.ÚS 255/26",
        "ecli": None,
        "decision_date": "2026-01-15",
        "judge": "Jan Novák",
        "text_url": None,
        "chunk_index": 0,
        "source": "nalus",
    }
    if with_scheme:
        payload["point_id_scheme"] = POINT_ID_SCHEME
    return SimpleNamespace(id=point_id, payload=payload, vector=[0.1, 0.2, 0.3])


class _FakeClient:
    def __init__(self, source_points: list[SimpleNamespace]) -> None:
        self._source_points = list(source_points)
        self._dest_points: dict[str, dict[str, SimpleNamespace]] = {}

    def get_collection(self, collection_name: str):
        del collection_name
        params = SimpleNamespace(
            vectors=SimpleNamespace(size=3, distance="Cosine"),
            sparse_vectors=None,
            shard_number=None,
            sharding_method=None,
            replication_factor=None,
            write_consistency_factor=None,
            on_disk_payload=True,
        )
        config = SimpleNamespace(
            params=params,
            hnsw_config=None,
            optimizer_config=None,
            wal_config=None,
            quantization_config=None,
            strict_mode_config=None,
        )
        return SimpleNamespace(config=config)

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self._dest_points

    def create_collection(self, collection_name: str, **kwargs) -> None:
        del kwargs
        self._dest_points[collection_name] = {}

    def delete_collection(self, collection_name: str) -> None:
        self._dest_points.pop(collection_name, None)

    def scroll(
        self,
        *,
        collection_name: str,
        limit: int,
        offset=None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ):
        del with_payload, with_vectors
        points = self._source_points if collection_name == "source" else list(
            self._dest_points.get(collection_name, {}).values()
        )
        start = int(offset or 0)
        end = start + limit
        next_offset = end if end < len(points) else None
        return points[start:end], next_offset

    def upsert(self, *, collection_name: str, points: list[SimpleNamespace]) -> None:
        bucket = self._dest_points.setdefault(collection_name, {})
        for point in points:
            bucket[str(point.id)] = SimpleNamespace(
                id=str(point.id),
                payload=point.payload,
                vector=point.vector,
            )

    def count(self, *, collection_name: str):
        if collection_name == "source":
            return SimpleNamespace(count=len(self._source_points))
        return SimpleNamespace(count=len(self._dest_points.get(collection_name, {})))


def test_inspect_duplicates_reports_duplicate_groups() -> None:
    client = _FakeClient(
        [
            _make_point("1", "chunk-1"),
            _make_point("2", "chunk-1"),
            _make_point("3", "chunk-2"),
        ]
    )

    report = inspect_duplicates(client, "source", batch_size=2)

    assert report.total_points == 3
    assert report.unique_original_ids == 2
    assert report.duplicate_groups == 1
    assert report.extra_duplicate_points == 1


def test_repair_collection_writes_unique_deterministic_points() -> None:
    deterministic_id = point_id_from_original_id("chunk-1")
    client = _FakeClient(
        [
            _make_point("100", "chunk-1", text="legacy duplicate"),
            _make_point(deterministic_id, "chunk-1", text="canonical", with_scheme=True),
            _make_point("200", "chunk-2", text="second chunk"),
        ]
    )

    stats = repair_collection(
        client,
        source_collection="source",
        destination_collection="repaired",
        batch_size=2,
        recreate_destination=False,
    )

    repaired_points = client._dest_points["repaired"]
    assert stats.total_points_scanned == 3
    assert stats.unique_chunks_written == 2
    assert stats.duplicate_groups == 1
    assert stats.extra_duplicate_points_removed == 1
    assert set(repaired_points.keys()) == {
        point_id_from_original_id("chunk-1"),
        point_id_from_original_id("chunk-2"),
    }
    assert repaired_points[point_id_from_original_id("chunk-1")].payload["text"] == "canonical"
    assert repaired_points[point_id_from_original_id("chunk-1")].payload["point_id_scheme"] == POINT_ID_SCHEME
