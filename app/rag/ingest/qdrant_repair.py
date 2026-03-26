from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from app.rag.ingest.qdrant_ingest import POINT_ID_SCHEME, payload_checksum, point_id_from_original_id


@dataclass(frozen=True)
class DuplicateReport:
    collection_name: str
    total_points: int
    unique_original_ids: int
    duplicate_groups: int
    extra_duplicate_points: int


@dataclass(frozen=True)
class RepairStats:
    source_collection: str
    destination_collection: str
    total_points_scanned: int
    unique_chunks_written: int
    duplicate_groups: int
    extra_duplicate_points_removed: int


@dataclass(frozen=True)
class _RepairCandidate:
    original_id: str
    current_point_id: str
    destination_point_id: str
    payload: dict[str, Any]
    vector: Any


def inspect_duplicates(
    client: Any,
    collection_name: str,
    *,
    batch_size: int = 256,
) -> DuplicateReport:
    counts = Counter[str]()
    total_points = 0

    for point in _scroll_points(
        client,
        collection_name,
        batch_size=batch_size,
        with_vectors=False,
    ):
        total_points += 1
        counts[_logical_original_id(point)] += 1

    duplicate_groups = sum(1 for count in counts.values() if count > 1)
    unique_original_ids = len(counts)
    extra_duplicate_points = total_points - unique_original_ids
    return DuplicateReport(
        collection_name=collection_name,
        total_points=total_points,
        unique_original_ids=unique_original_ids,
        duplicate_groups=duplicate_groups,
        extra_duplicate_points=extra_duplicate_points,
    )


def repair_collection(
    client: Any,
    *,
    source_collection: str,
    destination_collection: str,
    batch_size: int = 256,
    recreate_destination: bool = False,
) -> RepairStats:
    if source_collection == destination_collection:
        raise ValueError("source_collection and destination_collection must be different.")

    _create_destination_like(
        client,
        source_collection=source_collection,
        destination_collection=destination_collection,
        recreate_destination=recreate_destination,
    )

    counts = Counter[str]()
    best_candidates: dict[str, _RepairCandidate] = {}
    upsert_buffer: list[Any] = []

    from qdrant_client.models import PointStruct

    total_points_scanned = 0
    for point in _scroll_points(
        client,
        source_collection,
        batch_size=batch_size,
        with_vectors=True,
    ):
        total_points_scanned += 1
        candidate = _candidate_from_point(point)
        counts[candidate.original_id] += 1

        current_best = best_candidates.get(candidate.original_id)
        if current_best is not None and not _candidate_is_better(candidate, current_best):
            continue

        best_candidates[candidate.original_id] = candidate
        upsert_buffer.append(
            PointStruct(
                id=candidate.destination_point_id,
                vector=candidate.vector,
                payload=candidate.payload,
            )
        )
        if len(upsert_buffer) >= batch_size:
            client.upsert(collection_name=destination_collection, points=upsert_buffer)
            upsert_buffer.clear()

    if upsert_buffer:
        client.upsert(collection_name=destination_collection, points=upsert_buffer)

    duplicate_groups = sum(1 for count in counts.values() if count > 1)
    unique_chunks_written = len(best_candidates)
    extra_duplicate_points_removed = total_points_scanned - unique_chunks_written

    destination_count = client.count(collection_name=destination_collection).count
    if destination_count != unique_chunks_written:
        raise RuntimeError(
            "Destination collection size mismatch after repair: "
            f"expected {unique_chunks_written}, got {destination_count}."
        )

    return RepairStats(
        source_collection=source_collection,
        destination_collection=destination_collection,
        total_points_scanned=total_points_scanned,
        unique_chunks_written=unique_chunks_written,
        duplicate_groups=duplicate_groups,
        extra_duplicate_points_removed=extra_duplicate_points_removed,
    )


def update_alias(client: Any, *, alias_name: str, target_collection: str) -> None:
    from qdrant_client.models import CreateAlias, CreateAliasOperation, DeleteAlias, DeleteAliasOperation

    operations = []
    aliases = client.get_aliases().aliases
    if any(alias.alias_name == alias_name for alias in aliases):
        operations.append(
            DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=alias_name))
        )
    operations.append(
        CreateAliasOperation(
            create_alias=CreateAlias(
                collection_name=target_collection,
                alias_name=alias_name,
            )
        )
    )
    client.update_collection_aliases(change_aliases_operations=operations)


def _create_destination_like(
    client: Any,
    *,
    source_collection: str,
    destination_collection: str,
    recreate_destination: bool,
) -> None:
    source_info = client.get_collection(source_collection)
    if client.collection_exists(destination_collection):
        if not recreate_destination:
            raise RuntimeError(
                f"Destination collection '{destination_collection}' already exists. "
                "Use recreate_destination=True to replace it."
            )
        client.delete_collection(destination_collection)

    client.create_collection(
        collection_name=destination_collection,
        vectors_config=_config_value(source_info.config.params.vectors),
        sparse_vectors_config=_config_value(
            getattr(source_info.config.params, "sparse_vectors", None)
        ),
        shard_number=getattr(source_info.config.params, "shard_number", None),
        sharding_method=_config_value(getattr(source_info.config.params, "sharding_method", None)),
        replication_factor=getattr(source_info.config.params, "replication_factor", None),
        write_consistency_factor=getattr(
            source_info.config.params,
            "write_consistency_factor",
            None,
        ),
        on_disk_payload=getattr(source_info.config.params, "on_disk_payload", None),
        hnsw_config=_config_value(getattr(source_info.config, "hnsw_config", None)),
        optimizers_config=_config_value(getattr(source_info.config, "optimizer_config", None)),
        wal_config=_config_value(getattr(source_info.config, "wal_config", None)),
        quantization_config=_config_value(getattr(source_info.config, "quantization_config", None)),
        strict_mode_config=_config_value(getattr(source_info.config, "strict_mode_config", None)),
    )


def _scroll_points(
    client: Any,
    collection_name: str,
    *,
    batch_size: int,
    with_vectors: bool,
):
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
        )
        if not points:
            return
        for point in points:
            yield point
        if offset is None:
            return


def _candidate_from_point(point: Any) -> _RepairCandidate:
    original_id = _logical_original_id(point)
    payload = _normalized_payload(point.payload or {}, original_id=original_id)
    return _RepairCandidate(
        original_id=original_id,
        current_point_id=str(point.id),
        destination_point_id=point_id_from_original_id(original_id),
        payload=payload,
        vector=getattr(point, "vector", None),
    )


def _logical_original_id(point: Any) -> str:
    payload = point.payload or {}
    return str(payload.get("original_id") or point.id)


def _normalized_payload(payload: dict[str, Any], *, original_id: str) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["original_id"] = original_id
    normalized.setdefault("source", "nalus")
    normalized.setdefault("text", "")
    normalized.setdefault("case_reference", None)
    normalized.setdefault("ecli", None)
    normalized.setdefault("decision_date", None)
    normalized.setdefault("judge", None)
    normalized.setdefault("text_url", None)
    normalized.setdefault("chunk_index", None)
    normalized["point_id_scheme"] = POINT_ID_SCHEME
    normalized["content_checksum"] = payload_checksum(normalized)
    return normalized


def _candidate_is_better(candidate: _RepairCandidate, current: _RepairCandidate) -> bool:
    candidate_rank = _candidate_rank(candidate)
    current_rank = _candidate_rank(current)
    if candidate_rank != current_rank:
        return candidate_rank > current_rank
    return candidate.current_point_id < current.current_point_id


def _candidate_rank(candidate: _RepairCandidate) -> tuple[int, int, int, int]:
    payload = candidate.payload
    deterministic_id_match = int(candidate.current_point_id == candidate.destination_point_id)
    completeness = _payload_completeness(payload)
    text_length = len(payload.get("text") or "")
    vector_present = int(candidate.vector is not None)
    return (
        deterministic_id_match,
        completeness,
        text_length,
        vector_present,
    )


def _payload_completeness(payload: dict[str, Any]) -> int:
    keys = (
        "text",
        "case_reference",
        "ecli",
        "decision_date",
        "judge",
        "text_url",
        "chunk_index",
    )
    score = 0
    for key in keys:
        value = payload.get(key)
        if _has_meaningful_value(value):
            score += 1
    return score


def _has_meaningful_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def _config_value(value: Any) -> Any:
    if value is None:
        return None

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(exclude_none=True)

    to_dict = getattr(value, "dict", None)
    if callable(to_dict):
        return to_dict(exclude_none=True)

    return value
