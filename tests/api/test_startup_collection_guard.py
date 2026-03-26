from __future__ import annotations

from types import SimpleNamespace

from app.api.startup import _collection_supports_stable_sync, _ensure_collection
from app.rag.ingest.qdrant_ingest import POINT_ID_SCHEME, point_id_from_original_id


class _FakeClient:
    def __init__(
        self,
        *,
        collections: list[str] | None = None,
        aliases: list[str] | None = None,
        points: list[SimpleNamespace] | None = None,
    ) -> None:
        self._collections = collections or []
        self._aliases = aliases or []
        self._points = points or []
        self.create_calls: list[str] = []

    def get_collections(self):
        return SimpleNamespace(
            collections=[SimpleNamespace(name=name) for name in self._collections]
        )

    def get_aliases(self):
        return SimpleNamespace(
            aliases=[SimpleNamespace(alias_name=name) for name in self._aliases]
        )

    def create_collection(self, *, collection_name: str, vectors_config) -> None:
        del vectors_config
        self.create_calls.append(collection_name)

    def count(self, *, collection_name: str):
        del collection_name
        return SimpleNamespace(count=len(self._points))

    def scroll(
        self,
        *,
        collection_name: str,
        limit: int,
        offset=None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ):
        del collection_name, with_payload, with_vectors
        start = int(offset or 0)
        end = start + limit
        next_offset = end if end < len(self._points) else None
        return self._points[start:end], next_offset


def test_ensure_collection_does_not_create_when_alias_exists() -> None:
    client = _FakeClient(aliases=["nalus_live"])

    _ensure_collection(
        client,
        "nalus_live",
        vector_dim=10,
        distance="Cosine",
        vector_params_factory=lambda **kwargs: kwargs,
    )

    assert client.create_calls == []


def test_collection_supports_stable_sync_rejects_legacy_point_ids() -> None:
    client = _FakeClient(
        points=[
            SimpleNamespace(
                id="123",
                payload={
                    "original_id": "chunk-1",
                    "point_id_scheme": POINT_ID_SCHEME,
                },
            )
        ]
    )

    assert _collection_supports_stable_sync(client, "nalus") is False


def test_collection_supports_stable_sync_accepts_deterministic_points() -> None:
    original_id = "chunk-1"
    client = _FakeClient(
        points=[
            SimpleNamespace(
                id=point_id_from_original_id(original_id),
                payload={
                    "original_id": original_id,
                    "point_id_scheme": POINT_ID_SCHEME,
                },
            )
        ]
    )

    assert _collection_supports_stable_sync(client, "nalus") is True
