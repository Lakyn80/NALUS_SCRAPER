from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.indexing import LEGAL_V2_BM25_INDEX_ID, LEGAL_V2_COLLECTION_NAME  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture Legal v2 smoke-index safety snapshot.")
    parser.add_argument("--phase", choices=("prebuild", "postbuild"), required=True)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/legal_v2/smoke_index_20260730")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from qdrant_client import QdrantClient  # type: ignore[import-not-found]

    client = QdrantClient(url=args.qdrant_url, timeout=30)
    payload = {
        "schema": "legal_v2_smoke_safety_snapshot_v1",
        "phase": args.phase,
        "generated_at": _utc_now(),
        "qdrant_url": args.qdrant_url,
        "legal_v2_collection": LEGAL_V2_COLLECTION_NAME,
        "legal_v2_bm25_index_id": LEGAL_V2_BM25_INDEX_ID,
        "aliases": _aliases(client),
        "collections": _collections(client),
        "bm25_sidecars": _bm25_sidecars(PROJECT_ROOT / "storage/rag/bm25"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.phase}_snapshot.json"
    md_path = args.output_dir / f"{args.phase}_snapshot.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")
    print(json_path)
    print(md_path)
    return 0


def _aliases(client: Any) -> list[dict[str, Any]]:
    try:
        aliases = client.get_aliases().aliases
    except Exception:  # noqa: BLE001 - Qdrant versions differ in alias support.
        return []
    return [
        {"alias_name": str(alias.alias_name), "collection_name": str(alias.collection_name)}
        for alias in aliases
    ]


def _collections(client: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in client.get_collections().collections:
        name = str(item.name)
        try:
            info = client.get_collection(collection_name=name)
            vector_size = _vector_size(info)
        except Exception:  # noqa: BLE001 - snapshot should continue with bounded diagnostics.
            vector_size = None
        try:
            point_count = int(client.count(collection_name=name, exact=True).count)
        except Exception:  # noqa: BLE001
            point_count = None
        result.append(
            {
                "name": name,
                "point_count": point_count,
                "vector_size": vector_size,
                "is_legal_v2": name == LEGAL_V2_COLLECTION_NAME,
            }
        )
    return sorted(result, key=lambda item: item["name"])


def _vector_size(info: Any) -> int | None:
    params = getattr(getattr(info, "config", None), "params", None)
    vectors = getattr(params, "vectors", None)
    size = getattr(vectors, "size", None)
    if size is not None:
        return int(size)
    if isinstance(vectors, dict):
        first = next(iter(vectors.values()), None)
        if first is not None and getattr(first, "size", None) is not None:
            return int(first.size)
    return None


def _bm25_sidecars(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    files = sorted(item for item in path.glob("*.sqlite") if item.is_file())
    result = []
    for file_path in files:
        result.append(
            {
                "path": str(file_path),
                "size_bytes": file_path.stat().st_size,
                "sha256": _sha256(file_path),
                "row_count": _row_count(file_path),
                "is_legal_v2": LEGAL_V2_BM25_INDEX_ID in file_path.name,
            }
        )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_count(path: Path) -> int | None:
    try:
        with sqlite3.connect(path) as connection:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for table in ("bm25_chunks", "chunks", "rag_chunks"):
                if table in tables:
                    return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    except sqlite3.Error:
        return None
    return None


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Legal Retrieval v2 smoke safety snapshot",
        "",
        f"- Phase: `{payload['phase']}`",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Aliases: {len(payload['aliases'])}",
        f"- Collections: {len(payload['collections'])}",
        f"- BM25 sidecars: {len(payload['bm25_sidecars'])}",
        "",
        "## Collections",
        "",
    ]
    for item in payload["collections"]:
        lines.append(f"- `{item['name']}` points={item['point_count']} vector_size={item['vector_size']}")
    lines.extend(["", "## BM25 sidecars", ""])
    for item in payload["bm25_sidecars"]:
        lines.append(f"- `{item['path']}` rows={item['row_count']} size={item['size_bytes']} sha256={item['sha256']}")
    return "\n".join(lines) + "\n"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
