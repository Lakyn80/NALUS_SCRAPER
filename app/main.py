import json
import os
import re
import unicodedata
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from app.services.search_service import collect_results

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BATCHES_DIR = PROJECT_ROOT / "batches"


def _read_positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer. Received: {raw_value}") from exc

    if value < 1:
        raise SystemExit(f"{name} must be greater than or equal to 1.")

    return value


def _read_optional_positive_int(name: str) -> int | None:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return None

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer. Received: {raw_value}") from exc

    if value < 1:
        raise SystemExit(f"{name} must be greater than or equal to 1.")

    return value


def _read_non_negative_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default

    try:
        value = float(raw_value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a number. Received: {raw_value}") from exc

    if value < 0:
        raise SystemExit(f"{name} must be greater than or equal to 0.")

    return value


def _read_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise SystemExit(f"{name} must be a boolean value.")


def _slugify_query(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", normalized).strip("_").lower()
    return slug or "results"


def _default_output_path(
    query: str,
    max_results: int | None,
    page_start: int,
    page_end: int | None,
) -> Path:
    BATCHES_DIR.mkdir(exist_ok=True)
    query_slug = _slugify_query(query)
    if max_results is None and page_end == page_start:
        name = f"results_{query_slug}_p{page_start}.json"
    elif max_results is not None:
        name = f"results_{query_slug}_{max_results}.json"
    elif page_end is None:
        name = f"results_{query_slug}_bulk.json"
    else:
        name = f"results_{query_slug}_pages_{page_start}_{page_end}.json"
    return BATCHES_DIR / name


def main():
    load_dotenv()

    query = os.getenv("NALUS_QUERY", "rodinné právo").strip() or "rodinné právo"
    single_page = _read_positive_int("NALUS_PAGE", 1)
    page_start = _read_positive_int("NALUS_PAGE_START", single_page)
    max_results = _read_optional_positive_int("NALUS_MAX_RESULTS")
    explicit_page_end = _read_optional_positive_int("NALUS_PAGE_END")
    fetch_full_text = _read_bool("NALUS_FETCH_FULL_TEXT", True)
    batch_pages = _read_positive_int("NALUS_BATCH_PAGES", 10)
    batch_sleep_seconds = _read_non_negative_float("NALUS_BATCH_SLEEP_SECONDS", 2.0)

    page_end = explicit_page_end
    if page_end is None and max_results is None:
        page_end = page_start

    output_path_env = os.getenv("NALUS_OUTPUT_PATH", "").strip()
    output_path = (
        Path(output_path_env)
        if output_path_env
        else _default_output_path(query, max_results, page_start, page_end)
    )

    if page_end is not None and page_end < page_start:
        raise SystemExit("NALUS_PAGE_END must be greater than or equal to NALUS_PAGE_START.")

    results = collect_results(
        query=query,
        page_start=page_start,
        page_end=page_end,
        fetch_full_text=fetch_full_text,
        max_results=max_results,
        batch_pages=batch_pages,
        batch_sleep_seconds=batch_sleep_seconds,
    )

    print(f"QUERY: {query}")
    print(f"PAGES: {page_start}-{page_end if page_end is not None else 'auto'}")
    print(f"FULL TEXT: {fetch_full_text}")
    print(f"PARSED RESULTS: {len(results)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    print(f"Saved to {output_path}")

    auto_ingest = _read_bool("NALUS_AUTO_INGEST", True)
    if auto_ingest:
        _run_ingest(output_path, query)


def _run_ingest(batch_path: Path, query: str) -> None:
    """Ingest the freshly scraped batch into Qdrant and update the manifest."""
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "nalus")

    print(f"[INGEST] connecting to Qdrant at {qdrant_url}")

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

        from app.data.runtime_corpus import build_runtime_corpus, load_results_from_json
        from app.rag.ingest.qdrant_ingest import QdrantIngestor, _MOCK_VECTOR_DIM

        class _Adapter:
            def __init__(self, c):
                self._c = c
            def upsert(self, collection_name, points):
                self._c.upsert(collection_name=collection_name,
                               points=[PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points])
            def retrieve(self, *, collection_name, ids, with_payload=True, with_vectors=False):
                return self._c.retrieve(collection_name=collection_name, ids=ids,
                                        with_payload=with_payload, with_vectors=with_vectors)

        client = QdrantClient(url=qdrant_url, timeout=30)

        # Ensure collection exists
        existing = {c.name for c in client.get_collections().collections}
        if collection_name not in existing:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=_MOCK_VECTOR_DIM, distance=Distance.COSINE),
            )
            print(f"[INGEST] created collection '{collection_name}'")

        results = load_results_from_json(batch_path)
        corpus = build_runtime_corpus(results, source_label=str(batch_path))
        print(f"[INGEST] {len(results)} docs -> {len(corpus.chunks)} chunks")

        ingestor = QdrantIngestor(_Adapter(client), collection_name=collection_name, batch_size=200)
        stats = ingestor.ingest_chunks(corpus.chunks)
        print(
            f"[INGEST] done inserted={stats.inserted_points} "
            f"updated={stats.updated_points} skipped={stats.skipped_points}"
        )

        _update_manifest(batch_path, query, len(results), len(corpus.chunks))

    except Exception as exc:
        print(f"[INGEST] failed: {exc}")
        print("[INGEST] skipping auto-ingest — run scripts/ingest_batch.py manually")


def _update_manifest(batch_path: Path, query: str, doc_count: int, chunk_count: int) -> None:
    from datetime import datetime, timezone

    manifest_path = BATCHES_DIR / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {"version": 1, "description": "Tracking ingestovaných batchů do Qdrant kolekce 'nalus'", "batches": []}

    entry = {
        "file": batch_path.name,
        "query": query,
        "doc_count": doc_count,
        "chunk_count": chunk_count,
        "ingested_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
    }
    manifest["batches"] = [b for b in manifest["batches"] if b.get("file") != batch_path.name]
    manifest["batches"].append(entry)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[INGEST] manifest updated -> {manifest_path}")


if __name__ == "__main__":
    main()
