from __future__ import annotations

import argparse
import contextlib
import io
import warnings
from dataclasses import dataclass
from typing import Any

from app.nsoud.generate_embeddings import DEFAULT_MODEL_NAME, build_embedder, resolve_device


@dataclass(frozen=True)
class SearchResultRow:
    rank: int
    score: float
    case_number: str
    ecli: str
    decision_date: str
    document_type: str
    legal_area: str
    ns_section_hint: str
    chunk_id: str
    url: str
    text_preview: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local semantic search against the NSoud Qdrant test collection.")
    parser.add_argument("--query", required=True, help="Czech legal query to embed and search.")
    parser.add_argument("--collection", required=True, help="Qdrant collection name.")
    parser.add_argument("--qdrant-url", required=True, help="Qdrant base URL.")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to return.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Sentence-transformers model name.")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="auto",
        help="Embedding device selection.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def text_preview(text: str, limit: int = 700) -> str:
    normalized = " ".join(normalize_text(text).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


def detect_collection_vector_size(info: Any) -> int:
    vectors = getattr(info.config.params, "vectors", None)
    size = getattr(vectors, "size", None)
    return int(size or 0)


def verify_collection(client: Any, collection_name: str) -> int:
    if not client.collection_exists(collection_name):
        raise RuntimeError(f"Collection '{collection_name}' does not exist.")
    info = client.get_collection(collection_name)
    return detect_collection_vector_size(info)


def run_search(client: Any, *, collection_name: str, vector: list[float], limit: int) -> list[Any]:
    response = client.query_points(
        collection_name=collection_name,
        query=vector,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return list(response.points)


def map_result(rank: int, point: Any) -> SearchResultRow:
    payload = dict(point.payload or {})
    return SearchResultRow(
        rank=rank,
        score=float(point.score),
        case_number=normalize_text(payload.get("case_number")),
        ecli=normalize_text(payload.get("ecli")),
        decision_date=normalize_text(payload.get("decision_date")),
        document_type=normalize_text(payload.get("document_type")),
        legal_area=normalize_text(payload.get("legal_area")),
        ns_section_hint=normalize_text(payload.get("ns_section_hint")),
        chunk_id=normalize_text(payload.get("chunk_id")),
        url=normalize_text(payload.get("url")),
        text_preview=text_preview(normalize_text(payload.get("text"))),
    )


def print_results_table(results: list[SearchResultRow]) -> None:
    print("top results:")
    print("rank | score | case_number | document_type | legal_area | ns_section_hint | chunk_id")
    for result in results:
        print(
            f"{result.rank} | {result.score:.6f} | {result.case_number or '-'} | "
            f"{result.document_type or '-'} | {result.legal_area or '-'} | "
            f"{result.ns_section_hint or '-'} | {result.chunk_id or '-'}"
        )


def print_result_details(results: list[SearchResultRow]) -> None:
    for result in results:
        print("")
        print(f"[{result.rank}] score={result.score:.6f}")
        print(f"case_number: {result.case_number or '-'}")
        print(f"ecli: {result.ecli or '-'}")
        print(f"decision_date: {result.decision_date or '-'}")
        print(f"document_type: {result.document_type or '-'}")
        print(f"legal_area: {result.legal_area or '-'}")
        print(f"ns_section_hint: {result.ns_section_hint or '-'}")
        print(f"chunk_id: {result.chunk_id or '-'}")
        print(f"url: {result.url or '-'}")
        print(f"text preview: {result.text_preview or '-'}")


def main() -> int:
    args = parse_args()

    query = args.query.strip()
    if not query:
        print("search status: FAIL")
        print("error: --query must not be empty.")
        return 1

    if args.limit <= 0:
        print("search status: FAIL")
        print("error: --limit must be greater than 0.")
        return 1

    try:
        from qdrant_client import QdrantClient

        resolved_device = resolve_device(args.device)
        warnings.filterwarnings(
            "ignore",
            message=r"Qdrant client version .* is incompatible with server version .*",
        )
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            embedder = build_embedder(args.model_name, batch_size=1, device=resolved_device)
        query_vector = embedder.embed_query(query)

        client = QdrantClient(url=args.qdrant_url, timeout=30)
        collection_vector_size = verify_collection(client, args.collection)
        if len(query_vector) != collection_vector_size:
            raise RuntimeError(
                f"Embedding dimension mismatch: query vector has {len(query_vector)}, "
                f"collection expects {collection_vector_size}."
            )

        raw_results = run_search(
            client,
            collection_name=args.collection,
            vector=query_vector,
            limit=args.limit,
        )
        if not raw_results:
            raise RuntimeError("Search returned zero results.")

        results = [map_result(rank, point) for rank, point in enumerate(raw_results, start=1)]
    except Exception as exc:
        print("search status: FAIL")
        print(f"query: {query}")
        print(f"collection: {args.collection}")
        print("result count: 0")
        print(f"error: {exc}")
        return 1

    print("search status: PASS")
    print(f"query: {query}")
    print(f"collection: {args.collection}")
    print(f"result count: {len(results)}")
    print_results_table(results)
    print_result_details(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
