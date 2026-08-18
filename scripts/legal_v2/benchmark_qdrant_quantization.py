"""Read-only Qdrant search benchmark: full precision vs quantized search params.

Does not mutate Qdrant. Uses an existing stored vector so BGE-M3 is not required.
Never logs query text.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.retrieval.qdrant_quantization import QdrantQuantizationSearchPolicy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Qdrant query_points latency with ignore=True vs ignore=False."
    )
    parser.add_argument("--url", default="http://127.0.0.1:6333")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument("--oversampling", type=float, default=1.0)
    return parser.parse_args(argv)


def _first_vector(client: Any, collection_name: str) -> list[float]:
    points, _offset = client.scroll(
        collection_name=collection_name,
        limit=1,
        with_payload=False,
        with_vectors=True,
    )
    if not points:
        raise RuntimeError(f"Collection {collection_name!r} has no points to sample a vector from.")
    vector = getattr(points[0], "vector", None)
    if isinstance(vector, dict):
        vector = next(iter(vector.values()), None)
    if not isinstance(vector, list) or not vector:
        raise RuntimeError("Sampled point does not contain a dense vector.")
    return [float(value) for value in vector]


def _time_search(
    client: Any,
    *,
    collection_name: str,
    vector: list[float],
    limit: int,
    policy: QdrantQuantizationSearchPolicy,
    repeats: int,
) -> dict[str, Any]:
    search_params = policy.to_search_params()
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
            search_params=search_params,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        samples.append(elapsed_ms)
        if result.points is None:
            raise RuntimeError("query_points returned no points payload.")
    ordered = sorted(samples)
    return {
        "quantization_enabled": policy.enabled,
        "quantization_ignore": policy.ignore,
        "quantization_rescore": policy.rescore if policy.enabled else False,
        "quantization_oversampling": policy.oversampling if policy.enabled else 1.0,
        "repeats": repeats,
        "limit": limit,
        "vector_dimension": len(vector),
        "latency_ms_min": ordered[0],
        "latency_ms_median": statistics.median(ordered),
        "latency_ms_p90": ordered[max(0, int(round(0.9 * (len(ordered) - 1))))],
        "latency_ms_max": ordered[-1],
    }


def main(argv: list[str] | None = None, *, client: Any = None) -> int:
    args = parse_args(argv)
    if args.limit <= 0 or args.repeats <= 0:
        raise ValueError("--limit and --repeats must be positive.")
    if args.oversampling < 1.0:
        raise ValueError("--oversampling must be >= 1.0.")

    from qdrant_client import QdrantClient

    qdrant = client or QdrantClient(url=args.url, timeout=args.timeout)
    vector = _first_vector(qdrant, args.collection)
    off_policy = QdrantQuantizationSearchPolicy(enabled=False, rescore=False, oversampling=1.0)
    on_policy = QdrantQuantizationSearchPolicy(
        enabled=True,
        rescore=bool(args.rescore),
        oversampling=float(args.oversampling),
    )
    report = {
        "collection": args.collection,
        "url": args.url,
        "full_precision": _time_search(
            qdrant,
            collection_name=args.collection,
            vector=vector,
            limit=args.limit,
            policy=off_policy,
            repeats=args.repeats,
        ),
        "quantized_search": _time_search(
            qdrant,
            collection_name=args.collection,
            vector=vector,
            limit=args.limit,
            policy=on_policy,
            repeats=args.repeats,
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
