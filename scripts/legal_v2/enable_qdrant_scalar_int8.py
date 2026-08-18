"""Offline ops helper to enable Qdrant scalar INT8 on an existing collection.

This script is NOT part of the API runtime. It never runs at startup.
Dry-run is the default. Mutation requires --apply and a matching
--confirm-collection.

Compatible with Qdrant 1.13.6: uses ScalarType.INT8 + always_ram.
Does not use Qdrant 1.19+ memory tiers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or apply scalar INT8 quantization on an existing Qdrant collection. "
            "Does not re-embed. Dry-run unless --apply is set."
        )
    )
    parser.add_argument("--url", default="http://127.0.0.1:6333")
    parser.add_argument(
        "--collection",
        required=True,
        help="Exact collection name to inspect or update.",
    )
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument(
        "--always-ram",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Qdrant 1.13 always_ram flag (default: true).",
    )
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually call update_collection. Off by default.",
    )
    parser.add_argument(
        "--confirm-collection",
        default="",
        help="Must equal --collection when --apply is set.",
    )
    parser.add_argument(
        "--wait-green-seconds",
        type=int,
        default=0,
        help="After --apply, poll collection status until green or timeout.",
    )
    return parser.parse_args(argv)


def planned_quantization_config(*, quantile: float, always_ram: bool) -> dict[str, Any]:
    if not 0.5 <= quantile <= 1.0:
        raise ValueError("--quantile must be in [0.5, 1.0] for Qdrant scalar quantization.")
    return {
        "scalar": {
            "type": "int8",
            "quantile": quantile,
            "always_ram": bool(always_ram),
        }
    }


def build_quantization_config(*, quantile: float, always_ram: bool) -> Any:
    from qdrant_client import models

    return models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            quantile=quantile,
            always_ram=always_ram,
        )
    )


def inspect_collection(client: Any, collection_name: str) -> dict[str, Any]:
    info = client.get_collection(collection_name)
    config = getattr(info, "config", None)
    quantization = getattr(config, "quantization_config", None) if config is not None else None
    return {
        "collection": collection_name,
        "status": str(getattr(info, "status", "")),
        "points_count": getattr(info, "points_count", None),
        "indexed_vectors_count": getattr(info, "indexed_vectors_count", None),
        "optimizer_status": str(getattr(info, "optimizer_status", "")),
        "quantization_config": str(quantization),
    }


def apply_quantization(
    client: Any,
    *,
    collection_name: str,
    quantile: float,
    always_ram: bool,
) -> None:
    client.update_collection(
        collection_name=collection_name,
        quantization_config=build_quantization_config(
            quantile=quantile,
            always_ram=always_ram,
        ),
    )


def wait_until_green(client: Any, collection_name: str, *, timeout_seconds: int) -> dict[str, Any]:
    import time

    deadline = time.perf_counter() + timeout_seconds
    last = inspect_collection(client, collection_name)
    while time.perf_counter() < deadline:
        last = inspect_collection(client, collection_name)
        if str(last.get("status", "")).lower() == "green":
            return last
        time.sleep(2)
    raise RuntimeError(
        f"Collection {collection_name!r} did not become green within {timeout_seconds}s; "
        f"last_status={last.get('status')}"
    )


def main(argv: list[str] | None = None, *, client: Any = None) -> int:
    args = parse_args(argv)
    plan = planned_quantization_config(quantile=args.quantile, always_ram=args.always_ram)
    payload = {
        "mode": "apply" if args.apply else "dry-run",
        "url": args.url,
        "collection": args.collection,
        "planned_quantization_config": plan,
        "qdrant_api": "update_collection + ScalarQuantization INT8 + always_ram (1.13 compatible)",
    }
    if not args.apply:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.confirm_collection != args.collection:
        raise ValueError("--apply requires --confirm-collection to match --collection exactly.")

    from qdrant_client import QdrantClient

    qdrant = client or QdrantClient(url=args.url, timeout=args.timeout)
    before = inspect_collection(qdrant, args.collection)
    payload["before"] = before
    apply_quantization(
        qdrant,
        collection_name=args.collection,
        quantile=args.quantile,
        always_ram=args.always_ram,
    )
    after = inspect_collection(qdrant, args.collection)
    if args.wait_green_seconds > 0:
        after = wait_until_green(qdrant, args.collection, timeout_seconds=args.wait_green_seconds)
    payload["after"] = after
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
