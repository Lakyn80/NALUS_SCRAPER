from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from qdrant_client import QdrantClient

from app.rag.ingest.qdrant_repair import inspect_duplicates, repair_collection, update_alias


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deduplicated Qdrant collection with deterministic point IDs "
            "from a legacy NALUS collection."
        )
    )
    parser.add_argument("--url", default="http://localhost:6333")
    parser.add_argument("--source", default="nalus")
    parser.add_argument("--destination", default="")
    parser.add_argument("--alias", default="")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--recreate-destination", action="store_true")
    return parser.parse_args()


def _default_destination(source: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{source}_repaired_{timestamp}"


def main() -> int:
    args = _parse_args()
    destination = args.destination or _default_destination(args.source)
    client = QdrantClient(url=args.url, timeout=args.timeout)

    report = inspect_duplicates(
        client,
        args.source,
        batch_size=args.batch_size,
    )
    print(json.dumps(report.__dict__, ensure_ascii=False, indent=2))

    if args.report_only:
        return 0

    stats = repair_collection(
        client,
        source_collection=args.source,
        destination_collection=destination,
        batch_size=args.batch_size,
        recreate_destination=args.recreate_destination,
    )
    print(json.dumps(stats.__dict__, ensure_ascii=False, indent=2))

    if args.alias:
        update_alias(client, alias_name=args.alias, target_collection=destination)
        print(
            f"Alias '{args.alias}' now points to '{destination}'. "
            f"Set QDRANT_COLLECTION_NAME={args.alias} before restarting the API."
        )
    else:
        print(
            f"Repaired collection created: {destination}. "
            f"Set QDRANT_COLLECTION_NAME={destination} before restarting the API."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
