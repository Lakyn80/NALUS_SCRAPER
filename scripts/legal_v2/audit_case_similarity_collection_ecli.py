#!/usr/bin/env python3
"""Audit ECLI presence in a Legal v2 Qdrant collection (read-only).

Does not modify the collection. Reports indexed judgment identity health and
Case Similarity Golden v1 primary / hard-negative ECLI coverage.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.benchmark.case_similarity_golden import (  # noqa: E402
    DEFAULT_PILOT_DATASET,
    load_case_similarity_golden_jsonl,
)
from app.rag.legal_v2.identity import (  # noqa: E402
    ecli_key,
    is_valid_ecli,
    normalize_ecli,
)

DEFAULT_COLLECTION = "nalus_legal_paragraph_chunks_v2_pilot_600"
DEFAULT_OUT = (
    PROJECT_ROOT
    / "artifacts"
    / "legal_v2"
    / "case_similarity_golden_v1_pilot"
    / "collection_ecli_audit.json"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--qdrant-collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--golden", type=Path, default=DEFAULT_PILOT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    from qdrant_client import QdrantClient  # type: ignore[import-not-found]

    client = QdrantClient(url=args.qdrant_url, timeout=60)
    items = load_case_similarity_golden_jsonl(args.golden)

    chunk_count = 0
    by_document: dict[str, dict[str, Any]] = {}
    malformed: list[str] = []
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=args.qdrant_collection,
            limit=256,
            offset=next_offset,
            with_payload=[
                "document_id",
                "ecli",
                "canonical_document_id",
                "source_document_id",
                "case_reference",
            ],
            with_vectors=False,
        )
        for point in points:
            chunk_count += 1
            payload = point.payload or {}
            ecli_raw = payload.get("ecli") or payload.get("canonical_document_id") or payload.get("document_id")
            ecli = normalize_ecli(str(ecli_raw or ""))
            if ecli and is_valid_ecli(ecli):
                key = ecli_key(ecli)
                row = by_document.setdefault(
                    key,
                    {
                        "ecli": ecli,
                        "document_id": payload.get("document_id"),
                        "canonical_document_id": payload.get("canonical_document_id"),
                        "source_document_id": payload.get("source_document_id"),
                        "chunk_count": 0,
                        "document_id_equals_ecli": False,
                    },
                )
                row["chunk_count"] += 1
                doc_id = str(payload.get("document_id") or "")
                row["document_id_equals_ecli"] = bool(doc_id and ecli_key(doc_id) == key)
            else:
                malformed.append(str(ecli_raw or payload.get("document_id") or ""))
                fallback = str(payload.get("document_id") or f"unknown-{chunk_count}")
                row = by_document.setdefault(
                    f"non_ecli::{fallback}",
                    {
                        "ecli": None,
                        "document_id": payload.get("document_id"),
                        "canonical_document_id": payload.get("canonical_document_id"),
                        "source_document_id": payload.get("source_document_id"),
                        "chunk_count": 0,
                        "document_id_equals_ecli": False,
                    },
                )
                row["chunk_count"] += 1
        if next_offset is None:
            break

    judgments_with_ecli = [row for key, row in by_document.items() if not key.startswith("non_ecli::")]
    judgments_without_ecli = [row for key, row in by_document.items() if key.startswith("non_ecli::")]
    ecli_keys = {ecli_key(row["ecli"]) for row in judgments_with_ecli if row.get("ecli")}
    duplicate_eclis = [
        ecli for ecli, count in Counter(row["ecli"] for row in judgments_with_ecli).items() if count > 1
    ]

    primary_eclis = []
    hn_eclis = []
    for item in items:
        if item.expected_primary_ecli:
            primary_eclis.append(normalize_ecli(item.expected_primary_ecli))
        for row in item.hard_negative_rationales:
            if row.ecli:
                hn_eclis.append(normalize_ecli(row.ecli))

    primary_present = [ecli for ecli in primary_eclis if ecli_key(ecli) in ecli_keys]
    primary_missing = [ecli for ecli in primary_eclis if ecli_key(ecli) not in ecli_keys]
    hn_present = [ecli for ecli in hn_eclis if ecli_key(ecli) in ecli_keys]
    hn_missing = [ecli for ecli in hn_eclis if ecli_key(ecli) not in ecli_keys]
    blocked_primaries = [
        {
            "benchmark_id": item.benchmark_id,
            "source_document_id": item.source_document_id,
            "primary_identity_status": item.primary_identity_status,
        }
        for item in items
        if not item.expected_primary_ecli
    ]

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "collection": args.qdrant_collection,
        "indexed_chunks": chunk_count,
        "unique_indexed_judgments": len(by_document),
        "judgments_with_ecli": len(judgments_with_ecli),
        "judgments_without_ecli": len(judgments_without_ecli),
        "judgments_document_id_equals_ecli": sum(
            1 for row in judgments_with_ecli if row.get("document_id_equals_ecli")
        ),
        "malformed_ecli_values_sample": sorted(set(malformed))[:50],
        "duplicate_eclis": duplicate_eclis,
        "golden_primary_eclis_present": primary_present,
        "golden_primary_eclis_missing": primary_missing,
        "golden_hard_negative_eclis_present": hn_present,
        "golden_hard_negative_eclis_missing": hn_missing,
        "blocked_primaries_missing_verified_ecli": blocked_primaries,
        "indexing_note": (
            "Do not re-index under doc-*. Index missing verified judgments under their "
            "literal ECLI as document_id/canonical_document_id/ecli. Do not replace the "
            "live collection destructively; use an additive upsert or a versioned collection."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "collection",
        "indexed_chunks",
        "unique_indexed_judgments",
        "judgments_with_ecli",
        "judgments_without_ecli",
        "judgments_document_id_equals_ecli",
        "golden_primary_eclis_missing",
        "blocked_primaries_missing_verified_ecli",
    )}, ensure_ascii=False, indent=2))
    print(f"wrote={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
