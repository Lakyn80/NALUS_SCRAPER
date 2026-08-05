#!/usr/bin/env python3
"""Offline canonical schema v1 pilot export (no Qdrant / BM25 writes).

Selects up to three development-role documents from archetypes_v1.json, maps
parser v7 + hierarchical chunks onto the Phase 2 canonical contract, validates
reconstruction invariants, and writes JSON under artifacts/.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.ingest.chunking import HierarchicalChunkConfig, build_hierarchical_chunks  # noqa: E402
from app.rag.legal_v2.parser import parse_legal_document  # noqa: E402
from app.rag.legal_v2.schema.canonical_v1 import (  # noqa: E402
    DEFAULT_CHUNKING_PROFILE,
    bundle_to_dict,
    content_checksum,
    validate_bundle_invariants,
)
from app.rag.legal_v2.schema.map_from_legal_v2 import (  # noqa: E402
    line_inventory_from_review_rows,
    map_legal_v2_bundle,
)
from scripts.legal_v2.parser_review.models import DEFAULT_REVIEW_DIR, read_jsonl  # noqa: E402

DEFAULT_ARCHETYPES = (
    PROJECT_ROOT / "docs" / "architecture" / "parser_benchmark" / "archetypes_v1.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "legal_v2" / "canonical_schema_pilot"
_LINE_PREFIX_RE = re.compile(r"^\d{5}:\s?")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archetypes", type=Path, default=DEFAULT_ARCHETYPES)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-documents", type=int, default=3)
    args = parser.parse_args(argv)

    archetypes = json.loads(args.archetypes.read_text(encoding="utf-8"))
    selections = _select_development_documents(archetypes, max_documents=args.max_documents)
    if not selections:
        print(json.dumps({"status": "error", "reason": "no_development_documents"}, indent=2))
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for selection in selections:
        result = _export_one(
            selection=selection,
            review_dir=args.review_dir,
            output_dir=args.output_dir,
            parser_profile=str(archetypes.get("parser_profile") or ""),
        )
        results.append(result)

    summary = {
        "status": "ok" if all(item.get("status") == "ok" for item in results) else "partial",
        "schema_contract": "docs/architecture/CANONICAL_BLOCK_CHUNK_SCHEMA_V1.md",
        "chunking_profile": DEFAULT_CHUNKING_PROFILE,
        "output_dir": str(args.output_dir),
        "documents": results,
        "ok_count": sum(1 for item in results if item.get("status") == "ok"),
        "error_count": sum(1 for item in results if item.get("status") != "ok"),
    }
    summary_path = args.output_dir / "pilot_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "ok" else 2


def _select_development_documents(archetypes: dict[str, Any], *, max_documents: int) -> list[dict[str, Any]]:
    inventory = {
        int(item["review_number"]): item
        for item in archetypes.get("inventory", [])
        if item.get("review_number") is not None
    }
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for archetype in archetypes.get("archetypes", []):
        development = archetype.get("development") or {}
        if development.get("status") != "assigned":
            continue
        document_id = development.get("document_id")
        if not document_id or document_id in seen:
            continue
        review_number = development.get("review_number")
        inventory_row = inventory.get(int(review_number)) if review_number is not None else None
        selected.append(
            {
                "archetype_id": archetype.get("archetype_id"),
                "role": "development",
                "review_number": review_number,
                "document_id": document_id,
                "source_id": development.get("source_id")
                or (inventory_row or {}).get("source_id"),
                "case_number": development.get("case_number")
                or (inventory_row or {}).get("case_number"),
                "court": (inventory_row or {}).get("court") or archetype.get("court"),
                "decision_type": (inventory_row or {}).get("document_type")
                or archetype.get("decision_type"),
                "decision_date": (inventory_row or {}).get("decision_date"),
                "source_checksum": development.get("source_checksum")
                or (inventory_row or {}).get("source_checksum"),
            }
        )
        seen.add(document_id)
        if len(selected) >= max_documents:
            break
    return selected


def _export_one(
    *,
    selection: dict[str, Any],
    review_dir: Path,
    output_dir: Path,
    parser_profile: str,
) -> dict[str, Any]:
    document_id = str(selection["document_id"])
    text_path = review_dir / "documents" / document_id / "raw_numbered.txt"
    if not text_path.exists():
        return {
            "status": "skipped_missing_source",
            "document_id": document_id,
            "review_number": selection.get("review_number"),
            "reason": f"missing_review_text:{text_path}",
        }

    raw_numbered = text_path.read_text(encoding="utf-8")
    plain_text = _strip_numbered_prefix(raw_numbered)
    line_rows = [
        row
        for row in read_jsonl(review_dir / "review_lines.jsonl")
        if row.get("document_id") == document_id
    ]
    inventory = line_inventory_from_review_rows(line_rows)

    metadata = {
        "source_id": selection.get("source_id"),
        "source_document_id": selection.get("source_id"),
        "case_number": selection.get("case_number"),
        "court": selection.get("court"),
        "decision_type": selection.get("decision_type"),
        "document_type": selection.get("decision_type"),
        "decision_date": selection.get("decision_date"),
        "source_checksum": selection.get("source_checksum"),
        "language": "cs",
        "jurisdiction": "CZ",
        "archetype_id": selection.get("archetype_id"),
        "archetype_role": selection.get("role"),
    }
    parsed = parse_legal_document(document_id=document_id, text=plain_text, metadata=metadata)
    chunked = build_hierarchical_chunks(parsed, config=HierarchicalChunkConfig())
    bundle = map_legal_v2_bundle(
        parsed,
        chunked,
        line_inventory=inventory,
        parser_profile=parser_profile or parsed.metadata.get("parser_profile") or "",
        chunking_profile=DEFAULT_CHUNKING_PROFILE,
        source_document_id=selection.get("source_id"),
        source_checksum=selection.get("source_checksum") or content_checksum(plain_text),
    )
    report = validate_bundle_invariants(bundle)
    payload = {
        "selection": selection,
        "reconstruction": {
            "ok": report.ok,
            "failure_count": report.failure_count,
            "child_reconstruction_failures": report.child_reconstruction_failures,
            "parent_child_inconsistencies": report.parent_child_inconsistencies,
            "duplicate_ids": report.duplicate_ids,
            "cross_document_refs": report.cross_document_refs,
            "missing_block_refs": report.missing_block_refs,
        },
        "counts": {
            "blocks": len(bundle.blocks),
            "children": len(bundle.children),
            "parents": len(bundle.parents),
            "review_line_rows": len(line_rows),
        },
        "bundle": bundle_to_dict(bundle),
    }
    out_path = output_dir / f"{document_id}.canonical_v1.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "status": "ok" if report.ok else "reconstruction_failed",
        "document_id": document_id,
        "review_number": selection.get("review_number"),
        "archetype_id": selection.get("archetype_id"),
        "output_path": str(out_path),
        "counts": payload["counts"],
        "reconstruction_ok": report.ok,
        "failure_count": report.failure_count,
    }


def _strip_numbered_prefix(raw_numbered: str) -> str:
    lines: list[str] = []
    for line in raw_numbered.splitlines():
        lines.append(_LINE_PREFIX_RE.sub("", line, count=1))
    return "\n".join(lines).strip() + ("\n" if raw_numbered.endswith("\n") else "")


if __name__ == "__main__":
    raise SystemExit(main())
