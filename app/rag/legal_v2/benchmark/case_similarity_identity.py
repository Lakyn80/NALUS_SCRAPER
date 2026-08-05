"""Load verified source_document_id → ECLI mappings for case-similarity golden."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.rag.legal_v2.identity import (
    IDENTITY_STATUS_BLOCKED_MISSING_ECLI,
    IDENTITY_STATUS_VERIFIED,
    ecli_key,
    normalize_ecli,
    validate_decision_identity,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_IDENTITY_MAP = (
    PROJECT_ROOT / "benchmarks" / "legal_v2" / "case_similarity_document_identity_v1.json"
)


def load_case_similarity_identity_map(
    path: Path | str = DEFAULT_IDENTITY_MAP,
) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    documents = payload.get("documents") or []
    by_source: dict[str, dict[str, Any]] = {}
    by_ecli: dict[str, str] = {}
    for row in documents:
        source_id = str(row.get("source_document_id") or "").strip()
        if not source_id:
            raise ValueError("identity map row missing source_document_id")
        if source_id in by_source:
            raise ValueError(f"duplicate source_document_id in identity map: {source_id}")
        status = str(row.get("identity_status") or "").strip()
        ecli = row.get("ecli")
        canonical = row.get("canonical_document_id")
        if status == IDENTITY_STATUS_VERIFIED:
            normalized = validate_decision_identity(ecli=ecli, canonical_document_id=canonical)
            key = ecli_key(normalized)
            if key in by_ecli and by_ecli[key] != source_id:
                raise ValueError(
                    f"duplicate ECLI {normalized} mapped to {by_ecli[key]} and {source_id}"
                )
            by_ecli[key] = source_id
            row = dict(row)
            row["ecli"] = normalized
            row["canonical_document_id"] = normalized
            row["identity_status"] = IDENTITY_STATUS_VERIFIED
        elif status == IDENTITY_STATUS_BLOCKED_MISSING_ECLI:
            if ecli is not None or canonical is not None:
                raise ValueError(
                    f"blocked identity row must have null ecli/canonical for {source_id}"
                )
        else:
            raise ValueError(f"unsupported identity_status {status!r} for {source_id}")
        by_source[source_id] = row
    return by_source


def identity_for_source(
    source_document_id: str,
    mapping: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    table = mapping if mapping is not None else load_case_similarity_identity_map()
    if source_document_id not in table:
        raise KeyError(f"no identity mapping for {source_document_id}")
    return table[source_document_id]


def source_to_ecli(
    source_document_id: str,
    mapping: dict[str, dict[str, Any]] | None = None,
) -> str | None:
    row = identity_for_source(source_document_id, mapping)
    value = row.get("ecli")
    return normalize_ecli(value) if value else None
