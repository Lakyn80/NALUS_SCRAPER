from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.rag.legal_v2.audit import PARSER_VERSION

from .models import PROJECT_ROOT, REVIEW_SCHEMA_VERSION, STUDY_DIR, document_review_id, sha256_file

COURT_COUNTS = {
    "constitutional_court": 10,
    "high_court_prague": 5,
    "high_court_olomouc": 5,
}

IMMUTABLE_PATHS = (
    "app/rag/legal_v2/ingest/parser.py",
    "app/rag/legal_v2/audit.py",
    "app/rag/legal_v2/ingest/indexing.py",
    "tests/rag/test_legal_v2_parser_chunking.py",
)


@dataclass(frozen=True)
class ReviewDocument:
    review_id: str
    review_number: int
    source_id: str
    court: str
    source_checksum: str
    normalized_content_checksum: str
    source_format: str
    raw_path: Path
    source_url: str
    case_number: str
    decision_date: str | None
    document_type: str | None
    manifest_item: dict[str, Any]


def load_design_documents(manifest_path: Path | None = None) -> tuple[dict[str, Any], list[ReviewDocument]]:
    path = manifest_path or STUDY_DIR / "design_sample_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Design manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    selected = manifest.get("selected_documents")
    if not isinstance(selected, list):
        raise ValueError("Design manifest is missing selected_documents")
    if len(selected) != 20:
        raise ValueError(f"Design manifest must contain 20 documents, found {len(selected)}")
    counts = {court: 0 for court in COURT_COUNTS}
    checksums: list[str] = []
    documents: list[ReviewDocument] = []
    manifest_checksum = str(manifest.get("manifest_checksum") or "")
    for index, item in enumerate(selected, start=1):
        court = str(item.get("court") or "")
        if court not in counts:
            raise ValueError(f"Unexpected court in design manifest: {court}")
        counts[court] += 1
        raw_path = PROJECT_ROOT / Path(str(item.get("raw_path") or ""))
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw source for {item.get('source_id')}: {raw_path}")
        expected_checksum = str(item.get("source_checksum") or "")
        actual_checksum = sha256_file(raw_path)
        if expected_checksum != actual_checksum:
            raise ValueError(
                f"Raw checksum mismatch for {item.get('source_id')}: expected={expected_checksum} actual={actual_checksum} path={raw_path}"
            )
        if expected_checksum in checksums:
            raise ValueError(f"Duplicate raw source checksum in design manifest: {expected_checksum}")
        checksums.append(expected_checksum)
        source_id = str(item.get("source_id") or "")
        review_id = document_review_id(
            source_id=source_id,
            source_checksum=expected_checksum,
            manifest_checksum=manifest_checksum,
        )
        documents.append(
            ReviewDocument(
                review_id=review_id,
                review_number=index,
                source_id=source_id,
                court=court,
                source_checksum=expected_checksum,
                normalized_content_checksum=str(item.get("normalized_content_checksum") or ""),
                source_format=str(item.get("source_format") or ""),
                raw_path=raw_path,
                source_url=str(item.get("source_url") or ""),
                case_number=str(item.get("case_number") or ""),
                decision_date=item.get("decision_date"),
                document_type=item.get("decision_type"),
                manifest_item=dict(item),
            )
        )
    for court, expected in COURT_COUNTS.items():
        if counts[court] != expected:
            raise ValueError(f"Design manifest court distribution mismatch for {court}: expected={expected} actual={counts[court]}")
    return manifest, documents


def parser_git_identity() -> dict[str, Any]:
    blobs = {}
    for rel in IMMUTABLE_PATHS:
        blobs[rel] = subprocess.check_output(["git", "rev-parse", f"HEAD:{rel}"], cwd=PROJECT_ROOT, text=True).strip()
    return {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "parser_profile": PARSER_VERSION,
        "immutable_git_blobs": blobs,
    }


def base_manifest_payload(source_manifest: dict[str, Any], documents: list[ReviewDocument]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for document in documents:
        counts[document.court] = counts.get(document.court, 0) + 1
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "source_manifest_path": str((STUDY_DIR / "design_sample_manifest.json").relative_to(PROJECT_ROOT)),
        "source_manifest_checksum": source_manifest.get("manifest_checksum"),
        "source_manifest_sha256": sha256_file(STUDY_DIR / "design_sample_manifest.json"),
        "document_count": len(documents),
        "court_distribution": counts,
        **parser_git_identity(),
    }
