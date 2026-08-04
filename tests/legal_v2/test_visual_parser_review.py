from __future__ import annotations

import json

from scripts.legal_v2.parser_review.manifest import load_design_documents
from scripts.legal_v2.parser_review.models import REVIEW_SCHEMA_VERSION, line_review_id
from scripts.legal_v2.parser_review.snapshot import build_snapshot


def test_design_manifest_validation_and_stable_document_ids() -> None:
    manifest, documents = load_design_documents()
    counts = {}
    for document in documents:
        counts[document.court] = counts.get(document.court, 0) + 1
    assert len(documents) == 20
    assert counts == {"constitutional_court": 10, "high_court_prague": 5, "high_court_olomouc": 5}
    assert len({document.review_id for document in documents}) == 20
    assert manifest["kind"] == "design"


def test_line_id_is_deterministic() -> None:
    first = line_review_id(document_id="doc-x", raw_line_number=7, source_checksum="sha")
    second = line_review_id(document_id="doc-x", raw_line_number=7, source_checksum="sha")
    assert first == second
    assert "00007" in first


def test_snapshot_validate_only_uses_exact_design_source() -> None:
    result = build_snapshot(validate_only=True, document_filter="1")
    assert result["manifest"]["schema_version"] == REVIEW_SCHEMA_VERSION
    assert result["documents"] == 1
    assert result["lines"] > 0
    json.dumps(result, ensure_ascii=False)
