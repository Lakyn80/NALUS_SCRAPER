from __future__ import annotations

import json
from pathlib import Path

from scripts.legal_v2.parser_review.full_export import (
    DEFAULT_OUTPUT_DIR,
    GOLDEN_DOCUMENT_IDS,
    GOLDEN_REVIEW_NUMBERS,
    JSON_NAME,
    MARKDOWN_NAME,
    build_export_payload,
    export_full_review,
    render_markdown,
    semantic_fingerprint,
    validate_export_payload,
    validate_markdown,
)
from scripts.legal_v2.parser_review.models import DEFAULT_REVIEW_DIR, read_jsonl, sha256_file
from scripts.legal_v2.parser_review.status import GOLDEN_DIR, ParserValidationStatus
from scripts.legal_v2.parser_review.web_api import ReviewApi
from scripts.legal_v2.parser_review.web_server import EXPORT_FILES, ReviewRequestHandler


REVIEW_DIR = DEFAULT_REVIEW_DIR
DECISIONS = REVIEW_DIR / "manual_review_decisions.jsonl"
HISTORY = REVIEW_DIR / "manual_review_history.jsonl"


def _manual_fingerprint() -> tuple[str, int, str, int]:
    return (
        sha256_file(DECISIONS),
        DECISIONS.stat().st_size,
        sha256_file(HISTORY),
        HISTORY.stat().st_size,
    )


def test_exporter_finds_all_20_review_documents_and_classifies_golden_set() -> None:
    documents = read_jsonl(REVIEW_DIR / "review_documents.jsonl")
    assert len(documents) == 20
    payload = build_export_payload(snapshot_dir=REVIEW_DIR, first_commit="test-first")
    assert len(payload["remaining_documents"]) == 17
    golden_ids = {row["document_id"] for row in payload["golden_regressions"]}
    assert golden_ids == GOLDEN_DOCUMENT_IDS
    assert {int(row["review_index"]) for row in payload["golden_regressions"]} == GOLDEN_REVIEW_NUMBERS
    assert all(row["exact_golden_coverage"] is True for row in payload["golden_regressions"])
    assert all(row["exact_golden_coverage"] is False for row in payload["remaining_documents"])
    assert all(
        row["parser_validation_status"] != ParserValidationStatus.AUTO_VALIDATED_GOLDEN.value
        for row in payload["remaining_documents"]
    )
    assert all(row["verdict"] == "GOLDEN PASS" for row in payload["golden_regressions"])


def test_complete_json_and_markdown_cover_all_lines_boundaries_blocks(tmp_path: Path) -> None:
    before = _manual_fingerprint()
    result = export_full_review(
        snapshot_dir=REVIEW_DIR,
        output_dir=tmp_path,
        first_commit="test-first",
        verify_determinism=True,
    )
    after = _manual_fingerprint()
    assert before == after

    payload = json.loads((tmp_path / JSON_NAME).read_text(encoding="utf-8"))
    markdown = (tmp_path / MARKDOWN_NAME).read_text(encoding="utf-8")
    validate_export_payload(payload)
    validate_markdown(markdown, payload)
    assert result["documents"] == 17
    assert len(payload["remaining_documents"]) == 17
    assert "# Cross-document review candidates" in markdown

    for document in payload["remaining_documents"]:
        assert len(document["lines"]) == document["line_count"]
        assert len(document["boundaries"]) == document["boundary_count"] == document["line_count"] - 1
        assert len(document["blocks"]) == document["block_count"]
        assert all(line.get("current_parser_class") for line in document["lines"])
        assert all(boundary.get("parser_v6_decision") in {"SPLIT", "MERGE"} for boundary in document["boundaries"])
        assert all(block.get("stable_block_id") for block in document["blocks"])
        assert all(line.get("raw_text") is not None for line in document["lines"])
        assert all(boundary.get("full_text_before") is not None for boundary in document["boundaries"])
        assert all(boundary.get("full_text_after") is not None for boundary in document["boundaries"])
        assert all(block.get("complete_text") is not None for block in document["blocks"])
        assert all("previous_v5_class" in line for line in document["lines"])
        assert all(line.get("parser_validation_status") for line in document["lines"])
        assert all(line.get("manual_review_status") for line in document["lines"])
        assert "Fully correct." not in document["document_verdict"]
        assert "Human verified." not in document["document_verdict"]
        assert "Golden pass." not in document["document_verdict"]

    assert payload["corpus_summary"]["total_lines"] == sum(doc["line_count"] for doc in payload["remaining_documents"])
    assert payload["corpus_summary"]["total_boundaries"] == sum(
        doc["boundary_count"] for doc in payload["remaining_documents"]
    )
    assert payload["corpus_summary"]["total_blocks"] == sum(doc["block_count"] for doc in payload["remaining_documents"])
    import re

    assert len(re.findall(r"^# Document \d{2} —", markdown, flags=re.MULTILINE)) == 17
    assert "á" in markdown or "č" in markdown or "ř" in markdown or "ž" in markdown


def test_exporter_is_deterministic_across_temp_directories(tmp_path: Path) -> None:
    first_dir = tmp_path / "a"
    second_dir = tmp_path / "b"
    first = export_full_review(
        snapshot_dir=REVIEW_DIR,
        output_dir=first_dir,
        first_commit="abc",
        verify_determinism=True,
    )
    payload_a = json.loads((first_dir / JSON_NAME).read_text(encoding="utf-8"))
    payload_b = build_export_payload(
        snapshot_dir=REVIEW_DIR,
        first_commit="abc",
        generated_at=payload_a["generated_at"],
    )
    export_full_review(
        snapshot_dir=REVIEW_DIR,
        output_dir=second_dir,
        first_commit="abc",
    )
    payload_c = json.loads((second_dir / JSON_NAME).read_text(encoding="utf-8"))
    assert semantic_fingerprint(payload_a) == semantic_fingerprint(payload_b)
    assert semantic_fingerprint(payload_a) == semantic_fingerprint(
        {**payload_c, "generated_at": payload_a["generated_at"]}
    )
    assert render_markdown(payload_a) == render_markdown(payload_b)
    assert render_markdown(payload_a) == render_markdown(
        json.loads(json.dumps(payload_b, ensure_ascii=False, sort_keys=True))
    )
    assert first["documents"] == 17


def test_exporter_does_not_mutate_manual_or_golden_or_raw_sources(tmp_path: Path) -> None:
    before_decisions = sha256_file(DECISIONS)
    before_history = sha256_file(HISTORY)
    golden_before = {
        path.name: (path.stat().st_size, sha256_file(path))
        for path in GOLDEN_DIR.iterdir()
        if path.is_file()
    }
    raw_files = sorted((REVIEW_DIR / "documents").glob("*/raw_numbered.txt"))
    raw_before = {str(path): (path.stat().st_size, sha256_file(path)) for path in raw_files}

    export_full_review(snapshot_dir=REVIEW_DIR, output_dir=tmp_path, first_commit="safe")

    assert sha256_file(DECISIONS) == before_decisions
    assert sha256_file(HISTORY) == before_history
    golden_after = {
        path.name: (path.stat().st_size, sha256_file(path))
        for path in GOLDEN_DIR.iterdir()
        if path.is_file()
    }
    assert golden_before == golden_after
    raw_after = {str(path): (path.stat().st_size, sha256_file(path)) for path in raw_files}
    assert raw_before == raw_after


def test_source_order_conservation_and_no_duplication_in_export() -> None:
    payload = build_export_payload(snapshot_dir=REVIEW_DIR, first_commit="order")
    for document in payload["remaining_documents"]:
        numbers = [line["line_number"] for line in document["lines"]]
        assert numbers == sorted(numbers)
        assert len(numbers) == len(set(numbers))
        assert document["text_conservation"] is True
        assert document["duplication_count"] == 0
        assert document["ordering_failures"] == 0
        owned: set[int] = set()
        for block in document["blocks"]:
            for number in block["line_numbers"]:
                assert number not in owned
                owned.add(number)
        assert owned == set(numbers)


def test_html_full_corpus_api_and_copy_action_and_existing_views() -> None:
    api = ReviewApi(REVIEW_DIR)
    status, payload = api.get("/api/full-corpus-v7", {})
    assert status == 200
    assert len(payload["documents"]) == 20
    assert len(payload["golden_documents"]) == 3
    assert len(payload["remaining_documents"]) == 17
    assert all(doc["display_parser_label"] == "GOLDEN PASS" for doc in payload["golden_documents"])
    assert all(doc["exact_golden_coverage"] is False for doc in payload["remaining_documents"])
    assert all(
        doc["parser_validation_status"] != ParserValidationStatus.AUTO_VALIDATED_GOLDEN.value
        for doc in payload["remaining_documents"]
    )

    remaining_id = payload["remaining_documents"][0]["document_id"]
    status, markdown_payload = api.get(
        "/api/full-corpus-v7/document-markdown",
        {"document_id": [remaining_id]},
    )
    assert status == 200
    assert f"# Document {int(payload['remaining_documents'][0]['review_number']):02d}" in markdown_payload["markdown"]
    assert "## Complete line classification" in markdown_payload["markdown"]
    assert "## Complete boundaries" in markdown_payload["markdown"]
    assert "## Complete blocks" in markdown_payload["markdown"]

    status, historical = api.get("/api/full-corpus-v6", {})
    assert status == 200
    assert len(historical["documents"]) == 20

    for path in (
        "/api/documents",
        "/api/progress",
        "/api/problems",
        "/api/assisted/summary",
        "/api/parser-v7/changes",
        "/api/parser-v6/changes",
    ):
        status, _ = api.get(path, {"document_id": [remaining_id]} if "changes" in path or path == "/api/problems" else {})
        assert status == 200

    status, lines_payload = api.get("/api/lines", {"document_id": ["doc-b73cac9b3dfc8a42"]})
    assert status == 200
    assert len(lines_payload["lines"]) == 13
    status, boundaries_payload = api.get("/api/boundaries", {"document_id": ["doc-b73cac9b3dfc8a42"]})
    assert status == 200
    assert len(boundaries_payload["boundaries"]) == 12


def test_export_download_routes_point_to_generated_files(tmp_path: Path) -> None:
    assert f"/exports/{JSON_NAME}" in EXPORT_FILES
    assert f"/exports/{MARKDOWN_NAME}" in EXPORT_FILES
    export_full_review(snapshot_dir=REVIEW_DIR, output_dir=DEFAULT_OUTPUT_DIR, first_commit="routes")
    json_path, json_type = EXPORT_FILES[f"/exports/{JSON_NAME}"]
    md_path, md_type = EXPORT_FILES[f"/exports/{MARKDOWN_NAME}"]
    assert json_path.exists()
    assert md_path.exists()
    assert "json" in json_type
    assert "markdown" in md_type
    assert ReviewRequestHandler is not None
    # Keep temporary path argument exercised for isolation-friendly callers.
    assert tmp_path.exists()


def test_document_2_manual_completion_remains_unchanged_by_export(tmp_path: Path) -> None:
    before = _manual_fingerprint()
    export_full_review(snapshot_dir=REVIEW_DIR, output_dir=tmp_path, first_commit="doc2")
    after = _manual_fingerprint()
    assert before == after
    api = ReviewApi(REVIEW_DIR)
    status, payload = api.get("/api/document", {"id": ["doc-b73cac9b3dfc8a42"]})
    assert status == 200
    document = payload["document"]
    assert document["manual_line_reviewed"] == 13
    assert document["manual_line_total"] == 13
    assert document["manual_boundary_reviewed"] == 12
    assert document["manual_boundary_total"] == 12


def test_non_golden_documents_are_not_automatically_approved(tmp_path: Path) -> None:
    before = DECISIONS.read_text(encoding="utf-8")
    export_full_review(snapshot_dir=REVIEW_DIR, output_dir=tmp_path, first_commit="no-approve")
    after = DECISIONS.read_text(encoding="utf-8")
    assert before == after
    payload = json.loads((tmp_path / JSON_NAME).read_text(encoding="utf-8"))
    for document in payload["remaining_documents"]:
        assert document["parser_validation_status"] != ParserValidationStatus.AUTO_VALIDATED_GOLDEN.value
        assert document["exact_golden_coverage"] is False


def test_conflict_and_stale_categories_are_consistent() -> None:
    payload = build_export_payload(snapshot_dir=REVIEW_DIR, first_commit="conflict-check")
    validate_export_payload(payload)
    summary = payload["corpus_summary"]
    candidates = payload["cross_document_review_candidates"]
    assert summary["parser_manual_conflicts"] == len(candidates["parser_manual_conflicts"])
    assert summary["stale_manual_decisions"] == len(candidates["stale_manual_decisions"])
    conflict_keys = {(row["review_index"], row["locator"]) for row in candidates["parser_manual_conflicts"]}
    stale_keys = {(row["review_index"], row["locator"]) for row in candidates["stale_manual_decisions"]}
    assert not (conflict_keys & stale_keys)
    for document in payload["remaining_documents"]:
        for line in document["lines"]:
            if line.get("manual_review_status") == "MANUAL_DECISION_STALE" or line.get("stale_decision_flag"):
                assert line.get("manual_review_status") != "MANUAL_CONFLICT"


def test_opening_formula_overmerge_false_positives_suppressed() -> None:
    payload = build_export_payload(snapshot_dir=REVIEW_DIR, first_commit="overmerge-check")
    by_index = {int(row["review_index"]): row for row in payload["remaining_documents"]}
    for review_index, start, end in ((12, 1, 15), (17, 1, 12), (19, 1, 13)):
        document = by_index[review_index]
        opening = next(block for block in document["blocks"] if block["start_line"] == start)
        assert opening["end_line"] == end
        assert opening["suspicious_overmerge_flag"] is False
    # Genuine long non-opening blocks may still be flagged.
    flagged = [
        block
        for document in payload["remaining_documents"]
        for block in document["blocks"]
        if block["suspicious_overmerge_flag"]
    ]
    assert isinstance(flagged, list)
