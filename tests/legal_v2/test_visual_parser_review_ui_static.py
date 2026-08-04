from __future__ import annotations

from pathlib import Path


STATIC_DIR = Path("scripts/legal_v2/parser_review/static")


def test_review_ui_css_contains_viewport_bounded_app_shell() -> None:
    css = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    assert "grid-template-columns: minmax(260px, 320px) minmax(0, 1fr)" in css
    assert "height: 100dvh" in css
    assert "main {\n  min-width: 0;" in css
    assert "header > div {\n  min-width: 0;" in css
    assert "#work {\n  flex: 1 1 auto;" in css
    assert "overflow: auto" in css
    assert "body {\n  overflow-x: hidden;" not in css


def test_review_ui_bounds_tables_and_long_text() -> None:
    css = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert ".table-shell" in css
    assert "table-layout: fixed" in css
    assert "overflow-wrap: anywhere" in css
    assert ".controls { display: grid; grid-template-columns: minmax(0, 1fr);" in css
    assert '<div class="line-list">' in js
    assert "renderLineCard" in js
    assert "Parser v6 result" in js
    assert '<div class="boundary-list">' in js


def test_review_ui_has_responsive_single_column_breakpoint() -> None:
    css = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    assert "@media (max-width: 1100px)" in css
    assert "grid-template-columns: minmax(0, 1fr)" in css
    assert "max-height: 220px" in css


def test_boundary_review_renders_cards_with_explicit_decision_language() -> None:
    css = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert "boundary-card" in css
    assert "LINE BEFORE BOUNDARY" in js
    assert "LINE AFTER BOUNDARY" in js
    assert "PARSER v6: ${parserDisplay}" in js
    assert "PREVIOUS: ${card.previous_boundary.display}" in js
    assert "Accept parser: ${parserDisplay}" in js
    assert "Force SPLIT before line" in js
    assert "Force MERGE with line" in js
    assert "This will save: ${parserDisplay}" in js
    assert "Saved successfully. Revision" in js
    assert "Save failed:" in js
    assert "parser_proposed_boundary} ·" not in js
    assert "previous_automated_boundary_annotation}</td>" not in js


def test_review_ui_exposes_parser_v6_changed_queues() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    css = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert '<option value="parser-v6-changes">Changed by parser v6</option>' in html
    assert "/api/parser-v6/changes?document_id=" in js
    assert "renderParserV6Changes" in js
    assert "Changed Lines / Classes" in js
    assert "Changed Boundaries" in js
    assert "Changed Blocks" in js
    assert ".change-queue" in css
    assert ".change-section" in css


def test_review_ui_separates_parser_validation_from_manual_review() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    css = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert '<option value="problems">Problems</option>' in html
    assert '<option value="progress">Progress</option>' in html
    assert "Parser validation" in js
    assert "Manual review" in js
    assert "renderProgressView" in js
    assert "renderProblems" in js
    assert '<details class="manual-controls">' in js
    assert '<details class="manual-panel manual-controls">' in js
    assert "Manual review pending" not in js
    assert "<span class=\"pill\">${escapeHtml(card.status)}</span>" not in js
    assert ".status-badge.success" in css
    assert ".status-badge.review" in css
    assert ".status-badge.manual" in css


def test_review_ui_exposes_full_corpus_v6_view_and_exports() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    css = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert '<option value="full-corpus-v6">Full corpus v6 review</option>' in html
    assert 'id="copyDocumentReview"' in html
    assert "renderFullCorpusV6" in js
    assert "/api/full-corpus-v6" in js
    assert "/api/full-corpus-v6/document-markdown" in js
    assert "Download complete JSON" in js
    assert "Download complete Markdown" in js
    assert "Copy document review" in js
    assert ".full-corpus-view" in css
    assert ".corpus-card.golden" in css
    assert ".corpus-card.remaining" in css
