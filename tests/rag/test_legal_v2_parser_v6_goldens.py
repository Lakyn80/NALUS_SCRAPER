from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from app.rag.legal_v2.ingest.parser import parse_legal_document
from scripts.legal_v2.audit_parser_v6 import _olomouc_expected_classes
from scripts.legal_v2.parser_review.manifest import load_design_documents
from scripts.legal_v2.parser_review.snapshot import (
    _boundary_before,
    _line_class,
    _line_offsets,
    _paragraph_for_line,
    _raw_lines,
)

GOLDEN_DIR = Path("artifacts/legal_v2/parser_golden_inputs")


def _golden_spec() -> dict[str, Any]:
    return json.loads((GOLDEN_DIR / "corrected_golden_spec.json").read_text(encoding="utf-8"))


def _document(review_id: str) -> Any:
    _, documents = load_design_documents()
    return next(document for document in documents if document.review_id == review_id)


def _analyze(review_id: str) -> dict[str, Any]:
    document = _document(review_id)
    raw_lines = _raw_lines(document)
    text = "\n".join(raw_lines)
    first = parse_legal_document(document_id=document.source_id, text=text, metadata={"court": document.court})
    second = parse_legal_document(document_id=document.source_id, text=text, metadata={"court": document.court})
    block_for_line = [
        _paragraph_for_line(first.paragraphs, start, end)
        for start, end in _line_offsets(raw_lines)
    ]
    classes = {
        str(index): _line_class(line, block_for_line[index - 1], first, index, block_for_line)
        for index, line in enumerate(raw_lines, start=1)
    }
    boundaries = {
        str(index): "SPLIT" if block_for_line[index - 1] is not block_for_line[index] else "MERGE"
        for index in range(1, len(raw_lines))
    }
    ranges: list[list[int]] = []
    current = object()
    for index, paragraph in enumerate(block_for_line, start=1):
        if paragraph is not current:
            ranges.append([index, index])
            current = paragraph
        else:
            ranges[-1][1] = index
    return {
        "document": document,
        "raw_lines": raw_lines,
        "text": text,
        "parsed": first,
        "second": second,
        "block_for_line": block_for_line,
        "classes": classes,
        "boundaries": boundaries,
        "ranges": ranges,
    }


@pytest.mark.parametrize("review_id", ["doc-e5ac4b1fcd075062", "doc-cfa470876b0d5ed7"])
def test_v6_exact_golden_classes_boundaries_and_blocks(review_id: str) -> None:
    golden = next(doc for doc in _golden_spec()["documents"] if doc["doc_id"] == review_id)
    actual = _analyze(review_id)

    assert len(actual["raw_lines"]) == golden["line_count"]
    assert actual["classes"] == golden["expected_line_classes"]
    assert actual["boundaries"] == golden["expected_boundaries"]
    assert actual["ranges"] == golden["expected_block_ranges"]
    assert _non_whitespace(actual["parsed"].reconstruct_text()) == _non_whitespace(actual["text"])
    assert [p.paragraph_id for p in actual["parsed"].paragraphs] == [
        p.paragraph_id for p in actual["second"].paragraphs
    ]


def test_v6_constitutional_required_structural_anchors() -> None:
    actual = _analyze("doc-e5ac4b1fcd075062")

    assert actual["classes"]["3"] == "heading"
    assert [5, 7] in actual["ranges"]
    assert [item for item in actual["ranges"] if item in ([12, 13], [22, 23], [25, 26], [28, 29], [35, 36])] == [
        [12, 13],
        [22, 23],
        [25, 26],
        [28, 29],
        [35, 36],
    ]
    assert all(actual["classes"][str(line)] == "numbered_paragraph_start" for line in range(37, 51))
    assert actual["classes"]["52"] == "metadata"
    assert actual["ranges"][-1] == [53, 54]
    assert _primary_citation_count(actual["classes"]) == 0


def test_v6_prague_opening_and_nested_list() -> None:
    actual = _analyze("doc-cfa470876b0d5ed7")

    assert actual["ranges"][0] == [1, 12]
    assert actual["classes"]["12"] != "citation_continuation"
    assert [35, 42] in actual["ranges"]
    assert all(actual["classes"][str(line)] == "list_or_table" for line in range(36, 43))
    assert _primary_citation_count(actual["classes"]) == 0


def test_v6_olomouc_complete_hierarchy_fixture() -> None:
    golden = next(doc for doc in _golden_spec()["documents"] if doc["court"] == "high_court_olomouc")
    actual = _analyze(golden["doc_id"])
    expected_classes = _olomouc_expected_classes(
        golden,
        {
            "lines": [
                {"line": int(line), "text": actual["raw_lines"][int(line) - 1], "class": value}
                for line, value in actual["classes"].items()
            ]
        },
    )
    top_starts = [
        int(line)
        for line, value in actual["classes"].items()
        if value == "numbered_paragraph_start"
    ]
    top_numbers = [
        _required_leading_number(actual["raw_lines"][line - 1])
        for line in top_starts
    ]

    assert len(actual["raw_lines"]) == 698
    assert set(actual["classes"]) == {str(line) for line in range(1, 699)}
    assert actual["classes"] == expected_classes
    assert len(actual["boundaries"]) == 697
    assert actual["classes"]["173"] == "heading"
    assert top_starts == golden["exact_reasoning_top_level_paragraph_start_lines"]
    assert top_numbers == list(range(1, 75))
    assert top_starts[0] == 174
    assert top_starts[1] == 309
    assert top_starts[-1] == 695
    assert all(actual["classes"][str(line)] == "list_or_table" for line in golden["forbidden_false_top_level_starts"])
    assert all(not _boundary_before(line, actual["block_for_line"]) for line in golden["forbidden_false_top_level_starts"])
    assert all(actual["classes"][str(line)] == "list_or_table" for line in _semicolon_table_lines(actual["raw_lines"]))
    assert actual["classes"]["311"] == "numbered_paragraph_continuation"
    assert _primary_citation_count(actual["classes"]) == golden["primary_citation_class_expected_count"]
    assert _non_whitespace(actual["parsed"].reconstruct_text()) == _non_whitespace(actual["text"])
    assert len(actual["raw_lines"]) == len(set(enumerate(actual["raw_lines"], start=1)))
    assert [p.start_offset for p in actual["parsed"].paragraphs] == sorted(p.start_offset for p in actual["parsed"].paragraphs)


def test_v6_court_profiles_are_isolated() -> None:
    constitutional_text = "\n".join(_raw_lines(_document("doc-e5ac4b1fcd075062")))
    prague_text = "\n".join(_raw_lines(_document("doc-cfa470876b0d5ed7")))
    olomouc_text = "\n".join(_raw_lines(_document("doc-4f3c37d9c5a1afb7")))

    constitutional = parse_legal_document(document_id="us", text=constitutional_text, metadata={"court": "constitutional_court"})
    prague = parse_legal_document(document_id="ph", text=prague_text, metadata={"court": "high_court_prague"})
    olomouc = parse_legal_document(document_id="ol", text=olomouc_text, metadata={"court": "high_court_olomouc"})

    assert any(paragraph.original_text == "NÁLEZ Ústavního soudu Jménem republiky" for paragraph in constitutional.paragraphs)
    assert prague.paragraphs[0].original_text.startswith("Vrchní soud v Praze jako soud odvolací")
    assert not any(paragraph.original_text == "NÁLEZ Ústavního soudu Jménem republiky" for paragraph in prague.paragraphs)
    assert sum(1 for paragraph in olomouc.paragraphs if paragraph.numbering) == 82
    assert not olomouc.paragraphs[0].original_text.startswith("Vrchní soud v Praze jako soud odvolací")


def _semicolon_table_lines(lines: list[str]) -> list[int]:
    return [
        index
        for index, line in enumerate(lines, start=1)
        if line.count(";") >= 2 or line.casefold().startswith(("celkem", "; celkem"))
    ]


def _primary_citation_count(classes: dict[str, str]) -> int:
    return sum(1 for value in classes.values() if value == "citation_continuation")


def _required_leading_number(text: str) -> int:
    match = re.match(r"^\s*(\d+)", text)
    if match is None:
        raise AssertionError(f"Expected leading number: {text[:80]}")
    return int(match.group(1))


def _non_whitespace(value: str) -> str:
    return re.sub(r"\s+", "", value)
