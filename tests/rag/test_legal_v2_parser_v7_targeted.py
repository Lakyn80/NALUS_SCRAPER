from __future__ import annotations

from typing import Any

import pytest

from app.rag.legal_v2.audit import PARSER_VERSION
from app.rag.legal_v2.ingest.parser import parse_legal_document
from scripts.legal_v2.parser_review.manifest import load_design_documents
from scripts.legal_v2.parser_review.snapshot import (
    _line_class,
    _line_offsets,
    _paragraph_for_line,
    _raw_lines,
)

DOC17_STARTS = [18, 20, 37, 40, 41, 42, 44, 45, 46, 48, 50, 51, 53, 55, 56, 57, 58, 59, 60]
DOC17_CONTINUATIONS = [
    19,
    *range(21, 37),
    38,
    39,
    43,
    47,
    49,
    52,
    54,
    *range(61, 65),
]
DOC17_RANGES = [
    [18, 19],
    [20, 36],
    [37, 39],
    [40, 40],
    [41, 41],
    [42, 43],
    [44, 44],
    [45, 45],
    [46, 47],
    [48, 49],
    [50, 50],
    [51, 52],
    [53, 54],
    [55, 55],
    [56, 56],
    [57, 57],
    [58, 58],
    [59, 59],
    [60, 64],
]
EXPECTED_BLOCKS = {6: 42, 7: 44, 10: 33, 14: 30, 17: 25, 18: 51, 19: 20, 20: 29}


def _by_review_number() -> dict[int, Any]:
    _, documents = load_design_documents()
    return {int(document.review_number): document for document in documents}


def _analyze(review_number: int) -> dict[str, Any]:
    document = _by_review_number()[review_number]
    raw_lines = _raw_lines(document)
    text = "\n".join(raw_lines)
    first = parse_legal_document(document_id=document.source_id, text=text, metadata={"court": document.court})
    second = parse_legal_document(document_id=document.source_id, text=text, metadata={"court": document.court})
    block_for_line = [
        _paragraph_for_line(first.paragraphs, start, end) for start, end in _line_offsets(raw_lines)
    ]
    classes = {
        index: _line_class(line, block_for_line[index - 1], first, index, block_for_line)
        for index, line in enumerate(raw_lines, start=1)
    }
    boundaries = {
        index: "SPLIT" if block_for_line[index - 1] is not block_for_line[index] else "MERGE"
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
        "classes": classes,
        "boundaries": boundaries,
        "ranges": ranges,
        "block_for_line": block_for_line,
    }


def test_parser_profile_is_v8() -> None:
    assert PARSER_VERSION == "legal-decision-parser.cz-courts.v8"


@pytest.mark.parametrize(
    ("review_number", "line", "expected"),
    [
        (6, 33, "heading"),
        (6, 42, "heading"),
        (7, 30, "heading"),
        (10, 13, "heading"),
        (10, 17, "heading"),
    ],
)
def test_constitutional_subheading_classes(review_number: int, line: int, expected: str) -> None:
    actual = _analyze(review_number)
    assert actual["classes"][line] == expected


@pytest.mark.parametrize(
    ("review_number", "before", "after"),
    [
        (6, 32, 33),
        (6, 33, 34),
        (6, 41, 42),
        (6, 42, 43),
        (7, 29, 30),
        (7, 30, 31),
        (10, 12, 13),
        (10, 13, 14),
        (10, 16, 17),
        (10, 17, 18),
    ],
)
def test_constitutional_subheading_boundaries_split(review_number: int, before: int, after: int) -> None:
    actual = _analyze(review_number)
    assert actual["boundaries"][before] == "SPLIT"
    assert after == before + 1


def test_document_14_prague_opening_formula() -> None:
    actual = _analyze(14)
    assert actual["classes"][1] == "prose_start"
    assert all(actual["classes"][line] == "prose_continuation" for line in range(2, 12))
    assert all(actual["boundaries"][line] == "MERGE" for line in range(1, 11))
    assert actual["boundaries"][11] == "SPLIT"
    assert actual["ranges"][0] == [1, 11]
    assert len(actual["ranges"]) == EXPECTED_BLOCKS[14]


def test_document_17_olomouc_structure() -> None:
    actual = _analyze(17)
    for line in (14, 15, 16):
        assert actual["classes"][line] == "numbered_paragraph_start"
    assert actual["boundaries"][13] == "SPLIT"
    assert actual["boundaries"][14] == "SPLIT"
    assert actual["boundaries"][15] == "SPLIT"
    assert actual["boundaries"][16] == "SPLIT"
    starts = [
        line
        for line, cls in actual["classes"].items()
        if cls == "numbered_paragraph_start" and line >= 18
    ]
    assert starts == DOC17_STARTS
    numbers = []
    for line in starts:
        text = actual["raw_lines"][line - 1].lstrip()
        numbers.append(int(text.split(".", 1)[0]))
    assert numbers == list(range(1, 20))
    for line in DOC17_CONTINUATIONS:
        assert actual["classes"][line] == "numbered_paragraph_continuation"
    assert actual["classes"][33] == "numbered_paragraph_continuation"
    assert actual["classes"][33] != "list_or_table"
    assert actual["classes"][33] != "numbered_paragraph_start"
    reasoning_ranges = [row for row in actual["ranges"] if row[0] >= 18]
    assert reasoning_ranges == DOC17_RANGES
    assert len(actual["ranges"]) == EXPECTED_BLOCKS[17]


def test_document_18_olomouc_structure() -> None:
    actual = _analyze(18)
    assert actual["classes"][12] == "numbered_paragraph_start"
    assert actual["classes"][13] == "numbered_paragraph_start"
    assert actual["boundaries"][12] == "SPLIT"
    starts = [
        line
        for line, cls in actual["classes"].items()
        if cls == "numbered_paragraph_start" and 15 <= line <= 60
    ]
    assert starts == list(range(15, 61))
    numbers = []
    for line in starts:
        text = actual["raw_lines"][line - 1].lstrip()
        numbers.append(int(text.split(".", 1)[0]))
    assert numbers == list(range(1, 47))
    assert all(actual["boundaries"][line] == "SPLIT" for line in range(15, 60))
    assert actual["boundaries"][58] == "SPLIT"
    assert actual["boundaries"][59] == "SPLIT"
    assert len(actual["ranges"]) == EXPECTED_BLOCKS[18]


def test_document_19_olomouc_structure() -> None:
    actual = _analyze(19)
    assert actual["classes"][15] == "numbered_paragraph_start"
    assert actual["classes"][16] == "numbered_paragraph_start"
    assert actual["boundaries"][15] == "SPLIT"
    expected_starts = [18, 19, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    starts = [
        line
        for line, cls in actual["classes"].items()
        if cls == "numbered_paragraph_start" and line >= 18
    ]
    assert starts == expected_starts
    numbers = []
    for line in starts:
        text = actual["raw_lines"][line - 1].lstrip()
        numbers.append(int(text.split(".", 1)[0]))
    assert numbers == list(range(1, 16))
    assert all(actual["classes"][line] == "list_or_table" for line in (20, 21, 22))
    assert actual["boundaries"][19] == "MERGE"
    assert actual["boundaries"][20] == "MERGE"
    assert actual["boundaries"][21] == "MERGE"
    assert actual["boundaries"][22] == "SPLIT"
    assert [19, 22] in actual["ranges"] or any(row == [19, 22] for row in actual["ranges"])
    assert len(actual["ranges"]) == EXPECTED_BLOCKS[19]


def test_document_20_olomouc_structure() -> None:
    actual = _analyze(20)
    assert actual["classes"][10] == "numbered_paragraph_start"
    assert actual["classes"][11] == "numbered_paragraph_start"
    assert actual["boundaries"][10] == "SPLIT"
    starts = [
        line
        for line, cls in actual["classes"].items()
        if cls == "numbered_paragraph_start" and 13 <= line <= 36
    ]
    assert starts == list(range(13, 37))
    numbers = []
    for line in starts:
        text = actual["raw_lines"][line - 1].lstrip()
        numbers.append(int(text.split(".", 1)[0]))
    assert numbers == list(range(1, 25))
    assert all(actual["boundaries"][line] == "SPLIT" for line in range(13, 36))
    assert len(actual["ranges"]) == EXPECTED_BLOCKS[20]


@pytest.mark.parametrize("review_number", sorted(EXPECTED_BLOCKS))
def test_targeted_block_counts(review_number: int) -> None:
    actual = _analyze(review_number)
    assert len(actual["ranges"]) == EXPECTED_BLOCKS[review_number]


def test_all_20_documents_conservation_ordering_determinism() -> None:
    for review_number, document in sorted(_by_review_number().items()):
        raw_lines = _raw_lines(document)
        text = "\n".join(raw_lines)
        first = parse_legal_document(document_id=document.source_id, text=text, metadata={"court": document.court})
        second = parse_legal_document(document_id=document.source_id, text=text, metadata={"court": document.court})
        reconstructed = first.reconstruct_text()
        assert "".join(text.split()) == "".join(reconstructed.split()), review_number
        assert [paragraph.start_offset for paragraph in first.paragraphs] == sorted(
            paragraph.start_offset for paragraph in first.paragraphs
        )
        assert [paragraph.original_text for paragraph in first.paragraphs] == [
            paragraph.original_text for paragraph in second.paragraphs
        ]
        seen_offsets = set()
        for paragraph in first.paragraphs:
            key = (paragraph.start_offset, paragraph.end_offset)
            assert key not in seen_offsets
            seen_offsets.add(key)


def test_court_profile_isolation_criminal_golden_unchanged() -> None:
    actual = _analyze(16)
    assert len(actual["raw_lines"]) == 698
    assert len(actual["boundaries"]) == 697
    assert len(actual["ranges"]) == 94
    starts = [
        line
        for line, cls in actual["classes"].items()
        if cls == "numbered_paragraph_start" and line >= 174
    ]
    assert 182 not in starts
    assert all(line not in starts for line in range(296, 302))
