from __future__ import annotations

import re

from app.rag.legal_v2.ingest.parser import parse_legal_document
from scripts.legal_v2.parser_review.snapshot import (
    _boundary_before,
    _line_class,
    _line_offsets,
    _paragraph_for_line,
)


DOC2_STRUCTURAL_LINES = [
    "NALUS - databáze rozhodnutí Ústavního soudu",
    "I.ÚS 3299/24 ze dne 20. 12. 2024",
    "Česká republika",
    "USNESENÍ",
    "Ústavního soudu",
    "Ústavní soud rozhodl soudcem zpravodajem JUDr. Jaromírem Jirsou, ve věci návrhu 1) Aleny Houžvičkové a 2) Ing. Jaromíra Houžvičky, bez právního zastoupení, proti usnesení Krajského soudu v Hradci Králové č. j. 19 Co 182/2024-1218 ze dne 28. srpna 2024 a usnesení Okresního soudu v Jičíně č. j. 6 C 62/2007-1208 ze dne 31. května 2024, takto:",
    "Návrh se odmítá.",
    "Odůvodnění:",
    'Navrhovatelé se domáhali zrušení napadených rozhodnutí a vyloučení všech soudců Ústavního soudu podáním, které nesplňovalo náležitosti řádného návrhu na zahájení řízení před Ústavním soudem dané zákonem č. 182/1993 Sb., o Ústavním soudu, ve znění pozdějších předpisů (dále jen "zákon o Ústavním soudu"). Jelikož je z lustra patrné, že v minulosti byli navrhovatelé opakovaně informováni o zákonných náležitostech návrhu a postupu jak jeho nedostatky odstranit, považuje Ústavní soud za nadbytečné činit tak opakovaně a vyzývat je k odstranění vad, a proto jejich podání bez dalšího odmítá podle § 43 odst. 1 písm. a) zákona o Ústavním soudu.',
    "Poučení: Proti rozhodnutí Ústavního soudu není odvolání přípustné.",
    "V Brně dne 20. 12. 2024",
    "Jaromír Jirsa v. r.",
    "soudce zpravodaj",
]


def test_constitutional_court_document2_golden_structure() -> None:
    text = "\n".join(DOC2_STRUCTURAL_LINES)

    first = parse_legal_document(
        document_id="1-3299-24_1",
        text=text,
        metadata={"court": "constitutional_court"},
    )
    second = parse_legal_document(
        document_id="1-3299-24_1",
        text=text,
        metadata={"court": "constitutional_court"},
    )
    block_for_line = [
        _paragraph_for_line(first.paragraphs, start, end)
        for start, end in _line_offsets(DOC2_STRUCTURAL_LINES)
    ]
    block_ranges = []
    for paragraph in first.paragraphs:
        line_numbers = [index for index, block in enumerate(block_for_line, start=1) if block is paragraph]
        block_ranges.append((min(line_numbers), max(line_numbers)))

    assert len(DOC2_STRUCTURAL_LINES) == 13
    assert len(block_for_line) - 1 == 12
    assert len(first.paragraphs) == 11
    assert block_ranges == [
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 5),
        (6, 6),
        (7, 7),
        (8, 8),
        (9, 9),
        (10, 10),
        (11, 11),
        (12, 13),
    ]
    assert [
        _line_class(line, block_for_line[index - 1], first, index, block_for_line)
        for index, line in enumerate(DOC2_STRUCTURAL_LINES, start=1)
    ] == [
        "layout_noise",
        "metadata",
        "metadata",
        "heading",
        "heading",
        "prose_start",
        "prose_start",
        "heading",
        "prose_start",
        "instruction",
        "metadata",
        "signature",
        "signature",
    ]
    assert [_boundary_before(index, block_for_line) for index in range(2, 14)] == [
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
    assert _non_whitespace(first.reconstruct_text()) == _non_whitespace(text)
    assert len({paragraph.original_text for paragraph in first.paragraphs}) == len(first.paragraphs)
    assert [paragraph.start_offset for paragraph in first.paragraphs] == sorted(paragraph.start_offset for paragraph in first.paragraphs)
    assert [paragraph.paragraph_id for paragraph in first.paragraphs] == [paragraph.paragraph_id for paragraph in second.paragraphs]


def test_constitutional_profile_does_not_apply_to_high_court_title_pattern() -> None:
    text = "\n".join(
        [
            "NALUS - databáze rozhodnutí Ústavního soudu",
            "I.ÚS 3299/24 ze dne 20. 12. 2024",
            "USNESENÍ",
            "Ústavního soudu",
        ]
    )

    constitutional = parse_legal_document(document_id="DOC-US", text=text, metadata={"court": "constitutional_court"})
    high_court = parse_legal_document(document_id="DOC-HC", text=text, metadata={"court": "high_court_prague"})

    assert len(constitutional.paragraphs) == 3
    assert constitutional.paragraphs[2].original_text == "USNESENÍ Ústavního soudu"
    assert [paragraph.original_text for paragraph in high_court.paragraphs[-2:]] == ["USNESENÍ", "Ústavního soudu"]


def test_parser_profile_version_identifies_czech_courts_v6() -> None:
    from app.rag.legal_v2.audit import PARSER_VERSION

    assert PARSER_VERSION == "legal-decision-parser.cz-courts.v6"


def _non_whitespace(value: str) -> str:
    return re.sub(r"\s+", "", value)
