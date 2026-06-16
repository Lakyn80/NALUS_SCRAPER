from __future__ import annotations

from typing import Any

from app.nsoud.structure.patterns import (
    CLOSING_MARKERS,
    DOCUMENT_TYPE_MARKERS,
    MARKER_SPECS,
    NUMBERED_PARAGRAPH_PATTERNS,
    ODUVODNENI_MARKERS,
    POUCENI_MARKERS,
    ROMAN_MARKERS,
    ROMAN_SECTION_PATTERN,
    find_marker,
    is_valid_numbered_paragraph,
    is_valid_roman_section,
    marker_position,
    next_token,
)


def detect_roman_sections(full_text: str) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for match in ROMAN_SECTION_PATTERN.finditer(full_text):
        marker = match.group("roman")
        if marker not in ROMAN_MARKERS:
            continue
        following_token = next_token(full_text, match.end())
        if not is_valid_roman_section(following_token=following_token):
            continue
        detections.append({"marker": f"{marker}.", "position": int(match.start())})
    return detections


def detect_numbered_paragraphs(
    full_text: str,
    *,
    oduvodneni_position: int | None,
    takto_position: int | None,
    pouceni_position: int | None,
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    lower_bound = oduvodneni_position if oduvodneni_position is not None else (takto_position or 0)

    for label, pattern in NUMBERED_PARAGRAPH_PATTERNS.items():
        for match in pattern.finditer(full_text):
            number = int(match.group("num"))
            following_token = next_token(full_text, match.end())
            if not is_valid_numbered_paragraph(
                label=label,
                number=number,
                start=int(match.start()),
                following_token=following_token,
                lower_bound=lower_bound,
                pouceni_position=pouceni_position,
            ):
                continue
            detections.append(
                {
                    "marker_type": label,
                    "marker": match.group(0),
                    "number": number,
                    "position": int(match.start()),
                }
            )

    detections.sort(key=lambda item: int(item["position"]))
    return detections


def detect_ns_document_structure(
    *,
    full_text: str,
    metadata: dict | None = None,
) -> dict[str, Any]:
    text = full_text or ""
    metadata = metadata or {}
    marker_matches = {label: find_marker(text, pattern) for label, pattern in MARKER_SPECS.items()}
    oduvodneni_position = marker_position(marker_matches, *ODUVODNENI_MARKERS)
    pouceni_position = marker_position(marker_matches, *POUCENI_MARKERS)
    takto_position = marker_matches["takto:"].position
    closing_position = marker_position(marker_matches, *CLOSING_MARKERS)

    roman_sections = detect_roman_sections(text)
    numbered_paragraphs = detect_numbered_paragraphs(
        text,
        oduvodneni_position=oduvodneni_position,
        takto_position=takto_position,
        pouceni_position=pouceni_position,
    )

    section_candidates: list[dict[str, Any]] = [{"section": "header", "position": 0}]
    if takto_position is not None:
        section_candidates.append({"section": "vyrok", "position": takto_position})
    if oduvodneni_position is not None:
        section_candidates.append({"section": "oduvodneni", "position": oduvodneni_position})
    if pouceni_position is not None:
        section_candidates.append({"section": "pouceni", "position": pouceni_position})
    if closing_position is not None:
        section_candidates.append({"section": "closing/signature", "position": closing_position})
    section_candidates.sort(key=lambda item: int(item["position"]))

    observed_sections = [candidate["section"] for candidate in section_candidates]
    typical_section_order_exists = observed_sections == [
        "header",
        "vyrok",
        "oduvodneni",
        "pouceni",
        "closing/signature",
    ]

    return {
        "full_text_length": len(text),
        "detected_markers": {
            label: {
                "present": match.present,
                "position": match.position,
            }
            for label, match in marker_matches.items()
        },
        "detected_section_order": {
            "observed_sections": observed_sections,
            "typical_section_order_exists": typical_section_order_exists,
        },
        "detected_numbered_paragraph_count": len(numbered_paragraphs),
        "detected_numbered_paragraph_markers": numbered_paragraphs,
        "detected_roman_section_count": len(roman_sections),
        "detected_roman_sections": roman_sections,
        "section_candidates": section_candidates,
        "marker_flags": {
            "document_type": any(marker_matches[label].present for label in DOCUMENT_TYPE_MARKERS),
            "nejvyssi_soud_rozhodl": marker_matches["Nejvyšší soud rozhodl"].present,
            "takto": marker_matches["takto:"].present,
            "oduvodneni": oduvodneni_position is not None,
            "pouceni": pouceni_position is not None,
            "closing": closing_position is not None,
            "roman_sections": bool(roman_sections),
            "numbered_paragraphs": bool(numbered_paragraphs),
        },
        "metadata": metadata,
    }
