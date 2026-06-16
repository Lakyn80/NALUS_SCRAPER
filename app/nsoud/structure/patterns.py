from __future__ import annotations

import re
from dataclasses import dataclass


ROMAN_MARKERS = [
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "VII",
    "VIII",
    "IX",
    "X",
    "XI",
    "XII",
    "XIII",
    "XIV",
    "XV",
    "XVI",
    "XVII",
    "XVIII",
    "XIX",
    "XX",
]
DOCUMENT_TYPE_MARKERS = ("ROZSUDEK", "USNESENÍ", "STANOVISKO")
ODUVODNENI_MARKERS = ("Odůvodnění:", "O d ů v o d n ě n í:")
POUCENI_MARKERS = ("Poučení:", "P o u č e n í:")
CLOSING_MARKERS = ("V Brně dne", "předseda senátu", "předsedkyně senátu")
MAIN_SECTION_MARKERS = (
    "takto:",
    *ODUVODNENI_MARKERS,
    *POUCENI_MARKERS,
    *CLOSING_MARKERS,
)
STRUCTURE_PATTERN_LABELS = [
    ("document_type", "DTYPE"),
    ("nejvyssi_soud_rozhodl", "NSR"),
    ("takto", "TAKTO"),
    ("oduvodneni", "ODUV"),
    ("pouceni", "POUC"),
    ("closing", "CLOSE"),
    ("roman_sections", "ROMAN"),
    ("numbered_paragraphs", "NUM"),
]

ROMAN_SECTION_PATTERN = re.compile(
    r"(?:(?<=^)|(?<=[\s:;(\[]))"
    r"(?P<roman>XX|XIX|XVIII|XVII|XVI|XV|XIV|XIII|XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)\."
    r"(?=\s)",
)
NUMBERED_PARAGRAPH_PATTERNS = {
    "dot": re.compile(r"(?:(?<=^)|(?<=[\s:;]))(?P<num>[1-9][0-9]{0,2})\.(?=\s)"),
    "slash": re.compile(r"(?:(?<=^)|(?<=[\s:;]))(?P<num>[1-9][0-9]{0,2})/(?=\s)"),
    "bracket": re.compile(r"(?:(?<=^)|(?<=[\s:;]))\[(?P<num>[1-9][0-9]{0,2})\](?=\s)"),
    "paren": re.compile(r"(?:(?<=^)|(?<=[\s:;]))(?P<num>[1-9][0-9]{0,2})\)(?=\s)"),
}
MARKER_SPECS = {
    "ROZSUDEK": re.compile(r"\bROZSUDEK\b"),
    "USNESENÍ": re.compile(r"\bUSNESENÍ\b"),
    "STANOVISKO": re.compile(r"\bSTANOVISKO\b"),
    "JMÉNEM REPUBLIKY": re.compile(r"JMÉNEM\s+REPUBLIKY", re.IGNORECASE),
    "Nejvyšší soud rozhodl": re.compile(r"Nejvyšší\s+soud\s+rozhodl", re.IGNORECASE),
    "takto:": re.compile(r"\btakto\s*:", re.IGNORECASE),
    "Odůvodnění:": re.compile(r"Odůvodnění\s*:", re.IGNORECASE),
    "O d ů v o d n ě n í:": re.compile(r"O\s+d\s+ů\s+v\s+o\s+d\s+n\s+ě\s+n\s+í\s*:", re.IGNORECASE),
    "Poučení:": re.compile(r"Poučení\s*:", re.IGNORECASE),
    "P o u č e n í:": re.compile(r"P\s+o\s+u\s+č\s+e\s+n\s+í\s*:", re.IGNORECASE),
    "V Brně dne": re.compile(r"V\s+Brně\s+dne", re.IGNORECASE),
    "předseda senátu": re.compile(r"předseda\s+senátu", re.IGNORECASE),
    "předsedkyně senátu": re.compile(r"předsedkyně\s+senátu", re.IGNORECASE),
}


@dataclass(frozen=True)
class MarkerMatch:
    present: bool
    position: int | None


def find_marker(text: str, pattern: re.Pattern[str]) -> MarkerMatch:
    match = pattern.search(text)
    if not match:
        return MarkerMatch(present=False, position=None)
    return MarkerMatch(present=True, position=int(match.start()))


def marker_position(matches: dict[str, MarkerMatch], *labels: str) -> int | None:
    positions = [matches[label].position for label in labels if matches[label].position is not None]
    if not positions:
        return None
    return min(positions)


def next_non_space_index(text: str, index: int) -> int | None:
    for position in range(index, len(text)):
        if not text[position].isspace():
            return position
    return None


def next_token(text: str, index: int) -> str:
    next_index = next_non_space_index(text, index)
    if next_index is None:
        return ""
    match = re.match(r"[A-Za-zÁ-Žá-ž0-9]+", text[next_index:])
    if not match:
        return text[next_index]
    return match.group(0)


def is_uppercase_token_start(token: str) -> bool:
    if not token:
        return False
    first = token[0]
    return first.isalpha() and first.upper() == first


def is_valid_roman_section(*, following_token: str) -> bool:
    if following_token == "ÚS":
        return False
    if following_token and following_token[0].isdigit():
        return False
    if following_token and following_token[0].islower():
        return False
    return True


def is_valid_numbered_paragraph(
    *,
    label: str,
    number: int,
    start: int,
    following_token: str,
    lower_bound: int,
    pouceni_position: int | None,
) -> bool:
    if number > 200:
        return False
    if label == "paren" and start < lower_bound:
        return False
    if not following_token:
        return False
    if following_token[0].isdigit():
        return False
    if label in {"dot", "slash", "bracket"} and not (
        is_uppercase_token_start(following_token) or following_token[0] in {"„", "\"", "(", "["}
    ):
        return False
    if label == "paren" and not (
        is_uppercase_token_start(following_token) or following_token[0] in {"„", "\"", "("}
    ):
        return False
    if pouceni_position is not None and start > pouceni_position and label != "dot":
        return False
    return True
