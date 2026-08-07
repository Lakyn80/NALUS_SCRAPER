"""Centralized deterministic sentence scoring policy for extractive briefs."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.rag.legal_v2.query_input.boilerplate import is_boilerplate_sentence
from app.rag.legal_v2.query_input.identifiers import count_identifiers

_PROCEDURAL_RE = re.compile(
    r"(?i)\b(?:"
    r"ústavní\s+stížnost|odvolání|dovolání|žalob[ay]|správní\s+žalob|"
    r"insolvenční|exekučn|vazební|vazba|přípustnost|odmítnut"
    r")\b"
)
_DEFECT_RE = re.compile(
    r"(?i)\b(?:"
    r"advokát|právní(?:ho)?\s+zastoupení|formáln\w+\s+vad|"
    r"neodstraněn|chybí\s+odůvodnění|nedostatečn\w+\s+odůvodnění|"
    r"opožděn|nedostatek\s+aktivní\s+legitimace|prekluz|promlčen"
    r")\b"
)
_OUTCOME_RE = re.compile(
    r"(?i)\b(?:"
    r"odmítá\s+se|zamítá\s+se|zrušuje\s+se|vyhovuje\s+se|"
    r"navrhuje\s+zrušení|domáhá\s+zrušení"
    r")\b"
)
_ISSUE_RE = re.compile(
    r"(?i)\b(?:"
    r"promlčen|prekluz|náklad\w+\s+řízení|výpověď|nájem|"
    r"občanství|azyl|mezinárodní\s+ochran|únos|výživn|"
    r"svěření|úprav[ay]\s+styk|diskriminac"
    r")\b"
)
_FACT_RE = re.compile(
    r"(?i)\b(?:"
    r"nezletil|matk[ay]|otec|zaměstnavatel|zaměstnanec|"
    r"věřitel|dlužník|stěžovatel|žalobc|žalovan|"
    r"policie|orgán\s+sociálně"
    r")\b"
)
_NEGATION_RE = re.compile(
    r"(?i)\b(?:"
    r"nehledám|nechci|neřeším|nejde\s+mi\s+o|nejde\s+o|"
    r"nikoli|nýbrž|nybrz|ale\s+(?:o\s+)?|"
    r"pouze|jen\b|nikoliv"
    r")\b"
)
_REQUEST_RE = re.compile(r"(?i)\b(?:hledám|hledam|potřebuji|chci\s+najít|zajímá\s+mě)\b")


@dataclass(frozen=True)
class ScoringWeights:
    procedural: float = 3.0
    defect: float = 4.0
    outcome: float = 3.5
    issue: float = 3.0
    fact: float = 2.0
    negation: float = 5.0
    request: float = 3.5
    boilerplate_penalty: float = -8.0
    identifier_only_penalty: float = -6.0
    low_info_penalty: float = -2.0


DEFAULT_WEIGHTS = ScoringWeights()


def score_sentence(text: str, *, weights: ScoringWeights = DEFAULT_WEIGHTS) -> tuple[float, tuple[str, ...]]:
    cleaned = (text or "").strip()
    flags: list[str] = []
    score = 0.0

    if is_boilerplate_sentence(cleaned):
        score += weights.boilerplate_penalty
        flags.append("boilerplate")

    if _PROCEDURAL_RE.search(cleaned):
        score += weights.procedural
        flags.append("procedural")
    if _DEFECT_RE.search(cleaned):
        score += weights.defect
        flags.append("defect")
    if _OUTCOME_RE.search(cleaned):
        score += weights.outcome
        flags.append("outcome")
    if _ISSUE_RE.search(cleaned):
        score += weights.issue
        flags.append("issue")
    if _FACT_RE.search(cleaned):
        score += weights.fact
        flags.append("fact")
    if _NEGATION_RE.search(cleaned):
        score += weights.negation
        flags.append("negation")
    if _REQUEST_RE.search(cleaned):
        score += weights.request
        flags.append("request")

    id_count = count_identifiers(cleaned)
    if id_count and len(re.sub(r"\W+", "", cleaned)) < 40:
        score += weights.identifier_only_penalty
        flags.append("identifier_only")
    if len(cleaned) < 40:
        score += weights.low_info_penalty
        flags.append("low_info")

    # Mild length preference for informative mid-length sentences.
    if 60 <= len(cleaned) <= 260:
        score += 0.5

    return score, tuple(flags)


def extract_negative_focus(text: str) -> list[str]:
    """Pull negated scopes from contrast constructions."""
    cleaned = text or ""
    patterns = [
        re.compile(
            r"(?i)(?:nehledám|nechci|neřeším|nejde\s+mi\s+o|nejde\s+o|nikoli)\s+"
            r"(.+?)(?:,?\s+(?:ale|nýbrž|nybrz)\b|$|[.!?])"
        ),
        re.compile(r"(?i)pouze\s+.+?,\s*ne\s+(.+?)(?=$|[.!?])"),
    ]
    found: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(cleaned):
            body = re.sub(r"\s+", " ", match.group(1)).strip(" ,;")
            if len(body) >= 8:
                found.append(body[:160])
    return found


def extract_requested_focus(text: str) -> list[str]:
    cleaned = text or ""
    patterns = [
        re.compile(
            r"(?i)(?:ale\s+)?(?:hledám|hledam|potřebuji)\s+(?:jen|pouze|ted|nyní)?\s*(.+?)(?=$|[.!?])"
        ),
        re.compile(r"(?i)(?:ale|nýbrž|nybrz)\s+(.+?)(?=$|[.!?])"),
    ]
    found: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(cleaned):
            body = re.sub(r"\s+", " ", match.group(1)).strip(" ,;")
            if len(body) >= 8:
                found.append(body[:180])
    return found
