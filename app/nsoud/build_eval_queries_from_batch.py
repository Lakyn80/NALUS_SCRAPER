from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("pandas is required for this script.") from exc

try:
    import pyarrow  # noqa: F401
except ImportError:
    pyarrow = None


MIN_QUERY_COUNT = 15
MAX_QUERY_COUNT = 25
TARGET_QUERY_COUNT = 20
MAX_SOURCE_ITEMS = 5
TOP_EXTRACTED_TERMS_LIMIT = 15

REQUIRED_DOCUMENT_COLUMNS = {
    "case_number",
    "document_type",
    "legal_area",
    "title",
    "full_text",
}
REQUIRED_CHUNK_COLUMNS = {
    "case_number",
    "document_type",
    "legal_area",
    "chunk_id",
    "chunk_text",
    "ns_section_hint",
}

STOPWORDS = {
    "a",
    "aby",
    "ale",
    "ani",
    "ano",
    "asi",
    "bez",
    "by",
    "byl",
    "byla",
    "byli",
    "bylo",
    "byly",
    "co",
    "dále",
    "další",
    "dle",
    "dne",
    "do",
    "i",
    "jako",
    "je",
    "jeho",
    "její",
    "jejich",
    "jejím",
    "jen",
    "ji",
    "jsou",
    "k",
    "ke",
    "která",
    "které",
    "který",
    "kterých",
    "kterým",
    "když",
    "lze",
    "má",
    "mají",
    "mezi",
    "na",
    "nad",
    "ne",
    "nebo",
    "nebyl",
    "nejsou",
    "není",
    "než",
    "o",
    "od",
    "opakovaně",
    "pak",
    "po",
    "pod",
    "podle",
    "pokud",
    "poté",
    "pouze",
    "pro",
    "proto",
    "proti",
    "před",
    "při",
    "s",
    "se",
    "si",
    "strany",
    "své",
    "svých",
    "svůj",
    "tak",
    "také",
    "takto",
    "tato",
    "ten",
    "tento",
    "této",
    "to",
    "tohoto",
    "tomto",
    "toto",
    "u",
    "už",
    "v",
    "ve",
    "věci",
    "však",
    "z",
    "za",
    "ze",
    "že",
    "č",
    "česká",
    "české",
    "české republiky",
    "český",
    "j",
    "judr",
    "mgr",
    "ing",
    "ph",
    "phd",
    "republiky",
    "sb",
    "sp",
    "zn",
    "ř",
    "o",
    "s",
}

ANCHOR_TOKENS = {
    "265b",
    "265i",
    "243c",
    "237",
    "bydlení",
    "dovolací",
    "dovolání",
    "exekuce",
    "exekučního",
    "nákladů",
    "nájemného",
    "obohacení",
    "obrana",
    "obrany",
    "odmítnutí",
    "odnětí",
    "povinného",
    "právo",
    "přípustnost",
    "příslušnosti",
    "příslušnost",
    "rodinný",
    "skryté",
    "svobody",
    "trest",
    "trestného",
    "vady",
    "vady",
    "zastavení",
    "zjevně",
}

GENERIC_PHRASES = {
    "dovolací důvod",
    "hodnocení důkazů",
    "napadené rozhodnutí",
    "obsahem provedených důkazů",
    "opravný prostředek",
    "poučení proti tomuto",
    "prvního stupně",
    "proti rozsudku",
    "rozhodnutí odvolacího",
    "rozhodovací praxe",
    "rozsudek prvního stupně",
    "skutková zjištění",
    "státní zástupce",
    "trestného činu",
    "usnesení nejvyššího",
    "ustálené rozhodovací praxe",
}

PRIORITY_THEMES = [
    "criminal",
    "civil",
    "dovolani",
    "pripustnost_dovolani",
    "odmitnuti_dovolani",
    "exekuce",
    "mistni_prislusnost",
    "vady",
    "nutna_obrana",
    "naklady_rizeni",
]

OPTIONAL_THEMES = [
    "bydleni",
    "obohaceni",
    "trest",
    "criminal_dovolani",
]


@dataclass(frozen=True)
class ThemePattern:
    label: str
    regex: str
    base_score: int
    preferred_query: str | None = None
    extra_terms: tuple[str, ...] = ()


@dataclass
class EvidenceRow:
    case_number: str
    document_type: str
    legal_area: str
    chunk_id: str
    ns_section_hint: str
    chunk_text: str


@dataclass
class CandidateQuery:
    query: str
    normalized_query: str
    score: float = 0.0
    pattern_hits: int = 0
    source_terms: set[str] = field(default_factory=set)
    source_case_numbers: set[str] = field(default_factory=set)
    source_chunk_ids: set[str] = field(default_factory=set)
    legal_areas: Counter = field(default_factory=Counter)
    document_types: Counter = field(default_factory=Counter)
    themes: set[str] = field(default_factory=set)
    chunk_count: int = 0
    document_count: int = 0


THEME_PATTERNS = [
    ThemePattern(
        label="mistni_prislusnost",
        regex=r"určení místní příslušnosti(?: soudu| exekučního soudu)?",
        base_score=48,
    ),
    ThemePattern(
        label="mistni_prislusnost",
        regex=r"místní příslušnosti chybějí nebo je nelze zjistit",
        base_score=46,
    ),
    ThemePattern(
        label="exekuce",
        regex=r"zastavení exekuce",
        base_score=38,
    ),
    ThemePattern(
        label="exekuce",
        regex=r"pověření a nařízení exekuce",
        base_score=36,
    ),
    ThemePattern(
        label="naklady_rizeni",
        regex=r"náhradě nákladů dovolacího řízení",
        base_score=42,
        preferred_query="náhradě nákladů dovolacího řízení",
    ),
    ThemePattern(
        label="pripustnost_dovolani",
        regex=r"přípustnost dovolání(?: podle § 237 o\.?\s*s\.?\s*ř\.?)?",
        base_score=50,
        preferred_query="přípustnost dovolání podle § 237 o. s. ř.",
    ),
    ThemePattern(
        label="odmitnuti_dovolani",
        regex=r"odmítnutí dovolání",
        base_score=43,
    ),
    ThemePattern(
        label="odmitnuti_dovolani",
        regex=r"zjevně neopodstatněné",
        base_score=39,
        preferred_query="zjevně neopodstatněné dovolání",
        extra_terms=("dovolání",),
    ),
    ThemePattern(
        label="vady",
        regex=r"odpovědnosti za vady jako slevy z kupní ceny",
        base_score=78,
    ),
    ThemePattern(
        label="vady",
        regex=r"skryté vady",
        base_score=140,
    ),
    ThemePattern(
        label="vady",
        regex=r"rodinný dům",
        base_score=8,
    ),
    ThemePattern(
        label="bydleni",
        regex=r"právo bydlení",
        base_score=60,
    ),
    ThemePattern(
        label="obohaceni",
        regex=r"bezdůvodné obohacení za užívání bytu",
        base_score=64,
    ),
    ThemePattern(
        label="nutna_obrana",
        regex=r"nutná obrana(?: v úvahu v případech vzájemného napadání)?",
        base_score=52,
        preferred_query="nutná obrana vzájemné napadání",
    ),
    ThemePattern(
        label="criminal_dovolani",
        regex=r"dovolací důvod podle § 265b odst\. 1 písm\. [a-z]\)",
        base_score=50,
    ),
    ThemePattern(
        label="trest",
        regex=r"trest odnětí svobody",
        base_score=34,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic NSoud evaluation queries from a local batch.")
    parser.add_argument("--documents", type=Path, required=True, help="Input NSoud documents Parquet path.")
    parser.add_argument("--chunks", type=Path, required=True, help="Input NSoud chunks Parquet path.")
    parser.add_argument("--out-json", type=Path, required=True, help="Output JSON path.")
    parser.add_argument("--out-md", type=Path, required=True, help="Output Markdown report path.")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def slugify_text(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", ascii_text.lower()).strip()


def normalize_query_text(text: str) -> str:
    cleaned = collapse_spaces(text)
    cleaned = re.sub(r"^[,;:\-–—]+", "", cleaned).strip()
    return cleaned


def tokenize(text: str) -> list[str]:
    return re.findall(r"[0-9a-zá-ž]+", text.lower())


def normalize_phrase_key(text: str) -> str:
    simplified = slugify_text(text)
    parts = [part for part in simplified.split() if part not in {"na", "o", "u", "v", "ve", "z", "za"}]
    return " ".join(parts)


def ensure_columns(frame: pd.DataFrame, *, required: set[str], label: str) -> None:
    missing = sorted(column for column in required if column not in frame.columns)
    if missing:
        missing_display = ", ".join(missing)
        raise ValueError(f"{label} input is missing required columns: {missing_display}")


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def majority_label(counter: Counter) -> str:
    if not counter:
        return ""
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]


def natural_sort(values: Iterable[str]) -> list[str]:
    def sort_key(value: str) -> tuple[Any, ...]:
        parts = re.split(r"(\d+)", value)
        normalized_parts: list[Any] = []
        for part in parts:
            if part.isdigit():
                normalized_parts.append(int(part))
            else:
                normalized_parts.append(part.lower())
        return tuple(normalized_parts)

    return sorted((value for value in values if value), key=sort_key)


def phrase_tokens(text: str) -> list[str]:
    return [stem_token(token) for token in tokenize(text) if token]


def stem_token(token: str) -> str:
    simplified = slugify_text(token).replace(" ", "")
    if len(simplified) <= 4:
        return simplified
    for suffix in ("ami", "emi", "ove", "ovi", "eho", "ich", "ich", "imi", "ymi", "eni", "ani", "osti"):
        if simplified.endswith(suffix) and len(simplified) - len(suffix) >= 4:
            return simplified[: -len(suffix)]
    while len(simplified) > 5 and simplified.endswith(("a", "e", "i", "o", "u", "y")):
        simplified = simplified[:-1]
    return simplified


def is_meaningful_ngram(tokens: list[str]) -> bool:
    if not tokens:
        return False
    joined = " ".join(tokens)
    if joined in GENERIC_PHRASES:
        return False
    if all(token in STOPWORDS for token in tokens):
        return False
    if not any(token in ANCHOR_TOKENS or len(token) >= 6 for token in tokens):
        return False
    if not any(token in ANCHOR_TOKENS for token in tokens):
        return False
    non_stop = [token for token in tokens if token not in STOPWORDS]
    if len(non_stop) < 2:
        return False
    if sum(token.isdigit() for token in tokens) > 1:
        return False
    return True


def collect_evidence_rows(chunks_df: pd.DataFrame) -> list[EvidenceRow]:
    rows: list[EvidenceRow] = []
    for _, row in chunks_df.iterrows():
        rows.append(
            EvidenceRow(
                case_number=normalize_text(row.get("case_number")),
                document_type=normalize_text(row.get("document_type")),
                legal_area=normalize_text(row.get("legal_area")),
                chunk_id=normalize_text(row.get("chunk_id")),
                ns_section_hint=normalize_text(row.get("ns_section_hint")),
                chunk_text=normalize_text(row.get("chunk_text")),
            )
        )
    return rows


def infer_themes(query: str, legal_area: str) -> set[str]:
    normalized = slugify_text(query)
    themes: set[str] = set()
    if legal_area:
        themes.add(legal_area)
    if "dovol" in normalized:
        themes.add("dovolani")
    if "pripustnost" in normalized or "237" in normalized:
        themes.add("pripustnost_dovolani")
    if "odmit" in normalized or "neopodstat" in normalized:
        themes.add("odmitnuti_dovolani")
    if "exekuc" in normalized:
        themes.add("exekuce")
    if "prislusnost" in normalized:
        themes.add("mistni_prislusnost")
    if "vad" in normalized or "rodinny dum" in normalized:
        themes.add("vady")
    if "obrana" in normalized:
        themes.add("nutna_obrana")
    if "naklad" in normalized:
        themes.add("naklady_rizeni")
    if "bydlen" in normalized:
        themes.add("bydleni")
    if "obohacen" in normalized:
        themes.add("obohaceni")
    if "trest" in normalized or "265b" in normalized or "svobody" in normalized:
        themes.add("criminal")
    return themes


def add_candidate(
    candidates: dict[str, CandidateQuery],
    *,
    query: str,
    evidence: EvidenceRow,
    source_terms: Iterable[str],
    score_delta: float,
    extra_themes: Iterable[str] = (),
    pattern_based: bool = False,
) -> None:
    cleaned_query = normalize_query_text(query)
    if not cleaned_query:
        return
    normalized_query = normalize_phrase_key(cleaned_query)
    if not normalized_query:
        return
    candidate = candidates.get(normalized_query)
    if candidate is None:
        candidate = CandidateQuery(query=cleaned_query, normalized_query=normalized_query)
        candidates[normalized_query] = candidate

    candidate.score += score_delta
    if pattern_based:
        candidate.pattern_hits += 1
    candidate.chunk_count += 1
    candidate.source_case_numbers.add(evidence.case_number)
    candidate.source_chunk_ids.add(evidence.chunk_id)
    if evidence.legal_area:
        candidate.legal_areas[evidence.legal_area] += 1
    if evidence.document_type:
        candidate.document_types[evidence.document_type] += 1
    candidate.themes.update(infer_themes(cleaned_query, evidence.legal_area))
    candidate.themes.update(extra_themes)
    for term in source_terms:
        cleaned_term = normalize_query_text(term)
        if cleaned_term:
            candidate.source_terms.add(cleaned_term)


def extract_pattern_candidates(rows: list[EvidenceRow]) -> dict[str, CandidateQuery]:
    candidates: dict[str, CandidateQuery] = {}
    compiled_patterns = [(pattern, re.compile(pattern.regex, flags=re.IGNORECASE)) for pattern in THEME_PATTERNS]

    for row in rows:
        text = row.chunk_text
        if not text:
            continue
        for pattern, compiled in compiled_patterns:
            for match in compiled.finditer(text):
                matched_text = normalize_query_text(match.group(0))
                if not matched_text:
                    continue
                query_text = pattern.preferred_query or matched_text
                source_terms = [matched_text, *pattern.extra_terms]
                score_delta = pattern.base_score
                if row.ns_section_hint in {"header", "vyrok"}:
                    score_delta += 4
                add_candidate(
                    candidates,
                    query=query_text,
                    evidence=row,
                    source_terms=source_terms,
                    score_delta=score_delta,
                    extra_themes={pattern.label},
                    pattern_based=True,
                )
    return candidates


def extract_ngram_candidates(rows: list[EvidenceRow]) -> tuple[dict[str, CandidateQuery], Counter]:
    ngram_stats: dict[str, dict[str, Any]] = {}
    top_terms = Counter()

    for row in rows:
        combined = " ".join(
            part
            for part in [
                row.case_number,
                row.document_type,
                row.legal_area,
                row.ns_section_hint,
                row.chunk_text,
            ]
            if part
        )
        tokens = tokenize(combined)
        filtered_tokens: list[str] = []
        for token in tokens:
            if token in STOPWORDS:
                continue
            if len(token) < 3 and token not in {"§", "os", "tr"}:
                continue
            if token.isdigit() and token not in {"237"}:
                continue
            filtered_tokens.append(token)

        for n in range(2, 6):
            if len(filtered_tokens) < n:
                continue
            for start in range(len(filtered_tokens) - n + 1):
                gram_tokens = filtered_tokens[start : start + n]
                if not is_meaningful_ngram(gram_tokens):
                    continue
                phrase = " ".join(gram_tokens)
                if phrase in GENERIC_PHRASES:
                    continue
                stat = ngram_stats.setdefault(
                    phrase,
                    {
                        "chunk_ids": set(),
                        "case_numbers": set(),
                        "legal_areas": Counter(),
                        "document_types": Counter(),
                        "section_hints": Counter(),
                    },
                )
                stat["chunk_ids"].add(row.chunk_id)
                stat["case_numbers"].add(row.case_number)
                if row.legal_area:
                    stat["legal_areas"][row.legal_area] += 1
                if row.document_type:
                    stat["document_types"][row.document_type] += 1
                if row.ns_section_hint:
                    stat["section_hints"][row.ns_section_hint] += 1

    candidates: dict[str, CandidateQuery] = {}
    for phrase, stat in ngram_stats.items():
        chunk_count = len(stat["chunk_ids"])
        document_count = len(stat["case_numbers"])
        if chunk_count == 0 or document_count == 0:
            continue
        if chunk_count == 1 and document_count == 1 and len(phrase.split()) < 3:
            continue
        score = (chunk_count * 3.0) + (document_count * 4.0) + (len(phrase.split()) * 0.5)
        if "header" in stat["section_hints"]:
            score += 2.0
        if "vyrok" in stat["section_hints"]:
            score += 1.0
        top_terms[phrase] = score

        evidence = EvidenceRow(
            case_number=natural_sort(stat["case_numbers"])[0],
            document_type=majority_label(stat["document_types"]),
            legal_area=majority_label(stat["legal_areas"]),
            chunk_id=natural_sort(stat["chunk_ids"])[0],
            ns_section_hint=majority_label(stat["section_hints"]),
            chunk_text="",
        )
        add_candidate(
            candidates,
            query=phrase,
            evidence=evidence,
            source_terms=[phrase],
            score_delta=score,
        )
        candidate = candidates[normalize_phrase_key(phrase)]
        candidate.chunk_count = chunk_count
        candidate.document_count = document_count
        candidate.source_case_numbers = set(stat["case_numbers"])
        candidate.source_chunk_ids = set(stat["chunk_ids"])
        candidate.legal_areas = stat["legal_areas"]
        candidate.document_types = stat["document_types"]
    return candidates, top_terms


def merge_candidates(*candidate_groups: dict[str, CandidateQuery]) -> dict[str, CandidateQuery]:
    merged: dict[str, CandidateQuery] = {}
    for group in candidate_groups:
        for key, candidate in group.items():
            current = merged.get(key)
            if current is None:
                merged[key] = candidate
                continue
            current.score += candidate.score
            current.pattern_hits += candidate.pattern_hits
            current.chunk_count = max(current.chunk_count, candidate.chunk_count)
            current.document_count = max(current.document_count, candidate.document_count)
            current.source_terms.update(candidate.source_terms)
            current.source_case_numbers.update(candidate.source_case_numbers)
            current.source_chunk_ids.update(candidate.source_chunk_ids)
            current.legal_areas.update(candidate.legal_areas)
            current.document_types.update(candidate.document_types)
            current.themes.update(candidate.themes)
            if len(candidate.query) > len(current.query):
                current.query = candidate.query
    return merged


def compute_document_counts(candidates: dict[str, CandidateQuery]) -> None:
    for candidate in candidates.values():
        candidate.document_count = len(candidate.source_case_numbers)
        candidate.chunk_count = len(candidate.source_chunk_ids)
        candidate.score += candidate.document_count * 1.5
        candidate.score += min(candidate.chunk_count, 8) * 0.5
        if candidate.document_count > 1:
            candidate.score += 3.0


def is_near_duplicate(left: CandidateQuery, right: CandidateQuery) -> bool:
    left_tokens = set(phrase_tokens(left.normalized_query))
    right_tokens = set(phrase_tokens(right.normalized_query))
    if not left_tokens or not right_tokens:
        return False
    if left.normalized_query == right.normalized_query:
        return True
    overlap = len(left_tokens & right_tokens)
    min_size = min(len(left_tokens), len(right_tokens))
    max_size = max(len(left_tokens), len(right_tokens))
    if min_size == 0:
        return False
    if left.normalized_query in right.normalized_query or right.normalized_query in left.normalized_query:
        return overlap >= max(2, min_size - 1)
    return (overlap / min_size) >= 0.85 and (overlap / max_size) >= 0.7


def sorted_candidates(candidates: dict[str, CandidateQuery]) -> list[CandidateQuery]:
    return sorted(
        candidates.values(),
        key=lambda item: (
            -item.pattern_hits,
            -item.score,
            -item.document_count,
            -item.chunk_count,
            item.query.lower(),
        ),
    )


def is_selection_safe(candidate: CandidateQuery) -> bool:
    alpha_tokens = [token for token in tokenize(candidate.query) if re.search(r"[a-zá-ž]", token)]
    if len(alpha_tokens) < 2:
        return False
    if normalize_phrase_key(candidate.query) in {normalize_phrase_key(item) for item in GENERIC_PHRASES}:
        return False
    if len(candidate.query) < 10:
        return False
    if candidate.pattern_hits == 0 and len(alpha_tokens) < 3:
        return False
    if candidate.document_count == 1 and candidate.chunk_count == 1 and len(candidate.query.split()) < 3:
        return False
    return True


def select_best_for_theme(candidates: list[CandidateQuery], selected: list[CandidateQuery], theme: str) -> None:
    for candidate in candidates:
        if not is_selection_safe(candidate):
            continue
        if theme not in candidate.themes:
            continue
        if any(is_near_duplicate(candidate, existing) for existing in selected):
            continue
        selected.append(candidate)
        return


def select_queries(candidates: dict[str, CandidateQuery]) -> list[CandidateQuery]:
    ranked = [candidate for candidate in sorted_candidates(candidates) if is_selection_safe(candidate)]
    ranked_pattern = [candidate for candidate in ranked if candidate.pattern_hits > 0]
    selected: list[CandidateQuery] = []

    for theme in PRIORITY_THEMES:
        select_best_for_theme(ranked_pattern, selected, theme)

    for theme in OPTIONAL_THEMES:
        select_best_for_theme(ranked_pattern, selected, theme)

    for area in ("criminal", "civil"):
        select_best_for_theme(ranked_pattern, selected, area)

    for candidate in ranked_pattern:
        if len(selected) >= TARGET_QUERY_COUNT:
            break
        if any(is_near_duplicate(candidate, existing) for existing in selected):
            continue
        theme_counts = Counter(theme for item in selected for theme in item.themes)
        if any(theme_counts[theme] >= 4 for theme in candidate.themes if theme in PRIORITY_THEMES):
            continue
        selected.append(candidate)

    if len(selected) < MIN_QUERY_COUNT:
        for candidate in ranked:
            if len(selected) >= MIN_QUERY_COUNT:
                break
            if any(is_near_duplicate(candidate, existing) for existing in selected):
                continue
            selected.append(candidate)

    return selected[:MAX_QUERY_COUNT]


def validate_selected_queries(selected: list[CandidateQuery]) -> list[str]:
    errors: list[str] = []
    if not selected:
        errors.append("Zero queries were generated.")
    for candidate in selected:
        if not candidate.source_terms:
            errors.append(f"Query has no source_terms: {candidate.query}")
        if not candidate.source_case_numbers:
            errors.append(f"Query has no source_case_numbers: {candidate.query}")
        if not candidate.source_chunk_ids:
            errors.append(f"Query has no source_chunk_ids: {candidate.query}")
    return errors


def candidate_reason(candidate: CandidateQuery) -> str:
    area = majority_label(candidate.legal_areas) or "unknown"
    doc_type = majority_label(candidate.document_types) or "unknown"
    return (
        f"Extracted from {candidate.chunk_count} chunks across {candidate.document_count} documents; "
        f"dominant context is {area} / {doc_type}."
    )


def json_payload(
    *,
    status: str,
    created_at: str,
    documents_path: Path,
    chunks_path: Path,
    total_documents: int,
    total_chunks: int,
    selected: list[CandidateQuery],
) -> dict[str, Any]:
    return {
        "status": status,
        "created_at": created_at,
        "input_documents": str(documents_path),
        "input_chunks": str(chunks_path),
        "total_documents": total_documents,
        "total_chunks": total_chunks,
        "queries": [
            {
                "query": candidate.query,
                "reason": candidate_reason(candidate),
                "expected_legal_area": majority_label(candidate.legal_areas),
                "expected_document_type": majority_label(candidate.document_types),
                "source_terms": natural_sort(candidate.source_terms)[:MAX_SOURCE_ITEMS],
                "source_case_numbers": natural_sort(candidate.source_case_numbers)[:MAX_SOURCE_ITEMS],
                "source_chunk_ids": natural_sort(candidate.source_chunk_ids)[:MAX_SOURCE_ITEMS],
            }
            for candidate in selected
        ],
    }


def format_distribution(counter: Counter) -> str:
    if not counter:
        return "- none"
    return ", ".join(f"{label or '(empty)'}: {count}" for label, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def build_markdown_report(
    *,
    status: str,
    documents_path: Path,
    chunks_path: Path,
    total_documents: int,
    total_chunks: int,
    legal_area_distribution: Counter,
    document_type_distribution: Counter,
    top_terms: list[tuple[str, float]],
    selected: list[CandidateQuery],
    notes: list[str],
) -> str:
    lines = [
        "# NSoud Generated Evaluation Queries",
        "",
        f"- Status: **{status}**",
        f"- Documents input: `{documents_path}`",
        f"- Chunks input: `{chunks_path}`",
        f"- Total documents: **{total_documents}**",
        f"- Total chunks: **{total_chunks}**",
        f"- Generated query count: **{len(selected)}**",
        f"- Legal area distribution: {format_distribution(legal_area_distribution)}",
        f"- Document type distribution: {format_distribution(document_type_distribution)}",
        "",
        "## Top Extracted Legal Terms",
        "",
    ]

    if top_terms:
        for term, score in top_terms:
            lines.append(f"- `{term}` (score {score:.1f})")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Generated Queries",
            "",
            "| query | reason | expected_legal_area | expected_document_type | source_case_numbers | source_terms |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for candidate in selected:
        case_numbers = ", ".join(natural_sort(candidate.source_case_numbers)[:MAX_SOURCE_ITEMS]) or "-"
        source_terms = ", ".join(natural_sort(candidate.source_terms)[:MAX_SOURCE_ITEMS]) or "-"
        lines.append(
            f"| {candidate.query} | {candidate_reason(candidate)} | "
            f"{majority_label(candidate.legal_areas) or '-'} | {majority_label(candidate.document_types) or '-'} | "
            f"{case_numbers} | {source_terms} |"
        )

    lines.extend(["", "## Notes"])
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- None.")

    return "\n".join(lines)


def determine_status(*, selected_count: int, errors: list[str]) -> str:
    if errors:
        return "FAIL"
    if selected_count < MIN_QUERY_COUNT:
        return "WARN"
    return "PASS"


def build_top_terms(candidates: dict[str, CandidateQuery]) -> list[tuple[str, float]]:
    return [
        (candidate.query, candidate.score)
        for candidate in sorted_candidates(candidates)
        if is_selection_safe(candidate)
    ][:TOP_EXTRACTED_TERMS_LIMIT]


def main() -> int:
    args = parse_args()

    if pyarrow is None:
        print("status: FAIL")
        print("error: pyarrow is required for Parquet input.")
        return 1

    try:
        documents_df = load_parquet(args.documents)
        chunks_df = load_parquet(args.chunks)
        ensure_columns(documents_df, required=REQUIRED_DOCUMENT_COLUMNS, label="documents")
        ensure_columns(chunks_df, required=REQUIRED_CHUNK_COLUMNS, label="chunks")
    except Exception as exc:
        print("status: FAIL")
        print(f"error: {exc}")
        return 1

    rows = collect_evidence_rows(chunks_df)
    pattern_candidates = extract_pattern_candidates(rows)
    ngram_candidates, top_term_scores = extract_ngram_candidates(rows)
    merged = merge_candidates(pattern_candidates, ngram_candidates)
    compute_document_counts(merged)
    selected = select_queries(merged)

    errors = validate_selected_queries(selected)
    status = determine_status(selected_count=len(selected), errors=errors)

    created_at = datetime.now(timezone.utc).isoformat()
    legal_area_distribution = Counter(normalize_text(value) for value in documents_df["legal_area"].tolist())
    document_type_distribution = Counter(normalize_text(value) for value in documents_df["document_type"].tolist())

    notes = [
        "Queries are generated deterministically from local document and chunk artifacts only.",
        "No embedding model, external API, or Qdrant write operation is used by this script.",
        "Czech legal phrasing is surface-form based, so inflectional variants may still appear across selected queries.",
        "If the batch is narrow or repetitive, the final query set may underfill the 15-query target and return WARN.",
    ]
    if errors:
        notes.extend(errors)

    top_terms = build_top_terms(merged)
    payload = json_payload(
        status=status,
        created_at=created_at,
        documents_path=args.documents,
        chunks_path=args.chunks,
        total_documents=len(documents_df),
        total_chunks=len(chunks_df),
        selected=selected,
    )
    markdown = build_markdown_report(
        status=status,
        documents_path=args.documents,
        chunks_path=args.chunks,
        total_documents=len(documents_df),
        total_chunks=len(chunks_df),
        legal_area_distribution=legal_area_distribution,
        document_type_distribution=document_type_distribution,
        top_terms=top_terms,
        selected=selected,
        notes=notes,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out_md.write_text(markdown, encoding="utf-8")

    print(f"status: {status}")
    print(f"total documents: {len(documents_df)}")
    print(f"total chunks: {len(chunks_df)}")
    print(f"generated query count: {len(selected)}")
    print(f"output json path: {args.out_json}")
    print(f"output markdown path: {args.out_md}")
    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
