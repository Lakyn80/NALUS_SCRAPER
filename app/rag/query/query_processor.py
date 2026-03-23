"""
Rule-based query understanding layer for legal RAG retrieval.

Converts a free-text user query into a structured ProcessedQuery with
normalised text, extracted keywords, and mapped legal concepts.

Interface is intentionally decoupled from any LLM so it can be
swapped or extended later without touching callers.

Usage:
    from app.rag.query.query_processor import process_query

    pq = process_query("žena cizinka unesla dítě do Ruska")
    # pq.keywords      → ["žena", "cizinka", "unesla", "dítě", "ruska"]
    # pq.legal_concepts → ["mezinárodní únos dítěte", "rodičovská odpovědnost",
    #                       "mezinárodní prvek"]
"""

import re
from dataclasses import dataclass

from app.core.logging import get_logger
from app.core.tracing import trace_event

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Legal concept mapping
# Rule: if a keyword *contains* a map key as a substring → add the concept(s).
# Keys are lowercase.  Add more entries here to extend coverage.
# ---------------------------------------------------------------------------

_LEGAL_CONCEPTS_MAP: dict[str, list[str]] = {
    # Únos dítěte
    "únos":    ["mezinárodní únos dítěte"],
    "unos":    ["mezinárodní únos dítěte"],
    "unesl":   ["mezinárodní únos dítěte"],  # unesl / unesla / uneslo / unesení
    "unesení": ["mezinárodní únos dítěte"],
    "přemíst": ["mezinárodní únos dítěte"],  # přemístění, přemístil
    "haag":    ["mezinárodní únos dítěte", "Haagská úmluva"],

    # Dítě / péče
    "dítě":   ["rodičovská odpovědnost"],
    "dítět":  ["rodičovská odpovědnost"],   # dítěte, dítěti
    "dite":   ["rodičovská odpovědnost"],   # bez diakritiky
    "péče":   ["svěření do péče", "rodičovská odpovědnost"],
    "pece":   ["svěření do péče", "rodičovská odpovědnost"],
    "svěř":   ["svěření do péče"],          # svěření, svěřit
    "custody": ["svěření do péče"],

    # Styk
    "styk":   ["styk s dítětem"],

    # Výživa
    "aliment": ["vyživovací povinnost"],
    "výživ":   ["vyživovací povinnost"],
    "vyziv":   ["vyživovací povinnost"],

    # Mezinárodní prvek
    "cizin":  ["mezinárodní prvek"],   # cizinec, cizinka, cizinci
    "rusk":   ["mezinárodní prvek"],   # rusko, ruska, ruské, ruský
    "zahrani": ["mezinárodní prvek"],  # zahraničí, zahraniční

    # Rodinné právo
    "rozvod":  ["rodinné právo"],
    "manžel":  ["rodinné právo"],      # manžel, manželka, manželství
    "rozluk":  ["rodinné právo"],      # rozluka
    "rodič":   ["rodičovská odpovědnost", "rodinné právo"],
}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


@dataclass
class ProcessedQuery:
    original_query: str
    normalized_query: str
    keywords: list[str]
    legal_concepts: list[str]


def process_query(query: str) -> ProcessedQuery:
    """Analyse a free-text query and return a structured ProcessedQuery.

    Steps:
    1. Normalise (lowercase, collapse whitespace).
    2. Extract keywords (words ≥ 3 chars, deduplicated, order preserved).
    3. Map keywords to legal concepts via _LEGAL_CONCEPTS_MAP.
    4. Emit trace events at each step.
    """
    trace_event(logger, "query.start", query=query)

    normalized = _normalize(query)
    trace_event(logger, "query.normalized", normalized_query=normalized)

    keywords = _extract_keywords(normalized)
    trace_event(logger, "query.keywords", keywords=keywords)

    legal_concepts = _map_legal_concepts(keywords)
    trace_event(logger, "query.concepts", legal_concepts=legal_concepts)

    return ProcessedQuery(
        original_query=query,
        normalized_query=normalized,
        keywords=keywords,
        legal_concepts=legal_concepts,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize(query: str) -> str:
    """Lowercase and collapse whitespace runs to a single space."""
    return re.sub(r"\s+", " ", query.lower()).strip()


def _extract_keywords(normalized_query: str) -> list[str]:
    """Split on whitespace, drop words shorter than 3 chars, deduplicate.

    Insertion order is preserved (first occurrence wins).
    """
    seen: dict[str, None] = {}
    for word in normalized_query.split():
        if len(word) >= 3 and word not in seen:
            seen[word] = None
    return list(seen)


def _map_legal_concepts(keywords: list[str]) -> list[str]:
    """Map keywords to legal concepts using substring matching.

    A concept is added when any keyword *contains* a map key as a substring.
    Result is deduplicated; insertion order preserved.
    """
    seen: dict[str, None] = {}
    for keyword in keywords:
        for map_key, concepts in _LEGAL_CONCEPTS_MAP.items():
            if map_key in keyword:
                for concept in concepts:
                    if concept not in seen:
                        seen[concept] = None
    return list(seen)
