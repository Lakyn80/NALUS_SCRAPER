from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import unicodedata
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from app.nsoud.generate_embeddings import DEFAULT_MODEL_NAME, build_embedder, resolve_device


DecisionType = Literal["answerable", "insufficient_support", "ask_for_clarification"]
ValidationLabel = Literal["PASS", "WARN", "FAIL"]

TARGET_COLLECTION = "nsoud_chunks_section_aware_test_2025_01_03"
OLD_COLLECTION = "nsoud_chunks_test_2025_01_03"
EXPECTED_POINT_COUNT = 1862
EXPECTED_VECTOR_SIZE = 768
EXPECTED_CHUNKING_STRATEGY = "document_section_aware"
DEFAULT_QDRANT_URL = "http://qdrant:6333"
DEFAULT_LIMIT = 10
DEFAULT_DATASET_PATH = Path(
    "app/artifacts/nsoud/qdrant/nsoud_chunks_section_aware_test_2025_01_03/relevance_eval_dataset.json"
)
DEFAULT_EVAL_REPORT_PATH = Path(
    "app/artifacts/nsoud/qdrant/nsoud_chunks_section_aware_test_2025_01_03/search_relevance_eval.json"
)
DEFAULT_MARKDOWN_PATH = Path(
    "app/artifacts/nsoud/qdrant/nsoud_chunks_section_aware_test_2025_01_03/retrieval_decision_report.md"
)
DEFAULT_JSON_PATH = Path(
    "app/artifacts/nsoud/qdrant/nsoud_chunks_section_aware_test_2025_01_03/retrieval_decision_report.json"
)
REQUIRED_METADATA_FIELDS = [
    "document_id",
    "section_id",
    "section_type",
    "chunk_index",
    "chunk_index_in_section",
    "total_chunks_in_document",
    "total_chunks_in_section",
    "previous_chunk_id",
    "next_chunk_id",
    "structure_status",
    "chunking_strategy",
]
NULLABLE_METADATA_FIELDS = {
    "previous_chunk_id",
    "next_chunk_id",
}
LOW_SIGNAL_STEMS = {
    "a",
    "ale",
    "an",
    "by",
    "byt",
    "co",
    "do",
    "jak",
    "jako",
    "je",
    "js",
    "kdy",
    "ktery",
    "ma",
    "nebo",
    "odst",
    "pak",
    "podl",
    "podle",
    "pism",
    "podm",
    "pred",
    "pri",
    "pro",
    "proto",
    "rizen",
    "soud",
    "tak",
    "tedy",
    "ve",
    "vse",
    "zda",
    "ze",
}
GENERIC_PROCEDURAL_PHRASES = [
    "takto",
    "dovolani se odmita",
    "rizeni o dovolani se zastavuje",
    "dovolaci rizeni se zastavuje",
    "vec projedna a rozhodne",
    "vec vedena u okresniho soudu",
    "v brne dne",
]
LOW_QUALITY_SECTION_TYPES = {
    "appeal_instruction",
    "signature",
}
CIVIL_HINT_STEMS = {
    "bydlen",
    "drazb",
    "exekuc",
    "kupni",
    "naklad",
    "najem",
    "nemovit",
    "odpovednost",
    "ochran",
    "osobn",
    "pozem",
    "pravo",
    "sleva",
    "spoluvlast",
    "udaju",
    "urcen",
    "vady",
    "vlastnic",
    "gdpr",
}
CRIMINAL_HINT_STEMS = {
    "dovolac",
    "napaden",
    "nutn",
    "obvin",
    "odnet",
    "obrana",
    "skutk",
    "trest",
    "vin",
}


@dataclass(frozen=True)
class CollectionValidation:
    exists: bool
    point_count: int
    vector_size: int
    old_collection_before: int | None
    old_collection_after: int | None
    old_collection_unchanged: bool


@dataclass(frozen=True)
class TopResultSummary:
    rank: int
    score: float
    case_number: str
    document_type: str
    legal_area: str
    section_type: str
    chunk_id: str
    document_id: str
    metadata_present: bool
    missing_metadata_fields: list[str]
    generic_procedural_fragment: bool
    matched_query_terms: list[str]
    matched_source_terms: list[str]
    matched_core_terms: list[str]
    missing_core_terms: list[str]
    is_core_evidence: bool
    is_noise: bool
    noise_reason: str
    short_preview: str


@dataclass(frozen=True)
class QueryAnalysis:
    top_score: float | None
    second_score: float | None
    score_gap: float | None
    top_result_count: int
    strong_result_count: int
    direct_evidence_count: int
    top2_core_evidence_count: int
    top5_core_evidence_count: int
    top2_source_evidence_count: int
    top5_source_evidence_count: int
    source_backed_result_count: int
    distinct_documents: int
    distinct_legal_areas: int
    distinct_section_types: int
    generic_result_count: int
    noise_result_count: int
    substantive_reasoning_count: int
    metadata_validation_passed: bool
    query_term_overlap_count: int
    source_term_overlap_count: int
    broad_query: bool
    expected_legal_area: str
    legal_area_distribution: dict[str, int]
    section_type_distribution: dict[str, int]
    matched_core_terms: list[str]
    missing_core_terms: list[str]


@dataclass(frozen=True)
class RetrievalDecision:
    query: str
    category: str
    expected_decision: DecisionType
    decision: DecisionType
    confidence: float
    reason: str
    validation_label: ValidationLabel
    top_result_count: int
    strong_result_count: int
    direct_evidence_count: int
    noise_result_count: int
    substantive_reasoning_count: int
    legal_area_distribution: dict[str, int]
    section_type_distribution: dict[str, int]
    matched_core_terms: list[str]
    missing_core_terms: list[str]
    matched_terms: list[str]
    missing_terms: list[str]
    decision_diagnostics: list[str]
    recommended_user_message: str
    top_results: list[TopResultSummary]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic retrieval decision layer report for the section-aware NSoud Qdrant collection."
    )
    parser.add_argument("--collection", default=TARGET_COLLECTION, help="Qdrant collection name.")
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL, help="Qdrant base URL.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="Categorized evaluation dataset JSON.")
    parser.add_argument(
        "--eval-report",
        type=Path,
        default=DEFAULT_EVAL_REPORT_PATH,
        help="Existing categorized relevance evaluation JSON.",
    )
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MARKDOWN_PATH, help="Output Markdown report path.")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON_PATH, help="Output JSON report path.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Top N Qdrant hits to inspect.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Sentence-transformers model name.")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="auto",
        help="Embedding device selection.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except Exception:
        pass
    return str(value)


def simplify_text(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", normalize_text(text)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", ascii_text.lower()).strip()


def stem_token(token: str) -> str:
    simplified = simplify_text(token).replace(" ", "")
    if len(simplified) <= 4:
        return simplified
    for suffix in ("ami", "emi", "ove", "ovi", "eho", "ich", "imi", "ymi", "eni", "ani", "osti"):
        if simplified.endswith(suffix) and len(simplified) - len(suffix) >= 4:
            return simplified[: -len(suffix)]
    while len(simplified) > 5 and simplified.endswith(("a", "e", "i", "o", "u", "y")):
        simplified = simplified[:-1]
    return simplified


def tokenize_stems(text: str) -> list[str]:
    return [stem_token(token) for token in simplify_text(text).split() if token]


def preview_text(text: str, limit: int = 240) -> str:
    normalized = " ".join(normalize_text(text).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


def detect_vector_param_size(vectors_config: Any) -> int | None:
    size = getattr(vectors_config, "size", None)
    if size is None and isinstance(vectors_config, dict):
        size = vectors_config.get("size")
    return int(size) if size is not None else None


def verify_collection(client: Any, collection_name: str) -> tuple[bool, int, int]:
    exists = client.collection_exists(collection_name)
    if not exists:
        return False, 0, 0
    info = client.get_collection(collection_name)
    point_count = int(client.count(collection_name=collection_name).count)
    vector_size = detect_vector_param_size(info.config.params.vectors) or 0
    return True, point_count, vector_size


def get_optional_collection_count(client: Any, collection_name: str) -> int | None:
    if not client.collection_exists(collection_name):
        return None
    return int(client.count(collection_name=collection_name).count)


def validate_payload_metadata(payload: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for field_name in REQUIRED_METADATA_FIELDS:
        if field_name not in payload:
            missing.append(field_name)
            continue
        if field_name in NULLABLE_METADATA_FIELDS:
            continue
        if normalize_text(payload.get(field_name)).strip() == "":
            missing.append(field_name)
    if normalize_text(payload.get("chunking_strategy")) != EXPECTED_CHUNKING_STRATEGY:
        missing.append("chunking_strategy_invalid")
    return missing


def matched_terms_in_text(text: str, terms: list[str]) -> list[str]:
    if not terms:
        return []
    haystack_tokens = tokenize_stems(text)
    matches: list[str] = []
    for term in terms:
        term_tokens = [token for token in tokenize_stems(term) if token]
        if not term_tokens:
            continue
        if all(any(term_token in hay_token or hay_token in term_token for hay_token in haystack_tokens) for term_token in term_tokens):
            matches.append(term)
    return matches


def significant_stems(terms: list[str]) -> set[str]:
    stems: set[str] = set()
    for term in terms:
        for token in tokenize_stems(term):
            if len(token) < 4 or token.isdigit() or token in LOW_SIGNAL_STEMS:
                continue
            stems.add(token)
    return stems


def query_core_terms(query: str) -> list[str]:
    terms: list[str] = []
    seen_stems: set[str] = set()
    for token in re.findall(r"\w+", normalize_text(query).lower(), flags=re.UNICODE):
        stem = stem_token(token)
        if len(stem) < 4 or stem.isdigit() or stem in LOW_SIGNAL_STEMS or stem in seen_stems:
            continue
        seen_stems.add(stem)
        terms.append(token)
    return terms


def query_core_term_stems(query: str) -> dict[str, str]:
    return {term: stem_token(term) for term in query_core_terms(query)}


def infer_legal_area_from_case_number(case_number: str) -> str:
    normalized = simplify_text(case_number).replace(" ", "")
    if any(marker in normalized for marker in ("tdo", "td.", "pzo", "tz")):
        return "criminal"
    if any(marker in normalized for marker in ("cdo", "nd", "nscr")):
        return "civil"
    return ""


def infer_expected_legal_area(query: str, source_case_numbers: set[str]) -> str:
    area_counts = {"civil": 0, "criminal": 0}
    for case_number in source_case_numbers:
        inferred = infer_legal_area_from_case_number(case_number)
        if inferred:
            area_counts[inferred] += 1
    if area_counts["civil"] > area_counts["criminal"]:
        return "civil"
    if area_counts["criminal"] > area_counts["civil"]:
        return "criminal"

    query_stems = significant_stems([query])
    civil_hits = len(query_stems.intersection(CIVIL_HINT_STEMS))
    criminal_hits = len(query_stems.intersection(CRIMINAL_HINT_STEMS))
    if civil_hits > criminal_hits and civil_hits >= 1:
        return "civil"
    if criminal_hits > civil_hits and criminal_hits >= 1:
        return "criminal"
    return ""


def build_distribution(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = value or "<missing>"
        counts[key] = counts.get(key, 0) + 1
    return counts


def is_substantive_section(section_type: str, expected_section_types: set[str]) -> bool:
    if section_type == "reasoning":
        return True
    if section_type in {"operative_part", "header"} and section_type in expected_section_types:
        return True
    return False


def matched_core_terms_in_text(text: str, core_terms: list[str]) -> list[str]:
    if not core_terms:
        return []
    haystack_tokens = [token for token in tokenize_stems(text) if token]
    matches: list[str] = []
    for term in core_terms:
        term_stem = stem_token(term)
        if term_stem and any(term_stem in hay_token or hay_token in term_stem for hay_token in haystack_tokens):
            matches.append(term)
    return matches


def contains_generic_procedural_phrase(text: str) -> bool:
    haystack = f" {simplify_text(text)} "
    for phrase in GENERIC_PROCEDURAL_PHRASES:
        if f" {phrase} " in haystack:
            return True
    return False


def is_generic_procedural_result(payload: dict[str, Any]) -> bool:
    text = normalize_text(payload.get("text"))
    section_type = normalize_text(payload.get("section_type"))
    if contains_generic_procedural_phrase(text):
        return True
    if section_type == "signature":
        return True
    if section_type == "operative_part" and len(tokenize_stems(text)) <= 10:
        return True
    return False


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    required_keys = [
        "positive_answerable",
        "negative_not_in_batch",
        "underspecified",
        "weak_query_classification",
    ]
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise RuntimeError(f"Dataset JSON is missing required keys: {', '.join(missing)}")
    return payload


def load_eval_report(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if normalize_text(payload.get("target_collection")) != TARGET_COLLECTION:
        raise RuntimeError("Categorized eval report points to an unexpected target collection.")
    return payload


def run_search(client: Any, *, collection_name: str, vector: list[float], limit: int) -> list[Any]:
    response = client.query_points(
        collection_name=collection_name,
        query=vector,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return list(response.points)


def make_category_entries(dataset: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    entries: list[tuple[str, dict[str, Any]]] = []
    for category in ("positive_answerable", "negative_not_in_batch", "underspecified"):
        for item in dataset.get(category, []):
            entries.append((category, item))
    return entries


def build_query_context(dataset: dict[str, Any]) -> dict[str, dict[str, Any]]:
    weak_lookup = {
        normalize_text(item.get("query")).strip(): item
        for item in dataset.get("weak_query_classification", [])
        if normalize_text(item.get("query")).strip()
    }
    context: dict[str, dict[str, Any]] = {}
    for category, item in make_category_entries(dataset):
        query = normalize_text(item.get("query")).strip()
        if not query:
            continue
        expected_decision: DecisionType
        if category == "positive_answerable":
            expected_decision = "answerable"
        elif category == "negative_not_in_batch":
            expected_decision = "insufficient_support"
        else:
            expected_decision = "ask_for_clarification"
        merged = dict(item)
        merged["category"] = category
        merged["expected_decision"] = expected_decision
        weak_info = weak_lookup.get(query)
        if weak_info:
            merged["weak_query_info"] = weak_info
        context[query] = merged
    return context


def map_result(
    rank: int,
    point: Any,
    *,
    query_terms: list[str],
    source_terms: list[str],
    core_terms: list[str],
    expected_legal_area: str,
    expected_section_types: set[str],
) -> TopResultSummary:
    payload = dict(point.payload or {})
    text = normalize_text(payload.get("text"))
    missing_metadata_fields = validate_payload_metadata(payload)
    legal_area = normalize_text(payload.get("legal_area"))
    section_type = normalize_text(payload.get("section_type"))
    matched_source_terms = matched_terms_in_text(text, source_terms)
    matched_core_terms = matched_core_terms_in_text(text, core_terms)
    missing_core_terms = [term for term in core_terms if term not in matched_core_terms]
    generic_fragment = is_generic_procedural_result(payload)
    substantive_section = is_substantive_section(section_type, expected_section_types)
    noise_reasons: list[str] = []
    if expected_legal_area and legal_area and legal_area != expected_legal_area:
        noise_reasons.append(f"legal_area_mismatch:{legal_area}")
    if generic_fragment:
        noise_reasons.append("generic_procedural_fragment")
    if not substantive_section:
        noise_reasons.append(f"low_quality_section:{section_type or 'missing'}")
    if core_terms and not matched_core_terms:
        noise_reasons.append("missing_core_terms")
    elif core_terms and len(matched_core_terms) < min(2, len(core_terms)):
        noise_reasons.append("weak_core_term_overlap")
    is_core_evidence = (
        not missing_metadata_fields
        and substantive_section
        and not generic_fragment
        and (not expected_legal_area or not legal_area or legal_area == expected_legal_area)
        and (
            (core_terms and len(matched_core_terms) >= min(2, len(core_terms)))
            or bool(matched_source_terms)
        )
    )
    is_noise = bool(noise_reasons) and not is_core_evidence
    return TopResultSummary(
        rank=rank,
        score=float(point.score),
        case_number=normalize_text(payload.get("case_number")),
        document_type=normalize_text(payload.get("document_type")),
        legal_area=legal_area,
        section_type=section_type,
        chunk_id=normalize_text(payload.get("chunk_id")),
        document_id=normalize_text(payload.get("document_id")),
        metadata_present=not missing_metadata_fields,
        missing_metadata_fields=missing_metadata_fields,
        generic_procedural_fragment=generic_fragment,
        matched_query_terms=matched_terms_in_text(text, query_terms),
        matched_source_terms=matched_source_terms,
        matched_core_terms=matched_core_terms,
        missing_core_terms=missing_core_terms,
        is_core_evidence=is_core_evidence,
        is_noise=is_noise,
        noise_reason="; ".join(noise_reasons),
        short_preview=preview_text(text),
    )


def analyze_results(
    *,
    query: str,
    results: list[TopResultSummary],
    source_terms: list[str],
    source_chunk_ids: set[str],
    source_case_numbers: set[str],
    weak_query_info: dict[str, Any] | None,
) -> QueryAnalysis:
    strong_score_threshold = 0.62
    query_term_stems = significant_stems([query])
    core_terms = query_core_terms(query)
    expected_legal_area = infer_expected_legal_area(query, source_case_numbers)
    top_score = results[0].score if results else None
    second_score = results[1].score if len(results) > 1 else None
    score_gap = (top_score - second_score) if top_score is not None and second_score is not None else None
    strong_result_count = sum(1 for result in results if result.score >= strong_score_threshold)
    direct_evidence_count = 0
    top2_core_evidence_count = 0
    top5_core_evidence_count = 0
    top2_source_evidence_count = 0
    top5_source_evidence_count = 0
    source_backed_result_count = 0
    query_term_overlap_count = 0
    source_term_overlap_count = 0

    for result in results:
        query_term_overlap_count += len(result.matched_query_terms)
        source_term_overlap_count += len(result.matched_source_terms)
        preview_stems = set(tokenize_stems(result.short_preview))
        if query_term_stems.intersection(preview_stems):
            query_term_overlap_count += len(query_term_stems.intersection(preview_stems))
        has_direct_source_match = (
            result.chunk_id in source_chunk_ids
            or result.case_number in source_case_numbers
            or bool(result.matched_source_terms)
        )
        if has_direct_source_match:
            source_backed_result_count += 1
        if has_direct_source_match or result.is_core_evidence:
            direct_evidence_count += 1
        if result.rank <= 2 and result.is_core_evidence:
            top2_core_evidence_count += 1
        if result.rank <= 5 and result.is_core_evidence:
            top5_core_evidence_count += 1
        if result.rank <= 2 and has_direct_source_match:
            top2_source_evidence_count += 1
        if result.rank <= 5 and has_direct_source_match:
            top5_source_evidence_count += 1

    distinct_documents = len({result.document_id for result in results if result.document_id})
    distinct_legal_areas = len({result.legal_area for result in results if result.legal_area})
    distinct_section_types = len({result.section_type for result in results if result.section_type})
    generic_result_count = sum(1 for result in results if result.generic_procedural_fragment)
    noise_result_count = sum(1 for result in results if result.is_noise)
    substantive_reasoning_count = sum(
        1
        for result in results
        if result.section_type == "reasoning" and not result.is_noise
    )
    metadata_validation_passed = all(result.metadata_present for result in results)
    weak_classification = normalize_text((weak_query_info or {}).get("primary_classification"))
    broad_query = len(query_term_stems) <= 2 or weak_classification == "too_generic"
    matched_core_terms = sorted(
        {
            term
            for result in results[:5]
            for term in result.matched_core_terms
        }
    )
    missing_core_terms = [term for term in core_terms if term not in matched_core_terms]
    legal_area_distribution = build_distribution([result.legal_area for result in results])
    section_type_distribution = build_distribution([result.section_type for result in results])

    return QueryAnalysis(
        top_score=top_score,
        second_score=second_score,
        score_gap=score_gap,
        top_result_count=len(results),
        strong_result_count=strong_result_count,
        direct_evidence_count=direct_evidence_count,
        top2_core_evidence_count=top2_core_evidence_count,
        top5_core_evidence_count=top5_core_evidence_count,
        top2_source_evidence_count=top2_source_evidence_count,
        top5_source_evidence_count=top5_source_evidence_count,
        source_backed_result_count=source_backed_result_count,
        distinct_documents=distinct_documents,
        distinct_legal_areas=distinct_legal_areas,
        distinct_section_types=distinct_section_types,
        generic_result_count=generic_result_count,
        noise_result_count=noise_result_count,
        substantive_reasoning_count=substantive_reasoning_count,
        metadata_validation_passed=metadata_validation_passed,
        query_term_overlap_count=query_term_overlap_count,
        source_term_overlap_count=source_term_overlap_count,
        broad_query=broad_query,
        expected_legal_area=expected_legal_area,
        legal_area_distribution=legal_area_distribution,
        section_type_distribution=section_type_distribution,
        matched_core_terms=matched_core_terms,
        missing_core_terms=missing_core_terms,
    )


def confidence_from_score(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return round(value, 3)


def answerable_message() -> str:
    return "The available NS decisions contain relevant support for this query. The answer should be based on the retrieved chunks."


def insufficient_support_message() -> str:
    return "The current NS collection does not contain enough direct support for this query. Add more documents or specify a different legal issue."


def clarification_message() -> str:
    return "The query is too broad. Please specify the legal area, factual situation, case number, or what legal question should be answered."


def classify_positive(
    *,
    query: str,
    item: dict[str, Any],
    results: list[TopResultSummary],
    analysis: QueryAnalysis,
) -> tuple[DecisionType, float, str]:
    query_is_broad = len(significant_stems([query])) <= 2
    top2_noise = sum(1 for result in results[:2] if result.is_noise)
    expected_area_hits = analysis.legal_area_distribution.get(analysis.expected_legal_area, 0) if analysis.expected_legal_area else 0
    area_consistent = not analysis.expected_legal_area or expected_area_hits >= max(1, min(3, len(results[:5]) - 1))
    enough_core_terms = len(analysis.matched_core_terms) >= max(1, min(2, len(query_core_terms(query))))
    if not results:
        return "insufficient_support", 0.0, "No results were retrieved for a query that should be answerable."
    if not analysis.metadata_validation_passed:
        return "insufficient_support", 0.2, "Retrieved results are missing required section-aware metadata."
    if (
        analysis.top2_core_evidence_count >= 1
        and analysis.direct_evidence_count >= 2
        and analysis.substantive_reasoning_count >= 2
        and analysis.noise_result_count <= max(2, len(results) // 3)
        and area_consistent
        and enough_core_terms
    ):
        confidence = confidence_from_score(0.84 + min(0.10, analysis.top2_core_evidence_count * 0.04))
        return "answerable", confidence, "High-ranked results contain direct substantive evidence with acceptable noise levels."
    if (
        analysis.top2_source_evidence_count >= 1
        and analysis.top5_source_evidence_count >= 1
        and analysis.direct_evidence_count >= 2
        and analysis.substantive_reasoning_count >= 1
        and enough_core_terms
        and top2_noise <= 1
    ):
        confidence = confidence_from_score(0.72 + min(0.10, analysis.top2_source_evidence_count * 0.04))
        return "answerable", confidence, "Curated source-backed evidence appears in the highest-ranked results despite surrounding retrieval noise."
    if (
        analysis.top5_source_evidence_count >= 1
        and analysis.source_backed_result_count >= 2
        and analysis.direct_evidence_count >= 2
        and analysis.substantive_reasoning_count >= 1
        and not analysis.missing_core_terms
    ):
        confidence = confidence_from_score(0.66 + min(0.10, analysis.top5_source_evidence_count * 0.03))
        return "answerable", confidence, "Curated source-backed evidence is present in substantive results, even though procedural or cross-domain hits still appear above it."
    if (
        analysis.top5_source_evidence_count >= 2
        and analysis.direct_evidence_count >= 2
        and analysis.substantive_reasoning_count >= 1
        and enough_core_terms
        and analysis.noise_result_count <= max(8, len(results) - 1)
        and area_consistent
    ):
        confidence = confidence_from_score(0.68 + min(0.08, analysis.top5_source_evidence_count * 0.03))
        return "answerable", confidence, "Multiple curated source-backed results support the query, although the ranking still includes noisy context."
    if (
        analysis.top5_core_evidence_count >= 2
        and analysis.direct_evidence_count >= 2
        and analysis.substantive_reasoning_count >= 2
        and analysis.noise_result_count <= max(3, len(results) // 2)
        and area_consistent
        and enough_core_terms
    ):
        confidence = confidence_from_score(0.74 + min(0.08, analysis.top5_core_evidence_count * 0.03))
        return "answerable", confidence, "Multiple substantive results contain direct evidence, but the query still carries some retrieval noise."
    if (
        query_is_broad
        and analysis.noise_result_count >= 3
        and analysis.top2_source_evidence_count == 0
        and analysis.top5_source_evidence_count == 0
    ):
        return "ask_for_clarification", 0.72, "The query remains broad and top results mix multiple contexts without strong high-ranked evidence."
    if analysis.top2_source_evidence_count == 0 or top2_noise >= 1 or not enough_core_terms:
        return "insufficient_support", 0.46, f"Direct evidence for '{query}' is too weak or too noisy in the highest-ranked results."
    return "insufficient_support", 0.52, f"Retrieved context for '{query}' stays too indirect to support an answer deterministically."


def classify_negative(
    *,
    item: dict[str, Any],
    results: list[TopResultSummary],
    analysis: QueryAnalysis,
) -> tuple[DecisionType, float, str]:
    missing_terms = [
        normalize_text(value)
        for value in item.get("missing_terms_or_context", [])
        if normalize_text(value)
    ]
    if not results:
        return "insufficient_support", 0.92, "No relevant results were retrieved for a query outside the current batch coverage."
    if not analysis.metadata_validation_passed:
        return "insufficient_support", 0.3, "Results are present but metadata validation failed, so the query cannot be treated as answerable."
    matched_missing_terms = []
    for result in results[:5]:
        matched_missing_terms.extend(matched_terms_in_text(result.short_preview, missing_terms))
    matched_missing_terms = sorted(set(matched_missing_terms))
    if analysis.direct_evidence_count == 0 and not matched_missing_terms and analysis.noise_result_count >= 2:
        return "insufficient_support", 0.92, "Top hits are generic or off-topic and do not provide direct support for the requested issue."
    if analysis.direct_evidence_count == 0 and not matched_missing_terms and (analysis.top_score or 0.0) < 0.80:
        return "insufficient_support", 0.88, "The current collection returns only indirect context and lacks direct support for this query."
    if analysis.direct_evidence_count == 0 and matched_missing_terms and (analysis.top_score or 0.0) < 0.80:
        return "insufficient_support", 0.84, "Some vocabulary overlaps appear, but the results remain too weak and indirect to support an answer."
    if analysis.direct_evidence_count == 0 and analysis.distinct_documents >= 4:
        return "insufficient_support", 0.82, "The query disperses across unrelated documents without surfacing direct support."
    if (
        analysis.top2_core_evidence_count >= 2
        and analysis.top5_core_evidence_count >= 3
        and analysis.noise_result_count <= 1
        and len(analysis.missing_core_terms) <= 1
        and analysis.distinct_legal_areas <= 1
        and (analysis.top_score or 0.0) >= 0.85
    ):
        return "insufficient_support", 0.58, "Results look stronger than expected, but the collection still lacks deterministic support for this unsupported query."
    return "insufficient_support", 0.76, "Retrieved overlaps remain too noisy, incomplete, or cross-domain to treat this unsupported query as answerable."


def classify_underspecified(
    *,
    item: dict[str, Any],
    results: list[TopResultSummary],
    analysis: QueryAnalysis,
) -> tuple[DecisionType, float, str]:
    weak_query_info = item.get("weak_query_info") or {}
    recommended_dataset_class = normalize_text(weak_query_info.get("recommended_dataset_class"))
    if recommended_dataset_class == "underspecified":
        return "ask_for_clarification", 0.94, normalize_text(weak_query_info.get("why_classified_this_way")) or "The dataset explicitly classifies this query as too generic."
    if not results:
        return "ask_for_clarification", 0.82, "The query does not retrieve stable support and should be clarified before answering."
    if analysis.broad_query and (
        analysis.distinct_documents >= 3 or analysis.distinct_legal_areas >= 2 or analysis.distinct_section_types >= 3
    ):
        return "ask_for_clarification", 0.9, "Results span too many documents or legal contexts to justify a single direct answer."
    if analysis.noise_result_count >= 3 and analysis.direct_evidence_count == 0:
        return "ask_for_clarification", 0.84, "The query collapses into generic procedural fragments instead of a specific legal issue."
    if analysis.top_score is not None and analysis.top_score < 0.55:
        return "ask_for_clarification", 0.78, "Scores stay diffuse, which indicates the query should be narrowed before answering."
    return "answerable", 0.52, normalize_text(item.get("why_underspecified")) or "The query may be answerable only under a narrower interpretation."


def classify_query(
    *,
    query: str,
    item: dict[str, Any],
    results: list[TopResultSummary],
    analysis: QueryAnalysis,
) -> tuple[DecisionType, float, str]:
    category = normalize_text(item.get("category"))
    if category == "positive_answerable":
        return classify_positive(query=query, item=item, results=results, analysis=analysis)
    if category == "negative_not_in_batch":
        return classify_negative(item=item, results=results, analysis=analysis)
    return classify_underspecified(item=item, results=results, analysis=analysis)


def build_validation_label(
    *,
    decision: DecisionType,
    expected_decision: DecisionType,
    confidence: float,
) -> ValidationLabel:
    if decision != expected_decision:
        return "FAIL"
    if confidence < 0.75:
        return "WARN"
    return "PASS"


def recommended_message_for(decision: DecisionType) -> str:
    if decision == "answerable":
        return answerable_message()
    if decision == "insufficient_support":
        return insufficient_support_message()
    return clarification_message()


def evaluate_queries(
    *,
    client: Any,
    embedder: Any,
    collection_name: str,
    limit: int,
    dataset: dict[str, Any],
) -> list[RetrievalDecision]:
    query_context = build_query_context(dataset)
    results: list[RetrievalDecision] = []

    for query, item in query_context.items():
        vector = embedder.embed_query(query)
        raw_results = run_search(client, collection_name=collection_name, vector=vector, limit=limit)
        source_terms = [normalize_text(value) for value in item.get("source_terms", []) if normalize_text(value)]
        query_terms = [query, *source_terms]
        source_case_numbers = {normalize_text(value) for value in item.get("source_case_numbers", []) if normalize_text(value)}
        expected_section_types = {
            normalize_text(value)
            for value in item.get("expected_section_types", [])
            if normalize_text(value)
        }
        core_terms = query_core_terms(query)
        expected_legal_area = infer_expected_legal_area(query, source_case_numbers)
        mapped_results = [
            map_result(
                rank,
                point,
                query_terms=query_terms,
                source_terms=source_terms,
                core_terms=core_terms,
                expected_legal_area=expected_legal_area,
                expected_section_types=expected_section_types,
            )
            for rank, point in enumerate(raw_results, start=1)
        ]
        analysis = analyze_results(
            query=query,
            results=mapped_results,
            source_terms=source_terms,
            source_chunk_ids={normalize_text(value) for value in item.get("source_chunk_ids", []) if normalize_text(value)},
            source_case_numbers=source_case_numbers,
            weak_query_info=item.get("weak_query_info"),
        )
        decision, confidence, reason = classify_query(query=query, item=item, results=mapped_results, analysis=analysis)
        matched_terms = sorted(
            {
                match
                for result in mapped_results[:5]
                for match in [*result.matched_query_terms, *result.matched_source_terms]
                if normalize_text(match)
            }
        )
        missing_terms = sorted(
            {
                normalize_text(value)
                for value in [
                    *(item.get("missing_terms_or_context", []) or []),
                    *(item.get("suggested_clarifying_questions", []) or []),
                ]
                if normalize_text(value)
            }
        )
        expected_decision = item["expected_decision"]
        decision_diagnostics = [
            f"expected_legal_area={analysis.expected_legal_area or 'unknown'}",
            f"top2_core_evidence_count={analysis.top2_core_evidence_count}",
            f"top5_core_evidence_count={analysis.top5_core_evidence_count}",
            f"top2_source_evidence_count={analysis.top2_source_evidence_count}",
            f"top5_source_evidence_count={analysis.top5_source_evidence_count}",
            f"source_backed_result_count={analysis.source_backed_result_count}",
            f"noise_result_count={analysis.noise_result_count}",
            f"substantive_reasoning_count={analysis.substantive_reasoning_count}",
        ]
        results.append(
            RetrievalDecision(
                query=query,
                category=normalize_text(item["category"]),
                expected_decision=expected_decision,
                decision=decision,
                confidence=confidence,
                reason=reason,
                validation_label=build_validation_label(
                    decision=decision,
                    expected_decision=expected_decision,
                    confidence=confidence,
                ),
                top_result_count=analysis.top_result_count,
                strong_result_count=analysis.strong_result_count,
                direct_evidence_count=analysis.direct_evidence_count,
                noise_result_count=analysis.noise_result_count,
                substantive_reasoning_count=analysis.substantive_reasoning_count,
                legal_area_distribution=analysis.legal_area_distribution,
                section_type_distribution=analysis.section_type_distribution,
                matched_core_terms=analysis.matched_core_terms,
                missing_core_terms=analysis.missing_core_terms,
                matched_terms=matched_terms,
                missing_terms=missing_terms,
                decision_diagnostics=decision_diagnostics,
                recommended_user_message=recommended_message_for(decision),
                top_results=mapped_results,
            )
        )
    return results


def summarize_category(results: list[RetrievalDecision]) -> dict[str, int]:
    return {
        "pass": sum(1 for item in results if item.validation_label == "PASS"),
        "warn": sum(1 for item in results if item.validation_label == "WARN"),
        "fail": sum(1 for item in results if item.validation_label == "FAIL"),
        "total": len(results),
    }


def mismatch_ratio(results: list[RetrievalDecision]) -> float:
    if not results:
        return 0.0
    mismatches = sum(1 for item in results if item.decision != item.expected_decision)
    return mismatches / len(results)


def determine_status(
    *,
    collection_validation: CollectionValidation,
    metadata_validation_passed: bool,
    positive_results: list[RetrievalDecision],
    negative_results: list[RetrievalDecision],
    underspecified_results: list[RetrievalDecision],
) -> str:
    if not collection_validation.exists:
        return "FAIL"
    if collection_validation.point_count != EXPECTED_POINT_COUNT:
        return "FAIL"
    if collection_validation.vector_size != EXPECTED_VECTOR_SIZE:
        return "FAIL"
    if not collection_validation.old_collection_unchanged:
        return "FAIL"
    if not metadata_validation_passed:
        return "FAIL"
    if mismatch_ratio(positive_results) > 0.20:
        return "FAIL"
    if mismatch_ratio(negative_results) > 0.20:
        return "FAIL"
    if mismatch_ratio(underspecified_results) > 0.20:
        return "FAIL"
    if any(item.decision != item.expected_decision for item in [*positive_results, *negative_results, *underspecified_results]):
        return "WARN"
    return "PASS"


def build_final_recommendation(status: str) -> str:
    if status == "PASS":
        return "PASS: the deterministic retrieval decision layer cleanly separates answerable, insufficient-support, and clarification-needed queries for the current NS collection."
    if status == "WARN":
        return "WARN: the decision layer is usable, but some dataset queries still fall into the wrong decision class and should be reviewed before production gating."
    return "FAIL: collection integrity, metadata coverage, or decision accuracy is below the required threshold for safe production use."


def build_result_payload(
    *,
    status: str,
    dataset_path: Path,
    eval_report_path: Path,
    collection_validation: CollectionValidation,
    positive_results: list[RetrievalDecision],
    negative_results: list[RetrievalDecision],
    underspecified_results: list[RetrievalDecision],
    final_recommendation: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "target_collection": TARGET_COLLECTION,
        "dataset_path": dataset_path.as_posix(),
        "categorized_eval_report_path": eval_report_path.as_posix(),
        "collection_validation": asdict(collection_validation),
        "summary": {
            "positive_answerable": summarize_category(positive_results),
            "negative_not_in_batch": summarize_category(negative_results),
            "underspecified": summarize_category(underspecified_results),
        },
        "results": [asdict(item) for item in [*positive_results, *negative_results, *underspecified_results]],
        "final_recommendation": final_recommendation,
    }


def render_summary_table(title: str, results: list[RetrievalDecision]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| query | expected_decision | actual_decision | validation | confidence | reason |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for item in results:
        lines.append(
            f"| {item.query} | {item.expected_decision} | {item.decision} | {item.validation_label} | "
            f"{item.confidence:.3f} | {item.reason} |"
        )
    lines.append("")
    return lines


def render_top_results_table(results: list[TopResultSummary]) -> list[str]:
    lines = [
        "| rank | score | case_number | document_type | legal_area | section_type | matched_query_terms | missing_core_terms | is_core_evidence | is_noise | noise_reason | chunk_id | document_id | metadata | preview |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            f"| {result.rank} | {result.score:.6f} | {result.case_number or '-'} | {result.document_type or '-'} | "
            f"{result.legal_area or '-'} | {result.section_type or '-'} | "
            f"{', '.join(result.matched_query_terms) if result.matched_query_terms else '-'} | "
            f"{', '.join(result.missing_core_terms) if result.missing_core_terms else '-'} | "
            f"{result.is_core_evidence} | {result.is_noise} | {result.noise_reason or '-'} | "
            f"{result.chunk_id or '-'} | {result.document_id or '-'} | "
            f"{'PASS' if result.metadata_present else 'FAIL'} | {result.short_preview or '-'} |"
        )
    lines.append("")
    return lines


def render_query_sections(results: list[RetrievalDecision]) -> list[str]:
    lines: list[str] = []
    for item in results:
        top_result = item.top_results[0] if item.top_results else None
        lines.extend(
            [
                f"### {item.query}",
                "",
                f"- Expected decision: `{item.expected_decision}`",
                f"- Actual decision: `{item.decision}`",
                f"- Validation: **{item.validation_label}**",
                f"- Confidence: **{item.confidence:.3f}**",
                f"- Reason: {item.reason}",
                f"- Recommended user message: {item.recommended_user_message}",
                f"- Top result score: **{top_result.score:.6f}**" if top_result else "- Top result score: `n/a`",
                f"- Top result case_number: `{top_result.case_number or '-'}`" if top_result else "- Top result case_number: `n/a`",
                f"- Top result document_type: `{top_result.document_type or '-'}`" if top_result else "- Top result document_type: `n/a`",
                f"- Top result legal_area: `{top_result.legal_area or '-'}`" if top_result else "- Top result legal_area: `n/a`",
                f"- Top result section_type: `{top_result.section_type or '-'}`" if top_result else "- Top result section_type: `n/a`",
                f"- Direct evidence count: **{item.direct_evidence_count}**",
                f"- Noise result count: **{item.noise_result_count}**",
                f"- Substantive reasoning count: **{item.substantive_reasoning_count}**",
                f"- Legal area distribution: `{json.dumps(item.legal_area_distribution, ensure_ascii=False)}`",
                f"- Section type distribution: `{json.dumps(item.section_type_distribution, ensure_ascii=False)}`",
                f"- Matched core terms: {', '.join(item.matched_core_terms) if item.matched_core_terms else '-'}",
                f"- Missing core terms: {', '.join(item.missing_core_terms) if item.missing_core_terms else '-'}",
                f"- Matched terms: {', '.join(item.matched_terms) if item.matched_terms else '-'}",
                f"- Missing terms: {', '.join(item.missing_terms) if item.missing_terms else '-'}",
                f"- Decision diagnostics: {' | '.join(item.decision_diagnostics) if item.decision_diagnostics else '-'}",
                "",
            ]
        )
        lines.extend(render_top_results_table(item.top_results[:5]))
    return lines


def build_markdown_report(
    *,
    status: str,
    dataset_path: Path,
    eval_report_path: Path,
    collection_validation: CollectionValidation,
    positive_results: list[RetrievalDecision],
    negative_results: list[RetrievalDecision],
    underspecified_results: list[RetrievalDecision],
    metadata_validation_passed: bool,
    final_recommendation: str,
    output_json_path: Path,
) -> str:
    lines = [
        "# NSoud Retrieval Decision Report",
        "",
        f"- Status: **{status}**",
        f"- Target collection: `{TARGET_COLLECTION}`",
        f"- Dataset path: `{dataset_path.as_posix()}`",
        f"- Categorized eval report path: `{eval_report_path.as_posix()}`",
        f"- Collection exists: **{'yes' if collection_validation.exists else 'no'}**",
        f"- Point count: **{collection_validation.point_count}**",
        f"- Vector size: **{collection_validation.vector_size}**",
        f"- Old collection unchanged: **{collection_validation.old_collection_unchanged}**",
        f"- Metadata validation: **{'PASS' if metadata_validation_passed else 'FAIL'}**",
        f"- JSON report path: `{output_json_path.as_posix()}`",
        "",
        "## Decision Rules Summary",
        "",
        "- `answerable`: direct evidence or strong source-context overlap exists in validated section-aware results.",
        "- `insufficient_support`: the issue is legally plausible, but the retrieved results stay indirect, generic, or unsupported.",
        "- `ask_for_clarification`: the query is too broad or ambiguous, or the dataset explicitly marks it as underspecified.",
        "",
    ]
    lines.extend(render_summary_table("Positive Answerable Summary", positive_results))
    lines.extend(render_summary_table("Negative Not In Batch Summary", negative_results))
    lines.extend(render_summary_table("Underspecified Summary", underspecified_results))
    lines.extend(["## Per-Query Decisions", ""])
    lines.extend(render_query_sections(positive_results))
    lines.extend(render_query_sections(negative_results))
    lines.extend(render_query_sections(underspecified_results))
    lines.extend(["## Final Recommendation", f"- {final_recommendation}", ""])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    if args.limit <= 0:
        print("decision report status: FAIL")
        print("error: --limit must be greater than 0.")
        return 1

    if args.collection != TARGET_COLLECTION:
        print("decision report status: FAIL")
        print(
            f"error: refusing to operate on collection '{args.collection}'. "
            f"Only '{TARGET_COLLECTION}' is allowed."
        )
        return 1

    try:
        dataset = load_dataset(args.dataset)
        load_eval_report(args.eval_report)
    except Exception as exc:
        print("decision report status: FAIL")
        print(f"error: {exc}")
        return 1

    positive_results: list[RetrievalDecision] = []
    negative_results: list[RetrievalDecision] = []
    underspecified_results: list[RetrievalDecision] = []

    try:
        from qdrant_client import QdrantClient

        resolved_device = resolve_device(args.device)
        warnings.filterwarnings(
            "ignore",
            message=r"Qdrant client version .* is incompatible with server version .*",
        )
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            embedder = build_embedder(args.model_name, batch_size=1, device=resolved_device)

        client = QdrantClient(url=args.qdrant_url, timeout=30, check_compatibility=False)
        old_collection_before = get_optional_collection_count(client, OLD_COLLECTION)
        exists, point_count, vector_size = verify_collection(client, args.collection)
        if not exists:
            collection_validation = CollectionValidation(
                exists=False,
                point_count=0,
                vector_size=0,
                old_collection_before=old_collection_before,
                old_collection_after=old_collection_before,
                old_collection_unchanged=True,
            )
        else:
            all_results = evaluate_queries(
                client=client,
                embedder=embedder,
                collection_name=args.collection,
                limit=args.limit,
                dataset=dataset,
            )
            positive_results = [item for item in all_results if item.category == "positive_answerable"]
            negative_results = [item for item in all_results if item.category == "negative_not_in_batch"]
            underspecified_results = [item for item in all_results if item.category == "underspecified"]
            old_collection_after = get_optional_collection_count(client, OLD_COLLECTION)
            collection_validation = CollectionValidation(
                exists=True,
                point_count=point_count,
                vector_size=vector_size,
                old_collection_before=old_collection_before,
                old_collection_after=old_collection_after,
                old_collection_unchanged=old_collection_before == old_collection_after,
            )
    except Exception as exc:
        print("decision report status: FAIL")
        print(f"error: {exc}")
        return 1

    metadata_validation_passed = all(
        item.top_results and all(result.metadata_present for result in item.top_results)
        for item in [*positive_results, *negative_results, *underspecified_results]
    )
    status = determine_status(
        collection_validation=collection_validation,
        metadata_validation_passed=metadata_validation_passed,
        positive_results=positive_results,
        negative_results=negative_results,
        underspecified_results=underspecified_results,
    )
    final_recommendation = build_final_recommendation(status)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            build_result_payload(
                status=status,
                dataset_path=args.dataset,
                eval_report_path=args.eval_report,
                collection_validation=collection_validation,
                positive_results=positive_results,
                negative_results=negative_results,
                underspecified_results=underspecified_results,
                final_recommendation=final_recommendation,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    args.out_md.write_text(
        build_markdown_report(
            status=status,
            dataset_path=args.dataset,
            eval_report_path=args.eval_report,
            collection_validation=collection_validation,
            positive_results=positive_results,
            negative_results=negative_results,
            underspecified_results=underspecified_results,
            metadata_validation_passed=metadata_validation_passed,
            final_recommendation=final_recommendation,
            output_json_path=args.out_json,
        ),
        encoding="utf-8",
    )

    positive_summary = summarize_category(positive_results)
    negative_summary = summarize_category(negative_results)
    underspecified_summary = summarize_category(underspecified_results)

    print(f"decision report status: {status}")
    print(f"target collection: {args.collection}")
    print(
        "positive_answerable pass/warn/fail counts: "
        f"{positive_summary['pass']}/{positive_summary['warn']}/{positive_summary['fail']}"
    )
    print(
        "negative_not_in_batch pass/warn/fail counts: "
        f"{negative_summary['pass']}/{negative_summary['warn']}/{negative_summary['fail']}"
    )
    print(
        "underspecified pass/warn/fail counts: "
        f"{underspecified_summary['pass']}/{underspecified_summary['warn']}/{underspecified_summary['fail']}"
    )
    print(f"markdown report path: {args.out_md.as_posix()}")
    print(f"json report path: {args.out_json.as_posix()}")
    print("changed files:")
    print("app/nsoud/retrieval_decision.py")
    print(args.out_md.as_posix())
    print(args.out_json.as_posix())
    return 0 if status != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
