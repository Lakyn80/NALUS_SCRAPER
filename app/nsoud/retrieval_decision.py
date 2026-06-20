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
    short_preview: str


@dataclass(frozen=True)
class QueryAnalysis:
    top_score: float | None
    second_score: float | None
    score_gap: float | None
    top_result_count: int
    strong_result_count: int
    direct_evidence_count: int
    distinct_documents: int
    distinct_legal_areas: int
    distinct_section_types: int
    generic_result_count: int
    metadata_validation_passed: bool
    query_term_overlap_count: int
    source_term_overlap_count: int
    broad_query: bool


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
    matched_terms: list[str]
    missing_terms: list[str]
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


def map_result(rank: int, point: Any, *, query_terms: list[str], source_terms: list[str]) -> TopResultSummary:
    payload = dict(point.payload or {})
    text = normalize_text(payload.get("text"))
    missing_metadata_fields = validate_payload_metadata(payload)
    return TopResultSummary(
        rank=rank,
        score=float(point.score),
        case_number=normalize_text(payload.get("case_number")),
        document_type=normalize_text(payload.get("document_type")),
        legal_area=normalize_text(payload.get("legal_area")),
        section_type=normalize_text(payload.get("section_type")),
        chunk_id=normalize_text(payload.get("chunk_id")),
        document_id=normalize_text(payload.get("document_id")),
        metadata_present=not missing_metadata_fields,
        missing_metadata_fields=missing_metadata_fields,
        generic_procedural_fragment=is_generic_procedural_result(payload),
        matched_query_terms=matched_terms_in_text(text, query_terms),
        matched_source_terms=matched_terms_in_text(text, source_terms),
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
    top_score = results[0].score if results else None
    second_score = results[1].score if len(results) > 1 else None
    score_gap = (top_score - second_score) if top_score is not None and second_score is not None else None
    strong_result_count = sum(1 for result in results if result.score >= strong_score_threshold)
    direct_evidence_count = 0
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
            direct_evidence_count += 1

    distinct_documents = len({result.document_id for result in results if result.document_id})
    distinct_legal_areas = len({result.legal_area for result in results if result.legal_area})
    distinct_section_types = len({result.section_type for result in results if result.section_type})
    generic_result_count = sum(1 for result in results if result.generic_procedural_fragment)
    metadata_validation_passed = all(result.metadata_present for result in results)
    weak_classification = normalize_text((weak_query_info or {}).get("primary_classification"))
    broad_query = (
        len(query_term_stems) <= 2
        or weak_classification == "too_generic"
        or distinct_documents >= 6
        or distinct_legal_areas >= 2
        or distinct_section_types >= 3
    )

    return QueryAnalysis(
        top_score=top_score,
        second_score=second_score,
        score_gap=score_gap,
        top_result_count=len(results),
        strong_result_count=strong_result_count,
        direct_evidence_count=direct_evidence_count,
        distinct_documents=distinct_documents,
        distinct_legal_areas=distinct_legal_areas,
        distinct_section_types=distinct_section_types,
        generic_result_count=generic_result_count,
        metadata_validation_passed=metadata_validation_passed,
        query_term_overlap_count=query_term_overlap_count,
        source_term_overlap_count=source_term_overlap_count,
        broad_query=broad_query,
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
    expected_sections = {
        normalize_text(value)
        for value in item.get("expected_section_types", [])
        if normalize_text(value)
    }
    top_section_match = any(result.section_type in expected_sections for result in results[:3])
    early_source_match = any(result.matched_source_terms for result in results[:3])
    early_case_match = any(
        result.case_number in {normalize_text(value) for value in item.get("source_case_numbers", []) if normalize_text(value)}
        for result in results[:5]
    )
    if not results:
        return "insufficient_support", 0.0, "No results were retrieved for a query that should be answerable."
    if not analysis.metadata_validation_passed:
        return "insufficient_support", 0.2, "Retrieved results are missing required section-aware metadata."
    if analysis.direct_evidence_count >= 1 and analysis.strong_result_count >= 1:
        confidence = confidence_from_score(0.82 + min(0.14, analysis.direct_evidence_count * 0.03))
        return "answerable", confidence, "Top results contain direct source evidence for the expected answerable query."
    if early_source_match and top_section_match:
        confidence = confidence_from_score(0.78 + min(0.08, (analysis.top_score or 0.0) * 0.1))
        return "answerable", confidence, "Top results contain source-term overlap in the expected section context."
    if early_case_match and analysis.top_score is not None and analysis.top_score >= 0.55:
        confidence = confidence_from_score(0.76 + min(0.08, analysis.top_score * 0.1))
        return "answerable", confidence, "Source case context appears early enough to support an answer."
    if top_section_match and analysis.query_term_overlap_count >= 3 and analysis.top_score is not None and analysis.top_score >= 0.55:
        confidence = confidence_from_score(0.74 + min(0.08, analysis.top_score * 0.08))
        return "answerable", confidence, "Results contain sufficiently close legal context even though direct evidence is indirect."
    return "insufficient_support", 0.45, f"Retrieved context for '{query}' stays too indirect to support an answer deterministically."


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
    if analysis.direct_evidence_count == 0 and not matched_missing_terms and analysis.generic_result_count >= 2:
        return "insufficient_support", 0.92, "Top hits are generic or off-topic and do not provide direct support for the requested issue."
    if analysis.direct_evidence_count == 0 and not matched_missing_terms and (analysis.top_score or 0.0) < 0.80:
        return "insufficient_support", 0.88, "The current collection returns only indirect context and lacks direct support for this query."
    if analysis.direct_evidence_count == 0 and matched_missing_terms and (analysis.top_score or 0.0) < 0.80:
        return "insufficient_support", 0.84, "Some vocabulary overlaps appear, but the results remain too weak and indirect to support an answer."
    if analysis.direct_evidence_count == 0 and analysis.distinct_documents >= 4:
        return "insufficient_support", 0.82, "The query disperses across unrelated documents without surfacing direct support."
    return "answerable", 0.51, "The unsupported query unexpectedly retrieved narrow evidence and should be reviewed."


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
    if analysis.generic_result_count >= 3 and analysis.direct_evidence_count == 0:
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
        mapped_results = [
            map_result(rank, point, query_terms=query_terms, source_terms=source_terms)
            for rank, point in enumerate(raw_results, start=1)
        ]
        analysis = analyze_results(
            query=query,
            results=mapped_results,
            source_terms=source_terms,
            source_chunk_ids={normalize_text(value) for value in item.get("source_chunk_ids", []) if normalize_text(value)},
            source_case_numbers={normalize_text(value) for value in item.get("source_case_numbers", []) if normalize_text(value)},
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
                matched_terms=matched_terms,
                missing_terms=missing_terms,
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
        "| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata | preview |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            f"| {result.rank} | {result.score:.6f} | {result.case_number or '-'} | {result.document_type or '-'} | "
            f"{result.legal_area or '-'} | {result.section_type or '-'} | {result.chunk_id or '-'} | "
            f"{result.document_id or '-'} | {'PASS' if result.metadata_present else 'FAIL'} | {result.short_preview or '-'} |"
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
                f"- Matched terms: {', '.join(item.matched_terms) if item.matched_terms else '-'}",
                f"- Missing terms: {', '.join(item.missing_terms) if item.missing_terms else '-'}",
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
