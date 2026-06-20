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
from typing import Any

from app.nsoud.generate_embeddings import DEFAULT_MODEL_NAME, build_embedder, resolve_device


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
DEFAULT_MARKDOWN_PATH = Path(
    "app/artifacts/nsoud/qdrant/nsoud_chunks_section_aware_test_2025_01_03/search_relevance_eval.md"
)
DEFAULT_JSON_PATH = Path(
    "app/artifacts/nsoud/qdrant/nsoud_chunks_section_aware_test_2025_01_03/search_relevance_eval.json"
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
    "podl",
    "odst",
    "pism",
    "jak",
    "jako",
    "kter",
    "nejs",
    "soud",
    "rizen",
    "prav",
    "podm",
    "rozhodnut",
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
class SearchResultRow:
    rank: int
    score: float
    case_number: str
    document_type: str
    legal_area: str
    section_type: str
    chunk_id: str
    document_id: str
    section_id: str
    chunk_index: str
    chunk_index_in_section: str
    total_chunks_in_document: str
    total_chunks_in_section: str
    previous_chunk_id: str
    next_chunk_id: str
    structure_status: str
    chunking_strategy: str
    short_preview: str
    metadata_present: bool
    missing_metadata_fields: list[str]
    matched_source_terms: list[str]
    matched_missing_context_terms: list[str]


@dataclass(frozen=True)
class QueryResult:
    query: str
    category: str
    expected_behavior: str
    actual_label: str
    notes: list[str]
    result_count: int
    top_score: float | None
    metadata_validation_passed: bool
    all_results: list[SearchResultRow]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run category-aware search relevance evaluation for the section-aware NSoud Qdrant collection."
    )
    parser.add_argument("--collection", default=TARGET_COLLECTION, help="Qdrant collection name.")
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL, help="Qdrant base URL.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="Categorized evaluation dataset JSON.")
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MARKDOWN_PATH, help="Output Markdown report path.")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON_PATH, help="Output JSON report path.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Top N results per query.")
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
        if value != value:  # NaN
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


def contains_phrase(text: str, phrase: str) -> bool:
    haystack = f" {simplify_text(text)} "
    needle = f" {simplify_text(phrase)} "
    if not needle.strip():
        return False
    return needle in haystack


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


def matched_terms(text: str, terms: list[str]) -> list[str]:
    if not terms:
        return []
    haystack = tokenize_stems(text)
    matches: list[str] = []
    for term in terms:
        term_stems = tokenize_stems(term)
        if term_stems and all(any(stem in token for token in haystack) for stem in term_stems):
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


def map_result(rank: int, point: Any, *, source_terms: list[str], missing_context_terms: list[str]) -> SearchResultRow:
    payload = dict(point.payload or {})
    text = normalize_text(payload.get("text"))
    missing_metadata_fields = validate_payload_metadata(payload)
    return SearchResultRow(
        rank=rank,
        score=float(point.score),
        case_number=normalize_text(payload.get("case_number")),
        document_type=normalize_text(payload.get("document_type")),
        legal_area=normalize_text(payload.get("legal_area")),
        section_type=normalize_text(payload.get("section_type")),
        chunk_id=normalize_text(payload.get("chunk_id")),
        document_id=normalize_text(payload.get("document_id")),
        section_id=normalize_text(payload.get("section_id")),
        chunk_index=normalize_text(payload.get("chunk_index")),
        chunk_index_in_section=normalize_text(payload.get("chunk_index_in_section")),
        total_chunks_in_document=normalize_text(payload.get("total_chunks_in_document")),
        total_chunks_in_section=normalize_text(payload.get("total_chunks_in_section")),
        previous_chunk_id=normalize_text(payload.get("previous_chunk_id")),
        next_chunk_id=normalize_text(payload.get("next_chunk_id")),
        structure_status=normalize_text(payload.get("structure_status")),
        chunking_strategy=normalize_text(payload.get("chunking_strategy")),
        short_preview=preview_text(text),
        metadata_present=not missing_metadata_fields,
        missing_metadata_fields=missing_metadata_fields,
        matched_source_terms=matched_terms(text, source_terms),
        matched_missing_context_terms=[term for term in missing_context_terms if contains_phrase(text, term)],
    )


def load_dataset(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
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


def run_search(client: Any, *, collection_name: str, vector: list[float], limit: int) -> list[Any]:
    response = client.query_points(
        collection_name=collection_name,
        query=vector,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return list(response.points)


def classify_positive(item: dict[str, Any], results: list[SearchResultRow]) -> tuple[str, list[str]]:
    notes: list[str] = []
    if not results:
        return "FAIL", ["positive query returned zero results"]

    source_chunk_ids = {normalize_text(value) for value in item.get("source_chunk_ids", []) if normalize_text(value)}
    source_case_numbers = {normalize_text(value) for value in item.get("source_case_numbers", []) if normalize_text(value)}
    expected_sections = {normalize_text(value) for value in item.get("expected_section_types", []) if normalize_text(value)}
    evidence_stems = significant_stems([normalize_text(value) for value in item.get("source_terms", []) if normalize_text(value)])

    chunk_hit_ranks = [result.rank for result in results if result.chunk_id in source_chunk_ids]
    case_hit_ranks = [result.rank for result in results if result.case_number in source_case_numbers]
    section_hit_ranks = [result.rank for result in results if result.section_type in expected_sections]
    strong_term_ranks = [result.rank for result in results if result.matched_source_terms]
    stem_overlap_ranks = [
        result.rank
        for result in results
        if len(evidence_stems.intersection(set(tokenize_stems(result.short_preview)))) >= 2
    ]
    top_result = results[0]

    if chunk_hit_ranks:
        notes.append(f"source chunk evidence found at rank {min(chunk_hit_ranks)}")
        return "PASS", notes
    if case_hit_ranks and strong_term_ranks:
        notes.append(
            f"source case evidence found at rank {min(case_hit_ranks)} with source-term overlap at rank {min(strong_term_ranks)}"
        )
        return "PASS", notes
    if strong_term_ranks and top_result.section_type in expected_sections and top_result.score >= 0.70:
        notes.append(
            f"top result matches source terms with expected section type and score {top_result.score:.3f}"
        )
        return "PASS", notes
    if strong_term_ranks and min(strong_term_ranks) <= 3:
        notes.append(f"source-term overlap appears within top {min(strong_term_ranks)} results")
        return "PASS", notes
    if case_hit_ranks or (section_hit_ranks and strong_term_ranks):
        if case_hit_ranks:
            notes.append(f"source case evidence appears only indirectly at rank {min(case_hit_ranks)}")
        if section_hit_ranks and strong_term_ranks:
            notes.append(
                f"expected section and source-term overlap present but only indirectly (section rank {min(section_hit_ranks)}, term rank {min(strong_term_ranks)})"
            )
        return "WARN", notes
    if stem_overlap_ranks and top_result.score >= 0.50:
        notes.append(
            f"query stem overlap appears in top {min(stem_overlap_ranks)} results with score {top_result.score:.3f}, but source evidence is indirect"
        )
        return "WARN", notes
    if section_hit_ranks and top_result.score >= 0.55:
        notes.append(
            f"expected section context appears by rank {min(section_hit_ranks)} with top score {top_result.score:.3f}, but source evidence is indirect"
        )
        return "WARN", notes
    if top_result.score >= 0.55 and top_result.section_type in expected_sections:
        notes.append(
            f"results stay in an expected section type with score {top_result.score:.3f}, but source evidence is indirect"
        )
        return "WARN", notes
    notes.append("no source chunk, source case, or strong source-term overlap found in top 10")
    return "FAIL", notes


def classify_negative(item: dict[str, Any], results: list[SearchResultRow]) -> tuple[str, list[str]]:
    notes: list[str] = []
    if not results:
        notes.append("top 10 returned zero results, which is acceptable for insufficient-support detection")
        return "PASS", notes

    top_score = results[0].score
    direct_missing_term_hits = [result.rank for result in results if result.matched_missing_context_terms]
    unique_docs = len({result.document_id for result in results if result.document_id})
    unique_areas = len({result.legal_area for result in results if result.legal_area})

    if direct_missing_term_hits:
        notes.append(f"unexpected missing-context phrase overlap found at rank {min(direct_missing_term_hits)}")
        if top_score < 0.65 and unique_docs >= 3:
            notes.append("the phrase overlap appears incidental rather than as direct support")
            return "PASS", notes
        if top_score >= 0.80:
            return "FAIL", notes
        return "WARN", notes
    if top_score < 0.55 and unique_docs >= 4:
        notes.append("results are weak and dispersed, which matches insufficient-support behavior")
        return "PASS", notes
    if top_score < 0.68 and unique_areas >= 2:
        notes.append("results are ambiguous across legal contexts and do not show direct support")
        return "PASS", notes
    if top_score < 0.80 and unique_docs >= 3:
        notes.append("results do not contain direct support and remain broad enough for insufficient-support handling")
        return "PASS", notes
    if top_score < 0.85:
        notes.append("results exist but are ambiguous enough that dataset review may be needed")
        return "WARN", notes
    notes.append("unsupported query retrieved unexpectedly strong and narrow results")
    return "FAIL", notes


def classify_underspecified(item: dict[str, Any], results: list[SearchResultRow]) -> tuple[str, list[str]]:
    notes: list[str] = []
    if not results:
        notes.append("zero results still indicate the query cannot be answered directly without clarification")
        return "PASS", notes

    unique_docs = len({result.document_id for result in results if result.document_id})
    unique_sections = len({result.section_type for result in results if result.section_type})
    unique_areas = len({result.legal_area for result in results if result.legal_area})
    top_score = results[0].score

    if unique_docs >= 4 or unique_sections >= 3 or unique_areas >= 2:
        notes.append(
            f"results span {unique_docs} documents, {unique_sections} section types, and {unique_areas} legal areas"
        )
        return "PASS", notes
    if unique_docs >= 2:
        notes.append(f"query narrows to {unique_docs} documents, so clarification is still advisable")
        return "WARN", notes
    if top_score >= 0.85:
        notes.append("query collapsed into a single narrow interpretation with very strong score")
        return "FAIL", notes
    notes.append("query is broad but top results cluster into a narrow interpretation that may need review")
    return "WARN", notes


def summarize_category(results: list[QueryResult]) -> dict[str, int]:
    return {
        "pass": sum(1 for item in results if item.actual_label == "PASS"),
        "warn": sum(1 for item in results if item.actual_label == "WARN"),
        "fail": sum(1 for item in results if item.actual_label == "FAIL"),
        "total": len(results),
    }


def build_result_payload(
    *,
    status: str,
    collection_validation: CollectionValidation,
    positive_results: list[QueryResult],
    negative_results: list[QueryResult],
    underspecified_results: list[QueryResult],
    dataset_path: Path,
    final_recommendation: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "target_collection": TARGET_COLLECTION,
        "dataset_path": dataset_path.as_posix(),
        "collection_validation": asdict(collection_validation),
        "summary": {
            "positive_answerable": summarize_category(positive_results),
            "negative_not_in_batch": summarize_category(negative_results),
            "underspecified": summarize_category(underspecified_results),
        },
        "positive_answerable_results": [asdict(item) for item in positive_results],
        "negative_not_in_batch_results": [asdict(item) for item in negative_results],
        "underspecified_results": [asdict(item) for item in underspecified_results],
        "final_recommendation": final_recommendation,
    }


def render_summary_table(title: str, results: list[QueryResult]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| query | expected_behavior | actual_label | top_score | result_count | metadata_validation | notes |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for item in results:
        score = f"{item.top_score:.6f}" if item.top_score is not None else "-"
        lines.append(
            f"| {item.query} | {item.expected_behavior} | {item.actual_label} | {score} | {item.result_count} | "
            f"{'PASS' if item.metadata_validation_passed else 'FAIL'} | {' / '.join(item.notes) or '-'} |"
        )
    lines.append("")
    return lines


def render_detailed_results(results: list[QueryResult]) -> list[str]:
    lines: list[str] = []
    for item in results:
        lines.extend(
            [
                f"### {item.query}",
                "",
                f"- Expected behavior: `{item.expected_behavior}`",
                f"- Actual label: **{item.actual_label}**",
                f"- Metadata validation: **{'PASS' if item.metadata_validation_passed else 'FAIL'}**",
                f"- Result count: **{item.result_count}**",
                f"- Top score: **{item.top_score:.6f}**" if item.top_score is not None else "- Top score: `n/a`",
                f"- Notes: {' | '.join(item.notes) if item.notes else '-'}",
                "",
                "| rank | score | case_number | document_type | legal_area | section_type | chunk_id | document_id | metadata_validation | preview |",
                "| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for result in item.all_results:
            lines.append(
                f"| {result.rank} | {result.score:.6f} | {result.case_number or '-'} | {result.document_type or '-'} | "
                f"{result.legal_area or '-'} | {result.section_type or '-'} | {result.chunk_id or '-'} | "
                f"{result.document_id or '-'} | {'PASS' if result.metadata_present else 'FAIL'} | {result.short_preview or '-'} |"
            )
        lines.append("")
    return lines


def build_markdown_report(
    *,
    status: str,
    dataset_path: Path,
    collection_validation: CollectionValidation,
    positive_results: list[QueryResult],
    negative_results: list[QueryResult],
    underspecified_results: list[QueryResult],
    metadata_validation_passed: bool,
    final_recommendation: str,
    output_json_path: Path,
) -> str:
    lines = [
        "# NSoud Search Relevance Evaluation",
        "",
        f"- Status: **{status}**",
        f"- Target collection: `{TARGET_COLLECTION}`",
        f"- Dataset path: `{dataset_path.as_posix()}`",
        f"- Collection exists: **{'yes' if collection_validation.exists else 'no'}**",
        f"- Collection point count: **{collection_validation.point_count}**",
        f"- Collection vector size: **{collection_validation.vector_size}**",
        f"- Old collection unchanged: **{collection_validation.old_collection_unchanged}**",
        f"- Metadata validation: **{'PASS' if metadata_validation_passed else 'FAIL'}**",
        f"- JSON report path: `{output_json_path.as_posix()}`",
        "",
    ]
    lines.extend(render_summary_table("Positive Answerable Summary", positive_results))
    lines.extend(render_summary_table("Negative Not In Batch Summary", negative_results))
    lines.extend(render_summary_table("Underspecified Summary", underspecified_results))
    lines.extend(["## Per-Query Top Results", ""])
    lines.extend(render_detailed_results(positive_results))
    lines.extend(render_detailed_results(negative_results))
    lines.extend(render_detailed_results(underspecified_results))
    lines.extend(["## Final Recommendation", f"- {final_recommendation}", ""])
    return "\n".join(lines)


def build_final_recommendation(
    status: str,
    positive_summary: dict[str, int],
    negative_summary: dict[str, int],
    underspecified_summary: dict[str, int],
) -> str:
    if status == "PASS":
        return "PASS: category-aware evaluation behaves as expected across answerable, unsupported, and underspecified query classes."
    if status == "WARN":
        return (
            "WARN: collection integrity and metadata are intact, but some queries remain indirect or need dataset-review follow-up "
            "before this evaluation can be treated as a stronger regression gate."
        )
    return (
        "FAIL: collection validation, metadata coverage, or category-specific query behavior failed; "
        "fix those issues before relying on this evaluation."
    )


def determine_status(
    *,
    collection_validation: CollectionValidation,
    metadata_validation_passed: bool,
    positive_results: list[QueryResult],
    negative_results: list[QueryResult],
    underspecified_results: list[QueryResult],
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
    if any(item.result_count == 0 for item in positive_results):
        return "FAIL"
    if any(item.actual_label == "FAIL" for item in positive_results):
        return "FAIL"
    if any(item.actual_label == "FAIL" for item in negative_results):
        return "FAIL"
    if any(item.actual_label == "FAIL" for item in underspecified_results):
        return "FAIL"
    if any(item.actual_label == "WARN" for item in positive_results):
        return "WARN"
    if any(item.actual_label == "WARN" for item in negative_results):
        return "WARN"
    if any(item.actual_label == "WARN" for item in underspecified_results):
        return "WARN"
    return "PASS"


def evaluate_queries(
    *,
    client: Any,
    embedder: Any,
    collection_name: str,
    limit: int,
    category: str,
    expected_behavior: str,
    items: list[dict[str, Any]],
) -> list[QueryResult]:
    results: list[QueryResult] = []
    for item in items:
        query = normalize_text(item.get("query")).strip()
        if not query:
            continue
        vector = embedder.embed_query(query)
        raw_results = run_search(client, collection_name=collection_name, vector=vector, limit=limit)
        missing_context_terms = [
            normalize_text(value)
            for value in item.get("missing_terms_or_context", [])
            if normalize_text(value)
        ]
        mapped_results = [
            map_result(
                rank,
                point,
                source_terms=[normalize_text(value) for value in item.get("source_terms", []) if normalize_text(value)],
                missing_context_terms=missing_context_terms,
            )
            for rank, point in enumerate(raw_results, start=1)
        ]
        metadata_ok = all(result.metadata_present for result in mapped_results)
        if category == "positive_answerable":
            label, notes = classify_positive(item, mapped_results)
        elif category == "negative_not_in_batch":
            label, notes = classify_negative(item, mapped_results)
        else:
            label, notes = classify_underspecified(item, mapped_results)
        results.append(
            QueryResult(
                query=query,
                category=category,
                expected_behavior=expected_behavior,
                actual_label=label,
                notes=notes,
                result_count=len(mapped_results),
                top_score=mapped_results[0].score if mapped_results else None,
                metadata_validation_passed=metadata_ok,
                all_results=mapped_results,
            )
        )
    return results


def main() -> int:
    args = parse_args()

    if args.limit <= 0:
        print("eval status: FAIL")
        print("error: --limit must be greater than 0.")
        return 1

    if args.collection != TARGET_COLLECTION:
        print("eval status: FAIL")
        print(
            f"error: refusing to operate on collection '{args.collection}'. "
            f"Only '{TARGET_COLLECTION}' is allowed."
        )
        return 1

    try:
        dataset = load_dataset(args.dataset)
    except Exception as exc:
        print("eval status: FAIL")
        print(f"error: {exc}")
        return 1

    positive_results: list[QueryResult] = []
    negative_results: list[QueryResult] = []
    underspecified_results: list[QueryResult] = []

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
            positive_results = evaluate_queries(
                client=client,
                embedder=embedder,
                collection_name=args.collection,
                limit=args.limit,
                category="positive_answerable",
                expected_behavior="retrieval_returns_relevant_chunks",
                items=dataset["positive_answerable"],
            )
            negative_results = evaluate_queries(
                client=client,
                embedder=embedder,
                collection_name=args.collection,
                limit=args.limit,
                category="negative_not_in_batch",
                expected_behavior="insufficient_support",
                items=dataset["negative_not_in_batch"],
            )
            underspecified_results = evaluate_queries(
                client=client,
                embedder=embedder,
                collection_name=args.collection,
                limit=args.limit,
                category="underspecified",
                expected_behavior="ask_for_clarification",
                items=dataset["underspecified"],
            )
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
        print("eval status: FAIL")
        print(f"error: {exc}")
        return 1

    metadata_validation_passed = all(
        item.metadata_validation_passed
        for item in [*positive_results, *negative_results, *underspecified_results]
    )
    status = determine_status(
        collection_validation=collection_validation,
        metadata_validation_passed=metadata_validation_passed,
        positive_results=positive_results,
        negative_results=negative_results,
        underspecified_results=underspecified_results,
    )
    positive_summary = summarize_category(positive_results)
    negative_summary = summarize_category(negative_results)
    underspecified_summary = summarize_category(underspecified_results)
    final_recommendation = build_final_recommendation(
        status,
        positive_summary,
        negative_summary,
        underspecified_summary,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            build_result_payload(
                status=status,
                collection_validation=collection_validation,
                positive_results=positive_results,
                negative_results=negative_results,
                underspecified_results=underspecified_results,
                dataset_path=args.dataset,
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

    print(f"eval status: {status}")
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
    print(f"metadata validation: {'PASSED' if metadata_validation_passed else 'FAILED'}")
    print(f"markdown report path: {args.out_md.as_posix()}")
    print(f"json report path: {args.out_json.as_posix()}")
    print("changed files:")
    print("app/nsoud/evaluate_search_relevance.py")
    print(args.out_md.as_posix())
    print(args.out_json.as_posix())
    return 0 if status != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
