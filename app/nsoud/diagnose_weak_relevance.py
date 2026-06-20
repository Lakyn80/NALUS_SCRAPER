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

import pandas as pd

from app.nsoud.generate_embeddings import DEFAULT_MODEL_NAME, build_embedder, resolve_device


TARGET_COLLECTION = "nsoud_chunks_section_aware_test_2025_01_03"
OLD_COLLECTION = "nsoud_chunks_test_2025_01_03"
EXPECTED_POINT_COUNT = 1862
EXPECTED_VECTOR_SIZE = 768
EXPECTED_CHUNKING_STRATEGY = "document_section_aware"
DEFAULT_QDRANT_URL = "http://qdrant:6333"
DEFAULT_TOP_K = 10
DEFAULT_CHUNKS_PATH = Path("app/artifacts/nsoud/rag_ready/nsoud_chunks_2025_01_03.parquet")
DEFAULT_OUTPUT_MD = Path(
    "app/artifacts/nsoud/qdrant/nsoud_chunks_section_aware_test_2025_01_03/weak_relevance_diagnostics.md"
)
DEFAULT_OUTPUT_JSON = Path(
    "app/artifacts/nsoud/qdrant/nsoud_chunks_section_aware_test_2025_01_03/weak_relevance_diagnostics.json"
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


@dataclass(frozen=True)
class QueryConfig:
    query: str
    important_terms: list[str]
    generic_terms: list[str]
    expected_section_types: list[str]


@dataclass(frozen=True)
class SemanticResult:
    rank: int
    score: float
    chunk_id: str
    document_id: str
    case_number: str
    document_type: str
    legal_area: str
    section_type: str
    structure_status: str
    chunk_index: str
    total_chunks_in_document: str
    text_preview: str
    metadata_present: bool
    missing_metadata_fields: list[str]
    matches_all_important_terms: bool
    matched_important_terms: list[str]
    matched_generic_terms: list[str]


@dataclass(frozen=True)
class ExactCandidate:
    rank: int
    chunk_id: str
    document_id: str
    case_number: str
    document_type: str
    legal_area: str
    section_type: str
    structure_status: str
    phrase_match: bool
    matched_important_terms: list[str]
    matched_generic_terms: list[str]
    exact_score: int
    in_semantic_top_10: bool
    semantic_rank: int | None
    text_preview: str


@dataclass(frozen=True)
class QueryDiagnosis:
    query: str
    exact_match_exists: bool
    exact_match_count: int
    phrase_match_count: int
    semantic_top_10_count: int
    semantic_top_10_exact_overlap_count: int
    semantic_missed_exact_matches: bool
    likely_failure_reason: str
    recommendation: str
    notes: list[str]
    semantic_results: list[SemanticResult]
    exact_candidates: list[ExactCandidate]


WEAK_QUERY_CONFIGS = [
    QueryConfig(
        query="náhrada nákladů dovolacího řízení",
        important_terms=["náhrada nákladů", "dovolacího řízení"],
        generic_terms=["náklady", "dovolání", "řízení"],
        expected_section_types=["operative_part", "reasoning"],
    ),
    QueryConfig(
        query="zjevně neopodstatněné dovolání",
        important_terms=["zjevně neopodstatněné", "dovolání"],
        generic_terms=["dovolání"],
        expected_section_types=["operative_part", "reasoning"],
    ),
    QueryConfig(
        query="odmítnutí dovolání",
        important_terms=["odmítnutí dovolání"],
        generic_terms=["odmítnutí", "dovolání"],
        expected_section_types=["operative_part", "reasoning"],
    ),
    QueryConfig(
        query="rodinný dům",
        important_terms=["rodinný dům"],
        generic_terms=["dům", "nemovitost", "byt"],
        expected_section_types=["reasoning", "operative_part"],
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose weak search relevance queries for the section-aware NSoud Qdrant collection."
    )
    parser.add_argument("--collection", default=TARGET_COLLECTION, help="Qdrant collection name.")
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL, help="Qdrant base URL.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH, help="Chunk parquet path.")
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUTPUT_MD, help="Output Markdown report path.")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUTPUT_JSON, help="Output JSON report path.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Sentence-transformers model name.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top K semantic results.")
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
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value)


def preview_text(text: str, limit: int = 220) -> str:
    normalized = " ".join(normalize_text(text).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


def simplify_text(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", normalize_text(text)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", ascii_text.lower()).strip()


def contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = f" {simplify_text(text)} "
    normalized_phrase = f" {simplify_text(phrase)} "
    if not normalized_phrase.strip():
        return False
    return normalized_phrase in normalized_text


def matched_terms(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if contains_phrase(text, term)]


def detect_vector_param_size(vectors_config: Any) -> int | None:
    size = getattr(vectors_config, "size", None)
    if size is None and isinstance(vectors_config, dict):
        size = vectors_config.get("size")
    return int(size) if size is not None else None


def verify_collection(client: Any, collection_name: str) -> tuple[int, int]:
    if not client.collection_exists(collection_name):
        raise RuntimeError(f"Collection '{collection_name}' does not exist.")
    info = client.get_collection(collection_name)
    point_count = int(client.count(collection_name=collection_name).count)
    vector_size = detect_vector_param_size(info.config.params.vectors) or 0
    return point_count, vector_size


def get_optional_collection_count(client: Any, collection_name: str) -> int | None:
    if not client.collection_exists(collection_name):
        return None
    return int(client.count(collection_name=collection_name).count)


def validate_metadata(payload: dict[str, Any]) -> list[str]:
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


def run_search(client: Any, *, collection_name: str, vector: list[float], limit: int) -> list[Any]:
    response = client.query_points(
        collection_name=collection_name,
        query=vector,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return list(response.points)


def map_semantic_result(rank: int, point: Any, config: QueryConfig) -> SemanticResult:
    payload = dict(point.payload or {})
    text = normalize_text(payload.get("text"))
    missing_metadata_fields = validate_metadata(payload)
    return SemanticResult(
        rank=rank,
        score=float(point.score),
        chunk_id=normalize_text(payload.get("chunk_id")),
        document_id=normalize_text(payload.get("document_id")),
        case_number=normalize_text(payload.get("case_number")),
        document_type=normalize_text(payload.get("document_type")),
        legal_area=normalize_text(payload.get("legal_area")),
        section_type=normalize_text(payload.get("section_type")),
        structure_status=normalize_text(payload.get("structure_status")),
        chunk_index=normalize_text(payload.get("chunk_index")),
        total_chunks_in_document=normalize_text(payload.get("total_chunks_in_document")),
        text_preview=preview_text(text),
        metadata_present=not missing_metadata_fields,
        missing_metadata_fields=missing_metadata_fields,
        matches_all_important_terms=len(matched_terms(text, config.important_terms)) == len(config.important_terms),
        matched_important_terms=matched_terms(text, config.important_terms),
        matched_generic_terms=matched_terms(text, config.generic_terms),
    )


def load_chunks(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def validate_chunks_dataframe(df: pd.DataFrame) -> None:
    required_columns = [
        "chunk_id",
        "document_id",
        "case_number",
        "document_type",
        "legal_area",
        "section_type",
        "structure_status",
        "chunk_index",
        "total_chunks_in_document",
        "chunk_text",
        "chunking_strategy",
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise RuntimeError(f"Chunk parquet is missing required columns: {', '.join(missing)}")
    invalid_strategy_count = int(
        (df["chunking_strategy"].map(normalize_text) != EXPECTED_CHUNKING_STRATEGY).sum()
    )
    if invalid_strategy_count > 0:
        raise RuntimeError(
            f"Chunk parquet contains {invalid_strategy_count} rows without chunking_strategy='{EXPECTED_CHUNKING_STRATEGY}'."
        )


def build_exact_candidates(
    df: pd.DataFrame,
    config: QueryConfig,
    semantic_chunk_rank: dict[str, int],
) -> tuple[int, int, int, list[ExactCandidate]]:
    records: list[dict[str, Any]] = []
    query_phrase = config.query
    for _, row in df.iterrows():
        chunk_text = normalize_text(row["chunk_text"])
        phrase_match = contains_phrase(chunk_text, query_phrase)
        important_matches = matched_terms(chunk_text, config.important_terms)
        generic_matches = matched_terms(chunk_text, config.generic_terms)
        if not phrase_match and not important_matches and not generic_matches:
            continue
        exact_score = (100 if phrase_match else 0) + (20 * len(important_matches)) + (5 * len(generic_matches))
        records.append(
            {
                "chunk_id": normalize_text(row["chunk_id"]),
                "document_id": normalize_text(row["document_id"]),
                "case_number": normalize_text(row["case_number"]),
                "document_type": normalize_text(row["document_type"]),
                "legal_area": normalize_text(row["legal_area"]),
                "section_type": normalize_text(row["section_type"]),
                "structure_status": normalize_text(row["structure_status"]),
                "phrase_match": phrase_match,
                "matched_important_terms": important_matches,
                "matched_generic_terms": generic_matches,
                "exact_score": exact_score,
                "text_preview": preview_text(chunk_text),
            }
        )

    sorted_records = sorted(
        records,
        key=lambda item: (
            -item["exact_score"],
            item["case_number"],
            item["chunk_id"],
        ),
    )
    candidates: list[ExactCandidate] = []
    semantic_overlap_count = 0
    for rank, item in enumerate(sorted_records[:10], start=1):
        chunk_id = item["chunk_id"]
        semantic_rank = semantic_chunk_rank.get(chunk_id)
        if semantic_rank is not None:
            semantic_overlap_count += 1
        candidates.append(
            ExactCandidate(
                rank=rank,
                chunk_id=chunk_id,
                document_id=item["document_id"],
                case_number=item["case_number"],
                document_type=item["document_type"],
                legal_area=item["legal_area"],
                section_type=item["section_type"],
                structure_status=item["structure_status"],
                phrase_match=item["phrase_match"],
                matched_important_terms=item["matched_important_terms"],
                matched_generic_terms=item["matched_generic_terms"],
                exact_score=item["exact_score"],
                in_semantic_top_10=semantic_rank is not None,
                semantic_rank=semantic_rank,
                text_preview=item["text_preview"],
            )
        )
    return len(sorted_records), sum(1 for item in sorted_records if item["phrase_match"]), semantic_overlap_count, candidates


def classify_query(
    config: QueryConfig,
    semantic_results: list[SemanticResult],
    exact_match_count: int,
    phrase_match_count: int,
    exact_overlap_count: int,
    exact_candidates: list[ExactCandidate],
) -> tuple[str, str, list[str]]:
    notes: list[str] = []
    if not semantic_results:
        notes.append("Semantic search returned zero results for this weak query.")
        return "needs_hybrid_retrieval", "add exact/hybrid retrieval", notes

    top_result = semantic_results[0]
    top_semantic_has_exact_terms = top_result.matches_all_important_terms
    top_section_appropriate = top_result.section_type in config.expected_section_types
    top_score = top_result.score

    if exact_match_count == 0:
        notes.append("No exact candidate chunk contained the tracked phrase or important terms.")
        return "terms_not_present_in_batch", "remove query from generated eval if topic is not actually present in this batch", notes

    if phrase_match_count >= 25 or exact_match_count >= 40:
        notes.append("The query terms are broad and match many chunks exactly, which weakens diagnostic precision.")
        if top_semantic_has_exact_terms and top_score >= 0.65 and top_section_appropriate:
            return "evaluation_label_too_strict", "keep query as-is", notes
        return "query_too_generic", "rewrite query", notes

    if exact_overlap_count == 0:
        notes.append("Exact-match candidates exist, but none appear in semantic top 10.")
        return "semantic_retrieval_missed_exact_match", "add exact/hybrid retrieval", notes

    best_exact_semantic_rank = min(
        (candidate.semantic_rank for candidate in exact_candidates if candidate.semantic_rank is not None),
        default=None,
    )
    if best_exact_semantic_rank is not None and best_exact_semantic_rank > 3:
        notes.append(f"The best exact-match candidate appears only at semantic rank {best_exact_semantic_rank}.")
        return "expected_topic_present_but_low_ranked", "add exact/hybrid retrieval", notes

    if top_semantic_has_exact_terms and top_section_appropriate and top_score >= 0.60:
        notes.append("The top semantic result already matches the important terms and an appropriate section type.")
        return "evaluation_label_too_strict", "keep query as-is", notes

    if not top_section_appropriate and exact_overlap_count > 0:
        notes.append("Relevant chunks are present, but the top result lands in a less appropriate section type.")
        return "needs_hybrid_retrieval", "add exact/hybrid retrieval", notes

    notes.append("Exact candidates are present in semantic top 10, but only partial query evidence is matched.")
    return "query_too_generic", "rewrite query", notes


def diagnose_query(config: QueryConfig, *, client: Any, embedder: Any, chunks_df: pd.DataFrame, top_k: int, collection_name: str) -> QueryDiagnosis:
    vector = embedder.embed_query(config.query)
    raw_results = run_search(client, collection_name=collection_name, vector=vector, limit=top_k)
    semantic_results = [map_semantic_result(rank, point, config) for rank, point in enumerate(raw_results, start=1)]
    semantic_chunk_rank = {result.chunk_id: result.rank for result in semantic_results}
    exact_match_count, phrase_match_count, exact_overlap_count, exact_candidates = build_exact_candidates(
        chunks_df, config, semantic_chunk_rank
    )
    likely_failure_reason, recommendation, notes = classify_query(
        config,
        semantic_results,
        exact_match_count,
        phrase_match_count,
        exact_overlap_count,
        exact_candidates,
    )
    return QueryDiagnosis(
        query=config.query,
        exact_match_exists=exact_match_count > 0,
        exact_match_count=exact_match_count,
        phrase_match_count=phrase_match_count,
        semantic_top_10_count=len(semantic_results),
        semantic_top_10_exact_overlap_count=exact_overlap_count,
        semantic_missed_exact_matches=exact_match_count > 0 and exact_overlap_count == 0,
        likely_failure_reason=likely_failure_reason,
        recommendation=recommendation,
        notes=notes,
        semantic_results=semantic_results,
        exact_candidates=exact_candidates,
    )


def build_markdown_report(
    *,
    status: str,
    collection_name: str,
    point_count: int,
    vector_size: int,
    old_collection_before: int | None,
    old_collection_after: int | None,
    diagnoses: list[QueryDiagnosis],
    queries_with_exact_matches: list[str],
    semantic_missed_exact_queries: list[str],
    hybrid_queries: list[str],
    output_md: Path,
    output_json: Path,
) -> str:
    real_retrieval_problems = [
        item.query
        for item in diagnoses
        if item.likely_failure_reason in {"semantic_retrieval_missed_exact_match", "expected_topic_present_but_low_ranked", "needs_hybrid_retrieval"}
    ]
    generic_or_sparse = [
        item.query
        for item in diagnoses
        if item.likely_failure_reason in {"query_too_generic", "terms_not_present_in_batch", "evaluation_label_too_strict"}
    ]

    lines = [
        "# NSoud Weak Relevance Diagnostics",
        "",
        f"- Status: **{status}**",
        f"- Target collection: `{collection_name}`",
        f"- Point count: **{point_count}**",
        f"- Vector size: **{vector_size}**",
        f"- Weak query count: **{len(diagnoses)}**",
        f"- Queries with exact matches: **{len(queries_with_exact_matches)}**",
        f"- Queries where semantic missed exact matches: **{len(semantic_missed_exact_queries)}**",
        f"- Queries likely needing hybrid retrieval: **{len(hybrid_queries)}**",
        f"- Old collection count before/after: **{old_collection_before} -> {old_collection_after}**",
        f"- Markdown path: `{output_md.as_posix()}`",
        f"- JSON path: `{output_json.as_posix()}`",
        "",
    ]

    for diagnosis in diagnoses:
        lines.extend(
            [
                f"## {diagnosis.query}",
                "",
                f"- Likely failure reason: `{diagnosis.likely_failure_reason}`",
                f"- Recommendation: `{diagnosis.recommendation}`",
                f"- Exact match exists: **{'yes' if diagnosis.exact_match_exists else 'no'}**",
                f"- Exact match candidate count: **{diagnosis.exact_match_count}**",
                f"- Exact phrase-match count: **{diagnosis.phrase_match_count}**",
                f"- Semantic top 10 exact overlap count: **{diagnosis.semantic_top_10_exact_overlap_count}**",
                f"- Semantic missed exact matches: **{'yes' if diagnosis.semantic_missed_exact_matches else 'no'}**",
            ]
        )
        if diagnosis.notes:
            lines.append(f"- Notes: {' | '.join(diagnosis.notes)}")
        else:
            lines.append("- Notes: -")

        lines.extend(
            [
                "",
                "### Semantic Top 10",
                "",
                "| rank | score | case_number | document_type | legal_area | section_type | structure_status | chunk_id | document_id | metadata_present | matched_important_terms | preview |",
                "| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for result in diagnosis.semantic_results:
            lines.append(
                f"| {result.rank} | {result.score:.6f} | {result.case_number or '-'} | {result.document_type or '-'} | "
                f"{result.legal_area or '-'} | {result.section_type or '-'} | {result.structure_status or '-'} | "
                f"{result.chunk_id or '-'} | {result.document_id or '-'} | {'yes' if result.metadata_present else 'no'} | "
                f"{', '.join(result.matched_important_terms) or '-'} | {result.text_preview or '-'} |"
            )

        lines.extend(
            [
                "",
                "### Exact-match Diagnostics",
                "",
                "| rank | exact_score | phrase_match | in_semantic_top_10 | semantic_rank | case_number | document_type | legal_area | section_type | chunk_id | matched_important_terms | matched_generic_terms | preview |",
                "| --- | ---: | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        if diagnosis.exact_candidates:
            for candidate in diagnosis.exact_candidates:
                lines.append(
                    f"| {candidate.rank} | {candidate.exact_score} | {'yes' if candidate.phrase_match else 'no'} | "
                    f"{'yes' if candidate.in_semantic_top_10 else 'no'} | {candidate.semantic_rank or 0} | "
                    f"{candidate.case_number or '-'} | {candidate.document_type or '-'} | {candidate.legal_area or '-'} | "
                    f"{candidate.section_type or '-'} | {candidate.chunk_id or '-'} | "
                    f"{', '.join(candidate.matched_important_terms) or '-'} | {', '.join(candidate.matched_generic_terms) or '-'} | "
                    f"{candidate.text_preview or '-'} |"
                )
        else:
            lines.append("| - | - | - | - | - | - | - | - | - | - | - | - | No exact candidates found. |")
        lines.append("")

    lines.extend(
        [
            "## Final Summary",
            "",
            f"- Real retrieval problems: {', '.join(f'`{query}`' for query in real_retrieval_problems) if real_retrieval_problems else 'none'}",
            f"- Generic or sparse-topic weak queries: {', '.join(f'`{query}`' for query in generic_or_sparse) if generic_or_sparse else 'none'}",
            f"- Hybrid retrieval recommended before production-scale scrape: **{'yes' if hybrid_queries else 'no'}**",
            "",
        ]
    )
    return "\n".join(lines)


def build_json_report(
    *,
    status: str,
    collection_name: str,
    point_count: int,
    vector_size: int,
    old_collection_before: int | None,
    old_collection_after: int | None,
    diagnoses: list[QueryDiagnosis],
    queries_with_exact_matches: list[str],
    semantic_missed_exact_queries: list[str],
    hybrid_queries: list[str],
    output_md: Path,
    output_json: Path,
) -> dict[str, Any]:
    return {
        "status": status,
        "collection_name": collection_name,
        "point_count": point_count,
        "vector_size": vector_size,
        "weak_query_count": len(diagnoses),
        "queries_with_exact_matches": queries_with_exact_matches,
        "semantic_missed_exact_queries": semantic_missed_exact_queries,
        "hybrid_retrieval_queries": hybrid_queries,
        "old_collection_count_before": old_collection_before,
        "old_collection_count_after": old_collection_after,
        "markdown_path": output_md.as_posix(),
        "json_path": output_json.as_posix(),
        "diagnoses": [asdict(item) for item in diagnoses],
    }


def main() -> int:
    args = parse_args()

    if args.top_k <= 0:
        print("diagnostic status: FAIL")
        print("error: --top-k must be greater than 0.")
        return 1

    if args.collection != TARGET_COLLECTION:
        print("diagnostic status: FAIL")
        print(
            f"error: refusing to operate on collection '{args.collection}'. "
            f"Only '{TARGET_COLLECTION}' is allowed."
        )
        return 1

    try:
        chunks_df = load_chunks(args.chunks)
        validate_chunks_dataframe(chunks_df)
    except Exception as exc:
        print("diagnostic status: FAIL")
        print(f"error: {exc}")
        return 1

    warnings.filterwarnings(
        "ignore",
        message=r"Qdrant client version .* is incompatible with server version .*",
    )

    diagnoses: list[QueryDiagnosis] = []
    point_count = 0
    vector_size = 0
    old_collection_before: int | None = None
    old_collection_after: int | None = None

    try:
        from qdrant_client import QdrantClient

        resolved_device = resolve_device(args.device)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            embedder = build_embedder(args.model_name, batch_size=1, device=resolved_device)

        client = QdrantClient(url=args.qdrant_url, timeout=30, check_compatibility=False)
        old_collection_before = get_optional_collection_count(client, OLD_COLLECTION)
        point_count, vector_size = verify_collection(client, args.collection)

        if point_count != EXPECTED_POINT_COUNT:
            raise RuntimeError(f"Target collection point count is {point_count}, expected {EXPECTED_POINT_COUNT}.")
        if vector_size != EXPECTED_VECTOR_SIZE:
            raise RuntimeError(f"Target collection vector size is {vector_size}, expected {EXPECTED_VECTOR_SIZE}.")

        for config in WEAK_QUERY_CONFIGS:
            diagnoses.append(
                diagnose_query(
                    config,
                    client=client,
                    embedder=embedder,
                    chunks_df=chunks_df,
                    top_k=args.top_k,
                    collection_name=args.collection,
                )
            )

        old_collection_after = get_optional_collection_count(client, OLD_COLLECTION)
        if old_collection_before is not None and old_collection_before != old_collection_after:
            raise RuntimeError(
                f"Old collection '{OLD_COLLECTION}' changed from {old_collection_before} to {old_collection_after}."
            )
    except Exception as exc:
        print("diagnostic status: FAIL")
        print(f"error: {exc}")
        return 1

    queries_with_exact_matches = [item.query for item in diagnoses if item.exact_match_exists]
    semantic_missed_exact_queries = [item.query for item in diagnoses if item.semantic_missed_exact_matches]
    hybrid_queries = [
        item.query
        for item in diagnoses
        if item.recommendation == "add exact/hybrid retrieval"
    ]
    status = "PASS"

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(
        build_markdown_report(
            status=status,
            collection_name=args.collection,
            point_count=point_count,
            vector_size=vector_size,
            old_collection_before=old_collection_before,
            old_collection_after=old_collection_after,
            diagnoses=diagnoses,
            queries_with_exact_matches=queries_with_exact_matches,
            semantic_missed_exact_queries=semantic_missed_exact_queries,
            hybrid_queries=hybrid_queries,
            output_md=args.out_md,
            output_json=args.out_json,
        ),
        encoding="utf-8",
    )
    args.out_json.write_text(
        json.dumps(
            build_json_report(
                status=status,
                collection_name=args.collection,
                point_count=point_count,
                vector_size=vector_size,
                old_collection_before=old_collection_before,
                old_collection_after=old_collection_after,
                diagnoses=diagnoses,
                queries_with_exact_matches=queries_with_exact_matches,
                semantic_missed_exact_queries=semantic_missed_exact_queries,
                hybrid_queries=hybrid_queries,
                output_md=args.out_md,
                output_json=args.out_json,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"diagnostic status: {status}")
    print(f"weak query count: {len(diagnoses)}")
    print(f"queries with exact matches: {len(queries_with_exact_matches)}")
    print(f"queries where semantic missed exact matches: {len(semantic_missed_exact_queries)}")
    print(f"queries likely needing hybrid retrieval: {len(hybrid_queries)}")
    print(f"output markdown path: {args.out_md.as_posix()}")
    print(f"output json path: {args.out_json.as_posix()}")
    print("changed files:")
    print("app/nsoud/diagnose_weak_relevance.py")
    print(args.out_md.as_posix())
    print(args.out_json.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
