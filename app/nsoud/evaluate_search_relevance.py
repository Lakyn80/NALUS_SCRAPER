from __future__ import annotations

import argparse
import contextlib
import io
import re
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.nsoud.generate_embeddings import DEFAULT_MODEL_NAME, build_embedder, resolve_device


DEFAULT_COLLECTION = "nsoud_chunks_test_2025_01_03"
DEFAULT_LIMIT = 5
QUERIES = [
    "nutná obrana dovolání",
    "místní příslušnost exekuce povinný nemá pobyt v České republice",
    "skryté vady kupní smlouva nemovitost dovolání",
    "dovolání odmítnuto podle § 265i odst. 1 písm. e)",
    "určení místní příslušnosti podle § 11 odst. 3 občanského soudního řádu",
    "odpovědnost za vady rodinný dům skryté vady",
    "povinný nemá místo pobytu na území České republiky",
    "náhrada nákladů dovolacího řízení",
    "přípustnost dovolání Nejvyšší soud",
    "zjevně neopodstatněné dovolání",
]
QUERY_CONCEPTS: dict[str, list[list[str]]] = {
    "nutná obrana dovolání": [["nutn", "obran"], ["dovol"]],
    "místní příslušnost exekuce povinný nemá pobyt v České republice": [
        ["mistni", "prislusnost"],
        ["exekuc"],
        ["povin"],
        ["pobyt", "cesk"],
    ],
    "skryté vady kupní smlouva nemovitost dovolání": [
        ["skryt", "vad"],
        ["kupni", "smlouv"],
        ["nemovit"],
        ["dovolani"],
    ],
    "dovolání odmítnuto podle § 265i odst. 1 písm. e)": [
        ["265i"],
        ["odmit"],
        ["dovolani"],
    ],
    "určení místní příslušnosti podle § 11 odst. 3 občanského soudního řádu": [
        ["11"],
        ["mistni", "prislusnost"],
        ["os", "r"],
    ],
    "odpovědnost za vady rodinný dům skryté vady": [
        ["odpovednost", "vad"],
        ["skryt", "vad"],
        ["rodin", "dum"],
    ],
    "povinný nemá místo pobytu na území České republiky": [
        ["povin"],
        ["misto", "pobyt"],
        ["cesk"],
    ],
    "náhrada nákladů dovolacího řízení": [
        ["nahrad", "naklad"],
        ["dovolac", "rizen"],
    ],
    "přípustnost dovolání Nejvyšší soud": [
        ["pripust"],
        ["dovolani"],
        ["nejvyss"],
    ],
    "zjevně neopodstatněné dovolání": [
        ["zjevne", "neopodstat"],
        ["dovolani"],
    ],
}


@dataclass(frozen=True)
class SearchResultRow:
    rank: int
    score: float
    case_number: str
    ecli: str
    decision_date: str
    document_type: str
    legal_area: str
    ns_section_hint: str
    chunk_id: str
    url: str
    text_preview: str


@dataclass(frozen=True)
class QueryEvaluation:
    query: str
    result_count: int
    top_score: float | None
    top_case_number: str
    top_document_type: str
    top_legal_area: str
    top_chunk_id: str
    top_url: str
    top_text_preview: str
    label: str
    matched_concepts: int
    total_concepts: int
    rationale: str
    all_results: list[SearchResultRow]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run manual retrieval relevance evaluation against NSoud Qdrant.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Qdrant collection name.")
    parser.add_argument("--qdrant-url", default="http://qdrant:6333", help="Qdrant base URL.")
    parser.add_argument("--out", type=Path, required=True, help="Output Markdown report path.")
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
    return str(value)


def text_preview(text: str, limit: int = 700) -> str:
    normalized = " ".join(normalize_text(text).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


def simplify_text(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", ascii_text.lower()).strip()


def verify_collection(client: Any, collection_name: str) -> int:
    if not client.collection_exists(collection_name):
        raise RuntimeError(f"Collection '{collection_name}' does not exist.")
    info = client.get_collection(collection_name)
    vectors = getattr(info.config.params, "vectors", None)
    size = getattr(vectors, "size", None)
    return int(size or 0)


def run_search(client: Any, *, collection_name: str, vector: list[float], limit: int) -> list[Any]:
    response = client.query_points(
        collection_name=collection_name,
        query=vector,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return list(response.points)


def map_result(rank: int, point: Any) -> SearchResultRow:
    payload = dict(point.payload or {})
    return SearchResultRow(
        rank=rank,
        score=float(point.score),
        case_number=normalize_text(payload.get("case_number")),
        ecli=normalize_text(payload.get("ecli")),
        decision_date=normalize_text(payload.get("decision_date")),
        document_type=normalize_text(payload.get("document_type")),
        legal_area=normalize_text(payload.get("legal_area")),
        ns_section_hint=normalize_text(payload.get("ns_section_hint")),
        chunk_id=normalize_text(payload.get("chunk_id")),
        url=normalize_text(payload.get("url")),
        text_preview=text_preview(normalize_text(payload.get("text"))),
    )


def evaluate_label(query: str, top_result: SearchResultRow) -> tuple[str, int, int, str]:
    concept_groups = QUERY_CONCEPTS[query]
    searchable = " ".join(
        [
            top_result.case_number,
            top_result.document_type,
            top_result.legal_area,
            top_result.ns_section_hint,
            top_result.text_preview,
        ]
    )
    normalized = simplify_text(searchable)

    matched_groups: list[str] = []
    matched_group_sizes: list[int] = []
    for group in concept_groups:
        if all(token in normalized for token in group):
            matched_groups.append(" + ".join(group))
            matched_group_sizes.append(len(group))

    matched_count = len(matched_groups)
    total_count = len(concept_groups)
    core_phrase_match = any(size >= 2 for size in matched_group_sizes)
    if matched_count >= max(2, total_count - 1) or (core_phrase_match and top_result.score >= 0.65):
        label = "STRONG"
    elif matched_count >= 1:
        label = "MEDIUM"
    else:
        label = "WEAK"

    if matched_groups:
        rationale = f"matched concept groups: {', '.join(matched_groups)}"
    else:
        rationale = "no configured concept group matched the top result text/metadata"
    return label, matched_count, total_count, rationale


def build_report(
    *,
    status: str,
    collection_name: str,
    limit: int,
    evaluations: list[QueryEvaluation],
    strong_count: int,
    medium_count: int,
    weak_count: int,
    errors: list[str],
) -> str:
    total_result_rows = sum(item.result_count for item in evaluations)
    lines = [
        "# NSoud Search Relevance Evaluation",
        "",
        f"- Status: **{status}**",
        f"- Collection name: `{collection_name}`",
        f"- Query count: **{len(evaluations)}**",
        f"- Limit: **{limit}**",
        f"- Total result rows: **{total_result_rows}**",
        f"- STRONG count: **{strong_count}**",
        f"- MEDIUM count: **{medium_count}**",
        f"- WEAK count: **{weak_count}**",
        "",
        "## Query Summary",
        "",
        "| query | label | top_score | top_case_number | top_document_type | top_legal_area | top_chunk_id |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for item in evaluations:
        score = f"{item.top_score:.6f}" if item.top_score is not None else ""
        lines.append(
            f"| {item.query} | {item.label} | {score} | {item.top_case_number or '-'} | "
            f"{item.top_document_type or '-'} | {item.top_legal_area or '-'} | {item.top_chunk_id or '-'} |"
        )

    lines.extend(["", "## Detailed Results"])
    for item in evaluations:
        lines.extend(
            [
                "",
                f"### {item.query}",
                "",
                f"- Label: **{item.label}**",
                f"- Result count: **{item.result_count}**",
                f"- Top score: **{item.top_score:.6f}**" if item.top_score is not None else "- Top score: `n/a`",
                f"- Top case number: `{item.top_case_number or '-'}`",
                f"- Top document type: `{item.top_document_type or '-'}`",
                f"- Top legal area: `{item.top_legal_area or '-'}`",
                f"- Top chunk_id: `{item.top_chunk_id or '-'}`",
                f"- Top URL: `{item.top_url or '-'}`",
                f"- Heuristic rationale: {item.rationale}",
                f"- Top preview: {item.top_text_preview or '-'}",
                "",
                "| rank | score | case_number | document_type | legal_area | ns_section_hint | chunk_id |",
                "| --- | ---: | --- | --- | --- | --- | --- |",
            ]
        )
        for result in item.all_results:
            lines.append(
                f"| {result.rank} | {result.score:.6f} | {result.case_number or '-'} | "
                f"{result.document_type or '-'} | {result.legal_area or '-'} | "
                f"{result.ns_section_hint or '-'} | {result.chunk_id or '-'} |"
            )

    lines.extend(["", "## Errors"])
    if errors:
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Notes",
            "- Labels are heuristic only and are derived from simple token/concept matches in the top result metadata and preview.",
            "- This is a manual retrieval sanity check, not a benchmark with gold labels or graded relevance judgments.",
            "- Query wording, chunk boundaries, and lexical overlap can move a legally relevant result between STRONG and MEDIUM.",
            "- A WEAK label here means the top result looked generic for the configured concepts, not that the whole collection is unusable.",
            "",
        ]
    )
    return "\n".join(lines)


def status_from(*, errors: list[str], weak_count: int) -> str:
    if errors:
        return "FAIL"
    if weak_count > 0:
        return "WARN"
    return "PASS"


def main() -> int:
    args = parse_args()

    if args.limit <= 0:
        print("eval status: FAIL")
        print("error: --limit must be greater than 0.")
        return 1

    evaluations: list[QueryEvaluation] = []
    errors: list[str] = []

    try:
        from qdrant_client import QdrantClient

        resolved_device = resolve_device(args.device)
        warnings.filterwarnings(
            "ignore",
            message=r"Qdrant client version .* is incompatible with server version .*",
        )
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            embedder = build_embedder(args.model_name, batch_size=1, device=resolved_device)

        client = QdrantClient(url=args.qdrant_url, timeout=30)
        collection_vector_size = verify_collection(client, args.collection)

        for query in QUERIES:
            vector = embedder.embed_query(query)
            if len(vector) != collection_vector_size:
                raise RuntimeError(
                    f"Embedding dimension mismatch for query '{query}': "
                    f"{len(vector)} vs collection {collection_vector_size}."
                )
            raw_results = run_search(
                client,
                collection_name=args.collection,
                vector=vector,
                limit=args.limit,
            )
            if not raw_results:
                errors.append(f"Query returned zero results: `{query}`.")
                evaluations.append(
                    QueryEvaluation(
                        query=query,
                        result_count=0,
                        top_score=None,
                        top_case_number="",
                        top_document_type="",
                        top_legal_area="",
                        top_chunk_id="",
                        top_url="",
                        top_text_preview="",
                        label="WEAK",
                        matched_concepts=0,
                        total_concepts=len(QUERY_CONCEPTS[query]),
                        rationale="no search results returned",
                        all_results=[],
                    )
                )
                continue

            results = [map_result(rank, point) for rank, point in enumerate(raw_results, start=1)]
            top_result = results[0]
            label, matched_count, total_count, rationale = evaluate_label(query, top_result)
            evaluations.append(
                QueryEvaluation(
                    query=query,
                    result_count=len(results),
                    top_score=top_result.score,
                    top_case_number=top_result.case_number,
                    top_document_type=top_result.document_type,
                    top_legal_area=top_result.legal_area,
                    top_chunk_id=top_result.chunk_id,
                    top_url=top_result.url,
                    top_text_preview=top_result.text_preview,
                    label=label,
                    matched_concepts=matched_count,
                    total_concepts=total_count,
                    rationale=rationale,
                    all_results=results,
                )
            )
    except Exception as exc:
        errors.append(str(exc))

    strong_count = sum(1 for item in evaluations if item.label == "STRONG")
    medium_count = sum(1 for item in evaluations if item.label == "MEDIUM")
    weak_count = sum(1 for item in evaluations if item.label == "WEAK")
    zero_result_count = sum(1 for item in evaluations if item.result_count == 0)
    queries_passed = sum(1 for item in evaluations if item.result_count > 0)
    status = status_from(errors=errors, weak_count=weak_count)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        build_report(
            status=status,
            collection_name=args.collection,
            limit=args.limit,
            evaluations=evaluations,
            strong_count=strong_count,
            medium_count=medium_count,
            weak_count=weak_count,
            errors=errors,
        ),
        encoding="utf-8",
    )

    print(f"eval status: {status}")
    print(f"query count: {len(QUERIES)}")
    print(f"queries passed: {queries_passed}")
    print(f"queries with zero results: {zero_result_count}")
    print(f"STRONG count: {strong_count}")
    print(f"MEDIUM count: {medium_count}")
    print(f"WEAK count: {weak_count}")
    print(f"report path: {args.out}")
    return 0 if status != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
