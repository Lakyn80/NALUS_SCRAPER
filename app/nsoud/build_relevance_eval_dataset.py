from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


TARGET_COLLECTION = "nsoud_chunks_section_aware_test_2025_01_03"
DEFAULT_DOCUMENTS_PATH = Path("app/artifacts/nsoud/rag_ready/nsoud_documents_2025_01_03.parquet")
DEFAULT_CHUNKS_PATH = Path("app/artifacts/nsoud/rag_ready/nsoud_chunks_2025_01_03.parquet")
DEFAULT_PAYLOAD_PATH = Path("app/artifacts/nsoud/rag_ready/nsoud_qdrant_payload_preview_2025_01_03.parquet")
DEFAULT_GENERATED_QUERIES_PATH = Path("app/artifacts/nsoud/rag_ready/nsoud_generated_eval_queries_2025_01_03.json")
DEFAULT_OUTPUT_JSON = Path(
    "app/artifacts/nsoud/qdrant/nsoud_chunks_section_aware_test_2025_01_03/relevance_eval_dataset.json"
)
DEFAULT_OUTPUT_MD = Path(
    "app/artifacts/nsoud/qdrant/nsoud_chunks_section_aware_test_2025_01_03/relevance_eval_dataset.md"
)
EXPECTED_DOCUMENTS = 150
EXPECTED_CHUNKS = 1862
EXPECTED_CHUNKING_STRATEGY = "document_section_aware"
WEAK_QUERIES = {
    "náhrada nákladů dovolacího řízení",
    "zjevně neopodstatněné dovolání",
    "odmítnutí dovolání",
    "rodinný dům",
}


@dataclass(frozen=True)
class PositiveSupplement:
    query: str
    regex: str
    source_terms: list[str]
    why_answerable_hint: str


@dataclass(frozen=True)
class NegativeQuerySpec:
    query: str
    required_groups: list[list[str]]
    reason_not_answerable: str
    missing_terms_or_context: list[str]


@dataclass(frozen=True)
class UnderspecifiedQuerySpec:
    query: str
    why_underspecified: str
    suggested_clarifying_questions: list[str]


@dataclass(frozen=True)
class WeakQuerySpec:
    query: str
    required_groups: list[list[str]]
    recommended_dataset_class: str
    why_generic: str


POSITIVE_SUPPLEMENTS = [
    PositiveSupplement(
        query="dovolací důvod podle § 265b odst. 1 písm. h)",
        regex=r"dovolací důvod podle § 265b odst\.?\s*1 písm\.?\s*h\)",
        source_terms=["dovolací důvod podle § 265b odst. 1 písm. h)"],
        why_answerable_hint="The batch contains repeated explicit criminal dovolání reasoning anchored to § 265b odst. 1 písm. h).",
    ),
    PositiveSupplement(
        query="dovolací důvod podle § 265b odst. 1 písm. m)",
        regex=r"dovolací důvod podle § 265b odst\.?\s*1 písm\.?\s*m\)",
        source_terms=["dovolací důvod podle § 265b odst. 1 písm. m)"],
        why_answerable_hint="The batch contains repeated criminal dovolání reasoning explicitly discussing § 265b odst. 1 písm. m).",
    ),
]

NEGATIVE_QUERY_SPECS = [
    NegativeQuerySpec(
        query="mezinárodní ochrana a azyl",
        required_groups=[["mezinarodn", "ochran"], ["azyl"]],
        reason_not_answerable="This NS civil/criminal batch does not contain an asylum / international-protection topic slice.",
        missing_terms_or_context=["azyl", "mezinárodní ochrana", "cizinecké správní řízení"],
    ),
    NegativeQuerySpec(
        query="správní vyhoštění cizince",
        required_groups=[["spravn", "vyhosten"], ["cizin"]],
        reason_not_answerable="The batch is not a focused administrative-removal corpus and lacks that governing context.",
        missing_terms_or_context=["správní vyhoštění", "cizinec", "pobytové správní řízení"],
    ),
    NegativeQuerySpec(
        query="veřejná zakázka a zadávací řízení",
        required_groups=[["verejn", "zakazk"], ["zadavac", "rizen"]],
        reason_not_answerable="Public-procurement disputes are not materially represented in this 150-document NS sample.",
        missing_terms_or_context=["veřejná zakázka", "zadávací řízení", "zadavatel"],
    ),
    NegativeQuerySpec(
        query="ochrana osobních údajů podle GDPR",
        required_groups=[["ochran", "osobn", "udaj"], ["gdpr"]],
        reason_not_answerable="The batch lacks a GDPR / personal-data dispute cluster with enough explicit support.",
        missing_terms_or_context=["GDPR", "osobní údaje", "správce údajů"],
    ),
    NegativeQuerySpec(
        query="odpočet DPH u daně z přidané hodnoty",
        required_groups=[["odpoct", "dph"], ["dan", "pridan", "hodnot"]],
        reason_not_answerable="Tax-law support is absent from this Supreme Court civil/criminal subset.",
        missing_terms_or_context=["DPH", "daň z přidané hodnoty", "daňový odpočet"],
    ),
    NegativeQuerySpec(
        query="stavební povolení a územní rozhodnutí",
        required_groups=[["stavebn", "povolen"], ["uzemn", "rozhodnut"]],
        reason_not_answerable="The batch does not contain a meaningful planning / building-permit dispute track.",
        missing_terms_or_context=["stavební povolení", "územní rozhodnutí", "stavební úřad"],
    ),
]

UNDERSPECIFIED_QUERY_SPECS = [
    UnderspecifiedQuerySpec(
        query="náhrada nákladů dovolacího řízení",
        why_underspecified="The phrase occurs across many civil dovolání outcomes and does not identify the legal issue or desired answer target.",
        suggested_clarifying_questions=[
            "Které rozhodnutí nebo právní problém v dovolacím řízení vás zajímá?",
            "Chcete náklady po odmítnutí dovolání, po zastavení řízení, nebo po meritorním rozhodnutí?",
        ],
    ),
    UnderspecifiedQuerySpec(
        query="zjevně neopodstatněné dovolání",
        why_underspecified="The phrase spans many criminal dovolání decisions and does not specify a statute, issue, or factual context.",
        suggested_clarifying_questions=[
            "Který dovolací důvod nebo trestněprávní problém máte na mysli?",
            "Chcete rozhodnutí podle § 265i odst. 1 písm. e) tr. ř., nebo širší výklad zjevné neopodstatněnosti?",
        ],
    ),
    UnderspecifiedQuerySpec(
        query="odmítnutí dovolání",
        why_underspecified="The query is outcome-only and can refer to many unrelated civil or criminal dovolání decisions.",
        suggested_clarifying_questions=[
            "Jde vám o občanskoprávní nebo trestní dovolání?",
            "Má být dotaz navázán na konkrétní zákonné ustanovení nebo procesní důvod odmítnutí?",
        ],
    ),
    UnderspecifiedQuerySpec(
        query="rodinný dům",
        why_underspecified="The phrase is a broad property object, not a legal issue, and can point to ownership, defects, housing, or damages.",
        suggested_clarifying_questions=[
            "Jde o vlastnictví, vady, bydlení, náhradu škody, nebo jiný spor o rodinný dům?",
            "Má být dotaz zúžen na konkrétní právní otázku nebo skutkový typ?",
        ],
    ),
    UnderspecifiedQuerySpec(
        query="dovolání",
        why_underspecified="The batch contains many dovolání contexts; the bare term is too broad for a reliable answer.",
        suggested_clarifying_questions=[
            "Jaký dovolací důvod nebo právní problém řešíte?",
            "Má jít o přípustnost, odmítnutí, náklady, nebo konkrétní hmotněprávní otázku?",
        ],
    ),
    UnderspecifiedQuerySpec(
        query="místní příslušnost",
        why_underspecified="The phrase spans multiple exekuční and other process situations without specifying the procedural context.",
        suggested_clarifying_questions=[
            "Jde o exekuční věc, civilní spor, nebo jiný typ řízení?",
            "Má být dotaz zúžen na § 11 odst. 3 o. s. ř. nebo na konkrétní skutkovou situaci?",
        ],
    ),
]

WEAK_QUERY_SPECS = [
    WeakQuerySpec(
        query="náhrada nákladů dovolacího řízení",
        required_groups=[["nahrad", "naklad"], ["dovolac", "rizen"]],
        recommended_dataset_class="underspecified",
        why_generic="The query targets a repeated procedural outcome phrase rather than a concrete legal issue.",
    ),
    WeakQuerySpec(
        query="zjevně neopodstatněné dovolání",
        required_groups=[["zjevn", "neopodstat"], ["dovol"]],
        recommended_dataset_class="underspecified",
        why_generic="The query describes a broad criminal dovolání outcome without isolating the underlying issue.",
    ),
    WeakQuerySpec(
        query="odmítnutí dovolání",
        required_groups=[["odmit"], ["dovol"]],
        recommended_dataset_class="underspecified",
        why_generic="The query is an outcome label shared by many unrelated matters.",
    ),
    WeakQuerySpec(
        query="rodinný dům",
        required_groups=[["rodinn", "dum"]],
        recommended_dataset_class="underspecified",
        why_generic="The query names an object of dispute, not the legal question to answer.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic NSoud relevance evaluation dataset.")
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS_PATH, help="Input documents parquet path.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH, help="Input chunks parquet path.")
    parser.add_argument("--payload-preview", type=Path, default=DEFAULT_PAYLOAD_PATH, help="Input payload preview parquet path.")
    parser.add_argument(
        "--generated-queries",
        type=Path,
        default=DEFAULT_GENERATED_QUERIES_PATH,
        help="Optional existing generated eval queries JSON path.",
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUTPUT_JSON, help="Output JSON path.")
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUTPUT_MD, help="Output Markdown report path.")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def simplify_text(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", normalize_text(text)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", ascii_text.lower()).strip()


def stem_token(token: str) -> str:
    normalized = simplify_text(token).replace(" ", "")
    if len(normalized) <= 4:
        return normalized
    for suffix in ("ami", "emi", "ove", "ovi", "eho", "ich", "imi", "ymi", "eni", "ani", "osti", "ove"):
        if normalized.endswith(suffix) and len(normalized) - len(suffix) >= 4:
            return normalized[: -len(suffix)]
    while len(normalized) > 5 and normalized.endswith(("a", "e", "i", "o", "u", "y")):
        normalized = normalized[:-1]
    return normalized


def tokenize_stems(text: str) -> list[str]:
    return [stem_token(token) for token in simplify_text(text).split() if token]


def preview_text(text: str, limit: int = 180) -> str:
    normalized = " ".join(normalize_text(text).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


def ensure_columns(frame: pd.DataFrame, *, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{label} input is missing required columns: {', '.join(missing)}")


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def validate_chunking_strategy(chunks_df: pd.DataFrame) -> None:
    invalid = int((chunks_df["chunking_strategy"].map(normalize_text) != EXPECTED_CHUNKING_STRATEGY).sum())
    if invalid > 0:
        raise RuntimeError(
            f"Chunk parquet contains {invalid} rows without chunking_strategy='{EXPECTED_CHUNKING_STRATEGY}'."
        )


def load_generated_queries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    queries = payload.get("queries")
    if not isinstance(queries, list):
        raise RuntimeError(f"Generated query file `{path.as_posix()}` does not contain a `queries` list.")
    return [item for item in queries if isinstance(item, dict)]


def build_chunk_lookup(chunks_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for _, row in chunks_df.iterrows():
        lookup[normalize_text(row["chunk_id"])] = {column: row[column] for column in chunks_df.columns}
    return lookup


def top_section_types(rows: list[dict[str, Any]]) -> list[str]:
    counts = Counter(normalize_text(row.get("section_type")) for row in rows if normalize_text(row.get("section_type")))
    return [item[0] for item in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]]


def unique_sorted(values: list[str]) -> list[str]:
    return sorted({value for value in values if value})


def build_positive_from_generated(
    generated_queries: list[dict[str, Any]],
    chunk_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    positives: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    for item in generated_queries:
        query = normalize_text(item.get("query"))
        if not query or query in WEAK_QUERIES or query in seen_queries:
            continue
        source_chunk_ids = [
            chunk_id
            for chunk_id in [normalize_text(value) for value in item.get("source_chunk_ids", [])]
            if chunk_id in chunk_lookup
        ]
        if not source_chunk_ids:
            continue
        source_rows = [chunk_lookup[chunk_id] for chunk_id in source_chunk_ids]
        case_numbers = unique_sorted([normalize_text(row.get("case_number")) for row in source_rows])
        section_types = top_section_types(source_rows)
        document_ids = unique_sorted([normalize_text(row.get("document_id")) for row in source_rows])
        positives.append(
            {
                "query": query,
                "source_terms": unique_sorted([normalize_text(value) for value in item.get("source_terms", [])]),
                "source_case_numbers": case_numbers,
                "source_chunk_ids": source_chunk_ids,
                "expected_section_types": section_types,
                "why_answerable": (
                    f"Supported by {len(source_chunk_ids)} source chunks across {len(document_ids)} documents; "
                    f"evidence is concentrated in sections {', '.join(section_types) if section_types else 'unknown'}."
                ),
            }
        )
        seen_queries.add(query)
    return positives


def build_positive_from_supplements(
    chunks_df: pd.DataFrame,
    existing_queries: set[str],
) -> list[dict[str, Any]]:
    positives: list[dict[str, Any]] = []
    for spec in POSITIVE_SUPPLEMENTS:
        if spec.query in existing_queries:
            continue
        compiled = re.compile(spec.regex, flags=re.IGNORECASE)
        matched_rows: list[dict[str, Any]] = []
        for _, row in chunks_df.iterrows():
            chunk_text = normalize_text(row["chunk_text"])
            if compiled.search(chunk_text):
                matched_rows.append({column: row[column] for column in chunks_df.columns})
        if not matched_rows:
            continue
        source_rows = matched_rows[:5]
        source_chunk_ids = unique_sorted([normalize_text(row.get("chunk_id")) for row in source_rows])
        source_case_numbers = unique_sorted([normalize_text(row.get("case_number")) for row in source_rows])
        section_types = top_section_types(matched_rows)
        positives.append(
            {
                "query": spec.query,
                "source_terms": spec.source_terms,
                "source_case_numbers": source_case_numbers,
                "source_chunk_ids": source_chunk_ids,
                "expected_section_types": section_types,
                "why_answerable": (
                    f"{spec.why_answerable_hint} Matched {len(matched_rows)} chunks across "
                    f"{len({normalize_text(row.get('document_id')) for row in matched_rows})} documents."
                ),
            }
        )
    return positives


def chunk_matches_groups(chunk_text: str, required_groups: list[list[str]]) -> bool:
    stems = tokenize_stems(chunk_text)
    if not stems:
        return False
    for group in required_groups:
        if not all(any(token in stem for stem in stems) for token in group):
            return False
    return True


def collect_matching_rows(chunks_df: pd.DataFrame, required_groups: list[list[str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in chunks_df.iterrows():
        if chunk_matches_groups(normalize_text(row["chunk_text"]), required_groups):
            rows.append({column: row[column] for column in chunks_df.columns})
    return rows


def build_negative_queries(chunks_df: pd.DataFrame) -> list[dict[str, Any]]:
    negatives: list[dict[str, Any]] = []
    for spec in NEGATIVE_QUERY_SPECS:
        matched_rows = collect_matching_rows(chunks_df, spec.required_groups)
        matching_docs = len({normalize_text(row.get("document_id")) for row in matched_rows})
        if matching_docs > 1:
            continue
        negatives.append(
            {
                "query": spec.query,
                "expected_behavior": "insufficient_support",
                "reason_not_answerable": spec.reason_not_answerable,
                "missing_terms_or_context": spec.missing_terms_or_context,
                "matching_chunk_count": len(matched_rows),
                "matching_document_count": matching_docs,
            }
        )
    return negatives


def build_underspecified_queries() -> list[dict[str, Any]]:
    return [
        {
            "query": spec.query,
            "expected_behavior": "ask_for_clarification",
            "why_underspecified": spec.why_underspecified,
            "suggested_clarifying_questions": spec.suggested_clarifying_questions,
        }
        for spec in UNDERSPECIFIED_QUERY_SPECS
    ]


def build_weak_query_classification(
    chunks_df: pd.DataFrame,
    generated_queries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    generated_by_query = {
        normalize_text(item.get("query")): item
        for item in generated_queries
        if normalize_text(item.get("query"))
    }
    classifications: list[dict[str, Any]] = []
    for spec in WEAK_QUERY_SPECS:
        matched_rows = collect_matching_rows(chunks_df, spec.required_groups)
        matching_document_ids = unique_sorted([normalize_text(row.get("document_id")) for row in matched_rows])
        matching_case_numbers = unique_sorted([normalize_text(row.get("case_number")) for row in matched_rows])
        section_counts = Counter(normalize_text(row.get("section_type")) for row in matched_rows if normalize_text(row.get("section_type")))
        context_rich_count = sum(
            1
            for row in matched_rows
            if normalize_text(row.get("section_type")) in {"reasoning", "operative_part"}
        )
        source_chunk_ids = [
            normalize_text(value)
            for value in generated_by_query.get(spec.query, {}).get("source_chunk_ids", [])
            if normalize_text(value)
        ]
        classifications.append(
            {
                "query": spec.query,
                "primary_classification": "too_generic",
                "answerable_from_batch": len(matching_document_ids) > 0,
                "sparse_or_insufficiently_supported": False,
                "better_suited_for_hybrid_retrieval": False,
                "recommended_dataset_class": spec.recommended_dataset_class,
                "matching_chunk_count": len(matched_rows),
                "matching_document_count": len(matching_document_ids),
                "source_chunk_ids_available": source_chunk_ids,
                "source_case_numbers": matching_case_numbers[:5],
                "dominant_section_types": [item[0] for item in sorted(section_counts.items(), key=lambda item: (-item[1], item[0]))[:3]],
                "contains_enough_legal_context": context_rich_count > 0,
                "why_classified_this_way": spec.why_generic,
            }
        )
    return classifications


def determine_status(
    positive_answerable: list[dict[str, Any]],
    weak_query_classification: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    notes: list[str] = []
    if not positive_answerable:
        return "FAIL", ["No positive_answerable queries were generated."]
    if any(not item["source_chunk_ids"] for item in positive_answerable):
        return "FAIL", ["One or more positive_answerable queries are missing source_chunk_ids."]
    if not weak_query_classification:
        return "FAIL", ["Weak query classification is missing."]
    if len(positive_answerable) < 12:
        notes.append("Fewer than 12 positive_answerable queries were generated.")
        return "WARN", notes
    return "PASS", notes


def build_json_payload(
    *,
    status: str,
    created_at: str,
    args: argparse.Namespace,
    documents_df: pd.DataFrame,
    chunks_df: pd.DataFrame,
    positive_answerable: list[dict[str, Any]],
    negative_not_in_batch: list[dict[str, Any]],
    underspecified: list[dict[str, Any]],
    weak_query_classification: list[dict[str, Any]],
    notes: list[str],
) -> dict[str, Any]:
    return {
        "status": status,
        "created_at": created_at,
        "target_collection": TARGET_COLLECTION,
        "summary": {
            "input_documents": args.documents.as_posix(),
            "input_chunks": args.chunks.as_posix(),
            "input_payload_preview": args.payload_preview.as_posix(),
            "input_generated_queries": args.generated_queries.as_posix(),
            "total_documents": len(documents_df),
            "total_chunks": len(chunks_df),
            "positive_answerable_count": len(positive_answerable),
            "negative_not_in_batch_count": len(negative_not_in_batch),
            "underspecified_count": len(underspecified),
            "weak_query_classification_count": len(weak_query_classification),
            "notes": notes,
        },
        "positive_answerable": positive_answerable,
        "negative_not_in_batch": negative_not_in_batch,
        "underspecified": underspecified,
        "weak_query_classification": weak_query_classification,
    }


def render_positive_table(items: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| query | expected_section_types | source_case_numbers | source_chunk_ids | why_answerable |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in items:
        lines.append(
            f"| {item['query']} | {', '.join(item['expected_section_types']) or '-'} | "
            f"{', '.join(item['source_case_numbers']) or '-'} | {', '.join(item['source_chunk_ids']) or '-'} | "
            f"{item['why_answerable']} |"
        )
    return lines


def render_negative_table(items: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| query | matching_chunk_count | matching_document_count | expected_behavior | reason_not_answerable |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for item in items:
        lines.append(
            f"| {item['query']} | {item['matching_chunk_count']} | {item['matching_document_count']} | "
            f"{item['expected_behavior']} | {item['reason_not_answerable']} |"
        )
    return lines


def render_underspecified_table(items: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| query | expected_behavior | why_underspecified | suggested_clarifying_questions |",
        "| --- | --- | --- | --- |",
    ]
    for item in items:
        lines.append(
            f"| {item['query']} | {item['expected_behavior']} | {item['why_underspecified']} | "
            f"{' / '.join(item['suggested_clarifying_questions'])} |"
        )
    return lines


def render_weak_table(items: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| query | primary_classification | matching_chunk_count | matching_document_count | recommended_dataset_class | why_classified_this_way |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for item in items:
        lines.append(
            f"| {item['query']} | {item['primary_classification']} | {item['matching_chunk_count']} | "
            f"{item['matching_document_count']} | {item['recommended_dataset_class']} | {item['why_classified_this_way']} |"
        )
    return lines


def quoted_query_list(items: list[dict[str, Any]]) -> str:
    if not items:
        return "none"
    return ", ".join(f"`{item['query']}`" for item in items)


def build_markdown_report(
    *,
    status: str,
    args: argparse.Namespace,
    documents_df: pd.DataFrame,
    chunks_df: pd.DataFrame,
    positive_answerable: list[dict[str, Any]],
    negative_not_in_batch: list[dict[str, Any]],
    underspecified: list[dict[str, Any]],
    weak_query_classification: list[dict[str, Any]],
    notes: list[str],
) -> str:
    hybrid_later = any(item["better_suited_for_hybrid_retrieval"] for item in weak_query_classification)
    lines = [
        "# NSoud Relevance Evaluation Dataset",
        "",
        f"- Status: **{status}**",
        f"- Documents input: `{args.documents.as_posix()}`",
        f"- Chunks input: `{args.chunks.as_posix()}`",
        f"- Payload preview input: `{args.payload_preview.as_posix()}`",
        f"- Generated queries input: `{args.generated_queries.as_posix()}`",
        f"- Total documents: **{len(documents_df)}**",
        f"- Total chunks: **{len(chunks_df)}**",
        f"- positive_answerable count: **{len(positive_answerable)}**",
        f"- negative_not_in_batch count: **{len(negative_not_in_batch)}**",
        f"- underspecified count: **{len(underspecified)}**",
        f"- weak query classification count: **{len(weak_query_classification)}**",
        "",
        "## Positive Answerable",
        "",
    ]
    lines.extend(render_positive_table(positive_answerable))
    lines.extend(["", "## Negative Not In Batch", ""])
    lines.extend(render_negative_table(negative_not_in_batch))
    lines.extend(["", "## Underspecified", ""])
    lines.extend(render_underspecified_table(underspecified))
    lines.extend(["", "## Current Weak Query Classification", ""])
    lines.extend(render_weak_table(weak_query_classification))
    lines.extend(
        [
            "",
            "## Final Recommendation",
            "",
            f"- Retrieval quality testing should use: {quoted_query_list(positive_answerable)}",
            f"- Insufficient-support testing should use: {quoted_query_list(negative_not_in_batch)}",
            f"- Clarification behavior testing should use: {quoted_query_list(underspecified)}",
            f"- Hybrid retrieval should be added later: **{'yes' if hybrid_later else 'no'}**",
            "",
            "## Notes",
        ]
    )
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- None.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    try:
        documents_df = load_parquet(args.documents)
        chunks_df = load_parquet(args.chunks)
        payload_df = load_parquet(args.payload_preview)
    except Exception as exc:
        print("dataset build status: FAIL")
        print(f"error: {exc}")
        return 1

    try:
        ensure_columns(
            documents_df,
            required=["case_number", "document_type", "legal_area", "title", "full_text"],
            label="documents",
        )
        ensure_columns(
            chunks_df,
            required=[
                "document_id",
                "chunk_id",
                "case_number",
                "document_type",
                "legal_area",
                "section_type",
                "chunk_text",
                "chunking_strategy",
            ],
            label="chunks",
        )
        ensure_columns(payload_df, required=["point_id", "chunk_id", "text"], label="payload preview")
        validate_chunking_strategy(chunks_df)
    except Exception as exc:
        print("dataset build status: FAIL")
        print(f"error: {exc}")
        return 1

    generated_queries = load_generated_queries(args.generated_queries)
    chunk_lookup = build_chunk_lookup(chunks_df)
    positive_answerable = build_positive_from_generated(generated_queries, chunk_lookup)
    positive_answerable.extend(
        build_positive_from_supplements(chunks_df, {item["query"] for item in positive_answerable})
    )
    positive_answerable = sorted(positive_answerable, key=lambda item: item["query"].lower())

    negative_not_in_batch = build_negative_queries(chunks_df)
    underspecified = build_underspecified_queries()
    weak_query_classification = build_weak_query_classification(chunks_df, generated_queries)

    status, notes = determine_status(
        positive_answerable=positive_answerable,
        weak_query_classification=weak_query_classification,
    )
    if len(documents_df) != EXPECTED_DOCUMENTS:
        notes.append(f"Expected {EXPECTED_DOCUMENTS} documents but found {len(documents_df)}.")
    if len(chunks_df) != EXPECTED_CHUNKS:
        notes.append(f"Expected {EXPECTED_CHUNKS} chunks but found {len(chunks_df)}.")

    created_at = datetime.now(timezone.utc).isoformat()
    json_payload = build_json_payload(
        status=status,
        created_at=created_at,
        args=args,
        documents_df=documents_df,
        chunks_df=chunks_df,
        positive_answerable=positive_answerable,
        negative_not_in_batch=negative_not_in_batch,
        underspecified=underspecified,
        weak_query_classification=weak_query_classification,
        notes=notes,
    )
    markdown = build_markdown_report(
        status=status,
        args=args,
        documents_df=documents_df,
        chunks_df=chunks_df,
        positive_answerable=positive_answerable,
        negative_not_in_batch=negative_not_in_batch,
        underspecified=underspecified,
        weak_query_classification=weak_query_classification,
        notes=notes,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out_md.write_text(markdown, encoding="utf-8")

    print(f"dataset build status: {status}")
    print(f"positive_answerable count: {len(positive_answerable)}")
    print(f"negative_not_in_batch count: {len(negative_not_in_batch)}")
    print(f"underspecified count: {len(underspecified)}")
    print(f"weak queries classified count: {len(weak_query_classification)}")
    print(f"output json path: {args.out_json.as_posix()}")
    print(f"output markdown path: {args.out_md.as_posix()}")
    print("changed files:")
    print("app/nsoud/build_relevance_eval_dataset.py")
    print(args.out_json.as_posix())
    print(args.out_md.as_posix())
    return 0 if status != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
