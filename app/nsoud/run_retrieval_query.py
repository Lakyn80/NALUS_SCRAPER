"""Run one manual NS retrieval query against the section-aware Qdrant collection.

PowerShell / Docker usage example:
docker compose exec api python app/nsoud/run_retrieval_query.py --query "odpovědnost za vady nemovitosti sleva z kupní ceny" --include-full-decision --max-full-decisions 2
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shlex
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("pandas is required for manual query assembly.") from exc

from app.nsoud.generate_embeddings import DEFAULT_MODEL_NAME, build_embedder, resolve_device
from app.nsoud import retrieval_decision as decision_layer


DEFAULT_CHUNKS_PATH = Path("app/artifacts/nsoud/rag_ready/nsoud_chunks_2025_01_03.parquet")
DEFAULT_OUTPUT_DIR = Path("app/artifacts/nsoud/qdrant/nsoud_chunks_section_aware_test_2025_01_03")
DEFAULT_MARKDOWN_PATH = DEFAULT_OUTPUT_DIR / "manual_query_result.md"
DEFAULT_JSON_PATH = DEFAULT_OUTPUT_DIR / "manual_query_result.json"


@dataclass(frozen=True)
class FullDecisionResult:
    document_id: str
    case_number: str
    document_type: str
    legal_area: str
    chunk_count: int
    full_text: str


@dataclass(frozen=True)
class ManualQueryResult:
    query: str
    decision: decision_layer.DecisionType
    confidence: float
    reason: str
    recommended_user_message: str
    top_result_count: int
    collection_validation: decision_layer.CollectionValidation
    metadata_validation_passed: bool
    exact_dataset_match: bool
    direct_evidence_count: int
    noise_result_count: int
    substantive_reasoning_count: int
    legal_area_distribution: dict[str, int]
    section_type_distribution: dict[str, int]
    matched_core_terms: list[str]
    missing_core_terms: list[str]
    decision_diagnostics: list[str]
    top_results: list[decision_layer.TopResultSummary]
    full_decisions: list[FullDecisionResult]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a practical NS retrieval query using the deterministic decision layer. "
            "Example: docker compose exec api python app/nsoud/run_retrieval_query.py "
            "--query \"odpovědnost za vady nemovitosti sleva z kupní ceny\" --include-full-decision --max-full-decisions 2"
        )
    )
    parser.add_argument("--query", required=True, help="Manual user query to evaluate.")
    parser.add_argument("--top-k", type=int, default=10, help="Top K Qdrant results to retrieve.")
    parser.add_argument(
        "--include-full-decision",
        action="store_true",
        help="Assemble full decision text for the top distinct document_ids.",
    )
    parser.add_argument(
        "--include-noisy-full-decisions",
        action="store_true",
        help="Include full decisions even for noisy top results. By default only core evidence documents are included.",
    )
    parser.add_argument(
        "--max-full-decisions",
        type=int,
        default=3,
        help="Maximum number of distinct full decisions to assemble when --include-full-decision is used.",
    )
    parser.add_argument("--collection", default=decision_layer.TARGET_COLLECTION, help="Qdrant collection name.")
    parser.add_argument("--qdrant-url", default=decision_layer.DEFAULT_QDRANT_URL, help="Qdrant base URL.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output artifact directory.")
    parser.add_argument("--chunks-parquet", type=Path, default=DEFAULT_CHUNKS_PATH, help="NS chunk parquet path.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=decision_layer.DEFAULT_DATASET_PATH,
        help="Categorized relevance dataset JSON for exact query matches.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Sentence-transformers model name.")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="auto",
        help="Embedding device selection.",
    )
    return parser.parse_args()


def load_chunks_dataframe(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def result_preview_row(result: decision_layer.TopResultSummary) -> dict[str, Any]:
    return {
        "rank": result.rank,
        "score": result.score,
        "case_number": result.case_number,
        "document_type": result.document_type,
        "legal_area": result.legal_area,
        "section_type": result.section_type,
        "matched_query_terms": result.matched_query_terms,
        "missing_core_terms": result.missing_core_terms,
        "is_core_evidence": result.is_core_evidence,
        "is_noise": result.is_noise,
        "noise_reason": result.noise_reason,
        "chunk_id": result.chunk_id,
        "document_id": result.document_id,
        "short_preview": result.short_preview,
    }


def build_manual_reason_and_decision(
    *,
    query: str,
    results: list[decision_layer.TopResultSummary],
    analysis: decision_layer.QueryAnalysis,
) -> tuple[decision_layer.DecisionType, float, str]:
    query_stems = decision_layer.significant_stems([query])
    specific_query = len(query_stems) >= 4
    broad_query = len(query_stems) <= 2 or analysis.broad_query
    top_score = analysis.top_score or 0.0
    enough_core_terms = len(analysis.matched_core_terms) >= max(1, min(2, len(decision_layer.query_core_terms(query))))
    expected_area_hits = (
        analysis.legal_area_distribution.get(analysis.expected_legal_area, 0)
        if analysis.expected_legal_area
        else 0
    )
    area_consistent = not analysis.expected_legal_area or expected_area_hits >= max(1, min(3, len(results[:5]) - 1))
    top_noise = sum(1 for result in results[:3] if result.is_noise)
    top_query_match_count = sum(len(result.matched_query_terms) for result in results[:3])

    if not results:
        return (
            "insufficient_support",
            0.92,
            "The query returned no results, so the current NS collection does not provide direct support.",
        )
    if not analysis.metadata_validation_passed:
        return (
            "insufficient_support",
            0.25,
            "Retrieved results are missing required section-aware metadata, so they should not be used for answering.",
        )
    if (
        analysis.top2_core_evidence_count >= 1
        and analysis.direct_evidence_count >= 2
        and analysis.substantive_reasoning_count >= 2
        and analysis.noise_result_count <= max(2, len(results) // 3)
        and area_consistent
        and enough_core_terms
    ):
        return (
            "answerable",
            0.86,
            "High-ranked substantive results contain direct core evidence with acceptable noise levels.",
        )
    if (
        analysis.top5_core_evidence_count >= 2
        and analysis.direct_evidence_count >= 2
        and analysis.substantive_reasoning_count >= 2
        and analysis.top2_source_evidence_count >= 1
        and analysis.noise_result_count <= max(3, len(results) // 2)
        and area_consistent
        and enough_core_terms
    ):
        return (
            "answerable",
            0.77,
            "Multiple substantive results support the query and the highest-ranked evidence is strong enough for answering.",
        )
    if broad_query and analysis.noise_result_count >= 3 and analysis.top2_source_evidence_count == 0:
        return (
            "ask_for_clarification",
            0.88,
            "The query is broad and the top results span multiple documents or legal contexts, so clarification is needed.",
        )
    if analysis.noise_result_count >= 4 and analysis.direct_evidence_count == 0:
        return (
            "insufficient_support",
            0.83,
            "Top results are dominated by noisy or procedural matches and do not provide enough direct support.",
        )
    if specific_query and (analysis.top2_core_evidence_count == 0 or top_noise >= 1 or not enough_core_terms):
        return (
            "insufficient_support",
            0.52,
            "The query is specific, but the highest-ranked results do not contain strong enough direct core evidence.",
        )
    if top_score < 0.45:
        return (
            "insufficient_support",
            0.9,
            "Scores stay weak and the collection does not provide enough direct support for this query.",
        )
    if top_query_match_count == 0:
        return (
            "insufficient_support",
            0.87,
            "Top results do not contain meaningful overlap with the requested issue.",
        )
    return (
        "insufficient_support",
        0.76,
        "The collection returns partial context, but the evidence is still too indirect for a reliable answer.",
    )


def classify_manual_query(
    *,
    query: str,
    dataset_context: dict[str, Any] | None,
    results: list[decision_layer.TopResultSummary],
    analysis: decision_layer.QueryAnalysis,
) -> tuple[decision_layer.DecisionType, float, str, bool]:
    if dataset_context is not None:
        decision, confidence, reason = decision_layer.classify_query(
            query=query,
            item=dataset_context,
            results=results,
            analysis=analysis,
        )
        return decision, confidence, reason, True
    decision, confidence, reason = build_manual_reason_and_decision(
        query=query,
        results=results,
        analysis=analysis,
    )
    return decision, confidence, reason, False


def assemble_full_decisions(
    *,
    chunks_df: pd.DataFrame,
    top_results: list[decision_layer.TopResultSummary],
    max_full_decisions: int,
    include_noisy_full_decisions: bool,
) -> list[FullDecisionResult]:
    distinct_document_ids: list[str] = []
    for result in top_results:
        if not include_noisy_full_decisions and not result.is_core_evidence:
            continue
        if result.document_id and result.document_id not in distinct_document_ids:
            distinct_document_ids.append(result.document_id)
        if len(distinct_document_ids) >= max_full_decisions:
            break

    full_decisions: list[FullDecisionResult] = []
    for document_id in distinct_document_ids:
        doc_rows = chunks_df.loc[chunks_df["document_id"].map(str) == document_id].copy()
        if doc_rows.empty:
            continue
        doc_rows["chunk_index"] = doc_rows["chunk_index"].astype(int)
        doc_rows = doc_rows.sort_values("chunk_index")
        first_row = doc_rows.iloc[0]
        full_text = "\n\n".join(
            str(value).strip()
            for value in doc_rows["chunk_text"].tolist()
            if str(value).strip()
        )
        full_decisions.append(
            FullDecisionResult(
                document_id=document_id,
                case_number=decision_layer.normalize_text(first_row.get("case_number")),
                document_type=decision_layer.normalize_text(first_row.get("document_type")),
                legal_area=decision_layer.normalize_text(first_row.get("legal_area")),
                chunk_count=len(doc_rows),
                full_text=full_text,
            )
        )
    return full_decisions


def format_console_table(results: list[decision_layer.TopResultSummary]) -> str:
    headers = [
        ("rank", 4),
        ("score", 8),
        ("case_number", 24),
        ("document_type", 12),
        ("legal_area", 10),
        ("section_type", 16),
        ("matched_query_terms", 24),
        ("missing_core_terms", 24),
        ("core", 5),
        ("noise", 5),
        ("noise_reason", 26),
        ("chunk_id", 38),
        ("document_id", 32),
        ("short_preview", 72),
    ]
    lines = []
    header_line = " | ".join(label.ljust(width) for label, width in headers)
    separator = "-+-".join("-" * width for _, width in headers)
    lines.append(header_line)
    lines.append(separator)
    for result in results:
        row = [
            str(result.rank).ljust(4),
            f"{result.score:.6f}".ljust(8),
            result.case_number[:24].ljust(24),
            result.document_type[:12].ljust(12),
            result.legal_area[:10].ljust(10),
            result.section_type[:16].ljust(16),
            ", ".join(result.matched_query_terms)[:24].ljust(24),
            ", ".join(result.missing_core_terms)[:24].ljust(24),
            str(result.is_core_evidence).ljust(5),
            str(result.is_noise).ljust(5),
            result.noise_reason[:26].ljust(26),
            result.chunk_id[:38].ljust(38),
            result.document_id[:32].ljust(32),
            result.short_preview[:72].ljust(72),
        ]
        lines.append(" | ".join(row))
    return "\n".join(lines)


def build_json_payload(result: ManualQueryResult) -> dict[str, Any]:
    payload = {
        "query": result.query,
        "decision": result.decision,
        "confidence": result.confidence,
        "reason": result.reason,
        "recommended_user_message": result.recommended_user_message,
        "top_result_count": result.top_result_count,
        "collection_validation": asdict(result.collection_validation),
        "metadata_validation_passed": result.metadata_validation_passed,
        "exact_dataset_match": result.exact_dataset_match,
        "direct_evidence_count": result.direct_evidence_count,
        "noise_result_count": result.noise_result_count,
        "substantive_reasoning_count": result.substantive_reasoning_count,
        "legal_area_distribution": result.legal_area_distribution,
        "section_type_distribution": result.section_type_distribution,
        "matched_core_terms": result.matched_core_terms,
        "missing_core_terms": result.missing_core_terms,
        "decision_diagnostics": result.decision_diagnostics,
        "top_results": [result_preview_row(item) for item in result.top_results],
    }
    if result.full_decisions:
        payload["full_decisions"] = [asdict(item) for item in result.full_decisions]
    return payload


def build_markdown_output(result: ManualQueryResult) -> str:
    lines = [
        "# NSoud Manual Query Result",
        "",
        f"- Query: `{result.query}`",
        f"- Decision: **{result.decision}**",
        f"- Confidence: **{result.confidence:.3f}**",
        f"- Reason: {result.reason}",
        f"- Recommended user message: {result.recommended_user_message}",
        f"- Top result count: **{result.top_result_count}**",
        f"- Exact dataset match: **{result.exact_dataset_match}**",
        f"- Direct evidence count: **{result.direct_evidence_count}**",
        f"- Noise result count: **{result.noise_result_count}**",
        f"- Substantive reasoning count: **{result.substantive_reasoning_count}**",
        f"- Legal area distribution: `{json.dumps(result.legal_area_distribution, ensure_ascii=False)}`",
        f"- Section type distribution: `{json.dumps(result.section_type_distribution, ensure_ascii=False)}`",
        f"- Matched core terms: {', '.join(result.matched_core_terms) if result.matched_core_terms else '-'}",
        f"- Missing core terms: {', '.join(result.missing_core_terms) if result.missing_core_terms else '-'}",
        f"- Decision diagnostics: {' | '.join(result.decision_diagnostics) if result.decision_diagnostics else '-'}",
        "",
        "## Top Results",
        "",
        "| rank | score | case_number | document_type | legal_area | section_type | matched_query_terms | missing_core_terms | is_core_evidence | is_noise | noise_reason | chunk_id | document_id | short preview |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in result.top_results:
        lines.append(
            f"| {item.rank} | {item.score:.6f} | {item.case_number or '-'} | {item.document_type or '-'} | "
            f"{item.legal_area or '-'} | {item.section_type or '-'} | "
            f"{', '.join(item.matched_query_terms) if item.matched_query_terms else '-'} | "
            f"{', '.join(item.missing_core_terms) if item.missing_core_terms else '-'} | "
            f"{item.is_core_evidence} | {item.is_noise} | {item.noise_reason or '-'} | "
            f"{item.chunk_id or '-'} | {item.document_id or '-'} | {item.short_preview or '-'} |"
        )
    lines.append("")
    if result.full_decisions:
        lines.extend(["## Full Decisions", ""])
        for item in result.full_decisions:
            lines.extend(
                [
                    f"### {item.case_number or item.document_id}",
                    "",
                    f"- Document ID: `{item.document_id}`",
                    f"- Document type: `{item.document_type or '-'}`",
                    f"- Legal area: `{item.legal_area or '-'}`",
                    f"- Chunk count: **{item.chunk_count}**",
                    "",
                    "```text",
                    item.full_text,
                    "```",
                    "",
                ]
            )
    return "\n".join(lines)


def run_manual_query(
    *,
    client: Any,
    embedder: Any,
    query: str,
    top_k: int,
    dataset_context: dict[str, Any] | None,
    collection_validation: decision_layer.CollectionValidation,
    include_full_decision: bool,
    include_noisy_full_decisions: bool,
    chunks_df: pd.DataFrame | None,
    max_full_decisions: int,
) -> ManualQueryResult:
    vector = embedder.embed_query(query)
    raw_results = decision_layer.run_search(
        client,
        collection_name=decision_layer.TARGET_COLLECTION,
        vector=vector,
        limit=top_k,
    )
    query_terms = [query]
    source_terms: list[str] = []
    source_case_numbers: set[str] = set()
    source_chunk_ids: set[str] = set()
    expected_section_types: set[str] = set()
    if dataset_context is not None:
        source_terms = [
            decision_layer.normalize_text(value)
            for value in dataset_context.get("source_terms", [])
            if decision_layer.normalize_text(value)
        ]
        query_terms.extend(source_terms)
        source_case_numbers = {
            decision_layer.normalize_text(value)
            for value in dataset_context.get("source_case_numbers", [])
            if decision_layer.normalize_text(value)
        }
        source_chunk_ids = {
            decision_layer.normalize_text(value)
            for value in dataset_context.get("source_chunk_ids", [])
            if decision_layer.normalize_text(value)
        }
        expected_section_types = {
            decision_layer.normalize_text(value)
            for value in dataset_context.get("expected_section_types", [])
            if decision_layer.normalize_text(value)
        }
    core_terms = decision_layer.query_core_terms(query)
    expected_legal_area = decision_layer.infer_expected_legal_area(query, source_case_numbers)
    mapped_results = [
        decision_layer.map_result(
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
    analysis = decision_layer.analyze_results(
        query=query,
        results=mapped_results,
        source_terms=source_terms,
        source_chunk_ids=source_chunk_ids,
        source_case_numbers=source_case_numbers,
        weak_query_info=(dataset_context or {}).get("weak_query_info") if dataset_context else None,
    )
    decision, confidence, reason, exact_dataset_match = classify_manual_query(
        query=query,
        dataset_context=dataset_context,
        results=mapped_results,
        analysis=analysis,
    )
    full_decisions: list[FullDecisionResult] = []
    if include_full_decision:
        if chunks_df is None:
            raise RuntimeError("Full decision assembly requested but chunks parquet was not loaded.")
        full_decisions = assemble_full_decisions(
            chunks_df=chunks_df,
            top_results=mapped_results,
            max_full_decisions=max_full_decisions,
            include_noisy_full_decisions=include_noisy_full_decisions,
        )
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
    return ManualQueryResult(
        query=query,
        decision=decision,
        confidence=confidence,
        reason=reason,
        recommended_user_message=decision_layer.recommended_message_for(decision),
        top_result_count=len(mapped_results),
        collection_validation=collection_validation,
        metadata_validation_passed=analysis.metadata_validation_passed,
        exact_dataset_match=exact_dataset_match,
        direct_evidence_count=analysis.direct_evidence_count,
        noise_result_count=analysis.noise_result_count,
        substantive_reasoning_count=analysis.substantive_reasoning_count,
        legal_area_distribution=analysis.legal_area_distribution,
        section_type_distribution=analysis.section_type_distribution,
        matched_core_terms=analysis.matched_core_terms,
        missing_core_terms=analysis.missing_core_terms,
        decision_diagnostics=decision_diagnostics,
        top_results=mapped_results,
        full_decisions=full_decisions,
    )


def main() -> int:
    args = parse_args()

    if args.top_k <= 0:
        print("decision report status: FAIL")
        print("error: --top-k must be greater than 0.")
        return 1
    if args.max_full_decisions <= 0:
        print("decision report status: FAIL")
        print("error: --max-full-decisions must be greater than 0.")
        return 1
    if args.collection != decision_layer.TARGET_COLLECTION:
        print("decision report status: FAIL")
        print(
            f"error: refusing to operate on collection '{args.collection}'. "
            f"Only '{decision_layer.TARGET_COLLECTION}' is allowed."
        )
        return 1

    try:
        dataset = decision_layer.load_dataset(args.dataset)
        query_context = decision_layer.build_query_context(dataset)
    except Exception as exc:
        print("decision report status: FAIL")
        print(f"error: {exc}")
        return 1

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
        old_collection_before = decision_layer.get_optional_collection_count(client, decision_layer.OLD_COLLECTION)
        exists, point_count, vector_size = decision_layer.verify_collection(client, args.collection)
        if not exists:
            print("decision report status: FAIL")
            print(f"error: target collection '{args.collection}' does not exist.")
            return 1
        old_collection_after = decision_layer.get_optional_collection_count(client, decision_layer.OLD_COLLECTION)
        collection_validation = decision_layer.CollectionValidation(
            exists=True,
            point_count=point_count,
            vector_size=vector_size,
            old_collection_before=old_collection_before,
            old_collection_after=old_collection_after,
            old_collection_unchanged=old_collection_before == old_collection_after,
        )
        if point_count != decision_layer.EXPECTED_POINT_COUNT or vector_size != decision_layer.EXPECTED_VECTOR_SIZE:
            print("decision report status: FAIL")
            print(
                f"error: collection validation failed. expected count/vector "
                f"{decision_layer.EXPECTED_POINT_COUNT}/{decision_layer.EXPECTED_VECTOR_SIZE}, got {point_count}/{vector_size}."
            )
            return 1

        chunks_df = load_chunks_dataframe(args.chunks_parquet) if args.include_full_decision else None
        result = run_manual_query(
            client=client,
            embedder=embedder,
            query=args.query,
            top_k=args.top_k,
            dataset_context=query_context.get(args.query),
            collection_validation=collection_validation,
            include_full_decision=args.include_full_decision,
            include_noisy_full_decisions=args.include_noisy_full_decisions,
            chunks_df=chunks_df,
            max_full_decisions=args.max_full_decisions,
        )
    except Exception as exc:
        print("decision report status: FAIL")
        print(f"error: {exc}")
        return 1

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    out_md = output_dir / DEFAULT_MARKDOWN_PATH.name
    out_json = output_dir / DEFAULT_JSON_PATH.name
    out_json.write_text(json.dumps(build_json_payload(result), ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(build_markdown_output(result), encoding="utf-8")

    command_used = f"{Path(sys.executable).name} " + " ".join(shlex.quote(arg) for arg in sys.argv)
    print(f"command used: {command_used}")
    print(f"query: {result.query}")
    print(f"manual query decision: {result.decision}")
    print(f"confidence: {result.confidence:.3f}")
    print(f"reason: {result.reason}")
    print(f"recommended user message: {result.recommended_user_message}")
    print(f"direct evidence count: {result.direct_evidence_count}")
    print(f"noise result count: {result.noise_result_count}")
    print(f"matched core terms: {', '.join(result.matched_core_terms) if result.matched_core_terms else '-'}")
    print(f"missing core terms: {', '.join(result.missing_core_terms) if result.missing_core_terms else '-'}")
    print(f"top result count: {result.top_result_count}")
    print("top results:")
    print(format_console_table(result.top_results))
    print(f"output markdown path: {out_md.as_posix()}")
    print(f"output json path: {out_json.as_posix()}")
    print("changed files:")
    print("app/nsoud/run_retrieval_query.py")
    print(out_md.as_posix())
    print(out_json.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
