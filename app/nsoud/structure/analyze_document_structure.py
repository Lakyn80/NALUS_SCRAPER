from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("pandas is required for this script.") from exc

try:
    import pyarrow  # noqa: F401
except ImportError:
    pyarrow = None

from app.nsoud.structure.confidence import calculate_structure_confidence
from app.nsoud.structure.patterns import MARKER_SPECS, STRUCTURE_PATTERN_LABELS
from app.nsoud.structure.section_detector import detect_ns_document_structure


REQUIRED_COLUMNS = {
    "case_number",
    "ecli",
    "decision_date",
    "publication_date",
    "document_type",
    "legal_area",
    "full_text",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze deterministic NSoud document structure from local Parquet.")
    parser.add_argument("--input", type=Path, required=True, help="Input NSoud documents Parquet path.")
    parser.add_argument("--out-json", type=Path, required=True, help="Output JSON path.")
    parser.add_argument("--out-md", type=Path, required=True, help="Output Markdown path.")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def escape_markdown_cell(value: Any) -> str:
    return normalize_text(value).replace("|", "\\|")


def build_document_id(record: dict[str, Any], row_index: int) -> str:
    ecli = normalize_text(record.get("ecli"))
    if ecli:
        return ecli
    content_hash = normalize_text(record.get("content_hash"))
    if content_hash:
        return content_hash
    case_number = normalize_text(record.get("case_number")) or f"row-{row_index:04d}"
    safe_case = re.sub(r"[^A-Za-z0-9]+", "_", case_number).strip("_")
    return safe_case or f"row-{row_index:04d}"


def full_text_stats(texts: list[str]) -> dict[str, float]:
    lengths = [len(text) for text in texts]
    return {
        "min": min(lengths) if lengths else 0,
        "max": max(lengths) if lengths else 0,
        "avg": round(mean(lengths), 2) if lengths else 0.0,
    }


def combination_key(document: dict[str, Any]) -> str:
    labels: list[str] = []
    marker_flags = document["marker_flags"]
    for flag_name, short_label in STRUCTURE_PATTERN_LABELS:
        if marker_flags.get(flag_name):
            labels.append(short_label)
    return " + ".join(labels) if labels else "NO_MARKERS"


def marker_coverage_rows(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in MARKER_SPECS:
        count = sum(1 for document in documents if document["detected_markers"][label]["present"])
        rows.append(
            {
                "marker": label,
                "count": count,
                "pct": round((count / len(documents)) * 100, 2) if documents else 0.0,
            }
        )
    rows.append(
        {
            "marker": "roman_sections_any",
            "count": sum(1 for document in documents if document["detected_roman_section_count"] > 0),
            "pct": round(
                (
                    sum(1 for document in documents if document["detected_roman_section_count"] > 0)
                    / len(documents)
                )
                * 100,
                2,
            )
            if documents
            else 0.0,
        }
    )
    rows.append(
        {
            "marker": "numbered_paragraphs_any",
            "count": sum(1 for document in documents if document["detected_numbered_paragraph_count"] > 0),
            "pct": round(
                (
                    sum(1 for document in documents if document["detected_numbered_paragraph_count"] > 0)
                    / len(documents)
                )
                * 100,
                2,
            )
            if documents
            else 0.0,
        }
    )
    return rows


def chunking_recommendations(summary: dict[str, Any]) -> list[str]:
    coverage = summary["marker_coverage"]
    coverage_map = {row["marker"]: row["pct"] for row in coverage}
    recommendations: list[str] = []
    if coverage_map.get("takto:", 0.0) >= 80.0 and coverage_map.get("Odůvodnění:", 0.0) + coverage_map.get("O d ů v o d n ě n í:", 0.0) >= 80.0:
        recommendations.append("Prefer a first-pass split around `takto:` and `Odůvodnění` markers before any token-length chunking.")
    if coverage_map.get("numbered_paragraphs_any", 0.0) >= 80.0:
        recommendations.append("Preserve numbered legal paragraphs as atomic boundaries whenever possible.")
    if coverage_map.get("roman_sections_any", 0.0) >= 60.0:
        recommendations.append("Preserve Roman numeral verdict sections (`I.` to `XX.`) inside the `takto` block.")
    if coverage_map.get("Poučení:", 0.0) + coverage_map.get("P o u č e n í:", 0.0) >= 80.0:
        recommendations.append("Treat `Poučení` as a late-document boundary and avoid merging it into substantive reasoning chunks.")
    if coverage_map.get("V Brně dne", 0.0) == 100.0:
        recommendations.append("Use `V Brně dne` as a deterministic closing/signature boundary for trimming footer-only tails.")
    if coverage_map.get("O d ů v o d n ě n í:", 0.0) > 0.0 or coverage_map.get("P o u č e n í:", 0.0) > 0.0:
        recommendations.append("Support both regular and spaced marker spellings (`Odůvodnění` / `O d ů v o d n ě n í`, `Poučení` / `P o u č e n í`).")
    if not recommendations:
        recommendations.append("The batch is heterogeneous enough that chunking should rely on combined marker and paragraph heuristics.")
    return recommendations


def build_markdown_report(
    *,
    status: str,
    input_path: Path,
    summary: dict[str, Any],
) -> str:
    marker_rows = summary["marker_coverage"]
    weak_documents = summary["weak_examples"]
    section_order_examples = summary["section_order_examples"]
    recommendations = summary["chunking_recommendations"]
    metadata = summary["metadata_distribution"]
    confidence = summary["structure_confidence_summary"]

    lines = [
        "# NSoud Document Structure Analysis",
        "",
        f"- Status: **{status}**",
        f"- Input path: `{input_path}`",
        f"- Total documents: **{summary['total_documents']}**",
        "",
        "## Metadata Distribution",
        "",
        f"- Document type distribution: {json.dumps(metadata['document_type_distribution'], ensure_ascii=False)}",
        f"- Legal area distribution: {json.dumps(metadata['legal_area_distribution'], ensure_ascii=False)}",
        f"- Missing ecli count: **{metadata['missing_ecli_count']}**",
        f"- Missing decision_date count: **{metadata['missing_decision_date_count']}**",
        f"- Missing publication_date count: **{metadata['missing_publication_date_count']}**",
        f"- Missing legal_area count: **{metadata['missing_legal_area_count']}**",
        f"- Full text length min/max/avg: **{metadata['full_text_length']['min']} / {metadata['full_text_length']['max']} / {metadata['full_text_length']['avg']}**",
        "",
        "## Marker Coverage",
        "",
        "| marker | count | pct |",
        "| --- | ---: | ---: |",
    ]
    for row in marker_rows:
        lines.append(f"| {row['marker']} | {row['count']} | {row['pct']:.2f} |")

    lines.extend(
        [
            "",
            "## Structure Confidence Summary",
            "",
            f"- Strong structure count: **{confidence['strong_count']}**",
            f"- Medium structure count: **{confidence['medium_count']}**",
            f"- Weak structure count: **{confidence['weak_count']}**",
            f"- Needs review count: **{confidence['needs_review_count']}**",
            f"- Average structure confidence: **{confidence['avg_confidence']}**",
            "",
            "## Needs Review",
            "",
            "| document_id | case_number | document_type | legal_area | confidence | status | section_order |",
            "| --- | --- | --- | --- | ---: | --- | --- |",
        ]
    )
    if weak_documents:
        for document in weak_documents:
            lines.append(
                f"| {escape_markdown_cell(document['document_id'])} | {escape_markdown_cell(document['case_number'])} | "
                f"{escape_markdown_cell(document['document_type'])} | {escape_markdown_cell(document['legal_area'])} | "
                f"{document['structure_confidence']:.2f} | {document['structure_status']} | "
                f"{escape_markdown_cell(' > '.join(document['detected_section_order']['observed_sections']))} |"
            )
    else:
        lines.append("| - | - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Examples Of Detected Section Order",
            "",
            "| order | count | pct |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in section_order_examples:
        lines.append(f"| {escape_markdown_cell(row['order'])} | {row['count']} | {row['pct']:.2f} |")

    lines.extend(
        [
            "",
            "## Most Common Marker Combinations",
            "",
            "| combination | count | pct |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in summary["top_marker_combinations"]:
        lines.append(f"| {escape_markdown_cell(row['combination'])} | {row['count']} | {row['pct']:.2f} |")

    lines.extend(["", "## Recommendations For NS Chunking Rules", ""])
    for recommendation in recommendations:
        lines.append(f"- {recommendation}")

    return "\n".join(lines)


def analyze_documents(documents_df: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = documents_df.to_dict(orient="records")
    documents: list[dict[str, Any]] = []
    marker_pattern_counter: Counter[str] = Counter()
    section_order_counter: Counter[str] = Counter()

    for row_index, record in enumerate(records):
        structure = detect_ns_document_structure(
            full_text=normalize_text(record.get("full_text")),
            metadata={
                "document_type": normalize_text(record.get("document_type")),
                "legal_area": normalize_text(record.get("legal_area")),
                "case_number": normalize_text(record.get("case_number")),
                "ecli": normalize_text(record.get("ecli")),
            },
        )
        confidence = calculate_structure_confidence(structure)
        document = {
            "document_id": build_document_id(record, row_index),
            "case_number": normalize_text(record.get("case_number")),
            "ecli": normalize_text(record.get("ecli")),
            "document_type": normalize_text(record.get("document_type")),
            "legal_area": normalize_text(record.get("legal_area")),
            **{key: value for key, value in structure.items() if key != "metadata"},
            **confidence,
        }
        documents.append(document)
        marker_pattern_counter[combination_key(document)] += 1
        section_order_counter[" > ".join(document["detected_section_order"]["observed_sections"])] += 1

    full_texts = [normalize_text(record.get("full_text")) for record in records]
    summary = {
        "total_documents": len(records),
        "metadata_distribution": {
            "document_type_distribution": dict(
                sorted(
                    Counter(normalize_text(record.get("document_type")) for record in records).items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "legal_area_distribution": dict(
                sorted(
                    Counter(normalize_text(record.get("legal_area")) for record in records).items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "missing_ecli_count": sum(1 for record in records if not normalize_text(record.get("ecli"))),
            "missing_decision_date_count": sum(1 for record in records if not normalize_text(record.get("decision_date"))),
            "missing_publication_date_count": sum(1 for record in records if not normalize_text(record.get("publication_date"))),
            "missing_legal_area_count": sum(1 for record in records if not normalize_text(record.get("legal_area"))),
            "full_text_length": full_text_stats(full_texts),
        },
        "marker_coverage": marker_coverage_rows(documents),
        "structure_confidence_summary": {
            "strong_count": sum(1 for document in documents if document["structure_status"] == "strong"),
            "medium_count": sum(1 for document in documents if document["structure_status"] == "medium"),
            "weak_count": sum(1 for document in documents if document["structure_status"] == "weak"),
            "needs_review_count": sum(1 for document in documents if document["needs_review"]),
            "avg_confidence": round(mean(document["structure_confidence"] for document in documents), 3) if documents else 0.0,
        },
        "most_common_marker_combination": marker_pattern_counter.most_common(1)[0][0] if marker_pattern_counter else "",
        "top_marker_combinations": [
            {
                "combination": pattern,
                "count": count,
                "pct": round((count / len(documents)) * 100, 2) if documents else 0.0,
            }
            for pattern, count in marker_pattern_counter.most_common(20)
        ],
        "top_structural_marker_patterns": [
            {
                "pattern": pattern,
                "count": count,
                "pct": round((count / len(documents)) * 100, 2) if documents else 0.0,
            }
            for pattern, count in marker_pattern_counter.most_common(20)
        ],
        "weak_examples": [
            {
                "document_id": document["document_id"],
                "case_number": document["case_number"],
                "document_type": document["document_type"],
                "legal_area": document["legal_area"],
                "structure_confidence": document["structure_confidence"],
                "structure_status": document["structure_status"],
                "detected_section_order": document["detected_section_order"],
            }
            for document in documents
            if document["needs_review"]
        ][:10],
        "section_order_examples": [
            {
                "order": order,
                "count": count,
                "pct": round((count / len(documents)) * 100, 2) if documents else 0.0,
            }
            for order, count in section_order_counter.most_common(10)
        ],
    }
    summary["chunking_recommendations"] = chunking_recommendations(summary)
    return documents, summary


def validate_input(documents_df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing_columns = sorted(column for column in REQUIRED_COLUMNS if column not in documents_df.columns)
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        return errors
    empty_full_text_count = int(documents_df["full_text"].map(lambda value: normalize_text(value).strip() == "").sum())
    if empty_full_text_count > 0:
        errors.append(f"Found {empty_full_text_count} documents with missing or empty full_text.")
    return errors


def determine_status(errors: list[str], needs_review_count: int) -> str:
    if errors:
        return "FAIL"
    if needs_review_count > 0:
        return "WARN"
    return "PASS"


def main() -> int:
    args = parse_args()

    if pyarrow is None:
        print("status: FAIL")
        print("error: pyarrow is required for Parquet input.")
        return 1

    try:
        documents_df = pd.read_parquet(args.input)
    except Exception as exc:
        print("status: FAIL")
        print(f"error: {exc}")
        return 1

    errors = validate_input(documents_df)
    documents: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "total_documents": int(len(documents_df)),
        "structure_confidence_summary": {
            "strong_count": 0,
            "medium_count": 0,
            "weak_count": 0,
            "needs_review_count": 0,
            "avg_confidence": 0.0,
        },
    }

    if not errors:
        documents, summary = analyze_documents(documents_df)

    status = determine_status(errors, summary["structure_confidence_summary"]["needs_review_count"])
    payload = {
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(args.input),
        "total_documents": int(len(documents_df)),
        "summary": summary | {"errors": errors},
        "documents": documents,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out_md.write_text(
        build_markdown_report(status=status, input_path=args.input, summary=payload["summary"])
        if not errors
        else "\n".join(
            [
                "# NSoud Document Structure Analysis",
                "",
                f"- Status: **{status}**",
                f"- Input path: `{args.input}`",
                f"- Total documents: **{len(documents_df)}**",
                "",
                "## Errors",
                *(f"- {error}" for error in errors),
            ]
        ),
        encoding="utf-8",
    )

    confidence = summary["structure_confidence_summary"]
    print(f"status: {status}")
    print(f"total documents: {len(documents_df)}")
    print(f"strong structure count: {confidence['strong_count']}")
    print(f"medium structure count: {confidence['medium_count']}")
    print(f"weak structure count: {confidence['weak_count']}")
    print(f"needs_review count: {confidence['needs_review_count']}")
    print(f"output json path: {args.out_json}")
    print(f"output markdown path: {args.out_md}")
    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
