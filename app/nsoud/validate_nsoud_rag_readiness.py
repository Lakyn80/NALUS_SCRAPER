from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("pandas is required for this script.") from exc

try:
    import pyarrow  # noqa: F401
except ImportError:
    pyarrow = None

from app.nsoud.chunk_documents import (
    compute_document_id,
    detect_ns_structural_boundaries,
    load_documents,
    normalize_text,
    validate_paragraph_preservation,
)


CHANGES_MADE = [
    "Replaced generic marker scanning with isolated NS-specific boundary helpers.",
    "Added deterministic numbered paragraph detection for `1.` to `200.` with context checks.",
    "Added deterministic numbered slash detection for `1/` to `200/` with context checks.",
    "Added deterministic bracketed paragraph detection for `[1]` to `[200]` with context checks.",
    "Added deterministic parenthesized enumeration detection for `1)` to `200)` after list/sentence separators.",
    "Extended roman section detection from `I.` to `X.` with false-positive guards for citation patterns such as `I. ÚS`.",
    "Added NS section label detection for `takto:`, `Odůvodnění:`, `Poučení:`, spaced `P o u č e n í:`, and `V Brně dne`.",
    "Preserved section labels and markers inside the paragraph they introduce.",
    "Kept paragraph-preservation validation strict across all 150 documents.",
]


@dataclass(frozen=True)
class RemainingOverlongClassification:
    chunk_id: str
    case_number: str
    reason: str
    explanation: str
    audit_classification: str
    chunk_text_length: int


@dataclass(frozen=True)
class ReadinessSummary:
    chunking_status: str
    total_documents: int
    total_chunks: int
    overlong_chunk_count: int
    suspicious_possible_missed_boundary_count: int
    paragraph_preservation_passed: int
    paragraph_preservation_failed: int
    empty_chunk_count: int
    duplicate_chunk_id_count: int
    documents_with_zero_chunks: int
    final_readiness_status: str
    rag_ready: bool
    readiness_report_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate final NSoud chunking readiness for embeddings.")
    parser.add_argument("--documents", type=Path, required=True, help="Input NSoud documents Parquet path.")
    parser.add_argument("--chunks", type=Path, required=True, help="Input NSoud chunks Parquet path.")
    parser.add_argument("--audit-csv", type=Path, required=True, help="Input NSoud overlong audit CSV path.")
    parser.add_argument("--out", type=Path, required=True, help="Output Markdown readiness report path.")
    return parser.parse_args()


def load_chunks(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_audit_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def classify_remaining_overlong(
    audit_df: pd.DataFrame,
    chunks_df: pd.DataFrame,
) -> tuple[list[RemainingOverlongClassification], int]:
    classified_rows: list[RemainingOverlongClassification] = []
    unresolved_count = 0
    chunk_map = {
        normalize_text(row["chunk_id"]): normalize_text(row["chunk_text"])
        for _, row in chunks_df.loc[chunks_df["chunk_warning"] == "overlong_ns_paragraph", ["chunk_id", "chunk_text"]].iterrows()
    }

    for _, row in audit_df.iterrows():
        chunk_id = normalize_text(row.get("chunk_id"))
        boundaries = detect_ns_structural_boundaries(chunk_map.get(chunk_id, ""))
        reason = "unsafe_to_split"
        explanation = (
            "Audit heuristics detected inline marker-like text, but the deterministic NS detector found no safe additional "
            "structural boundary beyond the current paragraph start."
        )

        if normalize_text(row.get("audit_classification")) == "real long paragraph":
            reason = "real_long_paragraph"
            explanation = "No internal marker-like pattern was detected by the audit heuristics."
        elif len(boundaries) > 1:
            reason = "unresolved_boundary_issue"
            explanation = "The deterministic NS detector still finds additional internal boundaries inside the overlong chunk."
            unresolved_count += 1

        classified_rows.append(
            RemainingOverlongClassification(
                chunk_id=chunk_id,
                case_number=normalize_text(row.get("case_number")),
                reason=reason,
                explanation=explanation,
                audit_classification=normalize_text(row.get("audit_classification")),
                chunk_text_length=int(row.get("chunk_text_length", 0)),
            )
        )

    return classified_rows, unresolved_count

def compute_documents_with_zero_chunks(documents_df: pd.DataFrame, chunks_df: pd.DataFrame) -> int:
    chunk_counts = {str(key): int(value) for key, value in chunks_df["document_id"].value_counts().to_dict().items()}
    zero_count = 0
    for _, row in documents_df.iterrows():
        record = row.to_dict()
        if normalize_text(record.get("full_text")) and chunk_counts.get(compute_document_id(record), 0) == 0:
            zero_count += 1
    return zero_count


def build_markdown_report(
    *,
    documents_path: Path,
    chunks_path: Path,
    audit_csv_path: Path,
    summary: ReadinessSummary,
    classified_rows: list[RemainingOverlongClassification],
    unresolved_count: int,
) -> str:
    lines = [
        "# NSoud Chunking Readiness",
        "",
        f"- Documents input: `{documents_path}`",
        f"- Chunks input: `{chunks_path}`",
        f"- Audit input: `{audit_csv_path}`",
        f"- Final status: **{summary.final_readiness_status}**",
        f"- RAG_READY: **{'true' if summary.rag_ready else 'false'}**",
        f"- Final total documents: **{summary.total_documents}**",
        f"- Final total chunks: **{summary.total_chunks}**",
        f"- Final overlong chunk count: **{summary.overlong_chunk_count}**",
        f"- Final suspicious possible missed boundary count: **{summary.suspicious_possible_missed_boundary_count}**",
        f"- Paragraph preservation: **{summary.paragraph_preservation_passed} passed / {summary.paragraph_preservation_failed} failed**",
        f"- Empty chunk count: **{summary.empty_chunk_count}**",
        f"- Duplicate chunk_id count: **{summary.duplicate_chunk_id_count}**",
        f"- Documents with zero chunks: **{summary.documents_with_zero_chunks}**",
        f"- Unresolved boundary issue count: **{unresolved_count}**",
        "",
        "## Exact Changes Made To NS Boundary Detection",
    ]
    lines.extend(f"- {item}" for item in CHANGES_MADE)
    lines.extend(
        [
            "",
            "## Remaining Overlong Chunks",
            "",
            "| Chunk ID | Case Number | Reason | Audit Classification | Length | Explanation |",
            "| --- | --- | --- | --- | ---: | --- |",
        ]
    )
    if not classified_rows:
        lines.append("| - | - | - | - | 0 | none |")
    else:
        for row in classified_rows:
            explanation = row.explanation.replace("|", "\\|")
            lines.append(
                f"| {row.chunk_id} | {row.case_number or '-'} | {row.reason} | {row.audit_classification or '-'} | "
                f"{row.chunk_text_length} | {explanation} |"
            )
    lines.extend(
        [
            "",
            "## Embeddings Readiness",
            f"- Ready for embeddings: {'true' if summary.rag_ready else 'false'}",
            f"- Blocking issue present: {'true' if unresolved_count > 0 else 'false'}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    if pyarrow is None:
        print("chunking status: FAIL")
        print("error: pyarrow is required for Parquet input.")
        print("install command: pip install pyarrow")
        return 1

    try:
        documents_df = load_documents(args.documents)
        chunks_df = load_chunks(args.chunks)
        audit_df = load_audit_csv(args.audit_csv)
    except Exception as exc:
        print("chunking status: FAIL")
        print(f"error: {exc}")
        return 1

    paragraph_failures, paragraph_passed, paragraph_failed = validate_paragraph_preservation(documents_df, chunks_df)
    empty_chunk_count = int(chunks_df["chunk_text"].map(lambda value: normalize_text(value) == "").sum()) if not chunks_df.empty else 0
    duplicate_chunk_id_count = int(chunks_df["chunk_id"].duplicated(keep=False).sum()) if not chunks_df.empty else 0
    documents_with_zero_chunks = compute_documents_with_zero_chunks(documents_df, chunks_df)
    classified_rows, unresolved_count = classify_remaining_overlong(audit_df, chunks_df)

    if paragraph_failures or paragraph_failed > 0 or empty_chunk_count > 0 or duplicate_chunk_id_count > 0 or documents_with_zero_chunks > 0:
        final_status = "FAIL"
    elif unresolved_count > 0:
        final_status = "FAIL"
    elif len(classified_rows) > 0:
        final_status = "WARN"
    else:
        final_status = "PASS"

    rag_ready = final_status in {"PASS", "WARN"} and unresolved_count == 0
    summary = ReadinessSummary(
        chunking_status="PASS" if final_status != "FAIL" or not paragraph_failures else "FAIL",
        total_documents=len(documents_df),
        total_chunks=len(chunks_df),
        overlong_chunk_count=len(classified_rows),
        suspicious_possible_missed_boundary_count=int(audit_df["suspicious_possible_missed_boundary"].sum()) if not audit_df.empty else 0,
        paragraph_preservation_passed=paragraph_passed,
        paragraph_preservation_failed=paragraph_failed,
        empty_chunk_count=empty_chunk_count,
        duplicate_chunk_id_count=duplicate_chunk_id_count,
        documents_with_zero_chunks=documents_with_zero_chunks,
        final_readiness_status=final_status,
        rag_ready=rag_ready,
        readiness_report_path=args.out,
    )

    report = build_markdown_report(
        documents_path=args.documents,
        chunks_path=args.chunks,
        audit_csv_path=args.audit_csv,
        summary=summary,
        classified_rows=classified_rows,
        unresolved_count=unresolved_count,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")

    print(f"chunking status: {summary.chunking_status}")
    print(f"total documents: {summary.total_documents}")
    print(f"total chunks: {summary.total_chunks}")
    print(f"overlong NS paragraph chunk count: {summary.overlong_chunk_count}")
    print(f"suspicious possible missed boundary count: {summary.suspicious_possible_missed_boundary_count}")
    print(f"paragraph preservation passed/failed: {summary.paragraph_preservation_passed}/{summary.paragraph_preservation_failed}")
    print(f"final readiness status: {summary.final_readiness_status}")
    print(f"RAG_READY: {'true' if summary.rag_ready else 'false'}")
    print(f"readiness report path: {summary.readiness_report_path}")
    return 1 if summary.final_readiness_status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
