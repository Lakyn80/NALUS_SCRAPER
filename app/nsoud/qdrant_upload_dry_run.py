from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
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


REQUIRED_PAYLOAD_FIELDS = [
    "provider",
    "source",
    "court",
    "authority_level",
    "case_number",
    "document_id",
    "chunk_id",
    "chunk_index",
    "url",
]
OPTIONAL_PAYLOAD_FIELDS = [
    "ecli",
    "decision_date",
    "publication_date",
    "document_type",
    "legal_area",
    "title",
    "source_attribution",
    "content_hash",
    "chunk_text_length",
    "paragraph_count",
    "chunk_warning",
    "ns_section_hint",
]
REQUIRED_BASE_FIELDS = ["point_id", "text", "embedding", "embedding_dim", *REQUIRED_PAYLOAD_FIELDS]


@dataclass(frozen=True)
class DryRunSummary:
    dry_run_status: str
    qdrant_upload_ready: bool
    collection_name: str
    total_points: int
    vector_size: int
    duplicate_point_id_count: int
    missing_embedding_count: int
    inconsistent_embedding_dim_count: int
    empty_text_count: int
    report_path: Path
    plan_path: Path
    validation_status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate NSoud embeddings for Qdrant upload readiness.")
    parser.add_argument("--input", type=Path, required=True, help="Input NSoud embeddings parquet path.")
    parser.add_argument("--collection", required=True, help="Target Qdrant collection name.")
    parser.add_argument("--out-report", type=Path, required=True, help="Output Markdown report path.")
    parser.add_argument("--out-plan", type=Path, required=True, help="Output JSON upload plan path.")
    parser.add_argument(
        "--check-qdrant",
        action="store_true",
        help="Optionally check Qdrant connectivity without uploading anything.",
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


def count_missing_values(series: pd.Series, *, treat_empty_string_as_missing: bool) -> int:
    if treat_empty_string_as_missing:
        return int(series.map(lambda value: normalize_text(value).strip() == "").sum())
    return int(series.isna().sum())


def load_embeddings(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def distribution_counts(df: pd.DataFrame, column_name: str) -> dict[str, int]:
    if df.empty:
        return {}
    series = df[column_name].fillna("").map(lambda value: str(value).strip() or "<missing>")
    counts = series.value_counts(dropna=False).to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def render_distribution_table(title: str, counts: dict[str, int]) -> list[str]:
    lines = [f"## {title}", "", "| Value | Count |", "| --- | ---: |"]
    if not counts:
        lines.append("| - | 0 |")
    else:
        for value, count in counts.items():
            lines.append(f"| {value} | {count} |")
    lines.append("")
    return lines


def is_missing_embedding(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip() == ""
    if not hasattr(value, "__len__"):
        return True
    return len(value) == 0


def detect_vector_size(df: pd.DataFrame) -> tuple[int, int]:
    if df.empty:
        return 0, 0
    dims = df["embedding_dim"].dropna().tolist()
    unique_dims = sorted({int(value) for value in dims})
    if not unique_dims:
        return 0, 0
    if len(unique_dims) == 1:
        return unique_dims[0], 0
    inconsistent_count = int(df["embedding_dim"].map(lambda value: int(value) != unique_dims[0]).sum())
    return unique_dims[0], inconsistent_count


def recommended_batch_size(total_points: int, vector_size: int) -> int:
    if total_points <= 1000:
        return 64
    if vector_size >= 1024:
        return 64
    if total_points <= 5000:
        return 128
    return 256


def maybe_check_qdrant(enabled: bool) -> tuple[str | None, str | None]:
    if not enabled:
        return None, None
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        return "Qdrant client import failed.", str(exc)

    try:
        client = QdrantClient(url="http://qdrant:6333")
        collections = client.get_collections()
        return "Qdrant connectivity check passed.", f"collections_visible={len(collections.collections)}"
    except Exception as exc:
        return "Qdrant connectivity check failed.", str(exc)


def validate_dataframe(
    df: pd.DataFrame,
) -> tuple[str, bool, dict[str, int], dict[str, int], list[str], list[str], int, int, int, int, int]:
    failures: list[str] = []
    warnings: list[str] = []

    missing_required_counts: dict[str, int] = {}
    missing_optional_counts: dict[str, int] = {}

    for column_name in REQUIRED_BASE_FIELDS:
        if column_name not in df.columns:
            failures.append(f"Missing required column `{column_name}`.")

    if failures:
        return "FAIL", False, missing_required_counts, missing_optional_counts, failures, warnings, 0, 0, 0, 0, 0

    for field_name in ["point_id", "text", *REQUIRED_PAYLOAD_FIELDS]:
        missing_count = count_missing_values(df[field_name], treat_empty_string_as_missing=True)
        missing_required_counts[field_name] = missing_count
        if missing_count > 0:
            failures.append(f"Required field `{field_name}` contains missing values.")

    missing_required_counts["embedding"] = int(df["embedding"].map(is_missing_embedding).sum())
    if missing_required_counts["embedding"] > 0:
        failures.append("One or more rows are missing embeddings.")

    missing_required_counts["embedding_dim"] = count_missing_values(df["embedding_dim"], treat_empty_string_as_missing=False)
    if missing_required_counts["embedding_dim"] > 0:
        failures.append("One or more rows are missing embedding_dim.")

    duplicate_point_id_count = int(df["point_id"].duplicated(keep=False).sum())
    if duplicate_point_id_count > 0:
        failures.append("Duplicate point_id values detected.")

    empty_text_count = int(df["text"].map(lambda value: normalize_text(value).strip() == "").sum())
    if empty_text_count > 0:
        failures.append("One or more rows have empty text.")

    missing_embedding_count = int(df["embedding"].map(is_missing_embedding).sum())
    vector_size, inconsistent_embedding_dim_count = detect_vector_size(df)
    if vector_size == 0:
        failures.append("No valid embedding dimension was detected.")
    if inconsistent_embedding_dim_count > 0:
        failures.append("Embedding dimensions are inconsistent across rows.")

    for field_name in OPTIONAL_PAYLOAD_FIELDS:
        if field_name not in df.columns:
            missing_optional_counts[field_name] = len(df)
            warnings.append(f"Optional field `{field_name}` is not present.")
            continue
        treat_empty = field_name not in {"chunk_warning"}
        missing_count = count_missing_values(df[field_name], treat_empty_string_as_missing=treat_empty)
        missing_optional_counts[field_name] = missing_count

    if any(count > 0 for count in missing_optional_counts.values()):
        warnings.append("Some optional payload metadata fields are missing.")

    validation_status = "FAIL" if failures else "WARN" if warnings else "PASS"
    qdrant_upload_ready = not failures
    return (
        validation_status,
        qdrant_upload_ready,
        missing_required_counts,
        missing_optional_counts,
        failures,
        warnings,
        duplicate_point_id_count,
        missing_embedding_count,
        inconsistent_embedding_dim_count,
        empty_text_count,
        vector_size,
    )


def write_upload_plan(
    out_path: Path,
    *,
    collection_name: str,
    vector_size: int,
    total_points: int,
    input_path: Path,
) -> None:
    payload = {
        "collection_name": collection_name,
        "vector_size": vector_size,
        "distance": "Cosine",
        "total_points": total_points,
        "batch_size_recommendation": recommended_batch_size(total_points, vector_size),
        "input_path": str(input_path),
        "required_payload_fields": REQUIRED_PAYLOAD_FIELDS,
        "optional_payload_fields": OPTIONAL_PAYLOAD_FIELDS,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": True,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_report(
    *,
    input_path: Path,
    collection_name: str,
    report_path: Path,
    plan_path: Path,
    validation_status: str,
    qdrant_upload_ready: bool,
    total_points: int,
    vector_size: int,
    duplicate_point_id_count: int,
    missing_embedding_count: int,
    inconsistent_embedding_dim_count: int,
    empty_text_count: int,
    missing_required_counts: dict[str, int],
    missing_optional_counts: dict[str, int],
    failures: list[str],
    warnings: list[str],
    df: pd.DataFrame,
    qdrant_check_status: str | None,
    qdrant_check_detail: str | None,
) -> str:
    lines = [
        "# NSoud Qdrant Upload Dry Run",
        "",
        f"- Input: `{input_path}`",
        f"- Collection name: `{collection_name}`",
        f"- Validation status: **{validation_status}**",
        f"- QDRANT_UPLOAD_READY: **{'true' if qdrant_upload_ready else 'false'}**",
        f"- Total points: **{total_points}**",
        f"- Vector size: **{vector_size}**",
        f"- Duplicate point_id count: **{duplicate_point_id_count}**",
        f"- Missing embedding count: **{missing_embedding_count}**",
        f"- Inconsistent embedding_dim count: **{inconsistent_embedding_dim_count}**",
        f"- Empty text count: **{empty_text_count}**",
        f"- Report path: `{report_path}`",
        f"- Upload plan path: `{plan_path}`",
        "",
        "## Status",
    ]

    status_items = failures + warnings if failures or warnings else ["Qdrant upload dry run passed."]
    lines.extend(f"- {item}" for item in status_items)

    if qdrant_check_status is not None:
        lines.extend(["", "## Optional Qdrant Check", f"- {qdrant_check_status}"])
        if qdrant_check_detail:
            lines.append(f"- Detail: `{qdrant_check_detail}`")

    lines.extend(
        [
            "",
            "## Missing Required Metadata Counts",
            "",
            "| Field | Missing Count |",
            "| --- | ---: |",
        ]
    )
    for field_name in ["point_id", "text", "embedding", "embedding_dim", *REQUIRED_PAYLOAD_FIELDS]:
        lines.append(f"| `{field_name}` | {missing_required_counts.get(field_name, 0)} |")

    lines.extend(
        [
            "",
            "## Missing Optional Metadata Counts",
            "",
            "| Field | Missing Count |",
            "| --- | ---: |",
        ]
    )
    for field_name in OPTIONAL_PAYLOAD_FIELDS:
        lines.append(f"| `{field_name}` | {missing_optional_counts.get(field_name, 0)} |")

    lines.append("")
    lines.extend(render_distribution_table("Source Distribution", distribution_counts(df, "source")))
    lines.extend(render_distribution_table("Authority Level Distribution", distribution_counts(df, "authority_level")))
    lines.extend(render_distribution_table("Document Type Distribution", distribution_counts(df, "document_type")))
    lines.extend(render_distribution_table("Legal Area Distribution", distribution_counts(df, "legal_area")))
    lines.extend(
        [
            "## Recommended Next Docker Command",
            "",
            f"`docker compose exec api python app/nsoud/qdrant_upload_dry_run.py --input {input_path.as_posix()} --collection {collection_name} --out-report {report_path.as_posix()} --out-plan {plan_path.as_posix()}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    if pyarrow is None:
        print("dry run status: FAIL")
        print("error: pyarrow is required for Parquet input.")
        print("install command: pip install pyarrow")
        return 1

    try:
        df = load_embeddings(args.input)
    except Exception as exc:
        print("dry run status: FAIL")
        print("QDRANT_UPLOAD_READY: false")
        print(f"error: {exc}")
        return 1

    (
        validation_status,
        qdrant_upload_ready,
        missing_required_counts,
        missing_optional_counts,
        failures,
        warnings,
        duplicate_point_id_count,
        missing_embedding_count,
        inconsistent_embedding_dim_count,
        empty_text_count,
        vector_size,
    ) = validate_dataframe(df)

    qdrant_check_status, qdrant_check_detail = maybe_check_qdrant(args.check_qdrant)

    try:
        write_upload_plan(
            args.out_plan,
            collection_name=args.collection,
            vector_size=vector_size,
            total_points=len(df),
            input_path=args.input,
        )
        report = build_report(
            input_path=args.input,
            collection_name=args.collection,
            report_path=args.out_report,
            plan_path=args.out_plan,
            validation_status=validation_status,
            qdrant_upload_ready=qdrant_upload_ready,
            total_points=len(df),
            vector_size=vector_size,
            duplicate_point_id_count=duplicate_point_id_count,
            missing_embedding_count=missing_embedding_count,
            inconsistent_embedding_dim_count=inconsistent_embedding_dim_count,
            empty_text_count=empty_text_count,
            missing_required_counts=missing_required_counts,
            missing_optional_counts=missing_optional_counts,
            failures=failures,
            warnings=warnings,
            df=df,
            qdrant_check_status=qdrant_check_status,
            qdrant_check_detail=qdrant_check_detail,
        )
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        args.out_report.write_text(report, encoding="utf-8")
    except Exception as exc:
        print("dry run status: FAIL")
        print("QDRANT_UPLOAD_READY: false")
        print(f"error: {exc}")
        return 1

    summary = DryRunSummary(
        dry_run_status="PASS" if validation_status != "FAIL" else "FAIL",
        qdrant_upload_ready=qdrant_upload_ready,
        collection_name=args.collection,
        total_points=len(df),
        vector_size=vector_size,
        duplicate_point_id_count=duplicate_point_id_count,
        missing_embedding_count=missing_embedding_count,
        inconsistent_embedding_dim_count=inconsistent_embedding_dim_count,
        empty_text_count=empty_text_count,
        report_path=args.out_report,
        plan_path=args.out_plan,
        validation_status=validation_status,
    )
    print(f"dry run status: {summary.dry_run_status}")
    print(f"QDRANT_UPLOAD_READY: {'true' if summary.qdrant_upload_ready else 'false'}")
    print(f"collection name: {summary.collection_name}")
    print(f"total points: {summary.total_points}")
    print(f"vector size: {summary.vector_size}")
    print(f"duplicate point_id count: {summary.duplicate_point_id_count}")
    print(f"missing embedding count: {summary.missing_embedding_count}")
    print(f"inconsistent embedding_dim count: {summary.inconsistent_embedding_dim_count}")
    print(f"empty text count: {summary.empty_text_count}")
    print(f"report path: {summary.report_path}")
    print(f"upload plan path: {summary.plan_path}")
    print(f"validation status: {summary.validation_status}")
    return 1 if validation_status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
