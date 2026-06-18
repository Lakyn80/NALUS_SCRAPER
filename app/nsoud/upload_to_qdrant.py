from __future__ import annotations

import argparse
import json
import math
import uuid
from dataclasses import asdict, dataclass
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


EXPECTED_INPUT_ROWS = 1862
EXPECTED_VECTOR_SIZE = 768
EXPECTED_DISTANCE = "Cosine"
EXPECTED_CHUNKING_STRATEGY = "document_section_aware"
OLD_COLLECTION_NAME = "nsoud_chunks_test_2025_01_03"
TARGET_COLLECTION_NAME = "nsoud_chunks_section_aware_test_2025_01_03"
DEFAULT_INPUT_PATH = Path("app/artifacts/nsoud/rag_ready/nsoud_embeddings_2025_01_03.parquet")
DEFAULT_QDRANT_URL = "http://qdrant:6333"
DEFAULT_BATCH_SIZE = 128
DEFAULT_ARTIFACTS_ROOT = Path("app/artifacts/nsoud/qdrant")

REQUIRED_COLUMNS = [
    "point_id",
    "text",
    "document_id",
    "chunk_id",
    "section_id",
    "section_type",
    "section_index",
    "chunk_index",
    "chunk_index_in_section",
    "total_chunks_in_document",
    "total_chunks_in_section",
    "previous_chunk_id",
    "next_chunk_id",
    "previous_section_chunk_id",
    "next_section_chunk_id",
    "structure_confidence",
    "structure_status",
    "structure_needs_review",
    "section_source",
    "chunking_strategy",
    "embedding",
    "embedding_dim",
]
REQUIRED_NONNULL_COLUMNS = [
    "point_id",
    "text",
    "document_id",
    "chunk_id",
    "section_id",
    "section_type",
    "section_index",
    "chunk_index",
    "chunk_index_in_section",
    "total_chunks_in_document",
    "total_chunks_in_section",
    "structure_confidence",
    "structure_status",
    "structure_needs_review",
    "section_source",
    "chunking_strategy",
]
NULLABLE_METADATA_COLUMNS = [
    "previous_chunk_id",
    "next_chunk_id",
    "previous_section_chunk_id",
    "next_section_chunk_id",
]
PAYLOAD_FIELDS_TO_PRESERVE = [
    "point_id",
    "chunk_id",
    "text",
    "document_id",
    "section_id",
    "section_type",
    "section_index",
    "chunk_index",
    "chunk_index_in_section",
    "total_chunks_in_document",
    "total_chunks_in_section",
    "previous_chunk_id",
    "next_chunk_id",
    "previous_section_chunk_id",
    "next_section_chunk_id",
    "structure_confidence",
    "structure_status",
    "structure_needs_review",
    "section_source",
    "chunking_strategy",
]


@dataclass(frozen=True)
class ValidationResult:
    status: str
    total_rows: int
    vector_size: int
    duplicate_point_id_count: int
    duplicate_chunk_id_count: int
    missing_embedding_count: int
    empty_text_count: int
    missing_required_metadata_count: int
    non_section_aware_row_count: int
    errors: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class OldCollectionState:
    exists: bool
    point_count_before: int | None
    point_count_after: int | None
    untouched: bool


@dataclass(frozen=True)
class UploadResult:
    status: str
    total_uploaded_points: int
    upload_batch_count: int
    final_collection_point_count: int | None
    recreated_collection: bool
    distance: str
    old_collection_state: OldCollectionState
    errors: list[str]
    warnings: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the final section-aware NSoud embeddings into the approved Qdrant test collection."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Input embeddings parquet path.")
    parser.add_argument(
        "--collection",
        default=TARGET_COLLECTION_NAME,
        help="Target Qdrant collection name. Only the approved section-aware test collection is allowed.",
    )
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL, help="Qdrant base URL.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Upsert batch size.")
    parser.add_argument("--out-manifest", type=Path, help="Output upload manifest JSON path.")
    parser.add_argument("--out-report", type=Path, help="Output upload report Markdown path.")
    parser.add_argument("--recreate", action="store_true", help="Safely recreate the approved target collection.")
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_allowed_collection(collection_name: str) -> None:
    if collection_name != TARGET_COLLECTION_NAME:
        raise ValueError(
            f"Refusing to operate on collection '{collection_name}'. Only '{TARGET_COLLECTION_NAME}' is allowed."
        )


def default_output_paths(collection_name: str) -> tuple[Path, Path]:
    artifact_dir = DEFAULT_ARTIFACTS_ROOT / collection_name
    return artifact_dir / "upload_manifest.json", artifact_dir / "upload_report.md"


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value)


def normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


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


def count_missing(series: pd.Series, *, allow_empty_string: bool) -> int:
    if allow_empty_string:
        return int(series.isna().sum())
    return int(series.map(lambda value: normalize_text(value).strip() == "").sum())


def detect_vector_size(df: pd.DataFrame) -> tuple[int, int]:
    dims = [int(value) for value in df["embedding_dim"].dropna().tolist()]
    unique_dims = sorted(set(dims))
    if not unique_dims:
        return 0, 0
    expected = unique_dims[0]
    inconsistent_count = int(df["embedding_dim"].map(lambda value: int(value) != expected).sum())
    return expected, inconsistent_count


def load_embeddings(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def validate_dataframe(df: pd.DataFrame) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        return ValidationResult(
            status="FAIL",
            total_rows=len(df),
            vector_size=0,
            duplicate_point_id_count=0,
            duplicate_chunk_id_count=0,
            missing_embedding_count=0,
            empty_text_count=0,
            missing_required_metadata_count=len(missing_columns),
            non_section_aware_row_count=0,
            errors=errors,
            warnings=warnings,
        )

    total_rows = len(df)
    if total_rows != EXPECTED_INPUT_ROWS:
        errors.append(f"Input row count is {total_rows}, expected {EXPECTED_INPUT_ROWS}.")

    duplicate_point_id_count = int(df["point_id"].duplicated(keep=False).sum())
    if duplicate_point_id_count > 0:
        errors.append("Duplicate point_id values detected.")

    duplicate_chunk_id_count = int(df["chunk_id"].duplicated(keep=False).sum())
    if duplicate_chunk_id_count > 0:
        errors.append("Duplicate chunk_id values detected.")

    missing_embedding_count = int(df["embedding"].map(is_missing_embedding).sum())
    if missing_embedding_count > 0:
        errors.append("One or more rows are missing embeddings.")

    empty_text_count = int(df["text"].map(lambda value: normalize_text(value).strip() == "").sum())
    if empty_text_count > 0:
        errors.append("One or more rows have empty text.")

    missing_required_metadata_count = 0
    for column in REQUIRED_NONNULL_COLUMNS:
        missing_required_metadata_count += count_missing(df[column], allow_empty_string=False)
    if missing_required_metadata_count > 0:
        errors.append("Required non-null metadata contains missing values.")

    for column in NULLABLE_METADATA_COLUMNS:
        if column not in df.columns:
            errors.append(f"Missing required metadata column '{column}'.")

    vector_size, inconsistent_embedding_dim_count = detect_vector_size(df)
    if vector_size != EXPECTED_VECTOR_SIZE:
        errors.append(f"Embedding dimension is {vector_size}, expected {EXPECTED_VECTOR_SIZE}.")
    if inconsistent_embedding_dim_count > 0:
        errors.append("Embedding dimensions are inconsistent across rows.")

    non_section_aware_row_count = int(
        df["chunking_strategy"].map(lambda value: normalize_text(value).strip() != EXPECTED_CHUNKING_STRATEGY).sum()
    )
    if non_section_aware_row_count > 0:
        errors.append(
            f"Found {non_section_aware_row_count} rows without chunking_strategy='{EXPECTED_CHUNKING_STRATEGY}'."
        )

    status = "FAIL" if errors else "WARN" if warnings else "PASS"
    return ValidationResult(
        status=status,
        total_rows=total_rows,
        vector_size=vector_size,
        duplicate_point_id_count=duplicate_point_id_count,
        duplicate_chunk_id_count=duplicate_chunk_id_count,
        missing_embedding_count=missing_embedding_count,
        empty_text_count=empty_text_count,
        missing_required_metadata_count=missing_required_metadata_count,
        non_section_aware_row_count=non_section_aware_row_count,
        errors=errors,
        warnings=warnings,
    )


def to_qdrant_point_id(raw_point_id: Any) -> str:
    normalized = normalize_text(raw_point_id).strip()
    if not normalized:
        raise ValueError("point_id must not be empty.")
    try:
        return str(uuid.UUID(normalized))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"nsoud:{normalized}"))


def build_payload(row: pd.Series) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field_name in row.index:
        if field_name == "embedding":
            continue
        value = normalize_scalar(row[field_name])
        if field_name == "text":
            payload[field_name] = normalize_text(value)
        else:
            payload[field_name] = value
    return payload


def build_points(df: pd.DataFrame) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        points.append(
            {
                "id": to_qdrant_point_id(row["point_id"]),
                "vector": [float(value) for value in row["embedding"]],
                "payload": build_payload(row),
            }
        )
    return points


def chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def detect_distance_name(vectors_config: Any) -> str | None:
    if vectors_config is None:
        return None
    distance = getattr(vectors_config, "distance", None)
    if distance is None and isinstance(vectors_config, dict):
        distance = vectors_config.get("distance")
    if distance is None:
        return None
    name = getattr(distance, "name", None)
    if name:
        return str(name).title()
    return str(distance).split(".")[-1].title()


def detect_vector_param_size(vectors_config: Any) -> int | None:
    if vectors_config is None:
        return None
    size = getattr(vectors_config, "size", None)
    if size is None and isinstance(vectors_config, dict):
        size = vectors_config.get("size")
    return int(size) if size is not None else None


def capture_old_collection_state(client: Any) -> OldCollectionState:
    exists = client.collection_exists(OLD_COLLECTION_NAME)
    before_count = int(client.count(collection_name=OLD_COLLECTION_NAME).count) if exists else None
    return OldCollectionState(
        exists=exists,
        point_count_before=before_count,
        point_count_after=before_count,
        untouched=True,
    )


def finalize_old_collection_state(client: Any, before_state: OldCollectionState) -> OldCollectionState:
    if not before_state.exists:
        return before_state
    after_count = int(client.count(collection_name=OLD_COLLECTION_NAME).count)
    return OldCollectionState(
        exists=True,
        point_count_before=before_state.point_count_before,
        point_count_after=after_count,
        untouched=before_state.point_count_before == after_count,
    )


def ensure_collection(client: Any, *, collection_name: str, vector_size: int, recreate: bool) -> tuple[bool, str]:
    from qdrant_client.models import Distance, VectorParams

    if client.collection_exists(collection_name):
        if recreate:
            client.delete_collection(collection_name)
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            return True, EXPECTED_DISTANCE

        info = client.get_collection(collection_name)
        vectors_config = info.config.params.vectors
        configured_size = detect_vector_param_size(vectors_config)
        configured_distance = detect_distance_name(vectors_config)
        if configured_size != vector_size:
            raise RuntimeError(
                f"Collection '{collection_name}' has vector size {configured_size}, expected {vector_size}."
            )
        if configured_distance != EXPECTED_DISTANCE:
            raise RuntimeError(
                f"Collection '{collection_name}' uses distance {configured_distance}, expected {EXPECTED_DISTANCE}."
            )
        return False, configured_distance or EXPECTED_DISTANCE

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    return False, EXPECTED_DISTANCE


def upload_points(client: Any, *, collection_name: str, points: list[dict[str, Any]], batch_size: int) -> tuple[int, int]:
    from qdrant_client.models import PointStruct

    total_uploaded_points = 0
    for batch in chunked(points, batch_size):
        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(id=point["id"], vector=point["vector"], payload=point["payload"])
                for point in batch
            ],
        )
        total_uploaded_points += len(batch)

    final_count = int(client.count(collection_name=collection_name).count)
    return total_uploaded_points, final_count


def write_manifest(
    out_path: Path,
    *,
    collection_name: str,
    artifact_dir: Path,
    qdrant_url: str,
    input_path: Path,
    validation: ValidationResult,
    upload: UploadResult,
    batch_size: int,
    started_at: str,
    finished_at: str,
) -> None:
    payload = {
        "status": upload.status,
        "collection_name": collection_name,
        "artifact_directory": str(artifact_dir),
        "qdrant_url": qdrant_url,
        "input_path": str(input_path),
        "input_rows": validation.total_rows,
        "uploaded_points": upload.total_uploaded_points,
        "final_collection_point_count": upload.final_collection_point_count,
        "vector_size": validation.vector_size,
        "distance": upload.distance,
        "duplicate_point_id_count": validation.duplicate_point_id_count,
        "duplicate_chunk_id_count": validation.duplicate_chunk_id_count,
        "missing_embedding_count": validation.missing_embedding_count,
        "empty_text_count": validation.empty_text_count,
        "missing_required_metadata_count": validation.missing_required_metadata_count,
        "non_section_aware_row_count": validation.non_section_aware_row_count,
        "recreated_collection": upload.recreated_collection,
        "batch_size": batch_size,
        "old_collection_state": asdict(upload.old_collection_state),
        "preserved_payload_fields": PAYLOAD_FIELDS_TO_PRESERVE,
        "started_at": started_at,
        "finished_at": finished_at,
        "warnings": upload.warnings,
        "errors": upload.errors,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_report(
    *,
    collection_name: str,
    artifact_dir: Path,
    qdrant_url: str,
    input_path: Path,
    manifest_path: Path,
    report_path: Path,
    validation: ValidationResult,
    upload: UploadResult,
) -> str:
    old_collection_message = (
        f"yes ({upload.old_collection_state.point_count_before} -> {upload.old_collection_state.point_count_after})"
        if upload.old_collection_state.exists
        else "not present"
    )
    changed_files = [
        "app/nsoud/upload_to_qdrant.py",
        str(manifest_path).replace("\\", "/"),
        str(report_path).replace("\\", "/"),
    ]
    lines = [
        "# NSoud Qdrant Upload Report",
        "",
        f"- Status: **{upload.status}**",
        f"- Target collection: `{collection_name}`",
        f"- Qdrant URL: `{qdrant_url}`",
        f"- Artifact directory: `{artifact_dir}`",
        f"- Input path: `{input_path}`",
        f"- Input rows: **{validation.total_rows}**",
        f"- Uploaded points: **{upload.total_uploaded_points}**",
        f"- Final collection point count: **{upload.final_collection_point_count}**",
        f"- Vector size: **{validation.vector_size}**",
        f"- Distance: **{upload.distance}**",
        f"- Duplicate point_id count: **{validation.duplicate_point_id_count}**",
        f"- Duplicate chunk_id count: **{validation.duplicate_chunk_id_count}**",
        f"- Missing embedding count: **{validation.missing_embedding_count}**",
        f"- Empty text count: **{validation.empty_text_count}**",
        f"- Missing required metadata count: **{validation.missing_required_metadata_count}**",
        f"- Rows outside `{EXPECTED_CHUNKING_STRATEGY}`: **{validation.non_section_aware_row_count}**",
        f"- Old collection untouched: **{old_collection_message}**",
        f"- Manifest path: `{manifest_path}`",
        f"- Report path: `{report_path}`",
        "",
        "## Preserved Payload Metadata",
        "",
    ]
    lines.extend(f"- `{field_name}`" for field_name in PAYLOAD_FIELDS_TO_PRESERVE)
    lines.extend(["", "## Warnings", ""])
    if upload.warnings:
        lines.extend(f"- {warning}" for warning in upload.warnings)
    else:
        lines.append("- None.")
    lines.extend(["", "## Errors", ""])
    if upload.errors:
        lines.extend(f"- {error}" for error in upload.errors)
    else:
        lines.append("- None.")
    lines.extend(["", "## Changed Files", ""])
    lines.extend(f"- `{path}`" for path in changed_files)
    lines.append("")
    return "\n".join(lines)


def write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def print_summary(
    *,
    collection_name: str,
    artifact_dir: Path,
    manifest_path: Path,
    report_path: Path,
    validation: ValidationResult,
    upload: UploadResult,
) -> None:
    print(f"upload status: {upload.status}")
    print(f"target collection: {collection_name}")
    print(f"artifact directory: {artifact_dir}")
    print(f"input rows: {validation.total_rows}")
    print(f"uploaded points: {upload.total_uploaded_points}")
    print(f"final collection point count: {upload.final_collection_point_count}")
    print(f"vector size: {validation.vector_size}")
    print(f"duplicate point_id count: {validation.duplicate_point_id_count}")
    print(f"duplicate chunk_id count: {validation.duplicate_chunk_id_count}")
    print(f"missing embedding count: {validation.missing_embedding_count}")
    print(f"empty text count: {validation.empty_text_count}")
    print(f"missing required metadata count: {validation.missing_required_metadata_count}")
    print(
        "old collection untouched confirmation: "
        f"{'PASS' if upload.old_collection_state.untouched else 'FAIL'}"
    )
    if upload.old_collection_state.exists:
        print(
            f"old collection counts: {upload.old_collection_state.point_count_before} -> "
            f"{upload.old_collection_state.point_count_after}"
        )
    else:
        print("old collection counts: not present")
    print(f"manifest path: {manifest_path}")
    print(f"report path: {report_path}")
    print("changed files:")
    print("app/nsoud/upload_to_qdrant.py")
    print(str(manifest_path).replace("\\", "/"))
    print(str(report_path).replace("\\", "/"))


def main() -> int:
    args = parse_args()
    started_at = utc_now_iso()

    if pyarrow is None:
        print("upload status: FAIL")
        print("error: pyarrow is required for Parquet input.")
        return 1

    if args.batch_size <= 0:
        print("upload status: FAIL")
        print("error: --batch-size must be greater than 0.")
        return 1

    try:
        ensure_allowed_collection(args.collection)
    except Exception as exc:
        print("upload status: FAIL")
        print(f"error: {exc}")
        return 1

    manifest_path = args.out_manifest
    report_path = args.out_report
    if manifest_path is None or report_path is None:
        default_manifest_path, default_report_path = default_output_paths(args.collection)
        manifest_path = manifest_path or default_manifest_path
        report_path = report_path or default_report_path
    artifact_dir = manifest_path.parent

    try:
        df = load_embeddings(args.input)
    except Exception as exc:
        print("upload status: FAIL")
        print(f"error: failed to load embeddings: {exc}")
        return 1

    validation = validate_dataframe(df)
    upload = UploadResult(
        status="FAIL" if validation.errors else validation.status,
        total_uploaded_points=0,
        upload_batch_count=0,
        final_collection_point_count=None,
        recreated_collection=False,
        distance=EXPECTED_DISTANCE,
        old_collection_state=OldCollectionState(
            exists=False,
            point_count_before=None,
            point_count_after=None,
            untouched=True,
        ),
        errors=list(validation.errors),
        warnings=list(validation.warnings),
    )

    if not validation.errors:
        try:
            from qdrant_client import QdrantClient

            client = QdrantClient(url=args.qdrant_url, timeout=60, check_compatibility=False)
            old_collection_before = capture_old_collection_state(client)
            recreated_collection, distance = ensure_collection(
                client,
                collection_name=args.collection,
                vector_size=validation.vector_size,
                recreate=args.recreate,
            )
            points = build_points(df)
            total_uploaded_points, final_count = upload_points(
                client,
                collection_name=args.collection,
                points=points,
                batch_size=args.batch_size,
            )
            old_collection_after = finalize_old_collection_state(client, old_collection_before)

            errors = list(validation.errors)
            warnings = list(validation.warnings)
            if final_count != EXPECTED_INPUT_ROWS:
                errors.append(
                    f"Final collection point count is {final_count}, expected {EXPECTED_INPUT_ROWS}."
                )
            if not old_collection_after.untouched:
                errors.append(
                    f"Old collection '{OLD_COLLECTION_NAME}' changed from "
                    f"{old_collection_after.point_count_before} to {old_collection_after.point_count_after}."
                )

            status = "FAIL" if errors else "WARN" if warnings else "PASS"
            upload = UploadResult(
                status=status,
                total_uploaded_points=total_uploaded_points,
                upload_batch_count=math.ceil(total_uploaded_points / args.batch_size) if total_uploaded_points else 0,
                final_collection_point_count=final_count,
                recreated_collection=recreated_collection,
                distance=distance,
                old_collection_state=old_collection_after,
                errors=errors,
                warnings=warnings,
            )
        except Exception as exc:
            upload = UploadResult(
                status="FAIL",
                total_uploaded_points=upload.total_uploaded_points,
                upload_batch_count=upload.upload_batch_count,
                final_collection_point_count=upload.final_collection_point_count,
                recreated_collection=upload.recreated_collection,
                distance=upload.distance,
                old_collection_state=upload.old_collection_state,
                errors=[*upload.errors, str(exc)],
                warnings=upload.warnings,
            )

    finished_at = utc_now_iso()

    try:
        write_manifest(
            manifest_path,
            collection_name=args.collection,
            artifact_dir=artifact_dir,
            qdrant_url=args.qdrant_url,
            input_path=args.input,
            validation=validation,
            upload=upload,
            batch_size=args.batch_size,
            started_at=started_at,
            finished_at=finished_at,
        )
        write_report(
            report_path,
            build_report(
                collection_name=args.collection,
                artifact_dir=artifact_dir,
                qdrant_url=args.qdrant_url,
                input_path=args.input,
                manifest_path=manifest_path,
                report_path=report_path,
                validation=validation,
                upload=upload,
            ),
        )
    except Exception as exc:
        print("upload status: FAIL")
        print(f"error: failed to write artifacts: {exc}")
        return 1

    print_summary(
        collection_name=args.collection,
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        report_path=report_path,
        validation=validation,
        upload=upload,
    )
    return 0 if upload.status != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
