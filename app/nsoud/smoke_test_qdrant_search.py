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


TARGET_COLLECTION = "nsoud_chunks_section_aware_test_2025_01_03"
OLD_COLLECTION = "nsoud_chunks_test_2025_01_03"
EXPECTED_POINT_COUNT = 1862
EXPECTED_VECTOR_SIZE = 768
EXPECTED_CHUNKING_STRATEGY = "document_section_aware"
DEFAULT_QDRANT_URL = "http://qdrant:6333"
DEFAULT_TOP_K = 5
DEFAULT_EMBEDDINGS_PATH = Path("app/artifacts/nsoud/rag_ready/nsoud_embeddings_2025_01_03.parquet")
DEFAULT_OUTPUT_PATH = Path(
    "app/artifacts/nsoud/qdrant/nsoud_chunks_section_aware_test_2025_01_03/smoke_test.md"
)
REQUIRED_TOP_RESULT_PAYLOAD_FIELDS = [
    "point_id",
    "chunk_id",
    "text",
    "document_id",
    "section_id",
    "section_type",
    "chunk_index",
    "chunk_index_in_section",
    "total_chunks_in_document",
    "total_chunks_in_section",
    "previous_chunk_id",
    "next_chunk_id",
    "previous_section_chunk_id",
    "next_section_chunk_id",
    "structure_status",
    "section_source",
    "chunking_strategy",
]


@dataclass(frozen=True)
class SmokeCase:
    test_name: str
    row_index: int
    point_id: str
    chunk_id: str
    section_type: str
    structure_status: str
    vector: list[float]


@dataclass(frozen=True)
class SmokeCaseResult:
    test_name: str
    row_index: int
    query_chunk_id: str
    top_hit_chunk_id: str
    original_found_in_top_5: bool
    top_hit_score: float | None
    passed: bool
    errors: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the NSoud Qdrant smoke test for the section-aware collection.")
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL, help="Qdrant base URL.")
    parser.add_argument("--collection", default=TARGET_COLLECTION, help="Target Qdrant collection name.")
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS_PATH, help="Embeddings parquet path.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output Markdown report path.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of search results to inspect.")
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


def load_embeddings(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def detect_vector_size(df: pd.DataFrame) -> tuple[int, int]:
    dims = sorted({int(value) for value in df["embedding_dim"].dropna().tolist()})
    if not dims:
        return 0, 0
    expected_dim = dims[0]
    inconsistent_count = int(df["embedding_dim"].map(lambda value: int(value) != expected_dim).sum())
    return expected_dim, inconsistent_count


def select_first_matching_index(df: pd.DataFrame, column_name: str, expected_value: str) -> int | None:
    matches = df.index[df[column_name].fillna("").map(str) == expected_value].tolist()
    if not matches:
        return None
    return int(matches[0])


def build_smoke_case(df: pd.DataFrame, *, test_name: str, row_index: int) -> SmokeCase:
    row = df.iloc[row_index]
    return SmokeCase(
        test_name=test_name,
        row_index=row_index,
        point_id=normalize_text(row["point_id"]),
        chunk_id=normalize_text(row["chunk_id"]),
        section_type=normalize_text(row["section_type"]),
        structure_status=normalize_text(row["structure_status"]),
        vector=[float(value) for value in row["embedding"]],
    )


def select_smoke_cases(df: pd.DataFrame) -> tuple[list[SmokeCase], list[str]]:
    warnings: list[str] = []
    candidate_specs: list[tuple[str, int]] = [
        ("first_row", 0),
        ("middle_row", len(df) // 2),
        ("last_row", len(df) - 1),
    ]

    for test_name, field_name, expected_value in [
        ("section_type_operative_part", "section_type", "operative_part"),
        ("section_type_reasoning", "section_type", "reasoning"),
        ("section_type_appeal_instruction", "section_type", "appeal_instruction"),
        ("structure_status_medium", "structure_status", "medium"),
    ]:
        index = select_first_matching_index(df, field_name, expected_value)
        if index is None:
            warnings.append(f'No row with {field_name} = "{expected_value}" was found.')
            continue
        candidate_specs.append((test_name, index))

    cases: list[SmokeCase] = []
    seen_indexes: set[int] = set()
    for test_name, row_index in candidate_specs:
        if row_index in seen_indexes:
            continue
        seen_indexes.add(row_index)
        cases.append(build_smoke_case(df, test_name=test_name, row_index=row_index))
    return cases, warnings


def detect_vector_param_size(vectors_config: Any) -> int | None:
    size = getattr(vectors_config, "size", None)
    if size is None and isinstance(vectors_config, dict):
        size = vectors_config.get("size")
    return int(size) if size is not None else None


def verify_collection(client: Any, *, collection_name: str) -> tuple[int, int]:
    if not client.collection_exists(collection_name):
        raise RuntimeError(f"Collection '{collection_name}' does not exist.")
    info = client.get_collection(collection_name)
    vector_size = detect_vector_param_size(info.config.params.vectors) or 0
    point_count = int(client.count(collection_name=collection_name).count)
    return point_count, vector_size


def get_old_collection_count(client: Any, *, collection_name: str) -> int | None:
    if not client.collection_exists(collection_name):
        return None
    return int(client.count(collection_name=collection_name).count)


def search_top_k(client: Any, *, collection_name: str, vector: list[float], limit: int) -> list[Any]:
    result = client.query_points(
        collection_name=collection_name,
        query=vector,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return list(result.points)


def validate_top_hit_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field_name in REQUIRED_TOP_RESULT_PAYLOAD_FIELDS:
        if field_name not in payload:
            errors.append(f"Top result payload is missing `{field_name}`.")
            continue
        if field_name in {
            "previous_chunk_id",
            "next_chunk_id",
            "previous_section_chunk_id",
            "next_section_chunk_id",
        }:
            continue
        if normalize_text(payload.get(field_name)).strip() == "":
            errors.append(f"Top result payload has empty `{field_name}`.")
    if normalize_text(payload.get("chunking_strategy")) != EXPECTED_CHUNKING_STRATEGY:
        errors.append(
            f"Top result payload has chunking_strategy='{normalize_text(payload.get('chunking_strategy'))}', "
            f"expected '{EXPECTED_CHUNKING_STRATEGY}'."
        )
    return errors


def run_smoke_case(client: Any, *, collection_name: str, case: SmokeCase, top_k: int) -> SmokeCaseResult:
    errors: list[str] = []
    points = search_top_k(client, collection_name=collection_name, vector=case.vector, limit=top_k)
    if not points:
        errors.append("Search returned zero results.")
        return SmokeCaseResult(
            test_name=case.test_name,
            row_index=case.row_index,
            query_chunk_id=case.chunk_id,
            top_hit_chunk_id="",
            original_found_in_top_5=False,
            top_hit_score=None,
            passed=False,
            errors=errors,
        )

    top_hit_payload = dict(points[0].payload or {})
    top_hit_chunk_id = normalize_text(top_hit_payload.get("chunk_id"))
    original_found = False
    for point in points:
        payload = dict(point.payload or {})
        if normalize_text(payload.get("chunk_id")) == case.chunk_id:
            original_found = True
            break

    if not original_found:
        errors.append(f"Original chunk_id '{case.chunk_id}' was not found in top {top_k} results.")

    errors.extend(validate_top_hit_payload(top_hit_payload))
    return SmokeCaseResult(
        test_name=case.test_name,
        row_index=case.row_index,
        query_chunk_id=case.chunk_id,
        top_hit_chunk_id=top_hit_chunk_id,
        original_found_in_top_5=original_found,
        top_hit_score=float(points[0].score),
        passed=not errors,
        errors=errors,
    )


def build_report(
    *,
    status: str,
    collection_name: str,
    expected_point_count: int,
    actual_point_count: int,
    vector_size: int,
    old_collection_before: int | None,
    old_collection_after: int | None,
    smoke_cases: list[SmokeCase],
    results: list[SmokeCaseResult],
    warnings: list[str],
    errors: list[str],
    output_path: Path,
) -> str:
    passed_count = sum(1 for result in results if result.passed)
    failed_count = len(results) - passed_count
    changed_files = [
        "app/nsoud/smoke_test_qdrant_search.py",
        str(output_path).replace("\\", "/"),
    ]
    lines = [
        "# NSoud Qdrant Smoke Test",
        "",
        f"- Status: **{status}**",
        f"- Target collection: `{collection_name}`",
        f"- Expected point count: **{expected_point_count}**",
        f"- Actual point count: **{actual_point_count}**",
        f"- Vector size: **{vector_size}**",
        f"- Tests passed: **{passed_count}**",
        f"- Tests failed: **{failed_count}**",
        f"- Old collection count before/after: **{old_collection_before} -> {old_collection_after}**",
        f"- Output report path: `{output_path}`",
        "",
        "## Selected Cases",
        "",
        "| test_name | row_index | chunk_id | section_type | structure_status |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for case in smoke_cases:
        lines.append(
            f"| {case.test_name} | {case.row_index} | {case.chunk_id} | {case.section_type} | {case.structure_status} |"
        )

    lines.extend(
        [
            "",
            "## Search Results",
            "",
            "| test_name | query_chunk_id | top_hit_chunk_id | original_found_in_top_5 | top_hit_score |",
            "| --- | --- | --- | --- | ---: |",
        ]
    )
    for result in results:
        score = f"{result.top_hit_score:.6f}" if result.top_hit_score is not None else ""
        lines.append(
            f"| {result.test_name} | {result.query_chunk_id} | {result.top_hit_chunk_id} | "
            f"{'true' if result.original_found_in_top_5 else 'false'} | {score} |"
        )

    lines.extend(["", "## Warnings"])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None.")

    lines.extend(["", "## Errors"])
    if errors:
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.append("- None.")

    lines.extend(["", "## Changed Files"])
    lines.extend(f"- `{path}`" for path in changed_files)
    lines.append("")
    return "\n".join(lines)


def print_summary(
    *,
    status: str,
    collection_name: str,
    expected_point_count: int,
    actual_point_count: int,
    vector_size: int,
    passed_count: int,
    failed_count: int,
    old_collection_before: int | None,
    old_collection_after: int | None,
    output_path: Path,
) -> None:
    print(f"smoke status: {status}")
    print(f"target collection: {collection_name}")
    print(f"expected point count: {expected_point_count}")
    print(f"actual point count: {actual_point_count}")
    print(f"vector size: {vector_size}")
    print(f"tests passed: {passed_count}")
    print(f"tests failed: {failed_count}")
    print(f"old collection count before/after: {old_collection_before} -> {old_collection_after}")
    print(f"output report path: {output_path}")
    print("changed files:")
    print("app/nsoud/smoke_test_qdrant_search.py")
    print(str(output_path).replace("\\", "/"))


def main() -> int:
    args = parse_args()

    if pyarrow is None:
        print("smoke status: FAIL")
        print("error: pyarrow is required for Parquet input.")
        return 1

    if args.top_k <= 0:
        print("smoke status: FAIL")
        print("error: --top-k must be greater than 0.")
        return 1

    if args.collection != TARGET_COLLECTION:
        print("smoke status: FAIL")
        print(
            f"error: refusing to operate on collection '{args.collection}'. "
            f"Only '{TARGET_COLLECTION}' is allowed."
        )
        return 1

    try:
        df = load_embeddings(args.embeddings)
    except Exception as exc:
        print("smoke status: FAIL")
        print(f"error: failed to load embeddings: {exc}")
        return 1

    errors: list[str] = []
    warnings: list[str] = []
    if len(df) != EXPECTED_POINT_COUNT:
        errors.append(f"Embeddings parquet row count is {len(df)}, expected {EXPECTED_POINT_COUNT}.")

    vector_size, inconsistent_vector_size_count = detect_vector_size(df)
    if vector_size != EXPECTED_VECTOR_SIZE:
        errors.append(f"Embeddings vector size is {vector_size}, expected {EXPECTED_VECTOR_SIZE}.")
    if inconsistent_vector_size_count > 0:
        errors.append("Embeddings parquet contains inconsistent embedding_dim values.")

    smoke_cases, case_warnings = select_smoke_cases(df)
    warnings.extend(case_warnings)
    results: list[SmokeCaseResult] = []
    actual_point_count = 0
    old_collection_before: int | None = None
    old_collection_after: int | None = None

    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=args.qdrant_url, timeout=30, check_compatibility=False)
        old_collection_before = get_old_collection_count(client, collection_name=OLD_COLLECTION)
        actual_point_count, collection_vector_size = verify_collection(client, collection_name=args.collection)
        if actual_point_count != EXPECTED_POINT_COUNT:
            errors.append(
                f"Collection point count mismatch: expected {EXPECTED_POINT_COUNT}, got {actual_point_count}."
            )
        if collection_vector_size != EXPECTED_VECTOR_SIZE:
            errors.append(
                f"Collection vector size mismatch: expected {EXPECTED_VECTOR_SIZE}, got {collection_vector_size}."
            )

        if not errors:
            for case in smoke_cases:
                result = run_smoke_case(client, collection_name=args.collection, case=case, top_k=args.top_k)
                results.append(result)
                errors.extend(result.errors)

        old_collection_after = get_old_collection_count(client, collection_name=OLD_COLLECTION)
        if old_collection_before is not None and old_collection_before != 1785:
            errors.append(
                f"Old collection '{OLD_COLLECTION}' has count {old_collection_before}, expected 1785 before smoke test."
            )
        if old_collection_before != old_collection_after:
            errors.append(
                f"Old collection '{OLD_COLLECTION}' changed from {old_collection_before} to {old_collection_after}."
            )
    except Exception as exc:
        errors.append(str(exc))

    status = "FAIL" if errors else "PASS"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        build_report(
            status=status,
            collection_name=args.collection,
            expected_point_count=EXPECTED_POINT_COUNT,
            actual_point_count=actual_point_count,
            vector_size=vector_size,
            old_collection_before=old_collection_before,
            old_collection_after=old_collection_after,
            smoke_cases=smoke_cases,
            results=results,
            warnings=warnings,
            errors=errors,
            output_path=args.out,
        ),
        encoding="utf-8",
    )

    passed_count = sum(1 for result in results if result.passed)
    failed_count = len(results) - passed_count
    print_summary(
        status=status,
        collection_name=args.collection,
        expected_point_count=EXPECTED_POINT_COUNT,
        actual_point_count=actual_point_count,
        vector_size=vector_size,
        passed_count=passed_count,
        failed_count=failed_count,
        old_collection_before=old_collection_before,
        old_collection_after=old_collection_after,
        output_path=args.out,
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
