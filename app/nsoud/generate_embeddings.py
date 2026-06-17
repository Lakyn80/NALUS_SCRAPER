from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
from dataclasses import dataclass
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

from app.rag.retrieval.embedder import SentenceTransformersEmbedder


REQUIRED_COLUMNS = [
    "point_id",
    "text",
    "source",
    "provider",
    "court",
    "authority_level",
    "case_number",
    "url",
    "source_attribution",
    "content_hash",
    "document_id",
    "chunk_id",
    "chunk_index",
    "total_chunks_in_document",
    "section_id",
    "section_type",
    "section_index",
    "chunk_index_in_section",
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
NULLABLE_LINK_FIELDS = {
    "previous_chunk_id",
    "next_chunk_id",
    "previous_section_chunk_id",
    "next_section_chunk_id",
}
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_COLUMNS = ["embedding", "embedding_dim"]


@dataclass(frozen=True)
class EmbeddingSummary:
    embedding_status: str
    validation_status: str
    input_rows: int
    output_rows: int
    embedding_dim: int
    missing_embeddings_count: int
    duplicate_point_id_count: int
    duplicate_chunk_id_count: int
    empty_text_count: int
    metadata_preservation_status: str
    output_path: Path
    manifest_path: Path
    validation_path: Path


class EmbeddingBackend:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError

    @property
    def device(self) -> str:
        raise NotImplementedError

    @property
    def backend_name(self) -> str:
        raise NotImplementedError

    def close(self) -> None:
        return None


class LocalEmbeddingBackend(EmbeddingBackend):
    def __init__(self, embedder: SentenceTransformersEmbedder, device: str) -> None:
        self._embedder = embedder
        self._device = device

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_documents(texts)

    @property
    def dim(self) -> int:
        return self._embedder.dim

    @property
    def device(self) -> str:
        return self._device

    @property
    def backend_name(self) -> str:
        return "local"


DOCKER_WORKER_CODE = r"""
import contextlib
import io
import json
import sys

from sentence_transformers import SentenceTransformer

model_name = sys.argv[1]
device = sys.argv[2]
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
    model = SentenceTransformer(model_name, device=device, local_files_only=True)

dim_getter = getattr(model, "get_sentence_embedding_dimension", None)
embedding_dim = int(dim_getter()) if callable(dim_getter) and dim_getter() else None

for raw_line in sys.stdin:
    line = raw_line.strip()
    if not line:
        continue
    request = json.loads(line)
    if request.get("command") == "close":
        print(json.dumps({"ok": True}))
        sys.stdout.flush()
        break

    texts = request["texts"]
    encoded = model.encode(texts, batch_size=request["batch_size"], normalize_embeddings=True, show_progress_bar=False)
    vectors = []
    for vector in encoded:
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        vectors.append([float(value) for value in vector])

    response = {
        "ok": True,
        "embeddings": vectors,
        "embedding_dim": embedding_dim if embedding_dim is not None else (len(vectors[0]) if vectors else 0),
        "device": device,
    }
    print(json.dumps(response, ensure_ascii=False))
    sys.stdout.flush()
"""


class DockerEmbeddingBackend(EmbeddingBackend):
    def __init__(self, *, model_name: str, device: str, batch_size: int) -> None:
        self._batch_size = batch_size
        self._process = subprocess.Popen(
            [
                "docker",
                "compose",
                "exec",
                "-T",
                "api",
                "python",
                "-u",
                "-c",
                DOCKER_WORKER_CODE,
                model_name,
                device,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        self._embedding_dim: int | None = None
        self._device = device

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("Docker embedding worker is not available.")

        request = {"texts": texts, "batch_size": self._batch_size}
        self._process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self._process.stdin.flush()

        response_line = self._process.stdout.readline()
        if not response_line:
            stderr = ""
            if self._process.stderr is not None:
                stderr = self._process.stderr.read().strip()
            raise RuntimeError(f"Docker embedding worker returned no response. {stderr}".strip())

        response = json.loads(response_line)
        if not response.get("ok"):
            raise RuntimeError(f"Docker embedding worker failed: {response}")

        self._embedding_dim = int(response.get("embedding_dim", 0))
        self._device = str(response.get("device", self._device))
        return response["embeddings"]

    @property
    def dim(self) -> int:
        return self._embedding_dim or 0

    @property
    def device(self) -> str:
        return self._device

    @property
    def backend_name(self) -> str:
        return "docker-api"

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        try:
            if self._process.stdin is not None:
                self._process.stdin.write(json.dumps({"command": "close"}) + "\n")
                self._process.stdin.flush()
        except Exception:
            pass
        finally:
            self._process.terminate()
            self._process.wait(timeout=10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate local embeddings for NSoud Qdrant payload preview.")
    parser.add_argument("--input", type=Path, required=True, help="Input NSoud payload preview parquet path.")
    parser.add_argument("--out", type=Path, required=True, help="Output embeddings parquet path.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Sentence-transformers model name.")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of rows to embed from input.")
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


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def manifest_path_for_output(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}_manifest.json")


def validation_path_for_output(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}_validation.md")


def load_payload_preview(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        if requested_device == "cuda":
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("CUDA requested but torch is not available.") from exc
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but no CUDA device is available.")
        return requested_device

    try:
        import torch
    except ImportError:
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"


def build_embedder(model_name: str, batch_size: int, device: str) -> SentenceTransformersEmbedder:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for local embedding generation. "
            "Install it in the api container or local environment."
        ) from exc

    try:
        model = SentenceTransformer(model_name, device=device, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            "Embedding model is not available in local cache. "
            "This script is local-only and will not download models. "
            f"Cache or bake the model first: {model_name}"
        ) from exc
    return SentenceTransformersEmbedder(model_name=model_name, model=model, batch_size=batch_size)


def build_embedding_backend(model_name: str, batch_size: int, requested_device: str) -> EmbeddingBackend:
    errors: list[str] = []

    try:
        resolved_device = resolve_device(requested_device)
        embedder = build_embedder(model_name, batch_size, resolved_device)
        return LocalEmbeddingBackend(embedder, resolved_device)
    except Exception as exc:
        errors.append(f"local backend unavailable: {exc}")

    docker_device = "cpu" if requested_device == "auto" else requested_device
    try:
        return DockerEmbeddingBackend(model_name=model_name, device=docker_device, batch_size=batch_size)
    except Exception as exc:
        errors.append(f"docker backend unavailable: {exc}")

    raise RuntimeError(" ; ".join(errors))


def validate_input_dataframe(df: pd.DataFrame) -> tuple[list[str], int]:
    failures: list[str] = []
    for column_name in REQUIRED_COLUMNS:
        if column_name not in df.columns:
            failures.append(f"Missing required input column `{column_name}`.")

    if failures:
        return failures, 0

    empty_text_count = int(df["text"].map(lambda value: normalize_text(value).strip() == "").sum())
    if empty_text_count > 0:
        failures.append("Input payload contains empty text rows.")

    return failures, empty_text_count


def write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", index=False)


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


def generate_embeddings_dataframe(
    payload_df: pd.DataFrame,
    *,
    backend: EmbeddingBackend,
    batch_size: int,
) -> tuple[pd.DataFrame, int]:
    if payload_df.empty:
        empty_df = payload_df.copy()
        empty_df["embedding"] = []
        empty_df["embedding_dim"] = []
        return empty_df, backend.dim

    total_rows = len(payload_df)
    embedded_batches: list[pd.DataFrame] = []
    rows_written = 0

    for batch_number, start in enumerate(range(0, total_rows, batch_size), start=1):
        batch_df = payload_df.iloc[start : start + batch_size].copy()
        vectors = backend.embed_documents(batch_df["text"].tolist())
        batch_df["embedding"] = vectors
        batch_df["embedding_dim"] = [len(vector) for vector in vectors]
        embedded_batches.append(batch_df)
        rows_written += len(batch_df)
        print(f"batch number: {batch_number}")
        print(f"rows written: {rows_written}")

    return pd.concat(embedded_batches, ignore_index=True), backend.dim


def validate_document_sequences(df: pd.DataFrame) -> tuple[int, int, int, int, list[str]]:
    sequence_passed = 0
    sequence_failed = 0
    neighbor_passed = 0
    neighbor_failed = 0
    failures: list[str] = []

    for document_id, group in df.groupby("document_id", sort=False):
        sorted_group = group.sort_values("chunk_index").reset_index(drop=True)
        actual_sequence = sorted_group["chunk_index"].astype(int).tolist()
        expected_sequence = list(range(len(sorted_group)))
        total_chunks_values = sorted_group["total_chunks_in_document"].astype(int).unique().tolist()
        if actual_sequence == expected_sequence and total_chunks_values == [len(sorted_group)]:
            sequence_passed += 1
        else:
            sequence_failed += 1
            failures.append(f"Document `{document_id}` has invalid chunk_index or total_chunks_in_document metadata.")

        links_ok = True
        for index, row in sorted_group.iterrows():
            expected_previous = normalize_text(sorted_group.iloc[index - 1]["chunk_id"]) if index > 0 else ""
            expected_next = normalize_text(sorted_group.iloc[index + 1]["chunk_id"]) if index + 1 < len(sorted_group) else ""
            actual_previous = normalize_text(row["previous_chunk_id"])
            actual_next = normalize_text(row["next_chunk_id"])
            if actual_previous != expected_previous or actual_next != expected_next:
                links_ok = False
                failures.append(f"Document `{document_id}` has invalid previous_chunk_id/next_chunk_id links.")
                break

        if links_ok:
            neighbor_passed += 1
        else:
            neighbor_failed += 1

    return sequence_passed, sequence_failed, neighbor_passed, neighbor_failed, failures


def validate_section_sequences(df: pd.DataFrame) -> tuple[int, int, int, int, list[str]]:
    sequence_passed = 0
    sequence_failed = 0
    neighbor_passed = 0
    neighbor_failed = 0
    failures: list[str] = []

    for (document_id, section_id), group in df.groupby(["document_id", "section_id"], sort=False):
        sorted_group = group.sort_values("chunk_index_in_section").reset_index(drop=True)
        actual_sequence = sorted_group["chunk_index_in_section"].astype(int).tolist()
        expected_sequence = list(range(len(sorted_group)))
        total_chunks_values = sorted_group["total_chunks_in_section"].astype(int).unique().tolist()
        if actual_sequence == expected_sequence and total_chunks_values == [len(sorted_group)]:
            sequence_passed += 1
        else:
            sequence_failed += 1
            failures.append(
                f"Section `{section_id}` in document `{document_id}` has invalid chunk_index_in_section or total_chunks_in_section metadata."
            )

        links_ok = True
        for index, row in sorted_group.iterrows():
            expected_previous = normalize_text(sorted_group.iloc[index - 1]["chunk_id"]) if index > 0 else ""
            expected_next = normalize_text(sorted_group.iloc[index + 1]["chunk_id"]) if index + 1 < len(sorted_group) else ""
            actual_previous = normalize_text(row["previous_section_chunk_id"])
            actual_next = normalize_text(row["next_section_chunk_id"])
            if actual_previous != expected_previous or actual_next != expected_next:
                links_ok = False
                failures.append(
                    f"Section `{section_id}` in document `{document_id}` has invalid previous_section_chunk_id/next_section_chunk_id links."
                )
                break

        if links_ok:
            neighbor_passed += 1
        else:
            neighbor_failed += 1

    return sequence_passed, sequence_failed, neighbor_passed, neighbor_failed, failures


def validate_metadata_preservation(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
) -> tuple[str, list[str]]:
    failures: list[str] = []
    input_by_point_id = input_df.set_index("point_id", drop=False)
    output_by_point_id = output_df.set_index("point_id", drop=False)

    if len(input_by_point_id) != len(output_by_point_id):
        failures.append("Input and output row counts differ during metadata preservation validation.")
        return "FAIL", failures

    if set(input_by_point_id.index.tolist()) != set(output_by_point_id.index.tolist()):
        failures.append("Input and output point_id sets differ.")
        return "FAIL", failures

    fields_to_compare = [
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

    for point_id in input_by_point_id.index.tolist():
        input_row = input_by_point_id.loc[point_id]
        output_row = output_by_point_id.loc[point_id]
        for field_name in fields_to_compare:
            input_value = input_row[field_name]
            output_value = output_row[field_name]
            if field_name in NULLABLE_LINK_FIELDS:
                if normalize_text(input_value) != normalize_text(output_value):
                    failures.append(f"Metadata mismatch for point_id `{point_id}` field `{field_name}`.")
                    return "FAIL", failures
                continue
            if str(input_value) != str(output_value):
                failures.append(f"Metadata mismatch for point_id `{point_id}` field `{field_name}`.")
                return "FAIL", failures

    return "PASS", failures


def validate_embeddings_dataframe(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
) -> tuple[str, list[str], int, int, int, int, int, int, int, int, int, int, str]:
    failures: list[str] = []

    input_rows = len(input_df)
    output_rows = len(output_df)
    if input_rows != 1862:
        failures.append(f"Expected 1862 input rows, found {input_rows}.")
    if output_rows != 1862:
        failures.append(f"Expected 1862 output rows, found {output_rows}.")

    missing_embeddings_count = int(
        output_df["embedding"].map(lambda value: not isinstance(value, list) or len(value) == 0).sum()
    ) if not output_df.empty else 0
    if missing_embeddings_count > 0:
        failures.append("One or more rows are missing embeddings.")

    duplicate_point_id_count = int(output_df["point_id"].duplicated(keep=False).sum()) if not output_df.empty else 0
    if duplicate_point_id_count > 0:
        failures.append("Duplicate point_id values detected.")

    duplicate_chunk_id_count = int(output_df["chunk_id"].duplicated(keep=False).sum()) if not output_df.empty else 0
    if duplicate_chunk_id_count > 0:
        failures.append("Duplicate chunk_id values detected.")

    empty_text_count = int(output_df["text"].map(lambda value: normalize_text(value).strip() == "").sum()) if not output_df.empty else 0
    if empty_text_count > 0:
        failures.append("One or more rows have empty text.")

    missing_required_metadata_rows = pd.Series([False] * len(output_df))
    for field_name in REQUIRED_COLUMNS:
        if field_name not in output_df.columns:
            failures.append(f"Output is missing required column `{field_name}`.")
            continue
        field_missing = output_df[field_name].map(is_missing)
        if field_name in NULLABLE_LINK_FIELDS:
            field_missing = pd.Series([False] * len(output_df))
        missing_required_metadata_rows = missing_required_metadata_rows | field_missing
    missing_required_metadata_count = int(missing_required_metadata_rows.sum()) if not output_df.empty else 0
    if missing_required_metadata_count > 0:
        failures.append("One or more rows are missing required metadata.")

    invalid_chunking_strategy_count = int(
        (output_df["chunking_strategy"].map(normalize_text) != "document_section_aware").sum()
    ) if not output_df.empty else 0
    if invalid_chunking_strategy_count > 0:
        failures.append("One or more rows have invalid chunking_strategy values.")

    embedding_dims = sorted({int(value) for value in output_df["embedding_dim"].dropna().tolist()}) if not output_df.empty else []
    if not embedding_dims:
        failures.append("No embedding dimensions were produced.")
        embedding_dim = 0
    elif len(embedding_dims) > 1:
        failures.append("Embedding dimensions are inconsistent across rows.")
        embedding_dim = embedding_dims[0]
    else:
        embedding_dim = embedding_dims[0]

    document_sequence_passed, document_sequence_failed, document_neighbor_passed, document_neighbor_failed, document_failures = (
        validate_document_sequences(output_df)
    )
    section_sequence_passed, section_sequence_failed, section_neighbor_passed, section_neighbor_failed, section_failures = (
        validate_section_sequences(output_df)
    )
    failures.extend(document_failures)
    failures.extend(section_failures)

    metadata_preservation_status, metadata_failures = validate_metadata_preservation(input_df, output_df)
    failures.extend(metadata_failures)

    validation_status = "FAIL" if failures else "PASS"
    return (
        validation_status,
        sorted(set(failures)),
        output_rows,
        embedding_dim,
        missing_embeddings_count,
        duplicate_point_id_count,
        duplicate_chunk_id_count,
        empty_text_count,
        missing_required_metadata_count,
        document_sequence_passed,
        document_sequence_failed,
        section_sequence_passed,
        section_sequence_failed,
        document_neighbor_passed,
        document_neighbor_failed,
        section_neighbor_passed,
        section_neighbor_failed,
        metadata_preservation_status,
    )


def build_validation_report(
    df: pd.DataFrame,
    *,
    input_path: Path,
    output_path: Path,
    model_name: str,
    device: str,
    backend_name: str,
    validation_status: str,
    failures: list[str],
    input_rows: int,
    output_rows: int,
    embedding_dim: int,
    missing_embeddings_count: int,
    duplicate_point_id_count: int,
    duplicate_chunk_id_count: int,
    empty_text_count: int,
    missing_required_metadata_count: int,
    document_sequence_validation_passed: int,
    document_sequence_validation_failed: int,
    section_sequence_validation_passed: int,
    section_sequence_validation_failed: int,
    document_neighbor_validation_passed: int,
    document_neighbor_validation_failed: int,
    section_neighbor_validation_passed: int,
    section_neighbor_validation_failed: int,
    metadata_preservation_status: str,
) -> str:
    status_items = failures if failures else ["Embedding validation passed."]
    lines = [
        "# NSoud Embeddings Validation",
        "",
        f"- Input: `{input_path}`",
        f"- Output: `{output_path}`",
        f"- Validation status: **{validation_status}**",
        f"- Input rows: **{input_rows}**",
        f"- Output rows: **{output_rows}**",
        f"- Embedding dim: **{embedding_dim}**",
        f"- Missing embeddings count: **{missing_embeddings_count}**",
        f"- Duplicate point_id count: **{duplicate_point_id_count}**",
        f"- Duplicate chunk_id count: **{duplicate_chunk_id_count}**",
        f"- Empty text count: **{empty_text_count}**",
        f"- Missing required metadata count: **{missing_required_metadata_count}**",
        f"- Metadata preservation status: **{metadata_preservation_status}**",
        f"- Document sequence validation passed/failed: **{document_sequence_validation_passed}/{document_sequence_validation_failed}**",
        f"- Section sequence validation passed/failed: **{section_sequence_validation_passed}/{section_sequence_validation_failed}**",
        f"- Document neighbor validation passed/failed: **{document_neighbor_validation_passed}/{document_neighbor_validation_failed}**",
        f"- Section neighbor validation passed/failed: **{section_neighbor_validation_passed}/{section_neighbor_validation_failed}**",
        f"- Model name: `{model_name}`",
        f"- Device: `{device}`",
        f"- Backend: `{backend_name}`",
        "",
        "## Status",
    ]
    lines.extend(f"- {item}" for item in status_items)
    lines.append("")
    lines.extend(render_distribution_table("Provider Distribution", distribution_counts(df, "provider")))
    lines.extend(render_distribution_table("Document Type Distribution", distribution_counts(df, "document_type")))
    lines.extend(render_distribution_table("Legal Area Distribution", distribution_counts(df, "legal_area")))
    lines.extend(render_distribution_table("Section Type Distribution", distribution_counts(df, "section_type")))
    lines.extend(
        [
            "## Text Lengths",
            "",
            f"- min: {min(df['chunk_text_length'].astype(int).tolist()) if not df.empty else 0}",
            f"- max: {max(df['chunk_text_length'].astype(int).tolist()) if not df.empty else 0}",
            f"- avg: {mean(df['chunk_text_length'].astype(int).tolist()):.2f}" if not df.empty else "- avg: 0.00",
            "",
            "## Recommended Docker Command",
            "",
            f"`docker compose exec api python app/nsoud/generate_embeddings.py --input {input_path.as_posix()} --out {output_path.as_posix()} --batch-size 32 --device auto`",
            "",
        ]
    )
    return "\n".join(lines)


def write_manifest(
    manifest_path: Path,
    *,
    input_path: Path,
    output_path: Path,
    model_name: str,
    device: str,
    backend_name: str,
    embedding_dim: int,
    total_input_rows: int,
    total_output_rows: int,
    missing_embeddings_count: int,
    duplicate_point_id_count: int,
    duplicate_chunk_id_count: int,
    empty_text_count: int,
    missing_required_metadata_count: int,
    metadata_preservation_status: str,
) -> None:
    payload = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "model_name": model_name,
        "device": device,
        "backend_name": backend_name,
        "embedding_dim": embedding_dim,
        "total_input_rows": total_input_rows,
        "total_output_rows": total_output_rows,
        "missing_embeddings_count": missing_embeddings_count,
        "duplicate_point_id_count": duplicate_point_id_count,
        "duplicate_chunk_id_count": duplicate_chunk_id_count,
        "empty_text_count": empty_text_count,
        "missing_required_metadata_count": missing_required_metadata_count,
        "metadata_preservation_status": metadata_preservation_status,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()

    if pyarrow is None:
        print("embedding status: FAIL")
        print("error: pyarrow is required for Parquet output.")
        print("install command: pip install pyarrow")
        return 1

    manifest_path = manifest_path_for_output(args.out)
    validation_path = validation_path_for_output(args.out)

    try:
        payload_df = load_payload_preview(args.input)
    except Exception as exc:
        print("embedding status: FAIL")
        print(f"error: {exc}")
        return 1

    input_failures, input_empty_text_count = validate_input_dataframe(payload_df)
    if input_failures:
        print("embedding status: FAIL")
        for failure in input_failures:
            print(f"error: {failure}")
        return 1

    if args.limit is not None:
        payload_df = payload_df.head(args.limit).copy()

    try:
        backend = build_embedding_backend(args.model_name, args.batch_size, args.device)
    except Exception as exc:
        print("embedding status: FAIL")
        print(f"error: {exc}")
        return 1

    try:
        print(f"input rows: {len(payload_df)}")
        embedded_df, embedding_dim = generate_embeddings_dataframe(
            payload_df,
            backend=backend,
            batch_size=args.batch_size,
        )
        final_df = embedded_df.sort_values(["chunk_id"]).reset_index(drop=True) if not embedded_df.empty else embedded_df
        write_parquet(final_df, args.out)

        (
            validation_status,
            failures,
            output_rows,
            final_embedding_dim,
            missing_embeddings_count,
            duplicate_point_id_count,
            duplicate_chunk_id_count,
            empty_text_count,
            missing_required_metadata_count,
            document_sequence_passed,
            document_sequence_failed,
            section_sequence_passed,
            section_sequence_failed,
            document_neighbor_passed,
            document_neighbor_failed,
            section_neighbor_passed,
            section_neighbor_failed,
            metadata_preservation_status,
        ) = validate_embeddings_dataframe(payload_df, final_df)

        write_manifest(
            manifest_path,
            input_path=args.input,
            output_path=args.out,
            model_name=args.model_name,
            device=backend.device,
            backend_name=backend.backend_name,
            embedding_dim=final_embedding_dim or embedding_dim,
            total_input_rows=len(payload_df),
            total_output_rows=output_rows,
            missing_embeddings_count=missing_embeddings_count,
            duplicate_point_id_count=duplicate_point_id_count,
            duplicate_chunk_id_count=duplicate_chunk_id_count,
            empty_text_count=empty_text_count or input_empty_text_count,
            missing_required_metadata_count=missing_required_metadata_count,
            metadata_preservation_status=metadata_preservation_status,
        )

        report = build_validation_report(
            final_df,
            input_path=args.input,
            output_path=args.out,
            model_name=args.model_name,
            device=backend.device,
            backend_name=backend.backend_name,
            validation_status=validation_status,
            failures=failures,
            input_rows=len(payload_df),
            output_rows=output_rows,
            embedding_dim=final_embedding_dim or embedding_dim,
            missing_embeddings_count=missing_embeddings_count,
            duplicate_point_id_count=duplicate_point_id_count,
            duplicate_chunk_id_count=duplicate_chunk_id_count,
            empty_text_count=empty_text_count or input_empty_text_count,
            missing_required_metadata_count=missing_required_metadata_count,
            document_sequence_validation_passed=document_sequence_passed,
            document_sequence_validation_failed=document_sequence_failed,
            section_sequence_validation_passed=section_sequence_passed,
            section_sequence_validation_failed=section_sequence_failed,
            document_neighbor_validation_passed=document_neighbor_passed,
            document_neighbor_validation_failed=document_neighbor_failed,
            section_neighbor_validation_passed=section_neighbor_passed,
            section_neighbor_validation_failed=section_neighbor_failed,
            metadata_preservation_status=metadata_preservation_status,
        )
        validation_path.write_text(report, encoding="utf-8")
    except Exception as exc:
        print("embedding status: FAIL")
        print(f"error: {exc}")
        return 1
    finally:
        backend.close()

    summary = EmbeddingSummary(
        embedding_status="PASS",
        validation_status=validation_status,
        input_rows=len(payload_df),
        output_rows=output_rows,
        embedding_dim=final_embedding_dim or embedding_dim,
        missing_embeddings_count=missing_embeddings_count,
        duplicate_point_id_count=duplicate_point_id_count,
        duplicate_chunk_id_count=duplicate_chunk_id_count,
        empty_text_count=empty_text_count or input_empty_text_count,
        metadata_preservation_status=metadata_preservation_status,
        output_path=args.out,
        manifest_path=manifest_path,
        validation_path=validation_path,
    )
    print(f"embedding status: {summary.embedding_status}")
    print(f"validation status: {summary.validation_status}")
    print(f"input rows: {summary.input_rows}")
    print(f"output rows: {summary.output_rows}")
    print(f"embedding_dim: {summary.embedding_dim}")
    print(f"missing embeddings count: {summary.missing_embeddings_count}")
    print(f"duplicate point_id count: {summary.duplicate_point_id_count}")
    print(f"duplicate chunk_id count: {summary.duplicate_chunk_id_count}")
    print(f"empty text count: {summary.empty_text_count}")
    print(f"metadata preservation status: {summary.metadata_preservation_status}")
    print(f"output parquet path: {summary.output_path}")
    print(f"manifest path: {summary.manifest_path}")
    print(f"validation report path: {summary.validation_path}")
    print("changed files: app/nsoud/generate_embeddings.py")
    return 1 if summary.validation_status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
