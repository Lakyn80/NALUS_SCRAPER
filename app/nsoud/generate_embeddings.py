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
    "chunk_id",
    "provider",
    "court",
    "authority_level",
]
OPTIONAL_METADATA_FIELDS = [
    "ecli",
    "decision_date",
    "publication_date",
    "document_type",
    "legal_area",
    "title",
]
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


@dataclass(frozen=True)
class EmbeddingSummary:
    embedding_status: str
    validation_status: str
    total_input_rows_used: int
    output_rows: int
    embedding_dim: int
    duplicate_point_id_count: int
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


def manifest_path_for_output(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}_manifest.json")


def validation_path_for_output(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}_validation.md")


def load_payload_preview(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_existing_embeddings(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
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


def deduplicate_by_point_id(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.drop_duplicates(subset=["point_id"], keep="first").reset_index(drop=True)


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


def generate_new_embeddings(
    rows_to_embed_df: pd.DataFrame,
    *,
    backend: EmbeddingBackend,
    batch_size: int,
) -> tuple[pd.DataFrame, int]:
    if rows_to_embed_df.empty:
        return pd.DataFrame(columns=[]), backend.dim

    total_rows = len(rows_to_embed_df)
    embedded_batches: list[pd.DataFrame] = []
    rows_written = 0

    for batch_number, start in enumerate(range(0, total_rows, batch_size), start=1):
        batch_df = rows_to_embed_df.iloc[start : start + batch_size].copy()
        vectors = backend.embed_documents(batch_df["text"].tolist())
        batch_df["embedding"] = vectors
        batch_df["embedding_dim"] = [len(vector) for vector in vectors]
        embedded_batches.append(batch_df)
        rows_written += len(batch_df)
        print(f"batch number: {batch_number}")
        print(f"rows written: {rows_written}")

    return pd.concat(embedded_batches, ignore_index=True), backend.dim


def validate_embeddings_dataframe(
    df: pd.DataFrame,
) -> tuple[str, list[str], list[str], int, int, int, int, int]:
    failures: list[str] = []
    warnings: list[str] = []

    empty_text_count = int(df["text"].map(lambda value: normalize_text(value).strip() == "").sum()) if not df.empty else 0
    if empty_text_count > 0:
        failures.append("One or more rows have empty text.")

    duplicate_point_id_count = int(df["point_id"].duplicated(keep=False).sum()) if not df.empty else 0
    if duplicate_point_id_count > 0:
        failures.append("Duplicate point_id values detected.")

    duplicate_chunk_id_count = int(df["chunk_id"].duplicated(keep=False).sum()) if not df.empty else 0
    if duplicate_chunk_id_count > 0:
        failures.append("Duplicate chunk_id values detected.")

    missing_embeddings_count = 0
    if not df.empty:
        missing_embeddings_count = int(
            df["embedding"].map(lambda value: not isinstance(value, list) or len(value) == 0).sum()
        )
    if missing_embeddings_count > 0:
        failures.append("One or more rows are missing embeddings.")

    embedding_dims = sorted({int(value) for value in df["embedding_dim"].dropna().tolist()}) if not df.empty else []
    if not embedding_dims:
        failures.append("No embedding dimensions were produced.")
        embedding_dim = 0
    elif len(embedding_dims) > 1:
        failures.append("Embedding dimensions are inconsistent across rows.")
        embedding_dim = embedding_dims[0]
    else:
        embedding_dim = embedding_dims[0]

    optional_missing_total = 0
    for field_name in OPTIONAL_METADATA_FIELDS:
        optional_missing_total += int(df[field_name].map(lambda value: normalize_text(value).strip() == "").sum()) if not df.empty else 0
    if optional_missing_total > 0:
        warnings.append("Some optional metadata fields are missing.")

    validation_status = "FAIL" if failures else "WARN" if warnings else "PASS"
    return (
        validation_status,
        failures,
        warnings,
        duplicate_point_id_count,
        duplicate_chunk_id_count,
        empty_text_count,
        missing_embeddings_count,
        embedding_dim,
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
    warnings: list[str],
    duplicate_point_id_count: int,
    empty_text_count: int,
    missing_embeddings_count: int,
    embedding_dim: int,
) -> str:
    status_items = failures + warnings if failures or warnings else ["Embedding validation passed."]
    legal_area_missing_count = int(df["legal_area"].map(lambda value: normalize_text(value).strip() == "").sum()) if not df.empty else 0
    lines = [
        "# NSoud Embeddings Validation",
        "",
        f"- Input: `{input_path}`",
        f"- Output: `{output_path}`",
        f"- Validation status: **{validation_status}**",
        f"- Total rows: **{len(df)}**",
        f"- Embedding dim: **{embedding_dim}**",
        f"- Missing embeddings count: **{missing_embeddings_count}**",
        f"- Duplicate point_id count: **{duplicate_point_id_count}**",
        f"- Empty text count: **{empty_text_count}**",
        f"- Model name: `{model_name}`",
        f"- Device: `{device}`",
        f"- Backend: `{backend_name}`",
        f"- Legal area missing count: **{legal_area_missing_count}**",
        "",
        "## Status",
    ]
    lines.extend(f"- {item}" for item in status_items)
    lines.append("")
    lines.extend(render_distribution_table("Source Distribution", distribution_counts(df, "provider")))
    lines.extend(render_distribution_table("Document Type Distribution", distribution_counts(df, "document_type")))
    lines.extend(render_distribution_table("Legal Area Distribution", distribution_counts(df, "legal_area")))
    lines.extend(
        [
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
    embedding_dim: int,
    total_input_rows: int,
    total_output_rows: int,
    skipped_existing_rows: int,
    newly_embedded_rows: int,
    duplicate_point_id_count: int,
    empty_text_count: int,
) -> None:
    payload = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "model_name": model_name,
        "device": device,
        "embedding_dim": embedding_dim,
        "total_input_rows": total_input_rows,
        "total_output_rows": total_output_rows,
        "skipped_existing_rows": skipped_existing_rows,
        "newly_embedded_rows": newly_embedded_rows,
        "duplicate_point_id_count": duplicate_point_id_count,
        "empty_text_count": empty_text_count,
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
        existing_df = load_existing_embeddings(args.out)
        existing_point_ids = set(existing_df["point_id"].tolist()) if not existing_df.empty and "point_id" in existing_df.columns else set()
        rows_to_embed_df = payload_df.loc[~payload_df["point_id"].isin(existing_point_ids)].copy()

        print(f"total input rows: {len(payload_df)}")
        print(f"existing embedded rows: {len(existing_point_ids)}")
        print(f"rows to embed: {len(rows_to_embed_df)}")

        newly_embedded_df, embedding_dim = generate_new_embeddings(
            rows_to_embed_df,
            backend=backend,
            batch_size=args.batch_size,
        )

        if existing_df.empty:
            final_df = newly_embedded_df.copy()
        else:
            final_df = pd.concat([existing_df, newly_embedded_df], ignore_index=True)
        final_df = deduplicate_by_point_id(final_df)
        final_df = final_df.sort_values(["chunk_id"]).reset_index(drop=True) if not final_df.empty else final_df

        write_parquet(final_df, args.out)

        (
            validation_status,
            failures,
            warnings,
            duplicate_point_id_count,
            duplicate_chunk_id_count,
            empty_text_count,
            missing_embeddings_count,
            final_embedding_dim,
        ) = validate_embeddings_dataframe(final_df)

        write_manifest(
            manifest_path,
            input_path=args.input,
            output_path=args.out,
            model_name=args.model_name,
            device=backend.device,
            embedding_dim=final_embedding_dim or embedding_dim,
            total_input_rows=len(payload_df),
            total_output_rows=len(final_df),
            skipped_existing_rows=len(existing_point_ids.intersection(set(payload_df["point_id"].tolist()))),
            newly_embedded_rows=len(newly_embedded_df),
            duplicate_point_id_count=duplicate_point_id_count,
            empty_text_count=empty_text_count or input_empty_text_count,
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
            warnings=warnings,
            duplicate_point_id_count=duplicate_point_id_count,
            empty_text_count=empty_text_count or input_empty_text_count,
            missing_embeddings_count=missing_embeddings_count,
            embedding_dim=final_embedding_dim or embedding_dim,
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
        total_input_rows_used=len(payload_df),
        output_rows=len(final_df),
        embedding_dim=final_embedding_dim or embedding_dim,
        duplicate_point_id_count=duplicate_point_id_count,
        output_path=args.out,
        manifest_path=manifest_path,
        validation_path=validation_path,
    )
    print(f"test embedding status: {summary.embedding_status}")
    print(f"total input rows used: {summary.total_input_rows_used}")
    print(f"output rows: {summary.output_rows}")
    print(f"embedding_dim: {summary.embedding_dim}")
    print(f"duplicate point_id count: {summary.duplicate_point_id_count}")
    print(f"validation status: {summary.validation_status}")
    return 1 if summary.validation_status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
