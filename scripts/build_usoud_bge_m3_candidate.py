"""Guarded builder for parallel US/NALUS BGE-M3 candidate collections.

Supports smoke, pilot, and full modes. Full mode writes only to an explicit
non-production candidate collection. Production aliases and stable collections
must never be modified by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BUILDER_VERSION = "usoud-bge-m3-guarded-v5"
MVP_RECENT_3H_RECORD_LIMIT = 600
BGE_M3_MODEL_NAME = "BAAI/bge-m3"
BGE_M3_DIMENSION = 1024
PRODUCTION_RETRIEVAL_PROFILE = "nalus_bge_m3_dense_bm25_rrf_v1"
SOURCE_ID = 1
DEFAULT_TOP_K = 5
DEFAULT_CHUNK_SIZE = 1400
DEFAULT_CHUNK_OVERLAP_WORDS = 35
DEFAULT_BM25_INDEX_ID = PRODUCTION_RETRIEVAL_PROFILE
FULL_LIMIT_ALL = 0
FULL_EXECUTE_RECORD_BATCH_SIZE = 50
FULL_VALIDATION_BM25_CHUNK_LIMIT = 10_000
FULL_DRY_RUN_PROGRESS_INTERVAL = 5_000
EXECUTE_CHECKPOINT_FILENAME = "execute_checkpoint.json"
REPORT_PATH = PROJECT_ROOT / "artifacts/nalus_update/usoud_bge_m3_stage1_smoke_report.md"
STAGE2_PILOT_REPORT_PATH = PROJECT_ROOT / "artifacts/nalus_update/usoud_bge_m3_stage2_pilot_report.md"
STAGE3_FULL_REPORT_PATH = PROJECT_ROOT / "artifacts/nalus_update/usoud_bge_m3_stage3_full_report.md"
DEFAULT_FULL_MANIFEST = PROJECT_ROOT / "batches/manifest.json"
PRODUCTION_COLLECTION_DENYLIST = {
    "nalus",
    "nalus_live",
    "nalus_stable_20260326",
}
ALLOWED_STAGE1_NAME_MARKERS = ("smoke", "tmp", "pilot")
ALLOWED_PILOT_NAME_MARKERS = ("pilot", "tmp")
ALLOWED_FULL_NAME_MARKERS = ("full", "tmp", "mvp")
RECREATE_SMOKE_ALLOWED_MARKERS = ("smoke", "tmp")
RECREATE_PILOT_ALLOWED_MARKERS = ("pilot", "tmp")
RECREATE_FULL_ALLOWED_MARKERS = ("full", "tmp", "mvp")
DEFAULT_STAGE2_PILOT_LIMIT = 500
MAX_STAGE2_PILOT_LIMIT = 1000
SMOKE_QUERIES = (
    "právo na spravedlivý proces",
    "opomenuté důkazy",
    "odůvodnění rozhodnutí",
    "porušení základních práv",
)
PILOT_QUERIES = (
    "právo na spravedlivý proces",
    "opomenuté důkazy",
    "odůvodnění rozhodnutí",
    "porušení základních práv",
    "extrémní nesoulad",
    "rovnost účastníků řízení",
    "právo na zákonného soudce",
    "odmítnutí dovolání",
    "náklady řízení",
    "vlastnické právo",
    "svoboda projevu",
    "ochrana soukromí",
    "vazba",
    "trestní řízení",
    "civilní řízení",
    "ústavní stížnost",
    "proporcionalita",
    "retroaktivita",
    "legitimní očekávání",
    "právo na účinný prostředek nápravy",
)


class SafetyError(ValueError):
    """Raised when a safety guard refuses the requested operation."""


@dataclass(frozen=True)
class SourceRecord:
    identity: str
    source_document_id: str
    case_reference: str | None
    ecli: str | None
    decision_date: str | None
    detail_url: str | None
    text_url: str | None
    full_text: str
    origin_file: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class DecisionDateFilter:
    date_from: date | None
    date_to: date | None
    years_back: int | None
    ingest_slice: str

    def as_summary(self) -> dict[str, Any]:
        return {
            "ingest_slice": self.ingest_slice,
            "years_back": self.years_back,
            "decision_date_from": self.date_from.isoformat() if self.date_from else None,
            "decision_date_to": self.date_to.isoformat() if self.date_to else None,
        }


@dataclass(frozen=True)
class SmokeChunk:
    seq_id: int
    point_id: str
    text: str
    payload: dict[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guarded smoke/pilot/full builder for a US/NALUS BGE-M3 candidate collection."
    )
    parser.add_argument("--mode", choices=["smoke", "pilot", "full"], required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--collection-name", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-batch", type=Path)
    source.add_argument("--source-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument("--recreate-smoke-collection", action="store_true")
    parser.add_argument("--recreate-pilot-collection", action="store_true")
    parser.add_argument("--recreate-full-collection", action="store_true")
    parser.add_argument("--resume-full-collection", action="store_true")
    parser.add_argument("--no-alias-update", action="store_true", default=True)
    parser.add_argument("--top-k-smoke-test", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://qdrant:6333"))
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument(
        "--full-record-batch-size",
        type=int,
        default=FULL_EXECUTE_RECORD_BATCH_SIZE,
        help="Records per chunk/embed/upsert batch in full execute mode.",
    )
    parser.add_argument(
        "--years-back",
        type=int,
        default=None,
        help="Full mode only: rolling decision_date window from today minus N years through today.",
    )
    parser.add_argument(
        "--decision-date-from",
        type=str,
        default=None,
        help="Full mode only: inclusive lower bound (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--decision-date-to",
        type=str,
        default=None,
        help="Full mode only: inclusive upper bound (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--ingest-slice",
        type=str,
        default="",
        help="Full mode label for checkpointing and later incremental slices (e.g. mvp_5y).",
    )
    parser.add_argument(
        "--append-full-slice",
        action="store_true",
        help="Full execute only: append a new date slice to an existing candidate collection.",
    )
    parser.add_argument(
        "--newest-first",
        action="store_true",
        help=(
            "Full mode only: after date filtering, process decisions newest-to-oldest "
            f"(requires positive --limit, e.g. {MVP_RECENT_3H_RECORD_LIMIT} for ~3h CPU MVP)."
        ),
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.mode not in {"smoke", "pilot", "full"}:
        raise SafetyError("Only --mode smoke, pilot, or full is allowed.")

    validate_collection_name(args.collection_name, execute=args.execute, mode=args.mode)
    validate_limit(args.mode, args.limit)
    validate_top_k(args.top_k_smoke_test)

    if args.mode == "full" and args.source_batch is not None:
        raise SafetyError("Full mode requires --source-manifest for the complete corpus, not --source-batch.")

    if not args.no_alias_update:
        raise SafetyError("Alias updates are refused. --no-alias-update must remain enabled.")

    if args.mode == "smoke" and args.recreate_pilot_collection:
        raise SafetyError("--recreate-pilot-collection is allowed only in pilot mode.")

    if args.mode == "smoke" and args.recreate_full_collection:
        raise SafetyError("--recreate-full-collection is allowed only in full mode.")

    if args.mode == "pilot" and args.recreate_smoke_collection:
        raise SafetyError("--recreate-smoke-collection is allowed only in smoke mode.")

    if args.mode == "pilot" and args.recreate_full_collection:
        raise SafetyError("--recreate-full-collection is allowed only in full mode.")

    if args.mode == "full" and args.recreate_smoke_collection:
        raise SafetyError("--recreate-smoke-collection is allowed only in smoke mode.")

    if args.mode == "full" and args.recreate_pilot_collection:
        raise SafetyError("--recreate-pilot-collection is allowed only in pilot mode.")

    if args.recreate_smoke_collection and not _contains_marker(
        args.collection_name, RECREATE_SMOKE_ALLOWED_MARKERS
    ):
        raise SafetyError(
            "--recreate-smoke-collection is allowed only for collection names containing "
            "'smoke' or 'tmp'."
        )

    if args.recreate_pilot_collection and not _contains_marker(
        args.collection_name, RECREATE_PILOT_ALLOWED_MARKERS
    ):
        raise SafetyError(
            "--recreate-pilot-collection is allowed only for collection names containing "
            "'pilot' or 'tmp'."
        )

    if args.recreate_full_collection and not _contains_marker(
        args.collection_name, RECREATE_FULL_ALLOWED_MARKERS
    ):
        raise SafetyError(
            "--recreate-full-collection is allowed only for collection names containing "
            "'full' or 'tmp'."
        )

    if args.mode == "full" and args.full_record_batch_size <= 0:
        raise SafetyError("--full-record-batch-size must be greater than zero in full mode.")

    if args.recreate_full_collection and args.resume_full_collection:
        raise SafetyError("--recreate-full-collection and --resume-full-collection cannot be used together.")

    if args.resume_full_collection and args.mode != "full":
        raise SafetyError("--resume-full-collection is allowed only in full mode.")

    if args.resume_full_collection and not args.execute:
        raise SafetyError("--resume-full-collection requires --execute.")

    validate_decision_date_args(args)
    validate_newest_first_args(args)


def validate_collection_name(collection_name: str, *, execute: bool, mode: str = "smoke") -> None:
    normalized = collection_name.strip()
    if not normalized:
        raise SafetyError("--collection-name must be explicitly provided.")

    if normalized in PRODUCTION_COLLECTION_DENYLIST:
        raise SafetyError(f"Refusing to write to protected collection: {normalized}")

    if normalized.startswith("nalus_stable_"):
        raise SafetyError(f"Refusing to write to stable production collection: {normalized}")

    allowed_markers = {
        "pilot": ALLOWED_PILOT_NAME_MARKERS,
        "full": ALLOWED_FULL_NAME_MARKERS,
    }.get(mode, ALLOWED_STAGE1_NAME_MARKERS)
    if execute and not _contains_marker(normalized, allowed_markers):
        raise SafetyError(
            f"{mode.title()} execution requires collection name to include one of: "
            f"{', '.join(allowed_markers)}."
        )


def validate_limit(mode: str, limit: int) -> None:
    if mode == "full":
        validate_full_limit(limit)
        return
    if mode == "pilot":
        validate_pilot_limit(limit)
        return
    validate_smoke_limit(limit)


def validate_full_limit(limit: int) -> None:
    if limit < 0:
        raise SafetyError("Full mode --limit must be 0 (all deduplicated records) or a positive cap.")


def validate_smoke_limit(limit: int) -> None:
    if limit <= 0:
        raise SafetyError("--limit must be greater than zero.")
    if limit > 100:
        raise SafetyError("Smoke mode refuses --limit above 100.")


def validate_pilot_limit(limit: int) -> None:
    if limit <= 0:
        raise SafetyError("--limit must be greater than zero.")
    if limit > MAX_STAGE2_PILOT_LIMIT:
        raise SafetyError(f"Pilot mode refuses --limit above {MAX_STAGE2_PILOT_LIMIT}.")


def validate_top_k(top_k: int) -> None:
    if top_k <= 0:
        raise SafetyError("--top-k-smoke-test must be greater than zero.")
    if top_k > 20:
        raise SafetyError("--top-k-smoke-test above 20 is refused for smoke mode.")


def validate_decision_date_args(args: argparse.Namespace) -> None:
    if args.mode != "full":
        if any(
            (
                args.years_back is not None,
                args.decision_date_from,
                args.decision_date_to,
                args.ingest_slice,
                args.append_full_slice,
                args.newest_first,
            )
        ):
            raise SafetyError(
                "Decision-date filtering, ingest slices, and --newest-first are allowed only in full mode."
            )
        return

    if args.years_back is not None and args.years_back <= 0:
        raise SafetyError("--years-back must be a positive integer.")

    if args.decision_date_from:
        _parse_iso_date(args.decision_date_from, field_name="--decision-date-from")
    if args.decision_date_to:
        _parse_iso_date(args.decision_date_to, field_name="--decision-date-to")

    if args.append_full_slice and args.recreate_full_collection:
        raise SafetyError("--append-full-slice cannot be combined with --recreate-full-collection.")

    if args.append_full_slice and not args.execute:
        raise SafetyError("--append-full-slice requires --execute.")

    if args.append_full_slice and not (args.ingest_slice or args.decision_date_from or args.years_back):
        raise SafetyError(
            "--append-full-slice requires an explicit slice via --ingest-slice and/or date bounds."
        )


def validate_newest_first_args(args: argparse.Namespace) -> None:
    if not args.newest_first:
        return
    if args.mode != "full":
        raise SafetyError("--newest-first is allowed only in full mode.")
    if args.limit == FULL_LIMIT_ALL:
        raise SafetyError(
            f"--newest-first requires a positive --limit (e.g. {MVP_RECENT_3H_RECORD_LIMIT} for ~3h CPU MVP)."
        )


def validate_vector_dimension(vectors: list[list[float]], expected_dim: int = BGE_M3_DIMENSION) -> None:
    for index, vector in enumerate(vectors):
        if len(vector) != expected_dim:
            raise SafetyError(
                f"BGE-M3 vector dimension validation failed at vector {index}: "
                f"expected {expected_dim}, got {len(vector)}."
            )


def _contains_marker(value: str, markers: tuple[str, ...]) -> bool:
    normalized = value.lower()
    return any(marker in normalized for marker in markers)


def load_source_records(args: argparse.Namespace) -> list[SourceRecord]:
    if args.source_batch:
        return load_batch_records(resolve_project_path(args.source_batch))
    return load_manifest_records(resolve_project_path(args.source_manifest))


def resolve_project_path(path: Path | None) -> Path:
    if path is None:
        raise ValueError("Path must not be None.")
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_batch_records(path: Path) -> list[SourceRecord]:
    raw = _read_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"Expected list of records in {path}.")
    return _records_from_items(raw, origin_file=path.name)


def load_manifest_records(path: Path) -> list[SourceRecord]:
    manifest = _read_json(path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected manifest object in {path}.")

    records: list[SourceRecord] = []
    for entry in manifest.get("batches", []):
        if not isinstance(entry, dict) or not entry.get("file"):
            continue
        batch_path = path.parent / str(entry["file"])
        if batch_path.name == "manifest.json" or not batch_path.exists():
            continue
        records.extend(load_batch_records(batch_path))
    return records


def select_records(records: list[SourceRecord], *, limit: int) -> list[SourceRecord]:
    if limit == FULL_LIMIT_ALL:
        return records
    return records[:limit]


def order_records_by_decision_date(
    records: list[SourceRecord],
    *,
    newest_first: bool,
) -> list[SourceRecord]:
    if not newest_first:
        return records

    def sort_key(record: SourceRecord) -> tuple[int, int, str]:
        parsed = parse_decision_date(record.decision_date)
        if parsed is None:
            return (1, 0, record.identity)
        return (0, -parsed.toordinal(), record.identity)

    return sorted(records, key=sort_key)


def prepare_selected_records(
    args: argparse.Namespace,
) -> tuple[list[SourceRecord], DecisionDateFilter, dict[str, int]]:
    records = load_source_records(args)
    with_text = [record for record in records if record.full_text.strip()]
    deduped = deduplicate_records(with_text)
    date_filter = resolve_decision_date_filter(args)
    filtered, filter_stats = filter_records_by_decision_date(deduped, date_filter)
    ordered = order_records_by_decision_date(filtered, newest_first=args.newest_first)
    selected = select_records(ordered, limit=args.limit)
    return selected, date_filter, filter_stats


def filter_records_by_decision_date(
    records: list[SourceRecord],
    date_filter: DecisionDateFilter,
) -> tuple[list[SourceRecord], dict[str, int]]:
    if date_filter.date_from is None and date_filter.date_to is None:
        return records, {
            "deduplicated_record_count": len(records),
            "date_filtered_record_count": len(records),
            "date_unparseable_record_count": 0,
            "date_out_of_range_record_count": 0,
        }

    kept: list[SourceRecord] = []
    unparseable = 0
    out_of_range = 0
    for record in records:
        parsed = parse_decision_date(record.decision_date)
        if parsed is None:
            unparseable += 1
            continue
        if date_filter.date_from is not None and parsed < date_filter.date_from:
            out_of_range += 1
            continue
        if date_filter.date_to is not None and parsed > date_filter.date_to:
            out_of_range += 1
            continue
        kept.append(record)

    return kept, {
        "deduplicated_record_count": len(records),
        "date_filtered_record_count": len(kept),
        "date_unparseable_record_count": unparseable,
        "date_out_of_range_record_count": out_of_range,
    }


def resolve_decision_date_filter(args: argparse.Namespace) -> DecisionDateFilter:
    ingest_slice = (args.ingest_slice or "").strip()
    years_back = args.years_back
    date_from = _parse_iso_date(args.decision_date_from, field_name="--decision-date-from") if args.decision_date_from else None
    date_to = _parse_iso_date(args.decision_date_to, field_name="--decision-date-to") if args.decision_date_to else None

    if years_back is not None:
        today = _utc_today()
        rolling_from = _subtract_years(today, years_back)
        date_from = rolling_from if date_from is None else date_from
        date_to = today if date_to is None else date_to

    if not ingest_slice:
        if years_back is not None:
            ingest_slice = f"mvp_{years_back}y"
        elif date_from or date_to:
            ingest_slice = "dated_slice"
        else:
            ingest_slice = "full"

    return DecisionDateFilter(
        date_from=date_from,
        date_to=date_to,
        years_back=years_back,
        ingest_slice=ingest_slice,
    )


def parse_decision_date(value: str | None) -> date | None:
    cleaned = str(value or "").strip()
    if not cleaned:
        return None

    iso_match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", cleaned)
    if iso_match:
        return date(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3)))

    czech_match = re.fullmatch(r"(\d{1,2})\.\s*(\d{1,2})\.\s*(\d{4})", cleaned)
    if czech_match:
        return date(int(czech_match.group(3)), int(czech_match.group(2)), int(czech_match.group(1)))

    return None


def _parse_iso_date(value: str, *, field_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise SafetyError(f"{field_name} must use YYYY-MM-DD format.") from exc


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


def _subtract_years(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year - years)
    except ValueError:
        return day.replace(month=2, day=28, year=day.year - years)


def deduplicate_records(records: list[SourceRecord]) -> list[SourceRecord]:
    best: dict[str, SourceRecord] = {}
    for record in records:
        current = best.get(record.identity)
        if current is None or _record_rank(record) > _record_rank(current):
            best[record.identity] = record
    return list(best.values())


def chunk_records(
    records: list[SourceRecord],
    *,
    collection_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
    start_seq_id: int = 1,
) -> list[SmokeChunk]:
    chunks: list[SmokeChunk] = []
    seq_id = start_seq_id
    for record in records:
        text_chunks = split_text_into_chunks(
            record.full_text,
            chunk_size=chunk_size,
            overlap_words=overlap_words,
        )
        chunk_count = len(text_chunks)
        for chunk_index, text in enumerate(text_chunks):
            point_id = _point_id(collection_name, record.identity, chunk_index)
            payload = {
                "chunk_id": seq_id,
                "source_id": SOURCE_ID,
                "text": text,
                "source": "usoud / nalus",
                "document_id": record.identity,
                "court": "Ústavní soud",
                "decision_date": record.decision_date,
                "source_document_id": record.source_document_id,
                "case_reference": record.case_reference,
                "spisova_znacka": record.case_reference,
                "ecli": record.ecli,
                "detail_url": record.detail_url,
                "text_url": record.text_url,
                "origin_file": record.origin_file,
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "text_length": len(text),
                "builder_version": BUILDER_VERSION,
                "embedding_provider": "sentence_transformer",
                "embedding_model": BGE_M3_MODEL_NAME,
                "embedding_dimension": BGE_M3_DIMENSION,
                "retrieval_profile": PRODUCTION_RETRIEVAL_PROFILE,
                "ingest_run_id": BUILDER_VERSION,
                "qdrant_collection": collection_name,
                "bm25_index_id": DEFAULT_BM25_INDEX_ID,
                "content_checksum": _content_checksum(text),
            }
            chunks.append(SmokeChunk(seq_id=seq_id, point_id=point_id, text=text, payload=payload))
            seq_id += 1
    return chunks


def _content_checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def count_chunks_for_records(
    records: list[SourceRecord],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
    progress_label: str | None = None,
) -> int:
    total = 0
    for index, record in enumerate(records, start=1):
        total += len(
            split_text_into_chunks(
                record.full_text,
                chunk_size=chunk_size,
                overlap_words=overlap_words,
            )
        )
        if progress_label and index % FULL_DRY_RUN_PROGRESS_INTERVAL == 0:
            print(
                f"[full dry-run] {progress_label}: counted chunks for {index}/{len(records)} records "
                f"({total} chunks so far)",
                file=sys.stderr,
                flush=True,
            )
    return total


def split_text_into_chunks(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n+", normalized) if paragraph.strip()]
    if not paragraphs:
        paragraphs = [normalized]

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        parts = _split_long_paragraph(paragraph, chunk_size=chunk_size)
        for part in parts:
            if not current:
                current = part
                continue
            candidate = f"{current}\n\n{part}"
            if len(candidate) <= chunk_size:
                current = candidate
                continue
            chunks.append(current)
            current = _with_overlap(current, part, overlap_words=overlap_words, chunk_size=chunk_size)

    if current:
        chunks.append(current)

    return chunks


def build_summary(
    args: argparse.Namespace,
    selected: list[SourceRecord],
    chunks: list[SmokeChunk],
    *,
    chunk_count: int | None = None,
    date_filter: DecisionDateFilter | None = None,
    filter_stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    source_files = list_source_files(args)
    resolved_chunk_count = chunk_count if chunk_count is not None else len(chunks)
    resolved_filter = date_filter or resolve_decision_date_filter(args)
    resolved_stats = filter_stats or {}
    return {
        "generated_at": _utc_now(),
        "builder_version": BUILDER_VERSION,
        "script_path": "scripts/build_usoud_bge_m3_candidate.py",
        "mode": args.mode,
        "action": "execute" if args.execute else "dry-run",
        "command": _format_command(sys.argv),
        "input": str(args.source_batch or args.source_manifest),
        "source_files": source_files,
        "source_file_count": len(source_files),
        "collection_name": args.collection_name,
        "limit": args.limit,
        "newest_first": bool(args.newest_first),
        "ingest_slice": resolved_filter.ingest_slice,
        "decision_date_filter": resolved_filter.as_summary(),
        "record_filter_stats": resolved_stats,
        "selected_record_count": len(selected),
        "generated_chunk_count": resolved_chunk_count,
        "estimated_qdrant_points": resolved_chunk_count,
        "estimated_embedding_texts": resolved_chunk_count,
        "embedding_model": BGE_M3_MODEL_NAME,
        "expected_vector_dimension": BGE_M3_DIMENSION,
        "vector_dimension_validation": "not_run",
        "qdrant": {
            "collection_point_count_before": None,
            "collection_point_count_after": None,
            "inserted_point_count": 0,
            "nalus_live_before": None,
            "nalus_live_after": None,
            "nalus_stable_20260326_before": None,
            "nalus_stable_20260326_after": None,
            "aliases_before": [],
            "aliases_after": [],
            "nalus_live_target_before": None,
            "nalus_live_target_after": None,
            "aliases_changed": None,
        },
        "production_safety": {
            "production_touched": False,
            "aliases_changed": False,
            "nalus_live_changed": False,
            "nalus_stable_changed": False,
            "nalus_live_target_changed": False,
        },
        "qdrant_write_occurred": False,
        "payload_metadata_validation": "not_run",
        "sample_payloads": [],
        "bm25_status": "not_run",
        "hybrid_status": "not_run",
        "smoke_queries": [],
        "warnings": [],
        "failures": [],
        "production_api_touched": False,
        "retrieval_logic_changed": False,
        "clarification_gate_changed": False,
        "aliases_touched": False,
        "stage_recommendation": _stage_recommendation_for_mode(args.mode, dry_run=not args.execute),
        "final_status": "not_run",
    }


def run_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    selected, date_filter, filter_stats = prepare_selected_records(args)
    if args.mode == "full":
        order_label = "newest-first" if args.newest_first else "manifest-order"
        print(
            f"[full dry-run] slice={date_filter.ingest_slice} "
            f"from={date_filter.date_from} to={date_filter.date_to} "
            f"order={order_label}; "
            f"selected {len(selected)} records after date filter; counting chunks...",
            file=sys.stderr,
            flush=True,
        )
        chunk_count = count_chunks_for_records(
            selected,
            chunk_size=args.chunk_size,
            progress_label="chunk estimation",
        )
        chunks: list[SmokeChunk] = []
        summary = build_summary(
            args,
            selected,
            chunks,
            chunk_count=chunk_count,
            date_filter=date_filter,
            filter_stats=filter_stats,
        )
    else:
        chunks = chunk_records(selected, collection_name=args.collection_name, chunk_size=args.chunk_size)
        summary = build_summary(
            args,
            selected,
            chunks,
            date_filter=date_filter,
            filter_stats=filter_stats,
        )
    _record_read_only_qdrant_state(args, summary, phase="before")
    if args.mode == "full":
        snapshot = _production_snapshot_from_summary(summary, phase="before")
        write_production_safety_snapshot(args, snapshot, phase="before")
    summary["final_status"] = _final_status(summary)
    write_outputs(args, summary, dry_run=True)
    return summary


def run_execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "full":
        return run_execute_full_batched(args)
    selected, date_filter, filter_stats = prepare_selected_records(args)
    chunks = chunk_records(selected, collection_name=args.collection_name, chunk_size=args.chunk_size)
    summary = build_summary(
        args,
        selected,
        chunks,
        date_filter=date_filter,
        filter_stats=filter_stats,
    )
    dry_run_summary = _load_previous_dry_run(args.output_dir)
    if dry_run_summary:
        summary["dry_run_command"] = dry_run_summary.get("command")

    if not chunks:
        raise SafetyError("No chunks generated; refusing to create an empty smoke collection.")

    model = _load_bge_m3_model()
    vectors = _encode_chunks(
        model,
        [chunk.text for chunk in chunks],
        batch_size=args.embedding_batch_size,
        content_checksums=[str(chunk.payload.get("content_checksum") or "") for chunk in chunks],
    )
    validate_vector_dimension(vectors)
    summary["vector_dimension_validation"] = f"PASS ({BGE_M3_DIMENSION})"

    client = _qdrant_client(args.qdrant_url)
    production_before = capture_production_safety_snapshot(client)
    write_production_safety_snapshot(args, production_before, phase="before")

    aliases_before = production_before["aliases"]
    live_before = production_before["nalus_live_point_count"]
    stable_before = production_before["nalus_stable_20260326_point_count"]
    collection_before = _count_collection(client, args.collection_name)

    summary["qdrant"]["nalus_live_before"] = live_before
    summary["qdrant"]["nalus_stable_20260326_before"] = stable_before
    summary["qdrant"]["collection_point_count_before"] = collection_before
    summary["qdrant"]["aliases_before"] = aliases_before
    summary["qdrant"]["nalus_live_target_before"] = production_before["nalus_live_target"]

    _prepare_candidate_collection(
        client,
        collection_name=args.collection_name,
        recreate=_recreate_requested(args),
        existing_count=collection_before,
        mode=args.mode,
    )
    _upsert_chunks(client, collection_name=args.collection_name, chunks=chunks, vectors=vectors)
    summary["qdrant_write_occurred"] = True
    summary["qdrant"]["inserted_point_count"] = len(chunks)

    collection_after = _count_collection(client, args.collection_name)
    sample_payloads = _sample_payloads(client, args.collection_name)
    payload_check = _validate_sample_payloads(sample_payloads)
    summary["sample_payloads"] = sample_payloads
    summary["payload_metadata_validation"] = "PASS" if payload_check else "FAIL"
    if not payload_check:
        raise SafetyError("Qdrant payload verification failed: required payload metadata is missing.")

    query_results, bm25_status, hybrid_status = _run_smoke_queries(
        client=client,
        collection_name=args.collection_name,
        model=model,
        chunks=chunks,
        top_k=args.top_k_smoke_test,
        mode=args.mode,
    )

    aliases_after = _aliases_snapshot(client)
    live_after = _count_collection(client, "nalus_live")
    stable_after = _count_collection(client, "nalus_stable_20260326")
    production_after = capture_production_safety_snapshot(client)
    write_production_safety_snapshot(args, production_after, phase="after")
    safety_delta = compare_production_safety(production_before, production_after)

    summary["qdrant"]["collection_point_count_after"] = collection_after
    summary["qdrant"]["nalus_live_after"] = live_after
    summary["qdrant"]["nalus_stable_20260326_after"] = stable_after
    summary["qdrant"]["aliases_after"] = aliases_after
    summary["qdrant"]["nalus_live_target_after"] = production_after["nalus_live_target"]
    summary["qdrant"]["aliases_changed"] = safety_delta["aliases_changed"]
    summary["production_safety"] = safety_delta
    summary["bm25_status"] = bm25_status
    summary["hybrid_status"] = hybrid_status
    summary["smoke_queries"] = query_results

    if collection_after != len(chunks):
        summary["warnings"].append(
            f"Candidate collection point count is {collection_after}, expected {len(chunks)} after recreate."
        )
    if safety_delta["production_touched"]:
        raise SafetyError(
            "Production safety check failed: "
            f"nalus_live {live_before}->{live_after}, "
            f"nalus_stable_20260326 {stable_before}->{stable_after}, "
            f"aliases_changed={safety_delta['aliases_changed']}"
        )

    summary["stage_recommendation"] = _stage_recommendation_for_mode(args.mode, dry_run=False)
    summary["final_status"] = _final_status(summary)
    write_outputs(args, summary, dry_run=False)
    write_retrieval_validation(args, query_results)
    return summary


def run_execute_full_batched(args: argparse.Namespace) -> dict[str, Any]:
    selected, date_filter, filter_stats = prepare_selected_records(args)
    if not selected:
        raise SafetyError("No records selected; refusing to create an empty full collection.")

    dry_run_summary = _load_previous_dry_run(args.output_dir)
    expected_chunk_count = 0
    if dry_run_summary:
        expected_chunk_count = int(dry_run_summary.get("estimated_qdrant_points") or 0)

    summary = build_summary(
        args,
        selected,
        [],
        chunk_count=expected_chunk_count or None,
        date_filter=date_filter,
        filter_stats=filter_stats,
    )
    if dry_run_summary:
        summary["dry_run_command"] = dry_run_summary.get("command")

    model = _load_bge_m3_model()
    client = _qdrant_client(args.qdrant_url)
    production_before = capture_production_safety_snapshot(client)
    write_production_safety_snapshot(args, production_before, phase="before")

    live_before = production_before["nalus_live_point_count"]
    stable_before = production_before["nalus_stable_20260326_point_count"]
    collection_before = _count_collection(client, args.collection_name)

    summary["qdrant"]["nalus_live_before"] = live_before
    summary["qdrant"]["nalus_stable_20260326_before"] = stable_before
    summary["qdrant"]["collection_point_count_before"] = collection_before
    summary["qdrant"]["aliases_before"] = production_before["aliases"]
    summary["qdrant"]["nalus_live_target_before"] = production_before["nalus_live_target"]

    resume_state = resolve_full_execute_resume(
        args,
        selected=selected,
        date_filter=date_filter,
        collection_before=collection_before,
        expected_chunk_count=expected_chunk_count,
    )
    summary["full_execute_resume"] = resume_state["resume_mode"]
    summary["append_full_slice"] = bool(args.append_full_slice)

    _prepare_candidate_collection(
        client,
        collection_name=args.collection_name,
        recreate=_recreate_requested(args),
        existing_count=collection_before,
        mode=args.mode,
        resume=resume_state["resume_mode"] or bool(args.append_full_slice),
    )

    next_record_index = resume_state["next_record_index"]
    next_seq_id = resume_state["next_seq_id"]
    inserted = resume_state["inserted_point_count"]
    collection_at_start = resume_state["collection_point_count_at_start"]
    dimension_validated = inserted > 0 or (collection_at_start or 0) > 0
    if dimension_validated and inserted > 0:
        summary["vector_dimension_validation"] = f"PASS ({BGE_M3_DIMENSION})"

    record_batch_size = args.full_record_batch_size
    total_records = len(selected)

    if resume_state["resume_mode"]:
        print(
            f"[full execute] slice={date_filter.ingest_slice} resuming from record "
            f"{next_record_index}/{total_records}, slice chunks upserted {inserted}",
            file=sys.stderr,
            flush=True,
        )
    else:
        order_label = "newest-first" if args.newest_first else "manifest-order"
        print(
            f"[full execute] slice={date_filter.ingest_slice} "
            f"window={date_filter.date_from}..{date_filter.date_to} "
            f"order={order_label} "
            f"starting batched ingest for {total_records} records "
            f"(batch size {record_batch_size})",
            file=sys.stderr,
            flush=True,
        )

    checkpoint = resume_state["checkpoint"]
    for batch_start in range(next_record_index, total_records, record_batch_size):
        batch_records = selected[batch_start : batch_start + record_batch_size]
        chunks = chunk_records(
            batch_records,
            collection_name=args.collection_name,
            chunk_size=args.chunk_size,
            start_seq_id=next_seq_id,
        )
        if not chunks:
            checkpoint = _advance_execute_checkpoint(
                checkpoint,
                next_record_index=min(batch_start + record_batch_size, total_records),
                next_seq_id=next_seq_id,
                inserted_point_count=inserted,
                last_record_identity=batch_records[-1].identity if batch_records else None,
            )
            write_execute_checkpoint(args, checkpoint)
            continue
        vectors = _encode_chunks(
            model,
            [chunk.text for chunk in chunks],
            batch_size=args.embedding_batch_size,
            content_checksums=[str(chunk.payload.get("content_checksum") or "") for chunk in chunks],
        )
        validate_vector_dimension(vectors)
        if not dimension_validated:
            summary["vector_dimension_validation"] = f"PASS ({BGE_M3_DIMENSION})"
            dimension_validated = True
        _upsert_chunks(client, collection_name=args.collection_name, chunks=chunks, vectors=vectors)
        inserted += len(chunks)
        next_seq_id += len(chunks)
        processed_records = min(batch_start + record_batch_size, total_records)
        progress_target = expected_chunk_count or "?"
        print(
            f"[full execute] records {processed_records}/{total_records}, "
            f"chunks upserted {inserted}/{progress_target}",
            file=sys.stderr,
            flush=True,
        )
        checkpoint = _advance_execute_checkpoint(
            checkpoint,
            next_record_index=processed_records,
            next_seq_id=next_seq_id,
            inserted_point_count=inserted,
            last_record_identity=batch_records[-1].identity,
        )
        write_execute_checkpoint(args, checkpoint)

    if inserted == 0:
        raise SafetyError("No chunks generated; refusing to leave an empty full collection.")

    clear_execute_checkpoint(args)
    summary["qdrant_write_occurred"] = True
    summary["qdrant"]["inserted_point_count"] = inserted
    summary["slice_inserted_point_count"] = inserted
    summary["collection_point_count_at_start"] = collection_at_start
    summary["generated_chunk_count"] = inserted
    summary["estimated_qdrant_points"] = inserted
    summary["estimated_embedding_texts"] = inserted

    collection_after = _count_collection(client, args.collection_name)
    sample_payloads = _sample_payloads(client, args.collection_name)
    payload_check = _validate_sample_payloads(sample_payloads)
    summary["sample_payloads"] = sample_payloads
    summary["payload_metadata_validation"] = "PASS" if payload_check else "FAIL"
    if not payload_check:
        raise SafetyError("Qdrant payload verification failed: required payload metadata is missing.")

    validation_chunks = _load_bounded_chunks_from_qdrant(
        client,
        args.collection_name,
        limit=FULL_VALIDATION_BM25_CHUNK_LIMIT,
    )
    summary["validation_bm25_chunk_sample_size"] = len(validation_chunks)
    query_results, bm25_status, hybrid_status = _run_smoke_queries(
        client=client,
        collection_name=args.collection_name,
        model=model,
        chunks=validation_chunks,
        top_k=args.top_k_smoke_test,
        mode=args.mode,
    )

    production_after = capture_production_safety_snapshot(client)
    write_production_safety_snapshot(args, production_after, phase="after")
    safety_delta = compare_production_safety(production_before, production_after)

    summary["qdrant"]["collection_point_count_after"] = collection_after
    summary["qdrant"]["nalus_live_after"] = production_after["nalus_live_point_count"]
    summary["qdrant"]["nalus_stable_20260326_after"] = production_after["nalus_stable_20260326_point_count"]
    summary["qdrant"]["aliases_after"] = production_after["aliases"]
    summary["qdrant"]["nalus_live_target_after"] = production_after["nalus_live_target"]
    summary["qdrant"]["aliases_changed"] = safety_delta["aliases_changed"]
    summary["production_safety"] = safety_delta
    summary["bm25_status"] = bm25_status
    summary["hybrid_status"] = hybrid_status
    summary["smoke_queries"] = query_results

    expected_collection_after = (collection_at_start or 0) + inserted
    if collection_after != expected_collection_after:
        summary["warnings"].append(
            "Candidate collection point count is "
            f"{collection_after}, expected {expected_collection_after} "
            f"(start={collection_at_start}, slice_inserted={inserted})."
        )
    if safety_delta["production_touched"]:
        raise SafetyError(
            "Production safety check failed: "
            f"nalus_live {live_before}->{production_after['nalus_live_point_count']}, "
            f"nalus_stable_20260326 {stable_before}->{production_after['nalus_stable_20260326_point_count']}, "
            f"aliases_changed={safety_delta['aliases_changed']}"
        )

    summary["stage_recommendation"] = _stage_recommendation_for_mode(args.mode, dry_run=False)
    summary["final_status"] = _final_status(summary)
    write_outputs(args, summary, dry_run=False)
    write_retrieval_validation(args, query_results)
    return summary


def _checkpoint_path(args: argparse.Namespace) -> Path:
    return resolve_project_path(args.output_dir) / EXECUTE_CHECKPOINT_FILENAME


def load_execute_checkpoint(args: argparse.Namespace) -> dict[str, Any] | None:
    path = _checkpoint_path(args)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SafetyError(f"Execute checkpoint at {path} is invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SafetyError(f"Execute checkpoint at {path} must be a JSON object.")
    return data


def write_execute_checkpoint(args: argparse.Namespace, checkpoint: dict[str, Any]) -> None:
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint["updated_at"] = _utc_now()
    path = _checkpoint_path(args)
    path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clear_execute_checkpoint(args: argparse.Namespace) -> None:
    path = _checkpoint_path(args)
    if path.exists():
        path.unlink()


def _build_execute_checkpoint(
    args: argparse.Namespace,
    *,
    selected: list[SourceRecord],
    expected_chunk_count: int,
    date_filter: DecisionDateFilter,
    collection_point_count_at_start: int,
    initial_next_seq_id: int = 1,
    initial_inserted_point_count: int = 0,
) -> dict[str, Any]:
    return {
        "status": "in_progress",
        "builder_version": BUILDER_VERSION,
        "collection_name": args.collection_name,
        "source_manifest": str(args.source_manifest),
        "limit": args.limit,
        "chunk_size": args.chunk_size,
        "full_record_batch_size": args.full_record_batch_size,
        "embedding_batch_size": args.embedding_batch_size,
        "ingest_slice": date_filter.ingest_slice,
        "decision_date_filter": date_filter.as_summary(),
        "newest_first": bool(args.newest_first),
        "total_records": len(selected),
        "expected_chunk_count": expected_chunk_count,
        "collection_point_count_at_start": collection_point_count_at_start,
        "next_record_index": 0,
        "next_seq_id": initial_next_seq_id,
        "inserted_point_count": initial_inserted_point_count,
        "last_record_identity": None,
        "updated_at": _utc_now(),
    }


def _checkpoint_matches_slice(
    checkpoint: dict[str, Any],
    date_filter: DecisionDateFilter,
    *,
    newest_first: bool,
) -> bool:
    return (
        checkpoint.get("ingest_slice") == date_filter.ingest_slice
        and checkpoint.get("decision_date_filter") == date_filter.as_summary()
        and bool(checkpoint.get("newest_first", False)) == newest_first
    )


def _advance_execute_checkpoint(
    checkpoint: dict[str, Any],
    *,
    next_record_index: int,
    next_seq_id: int,
    inserted_point_count: int,
    last_record_identity: str | None,
) -> dict[str, Any]:
    updated = dict(checkpoint)
    updated["next_record_index"] = next_record_index
    updated["next_seq_id"] = next_seq_id
    updated["inserted_point_count"] = inserted_point_count
    updated["last_record_identity"] = last_record_identity
    updated["status"] = "in_progress"
    return updated


def _validate_execute_checkpoint(
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    *,
    selected: list[SourceRecord],
    expected_chunk_count: int,
    date_filter: DecisionDateFilter,
) -> None:
    if checkpoint.get("status") != "in_progress":
        raise SafetyError(
            f"Execute checkpoint status is {checkpoint.get('status')!r}; only in_progress checkpoints can resume."
        )
    if checkpoint.get("collection_name") != args.collection_name:
        raise SafetyError("Execute checkpoint collection_name does not match --collection-name.")
    if checkpoint.get("source_manifest") != str(args.source_manifest):
        raise SafetyError("Execute checkpoint source_manifest does not match current --source-manifest.")
    if checkpoint.get("limit") != args.limit:
        raise SafetyError("Execute checkpoint limit does not match current --limit.")
    if checkpoint.get("chunk_size") != args.chunk_size:
        raise SafetyError("Execute checkpoint chunk_size does not match current --chunk-size.")
    if checkpoint.get("full_record_batch_size") != args.full_record_batch_size:
        raise SafetyError(
            "Execute checkpoint full_record_batch_size does not match current --full-record-batch-size."
        )
    if checkpoint.get("embedding_batch_size") != args.embedding_batch_size:
        raise SafetyError(
            "Execute checkpoint embedding_batch_size does not match current --embedding-batch-size."
        )
    if not _checkpoint_matches_slice(checkpoint, date_filter, newest_first=bool(args.newest_first)):
        raise SafetyError(
            "Execute checkpoint ingest slice/date filter/newest-first flag does not match current run."
        )
    if checkpoint.get("total_records") != len(selected):
        raise SafetyError("Execute checkpoint total_records does not match current selected record count.")
    if expected_chunk_count and checkpoint.get("expected_chunk_count") != expected_chunk_count:
        raise SafetyError("Execute checkpoint expected_chunk_count does not match dry-run summary.")
    next_record_index = int(checkpoint.get("next_record_index") or 0)
    if next_record_index < 0 or next_record_index > len(selected):
        raise SafetyError("Execute checkpoint next_record_index is out of range.")


def resolve_full_execute_resume(
    args: argparse.Namespace,
    *,
    selected: list[SourceRecord],
    date_filter: DecisionDateFilter,
    collection_before: int | None,
    expected_chunk_count: int,
) -> dict[str, Any]:
    checkpoint = load_execute_checkpoint(args)
    recreate = _recreate_requested(args)
    explicit_resume = args.resume_full_collection
    collection_count = collection_before or 0

    if checkpoint is not None and not _checkpoint_matches_slice(
        checkpoint, date_filter, newest_first=bool(args.newest_first)
    ):
        if recreate or args.append_full_slice:
            clear_execute_checkpoint(args)
            checkpoint = None
        else:
            raise SafetyError(
                "Execute checkpoint belongs to a different ingest slice/date window/order. "
                "Use matching --ingest-slice/date/--newest-first args, --append-full-slice, "
                "or --recreate-full-collection."
            )

    if recreate:
        clear_execute_checkpoint(args)
        checkpoint = _build_execute_checkpoint(
            args,
            selected=selected,
            expected_chunk_count=expected_chunk_count,
            date_filter=date_filter,
            collection_point_count_at_start=0,
        )
        write_execute_checkpoint(args, checkpoint)
        return {
            "resume_mode": False,
            "next_record_index": 0,
            "next_seq_id": 1,
            "inserted_point_count": 0,
            "collection_point_count_at_start": 0,
            "checkpoint": checkpoint,
        }

    if checkpoint is not None:
        _validate_execute_checkpoint(
            args,
            checkpoint,
            selected=selected,
            expected_chunk_count=expected_chunk_count,
            date_filter=date_filter,
        )
        at_start = int(checkpoint.get("collection_point_count_at_start") or 0)
        slice_inserted = int(checkpoint.get("inserted_point_count") or 0)
        expected_qdrant = at_start + slice_inserted
        if collection_count != expected_qdrant:
            raise SafetyError(
                "Execute checkpoint progress does not match Qdrant collection count: "
                f"expected {expected_qdrant} (start={at_start}, slice_inserted={slice_inserted}), "
                f"got {collection_count}. Refusing to resume until counts match or "
                "--recreate-full-collection is used."
            )
        return {
            "resume_mode": True,
            "next_record_index": int(checkpoint.get("next_record_index") or 0),
            "next_seq_id": int(checkpoint.get("next_seq_id") or 1),
            "inserted_point_count": slice_inserted,
            "collection_point_count_at_start": at_start,
            "checkpoint": checkpoint,
        }

    if collection_count > 0:
        if args.append_full_slice:
            checkpoint = _build_execute_checkpoint(
                args,
                selected=selected,
                expected_chunk_count=expected_chunk_count,
                date_filter=date_filter,
                collection_point_count_at_start=collection_count,
                initial_next_seq_id=collection_count + 1,
            )
            write_execute_checkpoint(args, checkpoint)
            return {
                "resume_mode": False,
                "next_record_index": 0,
                "next_seq_id": collection_count + 1,
                "inserted_point_count": 0,
                "collection_point_count_at_start": collection_count,
                "checkpoint": checkpoint,
            }
        if explicit_resume:
            raise SafetyError(
                "No execute checkpoint found for --resume-full-collection. "
                "Cannot safely infer resume position."
            )
        raise SafetyError(
            f"Full collection {args.collection_name!r} already has {collection_count} points but no "
            "matching execute_checkpoint.json was found. Use --append-full-slice for a new date slice, "
            "or --recreate-full-collection to start over."
        )

    checkpoint = _build_execute_checkpoint(
        args,
        selected=selected,
        expected_chunk_count=expected_chunk_count,
        date_filter=date_filter,
        collection_point_count_at_start=0,
    )
    write_execute_checkpoint(args, checkpoint)
    return {
        "resume_mode": False,
        "next_record_index": 0,
        "next_seq_id": 1,
        "inserted_point_count": 0,
        "collection_point_count_at_start": 0,
        "checkpoint": checkpoint,
    }


def _load_bounded_chunks_from_qdrant(
    client: Any,
    collection_name: str,
    *,
    limit: int,
) -> list[SmokeChunk]:
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    chunks: list[SmokeChunk] = []
    for point in points:
        payload = dict(point.payload or {})
        chunks.append(
            SmokeChunk(
                seq_id=int(payload.get("chunk_id") or 0),
                point_id=str(point.id),
                text=str(payload.get("text") or ""),
                payload=payload,
            )
        )
    return chunks


def write_outputs(args: argparse.Namespace, summary: dict[str, Any], *, dry_run: bool) -> None:
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / ("dry_run_summary.json" if dry_run else "execute_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = _report_path(args)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summary), encoding="utf-8")


def write_production_safety_snapshot(
    args: argparse.Namespace,
    snapshot: dict[str, Any],
    *,
    phase: str,
) -> None:
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"production_safety_snapshot_{phase}.json"
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_retrieval_validation(args: argparse.Namespace, query_results: list[dict[str, Any]]) -> None:
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": _utc_now(),
        "collection_name": args.collection_name,
        "mode": args.mode,
        "query_count": len(query_results),
        "all_hits_from_candidate_collection": all(
            item.get("all_hits_from_candidate_collection") for item in query_results
        ),
        "queries": query_results,
    }
    path = output_dir / "retrieval_validation.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def capture_production_safety_snapshot(client: Any) -> dict[str, Any]:
    aliases = _aliases_snapshot(client)
    return {
        "captured_at": _utc_now(),
        "nalus_live_point_count": _count_collection(client, "nalus_live"),
        "nalus_stable_20260326_point_count": _count_collection(client, "nalus_stable_20260326"),
        "aliases": aliases,
        "nalus_live_target": _alias_target(aliases, "nalus_live"),
    }


def compare_production_safety(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    aliases_changed = before.get("aliases") != after.get("aliases")
    nalus_live_changed = before.get("nalus_live_point_count") != after.get("nalus_live_point_count")
    nalus_stable_changed = (
        before.get("nalus_stable_20260326_point_count") != after.get("nalus_stable_20260326_point_count")
    )
    nalus_live_target_changed = before.get("nalus_live_target") != after.get("nalus_live_target")
    production_touched = any(
        (
            aliases_changed,
            nalus_live_changed,
            nalus_stable_changed,
            nalus_live_target_changed,
        )
    )
    return {
        "production_touched": production_touched,
        "aliases_changed": aliases_changed,
        "nalus_live_changed": nalus_live_changed,
        "nalus_stable_changed": nalus_stable_changed,
        "nalus_live_target_changed": nalus_live_target_changed,
    }


def _production_snapshot_from_summary(summary: dict[str, Any], *, phase: str) -> dict[str, Any]:
    qdrant = summary["qdrant"]
    return {
        "captured_at": summary["generated_at"],
        "nalus_live_point_count": qdrant.get(f"nalus_live_{phase}"),
        "nalus_stable_20260326_point_count": qdrant.get(f"nalus_stable_20260326_{phase}"),
        "aliases": qdrant.get(f"aliases_{phase}") or [],
        "nalus_live_target": qdrant.get(f"nalus_live_target_{phase}"),
    }


def list_source_files(args: argparse.Namespace) -> list[str]:
    if args.source_batch:
        path = resolve_project_path(args.source_batch)
        return [_report_relative_path(path)]
    manifest_path = resolve_project_path(args.source_manifest)
    files: list[str] = []
    manifest = _read_json(manifest_path)
    for entry in manifest.get("batches", []):
        if not isinstance(entry, dict) or not entry.get("file"):
            continue
        batch_path = manifest_path.parent / str(entry["file"])
        if batch_path.name == "manifest.json" or not batch_path.exists():
            continue
        files.append(_report_relative_path(batch_path))
    return sorted(files)


def _report_relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _stage_recommendation_for_mode(mode: str, *, dry_run: bool) -> str:
    if mode == "full":
        return "safe_to_execute_full_after_review" if dry_run else "full_candidate_complete_review_required"
    if mode == "pilot":
        return "safe_to_execute_pilot_after_review" if dry_run else "safe_after_review"
    return "safe_after_execute_smoke_passes" if dry_run else "safe_after_review"


def _final_status(summary: dict[str, Any]) -> str:
    warnings = summary.get("warnings") or []
    failures = summary.get("failures") or []
    if failures:
        return "FAIL"
    if summary.get("vector_dimension_validation", "").startswith("PASS") is False and summary.get("action") == "execute":
        return "FAIL"
    if summary.get("payload_metadata_validation") != "PASS" and summary.get("action") == "execute":
        return "FAIL"
    if summary.get("production_safety", {}).get("production_touched"):
        return "FAIL"
    if summary.get("action") == "execute" and not summary.get("smoke_queries"):
        return "FAIL"
    if summary.get("action") == "execute" and summary.get("bm25_status") not in {"available", "not_available"}:
        return "FAIL"
    if summary.get("action") == "dry-run":
        return "PASS"
    all_hits_ok = all(item.get("all_hits_from_candidate_collection") for item in summary.get("smoke_queries") or [])
    if summary.get("action") == "execute" and not all_hits_ok:
        return "FAIL"
    return "PASS"


def _report_path(args: argparse.Namespace) -> Path:
    if args.mode == "full":
        return STAGE3_FULL_REPORT_PATH
    if args.mode == "pilot":
        return STAGE2_PILOT_REPORT_PATH
    return REPORT_PATH


def render_report(summary: dict[str, Any]) -> str:
    qdrant = summary["qdrant"]
    dry_run_command = summary.get("dry_run_command")
    execute_command = summary["command"] if summary["action"] == "execute" else None
    if summary["action"] == "dry-run":
        dry_run_command = summary["command"]
    stage_label = {
        "full": "Stage 3 Full",
        "pilot": "Stage 2 Pilot",
    }.get(summary["mode"], "Stage 1 Smoke")
    query_heading = {
        "full": "Full Retrieval Validation Queries",
        "pilot": "Pilot Query Results",
    }.get(summary["mode"], "Smoke Query Results")

    lines = [
        f"# Ustavni soud / NALUS - BGE-M3 {stage_label} Report",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Goal",
        "",
        (
            "- Run a guarded full-corpus candidate collection without touching production."
            if summary["mode"] == "full"
            else "- Run a guarded 500-decision pilot candidate collection without touching production."
            if summary["mode"] == "pilot"
            else "- Run a guarded smoke candidate collection without touching production."
        ),
        "",
        "## Run Summary",
        "",
        f"- Script path: `{summary['script_path']}`",
        f"- Builder version: `{summary['builder_version']}`",
        f"- Stage 1 commit reference: `4290559 Add guarded ÚS BGE-M3 smoke builder`",
        f"- Mode: `{summary['mode']}`",
        f"- Action: `{summary['action']}`",
        f"- Dry-run command: `{dry_run_command or 'not recorded'}`",
        f"- Execute command: `{execute_command or 'not run'}`",
        f"- Input: `{summary['input']}`",
        f"- Limit: `{summary['limit']}`",
        f"- Selected records: `{summary['selected_record_count']}`",
        f"- Source files: `{len(summary.get('source_files') or [])}` listed in JSON summary",
        f"- Generated chunks: `{summary['generated_chunk_count']}`",
        f"- Estimated Qdrant points: `{summary['estimated_qdrant_points']}`",
        f"- Estimated embedding texts: `{summary['estimated_embedding_texts']}`",
        f"- Embedding model: `{summary['embedding_model']}`",
        f"- Vector dimension validation: `{summary['vector_dimension_validation']}`",
        f"- Qdrant collection: `{summary['collection_name']}`",
        f"- Collection point count before: `{qdrant['collection_point_count_before']}`",
        f"- Collection point count after: `{qdrant['collection_point_count_after']}`",
        f"- Inserted point count: `{qdrant['inserted_point_count']}`",
        f"- Qdrant write occurred: `{summary['qdrant_write_occurred']}`",
        f"- `nalus_live` before/after: `{qdrant['nalus_live_before']}` / `{qdrant['nalus_live_after']}`",
        (
            "- `nalus_stable_20260326` before/after: "
            f"`{qdrant['nalus_stable_20260326_before']}` / `{qdrant['nalus_stable_20260326_after']}`"
        ),
        (
            "- `nalus_live` target before/after: "
            f"`{qdrant['nalus_live_target_before']}` / `{qdrant['nalus_live_target_after']}`"
        ),
        f"- BM25 status: `{summary['bm25_status']}`",
        f"- Hybrid/RRF status: `{summary['hybrid_status']}`",
        f"- Payload metadata validation: `{summary['payload_metadata_validation']}`",
        f"- Production API touched: `{summary['production_api_touched']}`",
        f"- Aliases touched: `{summary['aliases_touched']}`",
        f"- Aliases changed by verification: `{qdrant['aliases_changed']}`",
        f"- Retrieval logic changed: `{summary['retrieval_logic_changed']}`",
        f"- Clarification gate changed: `{summary['clarification_gate_changed']}`",
        f"- Production safety touched: `{summary.get('production_safety', {}).get('production_touched')}`",
        f"- Final status: `{summary.get('final_status', 'not_run')}`",
        f"- Stage recommendation: `{summary.get('stage_recommendation', 'not_ready')}`",
        "",
        "## Qdrant Aliases",
        "",
    ]
    if qdrant.get("aliases_after") or qdrant.get("aliases_before"):
        aliases = qdrant.get("aliases_after") or qdrant.get("aliases_before") or []
        for item in aliases:
            lines.append(f"- `{item['alias_name']}` -> `{item['collection_name']}`")
    else:
        lines.append("- Not available.")

    lines.extend(
        [
            "",
            "## Sample Payloads",
            "",
        ]
    )
    if summary.get("sample_payloads"):
        for payload in summary["sample_payloads"]:
            lines.append(
                "- "
                f"doc=`{payload.get('source_document_id')}` "
                f"date=`{payload.get('decision_date')}` "
                f"chunk=`{payload.get('chunk_id')}` "
                f"snippet=\"{payload.get('text_snippet')}\""
            )
    else:
        lines.append("- Not available.")

    lines.extend(
        [
            "",
            f"## {query_heading}",
            "",
        ]
    )

    if not summary["smoke_queries"]:
        lines.append("- Not run.")
    else:
        for item in summary["smoke_queries"]:
            lines.extend(
                [
                    f"### `{item['query']}`",
                    "",
                    f"- Dense results from scoped collection: `{item['dense_all_from_smoke_collection']}`",
                    f"- Dense results from candidate collection: `{item['dense_all_from_candidate_collection']}`",
                    f"- All reported hits from candidate collection: `{item.get('all_hits_from_candidate_collection')}`",
                    f"- BM25 results: `{len(item.get('bm25_results') or [])}`",
                    f"- Hybrid results: `{len(item.get('hybrid_results') or [])}`",
                    f"- Qualitative relevance: `{item.get('qualitative_relevance', 'not_assessed')}`",
                    "",
                ]
            )
            for result in item["dense_results"]:
                lines.append(
                    "- dense "
                    f"score=`{result['score']}` doc=`{result['source_document_id']}` "
                    f"date=`{result['decision_date']}` snippet=\"{result['snippet']}\""
                )
            if item.get("hybrid_results"):
                lines.append("")
                for result in item["hybrid_results"]:
                    lines.append(
                        "- hybrid "
                        f"score=`{result['score']}` doc=`{result['source_document_id']}` "
                        f"date=`{result['decision_date']}` snippet=\"{result['snippet']}\""
                    )
            lines.append("")

    lines.extend(["## Failures / Warnings", ""])
    warnings = summary.get("warnings") or []
    failures = summary.get("failures") or []
    if not warnings and not failures:
        lines.append("- None.")
    for warning in warnings:
        lines.append(f"- WARNING: {warning}")
    for failure in failures:
        lines.append(f"- FAILURE: {failure}")
    lines.append("")
    return "\n".join(lines)


def _load_bge_m3_model() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: sentence_transformers. Install project production "
            "requirements in Docker; no fallback embedding is allowed."
        ) from exc

    device = os.getenv("BGE_M3_DEVICE", "cpu")
    return SentenceTransformer(BGE_M3_MODEL_NAME, device=device)


def _encode_chunks(
    model: Any,
    texts: list[str],
    *,
    batch_size: int,
    content_checksums: list[str] | None = None,
) -> list[list[float]]:
    from app.rag.retrieval.embedding_cache import (
        build_embedding_cache,
        embed_texts_with_cache,
        embedding_cache_config_from_env,
    )

    cache_build = build_embedding_cache(
        profile_name=PRODUCTION_RETRIEVAL_PROFILE,
        embedding_model=BGE_M3_MODEL_NAME,
        embedding_dim=BGE_M3_DIMENSION,
    )
    cache_config = embedding_cache_config_from_env(
        profile_name=PRODUCTION_RETRIEVAL_PROFILE,
        embedding_model=BGE_M3_MODEL_NAME,
        embedding_dim=BGE_M3_DIMENSION,
    )

    def encode_batch(batch: list[str]) -> list[list[float]]:
        encoded = model.encode(
            batch,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [_to_float_vector(vector) for vector in encoded]

    vectors = embed_texts_with_cache(
        texts=texts,
        content_checksums=content_checksums,
        cache=cache_build.cache,
        config=cache_config,
        encode_batch=encode_batch,
        batch_size=batch_size,
    )
    for index, vector in enumerate(vectors):
        if len(vector) != BGE_M3_DIMENSION:
            raise SafetyError(
                f"BGE-M3 embedding dimension mismatch at vector {index}: "
                f"{len(vector)} != {BGE_M3_DIMENSION}"
            )
    return vectors


def _encode_query(model: Any, query: str) -> list[float]:
    encoded = model.encode([query], batch_size=1, normalize_embeddings=True, show_progress_bar=False)
    return _to_float_vector(encoded[0])


def _qdrant_client(qdrant_url: str) -> Any:
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: qdrant_client. Install project production requirements in Docker."
        ) from exc
    return QdrantClient(url=qdrant_url, timeout=120, check_compatibility=False)


def _recreate_requested(args: argparse.Namespace) -> bool:
    return bool(
        args.recreate_smoke_collection
        or args.recreate_pilot_collection
        or args.recreate_full_collection
    )


def _record_read_only_qdrant_state(args: argparse.Namespace, summary: dict[str, Any], *, phase: str) -> None:
    try:
        client = _qdrant_client(args.qdrant_url)
        aliases = _aliases_snapshot(client)
        live_count = _count_collection(client, "nalus_live")
        stable_count = _count_collection(client, "nalus_stable_20260326")
        collection_count = _count_collection(client, args.collection_name)
    except Exception as exc:  # noqa: BLE001 - dry-run must not write just because Qdrant reads fail.
        summary["warnings"].append(f"Read-only Qdrant state unavailable during dry-run: {exc}")
        return

    summary["qdrant"][f"nalus_live_{phase}"] = live_count
    summary["qdrant"][f"nalus_stable_20260326_{phase}"] = stable_count
    summary["qdrant"]["collection_point_count_before"] = collection_count
    summary["qdrant"][f"aliases_{phase}"] = aliases
    summary["qdrant"][f"nalus_live_target_{phase}"] = _alias_target(aliases, "nalus_live")
    if phase == "before":
        summary["qdrant"]["aliases_changed"] = False


def _prepare_candidate_collection(
    client: Any,
    *,
    collection_name: str,
    recreate: bool,
    existing_count: int | None,
    mode: str,
    resume: bool = False,
) -> None:
    from qdrant_client.models import Distance, VectorParams

    exists = existing_count is not None
    if exists and recreate:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=BGE_M3_DIMENSION, distance=Distance.COSINE),
        )
        return

    if exists and existing_count and not resume:
        recreate_flag = f"--recreate-{mode}-collection"
        raise SafetyError(
            f"{mode.title()} collection {collection_name!r} already has {existing_count} points. "
            f"Pass {recreate_flag} with a guarded collection name to recreate safely, "
            "or resume from execute_checkpoint.json without recreate."
        )

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=BGE_M3_DIMENSION, distance=Distance.COSINE),
        )


def _upsert_chunks(
    client: Any,
    *,
    collection_name: str,
    chunks: list[SmokeChunk],
    vectors: list[list[float]],
) -> None:
    from qdrant_client.models import PointStruct

    points = [
        PointStruct(id=chunk.point_id, vector=vector, payload=chunk.payload)
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    for start in range(0, len(points), 64):
        client.upsert(collection_name=collection_name, points=points[start : start + 64])


def _run_smoke_queries(
    *,
    client: Any,
    collection_name: str,
    model: Any,
    chunks: list[SmokeChunk],
    top_k: int,
    mode: str,
) -> tuple[list[dict[str, Any]], str, str]:
    bm25_index = None
    bm25_status = "not_available"
    hybrid_status = "not_available"
    rag_eval = _load_rag_eval_components()
    if rag_eval is not None:
        RagEvalChunk, BenchmarkRetrievalConfig, Bm25ChunkIndex, reciprocal_rank_fusion, RagEvalRetrievalResult = rag_eval
        bm25_chunks = [
            RagEvalChunk(
                chunk_id=chunk.seq_id,
                chunk_text=chunk.text,
                source_id=SOURCE_ID,
                chunk_metadata={k: v for k, v in chunk.payload.items() if k != "text"},
            )
            for chunk in chunks
        ]
        retrieval_config = BenchmarkRetrievalConfig(modes=["bm25"], bm25_k1=1.5, bm25_b=0.75)
        bm25_index = Bm25ChunkIndex(chunks=bm25_chunks, retrieval_config=retrieval_config)
        bm25_status = "available"
        hybrid_status = "available_rrf"
    else:
        reciprocal_rank_fusion = None
        RagEvalRetrievalResult = None

    results: list[dict[str, Any]] = []
    for query in _queries_for_mode(mode, collection_name):
        dense_results = _dense_retrieve(
            client=client,
            collection_name=collection_name,
            model=model,
            query=query,
            top_k=top_k,
        )
        bm25_results = []
        hybrid_results = []
        if bm25_index is not None and reciprocal_rank_fusion is not None and RagEvalRetrievalResult is not None:
            bm25_response = bm25_index.retrieve(
                query=query,
                source_id=SOURCE_ID,
                top_k=top_k,
                collection_name=collection_name,
            )
            bm25_results = [_result_to_report(item, collection_name) for item in bm25_response.results]
            dense_for_rrf = [
                RagEvalRetrievalResult(
                    chunk_id=item["chunk_id"],
                    source_id=SOURCE_ID,
                    score=item["score"],
                    text=item["text"],
                    qdrant_collection=collection_name,
                    payload_metadata=item["payload_metadata"],
                )
                for item in dense_results
            ]
            hybrid_response = reciprocal_rank_fusion(
                [dense_for_rrf, bm25_response.results],
                top_k=top_k,
                rrf_k=60,
            )
            hybrid_results = [_result_to_report(item, collection_name) for item in hybrid_response.results]

        dense_report = [_dense_result_to_report(item, collection_name) for item in dense_results]
        results.append(
            {
                "query": query,
                "top_k": top_k,
                "collection": collection_name,
                "dense_results": dense_report,
                "bm25_results": bm25_results,
                "hybrid_results": hybrid_results,
                "dense_all_from_smoke_collection": all(
                    item["collection"] == collection_name for item in dense_report
                ),
                "dense_all_from_candidate_collection": all(
                    item["collection"] == collection_name for item in dense_report
                ),
                "all_hits_from_candidate_collection": _all_hits_from_collection(
                    collection_name, dense_report, bm25_results, hybrid_results
                ),
                "qualitative_relevance": _assess_query_relevance(
                    query, dense_report, bm25_results, hybrid_results
                ),
            }
        )
    return results, bm25_status, hybrid_status


def _queries_for_mode(mode: str, collection_name: str) -> tuple[str, ...]:
    if mode == "full" or "full" in collection_name.lower():
        return PILOT_QUERIES
    if mode == "pilot" or "pilot" in collection_name.lower():
        return PILOT_QUERIES
    return SMOKE_QUERIES


def _all_hits_from_collection(
    collection_name: str,
    *result_groups: list[dict[str, Any]],
) -> bool:
    return all(
        result.get("collection") == collection_name
        for group in result_groups
        for result in group
    )


def _assess_query_relevance(
    query: str,
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    hybrid_results: list[dict[str, Any]],
) -> str:
    combined = dense_results[:3] + bm25_results[:3] + hybrid_results[:3]
    if not combined:
        return "no_hits"
    query_terms = {
        term
        for term in re.findall(r"\w+", query.lower())
        if len(term) >= 4
    }
    if not query_terms:
        return "hits_returned_not_assessed"
    joined = " ".join(str(item.get("snippet") or "").lower() for item in combined)
    matched = sorted(term for term in query_terms if term in joined)
    if len(matched) >= min(2, len(query_terms)):
        return "plausible_top_hits_keyword_overlap"
    if matched:
        return "partial_keyword_overlap"
    return "semantic_or_weak_keyword_overlap_review_needed"


def _dense_retrieve(
    *,
    client: Any,
    collection_name: str,
    model: Any,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    vector = _encode_query(model, query)
    validate_vector_dimension([vector])
    response = client.query_points(collection_name=collection_name, query=vector, limit=top_k)
    dense_results: list[dict[str, Any]] = []
    for point in response.points:
        payload = dict(point.payload or {})
        dense_results.append(
            {
                "chunk_id": int(payload.get("chunk_id")),
                "score": float(point.score),
                "text": str(payload.get("text") or ""),
                "payload_metadata": {k: v for k, v in payload.items() if k != "text"},
                "collection": collection_name,
            }
        )
    return dense_results


def _load_rag_eval_components() -> tuple[Any, Any, Any, Any, Any] | None:
    try:
        from rag_eval.adapters.base import RagEvalChunk, RagEvalRetrievalResult
        from rag_eval.config import BenchmarkRetrievalConfig
        from rag_eval.retrieval.bm25 import Bm25ChunkIndex
        from rag_eval.retrieval.fusion import reciprocal_rank_fusion
    except ImportError:
        return None
    return (
        RagEvalChunk,
        BenchmarkRetrievalConfig,
        Bm25ChunkIndex,
        reciprocal_rank_fusion,
        RagEvalRetrievalResult,
    )


def _dense_result_to_report(item: dict[str, Any], collection_name: str) -> dict[str, Any]:
    metadata = item["payload_metadata"]
    return {
        "collection": collection_name,
        "source_document_id": metadata.get("source_document_id"),
        "decision_date": metadata.get("decision_date"),
        "score": round(float(item["score"]), 6),
        "snippet": _snippet(item["text"]),
        "chunk_id": item["chunk_id"],
    }


def _result_to_report(item: Any, collection_name: str) -> dict[str, Any]:
    metadata = dict(item.payload_metadata or {})
    return {
        "collection": item.qdrant_collection or collection_name,
        "source_document_id": metadata.get("source_document_id"),
        "decision_date": metadata.get("decision_date"),
        "score": round(float(item.score), 6),
        "snippet": _snippet(item.text),
        "chunk_id": item.chunk_id,
    }


def _sample_payloads(client: Any, collection_name: str) -> list[dict[str, Any]]:
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=3,
        with_payload=True,
        with_vectors=False,
    )
    samples: list[dict[str, Any]] = []
    for point in points:
        payload = dict(point.payload or {})
        samples.append(
            {
                "point_id": str(point.id),
                "source_document_id": payload.get("source_document_id"),
                "decision_date": payload.get("decision_date"),
                "case_reference": payload.get("case_reference"),
                "ecli": payload.get("ecli"),
                "chunk_id": payload.get("chunk_id"),
                "chunk_index": payload.get("chunk_index"),
                "document_id": payload.get("document_id"),
                "embedding_provider": payload.get("embedding_provider"),
                "embedding_model": payload.get("embedding_model"),
                "embedding_dimension": payload.get("embedding_dimension"),
                "retrieval_profile": payload.get("retrieval_profile"),
                "ingest_run_id": payload.get("ingest_run_id"),
                "qdrant_collection": payload.get("qdrant_collection"),
                "bm25_index_id": payload.get("bm25_index_id"),
                "source": payload.get("source"),
                "content_checksum": payload.get("content_checksum"),
                "origin_file": payload.get("origin_file"),
                "collection": collection_name,
                "text_snippet": _snippet(str(payload.get("text") or ""), limit=180),
            }
        )
    return samples


def _validate_sample_payloads(sample_payloads: list[dict[str, Any]]) -> bool:
    required = (
        "source_document_id",
        "decision_date",
        "chunk_id",
        "chunk_index",
        "document_id",
        "embedding_provider",
        "embedding_model",
        "embedding_dimension",
        "retrieval_profile",
        "ingest_run_id",
        "qdrant_collection",
        "bm25_index_id",
        "source",
        "content_checksum",
        "text_snippet",
    )
    if not sample_payloads:
        return False
    return all(all(payload.get(key) is not None for key in required) for payload in sample_payloads)


def _count_collection(client: Any, collection_name: str) -> int | None:
    try:
        return int(client.count(collection_name=collection_name).count)
    except Exception:  # noqa: BLE001 - missing collection must be represented as None.
        return None


def _aliases_snapshot(client: Any) -> list[dict[str, str]]:
    try:
        aliases = client.get_aliases().aliases
    except Exception:  # noqa: BLE001 - handled as unavailable snapshot.
        return []
    return sorted(
        (
            {"alias_name": str(alias.alias_name), "collection_name": str(alias.collection_name)}
            for alias in aliases
        ),
        key=lambda item: (item["alias_name"], item["collection_name"]),
    )


def _alias_target(aliases: list[dict[str, str]], alias_name: str) -> str | None:
    for alias in aliases:
        if alias.get("alias_name") == alias_name:
            return alias.get("collection_name")
    return None


def _records_from_items(items: list[Any], *, origin_file: str) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        full_text = str(item.get("full_text") or "").strip()
        identity = _record_identity(item)
        if not identity:
            continue
        source_document_id = str(item.get("ecli") or item.get("case_reference") or item.get("result_id") or identity)
        records.append(
            SourceRecord(
                identity=identity,
                source_document_id=source_document_id,
                case_reference=_clean_optional(item.get("case_reference")),
                ecli=_clean_optional(item.get("ecli")),
                decision_date=_clean_optional(item.get("decision_date")),
                detail_url=_clean_optional(item.get("detail_url")),
                text_url=_clean_optional(item.get("text_url")),
                full_text=full_text,
                origin_file=origin_file,
                raw=item,
            )
        )
    return records


def _record_identity(item: dict[str, Any]) -> str:
    for key in ("ecli", "case_reference", "detail_url", "text_url"):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    result_id = item.get("result_id")
    return str(result_id).strip() if result_id is not None else ""


def _record_rank(record: SourceRecord) -> tuple[int, int, int, str]:
    metadata_count = sum(
        1
        for value in (
            record.case_reference,
            record.ecli,
            record.decision_date,
            record.detail_url,
            record.text_url,
        )
        if value
    )
    return (int(bool(record.full_text.strip())), len(record.full_text), metadata_count, record.identity)


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    collapsed = "\n".join(lines)
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
    return collapsed.strip()


def _split_long_paragraph(paragraph: str, *, chunk_size: int) -> list[str]:
    if len(paragraph) <= chunk_size:
        return [paragraph]

    words = paragraph.split()
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        additional = len(word) + (1 if current else 0)
        if current and current_len + additional > chunk_size:
            parts.append(" ".join(current))
            current = [word]
            current_len = len(word)
            continue
        current.append(word)
        current_len += additional
    if current:
        parts.append(" ".join(current))
    return parts


def _with_overlap(previous: str, next_part: str, *, overlap_words: int, chunk_size: int) -> str:
    previous_words = previous.split()
    overlap = " ".join(previous_words[-overlap_words:]) if previous_words and overlap_words > 0 else ""
    if not overlap:
        return next_part
    candidate = f"{overlap}\n\n{next_part}"
    if len(candidate) <= chunk_size:
        return candidate
    return next_part


def _point_id(collection_name: str, identity: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{collection_name}:{identity}:{chunk_index}"))


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _clean_optional(value: Any) -> str | None:
    cleaned = str(value or "").strip()
    return cleaned or None


def _to_float_vector(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]


def _snippet(text: str, limit: int = 220) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def _format_command(argv: list[str]) -> str:
    return "python " + " ".join(argv)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_previous_dry_run(output_dir: Path) -> dict[str, Any] | None:
    path = resolve_project_path(output_dir) / "dry_run_summary.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_args(args)
        summary = run_execute(args) if args.execute else run_dry_run(args)
    except Exception as exc:  # noqa: BLE001 - CLI must render a report on failure where possible.
        failure_summary = {
            "generated_at": _utc_now(),
            "builder_version": BUILDER_VERSION,
            "script_path": "scripts/build_usoud_bge_m3_candidate.py",
            "mode": getattr(args, "mode", "unknown"),
            "action": "execute" if getattr(args, "execute", False) else "dry-run",
            "command": _format_command(sys.argv),
            "input": str(getattr(args, "source_batch", None) or getattr(args, "source_manifest", None)),
            "collection_name": getattr(args, "collection_name", "unknown"),
            "limit": getattr(args, "limit", None),
            "selected_record_count": 0,
            "generated_chunk_count": 0,
            "estimated_qdrant_points": 0,
            "estimated_embedding_texts": 0,
            "embedding_model": BGE_M3_MODEL_NAME,
            "expected_vector_dimension": BGE_M3_DIMENSION,
            "vector_dimension_validation": "failed_or_not_run",
            "qdrant": {
                "collection_point_count_before": None,
                "collection_point_count_after": None,
                "inserted_point_count": 0,
                "nalus_live_before": None,
                "nalus_live_after": None,
                "nalus_stable_20260326_before": None,
                "nalus_stable_20260326_after": None,
                "aliases_before": [],
                "aliases_after": [],
                "nalus_live_target_before": None,
                "nalus_live_target_after": None,
                "aliases_changed": None,
            },
            "qdrant_write_occurred": False,
            "payload_metadata_validation": "failed_or_not_run",
            "sample_payloads": [],
            "bm25_status": "not_run",
            "hybrid_status": "not_run",
            "smoke_queries": [],
            "warnings": [],
            "failures": [str(exc)],
            "production_api_touched": False,
            "retrieval_logic_changed": False,
            "clarification_gate_changed": False,
            "aliases_touched": False,
            "stage_recommendation": "not_ready",
            "final_status": "FAIL",
            "production_safety": {
                "production_touched": False,
                "aliases_changed": False,
                "nalus_live_changed": False,
                "nalus_stable_changed": False,
                "nalus_live_target_changed": False,
            },
            "source_files": [],
            "source_file_count": 0,
        }
        report_path = _report_path(args)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_report(failure_summary), encoding="utf-8")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {_report_path(args)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
