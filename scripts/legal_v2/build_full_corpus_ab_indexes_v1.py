#!/usr/bin/env python3
"""Build production-scale Legal v2 A/B indexes from the full Constitutional Court corpus.

Architecture (unchanged quality config):
  FAST     -> A hierarchical collection
  BALANCED -> B contextual collection + ColBERT (built separately)
  PRECISE  -> same B collection + CE-7 at query time

Uses exact validated chunkers from the Slice-4 pilot:
  A = legal_v2_hierarchical_chunker_v1
  B = legal_contextual_packed_v1

Does NOT overwrite pilot *_300 / pilot_600 collections.
Requires CUDA for embedding (hard-stop on CPU fallback).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.audit import PARSER_VERSION  # noqa: E402
from app.rag.legal_v2.ingest import index_builder as ib  # noqa: E402
from app.rag.legal_v2.ingest.chunkers import chunk_document_for_experiment  # noqa: E402
from app.rag.legal_v2.ingest.chunkers.contextual_packed_v1 import (  # noqa: E402
    ContextualPackedConfigV1,
)
from app.rag.legal_v2.ingest.chunkers.names import (  # noqa: E402
    CHUNKER_A_CURRENT,
    CHUNKER_B_CONTEXTUAL_PACKED_V1,
)
from app.rag.legal_v2.ingest.indexing import (  # noqa: E402
    LEGAL_V2_PROFILE,
    payload_for_child_chunk,
)
from app.rag.legal_v2.parser import parse_legal_document  # noqa: E402
from app.rag.legal_v2.sources import discover_source_documents_by_ids  # noqa: E402
from app.rag.retrieval.provenance import content_checksum  # noqa: E402
from scripts.legal_v2.run_chunking_ab_pilot_300_chunk_qa import _policy_hash  # noqa: E402

EXPECTED_PARSER = "legal-decision-parser.cz-courts.v8"
EXPECTED_B_POLICY_HASH = (
    "8fa196c58a9c537d311af6849582481ac195324c4f358634e81fcecb8f3f5898"
)
EXPECTED_EMBEDDING_DIM = 1024

# Production full-corpus names (explicit, distinct from pilot).
COLLECTION_A_FULL = "nalus_legal_paragraph_chunks_v2_chunk_ab_v8_a_current_full"
COLLECTION_B_FULL = "nalus_legal_paragraph_chunks_v2_chunk_ab_v8_b_contextual_full"
BM25_ID_A_FULL = "nalus_legal_paragraph_bm25_v2_chunk_ab_v8_a_current_full"
BM25_ID_B_FULL = "nalus_legal_paragraph_bm25_v2_chunk_ab_v8_b_contextual_full"

# Isolated 2k calibration names (fixed-batch baseline).
COLLECTION_A_CAL2K = "nalus_legal_paragraph_chunks_v2_chunk_ab_v8_a_current_cal2k"
COLLECTION_B_CAL2K = "nalus_legal_paragraph_chunks_v2_chunk_ab_v8_b_contextual_cal2k"
BM25_ID_A_CAL2K = "nalus_legal_paragraph_bm25_v2_chunk_ab_v8_a_current_cal2k"
BM25_ID_B_CAL2K = "nalus_legal_paragraph_bm25_v2_chunk_ab_v8_b_contextual_cal2k"

# Isolated 2k adaptive-batching profile (does not overwrite baseline cal2k).
COLLECTION_A_CAL2K_ADAPTIVE = (
    "nalus_legal_paragraph_chunks_v2_chunk_ab_v8_a_current_cal2k_adaptive"
)
COLLECTION_B_CAL2K_ADAPTIVE = (
    "nalus_legal_paragraph_chunks_v2_chunk_ab_v8_b_contextual_cal2k_adaptive"
)
BM25_ID_A_CAL2K_ADAPTIVE = (
    "nalus_legal_paragraph_bm25_v2_chunk_ab_v8_a_current_cal2k_adaptive"
)
BM25_ID_B_CAL2K_ADAPTIVE = (
    "nalus_legal_paragraph_bm25_v2_chunk_ab_v8_b_contextual_cal2k_adaptive"
)

PROTECTED_COLLECTIONS = {
    "nalus",
    "nalus_live",
    "nalus_legal_paragraph_chunks_v2",
    "nalus_legal_paragraph_chunks_v2_pilot_600",
    "nalus_legal_paragraph_chunks_v2_chunk_ab_v8_a_current_300",
    "nalus_legal_paragraph_chunks_v2_chunk_ab_v8_b_contextual_300",
}

DEFAULT_BATCHES = PROJECT_ROOT.parent / "nalus-scraper" / "batches"
DEFAULT_OUT = PROJECT_ROOT / "artifacts" / "legal_v2" / "full_corpus_build_v1"
DEFAULT_STORAGE = PROJECT_ROOT.parent / "nalus-scraper" / "storage" / "rag" / "bm25"


def pack_indices_by_token_budget(
    token_counts: list[int],
    *,
    token_budget: int,
    max_items: int,
    length_bucket_edges: tuple[int, ...] = (256, 512, 1024, 2048, 4096, 8192),
) -> list[list[int]]:
    """Pack original indices into length-aware adaptive encode batches.

    VRAM safety notes:
    - Transformers pad to the longest sequence in a batch, so cost tracks
      ``batch_items * max_seq_len`` (padded tokens), not only ``sum(tokens)``.
    - Attention cost grows strongly with sequence length; mixing extreme-long
      with short texts in one batch is avoided via length buckets.

    Each returned pack is sorted by original index so encode outputs restore
    cleanly to caller order.
    """
    if token_budget < 1:
        raise ValueError("token_budget must be >= 1")
    if max_items < 1:
        raise ValueError("max_items must be >= 1")
    if not token_counts:
        return []

    def _bucket(tokens: int) -> int:
        for edge in length_bucket_edges:
            if tokens <= edge:
                return int(edge)
        return int(length_bucket_edges[-1]) if length_bucket_edges else int(tokens)

    # Group by similar post-truncation length, then pack within each bucket.
    by_bucket: dict[int, list[int]] = {}
    for idx, raw in enumerate(token_counts):
        tokens = max(1, int(raw))
        by_bucket.setdefault(_bucket(tokens), []).append(idx)

    batches: list[list[int]] = []
    for bucket_edge in sorted(by_bucket.keys()):
        # Longest-first inside the bucket keeps max_seq_len stable within a pack.
        order = sorted(
            by_bucket[bucket_edge],
            key=lambda idx: (-int(token_counts[idx]), idx),
        )
        current: list[int] = []
        current_sum = 0
        current_max = 0
        for idx in order:
            tokens = max(1, int(token_counts[idx]))
            next_n = len(current) + 1
            next_max = max(current_max, tokens) if current else tokens
            next_sum = current_sum + tokens
            next_padded = next_n * next_max
            fits_items = next_n <= max_items
            fits_sum = next_sum <= token_budget
            fits_padded = next_padded <= token_budget
            # Single over-budget item still forms its own pack (model truncates).
            if current and (not fits_items or not fits_sum or not fits_padded):
                batches.append(sorted(current))
                current = []
                current_sum = 0
                current_max = 0
                next_n = 1
                next_max = tokens
                next_sum = tokens
                next_padded = tokens
            current.append(idx)
            current_sum = next_sum
            current_max = next_max
        if current:
            batches.append(sorted(current))
    return batches


def _pack_stats(token_counts: list[int], pack: list[int]) -> dict[str, int]:
    lengths = [max(1, int(token_counts[idx])) for idx in pack]
    max_seq = max(lengths) if lengths else 0
    return {
        "batch_items": len(pack),
        "sum_tokens": int(sum(lengths)),
        "max_sequence_length": int(max_seq),
        "padded_tokens": int(len(pack) * max_seq),
    }


class EmbedProfileSink:
    """Accumulates encode-scheduler profiling counters (warmup excluded)."""

    def __init__(self) -> None:
        self.tokenization_seconds = 0.0
        self.scheduler_overhead_seconds = 0.0
        self.gpu_encode_seconds = 0.0
        self.end_to_end_embed_seconds = 0.0
        self.post_truncation_tokens: list[int] = []
        self.batch_sizes: list[int] = []
        self.tokens_per_batch: list[int] = []
        self.max_seq_len_per_batch: list[int] = []
        self.padded_tokens_per_batch: list[int] = []
        self.peak_vram_allocated_mb = 0.0
        self.peak_vram_reserved_mb = 0.0

    def note_vram(self) -> None:
        try:
            import torch

            if not torch.cuda.is_available():
                return
            allocated = torch.cuda.memory_allocated(0) / (1024**2)
            reserved = torch.cuda.memory_reserved(0) / (1024**2)
            self.peak_vram_allocated_mb = max(self.peak_vram_allocated_mb, allocated)
            self.peak_vram_reserved_mb = max(self.peak_vram_reserved_mb, reserved)
        except Exception:  # noqa: BLE001
            return

    def as_dict(self) -> dict[str, Any]:
        tokens = [float(x) for x in self.post_truncation_tokens]
        batch_sizes = [float(x) for x in self.batch_sizes]
        tokens_per_batch = [float(x) for x in self.tokens_per_batch]
        max_seq = [float(x) for x in self.max_seq_len_per_batch]
        padded = [float(x) for x in self.padded_tokens_per_batch]
        encode_s = self.gpu_encode_seconds or 1e-9
        e2e_s = self.end_to_end_embed_seconds or encode_s
        token_sum = int(sum(self.post_truncation_tokens))
        size_hist = Counter(self.batch_sizes)
        return {
            "tokenization_seconds": round(self.tokenization_seconds, 3),
            "scheduler_overhead_seconds": round(self.scheduler_overhead_seconds, 3),
            "gpu_encode_seconds": round(self.gpu_encode_seconds, 3),
            "end_to_end_embed_seconds": round(self.end_to_end_embed_seconds, 3),
            "notes": (
                "tokenization_seconds is an extra scheduler tokenizer pass; "
                "SentenceTransformer.encode() tokenizes again internally. "
                "Decide full-build readiness on end-to-end embed throughput, "
                "not GPU encode alone."
            ),
            "post_truncation_token_sum": token_sum,
            "post_truncation_tokens": {
                "mean": statistics.fmean(tokens) if tokens else 0.0,
                "p50": _percentile(tokens, 50),
                "p95": _percentile(tokens, 95),
                "max": max(tokens) if tokens else 0.0,
            },
            "gpu_encode_tokens_per_sec": round(token_sum / encode_s, 3),
            "end_to_end_tokens_per_sec": round(token_sum / e2e_s, 3),
            # Back-compat alias used by earlier summaries.
            "effective_tokens_per_sec": round(token_sum / e2e_s, 3),
            "batch_size_histogram": {
                str(k): v for k, v in sorted(size_hist.items(), key=lambda item: item[0])
            },
            "batch_sizes": {
                "count": len(batch_sizes),
                "mean": statistics.fmean(batch_sizes) if batch_sizes else 0.0,
                "p50": _percentile(batch_sizes, 50),
                "p95": _percentile(batch_sizes, 95),
                "max": max(batch_sizes) if batch_sizes else 0.0,
            },
            "sum_tokens_per_batch": {
                "mean": statistics.fmean(tokens_per_batch) if tokens_per_batch else 0.0,
                "p50": _percentile(tokens_per_batch, 50),
                "p95": _percentile(tokens_per_batch, 95),
                "max": max(tokens_per_batch) if tokens_per_batch else 0.0,
            },
            "tokens_per_batch": {
                "mean": statistics.fmean(tokens_per_batch) if tokens_per_batch else 0.0,
                "p50": _percentile(tokens_per_batch, 50),
                "p95": _percentile(tokens_per_batch, 95),
                "max": max(tokens_per_batch) if tokens_per_batch else 0.0,
            },
            "max_sequence_length_per_batch": {
                "mean": statistics.fmean(max_seq) if max_seq else 0.0,
                "p50": _percentile(max_seq, 50),
                "p95": _percentile(max_seq, 95),
                "max": max(max_seq) if max_seq else 0.0,
            },
            "padded_tokens_per_batch": {
                "mean": statistics.fmean(padded) if padded else 0.0,
                "p50": _percentile(padded, 50),
                "p95": _percentile(padded, 95),
                "max": max(padded) if padded else 0.0,
            },
            "peak_vram_allocated_mb": round(self.peak_vram_allocated_mb, 2),
            "peak_vram_reserved_mb": round(self.peak_vram_reserved_mb, 2),
        }


class GpuBgeM3Embedder:
    """BGE-M3 embedder pinned to an explicit device (cuda required for full builds)."""

    def __init__(self, *, model_path: str, device: str, encode_batch_size: int) -> None:
        self._model_path = model_path
        self._device = device
        self._encode_batch_size = max(1, int(encode_batch_size))
        self._model: Any | None = None
        self.warmup_seconds: float | None = None
        self.max_seq_length: int | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from sentence_transformers import SentenceTransformer

        if self._device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("HARD_STOP: CUDA requested but torch.cuda.is_available() is False")
        started = time.perf_counter()
        self._model = SentenceTransformer(
            self._model_path,
            device=self._device,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.max_seq_length = int(getattr(self._model, "max_seq_length", 8192) or 8192)
        # Warmup excluded from throughput rates.
        _ = self._model.encode(
            ["warmup"],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.warmup_seconds = time.perf_counter() - started
        actual = str(self._model.device)
        if self._device == "cuda" and "cuda" not in actual.lower():
            raise SystemExit(f"HARD_STOP: expected cuda embedder device, got {actual!r}")

    def _empty_cache(self) -> None:
        try:
            import torch

            if self._device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            return

    def _vectors_from_encoded(self, encoded: Any) -> list[list[float]]:
        vectors: list[list[float]] = []
        for vector in encoded:
            if hasattr(vector, "tolist"):
                vector = vector.tolist()
            vectors.append([float(value) for value in vector])
        for index, vector in enumerate(vectors):
            if len(vector) != LEGAL_V2_PROFILE.embedding_dimension:
                raise RuntimeError(
                    f"BGE-M3 dimension mismatch at {index}: "
                    f"{len(vector)} != {LEGAL_V2_PROFILE.embedding_dimension}"
                )
        return vectors

    def count_tokens_post_truncation(self, texts: list[str]) -> list[int]:
        """Token lengths after the model max_seq_length truncation bound."""
        self.load()
        assert self._model is not None
        tokenizer = self._model.tokenizer
        max_len = int(self.max_seq_length or getattr(self._model, "max_seq_length", 8192) or 8192)
        counts: list[int] = []
        # Tokenize one-by-one to avoid a second padded tensor allocation competing with encode VRAM.
        for text in texts:
            encoded = tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )
            ids = encoded["input_ids"]
            counts.append(min(len(ids), max_len))
        return counts

    def embed_texts(
        self,
        texts: list[str],
        *,
        profile: EmbedProfileSink | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        self.load()
        assert self._model is not None
        e2e_started = time.perf_counter()
        self._empty_cache()
        token_counts: list[int] = []
        if profile is not None:
            t0 = time.perf_counter()
            token_counts = self.count_tokens_post_truncation(texts)
            profile.tokenization_seconds += time.perf_counter() - t0
            profile.scheduler_overhead_seconds += time.perf_counter() - t0
            profile.post_truncation_tokens.extend(token_counts)
            # Fixed mode may internally re-split; record caller-visible batch only.
            stats = _pack_stats(token_counts, list(range(len(texts))))
            profile.batch_sizes.append(stats["batch_items"])
            profile.tokens_per_batch.append(stats["sum_tokens"])
            profile.max_seq_len_per_batch.append(stats["max_sequence_length"])
            profile.padded_tokens_per_batch.append(stats["padded_tokens"])
        t1 = time.perf_counter()
        encoded = self._model.encode(
            texts,
            batch_size=self._encode_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if profile is not None:
            profile.gpu_encode_seconds += time.perf_counter() - t1
            profile.end_to_end_embed_seconds += time.perf_counter() - e2e_started
            profile.note_vram()
        return self._vectors_from_encoded(encoded)

    def embed_texts_adaptive(
        self,
        texts: list[str],
        *,
        token_budget: int,
        max_items: int,
        profile: EmbedProfileSink | None = None,
    ) -> list[list[float]]:
        """Encode with length-aware token-budget packing; return vectors in caller order.

        Each adaptive pack is one SentenceTransformer.encode() call with
        ``batch_size=len(pack)`` so profiling matches the scheduler packs.
        """
        if not texts:
            return []
        self.load()
        assert self._model is not None
        e2e_started = time.perf_counter()
        t0 = time.perf_counter()
        token_counts = self.count_tokens_post_truncation(texts)
        tokenize_s = time.perf_counter() - t0
        if profile is not None:
            profile.tokenization_seconds += tokenize_s

        t_pack = time.perf_counter()
        packs = pack_indices_by_token_budget(
            token_counts,
            token_budget=token_budget,
            max_items=max_items,
        )
        pack_s = time.perf_counter() - t_pack
        if profile is not None:
            profile.scheduler_overhead_seconds += tokenize_s + pack_s
            profile.post_truncation_tokens.extend(token_counts)

        out: list[list[float] | None] = [None] * len(texts)
        for pack in packs:
            batch_texts = [texts[idx] for idx in pack]
            pack_stats = _pack_stats(token_counts, pack)
            if profile is not None:
                profile.batch_sizes.append(pack_stats["batch_items"])
                profile.tokens_per_batch.append(pack_stats["sum_tokens"])
                profile.max_seq_len_per_batch.append(pack_stats["max_sequence_length"])
                profile.padded_tokens_per_batch.append(pack_stats["padded_tokens"])
            self._empty_cache()
            # One scheduler pack == one actual encode batch (no second fixed split).
            t1 = time.perf_counter()
            encoded = self._model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            if profile is not None:
                profile.gpu_encode_seconds += time.perf_counter() - t1
                profile.note_vram()
            vectors = self._vectors_from_encoded(encoded)
            for local_i, original_idx in enumerate(pack):
                out[original_idx] = vectors[local_i]
        if profile is not None:
            profile.end_to_end_embed_seconds += time.perf_counter() - e2e_started
        if any(vector is None for vector in out):
            raise RuntimeError("adaptive embedder failed to restore all vectors")
        return [vector for vector in out if vector is not None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_head() -> str:
    for key in ("LEGAL_V2_GIT_COMMIT", "GIT_COMMIT", "SOURCE_COMMIT"):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _resolve_model_path() -> str:
    env = (os.environ.get("EMBEDDING_MODEL_NAME") or "").strip()
    if env and Path(env).exists():
        return env
    for candidate in (
        "/root/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/"
        "5617a9f61b028005a4858fdac845db406aefb181",
        "/app/models/BAAI/bge-m3",
        str(PROJECT_ROOT / "models" / "BAAI" / "bge-m3"),
        "BAAI/bge-m3",
    ):
        if candidate == "BAAI/bge-m3" or Path(candidate).exists():
            return candidate
    raise SystemExit("HARD_STOP: cannot resolve BGE-M3 model path")


def _gate_quality_config(*, device: str) -> dict[str, Any]:
    if PARSER_VERSION != EXPECTED_PARSER:
        raise SystemExit(f"HARD_STOP: parser={PARSER_VERSION} != {EXPECTED_PARSER}")
    if CHUNKER_A_CURRENT != "legal_v2_hierarchical_chunker_v1":
        raise SystemExit(f"HARD_STOP: A chunker changed: {CHUNKER_A_CURRENT}")
    if CHUNKER_B_CONTEXTUAL_PACKED_V1 != "legal_contextual_packed_v1":
        raise SystemExit(f"HARD_STOP: B chunker changed: {CHUNKER_B_CONTEXTUAL_PACKED_V1}")
    b_policy = _policy_hash(ContextualPackedConfigV1())
    if b_policy != EXPECTED_B_POLICY_HASH:
        raise SystemExit(f"HARD_STOP: B policy hash changed: {b_policy}")
    if LEGAL_V2_PROFILE.embedding_dimension != EXPECTED_EMBEDDING_DIM:
        raise SystemExit("HARD_STOP: embedding dimension changed")
    if device != "cuda":
        raise SystemExit("HARD_STOP: full/calibration builds require --device cuda")
    return {
        "parser_version": PARSER_VERSION,
        "a_chunker_version": CHUNKER_A_CURRENT,
        "b_chunker_version": CHUNKER_B_CONTEXTUAL_PACKED_V1,
        "b_policy_hash": b_policy,
        "embedding_model": LEGAL_V2_PROFILE.embedding_model,
        "embedding_dimension": LEGAL_V2_PROFILE.embedding_dimension,
        "device": device,
    }


def _side_names(mode: str, side: str) -> tuple[str, str]:
    if mode == "full":
        if side == "A":
            return COLLECTION_A_FULL, BM25_ID_A_FULL
        return COLLECTION_B_FULL, BM25_ID_B_FULL
    if mode == "cal2k":
        if side == "A":
            return COLLECTION_A_CAL2K, BM25_ID_A_CAL2K
        return COLLECTION_B_CAL2K, BM25_ID_B_CAL2K
    if mode == "cal2k_adaptive":
        if side == "A":
            return COLLECTION_A_CAL2K_ADAPTIVE, BM25_ID_A_CAL2K_ADAPTIVE
        return COLLECTION_B_CAL2K_ADAPTIVE, BM25_ID_B_CAL2K_ADAPTIVE
    raise ValueError(f"unknown mode={mode}")


def _artifact_subdir(mode: str) -> str:
    if mode == "cal2k":
        return "throughput_2k"
    if mode == "cal2k_adaptive":
        return "throughput_2k_adaptive"
    return ""


def _refuse_protected(collection: str, *, mode: str | None = None) -> None:
    if collection in PROTECTED_COLLECTIONS or collection.startswith("nalus_stable_"):
        raise SystemExit(f"HARD_STOP: refusing protected collection {collection}")
    if collection.endswith("_300") or "pilot_600" in collection:
        raise SystemExit(f"HARD_STOP: refusing pilot-like collection name {collection}")
    # Adaptive profile must never write into baseline stratified cal2k collections.
    if mode == "cal2k_adaptive" and collection in {COLLECTION_A_CAL2K, COLLECTION_B_CAL2K}:
        raise SystemExit(
            f"HARD_STOP: cal2k_adaptive refusing baseline collection {collection}"
        )


def _select_cal2k_ids(
    meta_path: Path,
    *,
    count: int,
    seed: int = 20260814,
) -> tuple[list[str], dict[str, Any]]:
    """Deterministic length-stratified sample for throughput calibration.

    Length quantiles (by text_len chars):
      0-20, 20-40, 40-60, 60-80, 80-95, 95-100

    Within each stratum, also spread across years when possible.
    """
    rows: list[dict[str, Any]] = []
    with meta_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if len(rows) < count:
        raise SystemExit(f"HARD_STOP: only {len(rows)} eligible docs for cal2k={count}")

    ordered = sorted(
        rows,
        key=lambda item: (int(item.get("text_len") or 0), str(item.get("document_id") or "")),
    )
    n = len(ordered)
    # Inclusive quantile bands as fractions of the ordered corpus.
    bands: list[tuple[str, float, float, float]] = [
        ("q00_20", 0.00, 0.20, 0.20),
        ("q20_40", 0.20, 0.40, 0.20),
        ("q40_60", 0.40, 0.60, 0.20),
        ("q60_80", 0.60, 0.80, 0.20),
        ("q80_95", 0.80, 0.95, 0.15),
        ("q95_100", 0.95, 1.00, 0.05),
    ]
    # Allocate target counts proportional to band weight; fix rounding on the last band.
    targets: list[int] = []
    allocated = 0
    for index, (_name, _lo, _hi, weight) in enumerate(bands):
        if index == len(bands) - 1:
            targets.append(count - allocated)
        else:
            take = int(round(count * weight))
            targets.append(take)
            allocated += take

    selected: list[str] = []
    seen: set[str] = set()
    stratum_stats: list[dict[str, Any]] = []

    for (name, lo, hi, _weight), target in zip(bands, targets, strict=True):
        start = int(lo * n)
        end = int(hi * n) if hi < 1.0 else n
        if end <= start:
            end = min(n, start + 1)
        bucket = ordered[start:end]
        # Deterministic within-stratum order: hash(seed, id), then soft year spread.
        bucket_sorted = sorted(
            bucket,
            key=lambda item: (
                hashlib.sha256(
                    f"{seed}:{name}:{item['document_id']}".encode("utf-8")
                ).hexdigest(),
                int(item.get("year") or 0),
                item["document_id"],
            ),
        )
        # Take evenly spaced indices so we cover the stratum's length range, not only the head.
        picked: list[dict[str, Any]] = []
        if target > 0 and bucket_sorted:
            if target >= len(bucket_sorted):
                picked = list(bucket_sorted)
            else:
                step = len(bucket_sorted) / float(target)
                for i in range(target):
                    idx = min(len(bucket_sorted) - 1, int(i * step))
                    picked.append(bucket_sorted[idx])
        lens = [int(item.get("text_len") or 0) for item in picked]
        years = sorted({int(item.get("year") or 0) for item in picked})
        for item in picked:
            doc_id = item["document_id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            selected.append(doc_id)
        stratum_stats.append(
            {
                "stratum": name,
                "quantile_lo": lo,
                "quantile_hi": hi,
                "corpus_slice_start": start,
                "corpus_slice_end": end,
                "corpus_slice_size": len(bucket),
                "target": target,
                "selected": len(picked),
                "text_len_min": min(lens) if lens else None,
                "text_len_max": max(lens) if lens else None,
                "text_len_mean": (sum(lens) / len(lens)) if lens else None,
                "years_represented": years,
            }
        )

    # Top up if duplicates across strata reduced the count.
    if len(selected) < count:
        for item in ordered:
            doc_id = item["document_id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            selected.append(doc_id)
            if len(selected) >= count:
                break

    selected = selected[:count]
    selected.sort(
        key=lambda doc_id: hashlib.sha256(f"{seed}:order:{doc_id}".encode("utf-8")).hexdigest()
    )
    selected_meta = {
        row["document_id"]: row for row in rows if row["document_id"] in set(selected)
    }
    selected_lens = [int(selected_meta[doc_id].get("text_len") or 0) for doc_id in selected]
    sampling_report = {
        "schema": "legal_v2_cal2k_length_stratified_v2",
        "seed": seed,
        "requested_count": count,
        "selected_count": len(selected),
        "bands": stratum_stats,
        "selected_text_len": {
            "min": min(selected_lens) if selected_lens else 0,
            "max": max(selected_lens) if selected_lens else 0,
            "mean": (sum(selected_lens) / len(selected_lens)) if selected_lens else 0.0,
            "p50": _percentile([float(x) for x in selected_lens], 50),
            "p95": _percentile([float(x) for x in selected_lens], 95),
        },
    }
    return selected, sampling_report


def _load_ordered_ids(path: Path, *, limit: int | None = None) -> list[str]:
    ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit is not None:
        ids = ids[:limit]
    return ids


def _chunk_document_payloads(
    *,
    document: Any,
    chunker_version: str,
    collection: str,
    bm25_id: str,
    inventory_id: str,
    ingest_run_id: str,
) -> list[dict[str, Any]]:
    court = str(
        (document.metadata or {}).get("court")
        or (document.metadata or {}).get("court_name")
        or ""
    )
    parsed = parse_legal_document(
        document_id=document.document_id,
        text=document.text,
        metadata={
            **dict(document.metadata or {}),
            "court": court,
            "ecli": document.document_id
            if str(document.document_id).upper().startswith("ECLI:")
            else (document.metadata or {}).get("ecli"),
        },
    )
    result = chunk_document_for_experiment(parsed, chunker_version=chunker_version)
    content_hash = content_checksum(parsed.normalized_text)
    payloads: list[dict[str, Any]] = []
    for child in result.child_chunks:
        enriched = replace(
            child,
            metadata={
                **dict(child.metadata or {}),
                "ecli": (
                    parsed.metadata.get("ecli")
                    or (
                        child.document_id
                        if str(child.document_id).upper().startswith("ECLI:")
                        else None
                    )
                ),
                "court": court or None,
                "parser_version": PARSER_VERSION,
                "chunker_version": chunker_version,
                "document_content_hash": content_hash,
                "inventory_id": inventory_id,
                "ingest_run_id": ingest_run_id,
                "source": document.source or "constitutional",
                "case_reference": parsed.metadata.get("case_reference")
                or parsed.metadata.get("spisova_znacka"),
                "decision_date": parsed.metadata.get("decision_date")
                or parsed.metadata.get("date"),
                "document_type": parsed.metadata.get("document_type")
                or parsed.metadata.get("decision_form"),
                "language": "cs",
            },
        )
        payload = payload_for_child_chunk(
            enriched,
            qdrant_collection=collection,
            bm25_index_id=bm25_id,
        )
        payload["inventory_id"] = inventory_id
        payloads.append(payload)
    return payloads


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1))))
    return float(ordered[idx])


def _bm25_chunk_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with sqlite3.connect(path) as connection:
        return {str(row[0]) for row in connection.execute("SELECT chunk_id FROM bm25_chunks")}


def _qdrant_point_count(client: Any, collection: str) -> int:
    info = client.get_collection(collection_name=collection)
    return int(getattr(info, "points_count", None) or getattr(info, "points_count", 0) or 0)


def _gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_cuda_is_available": False,
        "gpu_name": None,
        "dtype": None,
    }
    try:
        import torch

        info["torch_cuda_is_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["dtype"] = str(torch.get_default_dtype())
            info["vram_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024**2), 2)
            info["vram_reserved_mb"] = round(torch.cuda.memory_reserved(0) / (1024**2), 2)
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{exc.__class__.__name__}:{exc}"
    return info


def build_side(
    *,
    side: str,
    mode: str,
    document_ids: list[str],
    batches_dir: Path,
    out_dir: Path,
    storage_bm25_dir: Path,
    qdrant_url: str,
    device: str,
    batch_size: int,
    document_batch_size: int,
    force_recreate: bool,
    inventory_id: str,
    embed_scheduler: str = "fixed",
    token_budget: int = 24576,
    max_batch_items: int = 64,
) -> dict[str, Any]:
    quality = _gate_quality_config(device=device)
    collection, bm25_id = _side_names(mode, side)
    _refuse_protected(collection, mode=mode)
    print(
        json.dumps(
            {
                "event": "full_corpus_target_names",
                "side": side,
                "mode": mode,
                "collection": collection,
                "bm25_index_id": bm25_id,
                "embed_scheduler": embed_scheduler,
                "token_budget": token_budget,
                "max_batch_items": max_batch_items,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    subdir = _artifact_subdir(mode)
    side_dir = out_dir / (subdir if subdir else f"{side.lower()}_build")
    side_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = side_dir / f"checkpoint_{side}_{mode}.json"
    failures_path = side_dir / "failures.jsonl"

    documents = discover_source_documents_by_ids(document_ids, batches_dir=batches_dir)
    by_id = {doc.document_id: doc for doc in documents}
    missing = [doc_id for doc_id in document_ids if doc_id not in by_id]
    if missing:
        raise SystemExit(
            f"HARD_STOP: {len(missing)} document_ids not found in batches "
            f"(sample={missing[:5]})"
        )
    ordered_docs = [by_id[doc_id] for doc_id in document_ids]

    chunker_version = CHUNKER_A_CURRENT if side == "A" else CHUNKER_B_CONTEXTUAL_PACKED_V1
    ingest_run_id = f"full_corpus_{mode}_{side}_{_git_head()[:12]}"
    bm25_path = storage_bm25_dir / f"{bm25_id}.sqlite"

    from qdrant_client import QdrantClient

    client = QdrantClient(url=qdrant_url, timeout=120)
    # Document-level resume is owned by this script's checkpoint file.
    # Do not set LegalV2BuildConfig.resume=True (that blocks first create).
    config = ib.LegalV2BuildConfig(
        collection_name=collection,
        bm25_index_id=bm25_id,
        bm25_path=bm25_path,
        output_dir=side_dir,
        recreate_collection=force_recreate,
        overwrite_bm25=force_recreate,
        resume=False,
        allow_existing_collection=not force_recreate,
        batch_size=batch_size,
        document_batch_size=document_batch_size,
        checkpoint_path=checkpoint_path,
        source_selection={
            "inventory_id": inventory_id,
            "mode": mode,
            "side": side,
            "document_count": len(ordered_docs),
            "embed_scheduler": embed_scheduler,
            "token_budget": token_budget,
            "max_batch_items": max_batch_items,
        },
    )
    config.validate()

    completed_ids: set[str] = set()
    stats: dict[str, Any] = {
        "documents_attempted": 0,
        "documents_succeeded": 0,
        "documents_failed": 0,
        "total_chunks": 0,
        "chunks_per_doc": [],
        "embed_seconds": 0.0,
        "upsert_seconds": 0.0,
        "chunk_seconds": 0.0,
        "embed_batches": 0,
        "upsert_points": 0,
    }
    if checkpoint_path.exists() and not force_recreate:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        completed_ids = set(payload.get("completed_document_ids") or [])
        loaded_stats = dict(payload.get("stats") or {})
        tail = loaded_stats.pop("chunks_per_doc_tail", None)
        stats.update(loaded_stats)
        if isinstance(stats.get("chunks_per_doc"), list):
            pass
        elif isinstance(tail, list):
            stats["chunks_per_doc"] = list(tail)
        else:
            stats["chunks_per_doc"] = []

    if force_recreate:
        completed_ids = set()
        stats = {
            "documents_attempted": 0,
            "documents_succeeded": 0,
            "documents_failed": 0,
            "total_chunks": 0,
            "chunks_per_doc": [],
            "embed_seconds": 0.0,
            "upsert_seconds": 0.0,
            "chunk_seconds": 0.0,
            "embed_batches": 0,
            "upsert_points": 0,
        }
        if failures_path.exists():
            failures_path.unlink()
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    ib._prepare_collection(client, config)
    ib._prepare_bm25_sidecar(
        bm25_path,
        overwrite=force_recreate,
        resume=(not force_recreate and bm25_path.exists()),
    )

    model_path = _resolve_model_path()
    embedder = GpuBgeM3Embedder(
        model_path=model_path, device=device, encode_batch_size=batch_size
    )
    profile = EmbedProfileSink()
    gpu_before = _gpu_info()
    embedder.load()
    gpu_after_warmup = _gpu_info()

    started = time.perf_counter()
    remaining_docs = [doc for doc in ordered_docs if doc.document_id not in completed_ids]
    total = len(ordered_docs)

    for start in range(0, len(remaining_docs), document_batch_size):
        batch_docs = remaining_docs[start : start + document_batch_size]
        batch_payloads: list[dict[str, Any]] = []
        batch_success: list[str] = []
        for document in batch_docs:
            stats["documents_attempted"] = int(stats["documents_attempted"]) + 1
            try:
                t0 = time.perf_counter()
                payloads = _chunk_document_payloads(
                    document=document,
                    chunker_version=chunker_version,
                    collection=collection,
                    bm25_id=bm25_id,
                    inventory_id=inventory_id,
                    ingest_run_id=ingest_run_id,
                )
                stats["chunk_seconds"] = float(stats["chunk_seconds"]) + (
                    time.perf_counter() - t0
                )
                if not payloads:
                    raise RuntimeError("no_chunks_produced")
                batch_payloads.extend(payloads)
                batch_success.append(document.document_id)
                stats["documents_succeeded"] = int(stats["documents_succeeded"]) + 1
                stats["chunks_per_doc"].append(len(payloads))
                stats["total_chunks"] = int(stats["total_chunks"]) + len(payloads)
            except Exception as exc:  # noqa: BLE001
                stats["documents_failed"] = int(stats["documents_failed"]) + 1
                with failures_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "document_id": document.document_id,
                                "side": side,
                                "error": f"{exc.__class__.__name__}:{exc}",
                                "at": _utc_now(),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        if batch_payloads:
            existing = set()
            # Identity check against already-written chunk ids in this run's BM25 when resuming.
            if completed_ids:
                existing = _bm25_chunk_ids(bm25_path)
            ib._validate_payload_identity(batch_payloads, existing_chunk_ids=existing)
            texts = [p["text"] for p in batch_payloads]
            t_embed = time.perf_counter()
            before_batches = len(profile.batch_sizes)
            if embed_scheduler == "adaptive":
                vectors = embedder.embed_texts_adaptive(
                    texts,
                    token_budget=token_budget,
                    max_items=max_batch_items,
                    profile=profile,
                )
            else:
                vectors = embedder.embed_texts(texts, profile=profile)
            stats["embed_batches"] = int(stats["embed_batches"]) + max(
                1, len(profile.batch_sizes) - before_batches
            )
            stats["embed_seconds"] = float(stats["embed_seconds"]) + (
                time.perf_counter() - t_embed
            )
            ib._validate_vectors(vectors, LEGAL_V2_PROFILE.embedding_dimension)
            t_up = time.perf_counter()
            ib._upsert_payloads(
                client,
                collection_name=collection,
                payloads=batch_payloads,
                vectors=vectors,
                batch_size=batch_size,
            )
            ib._append_bm25_payloads(batch_payloads, bm25_path)
            stats["upsert_seconds"] = float(stats["upsert_seconds"]) + (
                time.perf_counter() - t_up
            )
            stats["upsert_points"] = int(stats["upsert_points"]) + len(batch_payloads)

        completed_ids.update(batch_success)
        checkpoint = {
            "side": side,
            "mode": mode,
            "collection": collection,
            "completed_document_ids": sorted(completed_ids),
            "completed_document_count": len(completed_ids),
            "total_document_count": total,
            "stats": {
                **stats,
                "chunks_per_doc": stats["chunks_per_doc"],
            },
            "updated_at": _utc_now(),
        }
        compact = dict(checkpoint)
        compact_stats = dict(stats)
        compact_stats["chunks_per_doc_tail"] = list(stats["chunks_per_doc"][-500:])
        compact_stats.pop("chunks_per_doc", None)
        compact["stats"] = compact_stats
        # Atomic replace: never leave a truncated checkpoint after crash/reboot mid-write.
        _tmp_ckpt = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        _tmp_ckpt.write_text(
            json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _tmp_ckpt.replace(checkpoint_path)

        done = len(completed_ids)
        elapsed = time.perf_counter() - started
        e2e_s = (
            float(profile.end_to_end_embed_seconds)
            or float(profile.gpu_encode_seconds)
            or float(stats["embed_seconds"])
            or 1e-9
        )
        gpu_s = float(profile.gpu_encode_seconds) or e2e_s
        print(
            json.dumps(
                {
                    "event": "full_corpus_progress",
                    "side": side,
                    "mode": mode,
                    "documents_processed": done,
                    "documents_total": total,
                    "percent": round(100.0 * done / max(1, total), 2),
                    "chunks_created": int(stats["total_chunks"]),
                    "failures": int(stats["documents_failed"]),
                    "embed_chunks_per_sec_e2e": round(
                        float(stats["upsert_points"]) / e2e_s, 2
                    ),
                    "embed_chunks_per_sec_gpu": round(
                        float(stats["upsert_points"]) / gpu_s, 2
                    ),
                    "embed_tokens_per_sec_e2e": round(
                        int(sum(profile.post_truncation_tokens)) / e2e_s, 2
                    ),
                    "elapsed_sec": round(elapsed, 1),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    elapsed_total = time.perf_counter() - started
    chunks_per_doc = [float(x) for x in stats["chunks_per_doc"]]
    point_count = _qdrant_point_count(client, collection)
    bm25_count = len(_bm25_chunk_ids(bm25_path))
    gpu_final = _gpu_info()
    profile_dict = profile.as_dict()
    e2e_s = (
        float(profile.end_to_end_embed_seconds)
        or float(profile.gpu_encode_seconds)
        or float(stats["embed_seconds"])
        or 1e-9
    )
    gpu_s = float(profile.gpu_encode_seconds) or e2e_s
    upsert_s = float(stats["upsert_seconds"]) or 1e-9

    summary = {
        "schema": "legal_v2_full_corpus_side_build_v1",
        "side": side,
        "mode": mode,
        "collection": collection,
        "bm25_index_id": bm25_id,
        "bm25_path": str(bm25_path),
        "git_commit": _git_head(),
        "quality_config": quality,
        "model_path": model_path,
        "max_seq_length": embedder.max_seq_length,
        "embed_scheduler": embed_scheduler,
        "token_budget": token_budget,
        "max_batch_items": max_batch_items,
        "batch_size": batch_size,
        "document_batch_size": document_batch_size,
        "documents_attempted": int(stats["documents_attempted"]),
        "documents_succeeded": int(stats["documents_succeeded"]),
        "documents_failed": int(stats["documents_failed"]),
        "total_chunks": int(stats["total_chunks"]),
        "chunks_per_document": {
            "mean": statistics.fmean(chunks_per_doc) if chunks_per_doc else 0.0,
            "p50": _percentile(chunks_per_doc, 50),
            "p95": _percentile(chunks_per_doc, 95),
            "max": max(chunks_per_doc) if chunks_per_doc else 0.0,
        },
        "gpu": {
            "before": gpu_before,
            "after_warmup": gpu_after_warmup,
            "final": gpu_final,
            "warmup_seconds": embedder.warmup_seconds,
            "embedding_device": device,
        },
        "timing": {
            "chunk_seconds": round(float(stats["chunk_seconds"]), 3),
            "embed_seconds": round(float(stats["embed_seconds"]), 3),
            "tokenization_seconds": profile_dict["tokenization_seconds"],
            "scheduler_overhead_seconds": profile_dict["scheduler_overhead_seconds"],
            "gpu_encode_seconds": profile_dict["gpu_encode_seconds"],
            "end_to_end_embed_seconds": profile_dict["end_to_end_embed_seconds"],
            "upsert_seconds": round(float(stats["upsert_seconds"]), 3),
            "total_wall_seconds_excluding_warmup": round(elapsed_total, 3),
        },
        "throughput": {
            # Decision metric for full build: end-to-end including scheduler tokenize pass.
            "embed_chunks_per_sec": round(float(stats["upsert_points"]) / e2e_s, 3),
            "embed_chunks_per_sec_e2e": round(float(stats["upsert_points"]) / e2e_s, 3),
            "embed_chunks_per_sec_gpu": round(float(stats["upsert_points"]) / gpu_s, 3),
            "embed_docs_per_sec": round(
                int(stats["documents_succeeded"])
                / max(1e-9, float(stats["chunk_seconds"]) + e2e_s),
                3,
            ),
            "effective_tokens_per_sec": profile_dict["effective_tokens_per_sec"],
            "gpu_encode_tokens_per_sec": profile_dict["gpu_encode_tokens_per_sec"],
            "end_to_end_tokens_per_sec": profile_dict["end_to_end_tokens_per_sec"],
            "upsert_points_per_sec": round(float(stats["upsert_points"]) / upsert_s, 3),
            "embed_batches": int(stats["embed_batches"]),
        },
        "profile": profile_dict,
        "qdrant": {
            "point_count": point_count,
            "upsert_points_this_run_stats": int(stats["upsert_points"]),
        },
        "bm25_chunk_count": bm25_count,
        "integrity": {
            "qdrant_points_vs_chunks_stat": point_count == int(stats["total_chunks"])
            or point_count >= int(stats["total_chunks"]),
            "bm25_vs_qdrant_count_equal": bm25_count == point_count,
            "failed_document_count": int(stats["documents_failed"]),
        },
        "finished_at": _utc_now(),
    }
    (side_dir / f"summary_{side}_{mode}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (side_dir / f"summary_{side}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (side_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=("cal2k", "cal2k_adaptive", "full"),
        required=True,
    )
    p.add_argument("--side", choices=("A", "B", "both"), default="both")
    p.add_argument("--batches-dir", type=Path, default=DEFAULT_BATCHES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--storage-bm25-dir", type=Path, default=DEFAULT_STORAGE)
    p.add_argument("--eligible-ids-file", type=Path, default=None)
    p.add_argument("--eligible-meta-file", type=Path, default=None)
    p.add_argument(
        "--document-ids-file",
        type=Path,
        default=None,
        help="Explicit ordered document IDs (required reuse path for cal2k_adaptive).",
    )
    p.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://nalus-scraper-qdrant-1:6333"))
    p.add_argument("--device", choices=("cuda",), default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--document-batch-size", type=int, default=16)
    p.add_argument("--cal2k-count", type=int, default=2000)
    p.add_argument(
        "--embed-scheduler",
        choices=("fixed", "adaptive"),
        default=None,
        help="Defaults: cal2k/full=fixed, cal2k_adaptive=adaptive.",
    )
    p.add_argument("--token-budget", type=int, default=24576)
    p.add_argument("--max-batch-items", type=int, default=64)
    p.add_argument("--force-recreate", action="store_true")
    p.add_argument("--inventory-id", default="full_corpus_constitutional_v1")
    p.add_argument(
        "--run-equivalence-check",
        action="store_true",
        help="Compare fixed vs adaptive embeddings on a small stratified sample; exit after.",
    )
    p.add_argument("--equivalence-sample-chunks", type=int, default=80)
    return p.parse_args(argv)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)


def run_equivalence_check(
    *,
    document_ids: list[str],
    batches_dir: Path,
    out_dir: Path,
    device: str,
    sample_chunks: int,
    token_budget: int,
    max_batch_items: int,
    batch_size: int,
) -> dict[str, Any]:
    """Fixed vs adaptive embeddings must match within tight numeric tolerance."""
    model_path = _resolve_model_path()
    embedder = GpuBgeM3Embedder(
        model_path=model_path, device=device, encode_batch_size=batch_size
    )
    embedder.load()
    docs = discover_source_documents_by_ids(document_ids[:40], batches_dir=batches_dir)
    texts: list[str] = []
    chunk_ids: list[str] = []
    for document in docs:
        payloads = _chunk_document_payloads(
            document=document,
            chunker_version=CHUNKER_A_CURRENT,
            collection="equivalence_check_dummy",
            bm25_id="equivalence_check_dummy",
            inventory_id="equivalence_check",
            ingest_run_id="equivalence_check",
        )
        for payload in payloads:
            texts.append(str(payload["text"]))
            chunk_ids.append(str(payload["chunk_id"]))
            if len(texts) >= sample_chunks:
                break
        if len(texts) >= sample_chunks:
            break
    if len(texts) < 20:
        raise SystemExit(f"HARD_STOP: equivalence sample too small ({len(texts)})")

    fixed = embedder.embed_texts(texts)
    adaptive = embedder.embed_texts_adaptive(
        texts,
        token_budget=token_budget,
        max_items=max_batch_items,
    )
    cosines = [_cosine(a, b) for a, b in zip(fixed, adaptive, strict=True)]
    max_abs = [
        max(abs(x - y) for x, y in zip(a, b, strict=True))
        for a, b in zip(fixed, adaptive, strict=True)
    ]
    report = {
        "schema": "legal_v2_embed_scheduler_equivalence_v1",
        "sample_chunks": len(texts),
        "chunk_id_sample": chunk_ids[:10],
        "token_budget": token_budget,
        "max_batch_items": max_batch_items,
        "fixed_batch_size": batch_size,
        "max_seq_length": embedder.max_seq_length,
        "cosine": {
            "min": min(cosines),
            "mean": statistics.fmean(cosines),
            "max": max(cosines),
        },
        "max_abs_diff": {
            "min": min(max_abs),
            "mean": statistics.fmean(max_abs),
            "max": max(max_abs),
        },
        "pass": min(cosines) >= 0.9999 and max(max_abs) <= 1e-4,
        "tolerance_policy": (
            "Do not require bit-identical GPU vectors. Different batch "
            "composition/padding may produce tiny float differences. Hard-stop "
            "only on material divergence (cosine_min < 0.9999 or max_abs > 1e-4)."
        ),
        "finished_at": _utc_now(),
    }
    out = out_dir / "throughput_2k_adaptive"
    out.mkdir(parents=True, exist_ok=True)
    (out / "equivalence_check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "equivalence_check", **report}, ensure_ascii=False), flush=True)
    if not report["pass"]:
        raise SystemExit(
            "HARD_STOP: fixed vs adaptive embeddings diverge beyond tolerance "
            f"(cosine_min={report['cosine']['min']}, max_abs={report['max_abs_diff']['max']})"
        )
    return report


def _write_profile_compare(
    *,
    out_dir: Path,
    adaptive_summaries: dict[str, Any],
) -> None:
    baseline_dir = out_dir / "throughput_2k"
    adaptive_dir = out_dir / "throughput_2k_adaptive"
    lines = [
        "# Adaptive vs baseline stratified 2k profile",
        "",
        "Baseline = fixed-batch stratified cal2k (`throughput_2k`).",
        "Adaptive = token-budget scheduler (`throughput_2k_adaptive`).",
        "",
        "Do **not** treat baseline A vs B cps as comparable (different batch settings / resume).",
        "",
    ]
    decisions: list[str] = []
    for side in ("A", "B"):
        base_path = baseline_dir / f"summary_{side}_cal2k.json"
        if not base_path.exists():
            base_path = baseline_dir / "summary.json"
        adaptive = adaptive_summaries.get(side)
        if adaptive is None:
            continue
        baseline = None
        if base_path.exists():
            payload = json.loads(base_path.read_text(encoding="utf-8"))
            if "sides" in payload:
                baseline = payload["sides"].get(side)
            elif payload.get("side") == side or "total_chunks" in payload:
                baseline = payload if payload.get("side", side) == side else None
            # summary_A_cal2k.json is direct
            if baseline is None and "total_chunks" in payload and side == "A":
                baseline = payload
        # Prefer dedicated summary files
        dedicated = baseline_dir / f"summary_{side}_cal2k.json"
        if dedicated.exists():
            baseline = json.loads(dedicated.read_text(encoding="utf-8"))
        lines += [f"## Side {side}", ""]
        if baseline is None:
            lines.append("- baseline summary missing")
            lines.append("")
            continue
        base_cps = float(baseline.get("throughput", {}).get("embed_chunks_per_sec") or 0)
        ada_cps = float(adaptive.get("throughput", {}).get("embed_chunks_per_sec") or 0)
        gain = (ada_cps / base_cps) if base_cps > 0 else None
        lines += [
            f"- baseline chunks: {baseline.get('total_chunks')}",
            f"- adaptive chunks: {adaptive.get('total_chunks')}",
            f"- baseline embed cps: {base_cps}",
            f"- adaptive embed cps: {ada_cps}",
            f"- cps gain ratio: {None if gain is None else round(gain, 3)}",
            f"- adaptive tokens/sec: {adaptive.get('throughput', {}).get('effective_tokens_per_sec')}",
            f"- adaptive gpu_encode_s: {adaptive.get('timing', {}).get('gpu_encode_seconds')}",
            f"- adaptive tokenization_s: {adaptive.get('timing', {}).get('tokenization_seconds')}",
            f"- adaptive upsert_s: {adaptive.get('timing', {}).get('upsert_seconds')}",
            f"- batch size hist: `{adaptive.get('profile', {}).get('batch_size_histogram')}`",
            f"- peak VRAM alloc MB: {adaptive.get('profile', {}).get('peak_vram_allocated_mb')}",
            "",
        ]
        if side == "A" and gain is not None:
            if gain >= 1.8:
                decisions.append(
                    "MEANINGFUL_GAIN: recommend starting full A later with adaptive scheduler."
                )
            elif gain < 1.15:
                decisions.append(
                    "LITTLE_GAIN: if GPU was saturated, stop further ingest optimization "
                    "and start full A with safe adaptive/fixed settings."
                )
            else:
                decisions.append(
                    "MODERATE_GAIN: worth using adaptive for full A; no further scheduler tuning."
                )
    lines += ["## Decision", ""]
    if decisions:
        for item in decisions:
            lines.append(f"- {item}")
    else:
        lines.append("- insufficient data for automated decision")
    lines += [
        "",
        "## Hard rules",
        "",
        "- Do not auto-start full A/B/ColBERT from this profile task.",
        "- Do not change retrieval quality configuration.",
        "",
    ]
    adaptive_dir.mkdir(parents=True, exist_ok=True)
    (adaptive_dir / "profile_compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    embed_scheduler = args.embed_scheduler
    if embed_scheduler is None:
        embed_scheduler = "adaptive" if args.mode == "cal2k_adaptive" else "fixed"

    ids_file = args.eligible_ids_file or (out_dir / "eligible_document_ids.txt")
    meta_file = args.eligible_meta_file or (out_dir / "eligible_document_meta.jsonl")
    baseline_ids = out_dir / "throughput_2k" / "document_ids.txt"

    sampling_report: dict[str, Any] | None = None
    if args.mode == "cal2k_adaptive":
        ids_path = args.document_ids_file or baseline_ids
        if not ids_path.exists():
            raise SystemExit(
                "HARD_STOP: cal2k_adaptive requires the same stratified IDs at "
                f"{ids_path} (do not resample)"
            )
        document_ids = _load_ordered_ids(ids_path)
        adaptive_dir = out_dir / "throughput_2k_adaptive"
        adaptive_dir.mkdir(parents=True, exist_ok=True)
        (adaptive_dir / "document_ids.txt").write_text(
            "\n".join(document_ids) + "\n", encoding="utf-8"
        )
        (adaptive_dir / "document_ids_source.txt").write_text(
            str(ids_path.resolve()) + "\n", encoding="utf-8"
        )
    elif args.mode == "cal2k":
        if not meta_file.exists():
            raise SystemExit(f"HARD_STOP: cal2k requires meta file: {meta_file}")
        document_ids, sampling_report = _select_cal2k_ids(
            meta_file, count=args.cal2k_count
        )
        cal_dir = out_dir / "throughput_2k"
        cal_dir.mkdir(parents=True, exist_ok=True)
        (cal_dir / "document_ids.txt").write_text(
            "\n".join(document_ids) + "\n", encoding="utf-8"
        )
        (cal_dir / "sampling_report.json").write_text(
            json.dumps(sampling_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps({"event": "cal2k_sampling", **sampling_report}, ensure_ascii=False),
            flush=True,
        )
    else:
        if not ids_file.exists():
            raise SystemExit(f"HARD_STOP: missing eligible ids file: {ids_file}")
        document_ids = _load_ordered_ids(ids_file)

    if args.run_equivalence_check:
        run_equivalence_check(
            document_ids=document_ids,
            batches_dir=args.batches_dir,
            out_dir=out_dir,
            device=args.device,
            sample_chunks=args.equivalence_sample_chunks,
            token_budget=args.token_budget,
            max_batch_items=args.max_batch_items,
            batch_size=max(8, min(args.batch_size, 16)),
        )
        return 0

    build_config = {
        "mode": args.mode,
        "document_count": len(document_ids),
        "collections": {
            "A": _side_names(args.mode, "A")[0],
            "B": _side_names(args.mode, "B")[0],
        },
        "bm25": {
            "A": _side_names(args.mode, "A")[1],
            "B": _side_names(args.mode, "B")[1],
        },
        "qdrant_url": args.qdrant_url,
        "device": args.device,
        "batch_size": args.batch_size,
        "document_batch_size": args.document_batch_size,
        "embed_scheduler": embed_scheduler,
        "token_budget": args.token_budget,
        "max_batch_items": args.max_batch_items,
        "force_recreate": args.force_recreate,
        "git_commit": _git_head(),
        "quality_config": _gate_quality_config(device=args.device),
        "created_at": _utc_now(),
    }
    config_name = (
        "build_config_adaptive.json"
        if args.mode == "cal2k_adaptive"
        else "build_config.json"
    )
    config_path = (
        (out_dir / "throughput_2k_adaptive" / config_name)
        if args.mode == "cal2k_adaptive"
        else (out_dir / config_name)
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(build_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "build_config", **build_config}, ensure_ascii=False), flush=True)

    sides = ["A", "B"] if args.side == "both" else [args.side]
    summaries: dict[str, Any] = {}
    for side in sides:
        summaries[side] = build_side(
            side=side,
            mode=args.mode,
            document_ids=document_ids,
            batches_dir=args.batches_dir,
            out_dir=out_dir,
            storage_bm25_dir=args.storage_bm25_dir,
            qdrant_url=args.qdrant_url,
            device=args.device,
            batch_size=args.batch_size,
            document_batch_size=args.document_batch_size,
            force_recreate=args.force_recreate,
            inventory_id=args.inventory_id,
            embed_scheduler=embed_scheduler,
            token_budget=args.token_budget,
            max_batch_items=args.max_batch_items,
        )

    if args.mode in {"cal2k", "cal2k_adaptive"}:
        if "A" in summaries and "B" in summaries:
            a_chunks = int(summaries["A"]["total_chunks"])
            b_chunks = int(summaries["B"]["total_chunks"])
            if a_chunks == b_chunks:
                raise SystemExit(
                    "HARD_STOP: stratified cal2k produced identical A/B chunk counts "
                    f"({a_chunks}). Verify builder uses distinct A/B chunkers before full build."
                )
        target = out_dir / _artifact_subdir(args.mode)
        payload = {"sampling": sampling_report, "sides": summaries, "build_config": build_config}
        (target / "summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        lines = [
            f"# 2k throughput ({args.mode})",
            "",
            f"- Documents: **{len(document_ids)}**",
            f"- Scheduler: `{embed_scheduler}`",
            f"- Token budget: `{args.token_budget}`",
            f"- Max batch items: `{args.max_batch_items}`",
            "",
        ]
        for side, summary in summaries.items():
            lines += [
                f"## Side {side}",
                f"- collection: `{summary['collection']}`",
                f"- docs ok/fail: {summary['documents_succeeded']}/{summary['documents_failed']}",
                f"- chunks: {summary['total_chunks']}",
                f"- chunks/doc: `{summary['chunks_per_document']}`",
                f"- embed chunks/sec: {summary['throughput']['embed_chunks_per_sec']}",
                f"- tokens/sec: {summary['throughput'].get('effective_tokens_per_sec')}",
                f"- upsert points/sec: {summary['throughput']['upsert_points_per_sec']}",
                f"- wall sec: {summary['timing']['total_wall_seconds_excluding_warmup']}",
                "",
            ]
        (target / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if args.mode == "cal2k_adaptive":
            _write_profile_compare(out_dir=out_dir, adaptive_summaries=summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
