from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.rag.retrieval.errors import RetrievalConfigurationError


@dataclass(frozen=True)
class RetrievalProfile:
    name: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    retrieval_mode: str
    fusion: str
    rrf_k: int
    bm25_k1: float
    bm25_b: float


@dataclass(frozen=True)
class ProductionRetrievalConfig:
    profile: RetrievalProfile
    qdrant_collection: str
    bm25_sidecar_path: Path
    bm25_index_id: str
    model_path: str
    local_files_only: bool
    trust_remote_code: bool
    device: str
    candidate_multiplier: int
    min_candidate_count: int
    max_candidate_count: int
    lexical_filter_enabled: bool


BGE_M3_DENSE_BM25_RRF = RetrievalProfile(
    name="nalus_bge_m3_dense_bm25_rrf_v1",
    embedding_provider="sentence_transformer",
    embedding_model="BAAI/bge-m3",
    embedding_dimension=1024,
    retrieval_mode="dense_plus_bm25",
    fusion="rrf",
    rrf_k=60,
    bm25_k1=1.5,
    bm25_b=0.75,
)

DEFAULT_QDRANT_COLLECTION = "nalus_bge_m3_chunks_v1"
DEFAULT_MODEL_PATH = "/app/models/BAAI/bge-m3"
DEFAULT_BM25_SIDECAR_PATH = (
    Path("storage") / "rag" / "bm25" / f"{BGE_M3_DENSE_BM25_RRF.name}.sqlite"
)


FORBIDDEN_MODEL_MARKERS = ("mpnet", "paraphrase-multilingual-mpnet")


def _validate_model_path(model_path: str) -> None:
    normalized = model_path.lower()
    if any(marker in normalized for marker in FORBIDDEN_MODEL_MARKERS):
        raise RetrievalConfigurationError(
            f"MPNet embeddings are forbidden. Use local BGE-M3 at {DEFAULT_MODEL_PATH}. Got: {model_path}"
        )


def production_retrieval_config_from_env() -> ProductionRetrievalConfig:
    profile = _profile_from_env()
    bm25_path = Path(os.getenv("BM25_SIDECAR_PATH", str(DEFAULT_BM25_SIDECAR_PATH)))
    bm25_index_id = os.getenv("BM25_INDEX_ID", profile.name)
    model_path = os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_MODEL_PATH)
    _validate_model_path(model_path)
    return ProductionRetrievalConfig(
        profile=profile,
        qdrant_collection=os.getenv("QDRANT_COLLECTION_NAME", DEFAULT_QDRANT_COLLECTION),
        bm25_sidecar_path=bm25_path,
        bm25_index_id=bm25_index_id,
        model_path=model_path,
        local_files_only=_read_bool_env("EMBEDDING_LOCAL_FILES_ONLY", default=True),
        trust_remote_code=_read_bool_env("EMBEDDING_TRUST_REMOTE_CODE", default=False),
        device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        candidate_multiplier=_read_int_env("NALUS_RETRIEVAL_CANDIDATE_MULTIPLIER", default=6),
        min_candidate_count=_read_int_env("NALUS_RETRIEVAL_MIN_CANDIDATES", default=50),
        max_candidate_count=_read_int_env("NALUS_RETRIEVAL_MAX_CANDIDATES", default=500),
        lexical_filter_enabled=_read_bool_env("NALUS_RETRIEVAL_LEXICAL_FILTER_ENABLED", default=True),
    )


def _profile_from_env() -> RetrievalProfile:
    profile_name = os.getenv("RETRIEVAL_PROFILE", BGE_M3_DENSE_BM25_RRF.name)
    retrieval_mode = os.getenv("RETRIEVAL_MODE", BGE_M3_DENSE_BM25_RRF.retrieval_mode)
    provider = os.getenv("EMBEDDING_PROVIDER", BGE_M3_DENSE_BM25_RRF.embedding_provider)
    dimension = int(os.getenv("EMBEDDING_DIMENSION", str(BGE_M3_DENSE_BM25_RRF.embedding_dimension)))

    if profile_name != BGE_M3_DENSE_BM25_RRF.name:
        raise RetrievalConfigurationError(
            f"Unsupported production retrieval profile: {profile_name}. "
            f"Expected {BGE_M3_DENSE_BM25_RRF.name}."
        )
    if retrieval_mode != BGE_M3_DENSE_BM25_RRF.retrieval_mode:
        raise RetrievalConfigurationError(
            f"Unsupported production retrieval mode: {retrieval_mode}. "
            f"Expected {BGE_M3_DENSE_BM25_RRF.retrieval_mode}."
        )
    if provider != BGE_M3_DENSE_BM25_RRF.embedding_provider:
        raise RetrievalConfigurationError(
            f"Unsupported embedding provider: {provider}. "
            f"Expected {BGE_M3_DENSE_BM25_RRF.embedding_provider}."
        )
    if dimension != BGE_M3_DENSE_BM25_RRF.embedding_dimension:
        raise RetrievalConfigurationError(
            f"Unsupported embedding dimension: {dimension}. "
            f"Expected {BGE_M3_DENSE_BM25_RRF.embedding_dimension}."
        )

    return BGE_M3_DENSE_BM25_RRF


def _read_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise RetrievalConfigurationError(f"{name} must be a boolean value.")


def _read_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RetrievalConfigurationError(f"{name} must be an integer value.") from exc
    if value <= 0:
        raise RetrievalConfigurationError(f"{name} must be greater than zero.")
    return value
