from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.rag.legal_v2.audit import PARSER_VERSION
from app.rag.legal_v2.ingest.chunking import RetrievalChildChunk
from app.rag.retrieval.provenance import build_embedding_provenance, content_checksum
from app.rag.retrieval.production_profile import RetrievalProfile

LEGAL_V2_COLLECTION_NAME = "nalus_legal_paragraph_chunks_v2"
LEGAL_V2_BM25_INDEX_ID = "nalus_legal_paragraph_bm25_v2"
LEGAL_V2_PROFILE = RetrievalProfile(
    name="nalus_legal_paragraph_chunks_v2",
    embedding_provider="sentence_transformer",
    embedding_model="BAAI/bge-m3",
    embedding_dimension=1024,
    retrieval_mode="dense_plus_bm25_parent_verification",
    fusion="rrf",
    rrf_k=60,
    bm25_k1=1.5,
    bm25_b=0.75,
)


@dataclass(frozen=True)
class LegalV2IndexingContract:
    enabled: bool = False
    profile: RetrievalProfile = LEGAL_V2_PROFILE
    proposed_collection_name: str = LEGAL_V2_COLLECTION_NAME
    proposed_bm25_index_id: str = LEGAL_V2_BM25_INDEX_ID
    proposed_bm25_sidecar_path: Path = Path("storage") / "rag" / "bm25" / "nalus_legal_paragraph_bm25_v2.sqlite"
    active_production_alias_unchanged: bool = True
    writes_to_current_collection: bool = False


def legal_v2_indexing_contract() -> LegalV2IndexingContract:
    return LegalV2IndexingContract()


def payload_for_child_chunk(
    chunk: RetrievalChildChunk,
    *,
    qdrant_collection: str = LEGAL_V2_COLLECTION_NAME,
    bm25_index_id: str = LEGAL_V2_BM25_INDEX_ID,
) -> dict[str, Any]:
    paragraph_indexes = [span.paragraph_index for span in chunk.source_spans]
    metadata = dict(chunk.metadata)
    ecli = str(metadata.get("ecli") or "").strip() or None
    source_document_id = str(metadata.get("source_document_id") or "").strip() or None
    canonical_document_id = str(metadata.get("canonical_document_id") or ecli or chunk.document_id).strip()
    base = {
        "original_id": chunk.chunk_id,
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "document_id": chunk.document_id,
        "canonical_document_id": canonical_document_id,
        "ecli": ecli or (chunk.document_id if str(chunk.document_id).upper().startswith("ECLI:") else None),
        "source_document_id": source_document_id,
        "chunk_index": chunk.chunk_index,
        "section_type": chunk.section_type.value,
        "paragraph_ids": list(chunk.paragraph_ids),
        "paragraph_start": min(paragraph_indexes) if paragraph_indexes else None,
        "paragraph_end": max(paragraph_indexes) if paragraph_indexes else None,
        "paragraph_texts": dict(chunk.paragraph_texts),
        "paragraph_original_texts": dict(chunk.paragraph_original_texts),
        "source_spans": [asdict(span) for span in chunk.source_spans],
        "start_offset": chunk.start_offset,
        "end_offset": chunk.end_offset,
        "source_order": chunk.source_order,
        "heading_context": list(chunk.heading_context),
        "parent_window_id": metadata.get("parent_window_id"),
        "parent_window_paragraph_ids": _metadata_list(metadata.get("parent_window_paragraph_ids")),
        "parent_window_child_chunk_ids": _metadata_list(metadata.get("parent_window_child_chunk_ids")),
        "parent_window_text_checksum": metadata.get("parent_window_text_checksum"),
        "parent_window_token_count": metadata.get("parent_window_token_count"),
        "parent_window_truncated": bool(metadata.get("parent_window_truncated", False)),
        "token_count": chunk.token_count,
        "language": metadata.get("language", "cs"),
        "source": metadata.get("source"),
        "court": metadata.get("court"),
        "case_reference": metadata.get("case_reference"),
        "decision_date": metadata.get("decision_date"),
        "document_type": metadata.get("document_type"),
        "is_boilerplate": metadata.get("is_boilerplate", False),
        "is_citation_block": metadata.get("is_citation_block", False),
        "parser_version": metadata.get("parser_version", PARSER_VERSION),
        "chunker_version": metadata.get("chunker_version", "legal_v2_hierarchical_chunker_v1"),
        "document_content_hash": metadata.get("document_content_hash"),
        "retrieval_profile": LEGAL_V2_PROFILE.name,
        "collection_version": "v2",
        "can_reconstruct_paragraph_evidence": True,
    }
    base["content_checksum"] = content_checksum(chunk.text)
    return build_embedding_provenance(
        payload=base,
        profile=LEGAL_V2_PROFILE,
        ingest_run_id=str(metadata.get("ingest_run_id") or "legal_v2_offline_builder"),
        qdrant_collection=qdrant_collection,
        bm25_index_id=bm25_index_id,
    )


def _metadata_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []
