from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.rag.legal_v2.chunking import RetrievalChildChunk
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


def payload_for_child_chunk(chunk: RetrievalChildChunk) -> dict[str, Any]:
    return {
        "original_id": chunk.chunk_id,
        "text": chunk.text,
        "document_id": chunk.document_id,
        "chunk_index": chunk.chunk_index,
        "section_type": chunk.section_type.value,
        "paragraph_ids": list(chunk.paragraph_ids),
        "paragraph_texts": dict(chunk.paragraph_texts),
        "paragraph_original_texts": dict(chunk.paragraph_original_texts),
        "source_spans": [asdict(span) for span in chunk.source_spans],
        "start_offset": chunk.start_offset,
        "end_offset": chunk.end_offset,
        "source_order": chunk.source_order,
        "heading_context": list(chunk.heading_context),
        "retrieval_profile": LEGAL_V2_PROFILE.name,
        "collection_version": "v2",
        "can_reconstruct_paragraph_evidence": True,
    }
