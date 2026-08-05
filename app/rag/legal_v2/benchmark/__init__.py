"""Legal v2 retrieval / parser benchmark helpers."""

from app.rag.legal_v2.benchmark.case_similarity_golden import (
    DEFAULT_PILOT_DATASET as DEFAULT_CASE_SIMILARITY_PILOT_DATASET,
    CaseSimilarityGoldenItem,
    load_case_similarity_golden_jsonl,
    validate_case_similarity_dataset,
)
from app.rag.legal_v2.benchmark.corpus import (
    DevelopmentCorpus,
    load_case_similarity_corpus,
    load_case_similarity_primary_document_ids,
    load_development_corpus,
    load_development_document_refs,
    load_reviewed_pool_corpus,
    load_reviewed_pool_document_refs,
    rank_blocks_by_token_overlap,
)
from app.rag.legal_v2.benchmark.retrieval_golden import (
    DEFAULT_PILOT_DATASET,
    RetrievalGoldenItem,
    load_retrieval_golden_jsonl,
    validate_retrieval_golden_dataset,
)

__all__ = [
    "DevelopmentCorpus",
    "load_development_corpus",
    "load_development_document_refs",
    "load_reviewed_pool_corpus",
    "load_reviewed_pool_document_refs",
    "load_case_similarity_corpus",
    "load_case_similarity_primary_document_ids",
    "rank_blocks_by_token_overlap",
    "DEFAULT_PILOT_DATASET",
    "RetrievalGoldenItem",
    "load_retrieval_golden_jsonl",
    "validate_retrieval_golden_dataset",
    "DEFAULT_CASE_SIMILARITY_PILOT_DATASET",
    "CaseSimilarityGoldenItem",
    "load_case_similarity_golden_jsonl",
    "validate_case_similarity_dataset",
]
