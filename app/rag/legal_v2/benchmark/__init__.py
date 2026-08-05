"""Legal v2 retrieval / parser benchmark helpers."""

from app.rag.legal_v2.benchmark.corpus import (
    DevelopmentCorpus,
    load_development_corpus,
    load_development_document_refs,
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
    "rank_blocks_by_token_overlap",
    "DEFAULT_PILOT_DATASET",
    "RetrievalGoldenItem",
    "load_retrieval_golden_jsonl",
    "validate_retrieval_golden_dataset",
]
