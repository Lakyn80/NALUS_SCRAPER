"""Long-input preprocessing / SearchBrief layer for Legal v2.

This package is a pre-retrieval boundary. It must not import FastAPI, Qdrant,
BM25, RRF, or QuerySpec internals.
"""

from app.rag.legal_v2.query_input.config import LongInputConfig, long_input_config_from_env
from app.rag.legal_v2.query_input.models import (
    CondensationMethod,
    InputClassification,
    PreparedQuery,
    SearchBrief,
)
from app.rag.legal_v2.query_input.service import QueryInputService, get_query_input_service

__all__ = [
    "CondensationMethod",
    "InputClassification",
    "LongInputConfig",
    "PreparedQuery",
    "QueryInputService",
    "SearchBrief",
    "get_query_input_service",
    "long_input_config_from_env",
]
