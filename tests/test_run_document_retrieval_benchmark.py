from __future__ import annotations

import pytest

from app.rag.retrieval.errors import RetrievalConfigurationError
from scripts.run_document_retrieval_benchmark import build_document_config, parse_args


def test_runner_builds_document_config_from_cli() -> None:
    args = parse_args(
        [
            "--dataset",
            "dataset.jsonl",
            "--retrieval-only",
            "--candidate-pool-size",
            "25",
            "--max-returned-documents",
            "10",
            "--max-supporting-chunks",
            "2",
            "--document-threshold",
            "0.25",
            "--latency-budget-ms",
            "1000",
        ]
    )

    config = build_document_config(args)

    assert config.enabled is True
    assert config.max_candidate_chunks == 25
    assert config.max_returned_documents == 10
    assert config.max_supporting_chunks_per_document == 2
    assert config.document_relevance_threshold == 0.25
    assert config.latency_budget_ms == 1000


def test_runner_rejects_invalid_document_config() -> None:
    args = parse_args(
        [
            "--dataset",
            "dataset.jsonl",
            "--retrieval-only",
            "--candidate-pool-size",
            "0",
        ]
    )

    with pytest.raises(RetrievalConfigurationError):
        build_document_config(args)
