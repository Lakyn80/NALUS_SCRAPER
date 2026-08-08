"""Optional real-model smoke for BGE reranker (skipped if cache missing)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.rag.legal_v2.rerank.config import CrossEncoderConfig
from app.rag.legal_v2.rerank.errors import RerankerModelLoadError
from app.rag.legal_v2.rerank.models import RerankPassage
from app.rag.legal_v2.rerank.providers.cross_encoder import (
    SentenceTransformersCrossEncoderProvider,
)


def _model_likely_cached(model_id: str = "BAAI/bge-reranker-v2-m3") -> bool:
    hub = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    token = "models--" + model_id.replace("/", "--")
    return (hub / token).exists()


@pytest.mark.skipif(
    not _model_likely_cached() and os.getenv("NALUS_LEGAL_V2_CE_ALLOW_DOWNLOAD", "0") not in {
        "1",
        "true",
        "yes",
        "on",
    },
    reason="BGE reranker weights not in local HF cache; set NALUS_LEGAL_V2_CE_ALLOW_DOWNLOAD=1 to fetch",
)
def test_real_bge_reranker_confusable_ordering() -> None:
    allow = os.getenv("NALUS_LEGAL_V2_CE_ALLOW_DOWNLOAD", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    provider = SentenceTransformersCrossEncoderProvider(
        CrossEncoderConfig(
            enabled=True,
            model_id="BAAI/bge-reranker-v2-m3",
            allow_download=allow,
            local_files_only=not allow,
            device="cpu",
            batch_size=2,
            max_length=512,
        )
    )
    try:
        provider.load()
    except RerankerModelLoadError as exc:
        pytest.skip(f"model unavailable: {exc}")

    query = "výpověď z nájmu bytu"
    passages = (
        RerankPassage(
            ecli="ECLI:NAJEM",
            text="Pronajímatel vypověděl nájem bytu pro hrubé porušení povinností nájemce.",
            chunk_id="najom",
            stage1_document_rank=2,
            passage_index=0,
        ),
        RerankPassage(
            ecli="ECLI:PRACE",
            text="Zaměstnavatel dal zaměstnanci výpověď z pracovního poměru pro nadbytečnost.",
            chunk_id="prace",
            stage1_document_rank=1,
            passage_index=0,
        ),
    )
    scores = {item.ecli: item.score for item in provider.score(query, passages)}
    assert scores["ECLI:NAJEM"] > scores["ECLI:PRACE"]
