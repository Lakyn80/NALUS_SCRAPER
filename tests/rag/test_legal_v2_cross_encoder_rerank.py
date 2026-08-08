"""Unit tests for Legal v2 Cross-Encoder reranking (no model download)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.rag.legal_v2.rerank.aggregation import aggregate_max_passage_scores
from app.rag.legal_v2.rerank.config import CrossEncoderConfig, cross_encoder_config_from_env
from app.rag.legal_v2.rerank.errors import RerankerInferenceError, RerankerUnavailableError
from app.rag.legal_v2.rerank.models import RerankCandidate, RerankPassage, RerankScore
from app.rag.legal_v2.rerank.passage_selection import (
    build_candidates_from_stage1_docs,
    select_passages_for_document,
)
from app.rag.legal_v2.rerank.service import (
    CrossEncoderRerankingService,
    reset_cross_encoder_reranking_service_for_tests,
)


@dataclass
class _Passage:
    text: str
    chunk_id: str


@dataclass
class _Doc:
    ecli: str
    rank: int
    score: float
    relevant_passages: list[_Passage]
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None
    metadata: dict | None = None


class _FakeProvider:
    model_id = "fake-ce"
    device = "injected"
    is_loaded = True
    model_revision = "fake"
    dtype = "float32"

    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self._scores = scores or {}
        self.load_calls = 0
        self.score_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    def score(self, query: str, passages):
        self.score_calls += 1
        assert "ECLI:" not in query
        out = []
        for p in passages:
            out.append(
                RerankScore(
                    ecli=p.ecli,
                    chunk_id=p.chunk_id,
                    score=float(self._scores.get(p.ecli, 0.1)),
                    passage_index=p.passage_index,
                    truncated=False,
                )
            )
        return tuple(out)


@pytest.fixture(autouse=True)
def _reset_service():
    reset_cross_encoder_reranking_service_for_tests()
    yield
    reset_cross_encoder_reranking_service_for_tests()


def test_config_defaults_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED", raising=False)
    cfg = cross_encoder_config_from_env()
    assert cfg.enabled is False
    assert cfg.candidate_documents == 30
    assert cfg.passages_per_document == 3
    assert cfg.aggregation == "max"


def test_provider_not_required_when_disabled() -> None:
    svc = CrossEncoderRerankingService(CrossEncoderConfig(enabled=False))
    assert svc.readiness()["status"] == "disabled"
    with pytest.raises(RerankerUnavailableError):
        svc.rerank("query", [_Doc("ECLI:CZ:US:1", 1, 0.1, [_Passage("a", "c1")])])


def test_passage_cap_and_dedupe() -> None:
    passages = select_passages_for_document(
        ecli="ECLI:A",
        stage1_rank=1,
        passage_texts=[
            ("c1", "alpha text"),
            ("c2", "alpha text"),
            ("c3", "beta text"),
            ("c4", "gamma text"),
            ("c5", "delta text"),
        ],
        max_passages=3,
    )
    assert len(passages) == 3
    assert [p.chunk_id for p in passages] == ["c1", "c3", "c4"]


def test_candidate_document_cap() -> None:
    docs = [
        _Doc(f"ECLI:{i}", i + 1, 1.0 - i * 0.01, [_Passage(f"text {i}", f"c{i}")])
        for i in range(10)
    ]
    candidates, _warnings = build_candidates_from_stage1_docs(
        docs, max_documents=3, max_passages=2
    )
    assert len(candidates) == 3
    assert candidates[0].ecli == "ECLI:0"


def test_max_aggregation_and_tie_break() -> None:
    candidates = (
        RerankCandidate(
            ecli="ECLI:B",
            stage1_rank=2,
            stage1_score=0.5,
            passages=(
                RerankPassage("ECLI:B", "x", "c1", 2, 0),
                RerankPassage("ECLI:B", "y", "c2", 2, 1),
            ),
        ),
        RerankCandidate(
            ecli="ECLI:A",
            stage1_rank=1,
            stage1_score=0.9,
            passages=(RerankPassage("ECLI:A", "z", "c3", 1, 0),),
        ),
        RerankCandidate(
            ecli="ECLI:C",
            stage1_rank=3,
            stage1_score=0.4,
            passages=(RerankPassage("ECLI:C", "w", "c4", 3, 0),),
        ),
    )
    scores = (
        RerankScore("ECLI:B", "c1", 0.2, 0),
        RerankScore("ECLI:B", "c2", 0.8, 1),
        RerankScore("ECLI:A", "c3", 0.8, 0),
        RerankScore("ECLI:C", "c4", 0.1, 0),
    )
    ranked = aggregate_max_passage_scores(candidates, scores)
    # B and A both max=0.8; Stage1 rank asc => A (1) before B (2)
    assert [d.ecli for d in ranked] == ["ECLI:A", "ECLI:B", "ECLI:C"]
    assert ranked[0].ce_score == 0.8
    assert ranked[1].ce_score == 0.8
    assert ranked[0].stage1_rank == 1
    assert ranked[1].stage1_rank == 2


def test_service_preserves_stage1_provenance() -> None:
    docs = [
        _Doc(
            "ECLI:CZ:US:NAJEM",
            1,
            0.55,
            [_Passage("pronajímatel vypověděl nájem bytu", "c1")],
        ),
        _Doc(
            "ECLI:CZ:US:PRACE",
            2,
            0.50,
            [_Passage("zaměstnavatel dal zaměstnanci výpověď", "c2")],
        ),
    ]
    provider = _FakeProvider({"ECLI:CZ:US:NAJEM": 0.9, "ECLI:CZ:US:PRACE": 0.2})
    svc = CrossEncoderRerankingService(
        CrossEncoderConfig(enabled=True, candidate_documents=30, passages_per_document=3),
        provider=provider,
    )
    result = svc.rerank("výpověď z nájmu bytu", docs)
    assert result.diagnostics.rerank_applied is True
    assert result.documents[0].ecli == "ECLI:CZ:US:NAJEM"
    assert result.documents[0].stage1_rank == 1
    assert result.documents[0].stage1_score == 0.55
    assert result.documents[0].ce_rank == 1
    assert result.documents[1].ecli == "ECLI:CZ:US:PRACE"
    assert result.documents[1].stage1_rank == 2
    assert provider.score_calls == 1


def test_missing_passages_handled() -> None:
    docs = [
        _Doc("ECLI:EMPTY", 1, 0.4, []),
        _Doc("ECLI:HAS", 2, 0.3, [_Passage("relevant text", "c1")]),
    ]
    provider = _FakeProvider({"ECLI:HAS": 0.7})
    svc = CrossEncoderRerankingService(
        CrossEncoderConfig(enabled=True),
        provider=provider,
    )
    result = svc.rerank("dotaz", docs)
    assert result.documents[0].ecli == "ECLI:HAS"
    assert result.documents[1].ecli == "ECLI:EMPTY"
    assert any(w.startswith("no_passages:") for w in result.diagnostics.warnings)


def test_inference_failure_is_explicit() -> None:
    class _Boom(_FakeProvider):
        def score(self, query, passages):
            raise RerankerInferenceError("boom")

    svc = CrossEncoderRerankingService(
        CrossEncoderConfig(enabled=True),
        provider=_Boom(),
    )
    with pytest.raises(RerankerInferenceError):
        svc.rerank("dotaz", [_Doc("ECLI:X", 1, 0.1, [_Passage("t", "c")])])


def test_batch_count_accounting() -> None:
    docs = [
        _Doc(f"ECLI:{i}", i + 1, 1.0, [_Passage(f"text {i}", f"c{i}"), _Passage(f"more {i}", f"d{i}")])
        for i in range(5)
    ]
    provider = _FakeProvider({f"ECLI:{i}": float(i) for i in range(5)})
    svc = CrossEncoderRerankingService(
        CrossEncoderConfig(enabled=True, batch_size=4, passages_per_document=2),
        provider=provider,
    )
    result = svc.rerank("dotaz", docs)
    # 5 docs * 2 passages = 10 pairs; batch_size 4 => 3 batches
    assert result.diagnostics.pair_count == 10
    assert result.diagnostics.batch_count == 3
