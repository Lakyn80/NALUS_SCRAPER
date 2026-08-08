"""High-level Cross-Encoder reranking orchestration for Legal v2 Stage 1."""

from __future__ import annotations

import time
from threading import Lock
from typing import Any, Sequence

from app.observability import legal_v2_metrics
from app.rag.legal_v2.rerank.aggregation import aggregate_max_passage_scores
from app.rag.legal_v2.rerank.config import CrossEncoderConfig, cross_encoder_config_from_env
from app.rag.legal_v2.rerank.errors import (
    RerankerInferenceError,
    RerankerInvalidCandidateError,
    RerankerModelLoadError,
    RerankerUnavailableError,
)
from app.rag.legal_v2.rerank.models import (
    RerankDiagnostics,
    RerankResult,
)
from app.rag.legal_v2.rerank.passage_selection import build_candidates_from_stage1_docs
from app.rag.legal_v2.rerank.providers.cross_encoder import (
    SentenceTransformersCrossEncoderProvider,
)
from app.rag.legal_v2.rerank.selectors.base import EvidencePassageSelector
from app.rag.legal_v2.rerank.selectors.policy import get_evidence_passage_selector


class CrossEncoderRerankingService:
    def __init__(
        self,
        config: CrossEncoderConfig | None = None,
        *,
        provider: Any | None = None,
        passage_selector: EvidencePassageSelector | None = None,
    ) -> None:
        self._config = config or cross_encoder_config_from_env()
        self._config.validate()
        self._provider = provider
        self._passage_selector = passage_selector or get_evidence_passage_selector(
            self._config.passage_selector
        )

    @property
    def config(self) -> CrossEncoderConfig:
        return self._config

    def _get_provider(self) -> Any:
        if self._provider is None:
            self._provider = SentenceTransformersCrossEncoderProvider(self._config)
        return self._provider

    def readiness(self) -> dict[str, Any]:
        if not self._config.enabled:
            return {
                "enabled": False,
                "status": "disabled",
                "model": self._config.model_id,
                "device": None,
            }
        provider = self._get_provider()
        if provider.is_loaded:
            return {
                "enabled": True,
                "status": "ready",
                "model": provider.model_id,
                "device": provider.device,
            }
        return {
            "enabled": True,
            "status": "not_loaded",
            "model": self._config.model_id,
            "device": None,
        }

    def rerank(
        self,
        query: str,
        stage1_documents: Sequence[object],
        *,
        require_success: bool = True,
    ) -> RerankResult:
        if not self._config.enabled:
            raise RerankerUnavailableError("cross-encoder reranking is disabled")

        cleaned_query = " ".join(str(query or "").split()).strip()
        if not cleaned_query:
            raise RerankerInvalidCandidateError("query must not be blank")
        if "ECLI:" in cleaned_query.upper() and cleaned_query.upper().count("ECLI:") > 0:
            # Soft check only — do not strip; Stage1 QuerySpec owns identity policy.
            # We still refuse if the entire query is an identifier-only shortcut.
            compact = cleaned_query.replace(" ", "")
            if compact.upper().startswith("ECLI:") and len(compact) < 80 and " " not in cleaned_query:
                raise RerankerInvalidCandidateError(
                    "identifier-only query is not valid for CE reranking"
                )

        docs = list(stage1_documents or [])
        if not docs:
            raise RerankerInvalidCandidateError("no Stage 1 candidates to rerank")

        shortlist = docs[: self._config.candidate_documents]
        candidates, warnings = build_candidates_from_stage1_docs(
            shortlist,
            max_documents=self._config.candidate_documents,
            max_passages=self._config.passages_per_document,
            selector=self._passage_selector,
            passage_selector_name=self._config.passage_selector,
        )
        if not candidates:
            raise RerankerInvalidCandidateError("no valid CE candidates after passage selection")

        all_passages = tuple(p for c in candidates for p in c.passages)
        pair_count = len(all_passages)
        selected_counts = [len(c.passages) for c in candidates]
        mean_selected = (
            float(sum(selected_counts)) / float(len(selected_counts))
            if selected_counts
            else 0.0
        )
        batch_count = (
            (pair_count + self._config.batch_size - 1) // self._config.batch_size
            if pair_count
            else 0
        )

        started = time.perf_counter()
        provider = self._get_provider()
        try:
            scores = provider.score(cleaned_query, all_passages)
            status = "ok"
        except (RerankerModelLoadError, RerankerInferenceError) as exc:
            legal_v2_metrics.record_rerank(
                status="error",
                device_class=_device_class(getattr(provider, "device", "unknown")),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                pair_count=pair_count,
            )
            if require_success:
                raise
            raise RerankerUnavailableError(str(exc)) from exc

        truncated = sum(1 for item in scores if item.truncated)
        ranked = aggregate_max_passage_scores(candidates, scores)
        latency_ms = (time.perf_counter() - started) * 1000.0
        legal_v2_metrics.record_rerank(
            status=status,
            device_class=_device_class(provider.device),
            latency_ms=latency_ms,
            pair_count=pair_count,
        )

        diagnostics = RerankDiagnostics(
            rerank_enabled=True,
            rerank_applied=True,
            reranker_model=provider.model_id,
            reranker_device=provider.device,
            candidate_document_count=len(candidates),
            passage_count=pair_count,
            pair_count=pair_count,
            batch_count=batch_count,
            truncated_pair_count=truncated,
            aggregation=self._config.aggregation,
            rerank_latency_ms=latency_ms,
            warnings=warnings,
            model_revision=getattr(provider, "model_revision", None),
            dtype=getattr(provider, "dtype", None),
            passage_selector=getattr(
                self._passage_selector, "policy_id", self._config.passage_selector
            ),
            requested_passages_per_document=self._config.passages_per_document,
            mean_selected_passages=mean_selected,
        )
        return RerankResult(documents=ranked, diagnostics=diagnostics)


_service: CrossEncoderRerankingService | None = None
_service_lock = Lock()


def get_cross_encoder_reranking_service(
    config: CrossEncoderConfig | None = None,
) -> CrossEncoderRerankingService:
    global _service
    if config is not None:
        return CrossEncoderRerankingService(config=config)
    if _service is not None:
        return _service
    with _service_lock:
        if _service is not None:
            return _service
        _service = CrossEncoderRerankingService(config=cross_encoder_config_from_env())
        return _service


def reset_cross_encoder_reranking_service_for_tests() -> None:
    global _service
    with _service_lock:
        _service = None


def _device_class(device: str | None) -> str:
    value = (device or "unknown").strip().lower()
    if value.startswith("cuda"):
        return "cuda"
    if value == "cpu":
        return "cpu"
    if value == "injected":
        return "injected"
    return "unknown"
