from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from app.rag.clarification.models import ClarificationDecision, RetrievalHitSummary
from app.rag.clarification.service import LegalQueryClarificationService
from app.rag.orchestrator.orchestrator_service import OrchestratorResult
from app.rag.retrieval.models import RetrievedChunk


class _OrchestratorLike(Protocol):
    def run(self, query: str) -> OrchestratorResult: ...

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]: ...


@dataclass(frozen=True)
class ClarificationTrace:
    query: str
    initial_decision: ClarificationDecision
    final_decision: ClarificationDecision
    retrieval_ran: bool
    llm_called: bool
    cache_hit: bool
    semantic_reuse_hit: bool
    preview_hit_ids: list[str] = field(default_factory=list)
    preview_hit_domains: list[str] = field(default_factory=list)
    returned_sources: list[str] = field(default_factory=list)


class ClarifyingOrchestratorService:
    """Ask clarifying questions before the full orchestration flow runs."""

    def __init__(
        self,
        delegate: _OrchestratorLike,
        *,
        clarification_service: LegalQueryClarificationService | None = None,
        preview_top_k: int = 5,
    ) -> None:
        self._delegate = delegate
        self._clarification = clarification_service or LegalQueryClarificationService.from_env()
        self._preview_top_k = preview_top_k
        self._last_trace: ClarificationTrace | None = None

    @property
    def last_trace(self) -> ClarificationTrace | None:
        return self._last_trace

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        return self._delegate.retrieve(query, top_k=top_k)

    def run(self, query: str) -> OrchestratorResult:
        result, trace = self.run_with_trace(query)
        self._last_trace = trace
        return result

    def run_with_trace(self, query: str) -> tuple[OrchestratorResult, ClarificationTrace]:
        initial_decision = self._clarification.evaluate(query)
        if initial_decision.decision == "ask_clarifying_question":
            trace = self._build_trace(
                query=query,
                initial_decision=initial_decision,
                final_decision=initial_decision,
                retrieval_ran=False,
                preview_hits=[],
                result=None,
            )
            return self._clarification_result(initial_decision), trace

        preview_hits = self._delegate.retrieve(query, top_k=self._preview_top_k)
        preview_summaries = _to_hit_summaries(preview_hits)
        final_decision = self._clarification.evaluate(query, retrieval_hits=preview_summaries)
        if final_decision.decision == "ask_clarifying_question":
            trace = self._build_trace(
                query=query,
                initial_decision=initial_decision,
                final_decision=final_decision,
                retrieval_ran=True,
                preview_hits=preview_hits,
                result=None,
            )
            return self._clarification_result(final_decision), trace

        result = self._delegate.run(query)
        trace = self._build_trace(
            query=query,
            initial_decision=initial_decision,
            final_decision=final_decision,
            retrieval_ran=True,
            preview_hits=preview_hits,
            result=result,
        )
        return result, trace

    def _build_trace(
        self,
        *,
        query: str,
        initial_decision: ClarificationDecision,
        final_decision: ClarificationDecision,
        retrieval_ran: bool,
        preview_hits: list[RetrievedChunk],
        result: OrchestratorResult | None,
    ) -> ClarificationTrace:
        preview_summaries = _to_hit_summaries(preview_hits)
        return ClarificationTrace(
            query=query,
            initial_decision=initial_decision,
            final_decision=final_decision,
            retrieval_ran=retrieval_ran,
            llm_called=bool(initial_decision.llm_called or final_decision.llm_called),
            cache_hit=bool(initial_decision.cache_hit or final_decision.cache_hit),
            semantic_reuse_hit=bool(
                initial_decision.semantic_cache_hit or final_decision.semantic_cache_hit
            ),
            preview_hit_ids=[hit.document_id for hit in preview_summaries],
            preview_hit_domains=[_infer_domain_name(hit.document_id) for hit in preview_summaries],
            returned_sources=[] if result is None else list(result.sources),
        )

    @staticmethod
    def _clarification_result(decision: ClarificationDecision) -> OrchestratorResult:
        return OrchestratorResult(
            answer=decision.clarification_question_cs,
            sources=[],
            plan_steps=[],
        )


def _to_hit_summaries(chunks: list[RetrievedChunk]) -> list[RetrievalHitSummary]:
    summaries: list[RetrievalHitSummary] = []
    for index, chunk in enumerate(chunks, start=1):
        summaries.append(
            RetrievalHitSummary(
                rank=index,
                document_id=_document_id_for_chunk(chunk),
                score=chunk.score,
            )
        )
    return summaries


def _document_id_for_chunk(chunk: RetrievedChunk) -> str:
    metadata = chunk.metadata or {}
    for key in ("document_id", "case_reference", "reference", "ecli", "original_id"):
        value = metadata.get(key)
        if value:
            return str(value)
    return chunk.id


def _infer_domain_name(document_id: str) -> str:
    from app.rag.clarification.retrieval_feedback import infer_domain_from_document_id

    return infer_domain_from_document_id(document_id)

