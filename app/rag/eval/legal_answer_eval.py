"""Deterministic no-LLM legal answer evaluation over gold retrieval results."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.rag.eval.evidence_window import (
    EvidenceWindow,
    EvidenceWindowConfig,
    build_evidence_window,
)
from app.rag.eval.legal_qa_benchmark import (
    LegalQaItem,
    SourceConstraints,
    load_dataset,
    normalize_for_match,
)
from app.rag.retrieval.errors import RetrievalConfigurationError

SUPPORT_LEVELS = frozenset(
    {"direct", "partial", "gap", "boilerplate_noise", "corpus_only"}
)
ANSWER_EVAL_STATUSES = frozenset({"pass", "partial", "gap", "needs_review", "skipped"})
BOILERPLATE_PATTERNS = (
    re.compile(r"^\s*takto\s*:", re.IGNORECASE),
    re.compile(r"^\s*dovol[aá]n[ií]\s+se\s+odm[ií]t[aá]", re.IGNORECASE),
    re.compile(r"^\s*usnesen[ií]\s", re.IGNORECASE),
    re.compile(r"^\s*rozsudek\s", re.IGNORECASE),
)
MIN_SUBSTANTIVE_SNIPPET_LEN = 40
MIN_STABLE_GOLD_COUNT = 10
NON_PASS_STATUSES = frozenset({"partial", "gap", "needs_review", "skipped"})


@dataclass(frozen=True)
class GoldRegistryEntry:
    question_id: str
    corpus: str
    source_pending: bool
    gold_available: bool
    corpus_only: bool
    invalid_gold_annotation: bool
    expected_ecli: str | None
    expected_answer_points: list[str]
    expected_keywords: list[str]


@dataclass(frozen=True)
class AnswerEvalResult:
    question_id: str
    question: str
    corpus: str
    source_pending: bool
    gold_available: bool
    gold_source_hit_at_1: bool | None
    gold_source_hit_at_3: bool | None
    gold_source_hit_at_5: bool | None
    support_level: str
    expected_answer_points: list[str]
    answer_skeleton: str
    citation_required: bool
    citation_available: bool
    answer_eval_status: str
    failure_reason: str | None = None
    gold_ecli: str | None = None
    gold_chunk_id: str | None = None
    gold_hit_rank: int | None = None
    support_keyword_coverage: float | None = None
    unsupported_answer_risk: bool = False
    evidence_window_enabled: bool = False
    evidence_window_anchor_chunk_id: str | None = None
    evidence_window_chunk_ids: list[str] = field(default_factory=list)
    evidence_window_chunk_indexes: list[int] = field(default_factory=list)
    evidence_window_document_id: str | None = None
    evidence_window_source: str | None = None
    evidence_window_truncated: bool = False
    evidence_window_provenance_valid: bool | None = None
    evidence_window_failure_reason: str | None = None
    evidence_window_construction_reason: str | None = None
    evidence_window_missing_neighbors: list[int] = field(default_factory=list)
    original_snippet_length: int | None = None
    combined_evidence_length: int | None = None


@dataclass(frozen=True)
class AnswerEvalMetrics:
    total_question_count: int
    gold_question_count: int
    missing_gold_count: int
    evaluable_question_count: int
    not_evaluable_missing_gold_count: int
    direct_support_count: int
    partial_support_count: int
    gap_count: int
    boilerplate_noise_count: int
    corpus_only_count: int
    citation_available_count: int
    citation_available_rate_gold: float
    corpus_routing_support_rate: float
    strict_direct_pass_rate_all: float
    strict_direct_pass_rate_gold: float
    usable_support_rate_gold: float
    unsupported_risk_rate_gold: float
    gold_retrieval_miss_count: int
    gold_retrieval_miss_rate: float
    answer_eval_pass_rate: float
    answer_eval_partial_rate: float
    answer_eval_gap_rate: float
    unsupported_answer_risk_count: int
    skipped_count: int
    needs_review_count: int
    evidence_window_used_count: int = 0
    evidence_window_failed_count: int = 0
    evidence_window_truncated_count: int = 0
    same_document_neighbor_count: int = 0

    @property
    def total_questions(self) -> int:
        return self.total_question_count

    @property
    def gold_available_count(self) -> int:
        return self.gold_question_count

    @property
    def citation_available_rate(self) -> float:
        return self.citation_available_rate_gold


@dataclass(frozen=True)
class FailedCaseReportEntry:
    run_id: str
    timestamp: str
    question_id: str
    dataset: str
    gold_type: str
    question: str
    expected_answer: str
    actual_answer: str
    normalized_expected_answer: str
    normalized_actual_answer: str
    expected_source: str
    retrieved_sources: list[str]
    citations: list[str]
    strict_direct_pass: bool
    usable_support: bool
    citation_available: bool
    unsupported_answer: bool
    failure_reason: str
    failure_category: str
    support_level: str
    answer_eval_status: str
    corpus: str
    is_real_failure: bool


def infer_corpus_from_run_name(run_name: str) -> str:
    name = run_name.lower()
    if name.startswith("mixed"):
        return "mixed"
    if name.startswith("nsoud"):
        return "nsoud"
    if name.startswith("usoud"):
        return "usoud"
    raise RetrievalConfigurationError(f"Cannot infer corpus from run name: {run_name!r}")


def build_summary_json_payload(
    *,
    run_name: str,
    metrics: AnswerEvalMetrics,
    generated_at: str,
    corpus: str | None = None,
) -> dict[str, Any]:
    resolved_corpus = corpus or infer_corpus_from_run_name(run_name)
    return {
        "generated_at": generated_at,
        "run_name": run_name,
        "corpus": resolved_corpus,
        "gold": metrics.gold_question_count,
        "total_question_count": metrics.total_question_count,
        "gold_question_count": metrics.gold_question_count,
        "missing_gold_count": metrics.missing_gold_count,
        "evaluable_question_count": metrics.evaluable_question_count,
        "not_evaluable_missing_gold_count": metrics.not_evaluable_missing_gold_count,
        "direct_support_count": metrics.direct_support_count,
        "partial_support_count": metrics.partial_support_count,
        "gap_count": metrics.gap_count,
        "boilerplate_noise_count": metrics.boilerplate_noise_count,
        "corpus_only_count": metrics.corpus_only_count,
        "citation_available_count": metrics.citation_available_count,
        "unsupported_answer_risk_count": metrics.unsupported_answer_risk_count,
        "unsupported_risk_rate_gold": metrics.unsupported_risk_rate_gold,
        "gold_retrieval_miss_count": metrics.gold_retrieval_miss_count,
        "gold_retrieval_miss_rate": metrics.gold_retrieval_miss_rate,
        "corpus_routing_support_rate": metrics.corpus_routing_support_rate,
        "strict_direct_pass_rate_all": metrics.strict_direct_pass_rate_all,
        "strict_direct_pass_rate_gold": metrics.strict_direct_pass_rate_gold,
        "usable_support_rate_gold": metrics.usable_support_rate_gold,
        "citation_available_rate_gold": metrics.citation_available_rate_gold,
        "citation_available_rate": metrics.citation_available_rate_gold,
        "evidence_window_used_count": metrics.evidence_window_used_count,
        "evidence_window_failed_count": metrics.evidence_window_failed_count,
        "evidence_window_truncated_count": metrics.evidence_window_truncated_count,
        "same_document_neighbor_count": metrics.same_document_neighbor_count,
    }


def normalize_answer_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("\xa0", " ").strip().lower()
    normalized = re.sub(r"\b(rub|руб\.?|рубля|рублей|р\.?)\b", " rub ", normalized)
    normalized = re.sub(r"(?<=\d)[\s\u00a0](?=\d{3}\b)", "", normalized)
    normalized = re.sub(r"(?<=\d),(?=\d)", ".", normalized)
    normalized = re.sub(r"[\"'`´]", "", normalized)
    normalized = re.sub(r"[.,;:!?()\[\]{}]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def load_retrieval_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise RetrievalConfigurationError(f"Retrieval results not found: {path}")
    by_id: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RetrievalConfigurationError(
                    f"Invalid JSON in {path} line {line_number}: {exc}"
                ) from exc
            question_id = str(payload.get("id") or "").strip()
            if not question_id:
                raise RetrievalConfigurationError(f"Missing id in {path} line {line_number}.")
            by_id[question_id] = payload
    if not by_id:
        raise RetrievalConfigurationError(f"No retrieval results in {path}.")
    return by_id


def load_gold_registry_from_dataset(items: list[LegalQaItem]) -> dict[str, GoldRegistryEntry]:
    registry: dict[str, GoldRegistryEntry] = {}
    for item in items:
        constraints = item.expected_source_constraints
        ecli = str(constraints.source_document_id or "").strip() or None
        corpus_only = not item.source_pending and item.corpus == "mixed" and not ecli
        invalid_gold_annotation = not item.source_pending and not corpus_only and not ecli
        gold_available = not item.source_pending and (bool(ecli) or corpus_only)
        registry[item.id] = GoldRegistryEntry(
            question_id=item.id,
            corpus=item.corpus,
            source_pending=item.source_pending,
            gold_available=gold_available,
            corpus_only=corpus_only,
            invalid_gold_annotation=invalid_gold_annotation,
            expected_ecli=ecli,
            expected_answer_points=list(item.expected_answer_points),
            expected_keywords=list(item.expected_keywords),
        )
    return registry


def validate_gold_review_path(path: Path) -> None:
    if not path.exists():
        raise RetrievalConfigurationError(f"Gold review file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if "Gold Source Review" not in text:
        raise RetrievalConfigurationError(f"Not a gold review document: {path}")


def _parse_chunk_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    raw = metadata.get("chunk_metadata")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return dict(decoded) if isinstance(decoded, dict) else {}
    return {}


def hit_document_id(hit: dict[str, Any]) -> str | None:
    metadata = dict(hit.get("metadata") or {})
    chunk_meta = _parse_chunk_metadata(metadata)
    for key in ("source_document_id", "document_id", "ecli"):
        value = hit.get(key) or metadata.get(key) or chunk_meta.get(key) or chunk_meta.get(
            "source_document_id"
        )
        if value not in {None, ""}:
            return str(value).strip()
    return None


def hit_matches_gold_ecli(hit: dict[str, Any], expected_ecli: str) -> bool:
    document_id = hit_document_id(hit)
    if not document_id:
        return False
    return normalize_for_match(document_id) == normalize_for_match(expected_ecli)


def gold_source_hit_at_k(hits: list[dict[str, Any]], expected_ecli: str, k: int) -> bool:
    return any(hit_matches_gold_ecli(hit, expected_ecli) for hit in hits[:k])


def _keyword_coverage_in_text(expected_keywords: list[str], text: str) -> float:
    if not expected_keywords:
        return 0.0
    normalized = normalize_for_match(text)
    matched = sum(1 for keyword in expected_keywords if normalize_for_match(keyword) in normalized)
    return matched / len(expected_keywords)


def is_boilerplate_snippet(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) < MIN_SUBSTANTIVE_SNIPPET_LEN:
        return True
    return any(pattern.search(cleaned) for pattern in BOILERPLATE_PATTERNS)


def _best_gold_hit(hits: list[dict[str, Any]], expected_ecli: str) -> dict[str, Any] | None:
    matching = [hit for hit in hits if hit_matches_gold_ecli(hit, expected_ecli)]
    if not matching:
        return None
    return min(matching, key=lambda hit: int(hit.get("rank") or 9999))


def _format_citation(hit: dict[str, Any] | None, expected_ecli: str | None) -> str:
    if hit is None:
        return expected_ecli or ""
    document_id = hit_document_id(hit) or expected_ecli or ""
    chunk_id = str(hit.get("chunk_id") or "").strip()
    if document_id and chunk_id:
        return f"{document_id} (chunk {chunk_id})"
    return document_id or chunk_id


def classify_support_level(
    *,
    gold: GoldRegistryEntry,
    hits: list[dict[str, Any]],
    evidence_text_by_chunk_id: dict[str, str] | None = None,
) -> tuple[str, dict[str, Any] | None, float | None, int | None]:
    if gold.corpus_only:
        return "corpus_only", None, None, None

    if not gold.expected_ecli:
        return "gap", None, None, None

    gold_hit = _best_gold_hit(hits, gold.expected_ecli)
    if gold_hit is None:
        return "gap", None, None, None

    chunk_id = str(gold_hit.get("chunk_id") or "").strip()
    using_evidence_window = bool(evidence_text_by_chunk_id and chunk_id in evidence_text_by_chunk_id)
    snippet = (
        evidence_text_by_chunk_id[chunk_id]
        if using_evidence_window
        else str(gold_hit.get("text_snippet") or "")
    )
    rank = int(gold_hit.get("rank") or 9999)
    coverage = _keyword_coverage_in_text(gold.expected_keywords, snippet)
    if is_boilerplate_snippet(snippet) and not (using_evidence_window and coverage > 0.0):
        return "boilerplate_noise", gold_hit, coverage, rank

    if rank == 1 and coverage >= 0.67:
        return "direct", gold_hit, coverage, rank
    if coverage > 0.0:
        return "partial", gold_hit, coverage, rank
    if rank <= 3:
        return "partial", gold_hit, coverage, rank
    return "gap", gold_hit, coverage, rank


def build_answer_skeleton(
    *,
    support_level: str,
    gold: GoldRegistryEntry,
    gold_hit: dict[str, Any] | None,
    citation_required: bool,
) -> tuple[str, bool]:
    citation = _format_citation(gold_hit, gold.expected_ecli)
    citation_available = bool(citation) and support_level in {"direct", "partial"}

    if support_level == "direct":
        lead = gold.expected_answer_points[0] if gold.expected_answer_points else gold.question_id
        if citation_required and citation_available:
            return f"Na základě ověřeného zdroje [{citation}]: {lead}", citation_available
        return f"Na základě ověřeného zdroje: {lead}", citation_available

    if support_level == "partial":
        lead = gold.expected_answer_points[0] if gold.expected_answer_points else "téma otázky"
        suffix = f" [{citation}]" if citation_available else ""
        return (
            f"Částečná podpora — opatrná odpověď vyžaduje další ověření{suffix}: {lead}",
            citation_available,
        )

    if support_level == "corpus_only":
        return (
            "Korpusově relevantní kontext nalezen; přesná citace dokumentu není k dispozici.",
            False,
        )

    if support_level == "boilerplate_noise":
        return (
            "Retrieved zdroj neobsahuje dostatečný substantivní text pro bezpečnou odpověď.",
            False,
        )

    return "Nedostatečná podpora v retrieved zdrojích pro bezpečnou odpověď.", False


def map_support_to_status(support_level: str, *, citation_required: bool, citation_available: bool) -> str:
    if support_level == "direct":
        if citation_required and not citation_available:
            return "needs_review"
        return "pass"
    if support_level == "partial":
        return "partial"
    if support_level == "corpus_only":
        return "partial"
    if support_level == "boilerplate_noise":
        return "needs_review"
    return "gap"


def _evidence_text_by_chunk_id(window: EvidenceWindow) -> dict[str, str]:
    if not window.used:
        return {}
    return {
        chunk_id: window.combined_text
        for chunk_id in window.ordered_chunk_ids
        if chunk_id == window.anchor_chunk_id
    }


def _evidence_window_result_fields(
    *,
    window_config: EvidenceWindowConfig,
    evidence_window: EvidenceWindow | None,
    gold_hit: dict[str, Any] | None,
) -> dict[str, Any]:
    original_snippet = str(gold_hit.get("text_snippet") or "") if gold_hit else ""
    if not window_config.enabled or evidence_window is None:
        return {
            "evidence_window_enabled": window_config.enabled,
            "original_snippet_length": len(original_snippet) if gold_hit else None,
            "combined_evidence_length": len(original_snippet) if gold_hit else None,
        }
    return {
        "evidence_window_enabled": True,
        "evidence_window_anchor_chunk_id": evidence_window.anchor_chunk_id,
        "evidence_window_chunk_ids": list(evidence_window.ordered_chunk_ids),
        "evidence_window_chunk_indexes": list(evidence_window.ordered_chunk_indexes),
        "evidence_window_document_id": evidence_window.document_id,
        "evidence_window_source": evidence_window.source,
        "evidence_window_truncated": evidence_window.truncated,
        "evidence_window_provenance_valid": evidence_window.provenance_valid,
        "evidence_window_failure_reason": evidence_window.failure_reason,
        "evidence_window_construction_reason": evidence_window.construction_reason,
        "evidence_window_missing_neighbors": list(evidence_window.missing_neighbors),
        "original_snippet_length": len(original_snippet) if gold_hit else None,
        "combined_evidence_length": len(evidence_window.combined_text),
    }


def evaluate_answer_item(
    *,
    item: LegalQaItem,
    gold: GoldRegistryEntry,
    retrieval: dict[str, Any],
    citation_required: bool,
    evidence_window_config: EvidenceWindowConfig | None = None,
    evidence_sidecar_path: Path | None = None,
) -> AnswerEvalResult:
    hits = list(retrieval.get("hits") or [])
    window_config = evidence_window_config or EvidenceWindowConfig(enabled=False)

    if gold.source_pending or not gold.gold_available:
        return AnswerEvalResult(
            question_id=item.id,
            question=item.question,
            corpus=item.corpus,
            source_pending=item.source_pending,
            gold_available=False,
            gold_source_hit_at_1=None,
            gold_source_hit_at_3=None,
            gold_source_hit_at_5=None,
            support_level="gap",
            expected_answer_points=list(item.expected_answer_points),
            answer_skeleton="",
            citation_required=citation_required,
            citation_available=False,
            answer_eval_status="skipped",
            failure_reason=(
                "invalid gold annotation"
                if gold.invalid_gold_annotation
                else "source_pending or no gold annotation"
            ),
            evidence_window_enabled=window_config.enabled,
            evidence_window_provenance_valid=None if window_config.enabled else None,
        )

    if gold.corpus_only:
        support_level = "corpus_only"
        gold_hit = None
        hit_at_1 = None
        hit_at_3 = bool(retrieval.get("corpus_hit_at_3"))
        hit_at_5 = bool(retrieval.get("corpus_hit_at_5"))
        support_keyword_coverage = None
        gold_hit_rank = None
    else:
        assert gold.expected_ecli is not None
        hit_at_1 = gold_source_hit_at_k(hits, gold.expected_ecli, 1)
        hit_at_3 = gold_source_hit_at_k(hits, gold.expected_ecli, 3)
        hit_at_5 = gold_source_hit_at_k(hits, gold.expected_ecli, 5)
        gold_hit = _best_gold_hit(hits, gold.expected_ecli)
        evidence_window = build_evidence_window(
            anchor_hit=gold_hit,
            hits=hits,
            config=window_config,
            sidecar_path=evidence_sidecar_path,
        )
        evidence_text_by_chunk_id = _evidence_text_by_chunk_id(evidence_window)
        support_level, gold_hit, support_keyword_coverage, gold_hit_rank = classify_support_level(
            gold=gold,
            hits=hits,
            evidence_text_by_chunk_id=evidence_text_by_chunk_id,
        )

    answer_skeleton, citation_available = build_answer_skeleton(
        support_level=support_level,
        gold=gold,
        gold_hit=gold_hit,
        citation_required=citation_required,
    )
    status = map_support_to_status(
        support_level,
        citation_required=citation_required,
        citation_available=citation_available,
    )

    failure_reason: str | None = None
    if status == "gap":
        failure_reason = "Insufficient retrieved support for a safe answer skeleton."
    elif status == "needs_review":
        failure_reason = "Boilerplate/noise or missing citation for required answer support."
    elif status == "partial":
        failure_reason = "Only partial support; cautious answer skeleton only."

    unsupported_risk = support_level in {"gap", "boilerplate_noise"} or (
        citation_required and not citation_available and support_level == "direct"
    )

    return AnswerEvalResult(
        question_id=item.id,
        question=item.question,
        corpus=item.corpus,
        source_pending=False,
        gold_available=True,
        gold_source_hit_at_1=hit_at_1,
        gold_source_hit_at_3=hit_at_3,
        gold_source_hit_at_5=hit_at_5,
        support_level=support_level,
        expected_answer_points=list(item.expected_answer_points),
        answer_skeleton=answer_skeleton,
        citation_required=citation_required,
        citation_available=citation_available,
        answer_eval_status=status,
        failure_reason=failure_reason,
        gold_ecli=gold.expected_ecli,
        gold_chunk_id=str(gold_hit.get("chunk_id")) if gold_hit else None,
        gold_hit_rank=gold_hit_rank,
        support_keyword_coverage=support_keyword_coverage,
        unsupported_answer_risk=unsupported_risk,
        **_evidence_window_result_fields(
            window_config=window_config,
            evidence_window=evidence_window if not gold.corpus_only else None,
            gold_hit=gold_hit,
        ),
    )


def run_answer_eval(
    *,
    items: list[LegalQaItem],
    registry: dict[str, GoldRegistryEntry],
    retrieval_by_id: dict[str, dict[str, Any]],
    citation_required: bool,
    limit: int | None = None,
    evidence_window_config: EvidenceWindowConfig | None = None,
    evidence_sidecar_path: Path | None = None,
) -> list[AnswerEvalResult]:
    results: list[AnswerEvalResult] = []
    selected = items[:limit] if limit is not None else items
    for item in selected:
        if item.id not in retrieval_by_id:
            raise RetrievalConfigurationError(f"Missing retrieval result for {item.id!r}.")
        gold = registry[item.id]
        results.append(
            evaluate_answer_item(
                item=item,
                gold=gold,
                retrieval=retrieval_by_id[item.id],
                citation_required=citation_required,
                evidence_window_config=evidence_window_config,
                evidence_sidecar_path=evidence_sidecar_path,
            )
        )
    return results


def aggregate_answer_metrics(results: list[AnswerEvalResult]) -> AnswerEvalMetrics:
    total = len(results)
    if total == 0:
        raise RetrievalConfigurationError("No answer eval results to aggregate.")

    gold_results = [result for result in results if result.gold_available]
    gold_count = len(gold_results)
    missing_gold_count = total - gold_count

    def _count_status(status: str) -> int:
        return sum(1 for result in results if result.answer_eval_status == status)

    def _count_support(level: str) -> int:
        return sum(1 for result in gold_results if result.support_level == level)

    direct_support_count = _count_support("direct")
    partial_support_count = _count_support("partial")
    gap_count = _count_support("gap")
    boilerplate_noise_count = _count_support("boilerplate_noise")
    corpus_only_count = _count_support("corpus_only")
    citation_available_count = sum(1 for r in gold_results if r.citation_available)
    unsupported_answer_risk_count = sum(1 for r in gold_results if r.unsupported_answer_risk)
    gold_retrieval_miss_count = sum(
        1
        for result in gold_results
        if result.gold_ecli and result.gold_source_hit_at_5 is False
    )
    usable_support_gold = sum(
        1
        for result in gold_results
        if result.support_level in {"direct", "partial", "corpus_only"}
    )
    evidence_window_results = [
        result for result in gold_results if result.evidence_window_enabled
    ]
    evidence_window_used_count = sum(
        1 for result in evidence_window_results if result.evidence_window_provenance_valid is True
    )
    evidence_window_failed_count = sum(
        1 for result in evidence_window_results if result.evidence_window_provenance_valid is False
    )
    evidence_window_truncated_count = sum(
        1 for result in evidence_window_results if result.evidence_window_truncated
    )
    same_document_neighbor_count = sum(
        max(0, len(result.evidence_window_chunk_ids) - 1)
        for result in evidence_window_results
        if result.evidence_window_provenance_valid is True
    )

    strict_direct_pass_rate_all = direct_support_count / total
    strict_direct_pass_rate_gold = direct_support_count / gold_count if gold_count else 0.0
    usable_support_rate_gold = usable_support_gold / gold_count if gold_count else 0.0
    citation_available_rate_gold = citation_available_count / gold_count if gold_count else 0.0
    unsupported_risk_rate_gold = (
        unsupported_answer_risk_count / gold_count if gold_count else 0.0
    )
    gold_retrieval_miss_rate = (
        gold_retrieval_miss_count / gold_count if gold_count else 0.0
    )
    corpus_routing_support_rate = corpus_only_count / gold_count if gold_count else 0.0

    return AnswerEvalMetrics(
        total_question_count=total,
        gold_question_count=gold_count,
        missing_gold_count=missing_gold_count,
        evaluable_question_count=gold_count,
        not_evaluable_missing_gold_count=missing_gold_count,
        direct_support_count=direct_support_count,
        partial_support_count=partial_support_count,
        gap_count=gap_count,
        boilerplate_noise_count=boilerplate_noise_count,
        corpus_only_count=corpus_only_count,
        citation_available_count=citation_available_count,
        citation_available_rate_gold=citation_available_rate_gold,
        corpus_routing_support_rate=corpus_routing_support_rate,
        strict_direct_pass_rate_all=strict_direct_pass_rate_all,
        strict_direct_pass_rate_gold=strict_direct_pass_rate_gold,
        usable_support_rate_gold=usable_support_rate_gold,
        unsupported_risk_rate_gold=unsupported_risk_rate_gold,
        gold_retrieval_miss_count=gold_retrieval_miss_count,
        gold_retrieval_miss_rate=gold_retrieval_miss_rate,
        answer_eval_pass_rate=strict_direct_pass_rate_all,
        answer_eval_partial_rate=_count_status("partial") / total,
        answer_eval_gap_rate=_count_status("gap") / total,
        unsupported_answer_risk_count=unsupported_answer_risk_count,
        skipped_count=_count_status("skipped"),
        needs_review_count=_count_status("needs_review"),
        evidence_window_used_count=evidence_window_used_count,
        evidence_window_failed_count=evidence_window_failed_count,
        evidence_window_truncated_count=evidence_window_truncated_count,
        same_document_neighbor_count=same_document_neighbor_count,
    )


def _joined_expected_answer(item: LegalQaItem) -> str:
    return " ".join(point.strip() for point in item.expected_answer_points if point.strip()).strip()


def _gold_type(gold: GoldRegistryEntry) -> str:
    if gold.gold_available:
        return "gold"
    if gold.source_pending:
        return "non_gold"
    return "unknown"


def _retrieved_sources(hits: list[dict[str, Any]], *, limit: int = 5) -> list[str]:
    sources: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        document_id = hit_document_id(hit)
        chunk_id = str(hit.get("chunk_id") or "").strip()
        value = document_id or chunk_id
        if not value or value in seen:
            continue
        seen.add(value)
        sources.append(value)
        if len(sources) >= limit:
            break
    return sources


def _retrieved_citations(hits: list[dict[str, Any]], *, limit: int = 5) -> list[str]:
    citations: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        citation = _format_citation(hit, None)
        if not citation or citation in seen:
            continue
        seen.add(citation)
        citations.append(citation)
        if len(citations) >= limit:
            break
    return citations


def _is_dataset_filter_mismatch(run_corpus: str, item_corpus: str) -> bool:
    if run_corpus == "mixed":
        return item_corpus != "mixed"
    return item_corpus != run_corpus


def _is_real_failure_category(category: str) -> bool:
    return category in {
        "invalid_gold_annotation",
        "true_retrieval_miss",
        "unsupported_boilerplate_or_gap",
    }


def classify_failure_category(
    *,
    item: LegalQaItem,
    gold: GoldRegistryEntry,
    result: AnswerEvalResult,
    run_corpus: str,
) -> str:
    expected_answer = _joined_expected_answer(item)
    actual_answer = result.answer_skeleton.strip()
    normalized_expected = normalize_answer_text(expected_answer)
    normalized_actual = normalize_answer_text(actual_answer)

    if _is_dataset_filter_mismatch(run_corpus, item.corpus):
        return "invalid_gold_annotation"
    if not expected_answer:
        return "invalid_gold_annotation"
    if gold.invalid_gold_annotation:
        return "invalid_gold_annotation"
    if gold.source_pending or not gold.gold_available:
        return "not_evaluable_missing_gold"
    if gold.corpus_only and result.support_level == "corpus_only":
        return "corpus_only_no_document_citation_expected"
    if result.support_level == "boilerplate_noise":
        return "unsupported_boilerplate_or_gap"
    if (
        normalized_expected
        and normalized_actual
        and normalized_expected == normalized_actual
        and result.answer_eval_status != "pass"
    ):
        return "usable_partial_support"
    if result.answer_eval_status == "gap":
        if result.gold_source_hit_at_5 is False:
            return "true_retrieval_miss"
        return "unsupported_boilerplate_or_gap"
    if result.answer_eval_status == "partial":
        if result.support_level == "partial":
            if (result.support_keyword_coverage or 0.0) > 0.0 and result.citation_available:
                return "usable_partial_support"
            return "weak_partial_support"
        if result.support_level == "corpus_only":
            return "corpus_only_no_document_citation_expected"
    if result.answer_eval_status != "pass":
        return "unsupported_boilerplate_or_gap"
    return "unknown_failure"


def build_failed_case_report_entries(
    *,
    run_name: str,
    generated_at: str,
    dataset_path: Path,
    corpus: str,
    items: list[LegalQaItem],
    registry: dict[str, GoldRegistryEntry],
    retrieval_by_id: dict[str, dict[str, Any]],
    results: list[AnswerEvalResult],
) -> list[FailedCaseReportEntry]:
    item_by_id = {item.id: item for item in items}
    entries: list[FailedCaseReportEntry] = []
    for result in results:
        if result.answer_eval_status not in NON_PASS_STATUSES:
            continue

        item = item_by_id[result.question_id]
        gold = registry[result.question_id]
        retrieval = retrieval_by_id[result.question_id]
        hits = list(retrieval.get("hits") or [])
        expected_answer = _joined_expected_answer(item)
        actual_answer = result.answer_skeleton.strip()
        category = classify_failure_category(
            item=item,
            gold=gold,
            result=result,
            run_corpus=corpus,
        )
        entries.append(
            FailedCaseReportEntry(
                run_id=run_name,
                timestamp=generated_at,
                question_id=result.question_id,
                dataset=str(dataset_path),
                gold_type=_gold_type(gold),
                question=result.question,
                expected_answer=expected_answer,
                actual_answer=actual_answer,
                normalized_expected_answer=normalize_answer_text(expected_answer),
                normalized_actual_answer=normalize_answer_text(actual_answer),
                expected_source=gold.expected_ecli or "",
                retrieved_sources=_retrieved_sources(hits),
                citations=_retrieved_citations(hits),
                strict_direct_pass=result.answer_eval_status == "pass",
                usable_support=result.support_level in {"direct", "partial", "corpus_only"},
                citation_available=result.citation_available,
                unsupported_answer=result.unsupported_answer_risk,
                failure_reason=result.failure_reason or category,
                failure_category=category,
                support_level=result.support_level,
                answer_eval_status=result.answer_eval_status,
                corpus=result.corpus,
                is_real_failure=_is_real_failure_category(category),
            )
        )
    return entries


def build_metric_failure_categories(
    *,
    failed_cases: list[FailedCaseReportEntry],
    metrics: AnswerEvalMetrics,
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for entry in failed_cases:
        counts[entry.failure_category] += 1
    if metrics.gold_question_count < MIN_STABLE_GOLD_COUNT:
        counts["metric_denominator_warning"] += 1
    return dict(sorted(counts.items()))


def determine_final_status(
    *,
    corpus: str,
    failure_category_counts: dict[str, int],
) -> tuple[str, str]:
    real_failures = sum(
        failure_category_counts.get(category, 0)
        for category in (
            "true_retrieval_miss",
            "invalid_gold_annotation",
            "unsupported_boilerplate_or_gap",
        )
    )
    warning_only = sum(
        failure_category_counts.get(category, 0)
        for category in (
            "not_evaluable_missing_gold",
            "usable_partial_support",
            "weak_partial_support",
            "corpus_only_no_document_citation_expected",
            "metric_denominator_warning",
        )
    )
    if real_failures > 0:
        if corpus == "nsoud" and (
            failure_category_counts.get("true_retrieval_miss", 0) > 0
            or failure_category_counts.get("unsupported_boilerplate_or_gap", 0) > 0
        ):
            return (
                "FAIL_WITH_REAL_NSOUD_RISK",
                "NSoud contains real unsupported or retrieval-risk items that require manual review.",
            )
        return ("FAIL", "Real gold-evaluable failures are present.")
    if warning_only > 0:
        return (
            "WARN",
            "Main issues are missing gold coverage, conservative strict-pass gating, or small denominators.",
        )
    return ("PASS", "No real failures detected and coverage is sufficient.")


def build_nsoud_qa_007_diagnostic(
    *,
    items: list[LegalQaItem],
    registry: dict[str, GoldRegistryEntry],
    retrieval_by_id: dict[str, dict[str, Any]],
    results: list[AnswerEvalResult],
) -> dict[str, Any] | None:
    question_id = "nsoud-qa-007"
    if question_id not in registry or question_id not in retrieval_by_id:
        return None
    result_by_id = {result.question_id: result for result in results}
    if question_id not in result_by_id:
        return None

    result = result_by_id[question_id]
    gold = registry[question_id]
    retrieval = retrieval_by_id[question_id]
    hits = list(retrieval.get("hits") or [])
    retrieved_top_k_ids = _retrieved_sources(hits, limit=10)
    expected_source_id = gold.expected_ecli or ""
    expected_source_present_top_k = bool(
        expected_source_id
        and any(
            normalize_for_match(source_id) == normalize_for_match(expected_source_id)
            for source_id in retrieved_top_k_ids
        )
    )
    criminal_tdo_hits = sum(1 for source_id in retrieved_top_k_ids[:5] if ".TDO." in source_id.upper())
    matcher_issue = False
    gold_annotation_mismatch = gold.invalid_gold_annotation
    answer_support_gap = expected_source_present_top_k and result.support_level == "gap"
    true_retrieval_miss = bool(expected_source_id) and not expected_source_present_top_k
    question_too_generic = bool(true_retrieval_miss and criminal_tdo_hits >= 3 and retrieval.get("keyword_coverage", 0) >= 1.0)
    conclusion = (
        "Conservative conclusion: true_retrieval_miss. Expected ECLI is absent from retrieved top-k. "
        "The query also appears generic across multiple criminal dovolani decisions, so genericity may contribute, "
        "but there is no evidence of matcher mismatch."
        if true_retrieval_miss
        else "Conservative conclusion: expected source is present; this is not a true retrieval miss."
    )
    return {
        "question_id": question_id,
        "gold_source_id": expected_source_id,
        "retrieved_top_k_ids": retrieved_top_k_ids,
        "expected_source_present_top_k": expected_source_present_top_k,
        "true_retrieval_miss": true_retrieval_miss,
        "gold_annotation_mismatch": gold_annotation_mismatch,
        "answer_support_gap": answer_support_gap,
        "matcher_issue": matcher_issue,
        "question_too_generic": question_too_generic,
        "conclusion": conclusion,
    }


def build_metrics_summary_payload(
    *,
    run_name: str,
    corpus: str,
    generated_at: str,
    dataset_path: Path,
    retrieval_results_path: Path,
    gold_review_path: Path,
    metrics: AnswerEvalMetrics,
    failure_category_counts: dict[str, int],
    final_status: str,
    status_reason: str,
) -> dict[str, Any]:
    gold_count = metrics.gold_question_count
    denominator_warning = (
        f"Gold denominator is small ({gold_count}); percentage metrics are unstable."
        if gold_count < MIN_STABLE_GOLD_COUNT
        else None
    )
    return {
        "generated_at": generated_at,
        "run_name": run_name,
        "corpus": corpus,
        "dataset": str(dataset_path),
        "retrieval_results": str(retrieval_results_path),
        "gold_review": str(gold_review_path),
        "metrics": build_summary_json_payload(
            run_name=run_name,
            metrics=metrics,
            generated_at=generated_at,
            corpus=corpus,
        ),
        "total_evaluated_count": metrics.total_question_count,
        "gold_count": gold_count,
        "failure_category_counts": failure_category_counts,
        "denominator_warning": denominator_warning,
        "final_status": final_status,
        "status_reason": status_reason,
        "interpretation_notes": [
            "strict_direct_pass_rate is intentionally conservative",
            "usable_support_rate_gold is the practical support metric",
            "missing gold does not mean retrieval failure",
            "corpus_only mixed items are not expected to have document citations",
        ],
    }


def write_failed_case_diagnostics(
    *,
    output_dir: Path,
    run_name: str,
    corpus: str,
    generated_at: str,
    dataset_path: Path,
    retrieval_results_path: Path,
    gold_review_path: Path,
    metrics: AnswerEvalMetrics,
    failed_cases: list[FailedCaseReportEntry],
    nsoud_qa_007_diagnostic: dict[str, Any] | None = None,
) -> None:
    failure_category_counts = build_metric_failure_categories(
        failed_cases=failed_cases,
        metrics=metrics,
    )
    final_status, status_reason = determine_final_status(
        corpus=corpus,
        failure_category_counts=failure_category_counts,
    )
    metrics_summary = build_metrics_summary_payload(
        run_name=run_name,
        corpus=corpus,
        generated_at=generated_at,
        dataset_path=dataset_path,
        retrieval_results_path=retrieval_results_path,
        gold_review_path=gold_review_path,
        metrics=metrics,
        failure_category_counts=failure_category_counts,
        final_status=final_status,
        status_reason=status_reason,
    )

    (output_dir / "failed_cases_report.json").write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "run_name": run_name,
                "corpus": corpus,
                "failed_case_count": len(failed_cases),
                "failed_cases": [asdict(entry) for entry in failed_cases],
                "nsoud_qa_007_diagnostic": nsoud_qa_007_diagnostic,
                "final_status": final_status,
                "status_reason": status_reason,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "metric_failure_categories.json").write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "run_name": run_name,
                "corpus": corpus,
                "failure_category_counts": failure_category_counts,
                "final_status": final_status,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(metrics_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    practical_support = sum(
        1
        for entry in failed_cases
        if entry.failure_category in {"usable_partial_support", "weak_partial_support"}
    )
    retrieval_support = sum(
        1
        for entry in failed_cases
        if entry.failure_category in {"true_retrieval_miss", "unsupported_boilerplate_or_gap"}
    )
    gold_data = sum(
        1
        for entry in failed_cases
        if entry.failure_category in {"not_evaluable_missing_gold", "invalid_gold_annotation"}
    )
    top_failed_cases = failed_cases[:5]

    lines = [
        "# Failed cases diagnostic",
        "",
        f"- Run timestamp: {generated_at}",
        f"- Run name: `{run_name}`",
        f"- Evaluated datasets: `{dataset_path}`",
        f"- Metric names: strict_direct_pass_rate_all, strict_direct_pass_rate_gold, usable_support_rate_gold, citation_available_rate_gold, unsupported_risk_rate_gold, gold_retrieval_miss_rate, support breakdown, gold count",
        f"- Metric values: strict_all={metrics.strict_direct_pass_rate_all:.3f}, strict_gold={metrics.strict_direct_pass_rate_gold:.3f}, usable_gold={metrics.usable_support_rate_gold:.3f}, citation_rate_gold={metrics.citation_available_rate_gold:.3f}, unsupported_risk_gold={metrics.unsupported_risk_rate_gold:.3f}, retrieval_miss_gold={metrics.gold_retrieval_miss_rate:.3f}",
        f"- Gold count: {metrics.gold_question_count}",
        f"- Total evaluated count: {metrics.total_question_count}",
        "",
        "## Failure category breakdown",
        "",
    ]
    for category, count in failure_category_counts.items():
        lines.append(f"- {category}: {count}")
    lines.extend(
        [
            "",
            "## Top failed cases",
            "",
        ]
    )
    if not top_failed_cases:
        lines.append("- None")
    else:
        for entry in top_failed_cases:
            lines.append(
                f"- {entry.question_id}: {entry.failure_category} | status={entry.answer_eval_status} | support={entry.support_level}"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- strict_direct_pass_rate is intentionally conservative",
            "- usable_support_rate_gold is the practical support metric",
            "- missing gold does not mean retrieval failure",
            "- corpus_only mixed items are not expected to have document citations",
            "- NSoud unsupported-risk items require manual review",
            "",
            "## Cause assessment",
            "",
            f"- usable partial support diagnostics: {practical_support}",
            f"- retrieval/support caused failures: {retrieval_support}",
            f"- bad/incomplete gold data caused failures: {gold_data}",
            f"- re-ingest needed: {'unknown' if retrieval_support else 'no'}",
            f"- production Qdrant touched: no",
            f"- final status: {final_status}",
            f"- status reason: {status_reason}",
            "",
        ]
    )
    denominator_warning = metrics_summary["denominator_warning"]
    if denominator_warning:
        lines.extend(
            [
                "## Notes",
                "",
                f"- {denominator_warning}",
                "",
            ]
        )
    if nsoud_qa_007_diagnostic is not None:
        lines.extend(
            [
                "## nsoud-qa-007 diagnostic",
                "",
                f"- gold source id: {nsoud_qa_007_diagnostic['gold_source_id']}",
                f"- retrieved top-k ids: {', '.join(nsoud_qa_007_diagnostic['retrieved_top_k_ids'])}",
                f"- expected source absent in top-k: {not nsoud_qa_007_diagnostic['expected_source_present_top_k']}",
                f"- true_retrieval_miss: {nsoud_qa_007_diagnostic['true_retrieval_miss']}",
                f"- gold_annotation_mismatch: {nsoud_qa_007_diagnostic['gold_annotation_mismatch']}",
                f"- answer_support_gap: {nsoud_qa_007_diagnostic['answer_support_gap']}",
                f"- matcher_issue: {nsoud_qa_007_diagnostic['matcher_issue']}",
                f"- question_too_generic: {nsoud_qa_007_diagnostic['question_too_generic']}",
                f"- conclusion: {nsoud_qa_007_diagnostic['conclusion']}",
                "",
            ]
        )
    (output_dir / "failed_cases_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_answer_eval_outputs(
    *,
    output_dir: Path,
    dataset_path: Path,
    retrieval_results_path: Path,
    gold_review_path: Path,
    items: list[LegalQaItem],
    registry: dict[str, GoldRegistryEntry],
    retrieval_by_id: dict[str, dict[str, Any]],
    results: list[AnswerEvalResult],
    metrics: AnswerEvalMetrics,
    no_llm: bool,
    citation_required: bool,
    corpus: str | None = None,
    evidence_window_config: EvidenceWindowConfig | None = None,
    evidence_sidecar_path: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    results_path = output_dir / "answer_eval_results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    metrics_payload = {
        "generated_at": generated_at,
        "dataset": str(dataset_path),
        "retrieval_results": str(retrieval_results_path),
        "gold_review": str(gold_review_path),
        "no_llm": no_llm,
        "citation_required": citation_required,
        "corpus": corpus,
        "evidence_window": asdict(evidence_window_config)
        if evidence_window_config is not None
        else asdict(EvidenceWindowConfig(enabled=False)),
        "evidence_sidecar": str(evidence_sidecar_path) if evidence_sidecar_path else None,
        **asdict(metrics),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    failures = [
        result
        for result in results
        if result.answer_eval_status in {"gap", "needs_review"}
    ]
    failure_lines = [
        "# Answer eval failures",
        "",
        f"Generated: {generated_at}",
        f"Failed or needs review: {len(failures)} / {len(results)}",
        "",
    ]
    for result in failures:
        failure_lines.extend(
            [
                f"## {result.question_id}",
                f"- Question: {result.question}",
                f"- Status: {result.answer_eval_status}",
                f"- Support level: {result.support_level}",
                f"- Reason: {result.failure_reason}",
                "",
            ]
        )
    (output_dir / "failures.md").write_text("\n".join(failure_lines), encoding="utf-8")

    summary_lines = [
        "# No-LLM answer eval summary",
        "",
        f"- Generated: {generated_at}",
        f"- Dataset: `{dataset_path}`",
        f"- Retrieval results: `{retrieval_results_path}`",
        f"- Gold review: `{gold_review_path}`",
        f"- Mode: deterministic no-LLM",
        f"- Citation required: {citation_required}",
        f"- Evidence window enabled: {bool(evidence_window_config and evidence_window_config.enabled)}",
        "",
        "## Interpretation",
        "",
        "- `direct` = strict pass (document gold + snippet support)",
        "- `partial` = usable support, not full direct answer pass",
        "- `gap` / `boilerplate_noise` = must not generate a confident answer",
        "- `corpus_only` = corpus routing only, no document citation",
        "",
        "## Support breakdown (gold items)",
        "",
        f"- direct_support_count: {metrics.direct_support_count}",
        f"- partial_support_count: {metrics.partial_support_count}",
        f"- gap_count: {metrics.gap_count}",
        f"- boilerplate_noise_count: {metrics.boilerplate_noise_count}",
        f"- corpus_only_count: {metrics.corpus_only_count}",
        "",
        "## Rates",
        "",
        f"- strict_direct_pass_rate_all: {metrics.strict_direct_pass_rate_all:.3f}",
        f"- strict_direct_pass_rate_gold: {metrics.strict_direct_pass_rate_gold:.3f}",
        f"- usable_support_rate_gold: {metrics.usable_support_rate_gold:.3f}",
        f"- citation_available_rate_gold: {metrics.citation_available_rate_gold:.3f}",
        f"- corpus_routing_support_rate: {metrics.corpus_routing_support_rate:.3f}",
        f"- unsupported_risk_rate_gold: {metrics.unsupported_risk_rate_gold:.3f}",
        f"- gold_retrieval_miss_rate: {metrics.gold_retrieval_miss_rate:.3f}",
        f"- answer_eval_pass_rate (alias): {metrics.answer_eval_pass_rate:.3f}",
        f"- answer_eval_partial_rate: {metrics.answer_eval_partial_rate:.3f}",
        f"- answer_eval_gap_rate: {metrics.answer_eval_gap_rate:.3f}",
        "",
        "## Risk / coverage",
        "",
        f"- total questions: {metrics.total_question_count}",
        f"- gold available: {metrics.gold_question_count}",
        f"- missing gold: {metrics.missing_gold_count}",
        f"- evaluable questions: {metrics.evaluable_question_count}",
        f"- not evaluable (missing gold): {metrics.not_evaluable_missing_gold_count}",
        f"- citation available count: {metrics.citation_available_count}",
        f"- unsupported_answer_risk_count: {metrics.unsupported_answer_risk_count}",
        f"- gold_retrieval_miss_count: {metrics.gold_retrieval_miss_count}",
        f"- skipped: {metrics.skipped_count}",
        f"- needs review: {metrics.needs_review_count}",
        f"- evidence_window_used_count: {metrics.evidence_window_used_count}",
        f"- evidence_window_failed_count: {metrics.evidence_window_failed_count}",
        f"- evidence_window_truncated_count: {metrics.evidence_window_truncated_count}",
        f"- same_document_neighbor_count: {metrics.same_document_neighbor_count}",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    run_name = output_dir.name
    summary_json = build_summary_json_payload(
        run_name=run_name,
        metrics=metrics,
        generated_at=generated_at,
        corpus=corpus,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    resolved_corpus = corpus or infer_corpus_from_run_name(run_name)
    failed_cases = build_failed_case_report_entries(
        run_name=run_name,
        generated_at=generated_at,
        dataset_path=dataset_path,
        corpus=resolved_corpus,
        items=items,
        registry=registry,
        retrieval_by_id=retrieval_by_id,
        results=results,
    )
    nsoud_qa_007_diagnostic = None
    if resolved_corpus == "nsoud":
        nsoud_qa_007_diagnostic = build_nsoud_qa_007_diagnostic(
            items=items,
            registry=registry,
            retrieval_by_id=retrieval_by_id,
            results=results,
        )
    write_failed_case_diagnostics(
        output_dir=output_dir,
        run_name=run_name,
        corpus=resolved_corpus,
        generated_at=generated_at,
        dataset_path=dataset_path,
        retrieval_results_path=retrieval_results_path,
        gold_review_path=gold_review_path,
        metrics=metrics,
        failed_cases=failed_cases,
        nsoud_qa_007_diagnostic=nsoud_qa_007_diagnostic,
    )


def resolve_retrieval_results_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    joined = ", ".join(str(path) for path in candidates)
    raise RetrievalConfigurationError(f"No retrieval results found. Checked: {joined}")
