"""Deterministic no-LLM legal answer evaluation over gold retrieval results."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class GoldRegistryEntry:
    question_id: str
    corpus: str
    source_pending: bool
    gold_available: bool
    corpus_only: bool
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
    unsupported_answer_risk: bool = False


@dataclass(frozen=True)
class AnswerEvalMetrics:
    total_questions: int
    gold_available_count: int
    direct_support_count: int
    partial_support_count: int
    gap_count: int
    boilerplate_noise_count: int
    corpus_only_count: int
    citation_available_rate: float
    answer_eval_pass_rate: float
    answer_eval_partial_rate: float
    answer_eval_gap_rate: float
    unsupported_answer_risk_count: int
    skipped_count: int
    needs_review_count: int


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
        ecli = constraints.source_document_id
        corpus_only = not item.source_pending and item.corpus == "mixed" and not ecli
        gold_available = not item.source_pending and (bool(ecli) or corpus_only)
        registry[item.id] = GoldRegistryEntry(
            question_id=item.id,
            corpus=item.corpus,
            source_pending=item.source_pending,
            gold_available=gold_available,
            corpus_only=corpus_only,
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
) -> tuple[str, dict[str, Any] | None]:
    if gold.corpus_only:
        return "corpus_only", None

    if not gold.expected_ecli:
        return "gap", None

    gold_hit = _best_gold_hit(hits, gold.expected_ecli)
    if gold_hit is None:
        return "gap", None

    snippet = str(gold_hit.get("text_snippet") or "")
    if is_boilerplate_snippet(snippet):
        return "boilerplate_noise", gold_hit

    coverage = _keyword_coverage_in_text(gold.expected_keywords, snippet)
    rank = int(gold_hit.get("rank") or 9999)

    if rank == 1 and coverage >= 0.67:
        return "direct", gold_hit
    if coverage > 0.0:
        return "partial", gold_hit
    if rank <= 3:
        return "partial", gold_hit
    return "gap", gold_hit


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


def evaluate_answer_item(
    *,
    item: LegalQaItem,
    gold: GoldRegistryEntry,
    retrieval: dict[str, Any],
    citation_required: bool,
) -> AnswerEvalResult:
    hits = list(retrieval.get("hits") or [])

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
            failure_reason="source_pending or no gold annotation",
        )

    if gold.corpus_only:
        support_level = "corpus_only"
        gold_hit = None
        hit_at_1 = None
        hit_at_3 = bool(retrieval.get("corpus_hit_at_3"))
        hit_at_5 = bool(retrieval.get("corpus_hit_at_5"))
    else:
        assert gold.expected_ecli is not None
        hit_at_1 = gold_source_hit_at_k(hits, gold.expected_ecli, 1)
        hit_at_3 = gold_source_hit_at_k(hits, gold.expected_ecli, 3)
        hit_at_5 = gold_source_hit_at_k(hits, gold.expected_ecli, 5)
        support_level, gold_hit = classify_support_level(gold=gold, hits=hits)

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
        unsupported_answer_risk=unsupported_risk,
    )


def run_answer_eval(
    *,
    items: list[LegalQaItem],
    registry: dict[str, GoldRegistryEntry],
    retrieval_by_id: dict[str, dict[str, Any]],
    citation_required: bool,
    limit: int | None = None,
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
            )
        )
    return results


def aggregate_answer_metrics(results: list[AnswerEvalResult]) -> AnswerEvalMetrics:
    total = len(results)
    if total == 0:
        raise RetrievalConfigurationError("No answer eval results to aggregate.")

    gold_results = [result for result in results if result.gold_available]
    gold_count = len(gold_results)
    citation_denominator = [r for r in gold_results if r.citation_required]
    citation_available_rate = (
        sum(r.citation_available for r in citation_denominator) / len(citation_denominator)
        if citation_denominator
        else 0.0
    )

    def _count_status(status: str) -> int:
        return sum(1 for result in results if result.answer_eval_status == status)

    def _count_support(level: str) -> int:
        return sum(1 for result in gold_results if result.support_level == level)

    return AnswerEvalMetrics(
        total_questions=total,
        gold_available_count=gold_count,
        direct_support_count=_count_support("direct"),
        partial_support_count=_count_support("partial"),
        gap_count=_count_support("gap"),
        boilerplate_noise_count=_count_support("boilerplate_noise"),
        corpus_only_count=_count_support("corpus_only"),
        citation_available_rate=citation_available_rate,
        answer_eval_pass_rate=_count_status("pass") / total,
        answer_eval_partial_rate=_count_status("partial") / total,
        answer_eval_gap_rate=_count_status("gap") / total,
        unsupported_answer_risk_count=sum(1 for r in results if r.unsupported_answer_risk),
        skipped_count=_count_status("skipped"),
        needs_review_count=_count_status("needs_review"),
    )


def write_answer_eval_outputs(
    *,
    output_dir: Path,
    dataset_path: Path,
    retrieval_results_path: Path,
    gold_review_path: Path,
    results: list[AnswerEvalResult],
    metrics: AnswerEvalMetrics,
    no_llm: bool,
    citation_required: bool,
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
        "",
        "## Metrics",
        "",
        f"- total questions: {metrics.total_questions}",
        f"- gold available: {metrics.gold_available_count}",
        f"- direct support: {metrics.direct_support_count}",
        f"- partial support: {metrics.partial_support_count}",
        f"- gap: {metrics.gap_count}",
        f"- boilerplate noise: {metrics.boilerplate_noise_count}",
        f"- corpus only: {metrics.corpus_only_count}",
        f"- citation available rate: {metrics.citation_available_rate:.3f}",
        f"- answer eval pass rate: {metrics.answer_eval_pass_rate:.3f}",
        f"- answer eval partial rate: {metrics.answer_eval_partial_rate:.3f}",
        f"- answer eval gap rate: {metrics.answer_eval_gap_rate:.3f}",
        f"- unsupported answer risk: {metrics.unsupported_answer_risk_count}",
        f"- skipped: {metrics.skipped_count}",
        f"- needs review: {metrics.needs_review_count}",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def resolve_retrieval_results_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    joined = ", ".join(str(path) for path in candidates)
    raise RetrievalConfigurationError(f"No retrieval results found. Checked: {joined}")
