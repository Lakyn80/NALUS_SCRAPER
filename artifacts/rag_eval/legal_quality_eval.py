"""Legal-quality evaluation for NALUS retrieval benchmark exports."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CLASSIFICATIONS = frozenset(
    {"exact_dataset_match", "alternate_relevant", "irrelevant", "uncertain"}
)
TOPIC_LEVELS = frozenset({"high", "medium", "low", "none"})
USEFULNESS_LEVELS = frozenset({"excellent", "good", "partial", "poor"})
ALIGNMENT_LEVELS = frozenset({"aligned", "partially_aligned", "misaligned"})
GENERIC_RISK_LEVELS = frozenset({"high", "medium", "low"})

_CZECH_STOPWORDS = frozenset(
    {
        "a",
        "i",
        "o",
        "u",
        "v",
        "z",
        "k",
        "s",
        "na",
        "do",
        "od",
        "po",
        "za",
        "podle",
        "nebo",
        "je",
        "jsou",
        "byl",
        "byla",
        "bylo",
        "jako",
        "pro",
        "při",
        "že",
        "se",
        "si",
    }
)

_SECTION_RE = re.compile(r"§\s*\d+[a-z]?(?:\s*odst\.\s*\d+)?(?:\s*písm\.\s*[a-z]\))?", re.I)
_HIGH_GENERIC_PATTERNS = (
    re.compile(r"dovolací\s+důvod\s+podle\s+§", re.I),
    re.compile(r"náhrad[ěe]\s+nákladů\s+dovolacího\s+řízení", re.I),
)


def marker_hit(text: str, marker: str, aliases: list[str] | None = None) -> bool:
    haystack = text.casefold()
    needles = [marker, *(aliases or [])]
    return any(needle and needle.casefold() in haystack for needle in needles)


def _normalize_token(token: str) -> str:
    return token.strip(".,;:!?()\"'").casefold()


def extract_topic_signals(question: str, markers: list[dict[str, Any]]) -> set[str]:
    signals: set[str] = set()
    for raw in question.split():
        token = _normalize_token(raw)
        if len(token) >= 4 and token not in _CZECH_STOPWORDS:
            signals.add(token)
    for marker_info in markers:
        marker = (marker_info.get("marker") or "").strip()
        if not marker:
            continue
        for token in marker.split():
            token = _normalize_token(token)
            if len(token) >= 4 and token not in _CZECH_STOPWORDS:
                signals.add(token)
        signals.add(marker.casefold())
    for section in _SECTION_RE.findall(question):
        signals.add(re.sub(r"\s+", " ", section).casefold())
    return signals


def assess_generic_question_risk(question: str) -> str:
    if any(pattern.search(question) for pattern in _HIGH_GENERIC_PATTERNS):
        return "high"
    if _SECTION_RE.search(question):
        return "medium"
    if len(question.split()) <= 5:
        return "medium"
    return "low"


def assess_legal_topic_match(
    text: str,
    *,
    question: str,
    markers: list[dict[str, Any]],
    signals: set[str] | None = None,
) -> str:
    text_cf = text.casefold()
    topic_signals = signals if signals is not None else extract_topic_signals(question, markers)

    if any(marker_hit(text, m.get("marker", ""), m.get("aliases")) for m in markers):
        return "high"

    section_hits = sum(1 for section in _SECTION_RE.findall(question) if section.casefold() in text_cf)
    token_hits = sum(1 for signal in topic_signals if len(signal) >= 4 and signal in text_cf)

    if section_hits and token_hits >= 2:
        return "high"
    if token_hits >= max(2, len(topic_signals) // 2):
        return "high"
    if token_hits >= 2 or section_hits:
        return "medium"
    if token_hits == 1:
        return "low"
    return "none"


def classify_hit(
    *,
    document_id: str | None,
    text: str,
    expected_ecli: list[str],
    question: str,
    markers: list[dict[str, Any]],
    signals: set[str] | None = None,
) -> tuple[str, str, bool, str]:
    """Return classification, legal_topic_match, marker_found, reason_cs."""
    doc_id = document_id or ""
    in_dataset = doc_id in expected_ecli
    marker_found = any(marker_hit(text, m.get("marker", ""), m.get("aliases")) for m in markers)
    topic = assess_legal_topic_match(
        text, question=question, markers=markers, signals=signals
    )

    if in_dataset and topic in {"high", "medium"}:
        reason = (
            "Chunk pochází z očekávaného dokumentu datasetu a řeší stejné právní téma."
            if marker_found
            else "Chunk pochází z očekávaného ECLI a tematicky odpovídá otázce."
        )
        return "exact_dataset_match", topic, marker_found, reason

    if not in_dataset and (marker_found or topic == "high"):
        reason = (
            "Jiné ECLI, ale chunk obsahuje očekávanou evidenci a řeší stejné právní téma."
            if marker_found
            else "Jiné ECLI, ale chunk řeší stejný právní institut jako otázka."
        )
        return "alternate_relevant", topic, marker_found, reason

    if not in_dataset and topic == "medium":
        return (
            "alternate_relevant",
            topic,
            marker_found,
            "Jiné ECLI, ale chunk je tematicky blízký právnímu problému z otázky.",
        )

    if in_dataset and topic == "low":
        return (
            "uncertain",
            topic,
            marker_found,
            "Očekávané ECLI, ale tematická shoda s otázkou je slabá.",
        )

    if topic == "low" and marker_found:
        return (
            "alternate_relevant",
            topic,
            marker_found,
            "Marker nalezen, ale právní kontext může být jen částečně relevantní.",
        )

    if topic == "none":
        return (
            "irrelevant",
            topic,
            marker_found,
            "Chunk se k právnímu problému z otázky zřejmě netýká.",
        )

    return (
        "uncertain",
        topic,
        marker_found,
        "Relevance chunku k otázce není jednoznačná.",
    )


def _is_relevant_classification(classification: str) -> bool:
    return classification in {"exact_dataset_match", "alternate_relevant"}


def assess_production_usefulness(
    hits: list[dict[str, Any]],
) -> str:
    relevant = [h for h in hits if _is_relevant_classification(h["classification"])]
    if not relevant:
        return "poor"
    best_rank = min(h["rank"] for h in relevant)
    best = next(h for h in relevant if h["rank"] == best_rank)
    if best_rank == 1 and best["legal_topic_match"] == "high":
        return "excellent"
    if best_rank <= 2 and best["legal_topic_match"] in {"high", "medium"}:
        return "good"
    if best_rank <= 5:
        return "partial"
    return "poor"


def assess_benchmark_alignment(
    *,
    expected_ecli: list[str],
    hits: list[dict[str, Any]],
) -> str:
    if hits and hits[0]["document_id"] in expected_ecli:
        return "aligned"
    if any(h["document_id"] in expected_ecli for h in hits):
        return "partially_aligned"
    if any(
        h["document_id"] in expected_ecli and h.get("marker_found")
        for h in hits
    ):
        return "partially_aligned"
    relevant = [h for h in hits if _is_relevant_classification(h["classification"])]
    if relevant:
        return "partially_aligned"
    return "misaligned"


def evaluate_case(
    *,
    case_id: str,
    question: str,
    expected_ecli: list[str],
    expected_markers: list[str],
    marker_defs: list[dict[str, Any]],
    retrieval_hits: list[dict[str, Any]],
) -> dict[str, Any]:
    signals = extract_topic_signals(question, marker_defs)
    generic_risk = assess_generic_question_risk(question)

    hits_out: list[dict[str, Any]] = []
    for hit in retrieval_hits:
        classification, topic, marker_found, reason = classify_hit(
            document_id=hit.get("document_id"),
            text=hit.get("text") or "",
            expected_ecli=expected_ecli,
            question=question,
            markers=marker_defs,
            signals=signals,
        )
        hits_out.append(
            {
                "rank": hit["rank"],
                "document_id": hit.get("document_id"),
                "classification": classification,
                "legal_topic_match": topic,
                "marker_found": marker_found,
                "reason_cs": reason,
            }
        )

    relevant = [h for h in hits_out if _is_relevant_classification(h["classification"])]
    best = min(relevant, key=lambda h: h["rank"]) if relevant else (hits_out[0] if hits_out else None)

    top1 = hits_out[0] if hits_out else None
    return {
        "case_id": case_id,
        "question": question,
        "expected_ecli": expected_ecli,
        "expected_markers": expected_markers,
        "generic_question_risk": generic_risk,
        "top1_classification": top1["classification"] if top1 else "uncertain",
        "best_relevant_rank": best["rank"] if best else None,
        "best_relevant_document_id": best.get("document_id") if best else None,
        "production_usefulness": assess_production_usefulness(hits_out),
        "benchmark_alignment": assess_benchmark_alignment(expected_ecli=expected_ecli, hits=hits_out),
        "summary_cs": _case_summary(
            question=question,
            best=best,
            top1=top1,
            generic_risk=generic_risk,
        ),
        "hits": hits_out,
        "notes": _case_notes(
            expected_ecli=expected_ecli,
            top1=top1,
            best=best,
            generic_risk=generic_risk,
        ),
    }


def _case_summary(
    *,
    question: str,
    best: dict[str, Any] | None,
    top1: dict[str, Any] | None,
    generic_risk: str,
) -> str:
    if not best:
        return f"K otázce „{question}“ retrieval nevrátil použitelný právní kontext."
    if best["classification"] == "exact_dataset_match" and best["rank"] == 1:
        return (
            f"Top výsledek pochází z očekávaného dokumentu datasetu a řeší právní problém z otázky."
        )
    if best["classification"] == "alternate_relevant":
        note = (
            "To je pro reálné vyhledávání obvykle přijatelné."
            if generic_risk in {"high", "medium"}
            else "Pro reálné vyhledávání jde o relevantní podobnou judikaturu."
        )
        if top1 and top1["rank"] != best["rank"]:
            return (
                f"Top-1 je jiné ECLI, ale na rank {best['rank']} je právně relevantní podobný případ. {note}"
            )
        return f"Retrieval našel jiné, ale právně relevantní rozhodnutí k tématu otázky. {note}"
    return f"Nejlepší hit má nejasnou relevanci; výsledek je pro praktické vyhledávání slabší."


def _case_notes(
    *,
    expected_ecli: list[str],
    top1: dict[str, Any] | None,
    best: dict[str, Any] | None,
    generic_risk: str,
) -> list[str]:
    notes: list[str] = []
    if generic_risk == "high":
        notes.append("Otázka je obecná; více ECLI může být právně relevantních.")
    if top1 and best and top1["document_id"] not in expected_ecli and best["classification"] == "alternate_relevant":
        notes.append("Top-1 není v dataset scope, ale nalezená judikatura je tematicky blízká.")
    if top1 and top1["classification"] == "exact_dataset_match" and top1["rank"] == 1:
        notes.append("Shoda s benchmark datasetem i produkční relevancí.")
    return notes


def load_dataset_cases(dataset_path: Path) -> dict[str, dict[str, Any]]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    return {case["id"]: case for case in dataset["cases"]}


def evaluate_winner_export(
    *,
    winner_qa_path: Path,
    dataset_path: Path,
) -> dict[str, Any]:
    export = json.loads(winner_qa_path.read_text(encoding="utf-8"))
    dataset_cases = load_dataset_cases(dataset_path)

    case_evals: list[dict[str, Any]] = []
    for case in export["cases"]:
        meta = dataset_cases[case["case_id"]]
        scope = meta.get("source_scope") or {}
        expected_ecli = list(scope.get("document_ids") or [])
        marker_defs = meta.get("required_evidence") or []
        case_evals.append(
            evaluate_case(
                case_id=case["case_id"],
                question=case["question"],
                expected_ecli=expected_ecli,
                expected_markers=[m.get("marker", "") for m in marker_defs],
                marker_defs=marker_defs,
                retrieval_hits=case.get("retrieval_hits") or [],
            )
        )

    classification_counts = Counter(
        hit["classification"] for case in case_evals for hit in case["hits"]
    )

    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "source_export": str(winner_qa_path),
        "source_dataset": str(dataset_path),
        "winner_config_id": export.get("winner_config_id"),
        "model_code": export.get("model_code"),
        "retrieval_mode": export.get("retrieval_mode"),
        "benchmark_metrics": export.get("metrics"),
        "case_count": len(case_evals),
        "classification_counts": dict(classification_counts),
        "cases": case_evals,
    }


def render_legal_quality_report(payload: dict[str, Any]) -> str:
    counts = payload["classification_counts"]
    cases = payload["cases"]

    usefulness_counts = Counter(c["production_usefulness"] for c in cases)
    alignment_counts = Counter(c["benchmark_alignment"] for c in cases)
    generic_high = [c for c in cases if c["generic_question_risk"] == "high"]

    excellent_or_good = sum(
        1 for c in cases if c["production_usefulness"] in {"excellent", "good"}
    )
    pilot_ready = excellent_or_good == len(cases) and not any(
        c["production_usefulness"] in {"poor", "partial"} for c in cases
    )

    lines = [
        "# NALUS RAG Eval — Legal Quality Report",
        "",
        "## Executive summary",
        "",
        (
            f"Vítězný retrieval (`{payload.get('winner_config_id')}`, `{payload.get('retrieval_mode')}`) "
            f"u {payload['case_count']} pilotních otázek vrací převážně právně užitečné podobné judikaturu. "
            f"Benchmark metriky (hit_rate={payload.get('benchmark_metrics', {}).get('hit_rate')}, "
            f"mrr={payload.get('benchmark_metrics', {}).get('mrr')}) měří technickou shodu s datasetem; "
            f"tento report hodnotí produkční právní užitečnost zvlášť."
        ),
        "",
        "## Benchmark vs. produkční užitečnost",
        "",
        "| Pojem | Co měří |",
        "| --- | --- |",
        "| `benchmark_alignment` | Shoda s ručně zvolenými ECLI v eval datasetu |",
        "| `production_usefulness` | Užitečnost pro reálné hledání podobné judikatury |",
        "| `classification` | Právní typ relevance každého hitu (exact / alternate / irrelevant) |",
        "",
        "Jiné ECLI než v datasetu **není automaticky chyba**, pokud jde o `alternate_relevant`.",
        "",
        "## Per-case přehled",
        "",
        "| case_id | production_usefulness | benchmark_alignment | top1_classification | best_relevant_rank | note |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for case in cases:
        note = case["notes"][0] if case["notes"] else case["summary_cs"]
        note = note.replace("|", "/")
        lines.append(
            f"| {case['case_id']} | {case['production_usefulness']} | "
            f"{case['benchmark_alignment']} | {case['top1_classification']} | "
            f"{case['best_relevant_rank']} | {note} |"
        )

    lines.extend(
        [
            "",
            "## Klasifikace hitů (součty přes všechny top-k)",
            "",
            f"- `exact_dataset_match`: {counts.get('exact_dataset_match', 0)}",
            f"- `alternate_relevant`: {counts.get('alternate_relevant', 0)}",
            f"- `irrelevant`: {counts.get('irrelevant', 0)}",
            f"- `uncertain`: {counts.get('uncertain', 0)}",
            "",
            "## Produční užitečnost (per case)",
            "",
            f"- `excellent`: {usefulness_counts.get('excellent', 0)}",
            f"- `good`: {usefulness_counts.get('good', 0)}",
            f"- `partial`: {usefulness_counts.get('partial', 0)}",
            f"- `poor`: {usefulness_counts.get('poor', 0)}",
            "",
            "## Benchmark alignment (per case)",
            "",
            f"- `aligned`: {alignment_counts.get('aligned', 0)}",
            f"- `partially_aligned`: {alignment_counts.get('partially_aligned', 0)}",
            f"- `misaligned`: {alignment_counts.get('misaligned', 0)}",
            "",
            "## Riziko obecných otázek",
            "",
        ]
    )

    if generic_high:
        for case in generic_high:
            lines.append(f"- `{case['case_id']}` ({case['generic_question_risk']}): {case['question']}")
    else:
        lines.append("- Žádná otázka není označena jako high risk.")

    lines.extend(
        [
            "",
            "## Finální verdikt",
            "",
        ]
    )

    if pilot_ready:
        lines.append(
            "**Ano** — BGE-M3 + BM25 hybrid je pro pilotní produkční retrieval připravený, "
            "za těchto podmínek:"
        )
        lines.append("- uživatel hledá podobnou judikaturu, ne jeden konkrétní ECLI z testu;")
        lines.append("- u obecných otázek (§ dovolací důvody, náklady řízení) očekávejte více validních ECLI;")
        lines.append("- doporučujeme doplnit LLM rerank / právní sumarizaci nad top-k chunky.")
    else:
        lines.append(
            "**Ne / podmíněně** — retrieval potřebuje před produkcí doladit slabší case "
            "nebo rozšířit eval sadu."
        )

    lines.append("")
    return "\n".join(lines)
