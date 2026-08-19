#!/usr/bin/env python3
"""Prepare human review artifacts for Legal v2 full-corpus golden v2."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.benchmark.case_similarity_golden_v2 import (  # noqa: E402
    DEFAULT_V2_DATASET,
    MAX_QUERY_WORDS,
    MIN_QUERY_WORDS,
    audit_query_leakage,
    count_words,
    load_case_similarity_golden_v2_jsonl,
)
from app.rag.legal_v2.identity import is_valid_ecli, normalize_ecli  # noqa: E402

DEFAULT_COLLECTION = "nalus_legal_paragraph_chunks_v2_chunk_ab_v8_a_current_full"
DEFAULT_ARTIFACTS = PROJECT_ROOT / "artifacts" / "legal_v2" / "golden_v2_full_corpus"
BENCHMARK_SCOPE = "current_full_A_constitutional_court_only"

_TOKEN_RE = re.compile(r"[0-9A-Za-zÁ-Žá-ž§]+", re.UNICODE)
_ECLI_RE = re.compile(r"\bECLI:[A-Z]{2}:[A-Z]{2}:[0-9]{4}:[^\s,;]+", re.IGNORECASE)
_CASE_REF_RE = re.compile(
    r"(?:"
    r"\b(?:I{1,3}|IV|V{0,3}|VI{0,3}|IX|X{0,3})\.?\s*ÚS\s+\d+/\d+\b"
    r"|\bPl\.?\s*ÚS\s+\d+/\d+\b"
    r"|\bsp\.\s*zn\.\s*[^\s,]{3,}"
    r")",
    re.IGNORECASE,
)
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")
_RARE_NAME_RE = re.compile(r"\b[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ][a-záčďéěíňóřšťúůýž]{2,}\b")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", default="http://nalus-scraper-qdrant-1:6333")
    parser.add_argument("--qdrant-collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_V2_DATASET)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument(
        "--dense-per-query",
        type=Path,
        default=DEFAULT_ARTIFACTS / "per_query_dense.jsonl",
    )
    return parser.parse_args(argv)


def _tokenize(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text)]


def _ngram_overlap_ratio(query: str, target_text: str, n: int = 8) -> float:
    q_tokens = _tokenize(query)
    t_tokens = _tokenize(target_text)
    if len(q_tokens) < n or len(t_tokens) < n:
        return 0.0
    q_ngrams = {" ".join(q_tokens[i : i + n]) for i in range(len(q_tokens) - n + 1)}
    t_ngrams = {" ".join(t_tokens[i : i + n]) for i in range(len(t_tokens) - n + 1)}
    if not q_ngrams:
        return 0.0
    return len(q_ngrams & t_ngrams) / len(q_ngrams)


def _sentence_overlap(query: str, target_text: str, *, min_tokens: int = 10) -> list[str]:
    overlaps: list[str] = []
    query_sents = [part.strip() for part in _SENTENCE_RE.findall(query) if part.strip()]
    target_sents = [part.strip() for part in _SENTENCE_RE.findall(target_text) if part.strip()]
    for q_sent in query_sents:
        q_tokens = _tokenize(q_sent)
        if len(q_tokens) < min_tokens:
            continue
        for t_sent in target_sents:
            ratio = SequenceMatcher(None, " ".join(q_tokens), " ".join(_tokenize(t_sent))).ratio()
            if ratio >= 0.72:
                overlaps.append(q_sent[:180])
                break
    return overlaps


def _load_dense_ranks(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        rows[str(payload["query_id"])] = payload
    return rows


def _fetch_target_context(client: Any, collection: str, ecli: str) -> dict[str, Any]:
    from qdrant_client.http import models as qmodels

    filt = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="ecli",
                match=qmodels.MatchValue(value=ecli),
            )
        ]
    )
    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=filt,
        limit=64,
        with_payload=["text", "section_type", "case_reference", "document_type", "decision_date"],
        with_vectors=False,
    )
    reasoning: list[str] = []
    header_bits: list[str] = []
    for point in points:
        payload = point.payload or {}
        text = str(payload.get("text") or "").strip()
        section = str(payload.get("section_type") or "")
        if not text:
            continue
        if section == "court_reasoning":
            reasoning.append(text)
        elif section in {"header", "operative_part", "procedural_history"}:
            header_bits.append(text)
    reasoning_text = " ".join(reasoning[:3])
    header_text = " ".join(header_bits[:2])
    summary_source = reasoning_text or header_text
    summary = _sanitize_summary(summary_source[:700])
    issue = _extract_central_issue(reasoning_text or summary_source)
    excerpt = _sanitize_summary((reasoning[0] if reasoning else summary_source)[:420])
    return {
        "target_decision_summary": summary,
        "central_legal_issue": issue,
        "target_reasoning_excerpt": excerpt,
        "reasoning_chunk_count": len(reasoning),
    }


def _sanitize_summary(text: str) -> str:
    cleaned = _ECLI_RE.sub("", text)
    cleaned = _CASE_REF_RE.sub("dané věci", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_central_issue(text: str) -> str:
    if not text.strip():
        return "Právní posouzení věci Ústavním soudem (dostupný kontext chybí)."
    sentences = [part.strip() for part in _SENTENCE_RE.findall(text) if len(part.strip()) >= 40]
    if not sentences:
        return _sanitize_summary(text[:220])
    return _sanitize_summary(sentences[0][:260])


def _verify_corpus_scope(client: Any, collection: str) -> dict[str, Any]:
    info = client.get_collection(collection)
    courts: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    sample_docs: set[str] = set()
    next_offset = None
    scanned = 0
    while scanned < 5000:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=next_offset,
            with_payload=["document_id", "ecli", "court", "source"],
            with_vectors=False,
        )
        for point in points:
            scanned += 1
            payload = point.payload or {}
            courts[str(payload.get("court") or "constitutional_court")] += 1
            sources[str(payload.get("source") or "constitutional")] += 1
            doc = payload.get("ecli") or payload.get("document_id")
            if doc:
                sample_docs.add(str(doc))
        if next_offset is None:
            break
    return {
        "collection": collection,
        "points": info.points_count,
        "sample_chunks_scanned": scanned,
        "sample_unique_documents": len(sample_docs),
        "courts_sample": courts.most_common(),
        "sources_sample": sources.most_common(),
        "benchmark_scope": BENCHMARK_SCOPE,
        "scope_note": (
            "Current full FAST A Qdrant collection is constitutional-court (ÚS) only. "
            "This benchmark does not cover NS/NSS until those courts are indexed."
        ),
    }


def _generation_kind(item: Any) -> str:
    method = str(getattr(item, "target_selection_method", "") or "")
    if "reused_v1" in method:
        return "reused"
    return "auto"


def _automated_flags(
    item: Any,
    *,
    target_context: dict[str, Any],
    all_items: list[Any],
) -> list[str]:
    flags: list[str] = []
    query = item.query_text
    reasoning = target_context.get("target_reasoning_excerpt") or ""
    flags.extend(audit_query_leakage(query, target_ecli=item.expected_primary_ecli, case_reference=item.case_reference))
    if _ngram_overlap_ratio(query, reasoning, n=8) >= 0.15:
        flags.append("high_8gram_overlap_with_target_reasoning")
    if _ngram_overlap_ratio(query, reasoning, n=12) >= 0.08:
        flags.append("high_12gram_overlap_with_target_reasoning")
    if _sentence_overlap(query, reasoning):
        flags.append("near_verbatim_sentence_overlap")
    if count_words(query) < MIN_QUERY_WORDS:
        flags.append("query_too_short")
    if count_words(query) > MAX_QUERY_WORDS:
        flags.append("query_too_long")
    if re.search(r"[A-Za-z]{4,}", query) and not re.search(r"[áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ]", query):
        flags.append("missing_czech_diacritics")
    rare_names = _RARE_NAME_RE.findall(reasoning)
    for name in rare_names[:12]:
        if len(name) >= 5 and name in query:
            flags.append("target_specific_name_in_query")
            break
    if query.lower().startswith("odůvodnění:"):
        flags.append("query_starts_with_reasoning_header")
    if "Chci podobnou judikaturu bez uvádění konkrétní spisové značky." in query and _generation_kind(item) == "auto":
        flags.append("boilerplate_closing_phrase")
    if len(query.split()) < 55 and _generation_kind(item) == "auto":
        flags.append("possibly_over_close_auto_paraphrase")
    if not reasoning:
        flags.append("missing_target_reasoning_context")
    if "positive_not_retrieved" == "":
        pass
    return sorted(set(flags))


def _near_duplicate_pairs(items: list[Any]) -> list[tuple[str, str, float]]:
    pairs: list[tuple[str, str, float]] = []
    token_sets = {item.query_id: set(_tokenize(item.query_text)) for item in items}
    ids = [item.query_id for item in items]
    for i, left_id in enumerate(ids):
        for right_id in ids[i + 1 :]:
            left = token_sets[left_id]
            right = token_sets[right_id]
            if not left or not right:
                continue
            jaccard = len(left & right) / len(left | right)
            if jaccard >= 0.72:
                pairs.append((left_id, right_id, round(jaccard, 4)))
    return pairs


def _write_human_review_md(path: Path, queue_rows: list[dict[str, Any]]) -> None:
    chunks: list[str] = [
        "# Legal v2 Golden v2 — Human Review",
        "",
        f"**Benchmark scope:** `{BENCHMARK_SCOPE}` (Ústavní soud only; not NS/NSS).",
        "",
        "Review each query: realistic Czech legal search, no target leakage, relevant target.",
        "Dense baseline rank is informational only — do not approve/reject based on retrieval.",
        "",
    ]
    for row in queue_rows:
        split = row["split"].upper()
        chunks.extend(
            [
                f"## {row['query_id']} — {split}",
                "",
                "**Query:**",
                row["query_text"],
                "",
                "**Target:**",
                f"{row['expected_court']} / {row.get('expected_year') or '?'} / "
                f"{row.get('expected_primary_ecli') or row['expected_primary_document_id']}",
                "",
                "**Central legal issue:**",
                row["central_legal_issue"],
                "",
                "**Target reasoning context:**",
                row["target_reasoning_excerpt"],
                "",
                f"**Generated:** {row['generation_kind']}",
                "",
                "**Automated flags:** "
                + (", ".join(row["automated_flags"]) if row["automated_flags"] else "none"),
                "",
                f"**Dense baseline rank (info only):** {row.get('dense_primary_rank')}",
                "",
                "**Review:**",
                "- [ ] approved",
                "- [ ] needs edit",
                "- [ ] rejected",
                "",
                "**Notes:**",
                "",
                "---",
                "",
            ]
        )
    path.write_text("\n".join(chunks), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from qdrant_client import QdrantClient
    except ModuleNotFoundError as exc:
        raise SystemExit("Run inside API container with qdrant_client installed.") from exc

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    items = load_case_similarity_golden_v2_jsonl(args.benchmark)
    dense_by_id = _load_dense_ranks(args.dense_per_query)
    client = QdrantClient(url=args.qdrant_url, timeout=120)
    corpus_scope = _verify_corpus_scope(client, args.qdrant_collection)

    context_cache: dict[str, dict[str, Any]] = {}
    queue_rows: list[dict[str, Any]] = []
    flag_counter: Counter[str] = Counter()
    for item in items:
        ecli = normalize_ecli(item.expected_primary_ecli or item.expected_primary_document_id)
        if ecli not in context_cache:
            context_cache[ecli] = _fetch_target_context(client, args.qdrant_collection, ecli)
        target_context = context_cache[ecli]
        automated_flags = _automated_flags(item, target_context=target_context, all_items=items)
        for flag in automated_flags:
            flag_counter[flag] += 1
        dense_row = dense_by_id.get(item.query_id) or {}
        queue_rows.append(
            {
                "query_id": item.query_id,
                "split": item.split,
                "query_text": item.query_text,
                "query_type": item.query_type,
                "legal_area": item.legal_area,
                "expected_primary_document_id": item.expected_primary_document_id,
                "expected_primary_ecli": item.expected_primary_ecli,
                "expected_court": item.expected_court,
                "expected_year": item.expected_year,
                "target_decision_summary": target_context["target_decision_summary"],
                "central_legal_issue": target_context["central_legal_issue"],
                "target_reasoning_excerpt": target_context["target_reasoning_excerpt"],
                "generation_kind": _generation_kind(item),
                "target_selection_method": item.target_selection_method,
                "automated_leakage_checks": audit_query_leakage(
                    item.query_text,
                    target_ecli=item.expected_primary_ecli,
                    case_reference=item.case_reference,
                ),
                "automated_flags": automated_flags,
                "dense_primary_rank": dense_row.get("primary_rank"),
                "dense_hit_at_10": dense_row.get("hit_at_10"),
                "dense_failure_type": dense_row.get("failure_type"),
                "review_status": "pending",
                "reviewer_notes": "",
                "revised_query_text": "",
            }
        )

    near_duplicates = _near_duplicate_pairs(items)
    duplicate_targets = [
        key for key, count in Counter(item.expected_primary_ecli for item in items).items() if count > 1
    ]

    queue_path = args.artifacts_dir / "human_review_queue.jsonl"
    queue_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in queue_rows) + "\n",
        encoding="utf-8",
    )
    md_path = args.artifacts_dir / "HUMAN_REVIEW.md"
    _write_human_review_md(md_path, queue_rows)

    auto_report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_scope": BENCHMARK_SCOPE,
        "corpus_scope": corpus_scope,
        "total_queries": len(items),
        "auto_generated": sum(1 for row in queue_rows if row["generation_kind"] == "auto"),
        "reused": sum(1 for row in queue_rows if row["generation_kind"] == "reused"),
        "dev_count": sum(1 for row in queue_rows if row["split"] == "dev"),
        "test_count": sum(1 for row in queue_rows if row["split"] == "test"),
        "flag_counts": dict(flag_counter),
        "ecli_leakage": flag_counter.get("ecli_leakage", 0),
        "case_number_leakage": flag_counter.get("case_reference_leakage", 0)
        + flag_counter.get("case_reference_folded_leakage", 0),
        "high_text_overlap": sum(
            flag_counter.get(key, 0)
            for key in (
                "high_8gram_overlap_with_target_reasoning",
                "high_12gram_overlap_with_target_reasoning",
                "near_verbatim_sentence_overlap",
            )
        ),
        "near_duplicate_query_pairs": near_duplicates,
        "duplicate_primary_targets": duplicate_targets,
        "queries_with_any_flag": sum(1 for row in queue_rows if row["automated_flags"]),
        "status": "READY_FOR_HUMAN_REVIEW",
    }
    report_path = args.artifacts_dir / "automated_review_report.json"
    report_path.write_text(json.dumps(auto_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    meta_path = args.benchmark.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "benchmark_scope": BENCHMARK_SCOPE,
                "corpus_collection": args.qdrant_collection,
                "corpus_scope_verified_at": auto_report["timestamp_utc"],
                "does_not_cover": ["nsoud", "nss", "supreme_administrative", "high_courts"],
                "covers": ["constitutional_court"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(auto_report, ensure_ascii=False, indent=2))
    print(f"wrote {queue_path}")
    print(f"wrote {md_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
