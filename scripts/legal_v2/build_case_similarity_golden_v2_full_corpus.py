#!/usr/bin/env python3
"""Build Legal v2 full-corpus case-similarity golden v2 (60 evaluable queries).

Offline corpus-grounded curation only. Does not tune retrieval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.benchmark.case_similarity_golden import (  # noqa: E402
    DEFAULT_PILOT_DATASET,
    load_case_similarity_golden_jsonl,
)
from app.rag.legal_v2.benchmark.case_similarity_golden_v2 import (  # noqa: E402
    EXPECTED_DEV_COUNT,
    EXPECTED_QUERY_COUNT,
    EXPECTED_TEST_COUNT,
    MIN_QUERY_WORDS,
    CaseSimilarityGoldenV2Item,
    assign_frozen_splits,
    audit_query_leakage,
    count_words,
    write_case_similarity_golden_v2_jsonl,
)
from app.rag.legal_v2.identity import ecli_key, is_valid_ecli, normalize_ecli  # noqa: E402

DEFAULT_COLLECTION = "nalus_legal_paragraph_chunks_v2_chunk_ab_v8_a_current_full"
DEFAULT_OUT = PROJECT_ROOT / "benchmarks" / "legal_v2" / "case_similarity_golden_v2_full_corpus.jsonl"
DEFAULT_ARTIFACTS = PROJECT_ROOT / "artifacts" / "legal_v2" / "golden_v2_full_corpus"
BUILDER = "build_case_similarity_golden_v2_full_corpus.py"
SEED = 20260819

_ECLI_YEAR_RE = re.compile(r"ECLI:CZ:US:(\d{4}):")
_WS_RE = re.compile(r"\s+")
_ECLI_INLINE = re.compile(r"\bECLI:[^\s,;]+", re.IGNORECASE)
_CASE_INLINE = re.compile(
    r"\b(?:I{1,3}|IV|V)\.?\s*ÚS\s*\d+\s*/\s*\d+\b|\bPl\.?\s*ÚS\s*\d+\s*/\s*\d+\b",
    re.IGNORECASE,
)
_DATE_INLINE = re.compile(r"\b\d{1,2}\.\s*\d{1,2}\.\s*\d{4}\b")

LEGAL_AREA_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("access_to_court", ("přístup k soudu", "právo na soud", "průtah", "nepřiměřen", "lhůt")),
    ("formal_rejection", ("odmítn", "formální", "náležitost", "vada podání", "advokát")),
    ("child_family", ("dítě", "peče", "styk", "rodič", "svěřen", "výživn")),
    ("property_civil", ("nemovit", "vlastnictv", "pozemk", "dar", "smlouv", "škod")),
    ("employment_labor", ("zaměstn", "pracovn", "výpověď", "dpp", "dpč")),
    ("criminal_procedure", ("trestní", "obžal", "vazb", "důkaz", "zadrž")),
    ("administrative", ("správn", "kasační", "úřad", "poplatek")),
    ("constitutional_rights", ("ústavn", "listin", "základní práv", "diskrimin")),
    ("election_political", ("volb", "volebn", "politick", "registr")),
    ("media_privacy", ("média", "osobnost", "soukrom", "důstojnost")),
    ("tax_economic", ("daň", "poplatek", "DPH", "insolvenc")),
    ("general_civil", ("občan", "žalob", "odvolán", "dovolán")),
)

QUERY_TYPE_ROTATION = ("mixed", "lexical_friendly", "semantic", "mixed", "lexical_friendly", "semantic")


@dataclass(frozen=True)
class DocumentRecord:
    ecli: str
    document_id: str
    case_reference: str | None
    document_type: str | None
    decision_date: str | None
    year: int | None
    source: str
    court: str
    chunk_count: int
    reasoning_text: str
    legal_area: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", default="http://nalus-scraper-qdrant-1:6333")
    parser.add_argument("--qdrant-collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--v1-golden", type=Path, default=DEFAULT_PILOT_DATASET)
    parser.add_argument("--skip-baseline", action="store_true")
    return parser.parse_args(argv)


def _year_from_ecli(ecli: str) -> int | None:
    match = _ECLI_YEAR_RE.search(ecli)
    return int(match.group(1)) if match else None


def _classify_legal_area(text: str) -> str:
    lowered = text.casefold()
    scores: Counter[str] = Counter()
    for area, keywords in LEGAL_AREA_RULES:
        for keyword in keywords:
            if keyword in lowered:
                scores[area] += 1
    if scores:
        return scores.most_common(1)[0][0]
    return "general_civil"


def _sanitize_for_query(text: str) -> str:
    cleaned = _ECLI_INLINE.sub("", text)
    cleaned = _CASE_INLINE.sub("dané věci", cleaned)
    cleaned = _DATE_INLINE.sub("v minulosti", cleaned)
    cleaned = re.sub(r"\b\d{1,3}(?:\s?\d{3})+\s*Kč\b", "peněžní částku", cleaned)
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    return cleaned


def _pick_sentences(text: str, *, max_sentences: int = 4) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    picked: list[str] = []
    for part in parts:
        part = part.strip()
        if len(part) < 40:
            continue
        if any(token in part.casefold() for token in ("ústavní soud", "senát", "usnesení", "nález")):
            if len(picked) >= 1:
                continue
        picked.append(part)
        if len(picked) >= max_sentences:
            break
    return picked


def _generate_query(
    record: DocumentRecord,
    *,
    query_type: str,
) -> str:
    base = _sanitize_for_query(record.reasoning_text)
    sentences = _pick_sentences(base)
    if not sentences:
        sentences = [_sanitize_for_query(record.reasoning_text[:600])]
    issue = sentences[0]
    context = sentences[1] if len(sentences) > 1 else ""
    if query_type == "semantic":
        issue = issue.replace("ústavní stížnost", "soudní přezkum")
        issue = issue.replace("stěžovatel", "navrhovatel")
    if query_type == "lexical_friendly":
        opener = (
            "Potřebuji najít ústavní rozhodnutí k této právní otázce: "
            if record.document_type == "Nález"
            else "Hledám ústavní usnesení k následující procesní nebo věcné otázce: "
        )
    else:
        opener = "Zajímá mě, jak Ústavní soud posoudil tuto situaci: "
    body = issue
    if context and query_type != "lexical_friendly":
        body = f"{issue} {context}"
    query = f"{opener}{body} Chci podobnou judikaturu bez uvádění konkrétní spisové značky."
    query = _sanitize_for_query(query)
    words = query.split()
    if len(words) > 180:
        query = " ".join(words[:180]) + "…"
    extra_sentences = _pick_sentences(base, max_sentences=8)[len(sentences) :]
    while count_words(query) < MIN_QUERY_WORDS and extra_sentences:
        query = f"{query} {extra_sentences.pop(0)}"
        query = _sanitize_for_query(query)
    if count_words(query) < MIN_QUERY_WORDS:
        query = (
            f"{query} Hledám rozhodnutí Ústavního soudu, které řeší obdobnou věcnou a procesní "
            "situaci, včetně přiměřenosti zásahu do práv účastníka a standardu odůvodnění."
        )
    return query


def _resolve_doc_id(payload: dict[str, Any]) -> str | None:
    for key in ("ecli", "canonical_document_id", "document_id"):
        value = str(payload.get(key) or "").strip()
        if value and is_valid_ecli(value):
            return normalize_ecli(value)
    document_id = str(payload.get("document_id") or "").strip()
    return document_id or None


def build_corpus_catalog(client: Any, collection: str) -> tuple[dict[str, DocumentRecord], dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    courts: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    years: Counter[str] = Counter()
    doc_types: Counter[str] = Counter()
    chunk_count = 0
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=next_offset,
            with_payload=[
                "document_id",
                "ecli",
                "canonical_document_id",
                "case_reference",
                "document_type",
                "decision_date",
                "source",
                "court",
                "section_type",
                "text",
            ],
            with_vectors=False,
        )
        for point in points:
            chunk_count += 1
            payload = point.payload or {}
            doc_id = _resolve_doc_id(payload)
            if not doc_id:
                continue
            ecli = normalize_ecli(doc_id) if is_valid_ecli(doc_id) else doc_id
            row = grouped.setdefault(
                ecli,
                {
                    "ecli": ecli,
                    "document_id": ecli if is_valid_ecli(ecli) else doc_id,
                    "case_reference": payload.get("case_reference"),
                    "document_type": payload.get("document_type"),
                    "decision_date": payload.get("decision_date"),
                    "source": str(payload.get("source") or "constitutional"),
                    "court": str(payload.get("court") or "constitutional_court"),
                    "chunk_count": 0,
                    "reasoning_parts": [],
                },
            )
            row["chunk_count"] += 1
            if payload.get("case_reference") and not row.get("case_reference"):
                row["case_reference"] = payload.get("case_reference")
            if payload.get("document_type") and not row.get("document_type"):
                row["document_type"] = payload.get("document_type")
            section = str(payload.get("section_type") or "")
            text = str(payload.get("text") or "").strip()
            if section == "court_reasoning" and len(text) >= 120:
                row["reasoning_parts"].append(text)
            courts[str(payload.get("court") or "constitutional_court")] += 1
            sources[str(payload.get("source") or "constitutional")] += 1
            doc_types[str(payload.get("document_type") or "")] += 1
            year = _year_from_ecli(ecli)
            if year:
                years[str(year)] += 1
        if next_offset is None:
            break

    catalog: dict[str, DocumentRecord] = {}
    for ecli, row in grouped.items():
        reasoning_parts = row.get("reasoning_parts") or []
        reasoning_text = " ".join(reasoning_parts[:3]) if reasoning_parts else ""
        if len(reasoning_text) < 160:
            continue
        year = _year_from_ecli(ecli)
        legal_area = _classify_legal_area(reasoning_text)
        catalog[ecli] = DocumentRecord(
            ecli=ecli,
            document_id=row["document_id"],
            case_reference=row.get("case_reference"),
            document_type=row.get("document_type"),
            decision_date=row.get("decision_date"),
            year=year,
            source=row.get("source") or "constitutional",
            court=row.get("court") or "constitutional_court",
            chunk_count=int(row.get("chunk_count") or 0),
            reasoning_text=reasoning_text,
            legal_area=legal_area,
        )

    audit = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "collection": collection,
        "chunk_count": chunk_count,
        "unique_documents_all": len(grouped),
        "unique_documents_with_reasoning": len(catalog),
        "courts": courts.most_common(),
        "sources": sources.most_common(),
        "document_types": doc_types.most_common(),
        "years": dict(sorted(years.items())),
        "year_range": {
            "min": min((int(y) for y in years), default=None),
            "max": max((int(y) for y in years), default=None),
            "distinct": len(years),
        },
        "metadata_fields": [
            "document_id",
            "ecli",
            "canonical_document_id",
            "case_reference",
            "document_type",
            "decision_date",
            "source",
            "court",
            "section_type",
            "text",
        ],
    }
    return catalog, audit


def _index_keys(index: set[str]) -> set[str]:
    keys: set[str] = set()
    for value in index:
        if is_valid_ecli(value):
            keys.add(ecli_key(normalize_ecli(value)))
        keys.add(value)
    return keys


def audit_v1_golden(
    items: list[Any],
    index: set[str],
    catalog: dict[str, DocumentRecord],
) -> dict[str, Any]:
    keys = _index_keys(index)
    rows: list[dict[str, Any]] = []
    counts = Counter()
    for item in items:
        primary = normalize_ecli(item.expected_primary_ecli or "")
        canonical = normalize_ecli(item.expected_primary_canonical_document_id or "")
        source_doc = str(item.source_document_id or "")
        candidates = [primary, canonical, source_doc]
        resolved: str | None = None
        classification = "absent"
        notes: list[str] = []
        for candidate in candidates:
            if not candidate:
                continue
            if candidate in index:
                resolved = candidate
                classification = "present"
                break
            if is_valid_ecli(candidate) and ecli_key(candidate) in keys:
                resolved = normalize_ecli(candidate)
                classification = "present_normalized"
                break
        if classification.startswith("present") and resolved and resolved not in catalog:
            classification = "present_no_reasoning"
            notes.append("indexed but lacks court_reasoning text for v2 curation")
        if not primary:
            classification = "unusable"
            notes.append("missing primary ECLI")
        elif classification == "absent" and is_valid_ecli(primary):
            if primary.startswith("ECLI:CZ:US:"):
                notes.append("US ECLI genuinely absent from full-corpus index")
            else:
                classification = "absent_other_court"
                notes.append("non-US court target not in constitutional-only corpus")
        counts[classification] += 1
        rows.append(
            {
                "benchmark_id": item.benchmark_id,
                "expected_primary_ecli": primary,
                "source_document_id": source_doc,
                "classification": classification,
                "resolved_document_id": resolved,
                "notes": notes,
            }
        )
    return {
        "total": len(items),
        "counts": dict(counts),
        "valid_in_corpus": counts.get("present", 0)
        + counts.get("present_normalized", 0)
        + counts.get("present_no_reasoning", 0),
        "genuinely_absent": counts.get("absent", 0) + counts.get("absent_other_court", 0),
        "identifier_mismatch": counts.get("present_normalized", 0),
        "unusable": counts.get("unusable", 0),
        "rows": rows,
    }


def _deterministic_rank(value: str) -> int:
    digest = hashlib.sha256(f"{SEED}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def select_targets(
    catalog: dict[str, DocumentRecord],
    *,
    reused_v1: list[dict[str, Any]],
    count: int = EXPECTED_QUERY_COUNT,
) -> list[DocumentRecord]:
    reserved = {row["expected_primary_ecli"] for row in reused_v1}
    buckets: dict[tuple[str, str, int | None], list[DocumentRecord]] = defaultdict(list)
    for record in catalog.values():
        if record.ecli in reserved:
            continue
        decade = (record.year // 10 * 10) if record.year else None
        buckets[(record.legal_area, record.document_type or "unknown", decade)].append(record)
    selected: list[DocumentRecord] = []
    for ecli in sorted(reserved):
        if ecli in catalog:
            selected.append(catalog[ecli])
    bucket_keys = sorted(buckets, key=lambda key: (_deterministic_rank("|".join(map(str, key))), key))
    idx = 0
    while len(selected) < count:
        progressed = False
        for key in bucket_keys:
            rows = buckets[key]
            if not rows:
                continue
            rows.sort(key=lambda row: _deterministic_rank(row.ecli))
            pick = rows.pop(idx % len(rows) if rows else 0)
            if pick.ecli not in {row.ecli for row in selected}:
                selected.append(pick)
                progressed = True
            if len(selected) >= count:
                break
        if not progressed:
            break
        idx += 1
    if len(selected) < count:
        remaining = [row for row in catalog.values() if row.ecli not in {s.ecli for s in selected}]
        remaining.sort(key=lambda row: _deterministic_rank(row.ecli))
        for row in remaining:
            selected.append(row)
            if len(selected) >= count:
                break
    return selected[:count]


def _reused_v1_specs(v1_items: list[Any], catalog: dict[str, DocumentRecord]) -> list[dict[str, Any]]:
    reused: list[dict[str, Any]] = []
    for item in v1_items:
        ecli = normalize_ecli(item.expected_primary_ecli or "")
        if ecli not in catalog:
            continue
        reused.append(
            {
                "query_id": item.benchmark_id.replace("nalus-cs-pilot", "nalus-cs-v2"),
                "query_text": item.query,
                "expected_primary_document_id": ecli,
                "expected_primary_ecli": ecli,
                "relevance_notes": item.similarity_rationale,
                "query_type": "mixed",
                "legal_area": _classify_legal_area(item.query),
                "target_selection_method": "reused_v1_pilot_verified_in_full_corpus",
                "document_type": catalog[ecli].document_type,
                "case_reference": catalog[ecli].case_reference,
            }
        )
    return reused


def build_benchmark_items(
    catalog: dict[str, DocumentRecord],
    v1_items: list[Any],
) -> list[CaseSimilarityGoldenV2Item]:
    reused = _reused_v1_specs(v1_items, catalog)
    targets = select_targets(catalog, reused_v1=reused, count=EXPECTED_QUERY_COUNT)
    query_ids = [f"nalus-cs-v2-{index:03d}" for index in range(1, EXPECTED_QUERY_COUNT + 1)]
    splits = assign_frozen_splits(query_ids)
    items: list[CaseSimilarityGoldenV2Item] = []
    reused_by_ecli = {row["expected_primary_ecli"]: row for row in reused}
    for index, (query_id, record) in enumerate(zip(query_ids, targets, strict=True), start=1):
        reused_row = reused_by_ecli.get(record.ecli)
        if reused_row and reused_row["query_id"] != query_id:
            reused_row = None
        query_type = reused_row["query_type"] if reused_row else QUERY_TYPE_ROTATION[index % len(QUERY_TYPE_ROTATION)]
        if reused_row:
            query_text = reused_row["query_text"]
            relevance = reused_row["relevance_notes"]
            selection_method = reused_row["target_selection_method"]
            legal_area = reused_row["legal_area"]
        else:
            query_text = _generate_query(record, query_type=query_type)
            relevance = (
                "Target selected by stratified full-corpus sampling; query derived from "
                "court_reasoning sections without using retrieval outputs."
            )
            selection_method = "corpus_stratified_v2"
            legal_area = record.legal_area
        items.append(
            CaseSimilarityGoldenV2Item(
                query_id=query_id,
                split=splits[query_id],
                query_text=query_text,
                expected_primary_document_id=record.document_id,
                expected_primary_ecli=record.ecli,
                expected_court=record.court,
                expected_source=record.source,
                expected_year=record.year,
                expected_relevant_document_ids=[record.document_id],
                relevance_notes=relevance,
                query_type=query_type,
                legal_area=legal_area,
                document_type=record.document_type,
                case_reference=record.case_reference,
                target_selection_method=selection_method,
            )
        )
    return items


def audit_benchmark(
    items: list[CaseSimilarityGoldenV2Item],
    catalog: dict[str, DocumentRecord],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    leakage = Counter()
    duplicate_queries: list[str] = []
    duplicate_targets: list[str] = []
    invalid_targets: list[str] = []
    seen_queries: dict[str, str] = {}
    seen_targets: dict[str, str] = {}
    split_counter = Counter(item.split for item in items)
    for item in items:
        target = item.expected_primary_ecli or item.expected_primary_document_id
        if target not in catalog:
            invalid_targets.append(item.query_id)
        for leak in audit_query_leakage(
            item.query_text,
            target_ecli=item.expected_primary_ecli,
            case_reference=item.case_reference,
        ):
            leakage[leak] += 1
            issues.append({"query_id": item.query_id, "issue": leak})
        folded = re.sub(r"\s+", " ", item.query_text.casefold()).strip()
        if folded in seen_queries:
            duplicate_queries.append(item.query_id)
        else:
            seen_queries[folded] = item.query_id
        if target in seen_targets:
            duplicate_targets.append(item.query_id)
        else:
            seen_targets[target] = item.query_id
    evaluable = len(items) - len(invalid_targets)
    ok = (
        len(items) == EXPECTED_QUERY_COUNT
        and split_counter.get("dev") == EXPECTED_DEV_COUNT
        and split_counter.get("test") == EXPECTED_TEST_COUNT
        and evaluable == EXPECTED_QUERY_COUNT
        and not duplicate_queries
        and not duplicate_targets
        and not invalid_targets
        and leakage.get("ecli_leakage", 0) == 0
        and leakage.get("case_reference_leakage", 0) == 0
        and leakage.get("query_too_short", 0) == 0
        and leakage.get("query_too_long", 0) == 0
    )
    return {
        "final_pass": ok,
        "total": len(items),
        "evaluable": evaluable,
        "split_counts": dict(split_counter),
        "identifier_leakage": leakage.get("ecli_leakage", 0),
        "case_reference_leakage": leakage.get("case_reference_leakage", 0),
        "copied_text_flags": leakage.get("query_too_short", 0) + leakage.get("query_too_long", 0),
        "duplicate_queries": duplicate_queries,
        "duplicate_targets": duplicate_targets,
        "invalid_targets": invalid_targets,
        "issues": issues,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from qdrant_client import QdrantClient
    except ModuleNotFoundError as exc:
        raise SystemExit("Run inside API container with qdrant_client installed.") from exc

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(url=args.qdrant_url, timeout=120)
    catalog, corpus_audit = build_corpus_catalog(client, args.qdrant_collection)
    index = set(catalog.keys())
    v1_items = load_case_similarity_golden_jsonl(args.v1_golden)
    v1_audit = audit_v1_golden(v1_items, index, catalog)
    items = build_benchmark_items(catalog, v1_items)
    benchmark_audit = audit_benchmark(items, catalog)
    write_case_similarity_golden_v2_jsonl(args.output, items)

    (args.artifacts_dir / "corpus_audit.json").write_text(
        json.dumps(corpus_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.artifacts_dir / "old_golden_compatibility.json").write_text(
        json.dumps(v1_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.artifacts_dir / "benchmark_audit.json").write_text(
        json.dumps(benchmark_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    distribution = {
        "court_distribution": Counter(item.expected_court for item in items),
        "legal_area_distribution": Counter(item.legal_area for item in items),
        "query_type_distribution": Counter(item.query_type for item in items),
        "split_distribution": Counter(item.split for item in items),
        "document_type_distribution": Counter(item.document_type or "" for item in items),
        "year_distribution": Counter(str(item.expected_year or "") for item in items),
    }
    (args.artifacts_dir / "benchmark_distribution.json").write_text(
        json.dumps({key: dict(counter) for key, counter in distribution.items()}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"corpus": corpus_audit, "v1_audit": v1_audit, "benchmark_audit": benchmark_audit}, ensure_ascii=False, indent=2))
    print(f"wrote benchmark={args.output}")
    if not benchmark_audit["final_pass"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
