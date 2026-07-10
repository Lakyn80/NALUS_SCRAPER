"""Read-only NSoud provenance check for pending legal QA gold annotations."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.eval.legal_answer_eval import is_boilerplate_snippet, load_retrieval_results
from app.rag.eval.legal_qa_benchmark import LegalQaItem, load_dataset, normalize_for_match


@dataclass(frozen=True)
class EnrichedHit:
    rank: int
    chunk_id: str
    document_id: str | None
    source_document_id: str | None
    ecli: str | None
    case_reference: str | None
    spisova_znacka: str | None
    decision_date: str | None
    source: str | None
    text_snippet: str
    metadata_keys_present: list[str]
    provenance_sufficient_for_gold: bool
    section_type: str | None
    legal_area: str | None
    keyword_hit_count: int
    keyword_hit_ratio: float
    anchor_hit_count: int
    support_level: str
    baseline_provenance_present: bool


@dataclass(frozen=True)
class CandidateRecord:
    question_id: str
    question: str
    candidate_rank: int | None
    candidate_chunk_id: str | None
    candidate_source_document_id: str | None
    candidate_ecli: str | None
    candidate_case_reference: str | None
    candidate_decision_date: str | None
    support_level: str
    classification: str
    reason: str
    recommended_action: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only NSoud gold provenance checker.")
    parser.add_argument("--retrieval-results", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--collection-name", required=True)
    parser.add_argument("--qdrant-url", default="http://qdrant:6333")
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--output-jsonl", required=True)
    return parser.parse_args()


def _qdrant_client(url: str) -> Any:
    from qdrant_client import QdrantClient

    return QdrantClient(url=url, timeout=30, check_compatibility=False)


def _parse_chunk_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("chunk_metadata")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _normalized_text(value: str | None) -> str:
    return normalize_for_match(value or "")


def _keyword_stems(item: LegalQaItem) -> list[str]:
    manual_map = {
        "odmítnutí": "odmit",
        "dovolání": "dovol",
        "přípustnost": "pripust",
        "dovolací": "dovolac",
        "důvod": "duvod",
        "občanské": "obcansk",
        "odůvodnění": "oduvodn",
        "usnesení": "usnesen",
        "trestní": "trest",
        "265b": "265b",
        "extrémní": "extrem",
        "nesoulad": "nesoulad",
        "procesní": "proces",
        "pochybení": "pochyben",
    }
    stems: list[str] = []
    for keyword in item.expected_keywords:
        normalized = _normalized_text(keyword)
        stem = manual_map.get(keyword, normalized[: max(5, min(len(normalized), 8))])
        if stem and stem not in stems:
            stems.append(stem)
    return stems


def _question_anchors(item: LegalQaItem) -> list[str]:
    normalized = _normalized_text(item.question)
    anchors: list[str] = []
    if "241a odst 1" in normalized:
        anchors.append("241a odst 1")
    if "265b" in normalized:
        anchors.append("265b")
    if "o s r" in normalized:
        anchors.append("o s r")
    if "tr r" in normalized:
        anchors.append("tr r")
    return anchors


def classify_support_level(*, item: LegalQaItem, text: str, legal_area: str | None) -> tuple[str, int, float, int]:
    if is_boilerplate_snippet(text):
        return "boilerplate_noise", 0, 0.0, 0

    normalized = _normalized_text(text)
    stems = _keyword_stems(item)
    anchors = _question_anchors(item)
    keyword_hits = sum(1 for stem in stems if stem and stem in normalized)
    anchor_hits = sum(1 for anchor in anchors if anchor and anchor in normalized)
    if legal_area == "civil" and "obcansk" in stems:
        keyword_hits = max(keyword_hits, 1)
    if legal_area == "criminal" and "trest" in stems:
        keyword_hits = max(keyword_hits, 1)
    ratio = keyword_hits / len(stems) if stems else 0.0

    if anchor_hits > 0 and ratio >= 0.33:
        return "direct", keyword_hits, ratio, anchor_hits
    if ratio >= 0.67:
        return "direct", keyword_hits, ratio, anchor_hits
    if ratio >= 0.33 or anchor_hits > 0:
        return "partial", keyword_hits, ratio, anchor_hits
    return "gap", keyword_hits, ratio, anchor_hits


def _lookup_point_by_chunk_id(client: Any, *, collection_name: str, chunk_id: str) -> dict[str, Any] | None:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    scroll_filter = Filter(must=[FieldCondition(key="chunk_id", match=MatchValue(value=int(chunk_id)))])
    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=scroll_filter,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        return None
    return dict(points[0].payload or {})


def enrich_hit(hit: dict[str, Any], *, item: LegalQaItem, client: Any, collection_name: str) -> EnrichedHit:
    payload = _lookup_point_by_chunk_id(client, collection_name=collection_name, chunk_id=str(hit["chunk_id"])) or {}
    chunk_meta = _parse_chunk_metadata(payload)
    document_id = str(payload.get("document_id") or "") or None
    source_document_id = (
        str(payload.get("source_document_id") or chunk_meta.get("source_document_id") or "") or None
    )
    ecli = str(payload.get("ecli") or document_id or source_document_id or "") or None
    case_number = str(
        payload.get("case_reference")
        or payload.get("spisova_znacka")
        or chunk_meta.get("case_reference")
        or chunk_meta.get("spisova_znacka")
        or chunk_meta.get("case_number")
        or ""
    ) or None
    decision_date = str(payload.get("decision_date") or chunk_meta.get("decision_date") or "") or None
    text_snippet = str(hit.get("text_snippet") or payload.get("text") or "")
    support_level, keyword_hit_count, keyword_hit_ratio, anchor_hit_count = classify_support_level(
        item=item,
        text=text_snippet,
        legal_area=str(chunk_meta.get("legal_area") or ""),
    )
    baseline_keys = set(hit.keys()) | set((hit.get("metadata") or {}).keys())
    baseline_provenance_present = any(
        key in baseline_keys for key in {"document_id", "source_document_id", "ecli", "case_reference", "spisova_znacka"}
    )
    return EnrichedHit(
        rank=int(hit["rank"]),
        chunk_id=str(hit["chunk_id"]),
        document_id=document_id,
        source_document_id=source_document_id or document_id,
        ecli=ecli,
        case_reference=case_number,
        spisova_znacka=case_number,
        decision_date=decision_date,
        source=str(payload.get("source") or hit.get("source") or ""),
        text_snippet=text_snippet,
        metadata_keys_present=sorted(set((hit.get("metadata") or {}).keys()) | set(payload.keys()) | set(chunk_meta.keys())),
        provenance_sufficient_for_gold=bool(source_document_id or document_id),
        section_type=str(chunk_meta.get("section_type") or ""),
        legal_area=str(chunk_meta.get("legal_area") or ""),
        keyword_hit_count=keyword_hit_count,
        keyword_hit_ratio=keyword_hit_ratio,
        anchor_hit_count=anchor_hit_count,
        support_level=support_level,
        baseline_provenance_present=baseline_provenance_present,
    )


def choose_candidate(hits: list[EnrichedHit]) -> EnrichedHit | None:
    scored = [hit for hit in hits if hit.provenance_sufficient_for_gold]
    if not scored:
        return None
    support_weight = {"direct": 3, "partial": 2, "gap": 1, "boilerplate_noise": 0}
    return max(
        scored,
        key=lambda hit: (
            support_weight[hit.support_level],
            -hit.rank,
            hit.keyword_hit_ratio,
            hit.anchor_hit_count,
        ),
    )


def classify_candidate(candidate: EnrichedHit | None, hits: list[EnrichedHit]) -> tuple[str, str, str]:
    if candidate is None:
        has_signal_without_provenance = any(hit.support_level in {"direct", "partial"} for hit in hits)
        if has_signal_without_provenance:
            return (
                "needs_provenance_export_fix",
                "No clean source_document_id/ECLI available after baseline inspection.",
                "fix_export",
            )
        return ("corpus_gap", "No sufficiently relevant NSoud hit found in top-10.", "skip")

    if candidate.support_level == "boilerplate_noise":
        return (
            "boilerplate_noise",
            "Best provenance-backed hit is boilerplate or operative text only.",
            "skip",
        )
    if candidate.support_level == "gap":
        return (
            "corpus_gap",
            "Top-10 hits have provenance, but snippet support for the question is too weak.",
            "skip",
        )
    if candidate.support_level == "direct" and candidate.rank == 1:
        return (
            "gold_ready_direct",
            "Read-only Qdrant confirms clean provenance and rank-1 snippet directly supports the question.",
            "annotate_gold",
        )
    if candidate.support_level == "direct" and candidate.rank <= 2:
        return (
            "gold_ready_partial",
            "Read-only Qdrant confirms clean provenance; a direct-support candidate exists in top-2, but not at rank-1.",
            "annotate_gold",
        )
    return (
        "needs_manual_review",
        "Qdrant resolved provenance, but the best supportive hit is still only partial or falls below the conservative rank threshold.",
        "manual_review",
    )


def _write_jsonl(path: Path, rows: list[CandidateRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def _render_hit_table_rows(hits: list[EnrichedHit]) -> list[str]:
    lines = [
        "| rank | chunk_id | document_id | source_document_id | ecli | case_reference | spisova_znacka | decision_date | source | support | provenance_ok | metadata keys | snippet |",
        "|------|----------|-------------|--------------------|------|----------------|----------------|---------------|--------|---------|---------------|---------------|---------|",
    ]
    for hit in hits:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(hit.rank),
                    hit.chunk_id,
                    hit.document_id or "",
                    hit.source_document_id or "",
                    hit.ecli or "",
                    hit.case_reference or "",
                    hit.spisova_znacka or "",
                    hit.decision_date or "",
                    hit.source or "",
                    hit.support_level,
                    "yes" if hit.provenance_sufficient_for_gold else "no",
                    ", ".join(hit.metadata_keys_present),
                    hit.text_snippet.replace("\n", " ")[:180],
                ]
            )
            + " |"
        )
    return lines


def _write_report(path: Path, rows: list[CandidateRecord], details: dict[str, list[EnrichedHit]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.classification] = counts.get(row.classification, 0) + 1

    lines = [
        "# NSoud Provenance Check — 2026-07-10",
        "",
        "Read-only provenance verification over pending NSoud legal QA items.",
        "",
        "## Summary",
        "",
        f"- Questions inspected: `{len(rows)}`",
        f"- Read-only Qdrant lookup used: `yes`",
        f"- Qdrant write occurred: `no`",
        f"- Aliases touched: `no`",
        f"- Retrieval logic changed: `no`",
        "",
        "## Classification Counts",
        "",
    ]
    for key in sorted(counts):
        lines.append(f"- `{key}`: {counts[key]}")
    lines.extend(
        [
            "",
            "## Candidate Summary",
            "",
            "| question_id | question | top_candidate_rank | candidate_ecli | candidate_source_document_id | candidate_case_reference | candidate_chunk_id | support_level | classification | reason | recommended_action |",
            "|-------------|----------|--------------------|----------------|------------------------------|--------------------------|--------------------|---------------|----------------|--------|--------------------|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.question_id,
                    row.question,
                    "" if row.candidate_rank is None else str(row.candidate_rank),
                    row.candidate_ecli or "",
                    row.candidate_source_document_id or "",
                    row.candidate_case_reference or "",
                    row.candidate_chunk_id or "",
                    row.support_level,
                    row.classification,
                    row.reason,
                    row.recommended_action,
                ]
            )
            + " |"
        )

    for row in rows:
        lines.extend(
            [
                "",
                f"## {row.question_id}",
                "",
                f"- Question: {row.question}",
                f"- Classification: `{row.classification}`",
                f"- Recommended action: `{row.recommended_action}`",
                "",
            ]
        )
        lines.extend(_render_hit_table_rows(details[row.question_id]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_check(args: argparse.Namespace) -> list[CandidateRecord]:
    retrieval_by_id = load_retrieval_results(Path(args.retrieval_results))
    dataset_items = load_dataset(Path(args.dataset))
    pending_items = [item for item in dataset_items if item.corpus == "nsoud" and item.source_pending]
    client = _qdrant_client(args.qdrant_url)

    rows: list[CandidateRecord] = []
    detail_map: dict[str, list[EnrichedHit]] = {}
    for item in pending_items:
        retrieval = retrieval_by_id[item.id]
        enriched_hits = [
            enrich_hit(hit, item=item, client=client, collection_name=args.collection_name)
            for hit in list(retrieval.get("hits") or [])[:10]
        ]
        candidate = choose_candidate(enriched_hits)
        classification, reason, recommended_action = classify_candidate(candidate, enriched_hits)
        detail_map[item.id] = enriched_hits
        rows.append(
            CandidateRecord(
                question_id=item.id,
                question=item.question,
                candidate_rank=candidate.rank if candidate else None,
                candidate_chunk_id=candidate.chunk_id if candidate else None,
                candidate_source_document_id=candidate.source_document_id if candidate else None,
                candidate_ecli=candidate.ecli if candidate else None,
                candidate_case_reference=candidate.case_reference if candidate else None,
                candidate_decision_date=candidate.decision_date if candidate else None,
                support_level=candidate.support_level if candidate else "gap",
                classification=classification,
                reason=reason,
                recommended_action=recommended_action,
            )
        )

    _write_jsonl(Path(args.output_jsonl), rows)
    _write_report(Path(args.output_report), rows, detail_map)
    return rows


def main() -> int:
    args = parse_args()
    rows = run_check(args)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.classification] = counts.get(row.classification, 0) + 1
    print(json.dumps({"checked": len(rows), "classification_counts": counts}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
