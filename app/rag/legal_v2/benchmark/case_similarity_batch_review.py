"""Batch relevance review helpers for Golden v3 (propose / apply / qrels).

Retrieval ranks must never determine grades. Grades come only from confirmed
human (or human-confirmed agent) judgments.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.rag.legal_v2.benchmark.case_similarity_golden_v3 import GRADE_LABELS
from app.rag.legal_v2.benchmark.case_similarity_graded_eval import (
    JUDGMENT_EXPLICIT_NOT_RELEVANT,
    JUDGMENT_GRADED,
    QrelEntry,
)

DEFAULT_ARTIFACTS = (
    Path(__file__).resolve().parents[4] / "artifacts" / "legal_v2" / "golden_v3_graded"
)
DEFAULT_QUEUE = DEFAULT_ARTIFACTS / "relevance_review_queue.jsonl"
DEFAULT_QUERY_REVIEW = DEFAULT_ARTIFACTS / "query_review.jsonl"
DEFAULT_BATCHES = DEFAULT_ARTIFACTS / "batches"

_TOKEN_RE = re.compile(r"[0-9A-Za-zÁ-Žá-ž§]+", re.UNICODE)

# Content tokens only — never use dense/bm25/hybrid rank for grading.
_STOP = frozenset(
    {
        "a",
        "i",
        "o",
        "u",
        "v",
        "ve",
        "na",
        "do",
        "ze",
        "z",
        "se",
        "je",
        "to",
        "k",
        "ke",
        "jak",
        "pro",
        "při",
        "podle",
        "nebo",
        "že",
        "který",
        "která",
        "které",
        "soud",
        "ústavní",
        "ústavního",
        "stížnost",
        "rozhodnutí",
        "řízení",
    }
)


@dataclass(frozen=True)
class BatchSelection:
    split: str
    batch_index: int
    query_ids: list[str]
    pending_only: bool = True


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def list_split_query_ids(queue_rows: Sequence[dict[str, Any]], split: str) -> list[str]:
    ids = sorted({str(row["query_id"]) for row in queue_rows if row.get("split") == split})
    return ids


def select_batch_query_ids(
    queue_rows: Sequence[dict[str, Any]],
    *,
    split: str,
    batch_size: int = 5,
    batch_index: int = 1,
    pending_only: bool = True,
) -> BatchSelection:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if batch_index <= 0:
        raise ValueError("batch_index must be positive")
    all_ids = list_split_query_ids(queue_rows, split)
    if pending_only:
        pending_ids: list[str] = []
        for query_id in all_ids:
            rows = [row for row in queue_rows if row.get("query_id") == query_id]
            if any(row.get("review_status") != "reviewed" or row.get("relevance_grade") is None for row in rows):
                pending_ids.append(query_id)
        candidate_ids = pending_ids
    else:
        candidate_ids = all_ids
    start = (batch_index - 1) * batch_size
    selected = candidate_ids[start : start + batch_size]
    return BatchSelection(split=split, batch_index=batch_index, query_ids=selected, pending_only=pending_only)


def _tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_RE.findall(text or "")
        if len(token) >= 3 and token.casefold() not in _STOP
    }


def content_overlap_score(query_text: str, candidate_text: str) -> float:
    q = _tokens(query_text)
    c = _tokens(candidate_text)
    if not q or not c:
        return 0.0
    return len(q & c) / len(q)


def propose_grade_from_content(
    *,
    query_text: str,
    candidate_summary: str,
    reasoning_excerpt: str,
    central_legal_issue: str,
    is_legacy_primary: bool,
) -> tuple[int | None, str, bool]:
    """Propose a grade from query/judgment text only. Never uses retrieval ranks."""
    blob = " ".join(
        part
        for part in (candidate_summary, reasoning_excerpt, central_legal_issue)
        if part
    )
    overlap = content_overlap_score(query_text, blob)
    query_l = (query_text or "").casefold()
    blob_l = blob.casefold()

    # Domain cues shared by formal-complaint / representation themes.
    cue_hits = 0
    for cue in (
        "advokát",
        "zastoupen",
        "formáln",
        "náležitost",
        "vadn",
        "poučit",
        "odmít",
        "vyživovac",
        "dítě",
        "dovolán",
        "ústní jednán",
        "restitu",
        "opatrovník",
        "squeeze",
        "podjat",
        "vazb",
    ):
        if cue in query_l and cue in blob_l:
            cue_hits += 1

    needs_check = False
    if is_legacy_primary and overlap >= 0.12:
        grade = 3
        reason = "Legacy target; text aligns with the query legal issue."
    elif is_legacy_primary:
        grade = 2
        reason = "Legacy target; partial textual alignment — verify centrality."
        needs_check = True
    elif overlap >= 0.45 and cue_hits >= 1:
        grade = 3
        reason = f"Strong content overlap ({overlap:.2f}) and shared legal cues."
    elif overlap >= 0.32 or cue_hits >= 2:
        grade = 2
        reason = f"Clear topical overlap ({overlap:.2f}, cues={cue_hits})."
    elif overlap >= 0.18 or cue_hits == 1:
        grade = 1
        reason = f"Adjacent / partial overlap ({overlap:.2f})."
        if 0.16 <= overlap <= 0.22 and cue_hits == 0:
            needs_check = True
            return None, "Borderline partial vs irrelevant — needs human check.", True
    elif overlap >= 0.10:
        grade = None
        reason = "Weak overlap; ambiguous relevance — needs human check."
        needs_check = True
    else:
        grade = 0
        reason = f"Little meaningful overlap with the query ({overlap:.2f})."

    if needs_check and grade is not None and grade in {1, 2}:
        return None, reason + " Marked NEEDS_HUMAN_CHECK.", True
    return grade, reason, needs_check


def build_proposal_rows(
    queue_rows: Sequence[dict[str, Any]],
    *,
    query_ids: Sequence[str],
    legacy_by_query: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    legacy_by_query = legacy_by_query or {}
    selected = {qid for qid in query_ids}
    proposals: list[dict[str, Any]] = []
    for row in queue_rows:
        query_id = str(row.get("query_id") or "")
        if query_id not in selected:
            continue
        doc_id = str(row.get("document_id") or row.get("ecli") or "")
        legacy = legacy_by_query.get(query_id) or ""
        is_legacy = bool(legacy) and doc_id == legacy
        grade, reason, needs_check = propose_grade_from_content(
            query_text=str(row.get("query_text") or ""),
            candidate_summary=str(row.get("candidate_summary") or ""),
            reasoning_excerpt=str(row.get("reasoning_excerpt") or ""),
            central_legal_issue=str(row.get("central_legal_issue") or ""),
            is_legacy_primary=is_legacy,
        )
        label = GRADE_LABELS.get(grade) if grade is not None else None
        proposals.append(
            {
                "query_id": query_id,
                "split": row.get("split"),
                "query_text": row.get("query_text"),
                "document_id": doc_id,
                "ecli": row.get("ecli") or doc_id,
                "case_reference": row.get("case_reference"),
                "court": row.get("court"),
                "decision_date": row.get("decision_date"),
                "document_type": row.get("document_type"),
                "candidate_summary": row.get("candidate_summary"),
                "central_legal_issue": row.get("central_legal_issue"),
                "reasoning_excerpt": row.get("reasoning_excerpt"),
                "is_legacy_primary": is_legacy,
                "found_by_dense": row.get("found_by_dense"),
                "dense_rank": row.get("dense_rank"),
                "found_by_bm25": row.get("found_by_bm25"),
                "bm25_rank": row.get("bm25_rank"),
                "found_by_hybrid": row.get("found_by_hybrid"),
                "hybrid_rank": row.get("hybrid_rank"),
                "proposed_grade": grade,
                "proposed_label": label,
                "proposed_reason": reason,
                "needs_human_check": needs_check,
                "final_grade": None,
                "final_reason": "",
                "review_status": "proposed",
            }
        )
    return proposals


def render_batch_summary(proposals: Sequence[dict[str, Any]], *, batch_name: str) -> str:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in proposals:
        by_query[str(row["query_id"])].append(row)
    lines = [
        f"# {batch_name} — Relevance Proposal Summary",
        "",
        "Retrieval ranks are discovery metadata only. Grades are content-proposed drafts.",
        "Confirm or edit, then write `*_confirmed.jsonl` and run apply.",
        "",
    ]
    for query_id in sorted(by_query):
        rows = by_query[query_id]
        query_text = str(rows[0].get("query_text") or "")
        hist = Counter(
            "NEEDS_HUMAN_CHECK" if row.get("needs_human_check") or row.get("proposed_grade") is None else str(row.get("proposed_grade"))
            for row in rows
        )
        lines.extend(
            [
                f"## {query_id}",
                "",
                f"**Query:** {query_text}",
                "",
                f"**Candidates:** {len(rows)}",
                f"**Histogram:** {dict(sorted(hist.items(), key=lambda item: str(item[0])))}",
                "",
                "### Proposed grade 3",
            ]
        )
        grade3 = [row for row in rows if row.get("proposed_grade") == 3]
        if not grade3:
            lines.append("- (none)")
        for row in grade3:
            lines.append(
                f"- `{row.get('ecli')}` — {row.get('proposed_reason')} "
                f"(legacy={row.get('is_legacy_primary')})"
            )
        lines.extend(["", "### Proposed grade 2"])
        grade2 = [row for row in rows if row.get("proposed_grade") == 2]
        if not grade2:
            lines.append("- (none)")
        for row in grade2[:12]:
            lines.append(f"- `{row.get('ecli')}` — {row.get('proposed_reason')}")
        if len(grade2) > 12:
            lines.append(f"- … +{len(grade2) - 12} more")
        lines.extend(["", "### NEEDS_HUMAN_CHECK"])
        needs = [row for row in rows if row.get("needs_human_check") or row.get("proposed_grade") is None]
        if not needs:
            lines.append("- (none)")
        for row in needs:
            lines.append(f"- `{row.get('ecli')}` — {row.get('proposed_reason')}")
        lines.extend(["", "---", ""])
    return "\n".join(lines)


def normalize_confirmed_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Accept confirmed rows; reject unresolved NEEDS_HUMAN_CHECK."""
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("needs_human_check") and row.get("final_grade") is None and row.get("proposed_grade") is None:
            raise ValueError(
                f"{row.get('query_id')} / {row.get('document_id')}: NEEDS_HUMAN_CHECK has no final_grade"
            )
        grade = row.get("final_grade")
        if grade is None:
            if row.get("needs_human_check"):
                raise ValueError(
                    f"{row.get('query_id')} / {row.get('document_id')}: unresolved NEEDS_HUMAN_CHECK"
                )
            grade = row.get("proposed_grade")
        if grade is None:
            raise ValueError(f"{row.get('query_id')} / {row.get('document_id')}: missing final_grade")
        grade_i = int(grade)
        if grade_i not in GRADE_LABELS:
            raise ValueError(f"invalid grade {grade_i}")
        reason = str(row.get("final_reason") or row.get("proposed_reason") or "").strip()
        out.append(
            {
                "query_id": str(row["query_id"]),
                "document_id": str(row.get("document_id") or row.get("ecli")),
                "final_grade": grade_i,
                "final_reason": reason,
                "review_status": "reviewed",
            }
        )
    return out


def apply_confirmed_to_queue(
    queue_rows: list[dict[str, Any]],
    confirmed: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    confirmed_norm = normalize_confirmed_rows(confirmed)
    index = {(row["query_id"], row["document_id"]): row for row in confirmed_norm}
    updated = 0
    out: list[dict[str, Any]] = []
    for row in queue_rows:
        key = (str(row.get("query_id")), str(row.get("document_id") or row.get("ecli")))
        patch = index.get(key)
        if patch is None:
            out.append(row)
            continue
        new_row = dict(row)
        new_row["relevance_grade"] = patch["final_grade"]
        new_row["relevance_label"] = GRADE_LABELS[patch["final_grade"]]
        new_row["reviewer_notes"] = patch["final_reason"]
        new_row["review_status"] = "reviewed"
        out.append(new_row)
        updated += 1
    missing = len(confirmed_norm) - updated
    if missing:
        raise ValueError(f"{missing} confirmed rows not found in review queue")
    return out, updated


def reviewed_qrel_entries(
    queue_rows: Sequence[dict[str, Any]],
    *,
    split: str | None = None,
) -> list[QrelEntry]:
    entries: list[QrelEntry] = []
    for row in queue_rows:
        if split and row.get("split") != split:
            continue
        if row.get("review_status") != "reviewed":
            continue
        grade = row.get("relevance_grade")
        if grade is None:
            continue
        grade_i = int(grade)
        state = JUDGMENT_EXPLICIT_NOT_RELEVANT if grade_i == 0 else JUDGMENT_GRADED
        entries.append(
            QrelEntry(
                query_id=str(row["query_id"]),
                document_id=str(row.get("document_id") or row.get("ecli")),
                grade=grade_i,
                judgment_state=state,
                review_reason=str(row.get("reviewer_notes") or ""),
            )
        )
    return entries


def split_review_complete(queue_rows: Sequence[dict[str, Any]], split: str) -> bool:
    rows = [row for row in queue_rows if row.get("split") == split]
    if not rows:
        return False
    query_ids = {str(row["query_id"]) for row in rows}
    for query_id in query_ids:
        qrows = [row for row in rows if row.get("query_id") == query_id]
        if any(row.get("review_status") != "reviewed" or row.get("relevance_grade") is None for row in qrows):
            return False
    return True


def assert_freeze_allowed(queue_rows: Sequence[dict[str, Any]]) -> None:
    if not split_review_complete(queue_rows, "dev"):
        raise RuntimeError("freeze blocked: DEV relevance review incomplete")
    if not split_review_complete(queue_rows, "test"):
        raise RuntimeError("freeze blocked: TEST relevance review incomplete")


def qrels_to_jsonl_rows(entries: Sequence[QrelEntry]) -> list[dict[str, Any]]:
    return [
        {
            "query_id": entry.query_id,
            "document_id": entry.document_id,
            "grade": entry.grade,
            "label": GRADE_LABELS[entry.grade],
            "judgment_state": entry.judgment_state,
            "review_reason": entry.review_reason,
        }
        for entry in entries
    ]


def load_legacy_primary_map(benchmark_path: Path | str) -> dict[str, str]:
    from app.rag.legal_v2.benchmark.case_similarity_golden_v3 import (
        load_case_similarity_golden_v3_jsonl,
    )

    items = load_case_similarity_golden_v3_jsonl(benchmark_path)
    return {
        item.query_id: str(item.legacy_primary_ecli or item.legacy_primary_document_id)
        for item in items
    }


def update_query_review_statuses(
    query_review_rows: list[dict[str, Any]],
    *,
    query_ids: Sequence[str],
    status: str,
) -> list[dict[str, Any]]:
    wanted = set(query_ids)
    out: list[dict[str, Any]] = []
    for row in query_review_rows:
        new_row = dict(row)
        if new_row.get("query_id") in wanted:
            new_row["review_status"] = status
        out.append(new_row)
    return out
