#!/usr/bin/env python3
"""Apply HUMAN_AUDIT_BATCH_3 confirms + merge query 015 ECLI alias into canonical."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts" / "legal_v2" / "golden_v3_graded"
PRIORITY = BASE / "human_audit_priority.jsonl"
QUEUE = BASE / "relevance_review_queue.jsonl"
PROGRESS = BASE / "REVIEW_PROGRESS.md"

CANONICAL_015 = "ECLI:CZ:US:1999:4.US.23.99.1"
ALIAS_015 = "ECLI:CZ:US:1999:4.US.23.99"
QUERY_015 = "nalus-cs-v2-015"

BATCH3: list[tuple[str, str]] = [
    ("nalus-cs-v2-020", "3.US.277.96"),
    ("nalus-cs-v2-022", "Pl.US.33.97"),
    ("nalus-cs-v2-023", "II.US.339.97"),
    ("nalus-cs-v2-024", "III.US.758.16"),
    ("nalus-cs-v2-026", "Pl.US.12.20"),
]

_ROMAN = {
    "I": "1",
    "II": "2",
    "III": "3",
    "IV": "4",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def extract_case_token(value: str) -> str:
    """Normalized case token like 2.US.339.97 or PL.US.33.97 (no year/ECLI prefix)."""
    s = str(value or "").strip().upper()
    s = s.replace("ÚS", "US").replace("U.S.", "US")
    s = re.sub(r"\s+", "", s)
    m = re.match(r"ECLI:[A-Z]+:[A-Z]+:\d{4}:(.+)$", s)
    if m:
        s = m.group(1)
    def rom(mm: re.Match[str]) -> str:
        tok = mm.group(1)
        return _ROMAN.get(tok, tok) + "."

    s = re.sub(r"^(IV|III|II|I)\.", rom, s)
    return s


def ecli_matches(document_id: str, needle: str) -> bool:
    doc = extract_case_token(document_id)
    nd = extract_case_token(needle)
    if not doc or not nd:
        return False
    if doc == nd:
        return True
    # Optional document ordinal is a single trailing digit segment (e.g. ".1"),
    # never the two-digit year suffix of a sp.zn token like ".97".
    if re.fullmatch(re.escape(nd) + r"\.\d", doc):
        return True
    if re.fullmatch(re.escape(doc) + r"\.\d", nd):
        return True
    return False


def best_rank(a: Any, b: Any) -> Any:
    vals = [x for x in (a, b) if isinstance(x, int)]
    return min(vals) if vals else None


def merge_015_priority(rows: list[dict[str, Any]]) -> dict[str, Any]:
    canon = None
    alias = None
    for r in rows:
        if r.get("query_id") != QUERY_015:
            continue
        did = str(r.get("document_id") or "")
        if did == CANONICAL_015:
            canon = r
        elif did == ALIAS_015:
            alias = r
    if canon is None or alias is None:
        raise RuntimeError(
            f"015 merge rows missing: canon={canon is not None} alias={alias is not None}"
        )

    for flag in ("found_by_dense", "found_by_bm25", "found_by_hybrid"):
        canon[flag] = bool(canon.get(flag)) or bool(alias.get(flag))
    for rk in ("dense_rank", "bm25_rank", "hybrid_rank"):
        canon[rk] = best_rank(canon.get(rk), alias.get(rk))

    aliases = list(canon.get("merged_from_aliases") or [])
    if ALIAS_015 not in aliases:
        aliases.append(ALIAS_015)
    canon["merged_from_aliases"] = aliases
    canon["human_status"] = "confirmed"
    canon["human_grade"] = 3
    canon["proposed_grade"] = 3
    canon["proposed_label"] = "HIGHLY_RELEVANT"
    canon["human_notes"] = (
        "HUMAN_AUDIT_BATCH_3_MERGE_015: confirmed grade 3 — direct match on přísedící conflict / "
        "podjatost (IV.ÚS 23/99). Merged alias ECLI:CZ:US:1999:4.US.23.99 into canonical "
        f"{CANONICAL_015}; ranks/found_by_* unioned."
    )
    canon.pop("is_duplicate", None)
    canon.pop("duplicate_of", None)
    canon.pop("dedup_status", None)
    canon.pop("exclude_from_qrels", None)

    alias["dedup_status"] = "merged_into_canonical"
    alias["duplicate_of"] = CANONICAL_015
    alias["merged_into"] = CANONICAL_015
    alias["is_duplicate"] = True
    alias["exclude_from_qrels"] = True
    alias["human_status"] = "changed"
    alias["human_grade"] = None
    alias["human_notes"] = (
        "HUMAN_AUDIT_BATCH_3_MERGE_015: merged away — not a separate judgment. "
        f"Canonical={CANONICAL_015}."
    )
    return {"canonical": CANONICAL_015, "alias": ALIAS_015, "query_id": QUERY_015}


def merge_015_queue(rows: list[dict[str, Any]]) -> None:
    canon = None
    alias = None
    for r in rows:
        if r.get("query_id") != QUERY_015:
            continue
        did = str(r.get("document_id") or "")
        if did == CANONICAL_015:
            canon = r
        elif did == ALIAS_015:
            alias = r
    if canon is None or alias is None:
        raise RuntimeError("015 queue rows missing")

    for flag in ("found_by_dense", "found_by_bm25", "found_by_hybrid"):
        canon[flag] = bool(canon.get(flag)) or bool(alias.get(flag))
    for rk in ("dense_rank", "bm25_rank", "hybrid_rank"):
        canon[rk] = best_rank(canon.get(rk), alias.get(rk))

    aliases = list(canon.get("merged_from_aliases") or [])
    if ALIAS_015 not in aliases:
        aliases.append(ALIAS_015)
    canon["merged_from_aliases"] = aliases
    canon["relevance_grade"] = 3
    canon["relevance_label"] = "HIGHLY_RELEVANT"
    canon["review_status"] = "human_reviewed"
    prior = str(canon.get("reviewer_notes") or "").strip()
    note = (
        "HUMAN_AUDIT_BATCH_3_MERGE_015: confirmed grade 3 — direct match on přísedící conflict / "
        f"podjatost. Merged alias {ALIAS_015} into canonical; ranks/found_by_* unioned."
    )
    canon["reviewer_notes"] = f"{note} | prior: {prior}" if prior else note
    canon.pop("is_duplicate", None)
    canon.pop("duplicate_of", None)
    canon.pop("dedup_status", None)
    canon.pop("exclude_from_qrels", None)

    alias["dedup_status"] = "merged_into_canonical"
    alias["duplicate_of"] = CANONICAL_015
    alias["merged_into"] = CANONICAL_015
    alias["is_duplicate"] = True
    alias["exclude_from_qrels"] = True
    prior_a = str(alias.get("reviewer_notes") or "").strip()
    note_a = (
        "HUMAN_AUDIT_BATCH_3_MERGE_015: merged away — not a separate judgment; "
        f"exclude_from_qrels=true; canonical={CANONICAL_015}."
    )
    alias["reviewer_notes"] = f"{note_a} | prior: {prior_a}" if prior_a else note_a
    alias["review_status"] = "human_reviewed"
    alias["human_status"] = "changed"
    alias["human_grade"] = None


def apply_batch3_confirm(
    priority: list[dict[str, Any]],
    queue: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    applied: list[dict[str, Any]] = []
    case_notes = {
        "nalus-cs-v2-020": "Direct highly relevant match (3.ÚS 277/96).",
        "nalus-cs-v2-022": "Direct highly relevant match (Pl.ÚS 33/97).",
        "nalus-cs-v2-023": "Direct highly relevant match (II.ÚS 339/97).",
        "nalus-cs-v2-024": "Direct highly relevant match (III.ÚS 758/16).",
        "nalus-cs-v2-026": "Direct highly relevant match (Pl.ÚS 12/20).",
    }
    for query_id, needle in BATCH3:
        pri_hits = [
            r
            for r in priority
            if r.get("query_id") == query_id and ecli_matches(str(r.get("document_id") or ""), needle)
        ]
        que_hits = [
            r
            for r in queue
            if r.get("query_id") == query_id and ecli_matches(str(r.get("document_id") or ""), needle)
        ]
        if len(pri_hits) != 1 or len(que_hits) != 1:
            raise RuntimeError(
                f"Batch3 match ambiguity for {query_id} needle={needle}: "
                f"priority={len(pri_hits)} {[r.get('document_id') for r in pri_hits]} "
                f"queue={len(que_hits)} {[r.get('document_id') for r in que_hits]}"
            )
        pr = pri_hits[0]
        qr = que_hits[0]
        human_note = case_notes[query_id]
        pr["human_status"] = "confirmed"
        pr["human_grade"] = 3
        pr["proposed_grade"] = 3
        pr["proposed_label"] = "HIGHLY_RELEVANT"
        pr["human_notes"] = human_note

        qr["relevance_grade"] = 3
        qr["relevance_label"] = "HIGHLY_RELEVANT"
        qr["review_status"] = "human_reviewed"
        prior = str(qr.get("reviewer_notes") or "").strip()
        qr["reviewer_notes"] = (
            f"HUMAN_AUDIT_BATCH_3: {human_note} | prior: {prior}"
            if prior
            else f"HUMAN_AUDIT_BATCH_3: {human_note}"
        )
        applied.append(
            {
                "query_id": query_id,
                "document_id": pr.get("document_id"),
                "needle": needle,
                "human_grade": 3,
            }
        )
    return applied


def update_progress(applied: list[dict[str, Any]]) -> None:
    ids = ", ".join(a["query_id"].split("-")[-1] for a in applied)
    body = f"""# Golden v3 Graded — Review Progress

## Status: PROVISIONAL_AGENT_QRELS + HUMAN_AUDIT_BATCH_3 applied (015 merge complete)

Current qrels remain **agent-first-pass** annotations only (not rebuilt after this audit).

- **NOT** final human-reviewed ground truth overall
- **NOT** `QRELS_FROZEN` as final
- **HUMAN_AUDIT_BATCH_1 applied** (2026-08-20) for: 002, 005, 006, 008, 009
- **HUMAN_AUDIT_BATCH_2 applied** (2026-08-20) for: 010, 013, 018, 019
  - Confirmed g3: Pl.US-st.36.13 (010), 2.US.1311.24.2 + 4.US.2692.20.1 (013), 3.US.205.97 (018), Pl.US-st.35.13 (019)
  - Changed 3→2: 3.US.4059.18.1 + 2.US.3196.25.1 (013)
- **HUMAN_AUDIT_BATCH_3 applied** (2026-08-20) for: {ids}
  - Confirmed g3: 3.US.277.96 (020), Pl.US.33.97 (022), 2.US.339.97 (023), 3.US.758.16.1 (024), Pl.US.12.20.1 (026)
  - **015 merge complete** — alias `ECLI:CZ:US:1999:4.US.23.99` → canonical `ECLI:CZ:US:1999:4.US.23.99.1`
    - Canonical: human_status=confirmed, human_grade=3; found_by_*/ranks unioned; `merged_from_aliases` set
    - Alias: `dedup_status=merged_into_canonical`, `exclude_from_qrels=true`, preserved for audit trail
    - See also `DUPLICATE_INTEGRITY_AUDIT.md` / `duplicate_integrity_audit.jsonl`
- Next DEV review batch (REVIEW-ONLY): remaining DEV with ≥1 g3 after 026
- **No A/B retrieval yet**
- **Do not rebuild qrels** until human audit batches are intentionally promoted

### Priority audit counts (baseline inventory; as of 2026-08-20T11:53:19.752631+00:00)

| Metric | Count |
| --- | ---: |
| n_grade_3 | 89 |
| n_grade_2 | 252 |
| n_provisional | 451 |
| unique priority | 792 |

See `HUMAN_AUDIT_PRIORITY.md` for instructions and preview tables.
"""
    PROGRESS.write_text(body, encoding="utf-8")


def main() -> int:
    priority = load_jsonl(PRIORITY)
    queue = load_jsonl(QUEUE)
    merge_info = merge_015_priority(priority)
    merge_015_queue(queue)
    applied = apply_batch3_confirm(priority, queue)
    write_jsonl(PRIORITY, priority)
    write_jsonl(QUEUE, queue)
    update_progress(applied)
    report = {
        "merge_015": merge_info,
        "batch3_applied": applied,
        "batch3_count": len(applied),
        "priority_path": str(PRIORITY),
        "queue_path": str(QUEUE),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
