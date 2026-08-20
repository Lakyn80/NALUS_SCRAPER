#!/usr/bin/env python3
"""Build human-audit priority set from golden_v3 relevance_review_queue.jsonl.

Does NOT change grades. Outputs:
  - artifacts/legal_v2/golden_v3_graded/human_audit_priority.jsonl
  - artifacts/legal_v2/golden_v3_graded/HUMAN_AUDIT_PRIORITY.md
  - updates benchmarks/.../case_similarity_golden_v3_graded.meta.json
  - updates artifacts/.../REVIEW_PROGRESS.md
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUEUE = PROJECT_ROOT / "artifacts/legal_v2/golden_v3_graded/relevance_review_queue.jsonl"
OUT_JSONL = PROJECT_ROOT / "artifacts/legal_v2/golden_v3_graded/human_audit_priority.jsonl"
OUT_MD = PROJECT_ROOT / "artifacts/legal_v2/golden_v3_graded/HUMAN_AUDIT_PRIORITY.md"
META = PROJECT_ROOT / "benchmarks/legal_v2/case_similarity_golden_v3_graded.meta.json"
REVIEW_PROGRESS = PROJECT_ROOT / "artifacts/legal_v2/golden_v3_graded/REVIEW_PROGRESS.md"
PROBE = PROJECT_ROOT / "artifacts/legal_v2/golden_v3_graded/_agent_fs_probe.txt"

BUCKET_GRADE3 = "grade_3"
BUCKET_GRADE2 = "grade_2"
BUCKET_PROVISIONAL = "PROVISIONAL_AGENT_RESOLUTION"
BUCKET_ORDER = {BUCKET_GRADE3: 0, BUCKET_GRADE2: 1, BUCKET_PROVISIONAL: 2}


def _trunc(text: Any, limit: int) -> str | None:
    if text is None:
        return None
    s = str(text)
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def _key(row: dict[str, Any]) -> tuple[str, str]:
    doc = str(row.get("document_id") or row.get("ecli") or "")
    return (str(row.get("query_id") or ""), doc)


def _buckets(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    grade = row.get("relevance_grade")
    if grade == 3:
        out.append(BUCKET_GRADE3)
    if grade == 2:
        out.append(BUCKET_GRADE2)
    notes = str(row.get("reviewer_notes") or "")
    if notes.startswith("PROVISIONAL_AGENT_RESOLUTION"):
        out.append(BUCKET_PROVISIONAL)
    return out


def _primary_bucket(buckets: list[str]) -> str:
    return min(buckets, key=lambda b: BUCKET_ORDER[b])


def load_queue(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def to_priority_row(row: dict[str, Any], buckets: list[str]) -> dict[str, Any]:
    split_raw = str(row.get("split") or "").strip().lower()
    split = "DEV" if split_raw == "dev" else "TEST" if split_raw == "test" else split_raw.upper()
    out: dict[str, Any] = {
        "query_id": row.get("query_id"),
        "split": split,
        "query_text": row.get("query_text"),
        "ecli": row.get("ecli") or row.get("document_id"),
        "document_id": row.get("document_id") or row.get("ecli"),
        "proposed_grade": row.get("relevance_grade"),
        "proposed_label": row.get("relevance_label"),
        "review_reason": row.get("reviewer_notes"),
        "candidate_summary": _trunc(row.get("candidate_summary"), 500),
        "reasoning_excerpt": _trunc(row.get("reasoning_excerpt"), 400),
        "dense_rank": row.get("dense_rank"),
        "bm25_rank": row.get("bm25_rank"),
        "hybrid_rank": row.get("hybrid_rank"),
        "found_by_dense": row.get("found_by_dense"),
        "found_by_bm25": row.get("found_by_bm25"),
        "found_by_hybrid": row.get("found_by_hybrid"),
        "priority_buckets": buckets,
        "annotation_source": "AGENT_FIRST_PASS",
        "human_status": "pending",
        "human_grade": None,
        "human_notes": None,
    }
    if row.get("case_reference") is not None:
        out["case_reference"] = row.get("case_reference")
    return out


def sort_key(row: dict[str, Any]) -> tuple:
    split_ord = 0 if row["split"] == "DEV" else 1
    primary = _primary_bucket(row["priority_buckets"])
    return (split_ord, str(row["query_id"]), BUCKET_ORDER[primary], str(row.get("document_id") or ""))


def compute_counts(selected: dict[tuple[str, str], list[str]]) -> dict[str, int]:
    a = {k for k, b in selected.items() if BUCKET_GRADE3 in b}
    bset = {k for k, b in selected.items() if BUCKET_GRADE2 in b}
    c = {k for k, b in selected.items() if BUCKET_PROVISIONAL in b}
    return {
        "n_grade_3": len(a),
        "n_grade_2": len(bset),
        "n_provisional": len(c),
        "n_a_and_b": len(a & bset),
        "n_a_and_c": len(a & c),
        "n_b_and_c": len(bset & c),
        "n_a_and_b_and_c": len(a & bset & c),
        "n_unique": len(selected),
    }


def write_md(path: Path, counts: dict[str, int], rows: list[dict[str, Any]]) -> None:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[_primary_bucket(row["priority_buckets"])].append(row)

    def preview(text: Any, n: int = 120) -> str:
        s = str(text or "").replace("|", "/").replace("\n", " ")
        return s if len(s) <= n else s[: n - 1] + "…"

    lines: list[str] = []
    lines.append("# Human Audit Priority — Golden v3 Graded")
    lines.append("")
    lines.append("> **BANNER / STATUS:** Current qrels are **PROVISIONAL_AGENT_QRELS** / **AGENT-FIRST-PASS**.")
    lines.append("> They are **NOT** final human-reviewed ground truth.")
    lines.append("> They are **NOT** `HUMAN_REVIEWED`. Do **NOT** treat `QRELS_FROZEN` as final.")
    lines.append("")
    lines.append("## Exact counts")
    lines.append("")
    lines.append(f"| Metric | Count |")
    lines.append(f"| --- | ---: |")
    lines.append(f"| n_grade_3 (set A) | {counts['n_grade_3']} |")
    lines.append(f"| n_grade_2 (set B) | {counts['n_grade_2']} |")
    lines.append(f"| n_provisional (set C) | {counts['n_provisional']} |")
    lines.append(f"| \\|A ∩ B\\| | {counts['n_a_and_b']} |")
    lines.append(f"| \\|A ∩ C\\| | {counts['n_a_and_c']} |")
    lines.append(f"| \\|B ∩ C\\| | {counts['n_b_and_c']} |")
    lines.append(f"| \\|A ∩ B ∩ C\\| | {counts['n_a_and_b_and_c']} |")
    lines.append(f"| **unique priority judgments** | **{counts['n_unique']}** |")
    lines.append("")
    lines.append("## Human review instructions")
    lines.append("")
    lines.append("1. Open `human_audit_priority.jsonl` (full list) and fill `human_status` / `human_grade` / `human_notes`.")
    lines.append("2. Confirm in order: **grade 3** first, then **grade 2**, then **provisional** (`PROVISIONAL_AGENT_RESOLUTION`).")
    lines.append("3. Do **not** auto-change agent grades in the queue while auditing; record human fields only.")
    lines.append("4. Do **not** treat existing qrels / meta `QRELS_FROZEN` as final ground truth.")
    lines.append("5. No A/B retrieval runs until the priority human audit is complete.")
    lines.append("")
    lines.append("Full judgment payloads (summaries, ranks, buckets) live in `human_audit_priority.jsonl`.")
    lines.append("Below: first ~30 of each priority group (primary bucket).")
    lines.append("")

    group_titles = [
        (BUCKET_GRADE3, "Grade 3"),
        (BUCKET_GRADE2, "Grade 2"),
        (BUCKET_PROVISIONAL, "Provisional (PROVISIONAL_AGENT_RESOLUTION)"),
    ]
    for bucket, title in group_titles:
        group = by_group.get(bucket, [])
        lines.append(f"## {title} (n={len(group)}; showing first {min(30, len(group))})")
        lines.append("")
        lines.append("| query_id | split | ecli | proposed_grade | reason preview |")
        lines.append("| --- | --- | --- | ---: | --- |")
        for row in group[:30]:
            lines.append(
                f"| {row['query_id']} | {row['split']} | {row.get('ecli') or ''} | "
                f"{row.get('proposed_grade')} | {preview(row.get('review_reason'))} |"
            )
        if len(group) > 30:
            lines.append("")
            lines.append(f"_… {len(group) - 30} more in `human_audit_priority.jsonl`._")
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def update_meta(path: Path, qrels_count: int) -> None:
    meta: dict[str, Any] = {}
    if path.exists():
        meta = json.loads(path.read_text(encoding="utf-8"))
    frozen_at = meta.pop("qrels_frozen_at_utc", None)
    if frozen_at and "provisional_qrels_built_at_utc" not in meta:
        meta["provisional_qrels_built_at_utc"] = frozen_at
    meta["human_review_status"] = "PROVISIONAL_AGENT_QRELS"
    meta["annotation_status"] = "PROVISIONAL_AGENT_QRELS"
    meta["qrels_count"] = qrels_count
    meta["notes"] = (
        "Agent-first-pass annotations awaiting human audit of the priority set "
        "(grade 3, grade 2, PROVISIONAL_AGENT_RESOLUTION). "
        "NOT FINAL; NOT HUMAN_REVIEWED; NOT QRELS_FROZEN."
    )
    meta["qrels_frozen_at_utc"] = None
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def update_review_progress(path: Path, counts: dict[str, int]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    body = f"""# Golden v3 Graded — Review Progress

## Status: PROVISIONAL_AGENT_QRELS

Current qrels are **agent-first-pass** annotations only.

- **NOT** final human-reviewed ground truth
- **NOT** `HUMAN_REVIEWED`
- **NOT** `QRELS_FROZEN`
- Awaiting human audit of the priority set in `human_audit_priority.jsonl`
- **No A/B retrieval yet**

### Priority audit counts (as of {now})

| Metric | Count |
| --- | ---: |
| n_grade_3 | {counts['n_grade_3']} |
| n_grade_2 | {counts['n_grade_2']} |
| n_provisional | {counts['n_provisional']} |
| unique priority | {counts['n_unique']} |

See `HUMAN_AUDIT_PRIORITY.md` for instructions and preview tables.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def main() -> int:
    if not QUEUE.exists():
        raise SystemExit(f"Missing queue: {QUEUE}")
    rows = load_queue(QUEUE)
    selected_buckets: dict[tuple[str, str], list[str]] = {}
    selected_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        buckets = _buckets(row)
        if not buckets:
            continue
        k = _key(row)
        # Union buckets if duplicate keys appear
        prev = selected_buckets.get(k, [])
        merged = list(dict.fromkeys([*prev, *buckets]))
        selected_buckets[k] = merged
        selected_rows[k] = row

    priority_rows = [
        to_priority_row(selected_rows[k], selected_buckets[k]) for k in selected_rows
    ]
    priority_rows.sort(key=sort_key)
    counts = compute_counts(selected_buckets)

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as fh:
        for row in priority_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_md(OUT_MD, counts, priority_rows)
    update_meta(META, qrels_count=len(rows))
    update_review_progress(REVIEW_PROGRESS, counts)
    if PROBE.exists():
        PROBE.unlink()

    summary = {
        "queue_rows": len(rows),
        **counts,
        "out_jsonl": str(OUT_JSONL),
        "out_md": str(OUT_MD),
        "meta_status": "PROVISIONAL_AGENT_QRELS",
        "review_progress_updated": True,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
