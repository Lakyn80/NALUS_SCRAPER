"""Apply 4 DEV QA upgrades, close DEV relevance, freeze DEV-only qrels.

Label: DEV_QRELS_FROZEN_WITH_AGENT_LOW_GRADE_TAIL
Does NOT freeze TEST / full QRELS_FROZEN / A/B / deploy.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "artifacts" / "legal_v2" / "golden_v3_graded"
PRIORITY = DIR / "human_audit_priority.jsonl"
QUEUE = DIR / "relevance_review_queue.jsonl"
PROGRESS = DIR / "REVIEW_PROGRESS.md"
QRELS_DEV = DIR / "qrels_dev_reviewed.jsonl"
FREEZE_META = DIR / "DEV_QRELS_FREEZE.json"
FREEZE_NOTE = DIR / "DEV_QRELS_FREEZE.md"

GRADE_LABELS = {
    0: "NOT_RELEVANT",
    1: "PARTIALLY_RELEVANT",
    2: "RELEVANT",
    3: "HIGHLY_RELEVANT",
}

BATCH_TAG = "DEV_QA_SPOTCHECK_UPGRADE"

OPS: list[tuple[str, str, int, str]] = [
    (
        "nalus-cs-v2-023",
        "2.US.2684.25",
        2,
        "QA upgrade: rejects as nepřípustná + refuses odložení vykonatelnosti hlavního líčení — prematurity / ongoing appellate main-hearing posture matching query hinge.",
    ),
    (
        "nalus-cs-v2-023",
        "2.US.403.24",
        2,
        "QA upgrade: complaint against non-final judgment treated as předčasná while appeal pending — usable prematurity doctrine.",
    ),
    (
        "nalus-cs-v2-008",
        "4.US.436.03",
        2,
        "QA upgrade: restitution/ownership proof failure (nedoložili vlastnické právo) — clearly usable for missing ownership-docs hinge.",
    ),
    (
        "nalus-cs-v2-009",
        "2.US.1510.21",
        2,
        "QA upgrade: parent/child conflict for filing ÚS complaint — closely related to kolizní opatrovník / child-not-filing standing hinge.",
    ),
]


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ecli_matches(ecli: str, needle: str) -> bool:
    return needle in (ecli or "").replace("ÚS", "US").replace("ú", "u")


def find_hits(rows: list[dict], query_id: str, needle: str) -> list[dict]:
    return [
        r
        for r in rows
        if r.get("query_id") == query_id
        and ecli_matches(str(r.get("ecli") or r.get("document_id") or ""), needle)
    ]


def find_one(rows: list[dict], query_id: str, needle: str) -> dict:
    hits = find_hits(rows, query_id, needle)
    if len(hits) != 1:
        raise RuntimeError(
            f"Match ambiguity for {query_id} {needle}: n={len(hits)} "
            f"{[r.get('ecli') for r in hits]}"
        )
    return hits[0]


def apply_queue(qr: dict, grade: int, notes: str) -> None:
    label = GRADE_LABELS[grade]
    qr["relevance_grade"] = grade
    qr["relevance_label"] = label
    qr["review_status"] = "human_reviewed"
    prior = str(qr.get("reviewer_notes") or "").strip()
    if prior and not prior.startswith(f"{BATCH_TAG}:"):
        qr["reviewer_notes"] = f"{BATCH_TAG}: {notes} | prior: {prior}"
    else:
        qr["reviewer_notes"] = f"{BATCH_TAG}: {notes}"


def apply_priority(pr: dict, grade: int, notes: str) -> None:
    label = GRADE_LABELS[grade]
    pr["human_status"] = "changed"
    pr["human_grade"] = grade
    pr["human_notes"] = notes
    pr["proposed_grade"] = grade
    pr["proposed_label"] = label


def excluded(row: dict) -> bool:
    if row.get("exclude_from_qrels") is True:
        return True
    status = str(row.get("dedup_status") or "")
    return status in {"merged_into_canonical", "pending_merge"} and row.get("is_duplicate") is True


def build_dev_qrels(queue: list[dict]) -> tuple[list[dict], dict]:
    """DEV qrels: human_reviewed preferred; agent 'reviewed' for remaining 0/1 tail."""
    rows = []
    human_n = 0
    agent_n = 0
    for r in queue:
        if str(r.get("split") or "").lower() != "dev":
            continue
        if excluded(r):
            continue
        grade = r.get("relevance_grade")
        if grade is None:
            continue
        grade_i = int(grade)
        status = r.get("review_status")
        if status == "human_reviewed":
            source = "human_reviewed"
            human_n += 1
        elif status == "reviewed":
            # agent-first-pass low-grade tail (and any other agent-reviewed)
            source = "agent_first_pass"
            agent_n += 1
        else:
            continue
        rows.append(
            {
                "query_id": r["query_id"],
                "document_id": r.get("document_id") or r.get("ecli"),
                "grade": grade_i,
                "label": GRADE_LABELS[grade_i],
                "judgment_state": "explicit_grade_0" if grade_i == 0 else "graded",
                "review_reason": str(r.get("reviewer_notes") or ""),
                "annotation_source": source,
            }
        )
    stats = {
        "human_reviewed_count": human_n,
        "agent_only_count": agent_n,
        "qrels_count": len(rows),
        "query_count": len({r["query_id"] for r in rows}),
        "grade_histogram": dict(sorted(Counter(r["grade"] for r in rows).items())),
        "relevant_ge2_count": sum(1 for r in rows if r["grade"] >= 2),
    }
    return rows, stats


def integrity_checks(queue: list[dict], qrels: list[dict]) -> list[str]:
    issues: list[str] = []
    # no excluded aliases in qrels
    excl_ids = {
        (r.get("query_id"), r.get("ecli") or r.get("document_id"))
        for r in queue
        if excluded(r) and str(r.get("split") or "").lower() == "dev"
    }
    for row in qrels:
        key = (row["query_id"], row["document_id"])
        if key in excl_ids:
            issues.append(f"excluded alias present in qrels: {key}")
    # unique (query, doc)
    seen: set[tuple[str, str]] = set()
    for row in qrels:
        key = (row["query_id"], str(row["document_id"]))
        if key in seen:
            issues.append(f"duplicate qrel key: {key}")
        seen.add(key)
    # grades in range
    for row in qrels:
        if int(row["grade"]) not in (0, 1, 2, 3):
            issues.append(f"bad grade: {row}")
    # every human_reviewed DEV queue row with grade should appear
    for r in queue:
        if str(r.get("split") or "").lower() != "dev":
            continue
        if excluded(r):
            continue
        if r.get("review_status") != "human_reviewed":
            continue
        if r.get("relevance_grade") is None:
            issues.append(f"human_reviewed missing grade: {r.get('query_id')} {r.get('ecli')}")
            continue
        key = (r["query_id"], r.get("document_id") or r.get("ecli"))
        if key not in seen and (r["query_id"], r.get("ecli")) not in {
            (q["query_id"], q["document_id"]) for q in qrels
        }:
            # document_id may differ slightly; check ecli match in qrels
            if not any(
                q["query_id"] == r["query_id"]
                and (q["document_id"] == r.get("document_id") or q["document_id"] == r.get("ecli"))
                for q in qrels
            ):
                issues.append(f"human_reviewed missing from qrels: {r.get('query_id')} {r.get('ecli')}")
    return issues


def main() -> None:
    priority = load_jsonl(PRIORITY)
    queue = load_jsonl(QUEUE)
    applied = []

    for query_id, needle, grade, notes in OPS:
        qr = find_one(queue, query_id, needle)
        prior_grade = int(qr.get("relevance_grade") if qr.get("relevance_grade") is not None else 1)
        apply_queue(qr, grade, notes)
        pr_hits = find_hits(priority, query_id, needle)
        if len(pr_hits) == 1:
            apply_priority(pr_hits[0], grade, notes)
            priority_updated = True
        elif len(pr_hits) == 0:
            priority_updated = False
        else:
            raise RuntimeError(f"priority ambiguity {query_id} {needle}")
        applied.append(
            {
                "query_id": query_id,
                "ecli": qr.get("ecli"),
                "from_grade": prior_grade,
                "human_grade": grade,
                "human_status": "changed",
                "priority_updated": priority_updated,
                "notes": notes,
            }
        )

    if len(applied) != 4:
        raise RuntimeError(f"expected 4 applied, got {len(applied)}")

    write_jsonl(PRIORITY, priority)
    write_jsonl(QUEUE, queue)

    qrels, stats = build_dev_qrels(queue)
    # stable sort
    qrels.sort(key=lambda r: (r["query_id"], str(r["document_id"])))
    write_jsonl(QRELS_DEV, qrels)

    issues = integrity_checks(queue, qrels)
    if issues:
        raise RuntimeError("integrity failed:\n" + "\n".join(issues[:20]))

    now = datetime.now(timezone.utc).isoformat()
    freeze = {
        "status": "DEV_QRELS_FROZEN_WITH_AGENT_LOW_GRADE_TAIL",
        "frozen_at_utc": now,
        "scope": "DEV only",
        "test_touched": False,
        "full_qrels_frozen": False,
        "claim": (
            "DEV qrels frozen for first Dense/BM25/Hybrid A/B. "
            "Human-reviewed judgments preferred where available; "
            "remaining unreviewed grade 0/1 candidates keep agent-first-pass grades. "
            "NOT every DEV 0/1 judgment was human-reviewed. "
            "TEST qrels untouched."
        ),
        "qa_upgrades_applied": applied,
        "stats": stats,
        "qrels_path": str(QRELS_DEV.relative_to(ROOT)).replace("\\", "/"),
        "exclude_from_qrels_honored": True,
        "binary_relevance_threshold": 2,
    }
    FREEZE_META.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    FREEZE_NOTE.write_text(
        f"""# DEV Qrels Freeze

## Status: `DEV_QRELS_FROZEN_WITH_AGENT_LOW_GRADE_TAIL`

Frozen at: `{now}`

- **DEV relevance audit CLOSED** (g3 + g2 + QA spot-check upgrades)
- Human-reviewed grades used where available
- Remaining unreviewed DEV grade 0/1 keep **agent-first-pass** annotations
- Aliases with `exclude_from_qrels` excluded
- **TEST qrels NOT frozen / NOT touched**
- Do **not** claim every DEV 0/1 was human-reviewed

### Stats

| Metric | Count |
| --- | ---: |
| queries | {stats['query_count']} |
| qrels | {stats['qrels_count']} |
| human-reviewed | {stats['human_reviewed_count']} |
| agent-only | {stats['agent_only_count']} |
| grade ≥ 2 (binary relevant) | {stats['relevant_ge2_count']} |

### Grade histogram

| grade | count |
| --- | ---: |
| 0 | {stats['grade_histogram'].get(0, 0)} |
| 1 | {stats['grade_histogram'].get(1, 0)} |
| 2 | {stats['grade_histogram'].get(2, 0)} |
| 3 | {stats['grade_histogram'].get(3, 0)} |

### QA upgrades applied (4)

- `nalus-cs-v2-023` / II.US.2684.25 → 2
- `nalus-cs-v2-023` / II.US.403.24 → 2
- `nalus-cs-v2-008` / IV.US.436.03 → 2
- `nalus-cs-v2-009` / II.US.1510.21 → 2

Next: Dense vs BM25 vs Hybrid A/B on this DEV freeze (no tuning yet).
""",
        encoding="utf-8",
        newline="\n",
    )

    # Patch REVIEW_PROGRESS
    old = PROGRESS.read_text(encoding="utf-8")
    old = old.replace(
        "## Status: PROVISIONAL_AGENT_QRELS + HUMAN_AUDIT_G2_BATCH_1..6 applied; **DEV grade-2 CLOSED**",
        "## Status: **DEV_QRELS_FROZEN_WITH_AGENT_LOW_GRADE_TAIL** (DEV relevance CLOSED)",
    )
    insert = f"""- **DEV_QA_SPOTCHECK_UPGRADE applied** (4): 023/2684.25, 023/403.24, 008/436.03, 009/1510.21 → grade 2
- **DEV relevance audit CLOSED**
- **DEV qrels frozen**: `DEV_QRELS_FROZEN_WITH_AGENT_LOW_GRADE_TAIL`
  - human-reviewed where available; agent-first-pass for remaining 0/1 tail
  - exclude_from_qrels honored; TEST untouched
  - See DEV_QRELS_FREEZE.md / DEV_QRELS_FREEZE.json
- No more DEV manual relevance batches
- Next: Dense vs BM25 vs Hybrid A/B (no tune / no deploy yet)
"""
    if "DEV_QA_SPOTCHECK_UPGRADE" not in old:
        # insert after DEV grade-2 CLOSED line if present
        marker = "- **DEV grade-2 audit CLOSED**"
        if marker in old:
            old = old.replace(marker, marker + "\n" + insert, 1)
        else:
            old = insert + "\n" + old
    PROGRESS.write_text(old, encoding="utf-8", newline="\n")

    print(json.dumps({"applied": applied, "freeze": freeze, "integrity_ok": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
