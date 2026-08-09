#!/usr/bin/env python3
"""Build immutable chunking A/B inventory of exactly 300 ECLIs.

Mandatory set: all golden primary + accepted-alternative + hard-negative ECLIs
(with verified identity). Remaining slots filled by deterministic stratified
sampling from the pilot_600 inventory.

Does NOT embed or modify production collections.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOLDEN = PROJECT_ROOT / "benchmarks" / "legal_v2" / "case_similarity_golden_v1_pilot.jsonl"
DEFAULT_IDENTITY = (
    PROJECT_ROOT / "benchmarks" / "legal_v2" / "case_similarity_document_identity_v1.json"
)
DEFAULT_PILOT_INV = (
    PROJECT_ROOT
    / "artifacts"
    / "legal_v2"
    / "pilot_600_judgment_inventory"
    / "pilot_600_judgment_inventory.json"
)
DEFAULT_OUT = PROJECT_ROOT / "artifacts" / "legal_v2" / "chunking_ab_pilot_300_v1"
DEFAULT_BENCHMARKS_OUT = (
    PROJECT_ROOT / "benchmarks" / "legal_v2" / "chunking_ab_pilot_300_v1"
)
INVENTORY_ID = "chunking_ab_pilot_300_v1"
TARGET_COUNT = 300
SAMPLE_SEED = 20260809


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _identity_maps(path: Path) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    docs = payload.get("documents") or []
    doc_to_ecli: dict[str, str] = {}
    by_ecli: dict[str, dict[str, Any]] = {}
    for row in docs:
        ecli = str(row.get("ecli") or "").strip()
        doc_id = str(row.get("source_document_id") or "").strip()
        status = str(row.get("identity_status") or "").strip().lower()
        if not ecli or not doc_id:
            continue
        if status and status != "verified":
            continue
        doc_to_ecli[doc_id] = ecli
        by_ecli[ecli] = row
    return doc_to_ecli, by_ecli


def _collect_mandatory(
    golden_rows: list[dict[str, Any]],
    doc_to_ecli: dict[str, str],
) -> tuple[list[str], dict[str, list[str]]]:
    ordered: list[str] = []
    reasons: dict[str, list[str]] = defaultdict(list)

    def add(ecli: str | None, reason: str) -> None:
        if not ecli:
            return
        cleaned = str(ecli).strip()
        if not cleaned:
            return
        if cleaned not in reasons:
            ordered.append(cleaned)
        reasons[cleaned].append(reason)

    for row in golden_rows:
        qid = str(row.get("benchmark_id") or row.get("query_id") or "unknown")
        primary = row.get("expected_primary_ecli")
        add(primary, f"golden_primary:{qid}")
        for alt_doc in row.get("accepted_alternative_document_ids") or []:
            add(doc_to_ecli.get(str(alt_doc)), f"golden_accepted_alt:{qid}")
        # Some schemas may already carry ECLI lists.
        for alt_ecli in row.get("accepted_alternative_eclis") or []:
            add(alt_ecli, f"golden_accepted_alt_ecli:{qid}")
        for hn_doc in row.get("hard_negative_document_ids") or []:
            mapped = doc_to_ecli.get(str(hn_doc))
            if mapped:
                add(mapped, f"golden_hard_negative:{qid}")
            else:
                reasons[f"UNRESOLVED:{hn_doc}"].append(f"unresolved_hn:{qid}")
        for hn_ecli in row.get("hard_negative_eclis") or []:
            add(hn_ecli, f"golden_hard_negative_ecli:{qid}")
    return ordered, dict(reasons)


def _length_bucket(chunk_count: int) -> str:
    if chunk_count <= 10:
        return "short"
    if chunk_count <= 25:
        return "medium"
    if chunk_count <= 50:
        return "long"
    return "very_long"


def _stratified_fill(
    *,
    pool: list[dict[str, Any]],
    already: set[str],
    need: int,
    seed: int,
) -> list[dict[str, Any]]:
    if need <= 0:
        return []
    rng = random.Random(seed)
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in pool:
        ecli = str(doc.get("ecli") or "").strip()
        if not ecli or ecli in already:
            continue
        doc_type = str(doc.get("document_type") or "unknown")
        bucket = _length_bucket(int(doc.get("chunk_count") or 0))
        key = f"{doc_type}|{bucket}"
        by_stratum[key].append(doc)
    for key in by_stratum:
        by_stratum[key].sort(key=lambda d: str(d.get("ecli") or ""))
        rng.shuffle(by_stratum[key])

    selected: list[dict[str, Any]] = []
    strata = sorted(by_stratum.keys())
    while len(selected) < need and strata:
        progressed = False
        for key in list(strata):
            bucket = by_stratum[key]
            if not bucket:
                strata.remove(key)
                continue
            selected.append(bucket.pop())
            progressed = True
            if len(selected) >= need:
                break
        if not progressed:
            break
    if len(selected) < need:
        raise RuntimeError(
            f"insufficient stratified pool to fill inventory: have {len(selected)}, need {need}"
        )
    return selected


def build_inventory(
    *,
    golden_path: Path,
    identity_path: Path,
    pilot_inventory_path: Path,
    target_count: int,
    seed: int,
) -> dict[str, Any]:
    golden_rows = _load_jsonl(golden_path)
    doc_to_ecli, _ = _identity_maps(identity_path)
    pilot = json.loads(pilot_inventory_path.read_text(encoding="utf-8"))
    pilot_docs = list(pilot.get("documents") or [])
    by_ecli = {
        str(doc.get("ecli") or "").strip(): doc
        for doc in pilot_docs
        if str(doc.get("ecli") or "").strip()
    }

    mandatory_ordered, reason_map = _collect_mandatory(golden_rows, doc_to_ecli)
    unresolved = [key for key in reason_map if key.startswith("UNRESOLVED:")]
    if unresolved:
        raise RuntimeError(
            "cannot resolve hard-negative document ids to verified ECLI: "
            + ", ".join(unresolved[:20])
        )

    missing_in_pilot = [ecli for ecli in mandatory_ordered if ecli not in by_ecli]
    if missing_in_pilot:
        raise RuntimeError(
            "mandatory golden/HN ECLIs missing from pilot inventory: "
            + ", ".join(missing_in_pilot[:20])
        )

    selected_docs: list[dict[str, Any]] = []
    for ecli in mandatory_ordered:
        src = by_ecli[ecli]
        selected_docs.append(
            {
                "ecli": ecli,
                "canonical_document_id": src.get("canonical_document_id") or ecli,
                "case_number": src.get("case_number"),
                "court": src.get("court"),
                "decision_date": src.get("decision_date"),
                "document_type": src.get("document_type"),
                "chunk_count_in_pilot_a": src.get("chunk_count"),
                "length_bucket": _length_bucket(int(src.get("chunk_count") or 0)),
                "selection_reason": "mandatory_golden_or_hn",
                "selection_details": reason_map.get(ecli, []),
            }
        )

    already = {row["ecli"] for row in selected_docs}
    need = target_count - len(selected_docs)
    if need < 0:
        raise RuntimeError(
            f"mandatory set ({len(selected_docs)}) exceeds target_count={target_count}"
        )
    fill = _stratified_fill(pool=pilot_docs, already=already, need=need, seed=seed)
    for doc in fill:
        ecli = str(doc.get("ecli") or "").strip()
        selected_docs.append(
            {
                "ecli": ecli,
                "canonical_document_id": doc.get("canonical_document_id") or ecli,
                "case_number": doc.get("case_number"),
                "court": doc.get("court"),
                "decision_date": doc.get("decision_date"),
                "document_type": doc.get("document_type"),
                "chunk_count_in_pilot_a": doc.get("chunk_count"),
                "length_bucket": _length_bucket(int(doc.get("chunk_count") or 0)),
                "selection_reason": "stratified_fill",
                "selection_details": [
                    f"seed={seed}",
                    f"stratum={doc.get('document_type')}|{_length_bucket(int(doc.get('chunk_count') or 0))}",
                ],
            }
        )

    ordered_eclis = [row["ecli"] for row in selected_docs]
    if len(ordered_eclis) != target_count:
        raise RuntimeError(
            f"inventory size mismatch: got {len(ordered_eclis)}, expected {target_count}"
        )
    if len(set(ordered_eclis)) != len(ordered_eclis):
        raise RuntimeError("duplicate ECLIs in inventory")

    inventory_set = set(ordered_eclis)
    query_eval: list[dict[str, Any]] = []
    blocked_queries: list[str] = []
    for row in golden_rows:
        qid = str(row.get("benchmark_id") or row.get("query_id") or "unknown")
        primary = str(row.get("expected_primary_ecli") or "").strip()
        missing_primary = (not primary) or (primary not in inventory_set)
        missing_alts: list[str] = []
        for alt_doc in row.get("accepted_alternative_document_ids") or []:
            mapped = doc_to_ecli.get(str(alt_doc))
            if mapped and mapped not in inventory_set:
                missing_alts.append(mapped)
        for alt_ecli in row.get("accepted_alternative_eclis") or []:
            cleaned = str(alt_ecli).strip()
            if cleaned and cleaned not in inventory_set:
                missing_alts.append(cleaned)
        missing_hns: list[str] = []
        for hn_doc in row.get("hard_negative_document_ids") or []:
            mapped = doc_to_ecli.get(str(hn_doc))
            if mapped and mapped not in inventory_set:
                missing_hns.append(mapped)
        for hn_ecli in row.get("hard_negative_eclis") or []:
            cleaned = str(hn_ecli).strip()
            if cleaned and cleaned not in inventory_set:
                missing_hns.append(cleaned)
        evaluable = not missing_primary
        if not evaluable:
            blocked_queries.append(qid)
        query_eval.append(
            {
                "benchmark_id": qid,
                "expected_primary_ecli": primary or None,
                "evaluable": evaluable,
                "missing_accepted_alts": sorted(set(missing_alts)),
                "missing_hard_negatives": sorted(set(missing_hns)),
            }
        )
    if blocked_queries:
        raise RuntimeError(
            "golden queries not evaluable on inventory (missing primary ECLI): "
            + ", ".join(blocked_queries[:20])
        )

    inventory_hash = hashlib.sha256(
        "\n".join(ordered_eclis).encode("utf-8")
    ).hexdigest()

    return {
        "inventory_id": INVENTORY_ID,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_count": target_count,
        "document_count": len(ordered_eclis),
        "sample_seed": seed,
        "source_pilot_inventory": str(pilot_inventory_path.as_posix()),
        "source_golden": str(golden_path.as_posix()),
        "source_identity": str(identity_path.as_posix()),
        "mandatory_count": len(mandatory_ordered),
        "stratified_fill_count": need,
        "ordered_eclis": ordered_eclis,
        "inventory_hash_sha256": inventory_hash,
        "golden_query_evaluability": {
            "query_count": len(query_eval),
            "evaluable_count": sum(1 for q in query_eval if q["evaluable"]),
            "blocked_count": 0,
            "queries": query_eval,
        },
        "documents": selected_docs,
        "notes": [
            "Immutable A/B experiment inventory.",
            "A300/B300 metrics are paired only; do not compare absolute scores to A622.",
            "Do not modify after seeing retrieval metrics.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--pilot-inventory", type=Path, default=DEFAULT_PILOT_INV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--benchmarks-out-dir",
        type=Path,
        default=DEFAULT_BENCHMARKS_OUT,
        help="Versioned copy under benchmarks/ (committed).",
    )
    parser.add_argument("--target-count", type=int, default=TARGET_COUNT)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    args = parser.parse_args()

    payload = build_inventory(
        golden_path=args.golden,
        identity_path=args.identity,
        pilot_inventory_path=args.pilot_inventory,
        target_count=args.target_count,
        seed=args.seed,
    )

    md_lines = [
        f"# {INVENTORY_ID}",
        "",
        f"- document_count: {payload['document_count']}",
        f"- mandatory_count: {payload['mandatory_count']}",
        f"- stratified_fill_count: {payload['stratified_fill_count']}",
        f"- inventory_hash_sha256: `{payload['inventory_hash_sha256']}`",
        f"- sample_seed: {payload['sample_seed']}",
        (
            f"- golden_queries_evaluable: "
            f"{payload['golden_query_evaluability']['evaluable_count']}/"
            f"{payload['golden_query_evaluability']['query_count']}"
        ),
        "",
        "## Length buckets",
    ]
    buckets: dict[str, int] = defaultdict(int)
    types: dict[str, int] = defaultdict(int)
    for doc in payload["documents"]:
        buckets[str(doc["length_bucket"])] += 1
        types[str(doc.get("document_type") or "unknown")] += 1
    for key, value in sorted(buckets.items()):
        md_lines.append(f"- {key}: {value}")
    md_lines.append("")
    md_lines.append("## Document types")
    for key, value in sorted(types.items()):
        md_lines.append(f"- {key}: {value}")
    report_text = "\n".join(md_lines) + "\n"
    manifest_text = json.dumps(payload, ensure_ascii=False, indent=2)
    eclis_text = "\n".join(payload["ordered_eclis"]) + "\n"

    for out_dir in (args.out_dir, args.benchmarks_out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "inventory_manifest.json").write_text(manifest_text, encoding="utf-8")
        (out_dir / "ordered_eclis.txt").write_text(eclis_text, encoding="utf-8")
        (out_dir / "inventory_report.md").write_text(report_text, encoding="utf-8")
        print(f"WROTE {out_dir / 'inventory_manifest.json'}")

    print(
        f"DONE count={payload['document_count']} mandatory={payload['mandatory_count']} "
        f"hash={payload['inventory_hash_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
