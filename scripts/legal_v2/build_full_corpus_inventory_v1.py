#!/usr/bin/env python3
"""Phase-1 inventory for Legal v2 full-corpus A/B index builds.

Canonical source: batches/ + batches/manifest.json (Constitutional Court only).
Does not scrape NALUS. Does not write Qdrant.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.identity import is_valid_ecli, normalize_ecli  # noqa: E402

DEFAULT_BATCHES = (
    PROJECT_ROOT.parent / "nalus-scraper" / "batches"
)
DEFAULT_OUT = PROJECT_ROOT / "artifacts" / "legal_v2" / "full_corpus_build_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batches-dir", type=Path, default=DEFAULT_BATCHES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--min-eligible",
        type=int,
        default=20_000,
        help="Hard-stop if eligible unique docs fall below this without explanation.",
    )
    return p.parse_args(argv)


def _resolve_document_id(item: dict[str, Any]) -> tuple[str, str]:
    """Return (document_id, identity_class)."""
    ecli = str(item.get("ecli") or "").strip()
    if ecli and is_valid_ecli(ecli):
        return normalize_ecli(ecli), "valid_ecli"
    for key in ("canonical_document_id", "document_id"):
        value = str(item.get(key) or "").strip()
        if value and is_valid_ecli(value):
            return normalize_ecli(value), "valid_ecli"
    for key in ("source_document_id", "case_reference", "spisova_znacka", "result_id"):
        value = str(item.get(key) or "").strip()
        if value:
            return value, "fallback_non_ecli"
    return "", "missing"


def _year_from(item: dict[str, Any], document_id: str) -> int | None:
    for key in ("decision_date", "date", "publication_date"):
        text = str(item.get(key) or "")
        match = re.search(r"(19\d{2}|20\d{2})", text)
        if match:
            return int(match.group(1))
    match = re.search(r"ECLI:CZ:US:(\d{4})", document_id.upper())
    if match:
        return int(match.group(1))
    return None


def _manifest_files(batches_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    manifest_path = batches_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"HARD_STOP: missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files: list[Path] = []
    missing: list[str] = []
    for entry in manifest.get("batches") or []:
        name = str(entry.get("file") or "").strip()
        if not name:
            continue
        path = batches_dir / name
        if path.exists():
            files.append(path)
        else:
            missing.append(name)
    if missing:
        raise SystemExit(f"HARD_STOP: manifest files missing on disk: {missing[:10]}")
    return files, manifest


def build_inventory(*, batches_dir: Path, out_dir: Path, min_eligible: int) -> dict[str, Any]:
    files, manifest = _manifest_files(batches_dir)
    raw = 0
    valid_ecli_hits = 0
    missing_id = 0
    empty_text = 0
    duplicate_skips = 0
    fallback_kept = 0
    unreadable_files = 0
    seen: set[str] = set()
    dup_ids: Counter[str] = Counter()
    years: Counter[int] = Counter()
    doc_types: Counter[str] = Counter()
    skipped_reasons: Counter[str] = Counter()
    text_lens: list[int] = []
    eligible: list[dict[str, Any]] = []

    for file_path in files:
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            unreadable_files += 1
            skipped_reasons["unreadable_file"] += 1
            continue
        if not isinstance(payload, list):
            skipped_reasons["json_root_not_list"] += 1
            continue
        for item in payload:
            raw += 1
            if not isinstance(item, dict):
                skipped_reasons["record_not_object"] += 1
                continue
            text = str(item.get("full_text") or "").strip()
            document_id, identity_class = _resolve_document_id(item)
            if identity_class == "valid_ecli":
                valid_ecli_hits += 1
            elif identity_class == "fallback_non_ecli":
                fallback_kept += 1
            if not document_id:
                missing_id += 1
                skipped_reasons["missing_document_id"] += 1
                continue
            if not text:
                empty_text += 1
                skipped_reasons["empty_text"] += 1
                continue
            if document_id in seen:
                duplicate_skips += 1
                dup_ids[document_id] += 1
                skipped_reasons["duplicate_document_id"] += 1
                continue
            seen.add(document_id)
            year = _year_from(item, document_id)
            years[year or 0] += 1
            doc_type = str(
                item.get("document_type")
                or item.get("decision_form")
                or item.get("form_decision")
                or "unknown"
            )[:80]
            doc_types[doc_type] += 1
            text_lens.append(len(text))
            eligible.append(
                {
                    "document_id": document_id,
                    "identity_class": identity_class,
                    "year": year,
                    "text_len": len(text),
                    "document_type": doc_type,
                    "source_file": file_path.name,
                }
            )

    text_lens_sorted = sorted(text_lens)

    def _pct(p: float) -> int:
        if not text_lens_sorted:
            return 0
        idx = min(
            len(text_lens_sorted) - 1,
            int(round((p / 100.0) * (len(text_lens_sorted) - 1))),
        )
        return text_lens_sorted[idx]

    hard_stop = False
    hard_stop_reasons: list[str] = []
    if len(eligible) < min_eligible:
        hard_stop = True
        hard_stop_reasons.append(
            f"eligible_unique_documents={len(eligible)} < min_eligible={min_eligible}"
        )
    # Large duplicate rate alone is expected (raw>unique); only fail if almost all are dups
    if raw > 0 and len(eligible) / raw < 0.05:
        hard_stop = True
        hard_stop_reasons.append("eligible/raw ratio < 5%")

    report: dict[str, Any] = {
        "schema": "legal_v2_full_corpus_inventory_v1",
        "batches_dir": str(batches_dir.resolve()),
        "manifest_batch_count": len(manifest.get("batches") or []),
        "manifest_sum_doc_count": sum(
            int(b.get("doc_count") or 0) for b in (manifest.get("batches") or [])
        ),
        "existing_batch_files": len(files),
        "raw_discovered_records": raw,
        "valid_ecli_identity_hits": valid_ecli_hits,
        "invalid_unparseable_records": skipped_reasons.get("record_not_object", 0)
        + skipped_reasons.get("json_root_not_list", 0),
        "documents_missing_ecli_or_document_id": missing_id,
        "duplicate_document_ids_skipped": duplicate_skips,
        "unique_ids_with_duplicates": len(dup_ids),
        "duplicate_id_samples": [
            {"document_id": doc_id, "extra_occurrences": count}
            for doc_id, count in dup_ids.most_common(20)
        ],
        "documents_with_empty_text": empty_text,
        "fallback_non_ecli_ids_kept": fallback_kept,
        "unreadable_files": unreadable_files,
        "documents_skipped_and_reasons": dict(skipped_reasons),
        "eligible_unique_document_count": len(eligible),
        "eligible_valid_ecli_count": sum(
            1 for row in eligible if row["identity_class"] == "valid_ecli"
        ),
        "year_histogram": {str(k): v for k, v in sorted(years.items())},
        "document_type_top20": doc_types.most_common(20),
        "text_len_chars": {
            "mean": (sum(text_lens) / len(text_lens)) if text_lens else 0.0,
            "p50": _pct(50),
            "p95": _pct(95),
            "max": text_lens_sorted[-1] if text_lens_sorted else 0,
        },
        "hard_stop": hard_stop,
        "hard_stop_reasons": hard_stop_reasons,
        "notes": (
            "Eligible = first occurrence of stable document_id with non-empty full_text. "
            "Duplicate document_ids are skipped (kept first). NSoud excluded."
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inventory.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "eligible_document_ids.txt").write_text(
        "\n".join(row["document_id"] for row in eligible) + ("\n" if eligible else ""),
        encoding="utf-8",
    )
    with (out_dir / "eligible_document_meta.jsonl").open("w", encoding="utf-8") as handle:
        for row in eligible:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# Full corpus inventory v1",
        "",
        f"- Raw discovered records: **{raw}**",
        f"- Eligible unique documents: **{len(eligible)}**",
        f"- Duplicate document_id skips: **{duplicate_skips}**",
        f"- Missing document_id: **{missing_id}**",
        f"- Empty text: **{empty_text}**",
        f"- Fallback non-ECLI ids kept: **{fallback_kept}**",
        f"- Hard stop: **{hard_stop}**",
        "",
        "## Skip reasons",
        "",
    ]
    for key, value in sorted(skipped_reasons.items(), key=lambda item: -item[1]):
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## Year histogram (newest first)", ""]
    for year, count in sorted(years.items(), key=lambda item: -(item[0] or 0))[:20]:
        lines.append(f"- {year}: {count}")
    (out_dir / "inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_inventory(
        batches_dir=args.batches_dir,
        out_dir=args.out_dir,
        min_eligible=args.min_eligible,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["hard_stop"]:
        print("HARD_STOP: inventory identity/size gate failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
