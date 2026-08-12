#!/usr/bin/env python3
"""Full chunk-only A/B QA on parser v8 for chunking_ab_pilot_300_v1.

Frozen inputs:
  - inventory chunking_ab_pilot_300_v1 (hash locked)
  - PARSER_VERSION legal-decision-parser.cz-courts.v8
  - A = production hierarchical chunker
  - B = legal_contextual_packed_v1 (policy hash locked)

Does NOT embed, index, retrieve, or modify parser/chunkers.
Writes a new directory (default chunk_qa_v8/) and a delta vs pre-v8 summary.
"""

from __future__ import annotations

import argparse
import os
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.rag.legal_v2.audit import PARSER_VERSION
from app.rag.legal_v2.ingest.chunkers import chunk_document_for_experiment
from app.rag.legal_v2.ingest.chunkers.contextual_packed_v1 import ContextualPackedConfigV1
from app.rag.legal_v2.ingest.chunkers.names import (
    CHUNKER_A_CURRENT,
    CHUNKER_B_CONTEXTUAL_PACKED_V1,
)
from app.rag.legal_v2.ingest.chunking import HierarchicalChunkingResult
from app.rag.legal_v2.parser import parse_legal_document

from scripts.legal_v2.run_chunking_ab_pilot_300_chunk_qa import (
    BGE_INPUT_LIMIT,
    BGE_MODEL_ID,
    CE_MAX_LENGTH,
    CE_MODEL_ID,
    EXTRAPOLATION_DOCS,
    _audit_branch,
    _branch_summary,
    _fetch_full_text,
    _policy_hash,
    _prefix_overlap_chars,
    _select_review_eclis,
    _load_tokenizer,
    _stats,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = (
    PROJECT_ROOT / "benchmarks" / "legal_v2" / "chunking_ab_pilot_300_v1" / "inventory_manifest.json"
)
DEFAULT_ARTIFACTS_INVENTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "legal_v2"
    / "chunking_ab_pilot_300_v1"
    / "inventory_manifest.json"
)
DEFAULT_BASELINE = (
    PROJECT_ROOT
    / "artifacts"
    / "legal_v2"
    / "chunking_ab_pilot_300_v1"
    / "chunk_qa"
    / "chunk_qa_summary.json"
)
DEFAULT_OUT = (
    PROJECT_ROOT / "artifacts" / "legal_v2" / "chunking_ab_pilot_300_v1" / "chunk_qa_v8"
)
DEFAULT_API = "http://127.0.0.1:8029"
EXPECTED_PARSER = "legal-decision-parser.cz-courts.v8"
EXPECTED_INVENTORY_HASH = "89233b9fe9b06eda8dea00abd99a48aa54940e616aa88c00860ced4ae49c011b"
EXPECTED_B_POLICY_HASH = "8fa196c58a9c537d311af6849582481ac195324c4f358634e81fcecb8f3f5898"
EXPECTED_A = "legal_v2_hierarchical_chunker_v1"
EXPECTED_B = "legal_contextual_packed_v1"


def _git_head(explicit: str | None = None) -> str:
    """Resolve git HEAD. Docker images often omit .git — allow CLI/env override."""
    if explicit and explicit.strip() and explicit.strip().lower() != "unknown":
        return explicit.strip()
    for key in ("LEGAL_V2_GIT_COMMIT", "GIT_COMMIT", "SOURCE_COMMIT"):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
            ).strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def _classify_v8(
    summary_a: dict[str, Any],
    summary_b: dict[str, Any],
    gates: dict[str, Any],
    section_class: str,
) -> str:
    if gates.get("blocked"):
        return "CHUNK_EXPERIMENT_BLOCKED"
    if section_class == "SECTION_BOUNDARIES_BLOCKING":
        return "SECTION_BOUNDARIES_BLOCKING"
    if gates.get("structural_regression"):
        return "CHUNK_B_STRUCTURAL_REGRESSION_V8"
    a_total = max(int(summary_a["total_child_chunks"]), 1)
    b_total = int(summary_b["total_child_chunks"])
    if b_total / a_total >= 1.5:
        return "CHUNK_B_EXCESSIVE_EXPANSION_V8"
    b_trunc = float(summary_b["ce_tokens_per_chunk"]["fraction_would_truncate_under_ce_max_length"])
    a_trunc = float(summary_a["ce_tokens_per_chunk"]["fraction_would_truncate_under_ce_max_length"])
    if b_trunc >= 0.10 and b_trunc >= (a_trunc + 0.05):
        return "CHUNK_B_TRUNCATION_RISK_V8"
    return "CHUNK_QA_PASS_V8"


def _section_boundary_class(
    *,
    header_share: float,
    header_suspicion_docs: int,
    docs: int,
    section_violations_b: int,
    deep_header_flags: int,
) -> str:
    if section_violations_b > 0 or header_share >= 0.20 or header_suspicion_docs / max(docs, 1) >= 0.15:
        return "SECTION_BOUNDARIES_BLOCKING"
    if deep_header_flags > 0 or header_share >= 0.08 or header_suspicion_docs > 5:
        return "SECTION_BOUNDARIES_NOISY_BUT_USABLE"
    return "SECTION_BOUNDARIES_CLEAN"


def _tiny_counts(chunks: list[dict[str, Any]], key: str = "token_count_bge") -> dict[str, Any]:
    vals = [int(c.get(key) or 0) for c in chunks]
    by_section: dict[str, Counter[str]] = defaultdict(Counter)
    for c in chunks:
        sec = str(c.get("section_type") or "unknown")
        t = int(c.get(key) or 0)
        if t <= 5:
            by_section[sec]["le5"] += 1
        if t <= 10:
            by_section[sec]["le10"] += 1
        if t <= 20:
            by_section[sec]["le20"] += 1
    return {
        "le5": sum(1 for v in vals if v <= 5),
        "le10": sum(1 for v in vals if v <= 10),
        "le20": sum(1 for v in vals if v <= 20),
        "by_section": {k: dict(v) for k, v in sorted(by_section.items())},
    }


def _delta_side(pre: dict[str, Any], cur: dict[str, Any]) -> dict[str, Any]:
    def g(block: dict[str, Any], *path: str, default: float = 0.0) -> float:
        node: Any = block
        for p in path:
            if not isinstance(node, dict) or p not in node:
                return default
            node = node[p]
        try:
            return float(node)
        except (TypeError, ValueError):
            return default

    return {
        "total_chunks_delta": int(cur.get("total_child_chunks") or 0)
        - int(pre.get("total_child_chunks") or 0),
        "mean_chunks_per_doc_delta": g(cur, "chunks_per_document", "mean")
        - g(pre, "chunks_per_document", "mean"),
        "median_chunks_per_doc_delta": g(cur, "chunks_per_document", "median")
        - g(pre, "chunks_per_document", "median"),
        "median_native_token_delta": g(cur, "native_tokens_per_chunk", "median")
        - g(pre, "native_tokens_per_chunk", "median"),
        "p95_native_token_delta": g(cur, "native_tokens_per_chunk", "p95")
        - g(pre, "native_tokens_per_chunk", "p95"),
        "median_bge_token_delta": g(cur, "bge_tokens_per_chunk", "median")
        - g(pre, "bge_tokens_per_chunk", "median"),
        "p95_bge_token_delta": g(cur, "bge_tokens_per_chunk", "p95")
        - g(pre, "bge_tokens_per_chunk", "p95"),
        "overlap_ratio_delta": g(cur, "overlap_ratio_mean") - g(pre, "overlap_ratio_mean"),
        "duplicate_text_ratio_delta": g(cur, "duplicate_text_ratio_mean")
        - g(pre, "duplicate_text_ratio_mean"),
        "ce_truncation_risk_delta": g(
            cur, "ce_tokens_per_chunk", "fraction_would_truncate_under_ce_max_length"
        )
        - g(pre, "ce_tokens_per_chunk", "fraction_would_truncate_under_ce_max_length"),
        "projected_75k_delta": int(cur.get("projected_chunks_75k") or 0)
        - int(pre.get("projected_chunks_75k") or 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=None)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--expected-count", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-model-tokenizers", action="store_true")
    parser.add_argument(
        "--git-commit",
        default="",
        help="Override git HEAD when .git is unavailable (e.g. Docker).",
    )
    args = parser.parse_args()

    if PARSER_VERSION != EXPECTED_PARSER:
        raise SystemExit(f"ABORT: PARSER_VERSION={PARSER_VERSION!r} != {EXPECTED_PARSER!r}")

    inv_path = args.inventory
    if inv_path is None:
        inv_path = DEFAULT_INVENTORY if DEFAULT_INVENTORY.exists() else DEFAULT_ARTIFACTS_INVENTORY
    inventory = json.loads(inv_path.read_text(encoding="utf-8"))
    ordered = list(inventory.get("ordered_eclis") or [])
    meta_by_ecli = {
        str(d.get("ecli")): d for d in (inventory.get("documents") or []) if d.get("ecli")
    }
    if inventory.get("inventory_id") != "chunking_ab_pilot_300_v1":
        raise SystemExit("ABORT: unexpected inventory_id")
    if inventory.get("inventory_hash_sha256") != EXPECTED_INVENTORY_HASH:
        raise SystemExit(
            f"ABORT: inventory hash mismatch: {inventory.get('inventory_hash_sha256')}"
        )
    if len(ordered) != args.expected_count:
        raise SystemExit(f"ABORT: count {len(ordered)} != {args.expected_count}")
    if len(set(ordered)) != len(ordered):
        raise SystemExit("ABORT: duplicate ECLIs")
    if args.limit and args.limit > 0:
        ordered = ordered[: args.limit]

    if CHUNKER_A_CURRENT != EXPECTED_A:
        raise SystemExit(f"ABORT: A version {CHUNKER_A_CURRENT} != {EXPECTED_A}")
    if CHUNKER_B_CONTEXTUAL_PACKED_V1 != EXPECTED_B:
        raise SystemExit(f"ABORT: B version {CHUNKER_B_CONTEXTUAL_PACKED_V1} != {EXPECTED_B}")

    b_config = ContextualPackedConfigV1()
    b_hash = _policy_hash(b_config)
    if b_hash != EXPECTED_B_POLICY_HASH:
        raise SystemExit(f"ABORT: B policy hash {b_hash} != {EXPECTED_B_POLICY_HASH}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "PRE_V8_BASELINE.txt").write_text(
        f"Historical pre-v8 QA: {args.baseline_summary.as_posix()}\n"
        "Do not overwrite. This run is parser-v8 only.\n",
        encoding="utf-8",
    )

    if args.skip_model_tokenizers:
        bge_count = lambda t: len(__import__("re").findall(r"\w+", t))  # noqa: E731
        ce_count = bge_count
        tokenizer_mode = "native_proxy_ONLY_FOR_DEBUG"
        bge_tok_id = "native_proxy"
        ce_tok_id = "native_proxy"
    else:
        print("Loading BGE-M3 tokenizer...", flush=True)
        bge_count = _load_tokenizer(BGE_MODEL_ID)
        print("Loading CE tokenizer...", flush=True)
        ce_count = _load_tokenizer(CE_MODEL_ID)
        tokenizer_mode = "actual_model_tokenizers"
        bge_tok_id = BGE_MODEL_ID
        ce_tok_id = CE_MODEL_ID

    audits_a: list[dict[str, Any]] = []
    audits_b: list[dict[str, Any]] = []
    per_doc: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    a_eclis: set[str] = set()
    b_eclis: set[str] = set()
    parsed_section_counts: Counter[str] = Counter()
    a_chunk_sections: Counter[str] = Counter()
    b_chunk_sections: Counter[str] = Counter()
    docs_with_header = 0
    header_suspicion_docs = 0
    deep_header_examples: list[dict[str, Any]] = []
    suspicious: list[dict[str, Any]] = []
    all_chunks_a: list[dict[str, Any]] = []
    all_chunks_b: list[dict[str, Any]] = []

    for index, ecli in enumerate(ordered, start=1):
        print(f"[{index}/{len(ordered)}] {ecli}", flush=True)
        try:
            text = _fetch_full_text(args.api, ecli)
            meta = meta_by_ecli.get(ecli) or {}
            court = str(meta.get("court") or "")
            parsed = parse_legal_document(
                document_id=ecli,
                text=text,
                metadata={"court": court, "ecli": ecli},
            )
            section_seq = [p.section_type.value for p in parsed.paragraphs]
            for sec in section_seq:
                parsed_section_counts[sec] += 1
            if "header" in section_seq:
                docs_with_header += 1

            # Header suspicion: header after first 20% of paragraphs with numbered body cues.
            header_suspect = False
            n_paras = max(len(parsed.paragraphs), 1)
            for p in parsed.paragraphs:
                if p.section_type.value != "header":
                    continue
                if p.paragraph_index >= max(3, int(0.2 * n_paras)) and len(p.normalized_text) >= 80:
                    header_suspect = True
                    if len(deep_header_examples) < 40:
                        deep_header_examples.append(
                            {
                                "ecli": ecli,
                                "paragraph_index": p.paragraph_index,
                                "token_estimate": len(p.normalized_text.split()),
                                "text_preview": p.normalized_text[:280],
                            }
                        )
            if header_suspect:
                header_suspicion_docs += 1

            for p in parsed.paragraphs:
                sec = p.section_type.value
                preview = p.normalized_text[:220]
                if sec == "participants" and any(
                    x in preview.lower()
                    for x in ("odmítl", "dospěl k závěru", "ústavní stížnost se odmítá")
                ):
                    suspicious.append(
                        {
                            "ecli": ecli,
                            "kind": "participants_looks_like_conclusion",
                            "paragraph_index": p.paragraph_index,
                            "text_preview": preview,
                        }
                    )
                if sec == "procedural_history" and any(
                    x in preview.lower()
                    for x in ("vzhledem ke shora", "jako návrh zjevně", "právní posouzení")
                ):
                    suspicious.append(
                        {
                            "ecli": ecli,
                            "kind": "procedural_history_looks_like_reasoning",
                            "paragraph_index": p.paragraph_index,
                            "text_preview": preview,
                        }
                    )

            result_a = chunk_document_for_experiment(parsed, chunker_version=CHUNKER_A_CURRENT)
            result_b = chunk_document_for_experiment(
                parsed, chunker_version=CHUNKER_B_CONTEXTUAL_PACKED_V1
            )
            assert result_a.__class__ is HierarchicalChunkingResult

            branch_a = _audit_branch(
                document=parsed,
                result=result_a,
                chunker_version=CHUNKER_A_CURRENT,
                bge_count=bge_count,
                ce_count=ce_count,
                is_b=False,
                b_config=b_config,
            )
            branch_b = _audit_branch(
                document=parsed,
                result=result_b,
                chunker_version=CHUNKER_B_CONTEXTUAL_PACKED_V1,
                bge_count=bge_count,
                ce_count=ce_count,
                is_b=True,
                b_config=b_config,
            )
            for c in branch_a["chunks"]:
                a_chunk_sections[c["section_type"]] += 1
                all_chunks_a.append(c)
            for c in branch_b["chunks"]:
                b_chunk_sections[c["section_type"]] += 1
                all_chunks_b.append(c)

            a_eclis.add(ecli)
            b_eclis.add(ecli)
            audits_a.append(branch_a)
            audits_b.append(branch_b)
            per_doc.append(
                {
                    "ecli": ecli,
                    "court": court or None,
                    "source_char_count": len(text),
                    "section_sequence": section_seq,
                    "A": {k: v for k, v in branch_a.items() if k != "chunks"},
                    "B": {k: v for k, v in branch_b.items() if k != "chunks"},
                    "_chunks_a": branch_a["chunks"],
                    "_chunks_b": branch_b["chunks"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"ecli": ecli, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  FAIL {exc}", flush=True)

    if failures:
        (out_dir / "chunk_qa_failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise SystemExit(f"ABORT: {len(failures)} failures")

    expected = set(ordered)
    gates = {"blocked": False, "structural_regression": False, "reasons": []}
    if a_eclis != expected or b_eclis != expected:
        gates["blocked"] = True
        gates["reasons"].append("ecli_set_mismatch")

    empty_a = sum(len(a["empty_chunk_ids"]) for a in audits_a)
    empty_b = sum(len(a["empty_chunk_ids"]) for a in audits_b)
    dup_a = sum(len(a["duplicate_chunk_ids"]) for a in audits_a)
    dup_b = sum(len(a["duplicate_chunk_ids"]) for a in audits_b)
    miss_a = sum(len(a["missing_paragraph_ids"]) for a in audits_a)
    miss_b = sum(len(a["missing_paragraph_ids"]) for a in audits_b)
    lost_a = sum(len(a["lost_paragraph_samples"]) for a in audits_a)
    lost_b = sum(len(a["lost_paragraph_samples"]) for a in audits_b)
    confirmed_loss_a = sum(len(a.get("confirmed_text_loss_ids") or []) for a in audits_a)
    confirmed_loss_b = sum(len(a.get("confirmed_text_loss_ids") or []) for a in audits_b)
    sec_b = sum(len(a["section_violations"]) for a in audits_b)
    ov_b = sum(len(a["b_overlap_policy_violations"]) for a in audits_b)
    order_b = sum(int(a["order_breaks"]) for a in audits_b)
    contam = sum(len(a["contamination_chunk_ids"]) for a in audits_a + audits_b)
    malformed = sum(1 for e in ordered if not str(e).startswith("ECLI:"))

    integrity = {
        "empty_chunks_a": empty_a,
        "empty_chunks_b": empty_b,
        "duplicate_chunk_ids_a": dup_a,
        "duplicate_chunk_ids_b": dup_b,
        "missing_paragraph_ids_a": miss_a,
        "missing_paragraph_ids_b": miss_b,
        "lost_paragraph_heuristic_a": lost_a,
        "lost_paragraph_heuristic_b": lost_b,
        "confirmed_text_loss_a": confirmed_loss_a,
        "confirmed_text_loss_b": confirmed_loss_b,
        "section_violations_b": sec_b,
        "b_overlap_policy_violations": ov_b,
        "order_breaks_b": order_b,
        "cross_document_contamination": contam,
        "malformed_ecli_count": malformed,
    }
    if any(
        integrity[k] > 0
        for k in (
            "empty_chunks_a",
            "empty_chunks_b",
            "duplicate_chunk_ids_a",
            "duplicate_chunk_ids_b",
            "missing_paragraph_ids_a",
            "missing_paragraph_ids_b",
            "cross_document_contamination",
            "malformed_ecli_count",
            "confirmed_text_loss_a",
            "confirmed_text_loss_b",
            "section_violations_b",
            "b_overlap_policy_violations",
            "order_breaks_b",
        )
    ):
        if (
            integrity["section_violations_b"]
            or integrity["b_overlap_policy_violations"]
            or integrity["order_breaks_b"]
        ):
            gates["structural_regression"] = True
            gates["reasons"].append("b_structural_issues")
        if any(
            integrity[k] > 0
            for k in (
                "empty_chunks_a",
                "empty_chunks_b",
                "duplicate_chunk_ids_a",
                "duplicate_chunk_ids_b",
                "missing_paragraph_ids_a",
                "missing_paragraph_ids_b",
                "cross_document_contamination",
                "malformed_ecli_count",
                "confirmed_text_loss_a",
                "confirmed_text_loss_b",
            )
        ):
            gates["blocked"] = True
            gates["reasons"].append("integrity_gate_failed")
        # lost_paragraph_heuristic is a substring probe and can false-positive on
        # intentionally split oversized paragraphs; confirmed_text_loss_* is authoritative.

    summary_a = _branch_summary(audits_a)
    summary_b = _branch_summary(audits_b)
    total_paras = sum(parsed_section_counts.values()) or 1
    header_share = parsed_section_counts.get("header", 0) / total_paras
    court_reasoning_share = parsed_section_counts.get("court_reasoning", 0) / total_paras
    section_class = _section_boundary_class(
        header_share=header_share,
        header_suspicion_docs=header_suspicion_docs,
        docs=len(ordered),
        section_violations_b=sec_b,
        deep_header_flags=len(deep_header_examples),
    )
    classification = _classify_v8(summary_a, summary_b, gates, section_class)
    multiplier = summary_b["total_child_chunks"] / max(summary_a["total_child_chunks"], 1)

    # Reviews
    review_eclis = _select_review_eclis(inventory, per_doc, target=28)
    review_docs = []
    for ecli in review_eclis:
        row = next(r for r in per_doc if r["ecli"] == ecli)

        def pick(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if len(chunks) <= 6:
                return [{k: v for k, v in c.items() if k != "text"} for c in chunks]
            idxs = sorted(set([0, 1, 2, len(chunks) // 2, len(chunks) - 2, len(chunks) - 1]))
            return [{k: v for k, v in chunks[i].items() if k != "text"} for i in idxs]

        chunks_b = row["_chunks_b"]
        overlap_markers = []
        for i in range(1, min(len(chunks_b), 12)):
            ov = _prefix_overlap_chars(chunks_b[i - 1]["text"], chunks_b[i]["text"])
            if ov > 0:
                overlap_markers.append(
                    {
                        "between": [chunks_b[i - 1]["chunk_id"], chunks_b[i]["chunk_id"]],
                        "overlap_chars": ov,
                    }
                )
        review_docs.append(
            {
                "ecli": ecli,
                "parser_version": PARSER_VERSION,
                "label": "PARSER_V8_UNREVIEWED",
                "document_length_chars": row["source_char_count"],
                "length_bucket": (meta_by_ecli.get(ecli) or {}).get("length_bucket"),
                "selection_reason": (meta_by_ecli.get(ecli) or {}).get("selection_reason"),
                "section_sequence": row["section_sequence"],
                "A": {
                    "child_chunk_count": row["A"]["child_chunk_count"],
                    "chunks": pick(row["_chunks_a"]),
                },
                "B": {
                    "child_chunk_count": row["B"]["child_chunk_count"],
                    "chunks": pick(chunks_b),
                    "overlap_markers": overlap_markers,
                },
            }
        )

    # Delta vs pre-v8
    delta: dict[str, Any] = {"baseline_available": False}
    changed_boundary_examples: list[dict[str, Any]] = []
    if args.baseline_summary.exists():
        pre = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
        pre_docs = {d["ecli"]: d for d in pre.get("documents") or []}
        cur_docs = {d["ecli"]: d for d in per_doc}
        common = sorted(set(pre_docs) & set(cur_docs))
        a_count_changed = 0
        b_count_changed = 0
        for ecli in common:
            pa = int(pre_docs[ecli].get("A", {}).get("child_chunk_count") or 0)
            pb = int(pre_docs[ecli].get("B", {}).get("child_chunk_count") or 0)
            ca = int(cur_docs[ecli]["A"]["child_chunk_count"])
            cb = int(cur_docs[ecli]["B"]["child_chunk_count"])
            if pa != ca:
                a_count_changed += 1
            if pb != cb:
                b_count_changed += 1
            if (pa != ca or pb != cb) and len(changed_boundary_examples) < 30:
                # Prefer docs that previously had header-heavy issues in v8 deep headers or mandatory.
                changed_boundary_examples.append(
                    {
                        "ecli": ecli,
                        "label": "UNREVIEWED",
                        "pre_v8_A_chunks": pa,
                        "v8_A_chunks": ca,
                        "pre_v8_B_chunks": pb,
                        "v8_B_chunks": cb,
                        "v8_section_sequence": cur_docs[ecli]["section_sequence"][:40],
                        "pre_v8_section_sequence": "unavailable_in_pre_v8_summary",
                        "A_preview_v8": [
                            {
                                "chunk_id": c["chunk_id"],
                                "section_type": c["section_type"],
                                "token_count_native": c["token_count_native"],
                                "text_preview": c["text_preview"],
                            }
                            for c in cur_docs[ecli]["_chunks_a"][:4]
                        ],
                        "B_preview_v8": [
                            {
                                "chunk_id": c["chunk_id"],
                                "section_type": c["section_type"],
                                "token_count_native": c["token_count_native"],
                                "text_preview": c["text_preview"],
                            }
                            for c in cur_docs[ecli]["_chunks_b"][:4]
                        ],
                    }
                )
        delta = {
            "baseline_available": True,
            "baseline_path": str(args.baseline_summary.as_posix()),
            "baseline_generated_at": pre.get("generated_at"),
            "note": (
                "pre-v8 summary lacks full SectionType sequences; "
                "sequence-change count unavailable for full 300."
            ),
            "A": _delta_side(pre.get("A") or {}, summary_a),
            "B": _delta_side(pre.get("B") or {}, summary_b),
            "docs_with_A_chunk_count_change": a_count_changed,
            "docs_with_B_chunk_count_change": b_count_changed,
            "docs_with_either_chunk_count_change": len(
                {
                    e
                    for e in common
                    if int(pre_docs[e].get("A", {}).get("child_chunk_count") or 0)
                    != int(cur_docs[e]["A"]["child_chunk_count"])
                    or int(pre_docs[e].get("B", {}).get("child_chunk_count") or 0)
                    != int(cur_docs[e]["B"]["child_chunk_count"])
                }
            ),
            "docs_with_section_sequence_change": None,
            "identical_child_chunk_pct": None,
            "identical_child_chunk_pct_note": (
                "Not computed: pre-v8 artifacts did not persist full chunk text/IDs for 300."
            ),
        }

    slice4_yes = (
        classification == "CHUNK_QA_PASS_V8"
        and confirmed_loss_a == 0
        and confirmed_loss_b == 0
    )
    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_head(args.git_commit),
        "parser_version": PARSER_VERSION,
        "parser_commit": "f9c8372",
        "inventory_id": inventory["inventory_id"],
        "inventory_hash_sha256": inventory["inventory_hash_sha256"],
        "document_count": len(ordered),
        "tokenizer_mode": tokenizer_mode,
        "bge_tokenizer_id": bge_tok_id,
        "ce_tokenizer_id": ce_tok_id,
        "ce_max_length": CE_MAX_LENGTH,
        "bge_input_limit": BGE_INPUT_LIMIT,
        "a_implementation": "app.rag.legal_v2.ingest.chunking.build_hierarchical_chunks",
        "a_chunker_version": CHUNKER_A_CURRENT,
        "b_implementation": (
            "app.rag.legal_v2.ingest.chunkers.contextual_packed_v1."
            "build_contextual_packed_chunks_v1"
        ),
        "b_chunker_version": CHUNKER_B_CONTEXTUAL_PACKED_V1,
        "b_policy": {
            "soft_min_tokens": b_config.soft_min_tokens,
            "soft_target_tokens": b_config.soft_target_tokens,
            "hard_max_tokens": b_config.hard_max_tokens,
            "overlap_max_tokens": b_config.overlap_max_tokens,
            "overlap_rule": "whole_paragraph_or_none",
            "policy_hash_sha256": b_hash,
        },
        "integrity_gates": integrity,
        "gate_evaluation": gates,
        "classification": classification,
        "section_boundary_classification": section_class,
        "parser_section_distribution": {
            "counts": dict(parsed_section_counts),
            "share": {k: v / total_paras for k, v in parsed_section_counts.items()},
            "docs_with_header": docs_with_header,
            "header_suspicion_docs": header_suspicion_docs,
            "header_share": header_share,
            "court_reasoning_share": court_reasoning_share,
        },
        "chunk_section_distribution": {
            "A": dict(a_chunk_sections),
            "B": dict(b_chunk_sections),
        },
        "tiny_structural_chunks": {
            "A_bge": _tiny_counts(all_chunks_a, "token_count_bge"),
            "B_bge": _tiny_counts(all_chunks_b, "token_count_bge"),
            "note": "Diagnostic only; heading attachment not fixed in this task.",
        },
        "A": summary_a,
        "B": summary_b,
        "relative_infrastructure_multiplier_b_over_a": {
            "embedding_count": multiplier,
            "bm25_row_count": multiplier,
            "qdrant_vector_count": multiplier,
            "label": "EXPERIMENTAL SAMPLE EXTRAPOLATION ONLY",
        },
        "projected_75k": {
            "label": "EXPERIMENTAL SAMPLE EXTRAPOLATION ONLY",
            "A_child_chunks": summary_a["projected_chunks_75k"],
            "B_child_chunks": summary_b["projected_chunks_75k"],
        },
        "SLICE_4_SAFE_TO_START": "YES" if slice4_yes else "NO",
        "no_embeddings_generated": True,
        "no_experimental_indexes_built": True,
        "no_retrieval_benchmark": True,
        "no_ce_scoring": True,
        "full_75k_corpus_untouched": True,
        "parser_unchanged_this_task": True,
        "chunkers_unchanged_this_task": True,
        "documents": [
            {
                "ecli": row["ecli"],
                "source_char_count": row["source_char_count"],
                "section_sequence": row["section_sequence"],
                "A": {
                    "child_chunk_count": row["A"]["child_chunk_count"],
                    "overlap_ratio": row["A"]["overlap_ratio"],
                    "duplicate_text_ratio": row["A"]["duplicate_text_ratio"],
                },
                "B": {
                    "child_chunk_count": row["B"]["child_chunk_count"],
                    "overlap_ratio": row["B"]["overlap_ratio"],
                    "duplicate_text_ratio": row["B"]["duplicate_text_ratio"],
                    "section_violations": row["B"]["section_violations"],
                },
            }
            for row in per_doc
        ],
    }

    (out_dir / "chunk_qa_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # aliases requested by prompt
    (out_dir / "chunk_qa_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "b_policy_freeze.json").write_text(
        json.dumps(summary["b_policy"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "boundary_review_v8_unreviewed.json").write_text(
        json.dumps(review_docs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "pre_v8_vs_v8_delta.json").write_text(
        json.dumps(delta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "changed_boundary_examples_unreviewed.json").write_text(
        json.dumps(changed_boundary_examples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "suspicious_section_examples.json").write_text(
        json.dumps(
            {
                "deep_header_examples": deep_header_examples,
                "other_suspicious": suspicious[:200],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    md = [
        "# Chunking A/B chunk-only QA (PARSER V8)",
        "",
        f"- classification: `{classification}`",
        f"- SLICE_4_SAFE_TO_START: `{summary['SLICE_4_SAFE_TO_START']}`",
        f"- section_boundary_classification: `{section_class}`",
        f"- parser_version: `{PARSER_VERSION}`",
        f"- git_commit: `{summary['git_commit']}`",
        f"- parser_commit: `{summary['parser_commit']}`",
        f"- b_policy_hash: `{b_hash}`",
        f"- confirmed_text_loss_a/b: `{confirmed_loss_a}/{confirmed_loss_b}`",
        f"- lost_paragraph_heuristic_a/b: `{lost_a}/{lost_b}`",
        f"- inventory_hash: `{inventory['inventory_hash_sha256']}`",
        f"- documents: {len(ordered)}",
        f"- A total chunks: {summary_a['total_child_chunks']}",
        f"- B total chunks: {summary_b['total_child_chunks']}",
        f"- relative multiplier B/A: {multiplier:.4f}",
        f"- header share: {header_share:.4f}",
        f"- header-suspicion docs: {header_suspicion_docs}/{len(ordered)}",
        f"- court_reasoning share: {court_reasoning_share:.4f}",
        f"- A CE trunc fraction: "
        f"{summary_a['ce_tokens_per_chunk']['fraction_would_truncate_under_ce_max_length']:.4f}",
        f"- B CE trunc fraction: "
        f"{summary_b['ce_tokens_per_chunk']['fraction_would_truncate_under_ce_max_length']:.4f}",
        "",
        "## Integrity",
        "",
        "```json",
        json.dumps(integrity, indent=2),
        "```",
        "",
        "## Pre-v8 delta (A)",
        "",
        "```json",
        json.dumps(delta.get("A"), indent=2),
        "```",
        "",
        "## Pre-v8 delta (B)",
        "",
        "```json",
        json.dumps(delta.get("B"), indent=2),
        "```",
        "",
        "STOP: Slice 4 not started. No embeddings / Qdrant / BM25 / retrieval / CE scoring.",
        "",
        "EXPERIMENTAL SAMPLE EXTRAPOLATION ONLY for 75k projections.",
    ]
    (out_dir / "chunk_qa_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    delta_md = [
        "# Pre-v8 vs parser-v8 chunk QA delta",
        "",
        "Purpose: quantify parser v8 effect only. Not used to pick A vs B.",
        "",
        f"- baseline: `{args.baseline_summary.as_posix()}`",
        f"- docs A chunk-count changed: {delta.get('docs_with_A_chunk_count_change')}",
        f"- docs B chunk-count changed: {delta.get('docs_with_B_chunk_count_change')}",
        f"- docs either changed: {delta.get('docs_with_either_chunk_count_change')}",
        f"- section sequence changes (full 300): {delta.get('docs_with_section_sequence_change')}",
        "",
        "## A",
        "",
        "```json",
        json.dumps(delta.get("A"), indent=2),
        "```",
        "",
        "## B",
        "",
        "```json",
        json.dumps(delta.get("B"), indent=2),
        "```",
    ]
    (out_dir / "pre_v8_vs_v8_delta.md").write_text("\n".join(delta_md) + "\n", encoding="utf-8")

    review_md = ["# Boundary review PARSER V8 (UNREVIEWED)", ""]
    for doc in review_docs:
        review_md.extend(
            [
                f"## {doc['ecli']}",
                f"- chars: {doc['document_length_chars']}",
                f"- bucket: {doc.get('length_bucket')}",
                f"- A chunks: {doc['A']['child_chunk_count']} | B chunks: {doc['B']['child_chunk_count']}",
                "",
            ]
        )
        for side in ("A", "B"):
            review_md.append(f"### {side}")
            for ch in doc[side]["chunks"]:
                review_md.append(
                    f"- `{ch['chunk_id']}` idx={ch['chunk_index']} sec={ch['section_type']} "
                    f"native={ch['token_count_native']} bge={ch['token_count_bge']} "
                    f"ce={ch['token_count_ce']}"
                )
                preview = (ch.get("text_preview") or "").replace("\n", " ")
                review_md.append(f"  - preview: {preview[:220]}")
            review_md.append("")
    (out_dir / "boundary_review_v8_unreviewed.md").write_text(
        "\n".join(review_md) + "\n", encoding="utf-8"
    )

    print(f"WROTE {out_dir / 'chunk_qa_summary.json'}", flush=True)
    print(f"CLASSIFICATION {classification}", flush=True)
    print(f"SLICE_4_SAFE_TO_START {summary['SLICE_4_SAFE_TO_START']}", flush=True)
    print(
        f"DONE A={summary_a['total_child_chunks']} B={summary_b['total_child_chunks']} "
        f"mult={multiplier:.4f}",
        flush=True,
    )
    return 0 if classification == "CHUNK_QA_PASS_V8" else 2


if __name__ == "__main__":
    raise SystemExit(main())
