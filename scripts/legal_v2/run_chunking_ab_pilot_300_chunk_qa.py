#!/usr/bin/env python3
"""Chunk-only A/B QA over chunking_ab_pilot_300_v1 (NO embeddings / NO indexes).

Loads each inventory ECLI's full text from the Stage1 API, re-parses with the
production parser, then chunks with:
  A = production build_hierarchical_chunks (via chunk_document_for_experiment)
  B = legal_contextual_packed_v1

Reports native sizing units separately from BGE-M3 / CE tokenizer lengths.
Hard-aborts on inventory / ECLI-set integrity failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.rag.legal_v2.ingest.chunkers import chunk_document_for_experiment
from app.rag.legal_v2.ingest.chunkers.contextual_packed_v1 import ContextualPackedConfigV1
from app.rag.legal_v2.ingest.chunkers.names import (
    CHUNKER_A_CURRENT,
    CHUNKER_B_CONTEXTUAL_PACKED_V1,
)
from app.rag.legal_v2.ingest.chunking import HierarchicalChunkingResult
from app.rag.legal_v2.models import LegalDocumentStructure
from app.rag.legal_v2.parser import parse_legal_document

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = (
    PROJECT_ROOT / "benchmarks" / "legal_v2" / "chunking_ab_pilot_300_v1" / "inventory_manifest.json"
)
DEFAULT_OUT = PROJECT_ROOT / "artifacts" / "legal_v2" / "chunking_ab_pilot_300_v1" / "chunk_qa"
DEFAULT_API = "http://127.0.0.1:8029"
NATIVE_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
BGE_MODEL_ID = "BAAI/bge-m3"
CE_MODEL_ID = "BAAI/bge-reranker-v2-m3"
BGE_INPUT_LIMIT = 8192
CE_MAX_LENGTH = 512
EXTRAPOLATION_DOCS = 75_000
NEAR_IDENTICAL_RATIO = 0.92


def _request_json(url: str, timeout: float = 180.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_full_text(api: str, ecli: str) -> str:
    url = f"{api.rstrip('/')}/api/rag/documents/{urllib.parse.quote(ecli, safe='')}"
    payload = _request_json(url, timeout=180.0)
    text = payload.get("full_text") or ""
    if not str(text).strip():
        raise RuntimeError(f"empty full_text for {ecli}")
    return str(text)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = (len(ordered) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo))


def _stats(values: list[int | float], *, extras: bool = False) -> dict[str, float]:
    floats = [float(v) for v in values]
    if not floats:
        base = {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
        return base
    out = {
        "count": float(len(floats)),
        "mean": statistics.fmean(floats),
        "median": float(statistics.median(floats)),
        "p10": _percentile(floats, 0.10),
        "p25": _percentile(floats, 0.25),
        "p75": _percentile(floats, 0.75),
        "p90": _percentile(floats, 0.90),
        "p95": _percentile(floats, 0.95),
        "p99": _percentile(floats, 0.99),
        "max": float(max(floats)),
    }
    if extras:
        return out
    return out


def _policy_hash(config: ContextualPackedConfigV1) -> str:
    payload = {
        "chunker_version": config.chunker_version,
        "soft_min_tokens": config.soft_min_tokens,
        "soft_target_tokens": config.soft_target_tokens,
        "hard_max_tokens": config.hard_max_tokens,
        "overlap_max_tokens": config.overlap_max_tokens,
        "overlap_rule": "whole_paragraph_or_none",
        "section_rule": "never_cross_SectionType",
        "oversized_rule": "sentence_then_clause_then_token",
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_tokenizer(model_id: str) -> Callable[[str], int]:
    from transformers import AutoTokenizer  # type: ignore[import]

    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

    def count(text: str) -> int:
        # Passage-only length; no special pairing with query here.
        return int(len(tok.encode(text, add_special_tokens=True)))

    return count


def _prefix_overlap_chars(a: str, b: str) -> int:
    """Longest suffix of a that equals a prefix of b (character-level)."""
    max_k = min(len(a), len(b))
    for k in range(max_k, 0, -1):
        if a[-k:] == b[:k]:
            return k
    return 0


def _near_identical(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    # Cheap Jaccard over whitespace tokens for near-identical adjacent chunks.
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    union = len(ta | tb)
    return (inter / union) >= NEAR_IDENTICAL_RATIO


def _audit_branch(
    *,
    document: LegalDocumentStructure,
    result: HierarchicalChunkingResult,
    chunker_version: str,
    bge_count: Callable[[str], int],
    ce_count: Callable[[str], int],
    is_b: bool,
    b_config: ContextualPackedConfigV1,
) -> dict[str, Any]:
    chunks = list(result.child_chunks)
    empty = [c.chunk_id for c in chunks if not c.text.strip()]
    dup_ids = [cid for cid, n in Counter(c.chunk_id for c in chunks).items() if n > 1]
    covered: set[str] = set()
    for chunk in chunks:
        covered.update(chunk.paragraph_ids)
    missing_paragraphs = [
        p.paragraph_id for p in document.paragraphs if p.paragraph_id not in covered
    ]
    # Contamination: chunk document_id / ecli mismatch.
    contamination = [
        c.chunk_id
        for c in chunks
        if getattr(c, "document_id", document.document_id) not in (None, document.document_id)
        and str(getattr(c, "document_id", "")) != document.document_id
    ]

    para_map = {p.paragraph_id: p for p in document.paragraphs}
    section_violations: list[dict[str, Any]] = []
    for chunk in chunks:
        sections = {
            para_map[pid].section_type for pid in chunk.paragraph_ids if pid in para_map
        }
        if len(sections) > 1:
            section_violations.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "sections": sorted(s.value for s in sections),
                }
            )

    # Order preservation: first paragraph index sequence should be non-decreasing
    # by original paragraph order.
    para_order = {p.paragraph_id: i for i, p in enumerate(document.paragraphs)}
    order_breaks = 0
    last_idx = -1
    for chunk in chunks:
        idxs = [para_order[pid] for pid in chunk.paragraph_ids if pid in para_order]
        if not idxs:
            continue
        first = min(idxs)
        if first < last_idx:
            order_breaks += 1
        last_idx = max(idxs)

    overlap_chars = 0
    total_chars = sum(len(c.text) for c in chunks) or 1
    near_identical_adj = 0
    adj_pairs = 0
    b_overlap_policy_violations: list[str] = []
    for i in range(1, len(chunks)):
        prev, cur = chunks[i - 1], chunks[i]
        ov = _prefix_overlap_chars(prev.text, cur.text)
        overlap_chars += ov
        adj_pairs += 1
        if _near_identical(prev.text, cur.text):
            near_identical_adj += 1
        if is_b and ov > 0:
            # Intentional B overlap reuses the same complete paragraph unit as the
            # last unit of prev and the first unit of cur (same paragraph_id).
            # Identical adjacent source paragraphs with different IDs can share
            # character prefixes without being overlap-policy violations.
            prev_last_pid = prev.paragraph_ids[-1] if prev.paragraph_ids else None
            cur_first_pid = cur.paragraph_ids[0] if cur.paragraph_ids else None
            intentional_overlap = (
                prev_last_pid is not None
                and cur_first_pid is not None
                and prev_last_pid == cur_first_pid
            )
            if intentional_overlap:
                para = para_map.get(prev_last_pid)
                para_text = (para.normalized_text.strip() if para is not None else "")
                ntok = len(NATIVE_TOKEN_RE.findall(para_text)) if para_text else 0
                starts_with_complete = bool(para_text) and cur.text.lstrip().startswith(
                    para_text
                )
                ends_with_complete = bool(para_text) and prev.text.rstrip().endswith(
                    para_text
                )
                if starts_with_complete and ends_with_complete:
                    if ntok > b_config.overlap_max_tokens:
                        # Oversized paragraph must never be used as overlap.
                        b_overlap_policy_violations.append(cur.chunk_id)
                    if prev.section_type != cur.section_type:
                        section_violations.append(
                            {
                                "chunk_id": cur.chunk_id,
                                "sections": [
                                    prev.section_type.value,
                                    cur.section_type.value,
                                ],
                                "kind": "overlap_cross_section",
                            }
                        )
                else:
                    # Shared paragraph_id after oversized split pieces is continuity,
                    # not whole-paragraph overlap. Still forbid cross-section bleed.
                    if prev.section_type != cur.section_type:
                        section_violations.append(
                            {
                                "chunk_id": cur.chunk_id,
                                "sections": [
                                    prev.section_type.value,
                                    cur.section_type.value,
                                ],
                                "kind": "overlap_cross_section",
                            }
                        )

    text_hashes = Counter(hashlib.sha256(c.text.encode("utf-8")).hexdigest() for c in chunks)
    duplicate_text_chunks = sum(n for n in text_hashes.values() if n > 1)

    native_tokens = [int(c.token_count) for c in chunks]
    bge_tokens = [bge_count(c.text) for c in chunks]
    ce_tokens = [ce_count(c.text) for c in chunks]

    # Source coverage: every normalized paragraph text should appear as substring
    # of the concatenation of unique paragraph texts covered by chunks.
    # Primary gate remains paragraph_id coverage.
    source_norm = "\n\n".join(p.normalized_text for p in document.paragraphs)
    chunk_concat = "\n\n".join(c.text for c in chunks)
    # Lost text heuristic: first-80-char substring probe on raw chunk_concat.
    # Can false-positive on whitespace / oversized split pieces.
    # Confirmed loss uses paragraph_id coverage + whitespace-normalized text coverage.
    lost_paragraphs = []
    confirmed_text_loss: list[str] = []
    ws_re = re.compile(r"\s+")

    def _ws(text: str) -> str:
        return ws_re.sub(" ", (text or "").strip())

    chunk_concat_ws = _ws(chunk_concat)
    pieces_by_pid: dict[str, list[str]] = {}
    for c in chunks:
        for pid in c.paragraph_ids:
            ptxt = (c.paragraph_texts or {}).get(pid)
            if ptxt:
                pieces_by_pid.setdefault(pid, []).append(ptxt)
            pieces_by_pid.setdefault(pid, []).append(c.text)
    for p in document.paragraphs:
        sample = p.normalized_text.strip()
        if len(sample) > 80:
            sample = sample[:80]
        if sample and sample not in chunk_concat:
            lost_paragraphs.append(p.paragraph_id)
        source_ws = _ws(p.normalized_text)
        if not source_ws:
            continue
        pid_present = p.paragraph_id in covered
        if not pid_present:
            confirmed_text_loss.append(p.paragraph_id)
            continue
        joined = _ws("\n\n".join(pieces_by_pid.get(p.paragraph_id) or []))
        covered_ok = (
            source_ws in chunk_concat_ws
            or source_ws in joined
            or (
                bool(source_ws.split())
                and set(source_ws.split()).issubset(set(joined.split()))
            )
        )
        if not covered_ok:
            confirmed_text_loss.append(p.paragraph_id)
    return {
        "chunker_version": chunker_version,
        "paragraph_count": len(document.paragraphs),
        "child_chunk_count": len(chunks),
        "empty_chunk_ids": empty,
        "duplicate_chunk_ids": dup_ids,
        "missing_paragraph_ids": missing_paragraphs,
        "lost_paragraph_samples": lost_paragraphs[:20],
        "confirmed_text_loss_ids": confirmed_text_loss[:50],
        "contamination_chunk_ids": contamination,
        "section_violations": section_violations,
        "order_breaks": order_breaks,
        "overlap_ratio": overlap_chars / total_chars,
        "duplicate_text_ratio": duplicate_text_chunks / max(len(chunks), 1),
        "near_identical_adjacent_ratio": (
            near_identical_adj / adj_pairs if adj_pairs else 0.0
        ),
        "b_overlap_policy_violations": b_overlap_policy_violations,
        "native_tokens": native_tokens,
        "bge_tokens": bge_tokens,
        "ce_tokens": ce_tokens,
        "bge_above_limit": sum(1 for t in bge_tokens if t > BGE_INPUT_LIMIT),
        "ce_truncation_risk": sum(1 for t in ce_tokens if t > CE_MAX_LENGTH),
        "ce_would_truncate": sum(1 for t in ce_tokens if t > CE_MAX_LENGTH),
        "source_char_count": len(source_norm),
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "chunk_index": c.chunk_index,
                "section_type": c.section_type.value,
                "paragraph_ids": list(c.paragraph_ids),
                "token_count_native": int(c.token_count),
                "token_count_bge": bge_tokens[i],
                "token_count_ce": ce_tokens[i],
                "char_count": len(c.text),
                "text_preview": c.text[:280],
                "text": c.text,
            }
            for i, c in enumerate(chunks)
        ],
        "diagnostics": {
            "split_overlong_paragraph_count": result.diagnostics.split_overlong_paragraph_count,
            "section_distribution": dict(result.diagnostics.section_distribution),
        },
    }


def _branch_summary(audits: list[dict[str, Any]]) -> dict[str, Any]:
    chunks_per_doc = [a["child_chunk_count"] for a in audits]
    native: list[int] = []
    bge: list[int] = []
    ce: list[int] = []
    bge_above = 0
    ce_trunc = 0
    total_chunks = 0
    for a in audits:
        native.extend(a["native_tokens"])
        bge.extend(a["bge_tokens"])
        ce.extend(a["ce_tokens"])
        bge_above += a["bge_above_limit"]
        ce_trunc += a["ce_would_truncate"]
        total_chunks += a["child_chunk_count"]
    return {
        "documents": len(audits),
        "total_child_chunks": total_chunks,
        "chunks_per_document": _stats(chunks_per_doc),
        "native_tokens_per_chunk": _stats(native),
        "bge_tokens_per_chunk": {
            **_stats(bge),
            "fraction_above_bge_input_limit": bge_above / max(total_chunks, 1),
            "bge_input_limit": BGE_INPUT_LIMIT,
            "count_above_limit": bge_above,
        },
        "ce_tokens_per_chunk": {
            **_stats(ce),
            "fraction_at_truncation_risk": ce_trunc / max(total_chunks, 1),
            "fraction_would_truncate_under_ce_max_length": ce_trunc / max(total_chunks, 1),
            "ce_max_length": CE_MAX_LENGTH,
            "count_would_truncate": ce_trunc,
        },
        "overlap_ratio_mean": statistics.fmean([a["overlap_ratio"] for a in audits]) if audits else 0.0,
        "duplicate_text_ratio_mean": (
            statistics.fmean([a["duplicate_text_ratio"] for a in audits]) if audits else 0.0
        ),
        "near_identical_adjacent_ratio_mean": (
            statistics.fmean([a["near_identical_adjacent_ratio"] for a in audits]) if audits else 0.0
        ),
        "projected_chunks_75k": round((total_chunks / max(len(audits), 1)) * EXTRAPOLATION_DOCS),
    }


def _classify(summary_a: dict[str, Any], summary_b: dict[str, Any], gates: dict[str, Any]) -> str:
    if gates.get("blocked"):
        return "CHUNK_EXPERIMENT_BLOCKED"
    if gates.get("structural_regression"):
        return "CHUNK_B_STRUCTURAL_REGRESSION"
    a_total = max(int(summary_a["total_child_chunks"]), 1)
    b_total = int(summary_b["total_child_chunks"])
    if b_total / a_total >= 1.5:
        return "CHUNK_B_EXCESSIVE_EXPANSION"
    # Absolute CE truncation can be high for BOTH A and B when passages are long
    # vs CE max_length=512. Flag B only when it is materially worse than A.
    b_trunc = float(summary_b["ce_tokens_per_chunk"]["fraction_would_truncate_under_ce_max_length"])
    a_trunc = float(summary_a["ce_tokens_per_chunk"]["fraction_would_truncate_under_ce_max_length"])
    if b_trunc >= 0.10 and b_trunc >= (a_trunc + 0.05):
        return "CHUNK_B_TRUNCATION_RISK"
    return "CHUNK_QA_PASS"


def _select_review_eclis(
    inventory: dict[str, Any],
    per_doc: list[dict[str, Any]],
    *,
    target: int = 28,
) -> list[str]:
    by_ecli = {row["ecli"]: row for row in inventory.get("documents") or []}
    audits = {row["ecli"]: row for row in per_doc}
    selected: list[str] = []

    def add(ecli: str | None) -> None:
        if not ecli or ecli in selected or ecli not in audits:
            return
        selected.append(ecli)

    for doc in inventory.get("documents") or []:
        if doc.get("selection_reason") == "mandatory_golden_or_hn":
            add(doc["ecli"])

    by_bucket: dict[str, list[str]] = defaultdict(list)
    for ecli, meta in by_ecli.items():
        by_bucket[str(meta.get("length_bucket") or "unknown")].append(ecli)
    rng = random.Random(20260809)
    for bucket in ("short", "medium", "long", "very_long"):
        pool = sorted(by_bucket.get(bucket) or [])
        rng.shuffle(pool)
        for ecli in pool[:5]:
            add(ecli)

    # Prefer documents with multiple section types in B.
    multi_section = []
    for row in per_doc:
        dist = row["B"]["diagnostics"]["section_distribution"]
        if len(dist) >= 2:
            multi_section.append(row["ecli"])
    rng.shuffle(multi_section)
    for ecli in multi_section:
        add(ecli)
        if len(selected) >= target:
            break
    for ecli in sorted(audits):
        add(ecli)
        if len(selected) >= target:
            break
    return selected[:target]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--expected-count", type=int, default=300)
    parser.add_argument("--skip-model-tokenizers", action="store_true")
    parser.add_argument("--persist-all-chunks", action="store_true")
    args = parser.parse_args()

    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    ordered = list(inventory.get("ordered_eclis") or [])
    if inventory.get("inventory_id") != "chunking_ab_pilot_300_v1":
        raise SystemExit(f"unexpected inventory_id={inventory.get('inventory_id')}")
    if len(ordered) != int(inventory.get("document_count") or -1):
        raise SystemExit("inventory ordered_eclis length != document_count")
    if len(ordered) != args.expected_count:
        raise SystemExit(
            f"ABORT: inventory count {len(ordered)} != expected_count {args.expected_count}"
        )
    if len(set(ordered)) != len(ordered):
        raise SystemExit("ABORT: duplicate ECLIs in inventory")

    if args.limit and args.limit > 0:
        ordered = ordered[: args.limit]

    b_config = ContextualPackedConfigV1()
    b_hash = _policy_hash(b_config)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_model_tokenizers:
        bge_count: Callable[[str], int] = lambda t: len(NATIVE_TOKEN_RE.findall(t))
        ce_count: Callable[[str], int] = lambda t: len(NATIVE_TOKEN_RE.findall(t))
        tokenizer_mode = "native_proxy_ONLY_FOR_DEBUG"
    else:
        print("Loading BGE-M3 tokenizer...", flush=True)
        bge_count = _load_tokenizer(BGE_MODEL_ID)
        print("Loading CE tokenizer...", flush=True)
        ce_count = _load_tokenizer(CE_MODEL_ID)
        tokenizer_mode = "actual_model_tokenizers"

    audits_a: list[dict[str, Any]] = []
    audits_b: list[dict[str, Any]] = []
    per_doc: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    a_eclis: set[str] = set()
    b_eclis: set[str] = set()

    for index, ecli in enumerate(ordered, start=1):
        print(f"[{index}/{len(ordered)}] {ecli}", flush=True)
        try:
            text = _fetch_full_text(args.api, ecli)
            parsed = parse_legal_document(document_id=ecli, text=text)

            result_a = chunk_document_for_experiment(parsed, chunker_version=CHUNKER_A_CURRENT)
            result_b = chunk_document_for_experiment(
                parsed, chunker_version=CHUNKER_B_CONTEXTUAL_PACKED_V1
            )
            # Prove A path is production hierarchical.
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
            a_eclis.add(ecli)
            b_eclis.add(ecli)
            audits_a.append(branch_a)
            audits_b.append(branch_b)
            per_doc.append(
                {
                    "ecli": ecli,
                    "source_char_count": len(text),
                    "A": {k: v for k, v in branch_a.items() if k != "chunks"},
                    "B": {k: v for k, v in branch_b.items() if k != "chunks"},
                    "_chunks_a": branch_a["chunks"],
                    "_chunks_b": branch_b["chunks"],
                }
            )
            if args.persist_all_chunks:
                (out_dir / "chunks_a").mkdir(exist_ok=True)
                (out_dir / "chunks_b").mkdir(exist_ok=True)
                safe = ecli.replace(":", "_")
                (out_dir / "chunks_a" / f"{safe}.json").write_text(
                    json.dumps(branch_a, ensure_ascii=False), encoding="utf-8"
                )
                (out_dir / "chunks_b" / f"{safe}.json").write_text(
                    json.dumps(branch_b, ensure_ascii=False), encoding="utf-8"
                )
        except Exception as exc:  # noqa: BLE001
            failures.append({"ecli": ecli, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  FAIL {exc}", flush=True)

    inventory_set = set(ordered)
    gates = {
        "blocked": False,
        "structural_regression": False,
        "reasons": [],
    }
    if failures:
        gates["blocked"] = True
        gates["reasons"].append(f"document_failures={len(failures)}")
        (out_dir / "chunk_qa_failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if a_eclis != inventory_set or b_eclis != inventory_set:
        # When --limit is used, compare against limited set.
        expected = set(ordered)
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
    sec_b = sum(len(a["section_violations"]) for a in audits_b)
    ov_b = sum(len(a["b_overlap_policy_violations"]) for a in audits_b)
    order_b = sum(int(a["order_breaks"]) for a in audits_b)
    contam = sum(len(a["contamination_chunk_ids"]) for a in audits_a + audits_b)

    integrity = {
        "empty_chunks_a": empty_a,
        "empty_chunks_b": empty_b,
        "duplicate_chunk_ids_a": dup_a,
        "duplicate_chunk_ids_b": dup_b,
        "missing_paragraph_ids_a": miss_a,
        "missing_paragraph_ids_b": miss_b,
        "lost_paragraph_heuristic_a": lost_a,
        "lost_paragraph_heuristic_b": lost_b,
        "section_violations_b": sec_b,
        "b_overlap_policy_violations": ov_b,
        "order_breaks_b": order_b,
        "cross_document_contamination": contam,
        "malformed_ecli_count": sum(1 for e in ordered if not str(e).startswith("ECLI:")),
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
        )
    ):
        gates["blocked"] = True
        gates["reasons"].append("integrity_gate_failed")
    if sec_b or ov_b or order_b or lost_b:
        gates["structural_regression"] = True
        gates["reasons"].append("b_structural_issues")

    summary_a = _branch_summary(audits_a)
    summary_b = _branch_summary(audits_b)
    classification = _classify(summary_a, summary_b, gates)
    multiplier = (
        summary_b["total_child_chunks"] / max(summary_a["total_child_chunks"], 1)
        if audits_a
        else 0.0
    )

    # Representative review artifact (strip full texts from main summary).
    review_eclis = _select_review_eclis(inventory, per_doc, target=28)
    review_docs = []
    for ecli in review_eclis:
        row = next(r for r in per_doc if r["ecli"] == ecli)
        chunks_a = row["_chunks_a"]
        chunks_b = row["_chunks_b"]
        # Keep first few + a mid + last for each side.
        def pick(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if len(chunks) <= 6:
                return [{k: v for k, v in c.items() if k != "text"} for c in chunks]
            idxs = sorted(set([0, 1, 2, len(chunks) // 2, len(chunks) - 2, len(chunks) - 1]))
            return [{k: v for k, v in chunks[i].items() if k != "text"} for i in idxs]

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
                "document_length_chars": row["source_char_count"],
                "length_bucket": next(
                    (
                        d.get("length_bucket")
                        for d in inventory.get("documents") or []
                        if d.get("ecli") == ecli
                    ),
                    None,
                ),
                "selection_reason": next(
                    (
                        d.get("selection_reason")
                        for d in inventory.get("documents") or []
                        if d.get("ecli") == ecli
                    ),
                    None,
                ),
                "A": {
                    "child_chunk_count": row["A"]["child_chunk_count"],
                    "chunks": pick(chunks_a),
                },
                "B": {
                    "child_chunk_count": row["B"]["child_chunk_count"],
                    "chunks": pick(chunks_b),
                    "overlap_markers": overlap_markers,
                },
            }
        )

    review_path = out_dir / "boundary_review_unreviewed.json"
    review_path.write_text(json.dumps(review_docs, ensure_ascii=False, indent=2), encoding="utf-8")
    review_md_lines = [
        "# Chunking A/B boundary review (UNREVIEWED)",
        "",
        "Manual inspection only. Do not treat as legal quality judgment.",
        "",
    ]
    for doc in review_docs:
        review_md_lines.extend(
            [
                f"## {doc['ecli']}",
                f"- chars: {doc['document_length_chars']}",
                f"- bucket: {doc.get('length_bucket')}",
                f"- reason: {doc.get('selection_reason')}",
                f"- A chunks: {doc['A']['child_chunk_count']} | B chunks: {doc['B']['child_chunk_count']}",
                "",
            ]
        )
        for side in ("A", "B"):
            review_md_lines.append(f"### {side}")
            for ch in doc[side]["chunks"]:
                review_md_lines.append(
                    f"- `{ch['chunk_id']}` idx={ch['chunk_index']} sec={ch['section_type']} "
                    f"native={ch['token_count_native']} bge={ch['token_count_bge']} "
                    f"ce={ch['token_count_ce']}"
                )
                preview = (ch.get("text_preview") or "").replace("\n", " ")
                review_md_lines.append(f"  - preview: {preview[:220]}")
            review_md_lines.append("")
    review_md_path = out_dir / "boundary_review_unreviewed.md"
    review_md_path.write_text("\n".join(review_md_lines) + "\n", encoding="utf-8")

    suspicious = []
    for row in per_doc:
        if row["B"]["section_violations"] or row["B"]["b_overlap_policy_violations"]:
            suspicious.append(
                {
                    "ecli": row["ecli"],
                    "section_violations": row["B"]["section_violations"],
                    "overlap_violations": row["B"]["b_overlap_policy_violations"],
                }
            )
        ratio = row["B"]["child_chunk_count"] / max(row["A"]["child_chunk_count"], 1)
        if ratio >= 2.0 or ratio <= 0.4:
            suspicious.append(
                {
                    "ecli": row["ecli"],
                    "chunk_ratio_b_over_a": ratio,
                    "a_chunks": row["A"]["child_chunk_count"],
                    "b_chunks": row["B"]["child_chunk_count"],
                }
            )

    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inventory_id": inventory["inventory_id"],
        "inventory_hash_sha256": inventory["inventory_hash_sha256"],
        "document_count": len(ordered),
        "tokenizer_mode": tokenizer_mode,
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
        "A": summary_a,
        "B": summary_b,
        "relative_infrastructure_multiplier_b_over_a": {
            "embedding_count": multiplier,
            "bm25_row_count": multiplier,
            "qdrant_vector_count": multiplier,
            "note": "extrapolation from child-chunk counts only; no indexes were built",
        },
        "projected_75k": {
            "label": "EXTRAPOLATION_ONLY",
            "A_child_chunks": summary_a["projected_chunks_75k"],
            "B_child_chunks": summary_b["projected_chunks_75k"],
        },
        "mandatory_coverage": {
            "mandatory_count": inventory.get("mandatory_count"),
            "golden_query_evaluability": inventory.get("golden_query_evaluability"),
        },
        "review_artifact": str(review_path.as_posix()),
        "suspicious_examples": suspicious[:50],
        "no_embeddings_generated": True,
        "no_experimental_indexes_built": True,
        "full_75k_corpus_untouched": True,
        "documents": [
            {
                "ecli": row["ecli"],
                "source_char_count": row["source_char_count"],
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
    summary_path = out_dir / "chunk_qa_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "b_policy_freeze.json").write_text(
        json.dumps(summary["b_policy"], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = [
        "# Chunking A/B chunk-only QA",
        "",
        f"- classification: `{classification}`",
        f"- inventory_hash: `{inventory['inventory_hash_sha256']}`",
        f"- documents: {len(ordered)}",
        f"- tokenizer_mode: `{tokenizer_mode}`",
        f"- A total chunks: {summary_a['total_child_chunks']}",
        f"- B total chunks: {summary_b['total_child_chunks']}",
        f"- relative multiplier B/A: {multiplier:.4f}",
        f"- A projected 75k chunks: {summary_a['projected_chunks_75k']} (EXTRAPOLATION)",
        f"- B projected 75k chunks: {summary_b['projected_chunks_75k']} (EXTRAPOLATION)",
        f"- A CE would-truncate fraction: "
        f"{summary_a['ce_tokens_per_chunk']['fraction_would_truncate_under_ce_max_length']:.4f}",
        f"- B CE would-truncate fraction: "
        f"{summary_b['ce_tokens_per_chunk']['fraction_would_truncate_under_ce_max_length']:.4f}",
        f"- B policy hash: `{b_hash}`",
        f"- review: `{review_path.name}`",
        "",
        "## Integrity",
        "",
        "```json",
        json.dumps(integrity, indent=2),
        "```",
        "",
        "STOP GATE: Slice 4 (embeddings/indexes) was NOT started.",
    ]
    (out_dir / "chunk_qa_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"WROTE {summary_path}", flush=True)
    print(f"CLASSIFICATION {classification}", flush=True)
    print(
        f"DONE A_chunks={summary_a['total_child_chunks']} "
        f"B_chunks={summary_b['total_child_chunks']} multiplier={multiplier:.4f}",
        flush=True,
    )
    return 0 if classification == "CHUNK_QA_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
