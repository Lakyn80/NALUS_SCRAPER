"""Verify lost_paragraph_heuristic cases from chunk QA v8.

Classifies each flagged case as FALSE_POSITIVE_HEURISTIC or ACTUAL_TEXT_LOSS.
Does not generate embeddings. Tokenizer-free.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from app.rag.legal_v2.ingest.chunkers import chunk_document_for_experiment
from app.rag.legal_v2.ingest.chunkers.names import (
    CHUNKER_A_CURRENT,
    CHUNKER_B_CONTEXTUAL_PACKED_V1,
)
from app.rag.legal_v2.parser import parse_legal_document

WS_RE = re.compile(r"\s+")


def _fetch(api: str, ecli: str) -> str:
    url = f"{api.rstrip('/')}/api/rag/documents/{urllib.parse.quote(ecli, safe='')}"
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    text = payload.get("full_text") or payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(f"empty text for {ecli}")
    return text


def _ws_norm(text: str) -> str:
    return WS_RE.sub(" ", (text or "").strip())


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _investigate_side(
    *,
    ecli: str,
    side: str,
    document: Any,
    result: Any,
) -> list[dict[str, Any]]:
    chunks = list(result.child_chunks)
    covered_ids: set[str] = set()
    for c in chunks:
        covered_ids.update(c.paragraph_ids)

    chunk_concat = "\n\n".join(c.text for c in chunks)
    # Per-paragraph reconstituted text from chunks that reference the pid.
    pieces_by_pid: dict[str, list[str]] = {}
    chunk_ids_by_pid: dict[str, list[str]] = {}
    for c in chunks:
        for pid in c.paragraph_ids:
            chunk_ids_by_pid.setdefault(pid, []).append(c.chunk_id)
            # Prefer exact unit text from chunk body when pid appears;
            # also keep paragraph_texts entry if present.
            ptxt = (c.paragraph_texts or {}).get(pid)
            if ptxt:
                pieces_by_pid.setdefault(pid, []).append(ptxt)

    cases: list[dict[str, Any]] = []
    for p in document.paragraphs:
        sample = p.normalized_text.strip()
        if len(sample) > 80:
            sample = sample[:80]
        heuristic_flag = bool(sample) and sample not in chunk_concat
        if not heuristic_flag:
            continue

        pid = p.paragraph_id
        in_chunk_ids = pid in covered_ids
        source_norm = _ws_norm(p.normalized_text)
        source_hash = _sha(p.normalized_text)

        # Confirm coverage: whitespace-normalized source must be covered by
        # (a) any chunk text, or (b) paragraph_texts for that pid, or
        # (c) concatenation of chunk texts that include the pid containing all tokens.
        covered_by_chunk_text = source_norm and source_norm in _ws_norm(chunk_concat)
        covered_by_paragraph_texts = False
        if pid in pieces_by_pid:
            joined = _ws_norm("\n\n".join(pieces_by_pid[pid]))
            covered_by_paragraph_texts = source_norm in joined or joined in source_norm or (
                # split oversized: all source tokens appear across pieces
                set(source_norm.split()).issubset(set(joined.split()))
                if source_norm
                else False
            )

        # Stronger split-aware check: collect raw chunk text segments containing pid
        # via source_spans / text inclusion of pieces.
        span_texts: list[str] = []
        for c in chunks:
            if pid not in c.paragraph_ids:
                continue
            # If paragraph was split, chunk text may contain only a piece.
            span_texts.append(c.text)
        span_join = _ws_norm("\n\n".join(span_texts))
        split_token_coverage = False
        if source_norm and span_join:
            src_tokens = source_norm.split()
            span_tokens = set(span_join.split())
            # For non-split, expect full source in span_join.
            if source_norm in span_join:
                split_token_coverage = True
            elif src_tokens and all(tok in span_tokens for tok in src_tokens):
                split_token_coverage = True

        actual_loss = not (
            in_chunk_ids
            and (covered_by_chunk_text or covered_by_paragraph_texts or split_token_coverage)
        )
        # Missing paragraph_id is definitive actual loss.
        if not in_chunk_ids:
            actual_loss = True

        reason = (
            "prefix80_not_in_chunk_concat"
            + ("; paragraph_id_present" if in_chunk_ids else "; paragraph_id_MISSING")
            + (
                "; likely_split_or_ws_mismatch"
                if in_chunk_ids and not covered_by_chunk_text
                else ""
            )
        )

        cases.append(
            {
                "ecli": ecli,
                "side": side,
                "paragraph_id": pid,
                "paragraph_index": p.paragraph_index,
                "section_type": p.section_type.value,
                "source_char_len": len(p.normalized_text),
                "source_sha256": source_hash,
                "source_text_preview": p.normalized_text[:400],
                "heuristic_sample_prefix80": sample,
                "heuristic_flag_reason": reason,
                "paragraph_id_in_chunks": in_chunk_ids,
                "chunk_ids_containing_paragraph": chunk_ids_by_pid.get(pid, []),
                "covered_by_ws_normalized_chunk_concat": covered_by_chunk_text,
                "covered_by_paragraph_texts": covered_by_paragraph_texts,
                "covered_by_split_token_union": split_token_coverage,
                "classification": (
                    "ACTUAL_TEXT_LOSS" if actual_loss else "FALSE_POSITIVE_HEURISTIC"
                ),
            }
        )
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--inventory",
        type=Path,
        default=Path("/app/artifacts/legal_v2/chunking_ab_pilot_300_v1/inventory_manifest.json"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/app/artifacts/legal_v2/chunking_ab_pilot_300_v1/chunk_qa_v8/"
            "lost_paragraph_heuristic_verification.json"
        ),
    )
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    args = ap.parse_args()

    inv = json.loads(args.inventory.read_text(encoding="utf-8"))
    ordered = list(inv["ordered_eclis"])
    meta = {d["ecli"]: d for d in inv.get("documents") or []}

    all_cases: list[dict[str, Any]] = []
    for i, ecli in enumerate(ordered, start=1):
        print(f"[{i}/{len(ordered)}] {ecli}", flush=True)
        text = _fetch(args.api, ecli)
        court = str((meta.get(ecli) or {}).get("court") or "")
        parsed = parse_legal_document(
            document_id=ecli, text=text, metadata={"court": court, "ecli": ecli}
        )
        ra = chunk_document_for_experiment(parsed, chunker_version=CHUNKER_A_CURRENT)
        rb = chunk_document_for_experiment(
            parsed, chunker_version=CHUNKER_B_CONTEXTUAL_PACKED_V1
        )
        all_cases.extend(_investigate_side(ecli=ecli, side="A", document=parsed, result=ra))
        all_cases.extend(_investigate_side(ecli=ecli, side="B", document=parsed, result=rb))

    a_cases = [c for c in all_cases if c["side"] == "A"]
    b_cases = [c for c in all_cases if c["side"] == "B"]
    confirmed_a = sum(1 for c in a_cases if c["classification"] == "ACTUAL_TEXT_LOSS")
    confirmed_b = sum(1 for c in b_cases if c["classification"] == "ACTUAL_TEXT_LOSS")
    fp_a = sum(1 for c in a_cases if c["classification"] == "FALSE_POSITIVE_HEURISTIC")
    fp_b = sum(1 for c in b_cases if c["classification"] == "FALSE_POSITIVE_HEURISTIC")

    payload = {
        "inventory_id": inv["inventory_id"],
        "inventory_hash_sha256": inv["inventory_hash_sha256"],
        "document_count": len(ordered),
        "heuristic_alerts_a": len(a_cases),
        "heuristic_alerts_b": len(b_cases),
        "false_positive_heuristic_a": fp_a,
        "false_positive_heuristic_b": fp_b,
        "confirmed_text_loss_a": confirmed_a,
        "confirmed_text_loss_b": confirmed_b,
        "cases": all_cases,
        "note": (
            "Heuristic uses first-80-char substring of normalized_text in raw chunk_concat. "
            "Confirmed loss requires missing paragraph_id or uncovered normalized text "
            "after whitespace normalization / split-aware piece union."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {args.out}", flush=True)
    print(
        f"SUMMARY heuristic_a={len(a_cases)} heuristic_b={len(b_cases)} "
        f"confirmed_a={confirmed_a} confirmed_b={confirmed_b} "
        f"fp_a={fp_a} fp_b={fp_b}",
        flush=True,
    )
    return 0 if confirmed_a == 0 and confirmed_b == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
