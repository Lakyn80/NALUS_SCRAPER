#!/usr/bin/env python3
"""Dump full judgment texts for an existing hybrid eval artifact.

No LLM calls. Prefers the pilot BM25 sqlite sidecar (same payloads as Qdrant).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--verified-only", action="store_true")
    parser.add_argument(
        "--bm25-sqlite",
        type=Path,
        default=Path(
            os.getenv(
                "NALUS_LEGAL_V2_BM25_SQLITE",
                "/app/storage/rag/bm25/nalus_legal_paragraph_bm25_v2_pilot_600.sqlite",
            )
        ),
    )
    args = parser.parse_args()

    artifact = json.loads(args.eval_json.read_text(encoding="utf-8"))
    rows = list(artifact.get("rows") or [])
    if args.verified_only:
        rows = [row for row in rows if row.get("status") == "verified_match"]

    output_dir = args.output_dir or (args.eval_json.parent / "document_reviews")
    output_dir.mkdir(parents=True, exist_ok=True)

    needed_ids = sorted(
        {
            str(candidate.get("document_id") or "").strip()
            for row in rows
            for candidate in (row.get("candidate_documents") or [])
            if candidate.get("document_id")
        }
    )
    texts_by_id = _load_from_bm25(args.bm25_sqlite, needed_ids)

    written: list[Path] = []
    missing_pairs: list[str] = []
    for row in sorted(rows, key=lambda item: str(item.get("id") or "")):
        query_id = str(row.get("id") or "unknown")
        enriched: list[dict[str, Any]] = []
        for candidate in row.get("candidate_documents") or []:
            document_id = str(candidate.get("document_id") or "").strip()
            payload = texts_by_id.get(document_id) or {}
            text = str(payload.get("document_text") or "")
            if document_id and not text:
                missing_pairs.append(f"{query_id}:{document_id}")
            item = dict(candidate)
            item.update(
                {
                    "document_text": text or None,
                    "document_paragraphs": payload.get("document_paragraphs"),
                    "document_paragraph_count": payload.get("document_paragraph_count") or 0,
                    "document_text_char_count": len(text),
                }
            )
            enriched.append(item)
        written.append(
            _write_review(
                review_dir=output_dir,
                query_id=query_id,
                query=str(row.get("query") or ""),
                status=str(row.get("status") or ""),
                candidate_documents=enriched,
            )
        )

    with_text = sum(1 for value in texts_by_id.values() if value.get("document_text"))
    index_path = output_dir / "INDEX_from_bm25_dump.md"
    lines = [
        "# Document reviews dumped from BM25 sqlite (no LLM)",
        "",
        f"- Source eval: `{args.eval_json.as_posix()}`",
        f"- BM25 sqlite: `{args.bm25_sqlite.as_posix()}`",
        f"- Queries written: `{len(written)}`",
        f"- Unique document_ids requested: `{len(needed_ids)}`",
        f"- Documents with text: `{with_text}`",
        f"- Missing query/document pairs: `{len(missing_pairs)}`",
        "",
        "## Start here (verified_match queries)",
        "",
    ]
    for row in rows:
        if row.get("status") != "verified_match":
            continue
        qid = str(row.get("id") or "")
        safe = _safe_id(qid)
        lines.append(f"- [`{safe}_full_documents.md`](./{safe}_full_documents.md)")
    lines.extend(["", "## All query files", ""])
    for path in written:
        lines.append(f"- [`{path.name}`](./{path.name})")
    if missing_pairs:
        lines.extend(["", "## Missing", ""])
        for item in missing_pairs[:100]:
            lines.append(f"- `{item}`")
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "queries_written": len(written),
                "unique_document_ids": len(needed_ids),
                "documents_with_text": with_text,
                "missing_pairs": len(missing_pairs),
                "output_dir": str(output_dir),
                "index": str(index_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _load_from_bm25(db_path: Path, document_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not document_ids:
        return {}
    if not db_path.exists():
        raise FileNotFoundError(f"BM25 sqlite not found: {db_path}")
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    by_doc: dict[str, dict[str, dict[str, Any]]] = {document_id: {} for document_id in document_ids}
    wanted = set(document_ids)
    # Load in chunks to keep the IN clause bounded.
    batch_size = 80
    for start in range(0, len(document_ids), batch_size):
        batch = document_ids[start : start + batch_size]
        placeholders = ",".join("?" for _ in batch)
        rows = cur.execute(
            f"""
            SELECT document_id, text, metadata, paragraph_ids, section_type
            FROM bm25_chunks
            WHERE document_id IN ({placeholders})
            ORDER BY document_id, chunk_id
            """,
            batch,
        ).fetchall()
        for document_id, text, metadata_raw, paragraph_ids_raw, section_type in rows:
            document_id = str(document_id or "").strip()
            if document_id not in wanted:
                continue
            metadata: dict[str, Any] = {}
            if metadata_raw:
                try:
                    parsed = json.loads(metadata_raw)
                    if isinstance(parsed, dict):
                        metadata = parsed
                except json.JSONDecodeError:
                    metadata = {}
            paragraph_texts = metadata.get("paragraph_texts")
            if isinstance(paragraph_texts, dict) and paragraph_texts:
                for paragraph_id, paragraph_text in paragraph_texts.items():
                    _remember(
                        by_doc[document_id],
                        paragraph_id=str(paragraph_id),
                        text=str(paragraph_text or ""),
                        metadata=metadata,
                        section_type=section_type,
                    )
                continue
            paragraph_ids: list[str]
            if paragraph_ids_raw:
                try:
                    parsed_ids = json.loads(paragraph_ids_raw)
                    paragraph_ids = [str(item) for item in parsed_ids] if isinstance(parsed_ids, list) else []
                except json.JSONDecodeError:
                    paragraph_ids = []
            else:
                paragraph_ids = []
            if not paragraph_ids:
                paragraph_ids = [str(metadata.get("paragraph_id") or f"chunk:{metadata.get('chunk_id') or text[:24]}")]
            for paragraph_id in paragraph_ids:
                _remember(
                    by_doc[document_id],
                    paragraph_id=str(paragraph_id),
                    text=str(text or ""),
                    metadata=metadata,
                    section_type=section_type,
                )
    con.close()

    out: dict[str, dict[str, Any]] = {}
    for document_id, paragraphs in by_doc.items():
        ordered = sorted(
            paragraphs.values(),
            key=lambda item: (
                int(item.get("source_order") if item.get("source_order") is not None else 10**9),
                int(item.get("paragraph_index") if item.get("paragraph_index") is not None else 10**9),
                str(item.get("paragraph_id") or ""),
            ),
        )
        # Deduplicate identical consecutive texts (overlapping child chunks).
        cleaned: list[dict[str, Any]] = []
        seen_text: set[str] = set()
        for item in ordered:
            body = str(item.get("text") or "").strip()
            if not body or body in seen_text:
                continue
            seen_text.add(body)
            cleaned.append(item)
        joined = "\n\n".join(str(item.get("text") or "").strip() for item in cleaned)
        out[document_id] = {
            "document_paragraphs": cleaned,
            "document_paragraph_count": len(cleaned),
            "document_text": joined,
            "document_text_char_count": len(joined),
        }
    return out


def _remember(
    bucket: dict[str, dict[str, Any]],
    *,
    paragraph_id: str,
    text: str,
    metadata: dict[str, Any],
    section_type: Any,
) -> None:
    if not paragraph_id:
        return
    existing = bucket.get(paragraph_id)
    if existing and len(str(existing.get("text") or "")) >= len(text or ""):
        return
    bucket[paragraph_id] = {
        "paragraph_id": paragraph_id,
        "paragraph_index": metadata.get("paragraph_index")
        if metadata.get("paragraph_index") is not None
        else metadata.get("paragraph_start"),
        "source_order": metadata.get("source_order"),
        "section_type": metadata.get("section_type") or section_type,
        "text": text,
    }


def _safe_id(query_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", query_id)


def _write_review(
    *,
    review_dir: Path,
    query_id: str,
    query: str,
    status: str,
    candidate_documents: list[dict[str, Any]],
) -> Path:
    path = review_dir / f"{_safe_id(query_id)}_full_documents.md"
    lines: list[str] = [
        f"# Document review: `{query_id}`",
        "",
        f"- Query: {query}",
        f"- Eval status: `{status}`",
        f"- Candidates: `{len(candidate_documents)}`",
        "- Source: BM25 sqlite reconstruction (no LLM re-run)",
        "",
    ]
    for item in candidate_documents:
        lines.extend(
            [
                f"## Rank {item.get('candidate_rank')} — `{item.get('ecli') or item.get('document_id')}`",
                "",
                f"- document_id: `{item.get('document_id')}`",
                f"- benchmark_label: `{item.get('benchmark_label')}`",
                f"- final_decision: `{item.get('final_decision')}`",
                f"- relevance_classification: `{item.get('relevance_classification')}`",
                f"- fast_decision/class: `{item.get('fast_decision')}` / `{item.get('fast_classification')}`",
                f"- thinking_used: `{item.get('thinking_fallback_used')}`",
                f"- paragraph_count: `{item.get('document_paragraph_count')}`",
                f"- char_count: `{item.get('document_text_char_count')}`",
                "",
                "### Full document text",
                "",
                "```text",
                str(item.get("document_text") or ""),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
