#!/usr/bin/env python3
"""Export a read-only human-review Markdown report for the case-similarity pilot."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.benchmark.case_similarity_golden import (  # noqa: E402
    DEFAULT_PILOT_DATASET,
    CaseSimilarityGoldenItem,
    best_supporting_block_token_overlap,
    count_sentences,
    count_words,
    detect_query_leakage,
    load_case_similarity_golden_jsonl,
    longest_verbatim_sentence_overlap_tokens,
)
from app.rag.legal_v2.benchmark.corpus import DevelopmentCorpus, load_case_similarity_corpus  # noqa: E402
from app.rag.legal_v2.schema.canonical_v1 import CanonicalBlock  # noqa: E402

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "legal_v2"
    / "case_similarity_golden_v1_pilot"
    / "manual_review.md"
)
EXPECTED_QUERY_COUNT = 20
_SECTION_RE = re.compile(r"^## Query: ", re.MULTILINE)


@dataclass(frozen=True)
class ExportStats:
    query_sections: int
    supporting_blocks_rendered: int
    alternative_docs_rendered: int
    hard_negative_docs_rendered: int


class ManualReviewExportError(RuntimeError):
    """Raised when the pilot cannot be exported safely for human review."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_PILOT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    items = load_case_similarity_golden_jsonl(args.dataset)
    corpus = load_case_similarity_corpus()
    markdown, stats = build_manual_review_markdown(items, corpus)
    _validate_exported_markdown(markdown, items)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(
        "\n".join(
            [
                f"output={args.output}",
                f"query_sections={stats.query_sections}",
                f"supporting_blocks_rendered={stats.supporting_blocks_rendered}",
                f"alternative_docs_rendered={stats.alternative_docs_rendered}",
                f"hard_negative_docs_rendered={stats.hard_negative_docs_rendered}",
            ]
        )
    )
    return 0


def build_manual_review_markdown(
    items: list[CaseSimilarityGoldenItem],
    corpus: DevelopmentCorpus,
) -> tuple[str, ExportStats]:
    if len(items) != EXPECTED_QUERY_COUNT:
        raise ManualReviewExportError(
            f"expected {EXPECTED_QUERY_COUNT} queries, found {len(items)}"
        )
    _preflight_items(items, corpus)

    refs = {ref.document_id: ref for ref in corpus.documents}
    lines: list[str] = [
        "# Case Similarity Golden v1 Pilot — Manual Review",
        "",
        "Read-only human audit export for document-level case-similarity retrieval.",
        "Dataset: `benchmarks/legal_v2/case_similarity_golden_v1_pilot.jsonl`",
        "",
        "Human verdict values: `PASS` | `FIX` | `REJECT`",
        "",
        "## Summary",
        "",
        "| ID | Style | Diff | Query (short) | Primary doc | Alts | Hard neg | Verdict |",
        "|---|---|---|---|---|---:|---:|---|",
    ]

    supporting_count = 0
    alt_count = 0
    hard_count = 0

    for item in items:
        short_query = _shorten(item.query, 90)
        lines.append(
            "| {bid} | {style} | {diff} | {query} | `{doc}` | {alts} | {hards} |  |".format(
                bid=item.benchmark_id,
                style=item.query_style,
                diff=item.difficulty,
                query=short_query.replace("|", "/"),
                doc=item.source_document_id,
                alts=len(item.accepted_alternative_document_ids),
                hards=len(item.hard_negative_document_ids),
            )
        )

    lines.extend(["", "---", ""])

    for item in items:
        primary = refs[item.source_document_id]
        lines.extend(
            [
                f"## Query: {item.benchmark_id}",
                "",
                "### Full query",
                "",
                item.query,
                "",
                "### Metadata",
                "",
                f"- query_style: `{item.query_style}`",
                f"- difficulty: `{item.difficulty}`",
                f"- split: `{item.split}`",
                f"- language: `{item.language}`",
                f"- factual_facets: {', '.join(item.factual_facets) or '—'}",
                f"- legal_issue_facets: {', '.join(item.legal_issue_facets) or '—'}",
                f"- procedural_facets: {', '.join(item.procedural_facets) or '—'}",
                f"- hard_negative_evaluable: `{item.hard_negative_evaluable}`",
                f"- hard_negative_blocker: `{item.hard_negative_blocker}`",
                f"- builder_notes: {item.notes or '—'}",
                "",
                "### Human verdict",
                "",
                "- [ ] PASS",
                "- [ ] FIX",
                "- [ ] REJECT",
                "",
                "### Reviewer notes",
                "",
                "_Write notes here._",
                "",
                "### Primary expected document",
                "",
                f"- document_id: `{item.source_document_id}`",
                f"- review_number: `{primary.review_number}`",
                f"- court: `{primary.court}`",
                f"- decision_type: `{primary.decision_type}`",
                f"- case_number (metadata only, must not appear in query): `{primary.case_number}`",
                f"- decision_date: `{primary.decision_date}`",
                "",
                "### Similarity rationale",
                "",
                item.similarity_rationale,
                "",
                "### Supporting blocks",
                "",
            ]
        )

        evidence_by_block = {row.block_id: row.excerpt for row in item.answer_evidence}
        for block_id in item.supporting_block_ids:
            block = corpus.blocks_by_id[block_id]
            supporting_count += 1
            lines.extend(_render_block(block, evidence_by_block.get(block_id)))

        lines.extend(["### Accepted alternatives", ""])
        if not item.accepted_alternative_document_ids:
            lines.append("_None._")
            lines.append("")
        else:
            alt_map = {row.document_id: row.rationale for row in item.accepted_alternative_rationales}
            for document_id in item.accepted_alternative_document_ids:
                alt_count += 1
                alt_ref = refs[document_id]
                lines.extend(
                    [
                        f"#### Alternative `{document_id}`",
                        "",
                        f"- court: `{alt_ref.court}`",
                        f"- decision_type: `{alt_ref.decision_type}`",
                        f"- rationale: {alt_map.get(document_id, '—')}",
                        "",
                        "Relevant block previews:",
                        "",
                    ]
                )
                lines.extend(_preview_blocks(corpus.blocks_for_document(document_id), limit=3))

        lines.extend(["### Hard negatives", ""])
        hard_map = {
            row.document_id: row for row in item.hard_negative_rationales
        }
        for document_id in item.hard_negative_document_ids:
            hard_count += 1
            hard_ref = refs[document_id]
            rationale = hard_map[document_id]
            lines.extend(
                [
                    f"#### Hard negative `{document_id}`",
                    "",
                    f"- court: `{hard_ref.court}`",
                    f"- decision_type: `{hard_ref.decision_type}`",
                    f"- looks similar because: {rationale.looks_similar_because}",
                    f"- materially incorrect because: {rationale.materially_incorrect_because}",
                    "",
                    "Relevant block previews:",
                    "",
                ]
            )
            lines.extend(_preview_blocks(corpus.blocks_for_document(document_id), limit=3))

        leaks = detect_query_leakage(
            item.query,
            document_ids=[item.source_document_id]
            + item.accepted_alternative_document_ids
            + item.hard_negative_document_ids,
            case_numbers=[
                refs[doc].case_number
                for doc in [item.source_document_id]
                + item.accepted_alternative_document_ids
                + item.hard_negative_document_ids
            ],
            source_ids=[
                refs[doc].source_id
                for doc in [item.source_document_id]
                + item.accepted_alternative_document_ids
                + item.hard_negative_document_ids
            ],
        )
        longest_overlap = 0
        longest_overlap_text = ""
        longest_overlap_block_id = "none"
        overlap_count, overlap_text, overlap_block_id = best_supporting_block_token_overlap(
            item.query,
            corpus.blocks_by_id,
            item.supporting_block_ids,
        )
        longest_overlap = overlap_count
        longest_overlap_text = overlap_text or "—"
        longest_overlap_block_id = overlap_block_id or "none"
        # Keep the hard-rule signal available for reviewers without changing failure semantics.
        sentence_leak = 0
        for block_id in item.supporting_block_ids:
            block = corpus.blocks_by_id[block_id]
            sentence_leak = max(
                sentence_leak,
                longest_verbatim_sentence_overlap_tokens(item.query, block.raw_text),
            )
        lines.extend(
            [
                "### Query leakage diagnostics",
                "",
                f"- word_count: `{count_words(item.query)}`",
                f"- sentence_count: `{count_sentences(item.query)}`",
                f"- longest_verbatim_overlap_tokens: `{longest_overlap}`",
                f"- longest_verbatim_overlap_text: `{longest_overlap_text}`",
                f"- source supporting_block_id: `{longest_overlap_block_id}`",
                f"- complete_sentence_leak_tokens_ge12: `{sentence_leak}`",
                f"- detected_source_identifiers: `{', '.join(leaks) if leaks else 'none'}`",
                "",
                "---",
                "",
            ]
        )

    stats = ExportStats(
        query_sections=len(items),
        supporting_blocks_rendered=supporting_count,
        alternative_docs_rendered=alt_count,
        hard_negative_docs_rendered=hard_count,
    )
    return "\n".join(lines).rstrip() + "\n", stats


def _preflight_items(items: list[CaseSimilarityGoldenItem], corpus: DevelopmentCorpus) -> None:
    for item in items:
        if item.source_document_id not in {ref.document_id for ref in corpus.documents}:
            raise ManualReviewExportError(f"missing source document {item.source_document_id}")
        for block_id in item.supporting_block_ids:
            if block_id not in corpus.blocks_by_id:
                raise ManualReviewExportError(f"missing supporting block {block_id}")


def _validate_exported_markdown(markdown: str, items: list[CaseSimilarityGoldenItem]) -> None:
    sections = _SECTION_RE.findall(markdown)
    if len(sections) != EXPECTED_QUERY_COUNT:
        raise ManualReviewExportError(
            f"exported markdown must contain {EXPECTED_QUERY_COUNT} query sections, found {len(sections)}"
        )
    for item in items:
        if f"## Query: {item.benchmark_id}" not in markdown:
            raise ManualReviewExportError(f"missing section for {item.benchmark_id}")


def _render_block(block: CanonicalBlock, excerpt: str | None) -> list[str]:
    primary_class = getattr(block, "primary_class", None) or "—"
    lines = [
        f"#### Block `{block.block_id}`",
        "",
        f"- block_index: `{block.block_index}`",
        f"- primary_class: `{primary_class}`",
        "",
        "Full text:",
        "",
        "```text",
        (block.raw_text or "").rstrip(),
        "```",
        "",
    ]
    if excerpt:
        lines.extend(["Evidence excerpt:", "", f"> {excerpt}", ""])
    return lines


def _preview_blocks(blocks: list[CanonicalBlock], *, limit: int) -> list[str]:
    lines: list[str] = []
    selected = [block for block in blocks if len((block.raw_text or "").strip()) >= 80][:limit]
    if not selected:
        selected = blocks[:limit]
    for block in selected:
        preview = _shorten((block.raw_text or "").replace("\n", " "), 240)
        lines.extend(
            [
                f"- `{block.block_id}` ({getattr(block, 'primary_class', None) or '—'}): {preview}",
            ]
        )
    lines.append("")
    return lines


def _shorten(text: str, limit: int) -> str:
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"


if __name__ == "__main__":
    raise SystemExit(main())
