#!/usr/bin/env python3
"""Export a read-only human-review Markdown report for the Step 4A pilot.

Does not modify the JSONL dataset or builder definitions.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.benchmark.corpus import DevelopmentCorpus, load_development_corpus  # noqa: E402
from app.rag.legal_v2.benchmark.retrieval_golden import (  # noqa: E402
    DEFAULT_PILOT_DATASET,
    RetrievalGoldenItem,
    load_retrieval_golden_jsonl,
)
from app.rag.legal_v2.schema.canonical_v1 import CanonicalBlock  # noqa: E402

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "legal_v2"
    / "retrieval_golden_v1_pilot"
    / "manual_review.md"
)
EXPECTED_QUERY_COUNT = 30
_SECTION_RE = re.compile(r"^## Query: ", re.MULTILINE)


@dataclass(frozen=True)
class ExportStats:
    query_sections: int
    primary_blocks_rendered: int
    alternative_blocks_rendered: int
    hard_negative_blocks_rendered: int
    negative_candidate_blocks_rendered: int


class ManualReviewExportError(RuntimeError):
    """Raised when the pilot cannot be exported safely for human review."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_PILOT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    items = load_retrieval_golden_jsonl(args.dataset)
    corpus = load_development_corpus()
    markdown, stats = build_manual_review_markdown(items, corpus)
    _validate_exported_markdown(markdown, items)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")

    print(
        "\n".join(
            [
                f"output={args.output}",
                f"query_sections={stats.query_sections}",
                f"primary_blocks_rendered={stats.primary_blocks_rendered}",
                f"alternative_blocks_rendered={stats.alternative_blocks_rendered}",
                f"hard_negative_blocks_rendered={stats.hard_negative_blocks_rendered}",
                f"negative_candidate_blocks_rendered={stats.negative_candidate_blocks_rendered}",
            ]
        )
    )
    return 0


def build_manual_review_markdown(
    items: list[RetrievalGoldenItem],
    corpus: DevelopmentCorpus,
) -> tuple[str, ExportStats]:
    if len(items) != EXPECTED_QUERY_COUNT:
        raise ManualReviewExportError(
            f"expected {EXPECTED_QUERY_COUNT} queries, found {len(items)}"
        )
    if not items:
        raise ManualReviewExportError("dataset is empty")

    _preflight_items(items, corpus)

    lines: list[str] = [
        "# Retrieval Golden v1 Pilot — Manual Review",
        "",
        "Read-only human audit export for Step 4A.",
        "Dataset: `benchmarks/legal_v2/retrieval_golden_v1_pilot.jsonl`",
        "",
        "Human verdict values: `PASS` | `FIX` | `REJECT`",
        "",
        "## Summary",
        "",
        "| query_id | shortened query | polarity | expected blocks | alternatives | hard negatives | human verdict |",
        "|---|---|---|---:|---:|---:|---|",
    ]

    for item in items:
        lines.append(
            "| {qid} | {query} | {polarity} | {expected} | {alts} | {hard} |  |".format(
                qid=_md_cell(item.query_id),
                query=_md_cell(_shorten(item.query, 72)),
                polarity="negative" if item.is_negative else "positive",
                expected=0 if item.is_negative else len(item.expected_block_ids),
                alts=0 if item.is_negative else len(item.accepted_alternative_block_ids),
                hard=0 if item.is_negative else len(item.hard_negative_block_ids),
            )
        )

    lines.extend(["", "---", ""])

    primary_count = 0
    alt_count = 0
    hard_count = 0
    candidate_count = 0

    for item in items:
        section, counts = _render_query_section(item, corpus)
        lines.append(section)
        lines.append("")
        primary_count += counts["primary"]
        alt_count += counts["alt"]
        hard_count += counts["hard"]
        candidate_count += counts["candidates"]

    markdown = "\n".join(lines).rstrip() + "\n"
    stats = ExportStats(
        query_sections=len(items),
        primary_blocks_rendered=primary_count,
        alternative_blocks_rendered=alt_count,
        hard_negative_blocks_rendered=hard_count,
        negative_candidate_blocks_rendered=candidate_count,
    )
    return markdown, stats


def _preflight_items(items: list[RetrievalGoldenItem], corpus: DevelopmentCorpus) -> None:
    seen: set[str] = set()
    for item in items:
        if not item.query_id.strip():
            raise ManualReviewExportError("query_id missing")
        if not item.query.strip():
            raise ManualReviewExportError(f"{item.query_id}: query text missing")
        if item.query_id in seen:
            raise ManualReviewExportError(f"duplicate query_id: {item.query_id}")
        seen.add(item.query_id)

        if item.is_negative:
            for candidate in item.inspected_negative_candidates:
                if candidate.block_id not in corpus.blocks_by_id:
                    raise ManualReviewExportError(
                        f"{item.query_id}: inspected candidate block missing: {candidate.block_id}"
                    )
            continue

        if not item.primary_expected_block_id:
            raise ManualReviewExportError(
                f"{item.query_id}: positive query has no primary expected block"
            )
        if item.primary_expected_block_id not in corpus.blocks_by_id:
            raise ManualReviewExportError(
                f"{item.query_id}: primary block missing: {item.primary_expected_block_id}"
            )

        expected = set(item.expected_block_ids)
        alternatives = set(item.accepted_alternative_block_ids)
        hard = set(item.hard_negative_block_ids)
        collision = hard & (expected | alternatives)
        if collision:
            raise ManualReviewExportError(
                f"{item.query_id}: hard-negative also expected/alternative: {sorted(collision)}"
            )

        for block_id in sorted(expected | alternatives | hard):
            if block_id not in corpus.blocks_by_id:
                raise ManualReviewExportError(
                    f"{item.query_id}: referenced block does not exist: {block_id}"
                )


def _render_query_section(
    item: RetrievalGoldenItem,
    corpus: DevelopmentCorpus,
) -> tuple[str, dict[str, int]]:
    lines: list[str] = [
        f"## Query: {item.query_id}",
        "",
        "### Metadata",
        "",
        f"- **query_id:** `{item.query_id}`",
        f"- **query:** {item.query}",
        f"- **query_type:** `{item.query_type}`",
        f"- **difficulty:** `{item.difficulty}`",
        f"- **court:** `{item.court or 'null'}`",
        f"- **source_document_id:** `{item.source_document_id or 'null'}`",
        f"- **split:** `{item.split}`",
        f"- **is_negative:** `{str(item.is_negative).lower()}`",
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
    ]

    counts = {"primary": 0, "alt": 0, "hard": 0, "candidates": 0}

    if item.is_negative:
        lines.extend(
            [
                "### Negative rationale",
                "",
                item.negative_rationale or "_missing_",
                "",
                "### Inspected negative candidates",
                "",
            ]
        )
        if not item.inspected_negative_candidates:
            lines.append("_No inspected candidates listed._")
            lines.append("")
        for candidate in item.inspected_negative_candidates:
            block = corpus.blocks_by_id[candidate.block_id]
            score = (
                "n/a"
                if candidate.overlap_score is None
                else f"{candidate.overlap_score:.4f}"
            )
            lines.extend(
                [
                    f"#### Candidate rank {candidate.rank}",
                    "",
                    f"- **document_id:** `{candidate.document_id}`",
                    f"- **block_id:** `{candidate.block_id}`",
                    f"- **overlap_score:** `{score}`",
                    f"- **rejection_reason:** {candidate.rejection_reason}",
                    "",
                    "##### Complete candidate block text",
                    "",
                    _fenced(block.raw_text),
                    "",
                ]
            )
            counts["candidates"] += 1
        return "\n".join(lines).rstrip(), counts

    primary = corpus.blocks_by_id[item.primary_expected_block_id or ""]
    lines.extend(
        [
            "### Primary expected canonical block",
            "",
            f"- **block_id:** `{primary.block_id}`",
            "",
            "#### Complete primary block text",
            "",
            _fenced(primary.raw_text),
            "",
            "### Evidence excerpt",
            "",
            _fenced(item.evidence_excerpt or ""),
            "",
            "### Expected blocks",
            "",
        ]
    )
    counts["primary"] += 1

    for block_id in item.expected_block_ids:
        block = corpus.blocks_by_id[block_id]
        role = "primary" if block_id == item.primary_expected_block_id else "expected"
        lines.extend(_render_block(block, role=role))

    lines.extend(["### Accepted alternative blocks", ""])
    if not item.accepted_alternative_block_ids:
        lines.extend(["_None._", ""])
    else:
        for block_id in item.accepted_alternative_block_ids:
            block = corpus.blocks_by_id[block_id]
            lines.extend(_render_block(block, role="accepted_alternative"))
            counts["alt"] += 1

    lines.extend(["### Hard-negative blocks", ""])
    if not item.hard_negative_block_ids:
        lines.extend(["_None._", ""])
    else:
        for block_id in item.hard_negative_block_ids:
            block = corpus.blocks_by_id[block_id]
            lines.extend(_render_block(block, role="hard_negative"))
            counts["hard"] += 1

    lines.extend(
        [
            "### Grounding note",
            "",
            item.grounding_note or "_None._",
            "",
        ]
    )
    return "\n".join(lines).rstrip(), counts


def _render_block(block: CanonicalBlock, *, role: str) -> list[str]:
    return [
        f"#### Block (`{role}`)",
        "",
        f"- **block_id:** `{block.block_id}`",
        f"- **document_id:** `{block.document_id}`",
        f"- **block_index:** `{block.block_index}`",
        f"- **primary_class:** `{block.primary_class}`",
        "",
        "##### Complete block text",
        "",
        _fenced(block.raw_text),
        "",
    ]


def _validate_exported_markdown(markdown: str, items: list[RetrievalGoldenItem]) -> None:
    sections = _SECTION_RE.findall(markdown)
    if len(sections) != EXPECTED_QUERY_COUNT:
        raise ManualReviewExportError(
            f"output contains {len(sections)} query sections, expected {EXPECTED_QUERY_COUNT}"
        )
    for item in items:
        heading = f"## Query: {item.query_id}"
        if heading not in markdown:
            raise ManualReviewExportError(f"missing query section for {item.query_id}")


def _shorten(value: str, width: int) -> str:
    text = " ".join(value.split())
    if len(text) <= width:
        return text
    return text[: max(0, width - 3)].rstrip() + "..."


def _md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _fenced(text: str) -> str:
    return "```text\n" + (text or "").rstrip() + "\n```"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ManualReviewExportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
