from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

from .models import DEFAULT_REVIEW_DIR, read_jsonl
from .progress import apply_manual_status, compute_progress


def render_list(review_dir: Path) -> str:
    rows = read_jsonl(review_dir / "review_documents.jsonl")
    lines = ["No  Court                    Source ID                         Review ID"]
    for row in rows:
        lines.append(f"{int(row['review_number']):02d}  {row['court']:<24} {row['source_id']:<32} {row['document_id']}")
    return "\n".join(lines)


def render_status(review_dir: Path) -> str:
    progress = compute_progress(review_dir)
    lines = [
        "Snapshot ready",
        f"Manual review {progress['manual_review_status']}",
        f"Documents: {progress['document_count']}",
        f"Lines reviewed: {progress['line_reviewed']}/{progress['line_total']}",
        f"Boundaries reviewed: {progress['boundary_reviewed']}/{progress['boundary_total']}",
        f"Incomplete documents: {progress['incomplete_documents']}",
        f"Unresolved items: {progress['unresolved_items']}",
        "",
    ]
    for doc in progress["documents"]:
        lines.append(
            f"{doc['review_number']:02d} {doc['court']:<24} {doc['source_id']:<30} "
            f"L {doc['line_reviewed']}/{doc['line_total']} B {doc['boundary_reviewed']}/{doc['boundary_total']}"
        )
    return "\n".join(lines)


def render_view(review_dir: Path, document_id: str, view: str = "lines") -> str:
    document = _document(review_dir, document_id)
    if view.lower() == "summary":
        return json.dumps(document, ensure_ascii=False, indent=2)
    if view.lower() == "raw":
        return (review_dir / "documents" / document["document_id"] / "raw_numbered.txt").read_text(encoding="utf-8")
    if view.lower() == "blocks":
        return (review_dir / "documents" / document["document_id"] / "parser_blocks.txt").read_text(encoding="utf-8")
    if view.lower() == "boundaries":
        return _render_boundaries(review_dir, document["document_id"])
    return _render_lines(review_dir, document["document_id"])


def _render_lines(review_dir: Path, document_id: str) -> str:
    rows = [row for row in read_jsonl(review_dir / "review_lines.jsonl") if row["document_id"] == document_id]
    rows = apply_manual_status(review_dir, rows, "line")
    output = ["Line  Status      Parser class                    Previous class                  Text"]
    for row in rows:
        text = textwrap.shorten(row["raw_text"], width=96, placeholder=" ...")
        output.append(
            f"{int(row['raw_line_number']):05d} {row.get('manual_decision_status','pending'):<11} "
            f"{str(row.get('parser_proposed_line_class')):<31} {str(row.get('previous_automated_annotation')):<31} {text}"
        )
    return "\n".join(output)


def _render_boundaries(review_dir: Path, document_id: str) -> str:
    rows = [row for row in read_jsonl(review_dir / "review_boundaries.jsonl") if row["document_id"] == document_id]
    rows = apply_manual_status(review_dir, rows, "boundary")
    output = ["Boundary ID                               Lines        Parser  Previous  Status"]
    for row in rows:
        output.append(
            f"{row['item_id'][-40:]:<40} {row['previous_line_number']:05d}->{row['next_line_number']:05d} "
            f"{str(row['parser_proposed_boundary']):<7} {str(row['previous_automated_boundary_annotation']):<9} "
            f"{row.get('manual_decision_status','pending')}"
        )
    return "\n".join(output)


def _document(review_dir: Path, document_id: str) -> dict[str, Any]:
    for row in read_jsonl(review_dir / "review_documents.jsonl"):
        if document_id in {str(row["document_id"]), str(row["source_id"]), str(row["review_number"])}:
            return row
    raise ValueError(f"Unknown document: {document_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render parser review views.")
    parser.add_argument("command", choices=["list", "status", "view"])
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--document-id")
    parser.add_argument("--view", default="lines")
    args = parser.parse_args(argv)
    if args.command == "list":
        print(render_list(args.review_dir))
    elif args.command == "status":
        print(render_status(args.review_dir))
    else:
        if not args.document_id:
            raise SystemExit("--document-id is required for view")
        print(render_view(args.review_dir, args.document_id, args.view))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
