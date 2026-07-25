from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.adapters import LegalAdapterRegistry  # noqa: E402
from app.rag.legal_v2.chunking import build_hierarchical_chunks  # noqa: E402
from app.rag.legal_v2.sources import discover_source_documents  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a human-reviewable Legal v2 parser QA artifact.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "artifacts/legal_v2/parser_quality_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/legal_v2/parser_quality_gate")
    parser.add_argument("--limit", type=int, default=12)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    documents = discover_source_documents(limit=args.limit)
    registry = LegalAdapterRegistry()
    statuses = _load_statuses(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reviewed = []
    lines = ["# Legal Retrieval v2 parser quality gate", ""]
    for document in documents:
        parsed = registry.adapter_for(document.source).parse(document)
        chunked = build_hierarchical_chunks(parsed)
        status = statuses.get(document.document_id, "needs_review")
        reviewed.append({"document_id": document.document_id, "source": document.source, "review_status": status})
        lines.extend(
            [
                f"## {document.document_id}",
                "",
                f"- Source: `{document.source}`",
                f"- Review status: `{status}`",
                f"- Paragraphs: {len(parsed.paragraphs)}",
                f"- Child chunks: {len(chunked.child_chunks)}",
                f"- Parent windows: {len(chunked.parent_windows)}",
                "",
                "### Parsed Sections",
                "",
            ]
        )
        for paragraph in parsed.paragraphs[:80]:
            lines.append(
                f"- `{paragraph.paragraph_id}` `{paragraph.section_type.value}` "
                f"boilerplate={paragraph.is_boilerplate} citation={paragraph.is_citation_block}: "
                f"{_snippet(paragraph.normalized_text)}"
            )
        lines.extend(["", "### Child Chunks", ""])
        for chunk in chunked.child_chunks[:40]:
            lines.append(f"- `{chunk.chunk_id}` paragraphs={chunk.paragraph_ids} tokens={chunk.token_count}")
        lines.extend(["", "### Parent Windows", ""])
        for window in chunked.parent_windows[:40]:
            lines.append(f"- `{window.window_id}` paragraphs={window.paragraph_ids} tokens={window.token_count}")
        lines.append("")
    payload = {"summary": {"status": "needs_review", "reviewed_documents": len(reviewed)}, "documents": reviewed}
    (args.output_dir / "parser_quality_gate.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "parser_quality_gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_dir / "parser_quality_gate.md")
    return 0


def _load_statuses(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    statuses = {}
    for item in data.get("documents", []):
        status = str(item.get("review_status") or "needs_review")
        if status not in {"approved", "rejected", "needs_review"}:
            status = "needs_review"
        statuses[str(item.get("document_id"))] = status
    return statuses


def _snippet(text: str, limit: int = 260) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
