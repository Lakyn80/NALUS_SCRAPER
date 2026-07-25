from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.audit import audit_documents, write_audit_report  # noqa: E402
from app.rag.legal_v2.sources import discover_source_documents  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a read-only Legal Retrieval v2 corpus parser audit.")
    parser.add_argument("--batches-dir", type=Path, default=PROJECT_ROOT / "batches")
    parser.add_argument(
        "--nsoud-chunks-path",
        type=Path,
        default=PROJECT_ROOT / "app/artifacts/nsoud/rag_ready/nsoud_chunks_2025_01_03.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/legal_v2/parse_audit")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    documents = discover_source_documents(
        batches_dir=args.batches_dir,
        nsoud_chunks_path=args.nsoud_chunks_path,
        limit=args.limit,
    )
    report = audit_documents(documents)
    json_path, markdown_path = write_audit_report(report, args.output_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    return 0 if report.summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
