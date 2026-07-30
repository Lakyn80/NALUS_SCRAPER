from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.source_inventory import build_source_inventory, write_source_inventory  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Legal Retrieval v2 source inventory artifact.")
    parser.add_argument("--batches-dir", type=Path, default=PROJECT_ROOT / "batches")
    parser.add_argument(
        "--nsoud-chunks-path",
        type=Path,
        default=PROJECT_ROOT / "app/artifacts/nsoud/rag_ready/nsoud_chunks_2025_01_03.jsonl",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/legal_v2/source_inventory_20260730.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/legal_v2/source_inventory_20260730.md",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_source_inventory(
        batches_dir=args.batches_dir,
        nsoud_chunks_path=args.nsoud_chunks_path,
    )
    json_path, markdown_path = write_source_inventory(report, args.json_output, args.markdown_output)
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
