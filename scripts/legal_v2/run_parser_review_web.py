from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.legal_v2.parser_review.models import DEFAULT_REVIEW_DIR  # noqa: E402
from scripts.legal_v2.parser_review.snapshot import build_snapshot  # noqa: E402
from scripts.legal_v2.parser_review.web_server import serve  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local browser UI for the Legal v2 parser review.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--build-if-missing", action="store_true")
    args = parser.parse_args(argv)
    if args.build_if_missing and not (args.review_dir / "review_manifest.json").exists():
        build_snapshot(review_dir=args.review_dir)
    serve(host=args.host, port=args.port, review_dir=args.review_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
