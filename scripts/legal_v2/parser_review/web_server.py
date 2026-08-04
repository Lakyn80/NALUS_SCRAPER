from __future__ import annotations

import json
from collections.abc import Callable
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .full_export import DEFAULT_OUTPUT_DIR, JSON_NAME, MARKDOWN_NAME
from .models import DEFAULT_REVIEW_DIR
from .security import assert_local_bind
from .web_api import ReviewApi

STATIC_DIR = Path(__file__).resolve().parent / "static"
EXPORT_FILES = {
    f"/exports/{JSON_NAME}": (DEFAULT_OUTPUT_DIR / JSON_NAME, "application/json; charset=utf-8"),
    f"/exports/{MARKDOWN_NAME}": (DEFAULT_OUTPUT_DIR / MARKDOWN_NAME, "text/markdown; charset=utf-8"),
}


class ReviewRequestHandler(SimpleHTTPRequestHandler):
    api: ReviewApi

    def __init__(self, *args: Any, directory: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, directory=directory or str(STATIC_DIR), **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._send_api(lambda: self.api.get(parsed.path, parse_qs(parsed.query)))
            return
        if parsed.path in EXPORT_FILES:
            self._send_export(parsed.path)
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        self._send_api(lambda: self.api.post(parsed.path, payload))

    def _send_api(self, fn: Callable[[], tuple[int, dict[str, Any]]]) -> None:
        try:
            status, payload = fn()
        except Exception as exc:  # noqa: BLE001 - local review UI returns bounded diagnostic errors.
            status, payload = 400, {"error": str(exc)}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_export(self, path: str) -> None:
        file_path, content_type = EXPORT_FILES[path]
        if not file_path.exists():
            self.send_error(404, "Export file not found. Generate it with export_parser_v6_full_review.py")
            return
        body = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Disposition", f'attachment; filename="{file_path.name}"')
        self.end_headers()
        self.wfile.write(body)


def serve(*, host: str = "127.0.0.1", port: int = 8765, review_dir: Path = DEFAULT_REVIEW_DIR) -> None:
    assert_local_bind(host)
    ReviewRequestHandler.api = ReviewApi(review_dir)
    handler = partial(ReviewRequestHandler, directory=str(STATIC_DIR))
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Serving parser review UI at http://{host}:{port}/")
    server.serve_forever()
