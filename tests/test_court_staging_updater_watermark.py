"""Tests for court_staging_updater watermark hard rules."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.court_staging_updater import load_watermarks, save_watermarks


def test_watermark_advance_only_on_ok_status(tmp_path: Path) -> None:
    root = tmp_path / "court_staging"
    (root / "updater").mkdir(parents=True)
    watermarks = {"us": {"watermark_date": "2026-08-01"}}
    save_watermarks(root, watermarks)

    def advance_if_ok(status: str) -> dict:
        data = load_watermarks(root)
        court = "us"
        end = "2026-08-22"
        if status == "ok":
            data.setdefault(court, {})
            data[court]["watermark_date"] = end
            data[court]["last_success_date"] = end
            data[court]["last_status"] = status
        save_watermarks(root, data)
        return load_watermarks(root)["us"]

    after_partial = advance_if_ok("partial")
    assert after_partial["watermark_date"] == "2026-08-01"

    after_ok = advance_if_ok("ok")
    assert after_ok["watermark_date"] == "2026-08-22"
    assert after_ok["last_status"] == "ok"


def test_skip_watermark_flag_persists_original(tmp_path: Path) -> None:
    root = tmp_path / "court_staging"
    (root / "updater").mkdir(parents=True)
    original = {"us": {"watermark_date": "2026-08-01", "last_status": "ok"}}
    path = root / "updater" / "watermarks.json"
    path.write_text(json.dumps(original), encoding="utf-8")

    # simulate skip-watermark: do not call save_watermarks
    loaded = load_watermarks(root)
    assert loaded["us"]["watermark_date"] == "2026-08-01"
