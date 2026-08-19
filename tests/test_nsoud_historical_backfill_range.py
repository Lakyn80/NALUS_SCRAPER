from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import app.nsoud.run_historical_backfill as backfill


def test_iter_months_reverse_date_bounds() -> None:
    months = backfill.iter_months(
        year_from=None,
        year_to=None,
        date_from=backfill.date(2000, 6, 1),
        date_to=backfill.date(2019, 12, 31),
        reverse=True,
    )
    assert months[0] == (2019, 12)
    assert months[1] == (2019, 11)
    assert months[-1] == (2000, 6)
    assert (1999, 12) not in months
    assert (2000, 5) not in months


def test_main_resume_skips_ok_retries_failed_and_respects_scope(monkeypatch, tmp_path: Path) -> None:
    out_dir = tmp_path / "ns" / "historical"
    out_dir.mkdir(parents=True, exist_ok=True)
    written_manifest: dict[str, object] = {}
    called_months: list[tuple[int, int]] = []

    def fake_parse_args() -> Namespace:
        return Namespace(
            year_from=None,
            year_to=None,
            date_from=backfill.date(2000, 6, 1),
            date_to=backfill.date(2000, 8, 31),
            reverse=True,
            delay=0.0,
            max_pages=10,
            out_dir=out_dir,
            seed_jsonl=[],
            resume=True,
        )

    def fake_load_manifest(_path: Path) -> dict[str, object]:
        return {
            "version": 1,
            "months": {
                "1999-12": {"status": "failed"},  # obsolete/out-of-scope
                "2000-08": {"status": "ok"},  # should be skipped by --resume
                "2000-07": {"status": "failed"},  # should be retried
                "2000-06": {"status": "failed"},  # should be retried
            },
        }

    def fake_run_month(*, year: int, month: int, out_dir: Path, delay: float, max_pages: int) -> backfill.MonthResult:
        called_months.append((year, month))
        return backfill.MonthResult(
            year=year,
            month=month,
            date_from=f"{year:04d}-{month:02d}-01",
            date_to=f"{year:04d}-{month:02d}-28",
            output_path=str(out_dir / f"nsoud_{year:04d}_{month:02d}.jsonl"),
            completeness={"status": "ok"},
            status="ok",
            error_message=None,
        )

    def fake_atomic_write_json(_path: Path, payload: dict[str, object]) -> None:
        written_manifest.clear()
        written_manifest.update(payload)

    monkeypatch.setattr(backfill, "parse_args", fake_parse_args)
    monkeypatch.setattr(backfill, "default_staging_root", lambda: tmp_path)
    monkeypatch.setattr(backfill, "ensure_staging_tree", lambda p: tmp_path)
    monkeypatch.setattr(backfill, "assert_safe_staging_path", lambda p, staging_root=None: Path(p))
    monkeypatch.setattr(backfill, "seed_known", lambda out_dir, seeds: {})
    monkeypatch.setattr(backfill, "load_manifest", fake_load_manifest)
    monkeypatch.setattr(backfill, "run_month", fake_run_month)
    monkeypatch.setattr(backfill, "atomic_write_json", fake_atomic_write_json)

    rc = backfill.main()
    assert rc == 0
    # reverse range 2000-08,07,06 with resume skip for 08 -> run only 07 then 06
    assert called_months == [(2000, 7), (2000, 6)]
    # out-of-scope obsolete month should remain untouched in manifest map
    assert written_manifest["months"]["1999-12"]["status"] == "failed"
