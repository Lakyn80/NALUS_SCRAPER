#!/usr/bin/env python3
"""Probe vyhledavac.nssoud.cz — discovery for NSS scraper (staging only)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.court_staging.jsonl_store import atomic_write_json
from app.court_staging.paths import assert_safe_staging_path, default_staging_root, ensure_staging_tree

BASE = "https://vyhledavac.nssoud.cz"
USER_AGENT = "nalus-scraper/nssoud-probe (+https://vyhledavac.nssoud.cz/)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Probe report JSON path under court_staging.",
    )
    args = parser.parse_args()

    try:
        import httpx
        from bs4 import BeautifulSoup
    except ImportError as exc:
        print(f"Missing dependency: {exc}")
        return 1

    staging = ensure_staging_tree(default_staging_root())
    out = assert_safe_staging_path(
        args.out or (staging / "nss" / "historical" / "pilot" / "probe_report.json"),
        staging_root=staging,
    )

    headers = {"User-Agent": USER_AGENT}
    report: dict = {
        "probed_at": datetime.now(timezone.utc).isoformat(),
        "base_url": BASE,
        "pages": [],
        "url_candidates": [],
        "form_actions": [],
        "notes": [],
    }

    with httpx.Client(timeout=60.0, follow_redirects=True, headers=headers) as client:
        for path in ("/", "/Home/Index?formular=4", "/Napoveda"):
            url = BASE + path if path.startswith("/") else path
            try:
                response = client.get(url)
            except Exception as exc:  # pragma: no cover
                report["pages"].append({"url": url, "error": str(exc)})
                continue
            html = response.text
            soup = BeautifulSoup(html, "lxml")
            forms = []
            for form in soup.find_all("form"):
                forms.append(
                    {
                        "action": form.get("action"),
                        "method": form.get("method"),
                        "id": form.get("id"),
                    }
                )
            report["pages"].append(
                {
                    "url": str(response.url),
                    "status_code": response.status_code,
                    "bytes": len(html),
                    "title": (soup.title.get_text(strip=True) if soup.title else None),
                    "forms": forms[:20],
                }
            )
            report["form_actions"].extend(forms)
            for match in re.findall(
                r"""['"](/[^'"]*(?:Search|Document|Api|Detail|Export|Home)[^'"]*)['"]""",
                html,
                flags=re.I,
            ):
                report["url_candidates"].append(match)

    report["url_candidates"] = sorted(set(report["url_candidates"]))[:100]
    # Dedup forms
    seen = set()
    unique_forms = []
    for form in report["form_actions"]:
        key = (form.get("action"), form.get("method"), form.get("id"))
        if key in seen:
            continue
        seen.add(key)
        unique_forms.append(form)
    report["form_actions"] = unique_forms[:50]
    report["notes"].append(
        "Primary source is vyhledavac.nssoud.cz. Sbírka is not the full-corpus target."
    )
    atomic_write_json(out, report)
    print(json.dumps({"out": str(out), "pages": len(report["pages"]), "url_candidates": len(report["url_candidates"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
