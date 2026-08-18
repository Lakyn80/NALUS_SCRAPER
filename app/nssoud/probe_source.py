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
from app.nssoud.form import summarize_findform

BASE = "https://vyhledavac.nssoud.cz"
USER_AGENT = "nalus-scraper/nssoud-probe (+https://vyhledavac.nssoud.cz/)"


def _js_hits(html: str) -> dict[str, bool]:
    lowered = html.lower()
    keys = (
        "findform",
        "setcondition",
        "myrestrrowscont",
        "zobrazenivysledkuvolba",
        "export",
        "btSubmit",
        "xmlhttprequest",
        "dxcfts",
    )
    return {key: key.lower() in lowered for key in keys}


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
        "findform": None,
        "js_behavior": {},
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
            page = {
                "url": str(response.url),
                "status_code": response.status_code,
                "bytes": len(html),
                "title": (soup.title.get_text(strip=True) if soup.title else None),
                "forms": forms[:20],
                "cookie_names": sorted(client.cookies.keys()),
            }
            if "formular=4" in str(response.url) or path.endswith("formular=4"):
                report["findform"] = summarize_findform(html)
                report["js_behavior"] = _js_hits(html)
                report["js_behavior"]["add_condition"] = "function AddCondition" in html
                report["js_behavior"]["more_rows_url"] = "/Home/MyResTRowsCont" in html
                report["js_behavior"]["csrf_cookie_present"] = any(
                    "Antiforgery" in name for name in client.cookies.keys()
                )
                page["findform_summary"] = {
                    "present": bool(report["findform"] and report["findform"].get("present")),
                    "method": (report["findform"] or {}).get("method"),
                    "csrf_field": (report["findform"] or {}).get("csrf_field"),
                    "condition_count": (report["findform"] or {}).get("condition_count"),
                    "has_decision_date": (report["findform"] or {}).get("has_decision_date"),
                }
            report["pages"].append(page)
            report["form_actions"].extend(forms)
            for match in re.findall(
                r"""['"](/[^'"]*(?:Search|Document|Api|Detail|Export|Home|SetCondition|MyResTRowsCont)[^'"]*)['"]""",
                html,
                flags=re.I,
            ):
                report["url_candidates"].append(match)

    report["url_candidates"] = sorted(set(report["url_candidates"]))[:100]
    seen = set()
    unique_forms = []
    for form in report["form_actions"]:
        key = (form.get("action"), form.get("method"), form.get("id"))
        if key in seen:
            continue
        seen.add(key)
        unique_forms.append(form)
    report["form_actions"] = unique_forms[:50]
    report["notes"].extend(
        [
            "Primary source is vyhledavac.nssoud.cz. Search is POST form#findform to the current Index URL.",
            "SetCondition is AJAX POST that injects extra condition rows; default form already has datumvydanirozhodnuti.",
            "MyResTRowsCont is POST infinite-scroll pagination (pageNum, vyhledavaciPodminky, zobrazeniVysledkuId).",
            "Detail URLs are /DokumentDetail/Index/{id}; full text is /DokumentOriginal/Html/{id}.",
            "Do not store antiforgery cookie or token values in git.",
        ]
    )
    atomic_write_json(out, report)
    print(
        json.dumps(
            {
                "out": str(out),
                "pages": len(report["pages"]),
                "url_candidates": len(report["url_candidates"]),
                "findform_present": bool(report.get("findform") and report["findform"].get("present")),
                "has_decision_date": (report.get("findform") or {}).get("has_decision_date"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
