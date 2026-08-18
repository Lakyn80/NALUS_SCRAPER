"""DXCFTS findform helpers for vyhledavac.nssoud.cz.

Derived from the live formular=4 HTML/JS (POST findform, infinite-scroll table,
MyResTRowsCont pagination). SetCondition is only used by the UI to add extra
condition rows; the default form already contains Datum vydání rozhodnutí.
"""

from __future__ import annotations

import codecs
import json
import re
from datetime import date
from typing import Any
from urllib.parse import urlencode, urljoin

from bs4 import BeautifulSoup

DATE_CONDITION_TECH = "datumvydanirozhodnuti"
FULLTEXT_CONDITION_TECH = "textDokumentu"
ECLI_CONDITION_TECH = "ecli"
DETAIL_PATH_RE = re.compile(r"/DokumentDetail/Index/(\d+)", re.I)
DOCUMENT_ID_RE = re.compile(r"/Dokument(?:Detail|Original)/(?:Index|Html|Text)/(\d+)", re.I)


def _scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "" if not value else _scalar(value[0])
    return str(value)


def serialize_findform(
    html: str,
    *,
    extra: dict[str, str] | None = None,
    include_submit: bool = True,
) -> list[tuple[str, str]]:
    """Build application/x-www-form-urlencoded pairs from form#findform.

    Skips catalog JSON (`ciselnikTreeData`) — live POST succeeded without it.
    Radios emit a single value (checked, else AND).
    """
    soup = BeautifulSoup(html, "lxml")
    form = soup.select_one("form#findform")
    if form is None:
        raise ValueError("NSS findform not found")
    pairs: list[tuple[str, str]] = []
    seen_radio: set[str] = set()
    for el in form.find_all(["input", "select", "textarea"]):
        name = el.get("name")
        if not name or el.has_attr("disabled"):
            continue
        name = str(name)
        if ".ciselnikTreeData" in name:
            continue
        tag = el.name
        typ = (el.get("type") or "text").lower()
        if tag == "input" and typ in {"button", "file", "image", "reset", "submit"}:
            continue
        if tag == "input" and typ == "radio":
            if name in seen_radio:
                continue
            checked = form.find("input", attrs={"name": name, "checked": True})
            chosen = checked or form.find("input", attrs={"name": name, "value": "AND"}) or el
            pairs.append((name, _scalar(chosen.get("value"))))
            seen_radio.add(name)
            continue
        if tag == "input" and typ == "checkbox":
            if el.has_attr("checked"):
                pairs.append((name, _scalar(el.get("value") or "on")))
            continue
        if tag == "select":
            selected = el.find("option", selected=True) or el.find("option")
            pairs.append((name, _scalar(selected.get("value") if selected else "")))
            continue
        if tag == "textarea":
            pairs.append((name, el.get_text() or ""))
            continue
        pairs.append((name, _scalar(el.get("value"))))
    if extra:
        for key, value in extra.items():
            replaced = False
            for index, (existing, _) in enumerate(pairs):
                if existing == key:
                    pairs[index] = (key, value)
                    replaced = True
                    break
            if not replaced:
                pairs.append((key, value))
    if include_submit and not any(name == "btSubmit" for name, _ in pairs):
        pairs.append(("btSubmit", ""))
    return [(str(name), str(value)) for name, value in pairs]


def encode_form_body(pairs: list[tuple[str, str]]) -> bytes:
    return urlencode(pairs, doseq=True).encode("utf-8")


def _condition_prefix_for_tech(pairs: list[tuple[str, str]], tech: str) -> str | None:
    for name, value in pairs:
        if not name.endswith(".TechnickyNazev"):
            continue
        if "vyhledavaciPodminkaHodnota" in name:
            continue
        if value == tech:
            return name[: -len(".TechnickyNazev")]
    return None


def apply_named_text(pairs: list[tuple[str, str]], *, tech: str, text: str) -> bool:
    prefix = _condition_prefix_for_tech(pairs, tech)
    if prefix is None:
        return False
    field = f"{prefix}.vyhledavaciPodminkaHodnota[0].HodnotaText"
    for index, (name, _) in enumerate(pairs):
        if name == field:
            pairs[index] = (name, text)
            return True
    pairs.append((field, text))
    return True


def format_nss_datetime(value: date, *, end_of_day: bool = False) -> str:
    clock = "23:59:59" if end_of_day else "00:00:00"
    return f"{value.day:02d}.{value.month:02d}.{value.year:04d} {clock}"


def apply_decision_date_window(
    pairs: list[tuple[str, str]],
    *,
    date_from: date | None,
    date_to: date | None,
) -> bool:
    """Set remote Datum vydání rozhodnutí (HodnotaDatumACasOd/Do)."""
    if date_from is None and date_to is None:
        return False
    prefix = _condition_prefix_for_tech(pairs, DATE_CONDITION_TECH)
    if prefix is None:
        return False
    mapping = {
        f"{prefix}.vyhledavaciPodminkaHodnota[0].HodnotaDatumACasOd": (
            format_nss_datetime(date_from, end_of_day=False) if date_from else ""
        ),
        f"{prefix}.vyhledavaciPodminkaHodnota[0].HodnotaDatumACasDo": (
            format_nss_datetime(date_to, end_of_day=True) if date_to else ""
        ),
    }
    for field, value in mapping.items():
        replaced = False
        for index, (name, _) in enumerate(pairs):
            if name == field:
                pairs[index] = (name, value)
                replaced = True
                break
        if not replaced:
            pairs.append((field, value))
    return True


def summarize_findform(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    form = soup.select_one("form#findform")
    if form is None:
        return {"present": False}
    conditions: list[dict[str, str | None]] = []
    seen: set[str] = set()
    for hid in form.find_all("input", attrs={"name": re.compile(r"TechnickyNazev$")}):
        name = hid.get("name") or ""
        if "vyhledavaciPodminkaHodnota" in name:
            continue
        tech = _scalar(hid.get("value"))
        if not tech or tech in seen:
            continue
        seen.add(tech)
        prefix = name.rsplit(".TechnickyNazev", 1)[0]
        label = form.find("input", attrs={"name": f"{prefix}.ZobrazovanyNazev"})
        dtype = form.find("input", attrs={"name": f"{prefix}.DatovyTyp"})
        conditions.append(
            {
                "technicky_nazev": tech,
                "label": _scalar(label.get("value") if label else None) or None,
                "datovy_typ": _scalar(dtype.get("value") if dtype else None) or None,
            }
        )
    token = form.find("input", attrs={"name": "__RequestVerificationToken"})
    return {
        "present": True,
        "id": form.get("id"),
        "method": (form.get("method") or "post").lower(),
        "action": form.get("action"),
        "csrf_field": "__RequestVerificationToken" if token is not None else None,
        "submit_name": "btSubmit",
        "condition_count": len(conditions),
        "conditions": conditions,
        "has_decision_date": any(item["technicky_nazev"] == DATE_CONDITION_TECH for item in conditions),
        "has_ecli": any(item["technicky_nazev"] == ECLI_CONDITION_TECH for item in conditions),
    }


def _decode_js_string(raw: str) -> str:
    return codecs.decode(raw, "unicode_escape")


def parse_infinite_scroll_state(html: str) -> dict[str, str | None]:
    params = re.search(r"var currParams = '([^']*)'", html)
    view = re.search(r"var currViewId = '([^']*)'", html)
    sort = re.search(r"var currSort = '([^']*)'", html)
    more = re.search(r"var moreRowsUrl = '([^']*)'", html)
    return {
        "vyhledavaci_podminky": _decode_js_string(params.group(1)) if params else None,
        "zobrazeni_vysledku_id": view.group(1) if view else None,
        "result_order": _decode_js_string(sort.group(1)) if sort else None,
        "more_rows_url": more.group(1) if more else None,
    }


def date_filter_was_applied(state: dict[str, str | None]) -> bool:
    raw = state.get("vyhledavaci_podminky") or ""
    if DATE_CONDITION_TECH not in raw:
        return False
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return DATE_CONDITION_TECH in raw
    items = payload if isinstance(payload, list) else [payload]
    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item.get("TechnickyNazev") or "") != DATE_CONDITION_TECH:
            continue
        values = item.get("vyhledavaciPodminkaHodnota") or []
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, dict):
                continue
            if value.get("HodnotaDatumACasOd") or value.get("HodnotaDatumACasDo"):
                return True
    return False


def extract_site_total(html: str) -> int | None:
    text = BeautifulSoup(html, "lxml").get_text(" ", strip=True)
    match = re.search(r"(?:nalezeno|počet(?: záznamů)?|záznamů)\s*[:=]?\s*(\d[\d\s]{0,8})", text, flags=re.I)
    if not match:
        match = re.search(r"(\d+)\s+(?:záznam|výsled)", text, flags=re.I)
    if not match:
        return None
    return int(re.sub(r"\s+", "", match.group(1)))


def _row_document_id(row: Any) -> str | None:
    hidden = row.find("input", attrs={"name": re.compile(r"ZobrazeneVysledky\[\d+\]\.ID$")})
    if hidden and hidden.get("value"):
        return str(hidden.get("value")).strip()
    for anchor in row.find_all("a", href=True):
        match = DOCUMENT_ID_RE.search(anchor["href"])
        if match:
            return match.group(1)
    return None


def extract_result_links(html: str, base_url: str) -> tuple[list[dict[str, str]], int | None]:
    """Extract NSS decision rows from #tresults / infinite-scroll fragments."""
    soup = BeautifulSoup(html, "lxml")
    total = extract_site_total(html)
    links: list[dict[str, str]] = []
    seen: set[str] = set()
    rows = soup.select("table#tresults tbody tr") or soup.select("table.infinite-scroll tbody tr")
    if not rows:
        rows = soup.find_all("tr")
    for row in rows:
        doc_id = _row_document_id(row)
        if not doc_id:
            continue
        detail_path = f"/DokumentDetail/Index/{doc_id}"
        url = urljoin(base_url, detail_path)
        if url in seen:
            continue
        seen.add(url)
        cells = [re.sub(r"\s+", " ", td.get_text(" ", strip=True)).strip() for td in row.find_all("td")]
        decision_date = next((cell for cell in cells if re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{4}", cell)), "")
        case_number = ""
        for cell in cells:
            if re.search(r"\d+\s+[A-Za-zÁ-ž]{1,8}\s+\d+/\d{4}", cell):
                case_number = cell
                break
        links.append(
            {
                "url": url,
                "document_id": doc_id,
                "html_url": urljoin(base_url, f"/DokumentOriginal/Html/{doc_id}"),
                "label": case_number or doc_id,
                "case_number": case_number,
                "decision_date_raw": decision_date,
            }
        )
    if not links:
        for anchor in soup.find_all("a", href=True):
            match = DETAIL_PATH_RE.search(anchor["href"])
            if not match:
                continue
            url = urljoin(base_url, f"/DokumentDetail/Index/{match.group(1)}")
            if url in seen:
                continue
            seen.add(url)
            links.append(
                {
                    "url": url,
                    "document_id": match.group(1),
                    "html_url": urljoin(base_url, f"/DokumentOriginal/Html/{match.group(1)}"),
                    "label": re.sub(r"\s+", " ", anchor.get_text(" ", strip=True)).strip(),
                    "case_number": "",
                    "decision_date_raw": "",
                }
            )
    return links, total


def detail_field(html: str, field_id: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    node = soup.select_one(f'.detcard[data-field-id="{field_id}"] .det-textval')
    if node is None:
        node = soup.select_one(f'[data-field-id="{field_id}"].det-textval')
    if node is None:
        return ""
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()


def is_search_or_nav_url(url: str) -> bool:
    lowered = url.lower()
    if "/dokumentdetail/index/" in lowered or "/dokumentoriginal/" in lowered:
        return False
    return any(token in lowered for token in ("/home/index", "formular=", "/napoveda", "/identity/"))
