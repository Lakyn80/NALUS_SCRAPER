"""Fixture tests for NSS DXCFTS findform POST discovery."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from app.nssoud.form import (
    apply_decision_date_window,
    apply_named_text,
    date_filter_was_applied,
    encode_form_body,
    extract_result_links,
    parse_infinite_scroll_state,
    serialize_findform,
    summarize_findform,
)
from app.nssoud.scraper import parse_decision_detail

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "nssoud"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_summarize_and_serialize_findform_controls() -> None:
    html = _read("findform.html")
    summary = summarize_findform(html)
    assert summary["present"] is True
    assert summary["method"] == "post"
    assert summary["csrf_field"] == "__RequestVerificationToken"
    assert summary["has_decision_date"] is True
    pairs = serialize_findform(html)
    names = [name for name, _ in pairs]
    assert names.count("vyhledavaciSekce[1].vyhledavaciPodminka[0].VazbaKDotazu") == 1
    assert ("__RequestVerificationToken", "TOKEN") in pairs
    assert ("btSubmit", "") in pairs
    assert not any("ciselnikTreeData" in name for name, _ in pairs)


def test_constructs_remote_date_and_fulltext_payload() -> None:
    pairs = serialize_findform(_read("findform.html"))
    assert apply_decision_date_window(
        pairs,
        date_from=date(2024, 1, 1),
        date_to=date(2024, 1, 31),
    )
    as_dict = dict(pairs)
    assert (
        as_dict[
            "vyhledavaciSekce[1].vyhledavaciPodminka[0].vyhledavaciPodminkaHodnota[0].HodnotaDatumACasOd"
        ]
        == "01.01.2024 00:00:00"
    )
    assert (
        as_dict[
            "vyhledavaciSekce[1].vyhledavaciPodminka[0].vyhledavaciPodminkaHodnota[0].HodnotaDatumACasDo"
        ]
        == "31.01.2024 23:59:59"
    )
    assert apply_named_text(pairs, tech="textDokumentu", text="kasační stížnost")
    body = encode_form_body(pairs).decode("utf-8")
    assert "HodnotaDatumACasOd=01.01.2024" in body
    assert "btSubmit" in body


def test_extracts_dokument_detail_result_links() -> None:
    links, total = extract_result_links(_read("results_table.html"), "https://vyhledavac.nssoud.cz")
    assert total is None
    assert [item["document_id"] for item in links] == ["718182", "718340"]
    assert links[0]["url"] == "https://vyhledavac.nssoud.cz/DokumentDetail/Index/718182"
    assert links[0]["html_url"] == "https://vyhledavac.nssoud.cz/DokumentOriginal/Html/718182"
    assert "1 As 262/2023" in links[0]["case_number"]
    assert links[0]["decision_date_raw"] == "31.01.2024"


def test_parse_decision_detail_from_real_field_ids() -> None:
    record = parse_decision_detail(
        _read("detail.html"),
        "https://vyhledavac.nssoud.cz/DokumentDetail/Index/718182",
        full_text_html=_read("original_html.html"),
    )
    assert record is not None
    assert record["ecli"] == "ECLI:CZ:NSS:2024:1.AS.262.2023.19"
    assert record["case_number"] == "1 As 262/2023-19"
    assert record["decision_date"] == "2024-01-31"
    assert "Kasační stížnost se odmítá" in record["full_text"]
    assert record["canonical_id"].startswith("ECLI:CZ:NSS:")


def test_pagination_state_and_date_condition_from_inline_js() -> None:
    state = parse_infinite_scroll_state(_read("findform.html"))
    assert state["more_rows_url"] == "/Home/MyResTRowsCont"
    assert state["zobrazeni_vysledku_id"] == "1"
    assert "pageNum" not in (state["vyhledavaci_podminky"] or "")
    assert date_filter_was_applied(state) is True
    assert "zvhdt1.Hodnota DESC" in (state["result_order"] or "")
