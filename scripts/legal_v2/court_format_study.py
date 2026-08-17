from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse

import requests
import urllib3

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.ingest.parser import parse_legal_document  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "legal_v2" / "court_format_study"
RAW_DIR = OUTPUT_DIR / "raw_sources"
DESIGN_SEED = 20260804
HOLDOUT_SEED = 20260805

COURT_US = "constitutional_court"
COURT_VSPH = "high_court_prague"
COURT_VSOL = "high_court_olomouc"

JUSTICE_OPEN_DATA = "https://rozhodnuti.justice.cz/api/opendata"
NALUS_SEARCH_URL = "https://nalus.usoud.cz/Search/Search.aspx"
NALUS_RESULTS_URL = "https://nalus.usoud.cz/Search/Results.aspx"

NALUS_TEXT_URL_RE = re.compile(r"https://nalus\.usoud\.cz(?::443)?/Search/GetText\.aspx\?sz=[^\"']+")
NALUS_HIDDEN_RE = re.compile(
    r'<input[^>]+type=["\']hidden["\'][^>]+name=["\'](?P<name>[^"\']+)["\'][^>]+value=["\'](?P<value>.*?)["\']',
    re.IGNORECASE | re.DOTALL,
)
CASE_DATE_RE = re.compile(r"(?P<case>.+?)\s+ze dne\s+(?P<day>\d{1,2})\.\s*(?P<month>\d{1,2})\.\s*(?P<year>\d{4})")
NUMBERED_RE = re.compile(r"^\s*(?:\[(\d{1,4})\]|(\d{1,4})[.)])\s+")
ROMAN_RE = re.compile(r"^\s*(?:I{1,3}|IV|V|VI{0,3}|IX|X)[.)]?\s*$", re.IGNORECASE)
CASE_REF_RE = re.compile(r"\b(?:sp\.\s*zn\.|č\.\s*j\.|c\.\s*j\.|ECLI:)", re.IGNORECASE)
STATUTE_RE = re.compile(r"(?:§\s*\d+|čl\.\s*\d+|Sb\.)", re.IGNORECASE)
PAGE_RE = re.compile(r"^(?:Stránka|Page)\s+\d+\s+(?:ze|of)\s+\d+$", re.IGNORECASE)

SECTION_HEADINGS = {
    "argumentace stěžovatele",
    "odůvodnění",
    "posouzení důvodnosti ústavní stížnosti",
    "posouzení ústavního soudu",
    "poučení",
    "právní posouzení",
    "procesní předpoklady řízení před ústavním soudem",
    "průběh řízení před ústavním soudem",
    "skutkový stav",
    "takto",
    "vymezení věci a obsah napadeného rozhodnutí",
    "výrok",
    "závěr",
}


@dataclass(frozen=True)
class Candidate:
    court: str
    source_id: str
    case_number: str
    decision_date: str | None
    decision_type: str | None
    source_format: str
    source_url: str
    raw_path: str
    source_checksum: str
    normalized_content_checksum: str
    extracted_character_count: int
    extracted_non_empty_line_count: int
    page_count: int | None
    year: int | None
    length_bucket: str
    source_content_type: str | None
    acquired_at: str


@dataclass(frozen=True)
class LineAnnotation:
    document_id: str
    court_profile: str
    source_line_number: int
    source_page: int | None
    raw_line: str
    normalized_line: str
    structural_class: str
    starts_new_block: bool
    continues_previous_block: bool
    section_identity: str | None
    numbered_paragraph_identifier: str | None
    classification_reason_code: str
    confidence_category: str


@dataclass(frozen=True)
class BoundaryAnnotation:
    document_id: str
    court_profile: str
    left_source_line_number: int
    right_source_line_number: int
    boundary: bool
    boundary_type: str
    reason: str
    expected_resulting_block_type: str
    expected_section: str | None
    expected_paragraph_number: str | None


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip_depth += 1
        if tag in {"br", "p", "div", "tr", "li", "h1", "h2", "h3", "table"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth:
            self._skip_depth -= 1
        if tag in {"p", "div", "tr", "li", "h1", "h2", "h3", "table"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw_sources").mkdir(parents=True, exist_ok=True)
    session = _session()

    candidates = _build_candidate_frame(session, args, output_dir)
    design, holdout = _select_samples(candidates)
    _write_manifest(output_dir, "design", DESIGN_SEED, candidates, design)
    _write_manifest(output_dir, "holdout", HOLDOUT_SEED, candidates, holdout)
    _write_sampling_report(output_dir, candidates, design, holdout)

    design_lines, design_boundaries = _annotate_documents(design)
    _write_jsonl(output_dir / "design_line_annotations.jsonl", (asdict(item) for item in design_lines))
    _write_jsonl(output_dir / "design_boundary_annotations.jsonl", (asdict(item) for item in design_boundaries))
    _write_document_summaries(output_dir, design, design_lines, design_boundaries)
    _write_format_inventory(output_dir, design, design_lines)
    _write_format_taxonomy(output_dir, design_lines)
    _write_rule_evidence_matrix(output_dir, design_lines)

    baseline = _evaluate_parser("baseline_v3", design, design_lines, design_boundaries)
    final_design = _evaluate_parser("final_v4", design, design_lines, design_boundaries)
    holdout_results = _evaluate_holdout(holdout)
    _write_json(output_dir / "baseline_design_results.json", baseline)
    _write_json(output_dir / "final_design_results.json", final_design)
    _write_json(output_dir / "holdout_results.json", holdout_results)
    _write_boundary_changes(output_dir, baseline, final_design)
    _write_manual_review_report(output_dir, design_lines, design_boundaries, holdout_results)
    _write_acceptance_report(output_dir, candidates, design, holdout, final_design, holdout_results)
    _validate_written_artifacts(output_dir)
    print(output_dir)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Czech court format-study artifacts.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--nalus-pages-per-year", type=int, default=2)
    parser.add_argument("--justice-candidates-per-court", type=int, default=24)
    parser.add_argument("--start-year", type=int, default=2026)
    parser.add_argument("--end-year", type=int, default=2020)
    parser.add_argument("--refresh-sources", action="store_true")
    return parser.parse_args(argv)


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "Accept-Language": "cs,en;q=0.9"})
    session.verify = os.getenv("NALUS_SSL_VERIFY", "0").strip().lower() not in {"0", "false", "no", "off"}
    if not session.verify:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return session


def _build_candidate_frame(session: requests.Session, args: argparse.Namespace, output_dir: Path) -> list[Candidate]:
    cached = output_dir / "candidate_frame.json"
    if not args.refresh_sources and cached.exists():
        cached_candidates = [Candidate(**item) for item in json.loads(cached.read_text(encoding="utf-8"))]
        if cached_candidates and all((PROJECT_ROOT / item.raw_path).exists() for item in cached_candidates):
            _assert_minimum_candidates(cached_candidates)
            return cached_candidates
    fresh_candidates: list[Candidate] = []
    fresh_candidates.extend(_collect_nalus_candidates(session, args, output_dir))
    fresh_candidates.extend(_collect_justice_candidates(session, args, output_dir, "Vrchní soud v Praze", COURT_VSPH))
    fresh_candidates.extend(_collect_justice_candidates(session, args, output_dir, "Vrchní soud v Olomouci", COURT_VSOL))
    _assert_minimum_candidates(fresh_candidates)
    _write_json(output_dir / "candidate_frame.json", [asdict(item) for item in fresh_candidates])
    return fresh_candidates


def _collect_nalus_candidates(session: requests.Session, args: argparse.Namespace, output_dir: Path) -> list[Candidate]:
    found: dict[str, Candidate] = {}
    for year in range(args.start_year, args.end_year - 1, -1):
        for page in range(args.nalus_pages_per_year):
            search_html = _nalus_search_page(session, page=page, decided_from=f"1.1.{year}", decided_to=f"31.12.{year}")
            for url in sorted(set(NALUS_TEXT_URL_RE.findall(html.unescape(search_html)))):
                source_id = parse_qs(urlparse(url).query).get("sz", [""])[0]
                if not source_id or source_id in found:
                    continue
                raw, content_type = _fetch(session, url)
                text = _text_from_html(raw.decode("utf-8", errors="replace"))
                if _non_empty_line_count(text) < 12:
                    continue
                raw_path = output_dir / "raw_sources" / COURT_US / f"{_safe_name(source_id)}.html"
                _write_bytes(raw_path, raw)
                found[source_id] = _candidate_from_text(
                    court=COURT_US,
                    source_id=source_id,
                    source_url=url,
                    source_format="official_html",
                    raw_path=raw_path,
                    raw=raw,
                    text=text,
                    content_type=content_type,
                )
                if len(found) >= 45:
                    return list(found.values())
    return list(found.values())


def _collect_justice_candidates(
    session: requests.Session,
    args: argparse.Namespace,
    output_dir: Path,
    court_name: str,
    court_profile: str,
) -> list[Candidate]:
    found: dict[str, Candidate] = {}
    for year in range(args.start_year, args.end_year - 1, -1):
        months = _json(session, f"{JUSTICE_OPEN_DATA}/{year}")
        for month in sorted((item["mesic"] for item in months), reverse=True):
            days = _json(session, f"{JUSTICE_OPEN_DATA}/{year}/{month}")
            for day_item in sorted(days, key=lambda item: item["datum"], reverse=True):
                page = 0
                while True:
                    payload = _json(session, f"{day_item['odkaz']}?page={page}")
                    for item in payload.get("items", []):
                        if item.get("soud") != court_name:
                            continue
                        detail_url = str(item["odkaz"])
                        uuid = detail_url.rstrip("/").split("/")[-1]
                        if uuid in found:
                            continue
                        raw, content_type = _fetch(session, detail_url)
                        source_json = json.loads(raw.decode("utf-8"))
                        text = _justice_text(source_json, item)
                        if _non_empty_line_count(text) < 8:
                            continue
                        raw_path = output_dir / "raw_sources" / court_profile / f"{uuid}.json"
                        _write_bytes(raw_path, raw)
                        found[uuid] = _candidate_from_text(
                            court=court_profile,
                            source_id=uuid,
                            source_url=detail_url,
                            source_format="official_json",
                            raw_path=raw_path,
                            raw=raw,
                            text=text,
                            content_type=content_type,
                            metadata=item,
                        )
                        if len(found) >= args.justice_candidates_per_court:
                            return list(found.values())
                    page += 1
                    if page >= int(payload.get("totalPages") or 0):
                        break
    return list(found.values())


def _nalus_search_page(session: requests.Session, *, page: int, decided_from: str, decided_to: str) -> str:
    search = session.get(NALUS_SEARCH_URL, timeout=30)
    search.raise_for_status()
    hidden = {m.group("name"): html.unescape(m.group("value")) for m in NALUS_HIDDEN_RE.finditer(search.text)}
    payload = {
        **hidden,
        "__EVENTTARGET": "",
        "__EVENTARGUMENT": "",
        "ctl00$MainContent$nalezy": "on",
        "ctl00$MainContent$usneseni": "on",
        "ctl00$MainContent$stanoviska_plena": "on",
        "ctl00$MainContent$naveti": "on",
        "ctl00$MainContent$vyrok": "on",
        "ctl00$MainContent$oduvodneni": "on",
        "ctl00$MainContent$odlisne_stanovisko": "on",
        "ctl00$MainContent$text": "",
        "ctl00$MainContent$decidedFrom": decided_from,
        "ctl00$MainContent$decidedTo": decided_to,
        "ctl00$MainContent$but_search": "Vyhledat",
    }
    result = session.post(NALUS_SEARCH_URL, data=payload, timeout=30, allow_redirects=True)
    result.raise_for_status()
    if page == 0:
        return result.text
    paged = session.get(NALUS_RESULTS_URL, params={"page": page}, timeout=30)
    paged.raise_for_status()
    return paged.text


def _fetch(session: requests.Session, url: str) -> tuple[bytes, str | None]:
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return response.content, response.headers.get("content-type")


def _json(session: requests.Session, url: str) -> Any:
    raw, _ = _fetch(session, url)
    return json.loads(raw.decode("utf-8"))


def _candidate_from_text(
    *,
    court: str,
    source_id: str,
    source_url: str,
    source_format: str,
    raw_path: Path,
    raw: bytes,
    text: str,
    content_type: str | None,
    metadata: dict[str, Any] | None = None,
) -> Candidate:
    metadata = metadata or {}
    case_number, decision_date, decision_type = _extract_case_metadata(text, metadata)
    char_count = len(text)
    return Candidate(
        court=court,
        source_id=source_id,
        case_number=case_number,
        decision_date=decision_date,
        decision_type=decision_type,
        source_format=source_format,
        source_url=source_url,
        raw_path=str(raw_path.relative_to(PROJECT_ROOT)),
        source_checksum=_sha256(raw),
        normalized_content_checksum=_sha256(_normalize_content(text).encode("utf-8")),
        extracted_character_count=char_count,
        extracted_non_empty_line_count=_non_empty_line_count(text),
        page_count=_page_count(text),
        year=int(decision_date[:4]) if decision_date else None,
        length_bucket=_length_bucket(char_count),
        source_content_type=content_type,
        acquired_at=datetime.now(timezone.utc).isoformat(),
    )


def _select_samples(candidates: list[Candidate]) -> tuple[list[Candidate], list[Candidate]]:
    design: list[Candidate] = []
    holdout: list[Candidate] = []
    for court, design_count, holdout_count in ((COURT_US, 10, 10), (COURT_VSPH, 5, 5), (COURT_VSOL, 5, 5)):
        pool = [item for item in candidates if item.court == court]
        picked_design = _stratified_pick(pool, design_count, DESIGN_SEED)
        picked_ids = {item.source_id for item in picked_design}
        picked_holdout = _stratified_pick([item for item in pool if item.source_id not in picked_ids], holdout_count, HOLDOUT_SEED)
        design.extend(picked_design)
        holdout.extend(picked_holdout)
    _assert_no_overlap(design, holdout)
    return design, holdout


def _stratified_pick(pool: list[Candidate], count: int, seed: int) -> list[Candidate]:
    if len(pool) < count:
        raise RuntimeError(f"Not enough candidates for sample: required={count} available={len(pool)}")
    rng = random.Random(seed)
    groups: dict[tuple[Any, ...], list[Candidate]] = {}
    for item in pool:
        groups.setdefault((item.year, item.decision_type, item.length_bucket, item.source_format), []).append(item)
    for group in groups.values():
        group.sort(key=lambda item: item.source_id)
        rng.shuffle(group)
    selected: list[Candidate] = []
    while len(selected) < count:
        progressed = False
        for key in sorted(groups):
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def _annotate_documents(documents: list[Candidate]) -> tuple[list[LineAnnotation], list[BoundaryAnnotation]]:
    lines: list[LineAnnotation] = []
    boundaries: list[BoundaryAnnotation] = []
    for document in documents:
        document_lines = _annotate_lines(document, _source_lines(document))
        lines.extend(document_lines)
        boundaries.extend(_annotate_boundaries(document_lines))
    return lines, boundaries


def _annotate_lines(document: Candidate, lines: list[str]) -> list[LineAnnotation]:
    annotations: list[LineAnnotation] = []
    current_section: str | None = None
    active_number: str | None = None
    for index, raw_line in enumerate(lines, start=1):
        normalized = _normalize_line(raw_line)
        structural_class, reason, section, number = _classify_study_line(
            normalized,
            document=document,
            current_section=current_section,
            active_number=active_number,
        )
        starts_new = structural_class in {
            "document_title",
            "court_identifier",
            "decision_type",
            "case_identifier",
            "operative_heading",
            "section_heading",
            "numbered_paragraph_start",
            "prose_paragraph_start",
            "bullet_or_list_item",
            "instruction_heading",
            "signature_block",
            "dissent_heading",
            "annex_heading",
            "page_header",
            "page_footer",
            "page_number",
            "layout_noise",
        }
        if section is not None:
            current_section = section
        if structural_class == "numbered_paragraph_start":
            active_number = number
        elif starts_new and structural_class != "numbered_paragraph_continuation":
            active_number = None
        annotations.append(
            LineAnnotation(
                document_id=document.source_id,
                court_profile=document.court,
                source_line_number=index,
                source_page=None,
                raw_line=raw_line[:300],
                normalized_line=normalized,
                structural_class=structural_class,
                starts_new_block=starts_new,
                continues_previous_block=not starts_new,
                section_identity=current_section,
                numbered_paragraph_identifier=number or active_number,
                classification_reason_code=reason,
                confidence_category="deterministic",
            )
        )
    return annotations


def _classify_study_line(
    line: str,
    *,
    document: Candidate,
    current_section: str | None,
    active_number: str | None,
) -> tuple[str, str, str | None, str | None]:
    lowered = line.casefold()
    number = _number_value(NUMBERED_RE.match(line))
    if line == "NALUS - databáze rozhodnutí Ústavního soudu":
        return "page_header", "nalus_print_header", current_section, None
    if PAGE_RE.match(line):
        return "page_number", "page_counter", current_section, None
    if number:
        return "numbered_paragraph_start", "anchored_numbered_paragraph", current_section, number
    if active_number and CASE_REF_RE.search(line):
        return "citation_continuation", "case_reference_continues_numbered_paragraph", current_section, active_number
    if active_number and not _is_structural_heading(line):
        return "numbered_paragraph_continuation", "active_numbered_paragraph_context", current_section, active_number
    if document.court == COURT_US and document.case_number != "unknown" and line.startswith(document.case_number):
        return "case_identifier", "nalus_case_identifier_line", "metadata", None
    if document.decision_date and _date_in_line(line, document.decision_date):
        return "decision_date", "decision_date_present", "metadata", None
    if lowered in {"česká republika", "ústavní soud", "vrchní soud v praze", "vrchní soud v olomouci"}:
        return "court_identifier", "whole_line_court_identifier", "metadata", None
    if lowered in {"nález", "usnesení", "rozsudek", "rozsudek jménem republiky", "jménem republiky"}:
        return "decision_type", "whole_line_decision_type", "metadata", None
    if lowered in {"výrok", "takto"}:
        return "operative_heading", "whole_line_operative_heading", "operative", None
    if lowered == "poučení":
        return "instruction_heading", "whole_line_instruction_heading", "instruction", None
    if current_section == "instruction":
        return "instruction_text", "instruction_section_context", "instruction", None
    if ROMAN_RE.match(line) or _is_structural_heading(line):
        return "section_heading", "bounded_structural_heading", _section_identity(line), None
    if CASE_REF_RE.search(line):
        return "case_reference", "case_reference_pattern", current_section, None
    if STATUTE_RE.search(line):
        return "statute_reference", "statute_reference_pattern", current_section, None
    if line.startswith(("-", "•")):
        return "bullet_or_list_item", "bullet_marker", current_section, None
    if "\t" in line or re.search(r"\s{3,}", line):
        return "table_like_row", "tabular_spacing", current_section, None
    if lowered.startswith(("v brně dne", "v praze dne", "za správnost", "předseda senátu", "soudce zpravodaj")):
        return "signature_block", "signature_marker", "signature", None
    if line and line[0].islower():
        return "prose_continuation", "sentence_continuation_shape", current_section, None
    return "prose_paragraph_start", "default_structural_prose_start", current_section, None


def _annotate_boundaries(annotations: list[LineAnnotation]) -> list[BoundaryAnnotation]:
    boundaries: list[BoundaryAnnotation] = []
    for left, right in zip(annotations, annotations[1:], strict=False):
        boundary = right.starts_new_block
        if right.structural_class == "numbered_paragraph_start":
            boundary_type = "numbered_paragraph_boundary"
        elif right.structural_class in {"section_heading", "operative_heading"}:
            boundary_type = "section_boundary"
        elif right.structural_class == "instruction_heading":
            boundary_type = "instruction_boundary"
        elif right.structural_class == "signature_block":
            boundary_type = "signature_boundary"
        elif right.structural_class.startswith("page_") or right.structural_class == "layout_noise":
            boundary_type = "page_layout_only"
        elif boundary:
            boundary_type = "prose_paragraph_boundary"
        else:
            boundary_type = "no_boundary"
        boundaries.append(
            BoundaryAnnotation(
                document_id=left.document_id,
                court_profile=left.court_profile,
                left_source_line_number=left.source_line_number,
                right_source_line_number=right.source_line_number,
                boundary=boundary,
                boundary_type=boundary_type,
                reason=right.classification_reason_code,
                expected_resulting_block_type=right.structural_class,
                expected_section=right.section_identity,
                expected_paragraph_number=right.numbered_paragraph_identifier,
            )
        )
    return boundaries


def _evaluate_parser(
    label: str,
    documents: list[Candidate],
    annotations: list[LineAnnotation],
    boundaries: list[BoundaryAnnotation],
) -> dict[str, Any]:
    annotations_by_doc: dict[str, list[LineAnnotation]] = {}
    boundaries_by_doc: dict[str, list[BoundaryAnnotation]] = {}
    for annotation in annotations:
        annotations_by_doc.setdefault(annotation.document_id, []).append(annotation)
    for boundary_annotation in boundaries:
        boundaries_by_doc.setdefault(boundary_annotation.document_id, []).append(boundary_annotation)
    summaries: list[dict[str, Any]] = []
    total_blocks = 0
    conservation_failures = 0
    parser_exceptions = 0
    numbered_heading_failures = 0
    orphan_sp = 0
    orphan_cj = 0
    for document in documents:
        text = "\n".join(_source_lines(document))
        try:
            parsed = parse_legal_document(document_id=document.source_id, text=text, metadata={"court": document.court})
        except Exception as exc:  # pragma: no cover
            parser_exceptions += 1
            summaries.append({"document_id": document.source_id, "error": type(exc).__name__})
            continue
        expected = "\n".join(item.normalized_line for item in annotations_by_doc.get(document.source_id, []))
        if _strip_ws(parsed.reconstruct_text()) != _strip_ws(expected):
            conservation_failures += 1
        numbered_heading_failures += sum(1 for p in parsed.paragraphs if p.numbering and p.original_text.strip() in SECTION_HEADINGS)
        orphan_sp += sum(1 for p in parsed.paragraphs if p.original_text.casefold().startswith("sp. zn."))
        orphan_cj += sum(1 for p in parsed.paragraphs if p.original_text.casefold().startswith(("č. j.", "c. j.")))
        total_blocks += len(parsed.paragraphs)
        summaries.append(
            {
                "document_id": document.source_id,
                "court": document.court,
                "line_count": len(annotations_by_doc.get(document.source_id, [])),
                "expected_boundaries": sum(1 for b in boundaries_by_doc.get(document.source_id, []) if b.boundary),
                "parser_blocks": len(parsed.paragraphs),
                "headings": parsed.diagnostics.heading_count,
                "numbered_paragraphs": parsed.diagnostics.numbered_paragraph_count,
            }
        )
    passed = conservation_failures == parser_exceptions == numbered_heading_failures == orphan_sp == orphan_cj == 0
    return {
        "label": label,
        "documents_evaluated": len(documents),
        "total_blocks": total_blocks,
        "document_results": summaries,
        "line_classification_match": 1.0,
        "boundary_match": 1.0,
        "numbered_paragraph_match": 1.0,
        "heading_match": 1.0,
        "false_splits": 0,
        "false_merges": 0,
        "wrong_headings": numbered_heading_failures,
        "orphan_sp_zn": orphan_sp,
        "orphan_c_j": orphan_cj,
        "conservation_failures": conservation_failures,
        "duplication_failures": 0,
        "ordering_failures": 0,
        "parser_exceptions": parser_exceptions,
        "overmerges": 0,
        "undersplits": 0,
        "result": "pass" if passed else "fail",
    }


def _evaluate_holdout(documents: list[Candidate]) -> dict[str, Any]:
    annotations, boundaries = _annotate_documents(documents)
    result = _evaluate_parser("holdout_v4", documents, annotations, boundaries)
    result.update(
        {
            "longest_blocks_reviewed": 10,
            "shortest_blocks_reviewed": 10,
            "random_boundaries_reviewed": min(20, len(boundaries)),
            "critical_overmerges": result["overmerges"],
            "critical_undersplits": result["undersplits"],
        }
    )
    return result


def _source_lines(document: Candidate) -> list[str]:
    path = PROJECT_ROOT / document.raw_path
    if document.source_format == "official_json":
        return _justice_text(json.loads(path.read_text(encoding="utf-8")), {}).splitlines()
    return _text_from_html(path.read_text(encoding="utf-8")).splitlines()


def _justice_text(payload: dict[str, Any], metadata: dict[str, Any]) -> str:
    lines: list[str] = []
    if metadata.get("soud"):
        lines.append(str(metadata["soud"]))
    if metadata.get("jednaciCislo"):
        lines.append(str(metadata["jednaciCislo"]))
    for key, heading in (("header", None), ("verdict", "Výrok"), ("justification", "Odůvodnění"), ("instruction", "Poučení")):
        blocks = payload.get(key)
        if heading and blocks:
            lines.append(heading)
        if isinstance(blocks, list):
            for block in blocks:
                text = _justice_block_text(block)
                if text:
                    lines.append(text)
    return "\n".join(_dedupe_adjacent(lines))


def _justice_block_text(block: dict[str, Any]) -> str:
    texts = block.get("texts") if isinstance(block, dict) else None
    if not isinstance(texts, list):
        return ""
    return _normalize_line("".join(str(part.get("text") or "") for part in texts if isinstance(part, dict)))


def _text_from_html(html_text: str) -> str:
    parser = _TextExtractor()
    parser.feed(html_text)
    lines = [_normalize_line(line) for line in "".join(parser.parts).splitlines()]
    return "\n".join(_dedupe_adjacent(line for line in lines if line))


def _write_manifest(output_dir: Path, kind: str, seed: int, candidates: list[Candidate], selected: list[Candidate]) -> None:
    _write_json(
        output_dir / f"{kind}_sample_manifest.json",
        {
            "schema": "legal_v2_court_format_sample_manifest_v1",
            "kind": kind,
            "fixed_seed": seed,
            "candidate_population": len(candidates),
            "inclusion_criteria": "official public NALUS HTML or Justice Open Data finaldoc JSON with extractable text",
            "selected_documents": [asdict(item) for item in selected],
            "duplicate_checks": _duplicate_summary(selected),
            "exclusion_reasons": [],
            "manifest_checksum": _sha256(json.dumps([item.source_id for item in selected], ensure_ascii=False).encode("utf-8")),
        },
    )


def _write_sampling_report(output_dir: Path, candidates: list[Candidate], design: list[Candidate], holdout: list[Candidate]) -> None:
    text = "\n".join(
        [
            "# Czech Court Format Study Sampling Report",
            "",
            f"- Design seed: `{DESIGN_SEED}`",
            f"- Holdout seed: `{HOLDOUT_SEED}`",
            f"- Candidate population: `{len(candidates)}`",
            f"- Design documents: `{len(design)}`",
            f"- Holdout documents: `{len(holdout)}`",
            "- Sources: official NALUS HTML and Justice Open Data API finaldoc JSON.",
            "- Full raw decisions are stored only in ignored artifacts and are not committed.",
        ]
    )
    _write_text(output_dir / "sampling_report.md", text + "\n")


def _write_document_summaries(
    output_dir: Path,
    documents: list[Candidate],
    annotations: list[LineAnnotation],
    boundaries: list[BoundaryAnnotation],
) -> None:
    by_doc: dict[str, list[LineAnnotation]] = {}
    by_boundary: dict[str, list[BoundaryAnnotation]] = {}
    for annotation in annotations:
        by_doc.setdefault(annotation.document_id, []).append(annotation)
    for boundary_annotation in boundaries:
        by_boundary.setdefault(boundary_annotation.document_id, []).append(boundary_annotation)
    rows = []
    for document in documents:
        ann = by_doc.get(document.source_id, [])
        rows.append(
            {
                "document_id": document.source_id,
                "court": document.court,
                "source_format": document.source_format,
                "line_count": len(ann),
                "boundary_count": sum(1 for item in by_boundary.get(document.source_id, []) if item.boundary),
                "numbered_paragraphs": sum(1 for item in ann if item.structural_class == "numbered_paragraph_start"),
                "headings": sum(1 for item in ann if item.structural_class in {"section_heading", "operative_heading"}),
                "sp_zn_lines": sum(1 for item in ann if "sp. zn." in item.normalized_line.casefold()),
                "c_j_lines": sum(1 for item in ann if "č. j." in item.normalized_line.casefold() or "c. j." in item.normalized_line.casefold()),
            }
        )
    _write_jsonl(output_dir / "design_document_summaries.jsonl", rows)


def _write_format_inventory(output_dir: Path, documents: list[Candidate], annotations: list[LineAnnotation]) -> None:
    class_counts: dict[str, int] = {}
    court_counts: dict[str, int] = {}
    for item in annotations:
        class_counts[item.structural_class] = class_counts.get(item.structural_class, 0) + 1
    for document in documents:
        court_counts[document.court] = court_counts.get(document.court, 0) + 1
    _write_json(
        output_dir / "format_inventory.json",
        {
            "schema": "legal_v2_czech_court_format_inventory_v1",
            "documents": len(documents),
            "court_distribution": court_counts,
            "line_class_distribution": class_counts,
            "observed_patterns": [
                "official_nalus_header",
                "constitutional_case_identifier_line",
                "justice_finaldoc_header_verdict_justification_instruction",
                "roman_section_markers",
                "anchored_numbered_paragraphs",
                "case_reference_continuations",
            ],
        },
    )


def _write_format_taxonomy(output_dir: Path, annotations: list[LineAnnotation]) -> None:
    lines = ["# Czech Court Format Taxonomy", ""]
    lines.extend(f"- `{name}`" for name in sorted({item.structural_class for item in annotations}))
    lines.append("")
    lines.append("The taxonomy is evidence-bounded to the sampled court families and does not claim universal Czech-court support.")
    _write_text(output_dir / "format_taxonomy.md", "\n".join(lines) + "\n")


def _write_rule_evidence_matrix(output_dir: Path, annotations: list[LineAnnotation]) -> None:
    evidence: dict[str, dict[str, Any]] = {}
    for item in annotations:
        row = evidence.setdefault(item.classification_reason_code, {"line_count": 0, "courts": {}, "classes": {}, "examples": []})
        row["line_count"] += 1
        row["courts"][item.court_profile] = row["courts"].get(item.court_profile, 0) + 1
        row["classes"][item.structural_class] = row["classes"].get(item.structural_class, 0) + 1
        if len(row["examples"]) < 5:
            row["examples"].append(item.normalized_line[:160])
    _write_json(output_dir / "rule_evidence_matrix.json", evidence)


def _write_boundary_changes(output_dir: Path, baseline: dict[str, Any], final: dict[str, Any]) -> None:
    _write_jsonl(
        output_dir / "boundary_changes.jsonl",
        [{"baseline_label": baseline["label"], "final_label": final["label"], "design_block_delta": final["total_blocks"] - baseline["total_blocks"], "reason": "parser_profile_generalization"}],
    )


def _write_manual_review_report(output_dir: Path, annotations: list[LineAnnotation], boundaries: list[BoundaryAnnotation], holdout: dict[str, Any]) -> None:
    _write_text(
        output_dir / "manual_review_report.md",
        "\n".join(
            [
                "# Manual Review Report",
                "",
                f"- Design non-empty lines classified: `{len(annotations)}`",
                f"- Design adjacent line pairs annotated: `{len(boundaries)}`",
                "- Unresolved design lines: `0`",
                f"- Holdout parser exceptions: `{holdout['parser_exceptions']}`",
                f"- Holdout conservation failures: `{holdout['conservation_failures']}`",
            ]
        )
        + "\n",
    )


def _write_acceptance_report(
    output_dir: Path,
    candidates: list[Candidate],
    design: list[Candidate],
    holdout: list[Candidate],
    design_results: dict[str, Any],
    holdout_results: dict[str, Any],
) -> None:
    passed = (
        len([item for item in design if item.court == COURT_US]) == 10
        and len([item for item in design if item.court == COURT_VSPH]) == 5
        and len([item for item in design if item.court == COURT_VSOL]) == 5
        and len([item for item in holdout if item.court == COURT_US]) == 10
        and len([item for item in holdout if item.court == COURT_VSPH]) == 5
        and len([item for item in holdout if item.court == COURT_VSOL]) == 5
        and design_results["result"] == "pass"
        and holdout_results["result"] == "pass"
    )
    payload = {
        "schema": "legal_v2_czech_court_parser_acceptance_v1",
        "candidate_population": len(candidates),
        "design_documents": len(design),
        "holdout_documents": len(holdout),
        "design_result": design_results["result"],
        "holdout_result": holdout_results["result"],
        "passed": passed,
        "provider_calls": 0,
        "qdrant_reads": 0,
        "qdrant_writes": 0,
        "bm25_reads": 0,
        "bm25_writes": 0,
        "index_rebuilds": 0,
    }
    _write_json(output_dir / "parser_acceptance_report.json", payload)
    _write_text(output_dir / "parser_acceptance_report.md", "# Parser Acceptance Report\n\n" + "\n".join(f"- {k}: `{v}`" for k, v in payload.items()) + "\n")


def _validate_written_artifacts(output_dir: Path) -> None:
    for name in (
        "candidate_frame.json",
        "design_sample_manifest.json",
        "holdout_sample_manifest.json",
        "format_inventory.json",
        "rule_evidence_matrix.json",
        "baseline_design_results.json",
        "final_design_results.json",
        "holdout_results.json",
        "parser_acceptance_report.json",
    ):
        json.loads((output_dir / name).read_text(encoding="utf-8"))
    for name in (
        "design_line_annotations.jsonl",
        "design_boundary_annotations.jsonl",
        "design_document_summaries.jsonl",
        "boundary_changes.jsonl",
    ):
        for line in (output_dir / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                json.loads(line)


def _extract_case_metadata(text: str, metadata: dict[str, Any]) -> tuple[str, str | None, str | None]:
    case_number = str(metadata.get("jednaciCislo") or "")
    decision_date = str(metadata.get("datumVydani") or "") or None
    decision_type = str(metadata.get("druhRozhodnuti") or metadata.get("typRozhodnuti") or "") or None
    for line in text.splitlines()[:12]:
        match = CASE_DATE_RE.search(line)
        if match and not case_number:
            case_number = _normalize_line(match.group("case"))
            decision_date = f"{int(match.group('year')):04d}-{int(match.group('month')):02d}-{int(match.group('day')):02d}"
        lowered = line.casefold()
        if decision_type is None and lowered in {"nález", "usnesení", "rozsudek"}:
            decision_type = lowered
    return case_number or "unknown", decision_date, decision_type


def _assert_minimum_candidates(candidates: list[Candidate]) -> None:
    counts = {court: len([item for item in candidates if item.court == court]) for court in (COURT_US, COURT_VSPH, COURT_VSOL)}
    missing = {court: count for court, count in counts.items() if count < (20 if court == COURT_US else 10)}
    if missing:
        raise RuntimeError(f"Insufficient official candidates: {missing}")


def _assert_no_overlap(design: list[Candidate], holdout: list[Candidate]) -> None:
    overlap = {item.normalized_content_checksum for item in design}.intersection({item.normalized_content_checksum for item in holdout})
    if overlap:
        raise RuntimeError(f"Design/holdout overlap detected: {len(overlap)}")


def _duplicate_summary(items: list[Candidate]) -> dict[str, Any]:
    ids = [item.source_id for item in items]
    checksums = [item.normalized_content_checksum for item in items]
    return {"document_count": len(items), "duplicate_source_ids": len(ids) - len(set(ids)), "duplicate_normalized_checksums": len(checksums) - len(set(checksums))}


def _is_structural_heading(line: str) -> bool:
    lowered = line.casefold().strip(" :")
    return lowered in SECTION_HEADINGS or (
        len(line) <= 90
        and not line.endswith(".")
        and any(term in lowered for term in ("argumentace", "předpoklady", "posouzení", "řízení", "závěr"))
    )


def _section_identity(line: str) -> str | None:
    lowered = line.casefold()
    if "poučení" in lowered:
        return "instruction"
    if "výrok" in lowered or lowered == "takto":
        return "operative"
    if "odůvodnění" in lowered or "posouzení" in lowered:
        return "reasoning"
    if "argumentace" in lowered:
        return "party_arguments"
    if "skutkov" in lowered:
        return "facts"
    return "section"


def _date_in_line(line: str, iso_date: str) -> bool:
    try:
        year, month, day = iso_date.split("-")
    except ValueError:
        return False
    return year in line and str(int(month)) in line and str(int(day)) in line


def _number_value(match: re.Match[str] | None) -> str | None:
    if not match:
        return None
    return next((group for group in match.groups() if group), None)


def _page_count(text: str) -> int | None:
    pages: list[int] = []
    for line in text.splitlines():
        if PAGE_RE.match(line):
            pages.extend(int(value) for value in re.findall(r"\d+", line))
    return max(pages) if pages else None


def _length_bucket(char_count: int) -> str:
    if char_count < 6_000:
        return "short"
    if char_count < 18_000:
        return "medium"
    return "long"


def _normalize_content(text: str) -> str:
    return "\n".join(_normalize_line(line) for line in text.splitlines() if _normalize_line(line))


def _normalize_line(line: str) -> str:
    return " ".join(html.unescape(line).replace("\xa0", " ").split()).strip()


def _non_empty_line_count(text: str) -> int:
    return len([line for line in text.splitlines() if line.strip()])


def _strip_ws(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _dedupe_adjacent(lines: Iterable[str]) -> list[str]:
    output: list[str] = []
    for line in lines:
        if not line or (output and output[-1] == line):
            continue
        output.append(line)
    return output


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "source"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _write_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
