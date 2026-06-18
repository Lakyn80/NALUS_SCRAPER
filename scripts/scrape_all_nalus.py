"""
Stahuje celý NALUS rok po roku a ingestuje do Qdrantu.

Strategie:
  - Prochazi roky 1993-2026 s filtrem decidedFrom/decidedTo
  - Prazdna textova query = vsechna rozhodnuti v danem roce
  - Kazdy rok ma max ~4000 rozhodnuti (~400 stranek) => server to neblokuje
  - Kazdy rok se ulozi do samostatneho souboru v batches/
  - Progress se ukla do manifestu => lze kdykoliv prerusit a navazat
  - Deduplika oproti existujicim batchim
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

BATCHES_DIR = PROJECT_ROOT / "batches"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoint.json"
FIRST_YEAR = 1993
CURRENT_YEAR = datetime.now().year

MAX_RETRIES = 3
RETRY_PAUSE = 90
PAGE_SLEEP = 3.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stahni cely NALUS rok po roku.")
    parser.add_argument("--resume", action="store_true", help="Navaz od posledniho nedokonceneho roku")
    parser.add_argument("--year", type=int, help="Stahni pouze tento rok")
    parser.add_argument("--from-year", type=int, default=FIRST_YEAR, help=f"Od roku (vychozi: {FIRST_YEAR})")
    parser.add_argument("--to-year", type=int, default=CURRENT_YEAR, help=f"Do roku (vychozi: {CURRENT_YEAR})")
    parser.add_argument("--no-ingest", action="store_true", help="Jen uloz JSON, neingestuj do Qdrantu")
    parser.add_argument("--dry-run", action="store_true", help="Jen zobraz co by se stahlo")
    parser.add_argument("--url", default="http://localhost:6333", help="Qdrant URL")
    parser.add_argument("--collection", default="nalus", help="Qdrant kolekce")
    return parser.parse_args()


def _atomic_write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, path)


def _load_manifest() -> dict:
    manifest_path = BATCHES_DIR / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"version": 1, "batches": []}


def _save_manifest(manifest: dict) -> None:
    _atomic_write_json(BATCHES_DIR / "manifest.json", manifest)


def _load_checkpoint() -> dict | None:
    if not CHECKPOINT_PATH.exists():
        return None

    try:
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
    except Exception as exc:
        print(f"[WARN] nelze nacist checkpoint: {exc}")
        return None

    year = checkpoint.get("year")
    page = checkpoint.get("page")
    if not isinstance(year, int) or not isinstance(page, int):
        print("[WARN] checkpoint ma neplatny format, ignoruji")
        return None

    return {"year": year, "page": max(1, page)}


def _save_checkpoint(year: int, page: int) -> None:
    _atomic_write_json(CHECKPOINT_PATH, {"year": year, "page": page})


def _completed_years(manifest: dict) -> set[int]:
    years = set()
    for entry in manifest.get("batches", []):
        if entry.get("complete") and entry.get("year"):
            years.add(int(entry["year"]))
    return years


def _find_resume_year(manifest: dict, from_year: int) -> int:
    completed = _completed_years(manifest)
    for year in range(from_year, CURRENT_YEAR + 1):
        if year not in completed:
            return year
    return CURRENT_YEAR + 1


def _partial_batch_path(year: int) -> Path:
    return BATCHES_DIR / f"year_{year}_partial.json"


def _load_partial_results(year: int) -> list:
    partial_path = _partial_batch_path(year)
    if not partial_path.exists():
        return []

    from app.models.search_result import NalusResult

    with open(partial_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [NalusResult(**item) for item in data]


def _save_partial_results(year: int, results: list) -> None:
    _atomic_write_json(_partial_batch_path(year), [asdict(r) for r in results])


def _delete_partial_results(year: int) -> None:
    partial_path = _partial_batch_path(year)
    if partial_path.exists():
        partial_path.unlink()


def _load_existing_ids() -> set[str]:
    ids: set[str] = set()
    for json_file in sorted(BATCHES_DIR.glob("*.json")):
        if json_file.name == "manifest.json":
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for d in data:
                if ecli := d.get("ecli"):
                    ids.add(ecli)
                if ref := d.get("case_reference"):
                    ids.add(ref)
                if rid := d.get("result_id"):
                    ids.add(str(rid))
        except Exception as exc:
            print(f"[WARN] nelze nacist {json_file.name}: {exc}")
    return ids


def _finalize_year(
    year: int,
    results_buffer: list,
    args: argparse.Namespace,
    manifest: dict,
) -> None:
    if results_buffer:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"year_{year}_{ts}.json"
        path = BATCHES_DIR / filename
        _atomic_write_json(path, [asdict(r) for r in results_buffer])
        print(f"[YEAR {year}] ulozeno {len(results_buffer)} rozhodnuti -> {filename}")

        entry = {
            "file": filename,
            "year": year,
            "doc_count": len(results_buffer),
            "complete": True,
            "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        }
        manifest["batches"] = [b for b in manifest.get("batches", []) if b.get("file") != filename]
        manifest["batches"].append(entry)
        _save_manifest(manifest)
        _delete_partial_results(year)

        if not args.no_ingest:
            try:
                _ingest_file(path, args.url, args.collection)
            except Exception as exc:
                print(f"[INGEST] rok {year} selhalo: {exc}")
    else:
        entry = {
            "year": year,
            "doc_count": 0,
            "complete": True,
            "note": "vse jiz existovalo jako duplikaty",
            "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        }
        manifest.setdefault("batches", []).append(entry)
        _save_manifest(manifest)
        _delete_partial_results(year)
        print(f"[YEAR {year}] vsechno duplikaty, rok oznacen jako hotovy")


def _scrape_year(
    year: int,
    existing_ids: set[str],
    args: argparse.Namespace,
    manifest: dict,
    start_page: int = 1,
) -> int:
    from app.crawler.extractor import extract_search_page
    from app.crawler.playwright_crawler import fetch_page_html
    from app.services.decision_service import enrich_results_with_text

    decided_from = f"1.1.{year}"
    decided_to = f"31.12.{year}"
    print(f"\n[YEAR {year}] stahuju {decided_from} - {decided_to}")

    results_buffer = _load_partial_results(year)
    current_page = max(1, start_page)
    total_new = 0
    total_skipped = 0
    consecutive_failures = 0

    if current_page > 1:
        try:
            first_html = fetch_page_html(
                query="",
                page=0,
                decided_from=decided_from,
                decided_to=decided_to,
            )
            first_search_page = extract_search_page(first_html, query="")
            if current_page > first_search_page.total_pages:
                print(
                    f"[YEAR {year}] checkpoint page={current_page} > total_pages={first_search_page.total_pages} -- preskakuji na dalsi rok"
                )
                _finalize_year(year, results_buffer, args, manifest)
                return total_new
        except Exception as exc:
            print(f"[YEAR {year}] nepodarilo se overit checkpoint page={current_page}: {exc}")

    while True:
        try:
            html = fetch_page_html(
                query="",
                page=current_page - 1,
                decided_from=decided_from,
                decided_to=decided_to,
            )
            search_page = extract_search_page(html, query="")
            consecutive_failures = 0
        except Exception as exc:
            consecutive_failures += 1
            print(f"[YEAR {year}] page_failed page={current_page} attempt={consecutive_failures}/{MAX_RETRIES} error={exc}")

            if consecutive_failures < MAX_RETRIES:
                time.sleep(5)
                continue

            print(f"[YEAR {year}] {MAX_RETRIES} selhani za sebou -- cekam {RETRY_PAUSE}s")
            time.sleep(RETRY_PAUSE)
            consecutive_failures = 0
            print(f"[YEAR {year}] opakuji stranku {current_page}")
            continue

        if current_page > search_page.total_pages:
            print(
                f"[YEAR {year}] checkpoint page={current_page} > total_pages={search_page.total_pages} -- preskakuji na dalsi rok"
            )
            break

        if not search_page.results:
            print(f"[YEAR {year}] prazdna stranka {current_page} -- konec")
            break

        to_enrich = []
        for result in search_page.results:
            uid = result.ecli or result.case_reference or str(result.result_id)
            if uid in existing_ids:
                total_skipped += 1
            else:
                to_enrich.append(result)

        if to_enrich:
            enriched = enrich_results_with_text(to_enrich)
            for result in enriched:
                uid = result.ecli or result.case_reference or str(result.result_id)
                existing_ids.add(uid)
                if result.ecli:
                    existing_ids.add(result.ecli)
                if result.case_reference:
                    existing_ids.add(result.case_reference)
                results_buffer.append(result)
                total_new += 1

        print(f"[YEAR {year}] page={current_page}/{search_page.total_pages} new={total_new} dup={total_skipped}")

        _save_partial_results(year, results_buffer)
        _save_checkpoint(year, current_page + 1)

        if current_page >= search_page.total_pages:
            break

        time.sleep(PAGE_SLEEP)
        current_page += 1

    _finalize_year(year, results_buffer, args, manifest)
    print(f"[YEAR {year}] hotovo: {total_new} novych, {total_skipped} duplikatu")
    return total_new


def _ingest_file(path: Path, qdrant_url: str, collection: str) -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    from app.data.runtime_corpus import build_runtime_corpus, load_results_from_json
    from app.rag.ingest.qdrant_ingest import QdrantIngestor, _MOCK_VECTOR_DIM

    class _Adapter:
        def __init__(self, c):
            self._c = c

        def upsert(self, collection_name, points):
            self._c.upsert(
                collection_name=collection_name,
                points=[PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points],
            )

        def retrieve(self, *, collection_name, ids, with_payload=True, with_vectors=False):
            return self._c.retrieve(
                collection_name=collection_name,
                ids=ids,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )

    client = QdrantClient(url=qdrant_url, timeout=60)
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=_MOCK_VECTOR_DIM, distance=Distance.COSINE),
        )

    results = load_results_from_json(path)
    corpus = build_runtime_corpus(results, source_label=str(path))
    ingestor = QdrantIngestor(_Adapter(client), collection_name=collection, batch_size=200)
    stats = ingestor.ingest_chunks(corpus.chunks)
    print(f"[INGEST] rok {path.stem}: inserted={stats.inserted_points} updated={stats.updated_points} skipped={stats.skipped_points}")


def main() -> int:
    args = _parse_args()
    BATCHES_DIR.mkdir(exist_ok=True)

    manifest = _load_manifest()
    completed = _completed_years(manifest)
    checkpoint = _load_checkpoint()

    if checkpoint is not None:
        while checkpoint["year"] in completed:
            checkpoint = {"year": checkpoint["year"] + 1, "page": 1}
        if checkpoint["year"] <= args.to_year and (not args.year or checkpoint["year"] == args.year):
            print(f"[RESUME] year={checkpoint['year']} page={checkpoint['page']}")
            _save_checkpoint(checkpoint["year"], checkpoint["page"])
        else:
            checkpoint = None

    if args.year:
        years = [args.year]
    else:
        from_year = args.from_year
        if args.resume:
            from_year = _find_resume_year(manifest, args.from_year)
            print(f"[INFO] --resume: navazuji od roku {from_year}")
        if checkpoint is not None:
            from_year = max(from_year, checkpoint["year"])
        years = list(range(from_year, args.to_year + 1))

    if args.dry_run:
        print(f"[DRY-RUN] roky ke stazeni: {[y for y in years if y not in completed]}")
        print(f"[DRY-RUN] jiz hotove roky: {sorted(completed & set(years))}")
        return 0

    existing_ids = _load_existing_ids()
    print(f"[INFO] nacteno {len(existing_ids)} existujicich ID z batches/")

    grand_total = 0

    for year in years:
        if year in completed:
            print(f"[YEAR {year}] jiz stazeno, preskakuji")
            continue

        start_page = 1
        if checkpoint is not None and year == checkpoint["year"]:
            start_page = checkpoint["page"]

        grand_total += _scrape_year(year, existing_ids, args, manifest, start_page=start_page)
        _save_checkpoint(year + 1, 1)
        checkpoint = None
        time.sleep(5)

    print(f"\n[INFO] Celkem novych rozhodnuti: {grand_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
