#!/usr/bin/env python3
"""FAST vs PRECISE GPU quality benchmark (15 hard queries + full judgments).

Measurement/export only. Does not tune retrieval.
HARD STOP if PRECISE CrossEncoder is not on CUDA.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import statistics
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "artifacts" / "legal_v2" / "fast_vs_precise_gpu_15q"
QUERIES_PATH = OUT_ROOT / "queries.md"
LIMIT = 50
FULLTEXT_API = os.getenv(
    "NALUS_FULLTEXT_API_BASE",
    "http://nalus-scraper-parser-forms-api-1:8000",
)
HEADING_ONLY_RE = re.compile(
    r"^(?:Odůvodnění|Výrok|Nález|Usnesení|Argumentace stěžovatele|"
    r"Skutkový stav věci a průběh předchozího řízení|"
    r"I{1,3}|IV|VI{0,3}|IX|X)\.?:?$",
    re.IGNORECASE,
)
WARMUP_QUERY = (
    "právo na spravedlivý proces a přiměřenost zásahu soudu do práv účastníka řízení"
)


@dataclass
class QuerySpec:
    qid: str
    query: str
    legal_intent: str
    dimensions: list[str]


@dataclass
class ProfileRun:
    profile: str
    qid: str
    query: str
    wall_retrieval_ms: float
    reported_latency_ms: float | None
    export_ms: float
    retrieval_stage: str | None
    result_count: int
    results: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    reranker_device: str | None = None
    reranker_dtype: str | None = None
    error: str | None = None
    heading_only_count: int = 0
    missing_best_passage: int = 0
    full_ok: int = 0
    full_missing: int = 0


def _configure_env() -> None:
    os.environ.setdefault("QDRANT_URL", "http://nalus-scraper-qdrant-1:6333")
    os.environ["NALUS_LEGAL_V2_CASE_SIMILARITY_ENABLED"] = "1"
    os.environ["NALUS_LEGAL_V2_SEARCH_ENABLED"] = "1"
    os.environ["NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED"] = "1"
    os.environ["NALUS_LEGAL_V2_CE_DEVICE"] = "cuda"
    os.environ.setdefault("NALUS_LEGAL_V2_CE_ALLOW_DOWNLOAD", "0")
    os.environ.setdefault("NALUS_LEGAL_V2_CE_BATCH_SIZE", "16")
    os.environ.setdefault("EMBEDDING_DEVICE", "cpu")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("EMBEDDING_LOCAL_FILES_ONLY", "1")
    os.environ.setdefault("NALUS_LEGAL_V2_MAX_RESULT_LIMIT", "50")
    os.environ.setdefault(
        "EMBEDDING_MODEL_NAME",
        "/root/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/"
        "5617a9f61b028005a4858fdac845db406aefb181",
    )


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(ROOT),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def _parse_queries(path: Path) -> list[QuerySpec]:
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"(?m)^## (Q\d+)\s*$", text)
    specs: list[QuerySpec] = []
    # blocks: [preamble, Q01, body, Q02, body, ...]
    for i in range(1, len(blocks), 2):
        qid = blocks[i].strip()
        body = blocks[i + 1]
        qm = re.search(r"### Query\n(.+?)(?=\n### |\Z)", body, re.S)
        im = re.search(r"### Legal intent\n(.+?)(?=\n### |\Z)", body, re.S)
        dm = re.search(r"### Expected relevance dimensions\n(.+?)(?=\n## |\Z)", body, re.S)
        if not qm:
            raise SystemExit(f"missing Query for {qid}")
        dims: list[str] = []
        if dm:
            for line in dm.group(1).splitlines():
                line = line.strip()
                if line.startswith("- "):
                    dims.append(line[2:].strip())
        specs.append(
            QuerySpec(
                qid=qid,
                query=" ".join(qm.group(1).strip().split()),
                legal_intent=(im.group(1).strip() if im else ""),
                dimensions=dims,
            )
        )
    if len(specs) != 15:
        raise SystemExit(f"expected 15 queries, got {len(specs)}")
    return specs


def _http_json(method: str, url: str, body: dict | None = None, timeout: float = 300.0) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {detail[:3000]}") from exc


def _md_escape(text: str) -> str:
    return text.replace("```", "``\u200b`")


def _fmt(v: Any) -> str:
    return "N/A" if v is None or v == "" else str(v)


def _is_heading_only(text: str) -> bool:
    return bool(HEADING_ONLY_RE.fullmatch((text or "").strip()))


def _norm_ecli(v: str) -> str:
    return str(v or "").strip().upper()


def _doc_to_dict(doc: Any) -> dict[str, Any]:
    passages = []
    for p in list(getattr(doc, "relevant_passages", None) or []):
        passages.append(
            {
                "text": getattr(p, "text", None),
                "chunk_id": getattr(p, "chunk_id", None),
                "section": getattr(p, "section", None),
                "score": getattr(p, "score", None),
                "rrf_rank": getattr(p, "rrf_rank", None),
                "dense_rank": getattr(p, "dense_rank", None),
                "bm25_rank": getattr(p, "bm25_rank", None),
            }
        )
    return {
        "rank": doc.rank,
        "document_id": doc.document_id,
        "ecli": doc.ecli,
        "court": doc.court,
        "case_number": doc.case_number,
        "decision_date": doc.decision_date,
        "document_type": doc.document_type,
        "score": doc.score,
        "dense_rank": doc.dense_rank,
        "bm25_rank": doc.bm25_rank,
        "rrf_score": doc.rrf_score,
        "stage1_rank": doc.stage1_rank,
        "stage1_score": doc.stage1_score,
        "ce_rank": doc.ce_rank,
        "ce_score": doc.ce_score,
        "relevant_passages": passages,
        "metadata": dict(doc.metadata or {}),
    }


def _best_passage(doc: dict[str, Any]) -> str:
    passages = list(doc.get("relevant_passages") or [])
    if not passages:
        return ""
    return str(passages[0].get("text") or "").strip()


def _verify_cuda_or_stop() -> dict[str, Any]:
    import torch

    available = bool(torch.cuda.is_available())
    name = torch.cuda.get_device_name(0) if available else None
    meta = {
        "cuda_available": available,
        "gpu_name": name,
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_count": int(torch.cuda.device_count()) if available else 0,
    }
    print(f"CUDA_CHECK {json.dumps(meta, ensure_ascii=False)}", flush=True)
    if not available:
        raise SystemExit("HARD STOP: torch.cuda.is_available() is False — refuse CPU PRECISE")
    return meta


def _assert_precise_on_cuda(diagnostics: dict[str, Any]) -> tuple[str, str | None]:
    rerank = diagnostics.get("rerank") if isinstance(diagnostics, dict) else None
    device = None
    dtype = None
    if isinstance(rerank, dict):
        device = rerank.get("reranker_device")
        dtype = rerank.get("dtype")
    device_s = str(device or "")
    if not device_s.lower().startswith("cuda"):
        raise SystemExit(
            f"HARD STOP: PRECISE reranker not on CUDA "
            f"(reranker_device={device!r}, diagnostics.rerank={rerank!r})"
        )
    return device_s, str(dtype) if dtype is not None else None


async def _warmup() -> dict[str, Any]:
    from app.rag.legal_v2.retrieve.case_similarity_search import (
        reset_case_similarity_stage1_runtime_for_tests,
        search_case_similarity_stage1,
        warmup_case_similarity_stage1_runtime,
    )

    reset_case_similarity_stage1_runtime_for_tests()
    print("WARMUP shared Stage1 (BGE-M3 + BM25)", flush=True)
    warm = await asyncio.to_thread(warmup_case_similarity_stage1_runtime)
    print(f"WARMUP stage1 done: {warm}", flush=True)

    print("WARMUP PRECISE (non-scored)", flush=True)
    t0 = time.perf_counter()
    result = await search_case_similarity_stage1(
        query=WARMUP_QUERY,
        limit=5,
        include_debug=False,
        retrieval_profile="precise",
    )
    wall = (time.perf_counter() - t0) * 1000.0
    device, dtype = _assert_precise_on_cuda(dict(result.diagnostics or {}))
    payload = {
        "warmup_query": WARMUP_QUERY,
        "wall_ms": wall,
        "retrieval_stage": result.retrieval_stage,
        "result_count": result.result_count,
        "reranker_device": device,
        "reranker_dtype": dtype,
        "rerank": (result.diagnostics or {}).get("rerank"),
        "stage1_warmup": warm,
    }
    print(f"WARMUP PRECISE OK device={device} dtype={dtype} wall_ms={wall:.1f}", flush=True)
    return payload


async def _search(profile: str, query: str) -> tuple[Any, float]:
    from app.rag.legal_v2.retrieve.case_similarity_search import (
        search_case_similarity_stage1,
    )

    t0 = time.perf_counter()
    result = await search_case_similarity_stage1(
        query=query,
        limit=LIMIT,
        include_debug=False,
        retrieval_profile=profile,
    )
    wall = (time.perf_counter() - t0) * 1000.0
    return result, wall


def _fetch_full_text(document_id: str) -> tuple[str, str, list[str]]:
    """Return (full_text_or_marker, chunk_count, diag_lines)."""
    encoded = urllib.parse.quote(document_id, safe="")
    url = f"{FULLTEXT_API.rstrip('/')}/api/rag/documents/{encoded}"
    try:
        full = _http_json("GET", url, None, timeout=300.0)
    except Exception as exc:  # noqa: BLE001
        return (
            "FULL_TEXT_NOT_AVAILABLE",
            "N/A",
            [f"- reason: full-document fetch failed: {type(exc).__name__}: {exc}", f"- source_endpoint: `{url}`"],
        )
    status = str(full.get("full_text_availability_status") or "")
    text = str(full.get("full_text") or "")
    diag = full.get("diagnostics") or {}
    chunk_count = "N/A"
    if isinstance(diag, dict) and diag.get("chunk_count") is not None:
        chunk_count = str(diag.get("chunk_count"))
    if status in {"available", "partial"} and text.strip():
        notes: list[str] = []
        if status == "partial":
            notes.append("- note: full_text_availability_status=`partial`")
        return _md_escape(text), chunk_count, notes
    return (
        "FULL_TEXT_NOT_AVAILABLE",
        chunk_count,
        [
            f"- full_text_availability_status: `{status or 'unknown'}`",
            f"- provenance_status: `{full.get('provenance_status')}`",
            f"- source_endpoint: `{url}`",
        ],
    )


def _relevance_pointers(full_text: str, dimensions: list[str]) -> dict[str, list[str]]:
    lowered = full_text.casefold()
    found: list[str] = []
    missing: list[str] = []
    for dim in dimensions:
        tokens = [t for t in re.split(r"[/,\s]+", dim.casefold()) if len(t) >= 4]
        hit = any(tok in lowered for tok in tokens) if tokens else False
        if hit:
            found.append(dim)
        else:
            missing.append(dim)
    pointers: list[str] = []
    for marker in (
        "Odůvodnění",
        "Výrok",
        "I.",
        "II.",
        "III.",
        "Nejlepší zájem",
        "spravedlivý proces",
        "proporcional",
        "Haagsk",
    ):
        idx = full_text.find(marker)
        if idx >= 0:
            snippet = " ".join(full_text[idx : idx + 160].split())
            pointers.append(f"near `{marker}` @ char {idx}: {snippet[:140]}…")
        if len(pointers) >= 4:
            break
    if not pointers and full_text.strip():
        pointers.append("inspect beginning of judgment + odůvodnění section")
    return {
        "relevant_legal_concepts_found": found,
        "potentially_irrelevant_or_missing_dimensions": missing,
        "important_sections_to_inspect": pointers,
    }


def _enrich_with_full_texts(
    run: ProfileRun,
    *,
    dimensions: list[str],
    stage: str | None,
) -> None:
    export_t0 = time.perf_counter()
    enriched: list[dict[str, Any]] = []
    for doc in run.results:
        document_id = str(doc.get("document_id") or doc.get("ecli") or "").strip()
        best = _best_passage(doc)
        if not best:
            run.missing_best_passage += 1
        elif _is_heading_only(best):
            run.heading_only_count += 1
        full_text = "FULL_TEXT_NOT_AVAILABLE"
        chunk_count = "N/A"
        diag_lines: list[str] = ["- reason: missing document_id"]
        if document_id:
            print(
                f"GET full [{run.profile} {run.qid} rank={doc.get('rank')}] {document_id}",
                flush=True,
            )
            full_text, chunk_count, diag_lines = _fetch_full_text(document_id)
        if full_text == "FULL_TEXT_NOT_AVAILABLE":
            run.full_missing += 1
            pointers = {
                "relevant_legal_concepts_found": [],
                "potentially_irrelevant_or_missing_dimensions": dimensions,
                "important_sections_to_inspect": ["FULL_TEXT_NOT_AVAILABLE"],
            }
        else:
            run.full_ok += 1
            pointers = _relevance_pointers(full_text, dimensions)
        enriched.append(
            {
                **doc,
                "best_passage": best,
                "full_text": full_text,
                "chunk_count": chunk_count,
                "full_diag_lines": diag_lines,
                "retrieval_stage": stage,
                "relevance_pointers": pointers,
            }
        )
    run.results = enriched
    run.export_ms = (time.perf_counter() - export_t0) * 1000.0


def _write_profile_md(path: Path, run: ProfileRun, query: str) -> None:
    lines = [
        f"# {run.profile.upper()} — {run.qid}",
        "",
        "## Query",
        "",
        query,
        "",
        "## Run metadata",
        "",
        f"- profile: `{run.profile}`",
        f"- qid: `{run.qid}`",
        f"- retrieval_stage: `{_fmt(run.retrieval_stage)}`",
        f"- requested_limit: `{LIMIT}`",
        f"- returned_result_count: `{run.result_count}`",
        f"- total_latency_ms: `{_fmt(run.reported_latency_ms)}` "
        f"(wall_retrieval_ms=`{run.wall_retrieval_ms:.1f}`)",
        f"- full_document_export_latency_ms: `{run.export_ms:.1f}`",
        f"- reranker_device: `{_fmt(run.reranker_device)}`",
        f"- reranker_dtype: `{_fmt(run.reranker_dtype)}`",
        f"- search_error: `{_fmt(run.error)}`",
        "",
        "---",
        "",
    ]
    for doc in run.results:
        collection = (run.diagnostics or {}).get("collection")
        lines.extend(
            [
                f"## Result {doc.get('rank')}",
                "",
                f"- Rank: {_fmt(doc.get('rank'))}",
                f"- Court: {_fmt(doc.get('court'))}",
                f"- ECLI: {_fmt(doc.get('ecli'))}",
                f"- Case number: {_fmt(doc.get('case_number'))}",
                f"- Date: {_fmt(doc.get('decision_date'))}",
                f"- Document ID: {_fmt(doc.get('document_id'))}",
                f"- Retrieval score: {_fmt(doc.get('score'))}",
                f"- Dense rank: {_fmt(doc.get('dense_rank'))}",
                f"- BM25 rank: {_fmt(doc.get('bm25_rank'))}",
                f"- RRF score: {_fmt(doc.get('rrf_score'))}",
                f"- CE score: {_fmt(doc.get('ce_score'))}",
                f"- Final rank: {_fmt(doc.get('rank'))}",
                f"- Final score: {_fmt(doc.get('score'))}",
                f"- Retrieval stage: {_fmt(doc.get('retrieval_stage') or run.retrieval_stage)}",
                f"- Source collection: `{_fmt(collection)}`",
                f"- Number of chunks: {_fmt(doc.get('chunk_count'))}",
                "",
                "### Best retrieved passage",
                "",
                doc.get("best_passage") or "_no passage returned_",
                "",
                "### Full judgment",
                "",
            ]
        )
        full = doc.get("full_text") or "FULL_TEXT_NOT_AVAILABLE"
        lines.append(full)
        lines.append("")
        if full == "FULL_TEXT_NOT_AVAILABLE":
            lines.append("Diagnostic metadata:")
            lines.append("")
            lines.extend(doc.get("full_diag_lines") or ["- reason: unknown"])
            lines.append("")
        lines.append("---")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _rank_map(run: ProfileRun) -> dict[str, int]:
    out: dict[str, int] = {}
    for doc in run.results:
        e = _norm_ecli(str(doc.get("ecli") or ""))
        if not e:
            continue
        try:
            out[e] = int(doc.get("rank"))
        except (TypeError, ValueError):
            continue
    return out


def _write_comparison(
    path: Path,
    *,
    q: QuerySpec,
    fast: ProfileRun,
    precise: ProfileRun,
) -> dict[str, Any]:
    f_map = _rank_map(fast)
    p_map = _rank_map(precise)
    f_top = [_norm_ecli(str(d.get("ecli") or "")) for d in fast.results[:10]]
    p_top = [_norm_ecli(str(d.get("ecli") or "")) for d in precise.results[:10]]
    shared = set(f_top) & set(p_top)
    entered = sorted(set(p_top) - set(f_top))
    left = sorted(set(f_top) - set(p_top))
    movements: list[dict[str, Any]] = []
    for ecli, f_rank in f_map.items():
        if ecli not in p_map:
            continue
        p_rank = p_map[ecli]
        movements.append(
            {
                "ecli": ecli,
                "fast_rank": f_rank,
                "precise_rank": p_rank,
                "delta": f_rank - p_rank,  # positive = improved toward 1
            }
        )
    movements.sort(key=lambda x: abs(x["delta"]), reverse=True)

    lines = [
        f"# Comparison — {q.qid}",
        "",
        "## Query",
        "",
        q.query,
        "",
        "### Legal intent",
        "",
        q.legal_intent,
        "",
        "## Run overview",
        "",
        "| Profile | Returned | Retrieval latency | Device | Stage |",
        "| ------- | -------: | ----------------: | ------ | ----- |",
        f"| FAST | {fast.result_count} | {fast.wall_retrieval_ms:.1f} ms "
        f"(diag `{_fmt(fast.reported_latency_ms)}`) | "
        f"`embedding={os.getenv('EMBEDDING_DEVICE','cpu')}` | `{_fmt(fast.retrieval_stage)}` |",
        f"| PRECISE | {precise.result_count} | {precise.wall_retrieval_ms:.1f} ms "
        f"(diag `{_fmt(precise.reported_latency_ms)}`) | "
        f"`{_fmt(precise.reranker_device)}` / dtype=`{_fmt(precise.reranker_dtype)}` | "
        f"`{_fmt(precise.retrieval_stage)}` |",
        "",
        "## Top 10 FAST",
        "",
        "```text",
    ]
    for d in fast.results[:10]:
        lines.append(
            f"{d.get('rank')} | {d.get('ecli')} | {d.get('court')} | {d.get('decision_date')}"
        )
    if not fast.results:
        lines.append("(no results)")
    lines += ["```", "", "## Top 10 PRECISE", "", "```text"]
    for d in precise.results[:10]:
        lines.append(
            f"{d.get('rank')} | {d.get('ecli')} | {d.get('court')} | {d.get('decision_date')}"
        )
    if not precise.results:
        lines.append("(no results)")
    lines += [
        "```",
        "",
        "## Top-10 overlap",
        "",
        f"- shared / 10: `{len(shared)} / 10`",
        f"- entered PRECISE top 10: {entered or []}",
        f"- left FAST top 10: {left or []}",
        "",
        "## Rank movement (shared documents)",
        "",
    ]
    if not movements:
        lines.append("_no shared documents_")
        lines.append("")
    else:
        lines.append("| ECLI | FAST | PRECISE | delta (toward 1) |")
        lines.append("| ---- | ---: | ------: | ---------------: |")
        for m in movements[:30]:
            sign = "+" if m["delta"] > 0 else ""
            lines.append(
                f"| {m['ecli']} | {m['fast_rank']} | {m['precise_rank']} | {sign}{m['delta']} |"
            )
        lines.append("")

    lines += [
        "## Best passage check",
        "",
        f"- FAST heading-only: `{fast.heading_only_count}`",
        f"- PRECISE heading-only: `{precise.heading_only_count}`",
        f"- FAST missing best passage: `{fast.missing_best_passage}`",
        f"- PRECISE missing best passage: `{precise.missing_best_passage}`",
        f"- FAST FULL_TEXT_NOT_AVAILABLE: `{fast.full_missing}`",
        f"- PRECISE FULL_TEXT_NOT_AVAILABLE: `{precise.full_missing}`",
        "",
        "## Manual quality-support (TOP 10 both profiles)",
        "",
    ]

    for profile_run in (fast, precise):
        for doc in profile_run.results[:10]:
            ptr = doc.get("relevance_pointers") or {}
            lines += [
                f"### {profile_run.profile.upper()} Rank {doc.get('rank')} — {doc.get('ecli')}",
                "",
                "#### Best passage",
                "",
                doc.get("best_passage") or "_no passage returned_",
                "",
                "#### Full-text relevance pointers",
                "",
                "- relevant legal concepts found in the judgment:",
            ]
            for item in ptr.get("relevant_legal_concepts_found") or []:
                lines.append(f"  - {item}")
            if not ptr.get("relevant_legal_concepts_found"):
                lines.append("  - (none matched by heuristic)")
            lines.append("- potentially irrelevant/mismatched concepts:")
            for item in ptr.get("potentially_irrelevant_or_missing_dimensions") or []:
                lines.append(f"  - {item}")
            if not ptr.get("potentially_irrelevant_or_missing_dimensions"):
                lines.append("  - (none)")
            lines.append("- important sections/paragraphs to inspect:")
            for item in ptr.get("important_sections_to_inspect") or []:
                lines.append(f"  - {item}")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "qid": q.qid,
        "top10_overlap": len(shared),
        "entered_precise_top10": entered,
        "left_fast_top10": left,
        "movements": movements,
        "fast_latency_ms": fast.wall_retrieval_ms,
        "precise_latency_ms": precise.wall_retrieval_ms,
        "fast_heading_only": fast.heading_only_count,
        "precise_heading_only": precise.heading_only_count,
        "fast_full_missing": fast.full_missing,
        "precise_full_missing": precise.full_missing,
        "fast_top10": f_top,
        "precise_top10": p_top,
        "fast_ranks": f_map,
        "precise_ranks": p_map,
    }


def _pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


def _write_summary(
    *,
    queries: list[QuerySpec],
    per_query: list[dict[str, Any]],
    fast_runs: dict[str, ProfileRun],
    precise_runs: dict[str, ProfileRun],
    cuda_meta: dict[str, Any],
    warmup: dict[str, Any],
) -> None:
    fast_lat = [float(x["fast_latency_ms"]) for x in per_query]
    prec_lat = [float(x["precise_latency_ms"]) for x in per_query]
    overlaps = [int(x["top10_overlap"]) for x in per_query]

    all_moves: list[dict[str, Any]] = []
    promotions_gt10_top10 = 0
    promotions_gt20_top10 = 0
    promotions_gt10_top5 = 0
    for item in per_query:
        for m in item["movements"]:
            all_moves.append({**m, "qid": item["qid"]})
            f_rank, p_rank = int(m["fast_rank"]), int(m["precise_rank"])
            if f_rank > 10 and p_rank <= 10:
                promotions_gt10_top10 += 1
            if f_rank > 20 and p_rank <= 10:
                promotions_gt20_top10 += 1
            if f_rank > 10 and p_rank <= 5:
                promotions_gt10_top5 += 1

    abs_moves = [abs(int(m["delta"])) for m in all_moves]
    up5 = sum(1 for m in all_moves if int(m["delta"]) >= 5)
    down5 = sum(1 for m in all_moves if int(m["delta"]) <= -5)
    largest_pos = sorted(all_moves, key=lambda m: int(m["delta"]), reverse=True)[:10]
    largest_neg = sorted(all_moves, key=lambda m: int(m["delta"]))[:10]

    heading_f = sum(r.heading_only_count for r in fast_runs.values())
    heading_p = sum(r.heading_only_count for r in precise_runs.values())
    miss_best_f = sum(r.missing_best_passage for r in fast_runs.values())
    miss_best_p = sum(r.missing_best_passage for r in precise_runs.values())
    full_miss_f = sum(r.full_missing for r in fast_runs.values())
    full_miss_p = sum(r.full_missing for r in precise_runs.values())
    full_ok_f = sum(r.full_ok for r in fast_runs.values())
    full_ok_p = sum(r.full_ok for r in precise_runs.values())

    lines = [
        "# FAST vs PRECISE GPU — benchmark summary",
        "",
        "## 0. Runtime / GPU verification",
        "",
        f"- cuda_available: `{cuda_meta.get('cuda_available')}`",
        f"- gpu_name: `{cuda_meta.get('gpu_name')}`",
        f"- torch_version: `{cuda_meta.get('torch_version')}`",
        f"- cuda_version: `{cuda_meta.get('cuda_version')}`",
        f"- warmup_reranker_device: `{warmup.get('reranker_device')}`",
        f"- warmup_reranker_dtype: `{warmup.get('reranker_dtype')}`",
        f"- warmup_wall_ms: `{warmup.get('wall_ms')}`",
        "",
        "## 1. Runtime (retrieval only; warmup excluded)",
        "",
        f"- FAST p50: `{_pct(fast_lat, 50):.1f} ms`",
        f"- FAST p95: `{_pct(fast_lat, 95):.1f} ms`",
        f"- FAST mean/min/max: `{statistics.mean(fast_lat):.1f}` / "
        f"`{min(fast_lat):.1f}` / `{max(fast_lat):.1f}` ms",
        f"- PRECISE GPU p50: `{_pct(prec_lat, 50):.1f} ms`",
        f"- PRECISE GPU p95: `{_pct(prec_lat, 95):.1f} ms`",
        f"- PRECISE mean/min/max: `{statistics.mean(prec_lat):.1f}` / "
        f"`{min(prec_lat):.1f}` / `{max(prec_lat):.1f}` ms",
        "",
        "## 2. Top-10 stability",
        "",
        f"- average FAST↔PRECISE top-10 overlap: `{statistics.mean(overlaps):.2f} / 10`",
        f"- median overlap: `{statistics.median(overlaps):.1f} / 10`",
        f"- min/max overlap: `{min(overlaps)}` / `{max(overlaps)}`",
        "",
        "## 3. Rank movement (shared docs)",
        "",
        f"- shared document pairs: `{len(all_moves)}`",
        f"- mean absolute rank movement: "
        f"`{(statistics.mean(abs_moves) if abs_moves else float('nan')):.2f}`",
        f"- median absolute rank movement: "
        f"`{(statistics.median(abs_moves) if abs_moves else float('nan')):.1f}`",
        f"- moved up by ≥5: `{up5}`",
        f"- moved down by ≥5: `{down5}`",
        "",
        "### Largest positive movements (PRECISE better)",
        "",
    ]
    for m in largest_pos:
        lines.append(
            f"- {m['qid']} `{m['ecli']}`: FAST {m['fast_rank']} → PRECISE {m['precise_rank']} "
            f"(+{m['delta']})"
        )
    lines += ["", "### Largest negative movements (PRECISE worse)", ""]
    for m in largest_neg:
        lines.append(
            f"- {m['qid']} `{m['ecli']}`: FAST {m['fast_rank']} → PRECISE {m['precise_rank']} "
            f"({m['delta']})"
        )

    lines += [
        "",
        "## 4. Candidate promotion",
        "",
        f"- PRECISE promotions from rank >10 into top 10: `{promotions_gt10_top10}`",
        f"- PRECISE promotions from rank >20 into top 10: `{promotions_gt20_top10}`",
        f"- PRECISE promotions from rank >10 into top 5: `{promotions_gt10_top5}`",
        "",
        "## 5. Snippet / full-text quality",
        "",
        f"- heading-only passages FAST: `{heading_f}`",
        f"- heading-only passages PRECISE: `{heading_p}`",
        f"- missing best passages FAST/PRECISE: `{miss_best_f}` / `{miss_best_p}`",
        f"- FULL_TEXT_NOT_AVAILABLE FAST/PRECISE: `{full_miss_f}` / `{full_miss_p}`",
        f"- full texts OK FAST/PRECISE: `{full_ok_f}` / `{full_ok_p}`",
        "",
        "## 6. Human relevance worksheet",
        "",
        "Fill grades manually after reading full judgments:",
        "",
        "```text",
        "GOLD = directly answers the legal research intent",
        "SILVER = materially relevant",
        "BRONZE = related but weak",
        "MISS = not useful",
        "```",
        "",
    ]
    for q in queries:
        item = next(x for x in per_query if x["qid"] == q.qid)
        lines += [
            f"### {q.qid}",
            "",
            f"Query: `{q.query}`",
            "",
            "| Rank | FAST ECLI | FAST grade | PRECISE ECLI | PRECISE grade |",
            "| ---: | --------- | ---------- | ------------ | ------------- |",
        ]
        for i in range(10):
            fe = item["fast_top10"][i] if i < len(item["fast_top10"]) else ""
            pe = item["precise_top10"][i] if i < len(item["precise_top10"]) else ""
            lines.append(f"| {i+1} | {fe} | [ ] | {pe} | [ ] |")
        lines.append("")

    (OUT_ROOT / "benchmark_summary.md").write_text("\n".join(lines), encoding="utf-8")


async def async_main() -> int:
    _configure_env()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not QUERIES_PATH.exists():
        raise SystemExit(f"missing finalized queries: {QUERIES_PATH}")

    cuda_meta = _verify_cuda_or_stop()
    queries = _parse_queries(QUERIES_PATH)
    print(f"Loaded {len(queries)} finalized queries from {QUERIES_PATH}", flush=True)

    warmup = await _warmup()

    fast_runs: dict[str, ProfileRun] = {}
    precise_runs: dict[str, ProfileRun] = {}
    per_query: list[dict[str, Any]] = []
    errors: list[str] = []

    for q in queries:
        qdir = OUT_ROOT / q.qid
        qdir.mkdir(parents=True, exist_ok=True)
        print(f"==== {q.qid} FAST ====", flush=True)
        try:
            result, wall = await _search("fast", q.query)
            diag = dict(result.diagnostics or {})
            run = ProfileRun(
                profile="fast",
                qid=q.qid,
                query=q.query,
                wall_retrieval_ms=wall,
                reported_latency_ms=diag.get("total_latency_ms"),
                export_ms=0.0,
                retrieval_stage=result.retrieval_stage,
                result_count=int(result.result_count or 0),
                results=[_doc_to_dict(d) for d in list(result.results or [])],
                diagnostics=diag,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{q.qid}/fast: {type(exc).__name__}: {exc}")
            run = ProfileRun(
                profile="fast",
                qid=q.qid,
                query=q.query,
                wall_retrieval_ms=0.0,
                reported_latency_ms=None,
                export_ms=0.0,
                retrieval_stage=None,
                result_count=0,
                error=f"{type(exc).__name__}: {exc}",
            )
        _enrich_with_full_texts(run, dimensions=q.dimensions, stage=run.retrieval_stage)
        _write_profile_md(qdir / "fast_full_judgments.md", run, q.query)
        fast_runs[q.qid] = run

        print(f"==== {q.qid} PRECISE ====", flush=True)
        try:
            result, wall = await _search("precise", q.query)
            diag = dict(result.diagnostics or {})
            device, dtype = _assert_precise_on_cuda(diag)
            run_p = ProfileRun(
                profile="precise",
                qid=q.qid,
                query=q.query,
                wall_retrieval_ms=wall,
                reported_latency_ms=diag.get("total_latency_ms"),
                export_ms=0.0,
                retrieval_stage=result.retrieval_stage,
                result_count=int(result.result_count or 0),
                results=[_doc_to_dict(d) for d in list(result.results or [])],
                diagnostics=diag,
                reranker_device=device,
                reranker_dtype=dtype,
            )
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{q.qid}/precise: {type(exc).__name__}: {exc}")
            # If search somehow returned without cuda assertion path:
            raise SystemExit(
                f"HARD STOP: PRECISE failed for {q.qid}: {type(exc).__name__}: {exc}"
            ) from exc
        _enrich_with_full_texts(
            run_p, dimensions=q.dimensions, stage=run_p.retrieval_stage
        )
        _write_profile_md(qdir / "precise_full_judgments.md", run_p, q.query)
        precise_runs[q.qid] = run_p

        cmp_stats = _write_comparison(
            qdir / "comparison.md",
            q=q,
            fast=run,
            precise=run_p,
        )
        per_query.append(cmp_stats)
        print(
            f"{q.qid} done fast={run.wall_retrieval_ms:.0f}ms "
            f"precise={run_p.wall_retrieval_ms:.0f}ms "
            f"overlap={cmp_stats['top10_overlap']}/10 "
            f"device={run_p.reranker_device}",
            flush=True,
        )

    _write_summary(
        queries=queries,
        per_query=per_query,
        fast_runs=fast_runs,
        precise_runs=precise_runs,
        cuda_meta=cuda_meta,
        warmup=warmup,
    )

    results_payload = {
        "schema": "fast_vs_precise_gpu_15q.v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "limit": LIMIT,
        "queries": [asdict(q) for q in queries],
        "per_query": per_query,
        "errors": errors,
        "fast": {
            qid: {
                "wall_retrieval_ms": r.wall_retrieval_ms,
                "reported_latency_ms": r.reported_latency_ms,
                "result_count": r.result_count,
                "retrieval_stage": r.retrieval_stage,
                "heading_only_count": r.heading_only_count,
                "full_ok": r.full_ok,
                "full_missing": r.full_missing,
                "eclis": [d.get("ecli") for d in r.results],
            }
            for qid, r in fast_runs.items()
        },
        "precise": {
            qid: {
                "wall_retrieval_ms": r.wall_retrieval_ms,
                "reported_latency_ms": r.reported_latency_ms,
                "result_count": r.result_count,
                "retrieval_stage": r.retrieval_stage,
                "reranker_device": r.reranker_device,
                "reranker_dtype": r.reranker_dtype,
                "heading_only_count": r.heading_only_count,
                "full_ok": r.full_ok,
                "full_missing": r.full_missing,
                "rerank": (r.diagnostics or {}).get("rerank"),
                "eclis": [d.get("ecli") for d in r.results],
            }
            for qid, r in precise_runs.items()
        },
    }
    (OUT_ROOT / "benchmark_results.json").write_text(
        json.dumps(results_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # representative precise diagnostics
    sample_rerank = None
    for r in precise_runs.values():
        if (r.diagnostics or {}).get("rerank"):
            sample_rerank = (r.diagnostics or {}).get("rerank")
            break
    run_meta = {
        "cuda_available": cuda_meta.get("cuda_available"),
        "gpu_name": cuda_meta.get("gpu_name"),
        "torch_version": cuda_meta.get("torch_version"),
        "cuda_version": cuda_meta.get("cuda_version"),
        "reranker_model": (sample_rerank or {}).get("reranker_model")
        or os.getenv("NALUS_LEGAL_V2_CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3"),
        "reranker_device": warmup.get("reranker_device"),
        "reranker_dtype": warmup.get("reranker_dtype"),
        "embedding_device": os.getenv("EMBEDDING_DEVICE", "cpu"),
        "git_commit": _git_commit(),
        "collection_fast_hint": "profile-bound Slice4 A for FAST / B for PRECISE",
        "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
        "fulltext_api": FULLTEXT_API,
        "warmup": warmup,
        "ce_device_env": os.getenv("NALUS_LEGAL_V2_CE_DEVICE"),
        "sample_precise_ce_diagnostics": sample_rerank,
        "errors": errors,
    }
    (OUT_ROOT / "run_metadata.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("DONE", flush=True)
    print(f"OUT_ROOT={OUT_ROOT}", flush=True)
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
