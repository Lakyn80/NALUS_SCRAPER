"""Export Stage-1 hit list + full judgment texts for manual review."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_QUERY = "matka odvezla/unesla dite do Ruska z Ceska"
DEFAULT_OUT = Path("artifacts/legal_v2/fe_query_child_russia")


def _request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body[:500]}") from exc


def _fetch_full_document(api: str, candidates: list[str]) -> dict[str, Any]:
    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate:
            continue
        url = f"{api.rstrip('/')}/api/rag/documents/{urllib.parse.quote(str(candidate), safe='')}"
        try:
            return _request_json(url, timeout=180.0)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"full document lookup failed: {last_error}")


def _passage_text(passage: Any) -> str:
    if isinstance(passage, dict):
        return str(passage.get("text") or passage.get("passage") or json.dumps(passage, ensure_ascii=False))
    return str(passage)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append(f"# Kompletni rozsudky — FE query ({payload['retrieval_profile']})")
    lines.append("")
    lines.append(f"- query: `{payload['query']}`")
    lines.append(f"- retrieval_profile: `{payload['retrieval_profile']}`")
    meta = payload.get("search_meta") or {}
    lines.append(f"- result_count: {meta.get('result_count')}")
    lines.append(f"- retrieval_stage: {meta.get('retrieval_stage')}")
    rerank = ((meta.get("diagnostics") or {}).get("rerank") or {})
    if rerank:
        lines.append(f"- rerank_applied: {rerank.get('rerank_applied')}")
        lines.append(f"- retrieval_profile_diag: {rerank.get('retrieval_profile')}")
    lines.append("")

    for entry in payload.get("decisions") or []:
        title = entry.get("ecli") or entry.get("document_id")
        lines.append(f"## Rank {entry.get('rank')}: {title}")
        lines.append("")
        for key in (
            "score",
            "court",
            "case_number",
            "decision_date",
            "document_type",
            "ce_rank",
            "ce_score",
        ):
            value = entry.get(key)
            if value is not None and value != "":
                lines.append(f"- {key}: {value}")
        lines.append(f"- document_id: `{entry.get('document_id')}`")
        passages = entry.get("relevant_passages") or []
        if passages:
            lines.append("- relevant_passages:")
            for passage in passages:
                lines.append(f"  - {_passage_text(passage)}")
        lines.append("")
        if entry.get("full_document_error"):
            lines.append(f"**FULL TEXT ERROR:** {entry['full_document_error']}")
            lines.append("")
            continue
        full = entry.get("full_document") or {}
        text = full.get("full_text") or ""
        lines.append(
            "### Full text "
            f"(availability={full.get('full_text_availability_status')}, chars={len(text)})"
        )
        lines.append("")
        lines.append("```")
        lines.append(text)
        lines.append("```")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8029")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--profile", default="ce7", choices=("fast", "ce7"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--search-timeout", type=float, default=900.0)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ready = _request_json(f"{args.api.rstrip('/')}/api/rag/legal-v2/case-similarity/ready", timeout=30.0)
    print(
        "ready=",
        json.dumps(
            {k: ready.get(k) for k in ("ready", "status", "warmup_status", "cross_encoder")},
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not ready.get("ready"):
        raise SystemExit("Stage 1 API is not ready yet")

    profile = args.profile
    print(f"search profile={profile} limit={args.limit}", flush=True)
    try:
        search = _request_json(
            f"{args.api.rstrip('/')}/api/rag/legal-v2/case-similarity/search",
            method="POST",
            payload={
                "query": args.query,
                "limit": args.limit,
                "retrieval_profile": profile,
            },
            timeout=args.search_timeout,
        )
    except Exception as exc:  # noqa: BLE001
        if profile != "fast":
            print(f"CE search failed ({type(exc).__name__}: {exc}); falling back to fast", flush=True)
            profile = "fast"
            search = _request_json(
                f"{args.api.rstrip('/')}/api/rag/legal-v2/case-similarity/search",
                method="POST",
                payload={
                    "query": args.query,
                    "limit": args.limit,
                    "retrieval_profile": profile,
                },
                timeout=300.0,
            )
        else:
            raise

    results = search.get("results") or []
    print(f"result_count={search.get('result_count')} fetched={len(results)}", flush=True)

    decisions: list[dict[str, Any]] = []
    for index, item in enumerate(results, start=1):
        ecli = item.get("ecli")
        print(f"full {index}/{len(results)} {ecli}", flush=True)
        entry: dict[str, Any] = {
            "rank": item.get("rank"),
            "score": item.get("score"),
            "ecli": ecli,
            "document_id": item.get("document_id"),
            "canonical_document_id": item.get("canonical_document_id"),
            "court": item.get("court"),
            "case_number": item.get("case_number"),
            "decision_date": item.get("decision_date"),
            "document_type": item.get("document_type"),
            "stage1_rank": item.get("stage1_rank"),
            "stage1_score": item.get("stage1_score"),
            "ce_rank": item.get("ce_rank"),
            "ce_score": item.get("ce_score"),
            "relevant_passages": item.get("relevant_passages") or [],
            "full_document": None,
            "full_document_error": None,
        }
        try:
            full = _fetch_full_document(
                args.api,
                [
                    str(item.get("document_id") or ""),
                    str(item.get("canonical_document_id") or ""),
                    str(ecli or ""),
                ],
            )
            text = full.get("full_text") or ""
            entry["full_document"] = {
                "document_id": full.get("document_id"),
                "ecli": full.get("ecli"),
                "full_text_availability_status": full.get("full_text_availability_status"),
                "full_text": text,
                "diagnostics": full.get("diagnostics"),
                "chunk_count": (full.get("diagnostics") or {}).get("chunk_count"),
            }
            print(
                f"  ok chars={len(text)} status={full.get('full_text_availability_status')}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            entry["full_document_error"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAIL {entry['full_document_error']}", flush=True)
        decisions.append(entry)

    payload = {
        "query": args.query,
        "retrieval_profile": profile,
        "search_meta": {
            "result_count": search.get("result_count"),
            "retrieval_stage": search.get("retrieval_stage"),
            "diagnostics": search.get("diagnostics"),
        },
        "decisions": decisions,
    }

    json_path = out_dir / f"full_decisions_{profile}.json"
    md_path = out_dir / f"full_decisions_{profile}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(md_path, payload)
    print(f"WROTE {json_path}", flush=True)
    print(f"WROTE {md_path}", flush=True)
    ok = sum(1 for d in decisions if d.get("full_document"))
    fail = sum(1 for d in decisions if d.get("full_document_error"))
    print(f"DONE profile={profile} full_ok={ok} full_fail={fail}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
