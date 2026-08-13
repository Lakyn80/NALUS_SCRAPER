#!/usr/bin/env python3
"""Staging smoke + concurrency check for FAST / BALANCED / PRECISE.

Exercises the same async Stage1 search path as the API (not HTTP), with:
- master-allow rejection checks
- one warm query per profile
- bounded concurrent load on BALANCED and PRECISE
- selected failure modes (missing ColBERT index)

Writes a self-validating report under artifacts/.../staging_smoke_v1/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.retrieve.case_similarity_search import (  # noqa: E402
    reset_case_similarity_stage1_runtime_for_tests,
    search_case_similarity_stage1,
    warmup_case_similarity_stage1_runtime,
)
from app.rag.legal_v2.retrieve.retrieval_profiles import (  # noqa: E402
    resolve_retrieval_profile,
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "legal_v2"
    / "chunking_ab_pilot_300_v1"
    / "staging_smoke_v1"
)
SMOKE_QUERY = (
    "Ústavní stížnost odmítnutá pro formální vady zastoupení advokátem"
)
EXPECTED_STAGE = {
    "fast": "hybrid_rrf_stage_1",
    "balanced": "hybrid_rrf_colbert",
    "precise": "hybrid_rrf_ce7",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--query", default=SMOKE_QUERY)
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--skip-live", action="store_true")
    p.add_argument("--skip-concurrency", action="store_true")
    return p.parse_args(argv)


def _ok(name: str, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "status": "pass", "detail": detail or {}}


def _fail(name: str, error: str, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "status": "fail", "error": error, "detail": detail or {}}


def _check_master_allows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # balanced without ColBERT allow
    prev_c = os.environ.get("NALUS_LEGAL_V2_COLBERT_ENABLED")
    os.environ["NALUS_LEGAL_V2_COLBERT_ENABLED"] = "0"
    try:
        resolve_retrieval_profile("balanced")
        rows.append(_fail("master_allow_balanced_off", "expected ValueError"))
    except ValueError as exc:
        rows.append(_ok("master_allow_balanced_off", {"message": str(exc)[:200]}))
    finally:
        if prev_c is None:
            os.environ.pop("NALUS_LEGAL_V2_COLBERT_ENABLED", None)
        else:
            os.environ["NALUS_LEGAL_V2_COLBERT_ENABLED"] = prev_c

    prev_ce = os.environ.get("NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED")
    os.environ["NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED"] = "0"
    try:
        resolve_retrieval_profile("precise")
        rows.append(_fail("master_allow_precise_off", "expected ValueError"))
    except ValueError as exc:
        rows.append(_ok("master_allow_precise_off", {"message": str(exc)[:200]}))
    finally:
        if prev_ce is None:
            os.environ.pop("NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED", None)
        else:
            os.environ["NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED"] = prev_ce

    # aliases / labels with allows on
    os.environ["NALUS_LEGAL_V2_COLBERT_ENABLED"] = "1"
    os.environ["NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED"] = "1"
    fast = resolve_retrieval_profile("fast")
    bal = resolve_retrieval_profile("balanced")
    prec = resolve_retrieval_profile("precise")
    ce7 = resolve_retrieval_profile("ce7")
    if (
        fast.label == "FAST"
        and bal.label == "BALANCED"
        and prec.label == "PRECISE"
        and ce7.profile_id == "precise"
    ):
        rows.append(
            _ok(
                "profile_labels",
                {
                    "fast": fast.profile_id,
                    "balanced": bal.profile_id,
                    "precise": prec.profile_id,
                    "ce7_alias": ce7.profile_id,
                },
            )
        )
    else:
        rows.append(_fail("profile_labels", "unexpected profile mapping"))
    return rows


async def _run_profile(
    *,
    profile: str,
    query: str,
    limit: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = await search_case_similarity_stage1(
        query=query,
        limit=limit,
        retrieval_profile=profile,
    )
    wall_ms = (time.perf_counter() - started) * 1000.0
    expected = EXPECTED_STAGE[profile]
    ok = (
        result.retrieval_stage == expected
        and int(result.result_count) >= 1
        and result.diagnostics.get("retrieval_profile")
        in {profile, "precise" if profile == "precise" else profile}
    )
    detail = {
        "retrieval_stage": result.retrieval_stage,
        "expected_stage": expected,
        "result_count": result.result_count,
        "wall_ms": wall_ms,
        "collection": result.diagnostics.get("collection"),
        "retrieval_profile": result.diagnostics.get("retrieval_profile"),
        "colbert_applied": result.diagnostics.get("colbert_applied"),
        "dense_latency_ms": result.diagnostics.get("dense_latency_ms"),
        "bm25_latency_ms": result.diagnostics.get("bm25_latency_ms"),
        "colbert_latency_ms": result.diagnostics.get("colbert_latency_ms"),
        "total_latency_ms": result.diagnostics.get("total_latency_ms"),
        "top_ecli": result.results[0].ecli if result.results else None,
    }
    if ok:
        return _ok(f"live_{profile}", detail)
    return _fail(
        f"live_{profile}",
        f"stage={result.retrieval_stage!r} expected={expected!r} "
        f"count={result.result_count}",
        detail,
    )


async def _concurrency_batch(
    *,
    profile: str,
    query: str,
    limit: int,
    n: int,
) -> dict[str, Any]:
    async def _one(i: int) -> dict[str, Any]:
        t0 = time.perf_counter()
        result = await search_case_similarity_stage1(
            query=f"{query} [{i}]",
            limit=limit,
            retrieval_profile=profile,
        )
        return {
            "i": i,
            "wall_ms": (time.perf_counter() - t0) * 1000.0,
            "stage": result.retrieval_stage,
            "count": result.result_count,
            "ok": result.retrieval_stage == EXPECTED_STAGE[profile]
            and result.result_count >= 1,
        }

    started = time.perf_counter()
    rows = await asyncio.gather(*[_one(i) for i in range(n)], return_exceptions=True)
    wall_ms = (time.perf_counter() - started) * 1000.0
    parsed: list[dict[str, Any]] = []
    errors: list[str] = []
    for row in rows:
        if isinstance(row, Exception):
            errors.append(f"{type(row).__name__}: {row}")
        else:
            parsed.append(row)
    walls = [float(r["wall_ms"]) for r in parsed]
    ok_count = sum(1 for r in parsed if r.get("ok"))
    detail = {
        "n": n,
        "ok_count": ok_count,
        "error_count": len(errors),
        "errors": errors[:5],
        "batch_wall_ms": wall_ms,
        "per_request_mean_ms": float(statistics.mean(walls)) if walls else None,
        "per_request_max_ms": float(max(walls)) if walls else None,
        "expected_stage": EXPECTED_STAGE[profile],
    }
    if errors or ok_count != n:
        return _fail(f"concurrency_{profile}", f"ok={ok_count}/{n} errors={len(errors)}", detail)
    return _ok(f"concurrency_{profile}", detail)


async def _failure_missing_colbert_index() -> dict[str, Any]:
    from app.rag.retrieval.errors import RetrievalConfigurationError

    reset_case_similarity_stage1_runtime_for_tests()
    prev = os.environ.get("NALUS_LEGAL_V2_COLBERT_INDEX_PATH")
    os.environ["NALUS_LEGAL_V2_COLBERT_INDEX_PATH"] = "/tmp/missing-colbert-index-does-not-exist"
    os.environ["NALUS_LEGAL_V2_COLBERT_ENABLED"] = "1"
    try:
        # Force runtime rebuild with real Qdrant/BM25, then fail on ColBERT lazy init.
        await search_case_similarity_stage1(
            query=SMOKE_QUERY,
            limit=3,
            retrieval_profile="balanced",
        )
        return _fail("failure_missing_colbert_index", "expected RetrievalConfigurationError")
    except (RetrievalConfigurationError, ValueError) as exc:
        return _ok(
            "failure_missing_colbert_index",
            {"error_type": type(exc).__name__, "message": str(exc)[:240]},
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(
            "failure_missing_colbert_index",
            f"unexpected {type(exc).__name__}: {exc}",
        )
    finally:
        if prev is None:
            os.environ.pop("NALUS_LEGAL_V2_COLBERT_INDEX_PATH", None)
        else:
            os.environ["NALUS_LEGAL_V2_COLBERT_INDEX_PATH"] = prev
        reset_case_similarity_stage1_runtime_for_tests()


async def async_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, Any]] = []
    checks.extend(_check_master_allows())

    # Ensure live master allows for remaining checks.
    os.environ["NALUS_LEGAL_V2_CASE_SIMILARITY_ENABLED"] = "1"
    os.environ["NALUS_LEGAL_V2_COLBERT_ENABLED"] = "1"
    os.environ["NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED"] = "1"

    if not args.skip_live:
        reset_case_similarity_stage1_runtime_for_tests()
        try:
            warmup = await asyncio.to_thread(warmup_case_similarity_stage1_runtime)
            checks.append(_ok("warmup", {"warmup": warmup}))
        except Exception as exc:  # noqa: BLE001
            checks.append(
                _fail("warmup", f"{type(exc).__name__}: {exc}", {"trace": traceback.format_exc()[-1200:]})
            )
            payload = _finalize(checks, args, output_dir, aborted=True)
            return 1 if payload["summary"]["failed"] else 0

        for profile in ("fast", "balanced", "precise"):
            try:
                checks.append(
                    await _run_profile(
                        profile=profile,
                        query=args.query,
                        limit=args.limit,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                checks.append(
                    _fail(
                        f"live_{profile}",
                        f"{type(exc).__name__}: {exc}",
                        {"trace": traceback.format_exc()[-1200:]},
                    )
                )

        if not args.skip_concurrency:
            for profile in ("balanced", "precise"):
                try:
                    checks.append(
                        await _concurrency_batch(
                            profile=profile,
                            query=args.query,
                            limit=args.limit,
                            n=max(2, int(args.concurrency)),
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    checks.append(
                        _fail(
                            f"concurrency_{profile}",
                            f"{type(exc).__name__}: {exc}",
                            {"trace": traceback.format_exc()[-1200:]},
                        )
                    )

        try:
            checks.append(await _failure_missing_colbert_index())
            # Restore live runtime after destructive failure test.
            reset_case_similarity_stage1_runtime_for_tests()
            await asyncio.to_thread(warmup_case_similarity_stage1_runtime)
        except Exception as exc:  # noqa: BLE001
            checks.append(
                _fail(
                    "failure_missing_colbert_index",
                    f"{type(exc).__name__}: {exc}",
                    {"trace": traceback.format_exc()[-1200:]},
                )
            )

    payload = _finalize(checks, args, output_dir, aborted=False)
    print(f"STAGING SMOKE VERDICT: {payload['verdict']['STAGING_SMOKE_VERDICT']}", flush=True)
    print(
        f"passed={payload['summary']['passed']} failed={payload['summary']['failed']} "
        f"total={payload['summary']['total']}",
        flush=True,
    )
    for row in checks:
        mark = "PASS" if row["status"] == "pass" else "FAIL"
        print(f"  [{mark}] {row['name']}", flush=True)
        if row["status"] != "pass":
            print(f"         {row.get('error')}", flush=True)
    return 0 if payload["summary"]["failed"] == 0 else 1


def _finalize(
    checks: list[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
    *,
    aborted: bool,
) -> dict[str, Any]:
    passed = sum(1 for c in checks if c["status"] == "pass")
    failed = sum(1 for c in checks if c["status"] != "pass")
    if aborted and failed:
        verdict = "STAGING_SMOKE_FAIL"
    elif failed == 0 and passed > 0:
        verdict = "STAGING_SMOKE_PASS"
    elif failed == 0:
        verdict = "STAGING_SMOKE_INCONCLUSIVE"
    else:
        verdict = "STAGING_SMOKE_FAIL"
    payload = {
        "schema": "staging_smoke_tiers.v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": args.query,
        "limit": args.limit,
        "concurrency": args.concurrency,
        "env": {
            "QDRANT_URL": os.getenv("QDRANT_URL"),
            "COLBERT_ENABLED": os.getenv("NALUS_LEGAL_V2_COLBERT_ENABLED"),
            "CE_ENABLED": os.getenv("NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED"),
            "EMBEDDING_DEVICE": os.getenv("EMBEDDING_DEVICE"),
            "COLBERT_DEVICE": os.getenv("NALUS_LEGAL_V2_COLBERT_DEVICE"),
            "CE_DEVICE": os.getenv("NALUS_LEGAL_V2_CROSS_ENCODER_DEVICE"),
        },
        "checks": checks,
        "summary": {"passed": passed, "failed": failed, "total": len(checks)},
        "verdict": {
            "STAGING_SMOKE_VERDICT": verdict,
            "aborted": aborted,
            "note": (
                "In-process Stage1 smoke (same async path as API). "
                "HTTP reverse-proxy surface not included."
            ),
        },
    }
    json_path = output_dir / "STAGING_SMOKE_RESULTS.json"
    md_path = output_dir / "STAGING_SMOKE_RESULTS.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Staging Smoke Results",
        "",
        f"## STAGING SMOKE VERDICT: {verdict}",
        "",
        f"- passed: {passed}",
        f"- failed: {failed}",
        f"- total: {len(checks)}",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for row in checks:
        detail = row.get("error") or json.dumps(row.get("detail") or {}, ensure_ascii=False)[:120]
        lines.append(f"| {row['name']} | {row['status']} | `{detail}` |")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {json_path}", flush=True)
    print(f"WROTE {md_path}", flush=True)
    return payload


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
