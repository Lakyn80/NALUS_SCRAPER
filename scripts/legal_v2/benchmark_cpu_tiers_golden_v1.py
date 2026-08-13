#!/usr/bin/env python3
"""CPU-only proxy latency benchmark for FAST / BALANCED / PRECISE.

Exercises the same async Stage1 path as the API on the 20 golden queries.

Phases per profile:
  cold  — first request after shared warmup (captures ColBERT/CE lazy load)
  warm  — 3 discard warmups, then timed golden set
  latency — p50/p95/mean from warm timed set
  concurrency — FAST 1/2/4; BALANCED/PRECISE 1 then 2 if c=1 is healthy

This host is a laptop CPU proxy, not a definitive target-VPS benchmark.
No quality re-scoring. Writes self-validating artifacts under cpu_latency_v1/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.benchmark.case_similarity_golden import (  # noqa: E402
    DEFAULT_PILOT_DATASET,
    load_case_similarity_golden_jsonl,
)
from app.rag.legal_v2.retrieve.case_similarity_search import (  # noqa: E402
    reset_case_similarity_stage1_runtime_for_tests,
    search_case_similarity_stage1,
    warmup_case_similarity_stage1_runtime,
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "legal_v2"
    / "chunking_ab_pilot_300_v1"
    / "cpu_latency_v1"
)

PROFILES = ("fast", "balanced", "precise")
EXPECTED_STAGE = {
    "fast": "hybrid_rrf_stage_1",
    "balanced": "hybrid_rrf_colbert",
    "precise": "hybrid_rrf_ce7",
}
PROFILE_LABEL = {
    "fast": "FAST",
    "balanced": "BALANCED",
    "precise": "PRECISE",
}

# Operational buckets for laptop CPU proxy (interactive UX heuristic).
# Not a VPS SLA; used only to classify this proxy run.
THRESHOLDS_MS = {
    "fast": {"safe_p50": 2500.0, "safe_p95": 6000.0, "slow_p50": 8000.0},
    "balanced": {"safe_p50": 4000.0, "safe_p95": 10000.0, "slow_p50": 20000.0},
    "precise": {"safe_p50": 12000.0, "safe_p95": 30000.0, "slow_p50": 60000.0},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", type=Path, default=DEFAULT_PILOT_DATASET)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--warmups", type=int, default=3)
    p.add_argument("--skip-concurrency", action="store_true")
    p.add_argument("--max-concurrency-heavy", type=int, default=2)
    return p.parse_args(argv)


def _git_meta() -> dict[str, Any]:
    def _run(args: list[str]) -> str:
        try:
            return (
                subprocess.check_output(args, cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL)
                .decode("utf-8")
                .strip()
            )
        except Exception:  # noqa: BLE001
            return "unknown"

    return {
        "git_head": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "branch", "--show-current"]),
        "dirty": bool(_run(["git", "status", "--porcelain"])),
    }


def _percentile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def _agg(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "min": None, "max": None, "n": 0}
    ordered = sorted(values)
    return {
        "mean": float(statistics.mean(ordered)),
        "p50": _percentile(ordered, 50),
        "p95": _percentile(ordered, 95),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "n": len(ordered),
    }


def _rss_mb() -> float | None:
    try:
        import psutil  # type: ignore

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001
        pass
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return float(parts[1]) / 1024.0
    except Exception:  # noqa: BLE001
        return None
    return None


def _env_snapshot() -> dict[str, Any]:
    keys = (
        "QDRANT_URL",
        "EMBEDDING_DEVICE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "NALUS_LEGAL_V2_COLBERT_ENABLED",
        "NALUS_LEGAL_V2_COLBERT_DEVICE",
        "NALUS_LEGAL_V2_COLBERT_ALLOW_DOWNLOAD",
        "NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED",
        "NALUS_LEGAL_V2_CE_DEVICE",
        "NALUS_LEGAL_V2_CE_ALLOW_DOWNLOAD",
        "NALUS_LEGAL_V2_COLBERT_INDEX_PATH",
    )
    return {k: os.getenv(k) for k in keys}


async def _timed_search(
    *,
    profile: str,
    query: str,
    limit: int,
    query_id: str | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        result = await search_case_similarity_stage1(
            query=query,
            limit=limit,
            retrieval_profile=profile,
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        expected = EXPECTED_STAGE[profile]
        ok = (
            result.retrieval_stage == expected
            and int(result.result_count) >= 1
        )
        return {
            "ok": ok,
            "error": None if ok else (
                f"stage={result.retrieval_stage!r} expected={expected!r} "
                f"count={result.result_count}"
            ),
            "wall_ms": wall_ms,
            "query_id": query_id,
            "retrieval_stage": result.retrieval_stage,
            "result_count": result.result_count,
            "dense_latency_ms": result.diagnostics.get("dense_latency_ms"),
            "bm25_latency_ms": result.diagnostics.get("bm25_latency_ms"),
            "colbert_latency_ms": result.diagnostics.get("colbert_latency_ms"),
            "total_latency_ms": result.diagnostics.get("total_latency_ms"),
            "rss_mb": _rss_mb(),
        }
    except Exception as exc:  # noqa: BLE001
        wall_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "wall_ms": wall_ms,
            "query_id": query_id,
            "trace": traceback.format_exc()[-1200:],
            "rss_mb": _rss_mb(),
        }


async def _concurrency_batch(
    *,
    profile: str,
    queries: list[tuple[str, str]],
    limit: int,
    concurrency: int,
) -> dict[str, Any]:
    # Cycle golden queries so concurrent requests are not identical strings.
    selected = [queries[i % len(queries)] for i in range(concurrency)]

    async def _one(i: int, qid: str, qtext: str) -> dict[str, Any]:
        row = await _timed_search(
            profile=profile,
            query=qtext,
            limit=limit,
            query_id=f"{qid}#c{concurrency}.{i}",
        )
        row["i"] = i
        return row

    started = time.perf_counter()
    rows = await asyncio.gather(
        *[_one(i, qid, qtext) for i, (qid, qtext) in enumerate(selected)],
        return_exceptions=False,
    )
    batch_wall_ms = (time.perf_counter() - started) * 1000.0
    walls = [float(r["wall_ms"]) for r in rows]
    ok_count = sum(1 for r in rows if r.get("ok"))
    errors = [str(r.get("error")) for r in rows if not r.get("ok")]
    return {
        "concurrency": concurrency,
        "ok_count": ok_count,
        "error_count": len(errors),
        "errors": errors[:5],
        "batch_wall_ms": batch_wall_ms,
        "per_request": _agg(walls),
        "peak_rss_mb": max((r.get("rss_mb") or 0.0) for r in rows) if rows else None,
        "rows": rows,
        "ok": ok_count == concurrency and not errors,
    }


def _classify_profile(profile: str, warm_agg: dict[str, Any], conc: list[dict[str, Any]]) -> str:
    p50 = warm_agg.get("p50")
    p95 = warm_agg.get("p95")
    if p50 is None:
        return "CPU_BENCH_FAILED"
    thr = THRESHOLDS_MS[profile]
    conc_ok = all(bool(c.get("ok")) for c in conc) if conc else True
    if not conc_ok:
        return "GPU_WORKER_RECOMMENDED"
    if float(p50) <= thr["safe_p50"] and (p95 is None or float(p95) <= thr["safe_p95"]):
        return "CPU_PRODUCTION_SAFE"
    if float(p50) <= thr["slow_p50"]:
        return "CPU_FUNCTIONAL_BUT_SLOW"
    return "GPU_WORKER_RECOMMENDED"


async def _run_profile(
    *,
    profile: str,
    queries: list[tuple[str, str]],
    limit: int,
    warmups: int,
    skip_concurrency: bool,
    max_concurrency_heavy: int,
) -> dict[str, Any]:
    label = PROFILE_LABEL[profile]
    print(f"\n=== {label} ({profile}) ===", flush=True)

    # Cold: first request (may include ColBERT/CE lazy init).
    cold_qid, cold_q = queries[0]
    print(f"  cold first query…", flush=True)
    cold = await _timed_search(
        profile=profile,
        query=cold_q,
        limit=limit,
        query_id=f"{cold_qid}#cold",
    )
    print(
        f"  cold wall_ms={cold['wall_ms']:.1f} ok={cold['ok']} "
        f"rss_mb={cold.get('rss_mb')}",
        flush=True,
    )
    if not cold["ok"]:
        return {
            "profile": profile,
            "label": label,
            "ok": False,
            "error": cold.get("error"),
            "cold": cold,
            "cpu_proxy_verdict": "CPU_BENCH_FAILED",
        }

    # Discard warmups (separate from timed set).
    for i in range(max(0, warmups)):
        w_qid, w_q = queries[(i + 1) % len(queries)]
        w = await _timed_search(
            profile=profile,
            query=w_q,
            limit=limit,
            query_id=f"{w_qid}#warmup{i}",
        )
        print(f"  warmup[{i}] wall_ms={w['wall_ms']:.1f} ok={w['ok']}", flush=True)
        if not w["ok"]:
            return {
                "profile": profile,
                "label": label,
                "ok": False,
                "error": w.get("error"),
                "cold": cold,
                "warmup_fail": w,
                "cpu_proxy_verdict": "CPU_BENCH_FAILED",
            }

    # Timed warm latency over full golden set.
    timed: list[dict[str, Any]] = []
    for idx, (qid, qtext) in enumerate(queries):
        row = await _timed_search(
            profile=profile,
            query=qtext,
            limit=limit,
            query_id=qid,
        )
        timed.append(row)
        mark = "OK" if row["ok"] else "FAIL"
        print(
            f"  warm[{idx + 1:02d}/{len(queries)}] {mark} "
            f"wall_ms={row['wall_ms']:.1f} id={qid}",
            flush=True,
        )
        if not row["ok"]:
            return {
                "profile": profile,
                "label": label,
                "ok": False,
                "error": row.get("error"),
                "cold": cold,
                "warm_partial": timed,
                "cpu_proxy_verdict": "CPU_BENCH_FAILED",
            }

    walls = [float(r["wall_ms"]) for r in timed]
    warm_agg = _agg(walls)
    print(
        f"  warm latency p50={warm_agg['p50']:.1f} p95={warm_agg['p95']:.1f} "
        f"mean={warm_agg['mean']:.1f} ms",
        flush=True,
    )

    concurrency_results: list[dict[str, Any]] = []
    if not skip_concurrency:
        if profile == "fast":
            levels = [1, 2, 4]
        else:
            levels = [1]
            # Escalate to 2 only if c=1 is healthy and wall is not already extreme.
            if (
                warm_agg["p50"] is not None
                and float(warm_agg["p50"]) <= THRESHOLDS_MS[profile]["slow_p50"]
            ):
                levels.append(max(2, int(max_concurrency_heavy)))

        for c in levels:
            print(f"  concurrency c={c}…", flush=True)
            batch = await _concurrency_batch(
                profile=profile,
                queries=queries,
                limit=limit,
                concurrency=c,
            )
            # Drop per-row payloads from summary to keep artifact smaller.
            summary = {k: v for k, v in batch.items() if k != "rows"}
            concurrency_results.append(summary)
            print(
                f"  concurrency c={c} ok={batch['ok']} "
                f"batch_wall_ms={batch['batch_wall_ms']:.1f} "
                f"per_p50={batch['per_request'].get('p50')}",
                flush=True,
            )
            if not batch["ok"]:
                break

    verdict = _classify_profile(profile, warm_agg, concurrency_results)
    peak_rss = max(
        [cold.get("rss_mb") or 0.0]
        + [r.get("rss_mb") or 0.0 for r in timed]
        + [c.get("peak_rss_mb") or 0.0 for c in concurrency_results]
    )
    return {
        "profile": profile,
        "label": label,
        "ok": True,
        "expected_stage": EXPECTED_STAGE[profile],
        "cold": {
            "wall_ms": cold["wall_ms"],
            "ok": cold["ok"],
            "rss_mb": cold.get("rss_mb"),
            "colbert_latency_ms": cold.get("colbert_latency_ms"),
            "dense_latency_ms": cold.get("dense_latency_ms"),
            "bm25_latency_ms": cold.get("bm25_latency_ms"),
        },
        "warm_latency_ms": warm_agg,
        "warm_rows": [
            {
                "query_id": r.get("query_id"),
                "wall_ms": r.get("wall_ms"),
                "ok": r.get("ok"),
                "dense_latency_ms": r.get("dense_latency_ms"),
                "bm25_latency_ms": r.get("bm25_latency_ms"),
                "colbert_latency_ms": r.get("colbert_latency_ms"),
            }
            for r in timed
        ],
        "concurrency": concurrency_results,
        "peak_rss_mb": peak_rss if peak_rss else None,
        "cpu_proxy_verdict": verdict,
        "thresholds_ms": THRESHOLDS_MS[profile],
    }


def _overall_verdict(profiles: dict[str, Any]) -> dict[str, Any]:
    order = {
        "CPU_PRODUCTION_SAFE": 0,
        "CPU_FUNCTIONAL_BUT_SLOW": 1,
        "GPU_WORKER_RECOMMENDED": 2,
        "CPU_BENCH_FAILED": 3,
    }
    per = {
        pid: (profiles.get(pid) or {}).get("cpu_proxy_verdict", "CPU_BENCH_FAILED")
        for pid in PROFILES
    }
    worst = max(per.values(), key=lambda v: order.get(v, 99))
    return {
        "scope": "laptop_cpu_proxy_not_target_vps",
        "per_profile": per,
        "worst": worst,
        "balanced_cpu_usable": per["balanced"]
        in {"CPU_PRODUCTION_SAFE", "CPU_FUNCTIONAL_BUT_SLOW"},
        "precise_cpu_usable": per["precise"]
        in {"CPU_PRODUCTION_SAFE", "CPU_FUNCTIONAL_BUT_SLOW"},
        "note": (
            "Usable means the profile completed correctly on this laptop CPU proxy "
            "within FUNCTIONAL_BUT_SLOW bounds. It is NOT a production VPS SLA pass."
        ),
    }


async def async_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_case_similarity_golden_jsonl(args.benchmark)
    queries = [(str(it.benchmark_id), str(it.query)) for it in items]
    if len(queries) != 20:
        raise SystemExit(f"expected 20 golden queries, got {len(queries)}")

    # Force master-allows for bench process (caller should already set devices/offline).
    os.environ["NALUS_LEGAL_V2_CASE_SIMILARITY_ENABLED"] = "1"
    os.environ["NALUS_LEGAL_V2_COLBERT_ENABLED"] = "1"
    os.environ["NALUS_LEGAL_V2_CROSS_ENCODER_ENABLED"] = "1"

    print("CPU TIER BENCH — shared warmup (BGE-M3 + BM25)", flush=True)
    reset_case_similarity_stage1_runtime_for_tests()
    shared_t0 = time.perf_counter()
    try:
        warmup = await asyncio.to_thread(warmup_case_similarity_stage1_runtime)
    except Exception as exc:  # noqa: BLE001
        payload = {
            "schema": "cpu_latency_tiers.v1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ok": False,
            "error": f"shared_warmup_failed: {type(exc).__name__}: {exc}",
            "trace": traceback.format_exc()[-2000:],
            "env": _env_snapshot(),
            "git": _git_meta(),
        }
        (output_dir / "CPU_LATENCY_RESULTS.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"ABORT: shared warmup failed: {exc}", flush=True)
        return 1
    shared_warmup_ms = (time.perf_counter() - shared_t0) * 1000.0
    print(
        f"shared_warmup_ms={shared_warmup_ms:.1f} "
        f"status={warmup.get('warmup_status')} rss_mb={_rss_mb()}",
        flush=True,
    )

    profiles_out: dict[str, Any] = {}
    for profile in PROFILES:
        profiles_out[profile] = await _run_profile(
            profile=profile,
            queries=queries,
            limit=int(args.limit),
            warmups=int(args.warmups),
            skip_concurrency=bool(args.skip_concurrency),
            max_concurrency_heavy=int(args.max_concurrency_heavy),
        )

    overall = _overall_verdict(profiles_out)
    payload = {
        "schema": "cpu_latency_tiers.v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ok": all(bool(profiles_out[p].get("ok")) for p in PROFILES),
        "benchmark": str(args.benchmark),
        "query_count": len(queries),
        "limit": int(args.limit),
        "warmups": int(args.warmups),
        "host_role": "laptop_cpu_proxy",
        "not_target_vps": True,
        "devices": {
            "embedding": os.getenv("EMBEDDING_DEVICE"),
            "colbert": os.getenv("NALUS_LEGAL_V2_COLBERT_DEVICE"),
            "ce": os.getenv("NALUS_LEGAL_V2_CE_DEVICE"),
        },
        "env": _env_snapshot(),
        "git": _git_meta(),
        "shared_warmup": {
            "wall_ms": shared_warmup_ms,
            "detail": warmup,
            "rss_mb_after": _rss_mb(),
        },
        "profiles": profiles_out,
        "verdict": overall,
        "HARD_STOP": True,
    }

    json_path = output_dir / "CPU_LATENCY_RESULTS.json"
    md_path = output_dir / "CPU_LATENCY_RESULTS.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# CPU Latency Tier Results (laptop proxy)",
        "",
        f"- timestamp: `{payload['timestamp']}`",
        f"- host_role: **laptop_cpu_proxy** (not target VPS)",
        f"- queries: `{payload['query_count']}`",
        f"- shared_warmup_ms: `{shared_warmup_ms:.1f}`",
        "",
        "## Per-profile",
        "",
        "| Profile | Warm p50 (ms) | Warm p95 (ms) | Cold (ms) | Peak RSS (MB) | Verdict |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for pid in PROFILES:
        row = profiles_out[pid]
        warm = row.get("warm_latency_ms") or {}
        cold = (row.get("cold") or {}).get("wall_ms")
        lines.append(
            f"| {row.get('label')} | {_fmt(warm.get('p50'))} | {_fmt(warm.get('p95'))} | "
            f"{_fmt(cold)} | {_fmt(row.get('peak_rss_mb'))} | `{row.get('cpu_proxy_verdict')}` |"
        )
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- worst: `{overall['worst']}`",
            f"- BALANCED CPU usable (proxy): `{overall['balanced_cpu_usable']}`",
            f"- PRECISE CPU usable (proxy): `{overall['precise_cpu_usable']}`",
            f"- note: {overall['note']}",
            "",
            "## HARD STOP",
            "",
            "Benchmark complete. No production activation. No Compose persistence. No commit/push.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n======== CPU PROXY VERDICT ========", flush=True)
    print(json.dumps(overall, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {json_path}", flush=True)
    print("HARD STOP", flush=True)
    return 0 if payload["ok"] else 1


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.1f}"
    except Exception:  # noqa: BLE001
        return str(v)


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
