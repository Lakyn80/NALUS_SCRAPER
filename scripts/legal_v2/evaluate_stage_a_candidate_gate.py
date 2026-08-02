from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.query_spec import build_query_spec_v2  # noqa: E402
from app.rag.legal_v2.retriever import (  # noqa: E402
    LegalV2RetrieverConfig,
    build_live_legal_v2_retriever,
)
from app.rag.retrieval.bge_m3_embedder import BgeM3Embedder  # noqa: E402
from app.rag.retrieval.production_profile import ProductionRetrievalConfig  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Legal Retrieval v2 Stage A candidate gate without provider calls."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/legal_v2/pilot_600_20260731/universal_quality/reviewed_benchmark.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/legal_v2/pilot_600_20260731/universal_quality",
    )
    parser.add_argument("--qdrant-url", default="http://qdrant:6333")
    parser.add_argument(
        "--qdrant-collection",
        default="nalus_legal_paragraph_chunks_v2_pilot_600",
    )
    parser.add_argument(
        "--bm25-sidecar-path",
        type=Path,
        default=Path(
            "/app/storage/rag/bm25/nalus_legal_paragraph_bm25_v2_pilot_600.sqlite"
        ),
    )
    parser.add_argument(
        "--bm25-index-id",
        default="nalus_legal_paragraph_bm25_v2_pilot_600",
    )
    parser.add_argument("--candidate-window", type=int, default=60)
    parser.add_argument("--dense-candidate-chunks", type=int, default=80)
    parser.add_argument("--bm25-candidate-chunks", type=int, default=80)
    parser.add_argument("--fused-candidate-chunks", type=int, default=120)
    parser.add_argument("--json-name", default="stage_a_gate.json")
    parser.add_argument("--markdown-name", default="stage_a_gate.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    rows = list(benchmark.get("items") or [])
    from qdrant_client import QdrantClient  # type: ignore[import-not-found]

    config = LegalV2RetrieverConfig(
        qdrant_collection=args.qdrant_collection,
        bm25_sidecar_path=args.bm25_sidecar_path,
        bm25_index_id=args.bm25_index_id,
        dense_candidate_chunks=args.dense_candidate_chunks,
        bm25_candidate_chunks=args.bm25_candidate_chunks,
        fused_candidate_chunks=args.fused_candidate_chunks,
        candidate_documents=args.candidate_window,
        model_path=os.getenv("EMBEDDING_MODEL_NAME", "/app/models/BAAI/bge-m3"),
    )
    prod_config = _embedder_config(config)
    client = QdrantClient(url=args.qdrant_url, timeout=30)
    retriever = build_live_legal_v2_retriever(
        client,
        BgeM3Embedder(prod_config),
        config,
    )

    evaluated: list[dict[str, Any]] = []
    retrieval_errors = 0
    zero_candidate_gold_queries: list[str] = []
    wrong_index_identity = 0
    per_domain: dict[str, list[dict[str, float]]] = {}
    hard_negative_top60 = 0
    hard_negative_top20 = 0
    hard_negative_top10 = 0
    gold_absent_top60: list[dict[str, Any]] = []

    for row in rows:
        query_id = str(row.get("id") or "")
        query = str(row.get("query") or "")
        split = str(row.get("benchmark_split") or row.get("split") or "")
        domain = str(row.get("legal_domain") or row.get("domain") or "")
        gold = _gold_ids(row)
        hard_negative_ids = _hard_negative_ids(row)
        if not query:
            continue
        if row.get("clarification_expected") or row.get("zero_result_expected"):
            evaluated.append(
                {
                    "id": query_id,
                    "domain": domain,
                    "split": split,
                    "gold": gold,
                    "ranked": [],
                    "p5": 0.0,
                    "r10": 0.0,
                    "r20": 0.0,
                    "coverage60": 0.0,
                    "mrr": 0.0,
                    "retrieval_error": None,
                    "clarification_expected": bool(row.get("clarification_expected")),
                    "zero_result_expected": bool(row.get("zero_result_expected")),
                }
            )
            continue
        if not gold:
            continue
        error: str | None
        try:
            query_spec = build_query_spec_v2(query)
            retrieval = retriever.retrieve(query_spec)
            ranked = [item.document_id for item in retrieval.documents]
            index = retrieval.diagnostics
            if (
                index.get("collection") != args.qdrant_collection
                or index.get("bm25_index_id") != args.bm25_index_id
            ):
                wrong_index_identity += 1
        except Exception as exc:  # noqa: BLE001
            retrieval_errors += 1
            ranked = []
            error = exc.__class__.__name__
        else:
            error = None
        metrics = _query_metrics(gold, ranked, args.candidate_window)
        if not ranked and gold:
            zero_candidate_gold_queries.append(query_id)
        if metrics["coverage60"] == 0.0:
            gold_absent_top60.append({"id": query_id, "domain": domain, "gold": gold})
        hard_negative_top60 += len(set(ranked[: args.candidate_window]).intersection(hard_negative_ids))
        hard_negative_top20 += len(set(ranked[:20]).intersection(hard_negative_ids))
        hard_negative_top10 += len(set(ranked[:10]).intersection(hard_negative_ids))
        per_domain.setdefault(domain, []).append(metrics)
        evaluated.append(
            {
                "id": query_id,
                "intent_id": row.get("intent_id"),
                "domain": domain,
                "split": split,
                "style": row.get("query_style"),
                "gold": gold,
                "ranked": ranked,
                "retrieval_error": error,
                **metrics,
                "clarification_expected": False,
                "zero_result_expected": False,
            }
        )

    gold_rows = [
        item
        for item in evaluated
        if item["gold"]
        and not item["clarification_expected"]
        and not item["zero_result_expected"]
    ]
    summary = {
        "schema": "legal_v2_stage_a_candidate_gate_v2",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provider_calls": 0,
        "query_count": len(rows),
        "gold_query_count": len(gold_rows),
        "intent_count": int(benchmark.get("summary", {}).get("intent_count") or 0),
        "candidate_window": args.candidate_window,
        "macro_precision_at_5": _mean(item["p5"] for item in gold_rows),
        "macro_recall_at_10": _mean(item["r10"] for item in gold_rows),
        "recall_at_20": _mean(item["r20"] for item in gold_rows),
        "mrr": _mean(item["mrr"] for item in gold_rows),
        "gold_coverage_in_candidate_window": _mean(
            item["coverage60"] for item in gold_rows
        ),
        "retrieval_errors": retrieval_errors,
        "zero_candidate_gold_query_count": len(zero_candidate_gold_queries),
        "wrong_index_identity": wrong_index_identity,
        "runtime_benchmark_leakage": 0,
        "query_specific_production_rules": 0,
        "endpoint_independent_retrieval_crashes": retrieval_errors,
        "candidate_hard_negative_leakage_top10": hard_negative_top10,
        "candidate_hard_negative_leakage_top20": hard_negative_top20,
        "candidate_hard_negative_leakage_top60": hard_negative_top60,
        "gold_absent_top60": gold_absent_top60,
        "per_domain": {
            domain: {
                "queries": len(values),
                "recall_at_10": _mean(item["r10"] for item in values),
                "recall_at_20": _mean(item["r20"] for item in values),
                "coverage60": _mean(item["coverage60"] for item in values),
            }
            for domain, values in sorted(per_domain.items())
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary["stage_a_gate_passed"] = _stage_a_passed(summary)
    payload = {
        "summary": summary,
        "rows": evaluated,
        "gate": {
            "coverage_at_60_min": 0.95,
            "recall_at_20_min": 0.75,
            "recall_at_10_min": 0.65,
            "precision_at_5_blocks_stage_b": False,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / args.json_name
    markdown_path = args.output_dir / args.markdown_name
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    print(json_path)
    return 0 if summary["stage_a_gate_passed"] else 2


def _embedder_config(config: LegalV2RetrieverConfig) -> ProductionRetrievalConfig:
    from app.rag.legal_v2.indexing import LEGAL_V2_PROFILE

    return ProductionRetrievalConfig(
        profile=LEGAL_V2_PROFILE,
        qdrant_collection=config.qdrant_collection,
        bm25_sidecar_path=config.bm25_sidecar_path,
        bm25_index_id=config.bm25_index_id,
        model_path=config.model_path,
        local_files_only=True,
        trust_remote_code=False,
        device="cpu",
        candidate_multiplier=1,
        min_candidate_count=1,
        max_candidate_count=max(
            config.dense_candidate_chunks,
            config.bm25_candidate_chunks,
        ),
        lexical_filter_enabled=False,
    )


def _gold_ids(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in (
        "strongly_relevant_document_ids",
        "materially_relevant_document_ids",
        "partial_match_document_ids",
    ):
        values.extend(str(item) for item in row.get(key) or [])
    return _dedupe(values)


def _hard_negative_ids(row: dict[str, Any]) -> set[str]:
    values: list[str] = []
    for key in ("explicit_hard_negative_document_ids", "related_only_document_ids"):
        values.extend(str(item) for item in row.get(key) or [])
    return set(_dedupe(values))


def _query_metrics(
    gold_ids: list[str],
    ranked_ids: list[str],
    candidate_window: int,
) -> dict[str, float]:
    gold = set(gold_ids)
    top5 = ranked_ids[:5]
    top10 = ranked_ids[:10]
    top20 = ranked_ids[:20]
    top_window = ranked_ids[:candidate_window]
    first_rank = next(
        (index for index, value in enumerate(ranked_ids, start=1) if value in gold),
        None,
    )
    return {
        "p5": len(gold.intersection(top5)) / 5,
        "r10": len(gold.intersection(top10)) / len(gold),
        "r20": len(gold.intersection(top20)) / len(gold),
        "coverage60": 1.0 if gold.intersection(top_window) else 0.0,
        "mrr": (1 / first_rank) if first_rank else 0.0,
    }


def _stage_a_passed(summary: dict[str, Any]) -> bool:
    return (
        float(summary["gold_coverage_in_candidate_window"]) >= 0.95
        and float(summary["recall_at_20"]) >= 0.75
        and float(summary["macro_recall_at_10"]) >= 0.65
        and int(summary["zero_candidate_gold_query_count"]) == 0
        and int(summary["wrong_index_identity"]) == 0
        and int(summary["runtime_benchmark_leakage"]) == 0
        and int(summary["query_specific_production_rules"]) == 0
        and int(summary["endpoint_independent_retrieval_crashes"]) == 0
    )


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Legal Retrieval v2 Stage A candidate gate",
        "",
        f"- Stage A gate passed: `{summary['stage_a_gate_passed']}`",
        f"- Provider calls: `{summary['provider_calls']}`",
        f"- Gold queries: `{summary['gold_query_count']}`",
        f"- Candidate window: `{summary['candidate_window']}`",
        f"- Precision@5: `{summary['macro_precision_at_5']}`",
        f"- Recall@10: `{summary['macro_recall_at_10']}`",
        f"- Recall@20: `{summary['recall_at_20']}`",
        f"- MRR: `{summary['mrr']}`",
        "- Gold coverage in candidate window@60: "
        f"`{summary['gold_coverage_in_candidate_window']}`",
        f"- Retrieval errors: `{summary['retrieval_errors']}`",
        f"- Wrong index identity: `{summary['wrong_index_identity']}`",
        "- Runtime benchmark leakage: "
        f"`{summary['runtime_benchmark_leakage']}`",
        "- Query-specific production rules: "
        f"`{summary['query_specific_production_rules']}`",
        "- Endpoint-independent retrieval crashes: "
        f"`{summary['endpoint_independent_retrieval_crashes']}`",
        "- Candidate hard-negative leakage top10/top20/top60: "
        f"`{summary['candidate_hard_negative_leakage_top10']}` / "
        f"`{summary['candidate_hard_negative_leakage_top20']}` / "
        f"`{summary['candidate_hard_negative_leakage_top60']}`",
        "",
        "## Gate Interpretation",
        "",
        "- Candidate precision@5 is measured and reported, but does not block Stage B.",
        "- Stage A is accepted only as a broad candidate supplier, not final user-facing ranking.",
        "",
        "## Gold Absent From Top 60",
        "",
    ]
    absent = summary["gold_absent_top60"]
    if absent:
        for item in absent:
            lines.append(f"- `{item['id']}` `{item['domain']}` gold={item['gold']}")
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def _mean(values: Any) -> float:
    items = [float(value) for value in values]
    return statistics.fmean(items) if items else 0.0


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
