from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.legal_v2.indexing import LEGAL_V2_PROFILE  # noqa: E402
from app.rag.legal_v2.interpreter import DeepSeekQuerySpecProvider  # noqa: E402
from app.rag.legal_v2.pipeline import search_legal_v2  # noqa: E402
from app.rag.legal_v2.retriever import (  # noqa: E402
    build_live_legal_v2_retriever,
    legal_v2_retriever_config_from_env,
)
from app.rag.legal_v2.verifier import DeepSeekSemanticVerifierProvider  # noqa: E402
from app.rag.llm.config import effective_llm_config_from_env  # noqa: E402
from app.rag.retrieval.bge_m3_embedder import BgeM3Embedder  # noqa: E402
from app.rag.retrieval.production_profile import ProductionRetrievalConfig  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explicit live Legal Retrieval v2 DeepSeek smoke.")
    parser.add_argument("--query", default="únos dítěte matkou z Česka do Ruska")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/legal_v2/live_smoke")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    llm_config = effective_llm_config_from_env()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not llm_config.api_key_configured:
        _write_blocked(
            args.output_dir,
            "DeepSeek credentials are not configured.",
            llm_config=llm_config.to_safe_dict(),
        )
        return 2
    from qdrant_client import QdrantClient  # type: ignore[import-not-found]

    config = legal_v2_retriever_config_from_env()
    if not config.bm25_sidecar_path.exists():
        _write_blocked(args.output_dir, f"BM25 sidecar is missing: {config.bm25_sidecar_path}")
        return 2
    prod_config = ProductionRetrievalConfig(
        profile=LEGAL_V2_PROFILE,
        qdrant_collection=config.qdrant_collection,
        bm25_sidecar_path=config.bm25_sidecar_path,
        bm25_index_id=config.bm25_index_id,
        model_path=config.model_path,
        local_files_only=True,
        trust_remote_code=False,
        device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        candidate_multiplier=1,
        min_candidate_count=1,
        max_candidate_count=1,
        lexical_filter_enabled=False,
    )
    client = QdrantClient(url=args.qdrant_url, timeout=20)
    if config.qdrant_collection not in {item.name for item in client.get_collections().collections}:
        _write_blocked(args.output_dir, f"Qdrant collection is missing: {config.qdrant_collection}")
        return 2
    result = search_legal_v2(
        query=args.query,
        retriever=build_live_legal_v2_retriever(client, BgeM3Embedder(prod_config), config),
        verifier=DeepSeekSemanticVerifierProvider(api_key, model=llm_config.deepseek_model),
        config=config,
        query_provider=DeepSeekQuerySpecProvider(api_key, model=llm_config.deepseek_model),
        debug=True,
    )
    payload = asdict(result)
    payload["llm_config"] = llm_config.to_safe_dict()
    (args.output_dir / "live_smoke.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (args.output_dir / "live_smoke.md").write_text(
        f"# Legal v2 live smoke\n\n- Status: `{result.status}`\n- Verified documents: {len(result.verified_documents)}\n",
        encoding="utf-8",
    )
    print(args.output_dir / "live_smoke.json")
    verifier_errors = int(result.rejection_counts.get("verifier_error", 0))
    clean_terminal_status = result.status in {"verified_match", "no_verified_results"}
    return 0 if clean_terminal_status and verifier_errors == 0 else 1


def _write_blocked(output_dir: Path, reason: str, *, llm_config: dict | None = None) -> None:
    payload = {
        "summary": {
            "status": "blocked",
            "reason": reason,
            "provider_calls_total": 0,
            "provider_cost_total": 0,
        },
        "llm_config": llm_config or {},
    }
    (output_dir / "live_smoke.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "live_smoke.md").write_text(f"# Legal v2 live smoke\n\n- Status: `blocked`\n- Reason: {reason}\n", encoding="utf-8")
    print(reason)


if __name__ == "__main__":
    raise SystemExit(main())
