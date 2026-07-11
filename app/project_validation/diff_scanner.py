from __future__ import annotations

import re
from dataclasses import dataclass

from app.project_validation.schemas import ClassifiedFile, RiskFinding, SafetySummary


@dataclass(frozen=True)
class DiffRule:
    rule_id: str
    severity: str
    pattern: str
    message: str
    regex: bool = False


HARD_FAIL_RULES = (
    DiffRule("qdrant_upsert", "fail", "qdrant.upsert", "Detected Qdrant write call."),
    DiffRule("qdrant_upsert_points", "fail", "upsert_points", "Detected Qdrant upsert_points call."),
    DiffRule("qdrant_recreate_collection", "fail", "recreate_collection", "Detected Qdrant collection recreation."),
    DiffRule("qdrant_delete_collection", "fail", "delete_collection", "Detected Qdrant collection deletion."),
    DiffRule("qdrant_update_collection", "fail", "update_collection", "Detected Qdrant collection update."),
    DiffRule("qdrant_create_alias", "fail", "create_alias", "Detected alias creation."),
    DiffRule("qdrant_update_alias", "fail", "update_alias", "Detected alias update."),
    DiffRule("qdrant_delete_alias", "fail", "delete_alias", "Detected alias deletion."),
    DiffRule("protected_alias_live", "fail", "nalus_live", "Detected protected alias reference."),
    DiffRule("protected_alias_stable", "fail", "nalus_stable_20260326", "Detected protected stable alias reference."),
    DiffRule("snapshot_download", "fail", "snapshot_download", "Detected snapshot download logic."),
    DiffRule("hf_hub_download", "fail", "hf_hub_download", "Detected model download logic."),
    DiffRule("from_pretrained", "fail", "from_pretrained", "Detected model loading helper."),
    DiffRule("automodel", "fail", "AutoModel", "Detected AutoModel usage."),
    DiffRule("mpnet", "fail", "MPNet", "Detected MPNet reference."),
    DiffRule("mock_embedding", "fail", "mock embedding", "Detected mock embedding fallback text."),
    DiffRule("hash_embedding", "fail", "hash embedding", "Detected hash embedding fallback text."),
    DiffRule("fallback_embedding", "fail", "fallback embedding", "Detected fallback embedding text."),
    DiffRule("deepseek_call", "fail", "deepseek", "Detected DeepSeek reference in changed diff."),
    DiffRule("openai_api_key", "fail", "OPENAI_API_KEY", "Detected API key environment usage in diff."),
    DiffRule("raw_secret", "fail", r"sk-[A-Za-z0-9]{20,}", "Detected raw secret-like token.", regex=True),
    DiffRule("nalus_legal_rag_import", "fail", "nalus_legal_rag", "Detected forbidden nalus_legal_rag reference."),
    DiffRule("ai_legal_system_import", "fail", "ai-legal-system", "Detected forbidden AI-LEGAL-SYSTEM reference."),
)

WARNING_RULES = (
    DiffRule("top_k_change", "warning", "top_k", "Detected top_k-related diff."),
    DiffRule("rrf_change", "warning", "rrf", "Detected RRF-related diff."),
    DiffRule("bm25_change", "warning", "bm25", "Detected BM25-related diff."),
    DiffRule("dense_change", "warning", "dense", "Detected dense retrieval-related diff."),
    DiffRule("redis_change", "warning", "redis", "Detected Redis-related diff."),
    DiffRule("logger_change", "warning", "logger.", "Detected logger call change; verify no raw text logging."),
)


def _rule_allowed(rule: DiffRule, allow_risks: set[str]) -> bool:
    return rule.rule_id in allow_risks or rule.pattern in allow_risks


def scan_diff_text(path: str, diff_text: str, *, allow_risks: set[str] | None = None) -> list[RiskFinding]:
    allowed = allow_risks or set()
    lowered = diff_text.lower()
    findings: list[RiskFinding] = []
    for rule in (*HARD_FAIL_RULES, *WARNING_RULES):
        if _rule_allowed(rule, allowed):
            continue
        matched = None
        if rule.regex:
            regex = re.search(rule.pattern, diff_text, flags=re.IGNORECASE)
            if regex:
                matched = regex.group(0)
        else:
            if rule.pattern.lower() in lowered:
                matched = rule.pattern
        if matched is None:
            continue
        findings.append(
            RiskFinding(
                severity=rule.severity,  # type: ignore[arg-type]
                rule_id=rule.rule_id,
                message=rule.message,
                path=path,
                matched_term=matched,
                source="diff_scan",
            )
        )

    if path.endswith(("docker-compose.yml", "docker-compose.yaml", "requirements.txt", "pyproject.toml")):
        findings.append(
            RiskFinding(
                severity="warning",
                rule_id="infra_or_dependency_change",
                message="Detected infrastructure or dependency file change.",
                path=path,
                source="diff_scan",
            )
        )
    return findings


def build_safety_summary(classified_files: list[ClassifiedFile], findings: list[RiskFinding]) -> SafetySummary:
    paths = {item.path for item in classified_files}
    finding_ids = {finding.rule_id for finding in findings}

    def _state(condition: bool) -> str:
        return "yes" if condition else "no"

    retrieval_logic_changed = any(path.startswith("app/rag/retrieval/") for path in paths)
    embedding_logic_changed = any(
        path.startswith("app/rag/retrieval/") and "embed" in path.lower() for path in paths
    )
    bm25_behavior_changed = any("bm25" in path.lower() for path in paths) or "bm25_change" in finding_ids
    rrf_behavior_changed = any(path.endswith("/rrf.py") for path in paths) or "rrf_change" in finding_ids
    qdrant_modified = any(rule.startswith("qdrant_") for rule in finding_ids) or any(
        "nalus_live" in path or "nalus_stable_20260326" in path for path in paths
    )
    redis_behavior_changed = any("redis" in path.lower() for path in paths) or "redis_change" in finding_ids
    model_download_introduced = any(
        rule in finding_ids for rule in {"snapshot_download", "hf_hub_download", "from_pretrained", "automodel", "mpnet"}
    )
    fallback_introduced = any(
        rule in finding_ids for rule in {"mock_embedding", "hash_embedding", "fallback_embedding"}
    )
    llm_or_deepseek_called = any(
        rule in finding_ids for rule in {"deepseek_call", "openai_api_key"}
    )

    return SafetySummary(
        retrieval_logic_changed=_state(retrieval_logic_changed),  # type: ignore[arg-type]
        embedding_logic_changed=_state(embedding_logic_changed),  # type: ignore[arg-type]
        bm25_behavior_changed=_state(bm25_behavior_changed),  # type: ignore[arg-type]
        rrf_behavior_changed=_state(rrf_behavior_changed),  # type: ignore[arg-type]
        qdrant_modified=_state(qdrant_modified),  # type: ignore[arg-type]
        redis_behavior_changed=_state(redis_behavior_changed),  # type: ignore[arg-type]
        model_download_introduced=_state(model_download_introduced),  # type: ignore[arg-type]
        fallback_introduced=_state(fallback_introduced),  # type: ignore[arg-type]
        llm_or_deepseek_called=_state(llm_or_deepseek_called),  # type: ignore[arg-type]
    )
