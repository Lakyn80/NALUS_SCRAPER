from __future__ import annotations

import re

from app.project_validation.schemas import ClassifiedFile, FileClassification, GitStatusEntry

_BASELINE_DIR_RE = re.compile(r"^artifacts/rag_eval/legal_qa/runs/[^/]*_baseline/", re.IGNORECASE)
_BASELINE_ANSWER_EVAL_RE = re.compile(
    r"^artifacts/rag_eval/legal_qa/answer_eval/(usoud_no_llm_baseline|mixed_no_llm_baseline|nsoud_no_llm_baseline)/",
    re.IGNORECASE,
)
_CANDIDATE_RE = re.compile(r"^artifacts/rag_eval/legal_qa/(runs|answer_eval)/([^/]+)/", re.IGNORECASE)


def normalize_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _is_local_noise(path: str) -> bool:
    return path.startswith((".idea/", ".vscode/", "__pycache__/", ".pytest_cache/", "node_modules/", "dist/", "build/")) or path.endswith(
        (".pyc", ".log")
    )


def _is_model_cache(path: str) -> bool:
    return path.startswith(("models/", ".cache/")) or path == ".env" or path.startswith(".env.")


def _is_infrastructure_config(path: str) -> bool:
    return (
        path
        in {
            ".env.example",
            "docker-compose.yml",
            "docker-compose.yaml",
            "pyproject.toml",
            "requirements.txt",
            "requirements-ci.txt",
            "requirements-local.txt",
        }
        or path.startswith("monitoring/")
    )


def _is_candidate_eval_artifact(path: str, candidate_runs: set[str]) -> bool:
    match = _CANDIDATE_RE.match(path)
    if match is None:
        return False
    run_name = match.group(2)
    return (
        run_name in candidate_runs
        or "candidate" in run_name.lower()
        or "repaired" in run_name.lower()
    )


def classify_path(path: str, *, candidate_runs: set[str] | None = None) -> FileClassification:
    normalized = normalize_path(path)
    candidate_run_names = candidate_runs or set()

    if normalized == "PROJECT_PROGRESS.md":
        return "project_progress"
    if _is_local_noise(normalized):
        return "local_noise"
    if _is_infrastructure_config(normalized):
        return "infra_config"
    if _is_model_cache(normalized):
        return "model_cache"
    if normalized.startswith("artifacts/evaluation_quality/") and normalized.endswith((".md", ".json", ".jsonl")):
        return "eval_reports"
    if _BASELINE_ANSWER_EVAL_RE.match(normalized) or _BASELINE_DIR_RE.match(normalized):
        return "generated_baseline_artifacts"
    if _is_candidate_eval_artifact(normalized, candidate_run_names):
        return "candidate_eval_artifacts"
    if normalized.startswith("tests/"):
        return "tests"
    if normalized.startswith("docs/") or normalized in {
        "PROJECT_EXECUTION_PROTOCOL.md",
        "readme.dev",
        "README.dev",
        "README.dev.md",
    }:
        return "docs"
    if normalized.startswith("app/") or normalized.startswith("scripts/"):
        return "source_code"
    return "unknown"


def classify_entries(
    entries: list[GitStatusEntry],
    *,
    candidate_runs: set[str] | None = None,
) -> list[ClassifiedFile]:
    return [
        ClassifiedFile(
            path=normalize_path(entry.path),
            classification=classify_path(entry.path, candidate_runs=candidate_runs),
            status_code=entry.status_code,
            staged=entry.is_staged,
            unstaged=entry.is_unstaged,
        )
        for entry in entries
    ]
