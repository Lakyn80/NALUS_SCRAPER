from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

ValidationMode = Literal["audit", "implementation", "artifact_only", "runtime_change", "eval_change"]
ValidationStatus = Literal["PASS", "WARN", "FAIL"]
TriState = Literal["yes", "no", "unknown"]
FindingSeverity = Literal["warning", "fail"]
FileClassification = Literal[
    "source_code",
    "infra_config",
    "tests",
    "docs",
    "project_progress",
    "eval_reports",
    "candidate_eval_artifacts",
    "generated_baseline_artifacts",
    "local_noise",
    "model_cache",
    "unknown",
]


@dataclass(frozen=True)
class GitStatusEntry:
    path: str
    status_code: str
    staged_status: str
    unstaged_status: str
    tracked: bool

    @property
    def is_untracked(self) -> bool:
        return self.status_code == "??"

    @property
    def is_staged(self) -> bool:
        return self.staged_status not in {" ", "?"}

    @property
    def is_unstaged(self) -> bool:
        return self.unstaged_status not in {" ", "?"}


@dataclass(frozen=True)
class ClassifiedFile:
    path: str
    classification: FileClassification
    status_code: str
    staged: bool
    unstaged: bool


@dataclass(frozen=True)
class RiskFinding:
    severity: FindingSeverity
    rule_id: str
    message: str
    path: str | None = None
    matched_term: str | None = None
    source: str = "policy"


@dataclass(frozen=True)
class GitSummary:
    branch: str
    expected_branch: str | None
    branch_ok: bool
    entries: list[GitStatusEntry]
    staged_paths: list[str]
    unstaged_paths: list[str]
    untracked_paths: list[str]


@dataclass(frozen=True)
class DocumentationCheck:
    required: bool
    passed: bool
    status: ValidationStatus
    message: str


@dataclass(frozen=True)
class TestExpectationCheck:
    required: bool
    passed: bool
    status: ValidationStatus
    message: str
    allow_reason: str | None = None


@dataclass(frozen=True)
class SafetySummary:
    retrieval_logic_changed: TriState
    embedding_logic_changed: TriState
    bm25_behavior_changed: TriState
    rrf_behavior_changed: TriState
    qdrant_modified: TriState
    redis_behavior_changed: TriState
    model_download_introduced: TriState
    fallback_introduced: TriState
    llm_or_deepseek_called: TriState


@dataclass(frozen=True)
class ValidationResult:
    task_name: str
    mode: ValidationMode
    status: ValidationStatus
    exit_code: int
    git_summary: GitSummary
    classified_files: list[ClassifiedFile]
    findings: list[RiskFinding]
    safety_summary: SafetySummary
    documentation_check: DocumentationCheck
    test_expectation_check: TestExpectationCheck
    recommended_next_action: str
    allowed_risks: list[str] = field(default_factory=list)
    candidate_runs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
