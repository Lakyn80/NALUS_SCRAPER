from __future__ import annotations

import os
import subprocess
from pathlib import Path

from app.project_validation.schemas import GitStatusEntry


class GitValidationError(RuntimeError):
    """Raised when git state cannot be collected safely."""


def _run_git(repo_root: Path, args: list[str], *, allowed_returncodes: set[int] | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    accepted = allowed_returncodes or {0}
    if completed.returncode not in accepted:
        message = completed.stderr.strip() or completed.stdout.strip() or f"git {' '.join(args)} failed"
        raise GitValidationError(message)
    return completed.stdout


def get_current_branch(repo_root: Path) -> str:
    return _run_git(repo_root, ["branch", "--show-current"]).strip()


def get_recent_commits(repo_root: Path, count: int = 5) -> list[str]:
    output = _run_git(repo_root, ["log", f"--oneline", f"-{count}"])
    return [line.strip() for line in output.splitlines() if line.strip()]


def parse_status_output(output: str) -> list[GitStatusEntry]:
    entries: list[GitStatusEntry] = []
    for raw_line in output.splitlines():
        if not raw_line.strip():
            continue
        if raw_line.startswith("?? "):
            path = raw_line[3:].strip()
            entries.append(
                GitStatusEntry(
                    path=path,
                    status_code="??",
                    staged_status="?",
                    unstaged_status="?",
                    tracked=False,
                )
            )
            continue
        if len(raw_line) < 4:
            raise GitValidationError(f"Unrecognized git status line: {raw_line!r}")
        status_code = raw_line[:2]
        path = raw_line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", maxsplit=1)[1]
        entries.append(
            GitStatusEntry(
                path=path,
                status_code=status_code,
                staged_status=status_code[0],
                unstaged_status=status_code[1],
                tracked=True,
            )
        )
    return entries


def get_status_entries(repo_root: Path) -> list[GitStatusEntry]:
    output = _run_git(repo_root, ["status", "--short"])
    return parse_status_output(output)


def _null_path() -> str:
    return os.devnull


def get_diff_text(repo_root: Path, entry: GitStatusEntry) -> str:
    path = entry.path
    chunks: list[str] = []
    if entry.is_staged:
        chunks.append(_run_git(repo_root, ["diff", "--cached", "--", path]))
    if entry.is_unstaged:
        chunks.append(_run_git(repo_root, ["diff", "--", path]))
    if entry.is_untracked:
        chunks.append(
            _run_git(
                repo_root,
                ["diff", "--no-index", "--", _null_path(), path],
                allowed_returncodes={0, 1},
            )
        )
    return "\n".join(chunk for chunk in chunks if chunk.strip())
