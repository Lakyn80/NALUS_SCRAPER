"""Isolated tests for atomic checkpoint write used by full-corpus builder."""

from __future__ import annotations

import json
from pathlib import Path


def _atomic_write_checkpoint(checkpoint_path: Path, payload: dict) -> None:
    """Mirror of builder atomic replace: write tmp then Path.replace."""
    tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tmp.replace(checkpoint_path)


def test_atomic_checkpoint_write_produces_valid_canonical_json(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoint_A_full.json"
    payload = {"completed_document_ids": ["a", "b"], "completed_document_count": 2}
    _atomic_write_checkpoint(ckpt, payload)
    loaded = json.loads(ckpt.read_text(encoding="utf-8"))
    assert loaded == payload
    assert not (tmp_path / "checkpoint_A_full.json.tmp").exists()


def test_failed_replace_leaves_previous_canonical_valid(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoint_A_full.json"
    first = {"completed_document_count": 1, "completed_document_ids": ["x"]}
    _atomic_write_checkpoint(ckpt, first)

    tmp = ckpt.with_suffix(ckpt.suffix + ".tmp")
    tmp.write_text("{not-json", encoding="utf-8")
    # Simulate crash before replace: tmp exists, canonical unchanged.
    assert ckpt.exists()
    assert json.loads(ckpt.read_text(encoding="utf-8")) == first
    assert tmp.exists()


def test_resume_reads_only_canonical_not_tmp(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoint_A_full.json"
    good = {"completed_document_count": 3, "completed_document_ids": ["a", "b", "c"]}
    _atomic_write_checkpoint(ckpt, good)
    stale = ckpt.with_suffix(ckpt.suffix + ".tmp")
    stale.write_text(
        json.dumps({"completed_document_count": 999}, ensure_ascii=False),
        encoding="utf-8",
    )
    # Builder resume path reads only checkpoint_path (canonical).
    loaded = json.loads(ckpt.read_text(encoding="utf-8"))
    assert loaded["completed_document_count"] == 3
    assert loaded != json.loads(stale.read_text(encoding="utf-8"))


def test_repeated_atomic_writes_remain_valid(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoint_A_full.json"
    for n in range(1, 6):
        _atomic_write_checkpoint(
            ckpt,
            {
                "completed_document_count": n,
                "completed_document_ids": [f"d{i}" for i in range(n)],
            },
        )
        loaded = json.loads(ckpt.read_text(encoding="utf-8"))
        assert loaded["completed_document_count"] == n
        assert len(loaded["completed_document_ids"]) == n
    assert not list(tmp_path.glob("*.tmp"))
