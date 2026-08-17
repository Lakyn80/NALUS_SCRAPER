"""Shared pytest hooks for optional local artifact-backed suites."""

from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]

_REVIEW_MANIFEST = _ROOT / "artifacts/legal_v2/visual_parser_review/review_manifest.json"
_DESIGN_MANIFEST = _ROOT / "artifacts/legal_v2/court_format_study/design_sample_manifest.json"
_GOLDEN_SPEC = _ROOT / "artifacts/legal_v2/parser_golden_inputs/corrected_golden_spec.json"

# Relative to tests/ — modules that need durable local artifacts (gitignored).
_REQUIRES_REVIEW_ARTIFACTS = frozenset(
    {
        "legal_v2/test_parser_v6_full_export.py",
        "legal_v2/test_visual_parser_review.py",
        "legal_v2/test_visual_parser_review_status.py",
        "legal_v2/test_visual_parser_review_validation.py",
        "legal_v2/test_visual_parser_review_web.py",
        "rag/test_legal_v2_case_similarity_golden_v1_pilot.py",
        "rag/test_legal_v2_retrieval_golden_v1_pilot.py",
        "rag/test_legal_v2_constitutional_profile.py",
    }
)
_REQUIRES_DESIGN_MANIFEST = frozenset(
    {
        "legal_v2/test_visual_parser_review.py",
        "rag/test_legal_v2_parser_v6_goldens.py",
        "rag/test_legal_v2_parser_v7_targeted.py",
        "rag/test_legal_v2_constitutional_profile.py",
    }
)
_REQUIRES_GOLDEN_SPEC = frozenset(
    {
        "rag/test_legal_v2_parser_v6_goldens.py",
        "legal_v2/test_parser_v6_full_export.py",
    }
)


def _module_rel(item: pytest.Item) -> str:
    path = Path(str(item.fspath)).resolve()
    try:
        return path.relative_to(_ROOT / "tests").as_posix()
    except ValueError:
        return path.name


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    del config  # unused; required by pytest hook signature
    skip_review = pytest.mark.skip(
        reason="missing local artifacts/legal_v2/visual_parser_review (gitignored)"
    )
    skip_design = pytest.mark.skip(
        reason="missing local artifacts/legal_v2/court_format_study design manifest (gitignored)"
    )
    skip_golden = pytest.mark.skip(
        reason="missing local artifacts/legal_v2/parser_golden_inputs (gitignored)"
    )
    has_review = _REVIEW_MANIFEST.is_file()
    has_design = _DESIGN_MANIFEST.is_file()
    has_golden = _GOLDEN_SPEC.is_file()
    for item in items:
        rel = _module_rel(item)
        if not has_review and rel in _REQUIRES_REVIEW_ARTIFACTS:
            item.add_marker(skip_review)
        if not has_design and rel in _REQUIRES_DESIGN_MANIFEST:
            item.add_marker(skip_design)
        if not has_golden and rel in _REQUIRES_GOLDEN_SPEC:
            item.add_marker(skip_golden)
