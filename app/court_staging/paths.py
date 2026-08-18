"""Path guards: court staging must never write into Full B frozen inputs."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COURT_STAGING_ROOT = PROJECT_ROOT / "artifacts" / "court_staging"

_FORBIDDEN_PATH_MARKERS = (
    "batches",
    "full_corpus_build_v1",
    "eligible_document_ids.txt",
    "checkpoint_B_full.json",
    "nalus_legal_paragraph_bm25_v2_chunk_ab_v8_b_contextual_full",
    "nalus_legal_paragraph_chunks_v2_chunk_ab_v8_b_contextual_full",
)


def default_staging_root() -> Path:
    return COURT_STAGING_ROOT


def assert_safe_staging_path(path: Path, *, staging_root: Path | None = None) -> Path:
    """Reject writes under batches/, Full B inventory, or Full B BM25/Qdrant artifacts.

    Returns the resolved absolute path.
    """
    resolved = path.expanduser().resolve()
    root = (staging_root or COURT_STAGING_ROOT).expanduser().resolve()

    # Prefer staging_root containment when caller intends court staging.
    try:
        resolved.relative_to(root)
        under_staging = True
    except ValueError:
        under_staging = False

    lowered_parts = {part.lower() for part in resolved.parts}
    name = resolved.name.lower()

    if "batches" in lowered_parts and resolved.parent.name.lower() != "court_staging":
        # Allow .../court_staging/.../batches-like names only if under staging root.
        if not under_staging:
            raise ValueError(
                f"Refusing path under batches/ (Full B frozen input): {resolved}"
            )

    # Hard reject common Full B / inventory paths even if somehow nested oddly.
    path_str = str(resolved).replace("\\", "/").lower()
    for marker in _FORBIDDEN_PATH_MARKERS:
        marker_l = marker.lower()
        if marker_l in path_str and not under_staging:
            raise ValueError(f"Refusing protected Full B / corpus path ({marker}): {resolved}")

    if name == "manifest.json" and "batches" in path_str and not under_staging:
        raise ValueError(f"Refusing batches manifest path: {resolved}")

    if not under_staging:
        # Still allow explicit non-staging outs only when they do not hit forbidden markers.
        for marker in ("\\batches\\", "/batches/", "full_corpus_build_v1"):
            if marker in path_str:
                raise ValueError(f"Refusing protected path: {resolved}")

    return resolved


def ensure_staging_tree(staging_root: Path | None = None) -> Path:
    root = assert_safe_staging_path(
        staging_root or COURT_STAGING_ROOT,
        staging_root=staging_root or COURT_STAGING_ROOT,
    )
    for relative in (
        "us/incremental",
        "ns/historical",
        "ns/incremental",
        "nss/historical/pilot",
        "nss/incremental",
        "updater",
        "merge_dry_run",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
        keep = root / relative / ".gitkeep"
        if not keep.exists():
            keep.write_text("", encoding="utf-8")
    return root
