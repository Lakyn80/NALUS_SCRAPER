"""Load development-role canonical blocks for retrieval-golden grounding/validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.rag.legal_v2.parser import parse_legal_document
from app.rag.legal_v2.schema.canonical_v1 import CanonicalBlock, CanonicalDocumentBundle
from app.rag.legal_v2.schema.map_from_legal_v2 import line_inventory_from_review_rows, map_legal_v2_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ARCHETYPES_PATH = (
    PROJECT_ROOT / "docs" / "architecture" / "parser_benchmark" / "archetypes_v1.json"
)
DEFAULT_REVIEW_DIR = PROJECT_ROOT / "artifacts" / "legal_v2" / "visual_parser_review"
DEFAULT_RAW_SOURCES_DIR = (
    PROJECT_ROOT / "artifacts" / "legal_v2" / "court_format_study" / "raw_sources"
)
_LINE_PREFIX_RE = re.compile(r"^\d{5}:\s?")

# Supplemental criminal appeals available in local raw_sources but outside the
# 20-document reviewed pilot pool. Used only as hard-negative candidates.
CASE_SIMILARITY_SUPPLEMENTAL_CRIMINAL_SOURCES: tuple[dict[str, str], ...] = (
    {
        "document_id": "doc-4fbdc1db957f44e7",
        "source_id": "4fbdc1db-957f-44e7-a9d3-18abddb9cdce",
        "relative_path": "high_court_olomouc/4fbdc1db-957f-44e7-a9d3-18abddb9cdce.json",
        "court": "high_court_olomouc",
        "decision_type": "criminal_appeal",
        "case_number": "6 To 41/2024",
        "decision_date": "2025-02-20",
    },
    {
        "document_id": "doc-68c126d146c84fa1",
        "source_id": "68c126d1-46c8-4fa1-aa5b-0602e2e7bb6e",
        "relative_path": "high_court_olomouc/68c126d1-46c8-4fa1-aa5b-0602e2e7bb6e.json",
        "court": "high_court_olomouc",
        "decision_type": "criminal_appeal",
        "case_number": "6 To 42/2024",
        "decision_date": "2025-02-27",
    },
)


@dataclass(frozen=True)
class DevelopmentDocumentRef:
    archetype_id: str
    review_number: int
    document_id: str
    source_id: str | None
    case_number: str | None
    court: str | None
    decision_type: str | None
    decision_date: str | None
    source_checksum: str | None
    ecli: str | None = None
    canonical_document_id: str | None = None
    identity_status: str | None = None


@dataclass(frozen=True)
class DevelopmentCorpus:
    documents: list[DevelopmentDocumentRef]
    bundles: dict[str, CanonicalDocumentBundle]
    blocks_by_id: dict[str, CanonicalBlock]

    def block(self, block_id: str) -> CanonicalBlock:
        return self.blocks_by_id[block_id]

    def blocks_for_document(self, document_id: str) -> list[CanonicalBlock]:
        return [block for block in self.blocks_by_id.values() if block.document_id == document_id]


def load_development_document_refs(
    archetypes_path: Path | str = DEFAULT_ARCHETYPES_PATH,
) -> list[DevelopmentDocumentRef]:
    payload = json.loads(Path(archetypes_path).read_text(encoding="utf-8"))
    inventory = {
        int(item["review_number"]): item
        for item in payload.get("inventory", [])
        if item.get("review_number") is not None
    }
    refs: list[DevelopmentDocumentRef] = []
    seen: set[str] = set()
    for archetype in payload.get("archetypes", []):
        development = archetype.get("development") or {}
        if development.get("status") != "assigned":
            continue
        document_id = development.get("document_id")
        if not document_id or document_id in seen:
            continue
        review_number = int(development["review_number"])
        inv = inventory.get(review_number, {})
        refs.append(
            DevelopmentDocumentRef(
                archetype_id=str(archetype.get("archetype_id") or ""),
                review_number=review_number,
                document_id=str(document_id),
                source_id=_first(development.get("source_id"), inv.get("source_id")),
                case_number=_first(development.get("case_number"), inv.get("case_number")),
                court=_first(inv.get("court"), archetype.get("court")),
                decision_type=_first(inv.get("document_type"), archetype.get("decision_type")),
                decision_date=_first(inv.get("decision_date")),
                source_checksum=_first(development.get("source_checksum"), inv.get("source_checksum")),
            )
        )
        seen.add(str(document_id))
    return refs


def load_reviewed_pool_document_refs(
    archetypes_path: Path | str = DEFAULT_ARCHETYPES_PATH,
) -> list[DevelopmentDocumentRef]:
    """Load all reviewed inventory documents (parser design pool of 20)."""
    payload = json.loads(Path(archetypes_path).read_text(encoding="utf-8"))
    refs: list[DevelopmentDocumentRef] = []
    seen: set[str] = set()
    for item in payload.get("inventory", []):
        document_id = item.get("document_id")
        review_number = item.get("review_number")
        if not document_id or review_number is None:
            continue
        if document_id in seen:
            continue
        refs.append(
            DevelopmentDocumentRef(
                archetype_id="",
                review_number=int(review_number),
                document_id=str(document_id),
                source_id=_first(item.get("source_id")),
                case_number=_first(item.get("case_number")),
                court=_first(item.get("court")),
                decision_type=_first(item.get("document_type")),
                decision_date=_first(item.get("decision_date")),
                source_checksum=_first(item.get("source_checksum")),
            )
        )
        seen.add(str(document_id))
    refs.sort(key=lambda ref: ref.review_number)
    return refs


def load_development_corpus(
    *,
    archetypes_path: Path | str = DEFAULT_ARCHETYPES_PATH,
    review_dir: Path | str = DEFAULT_REVIEW_DIR,
) -> DevelopmentCorpus:
    refs = load_development_document_refs(archetypes_path)
    return _load_corpus_for_refs(
        refs,
        archetypes_path=archetypes_path,
        review_dir=review_dir,
        archetype_role="development",
    )


def load_reviewed_pool_corpus(
    *,
    archetypes_path: Path | str = DEFAULT_ARCHETYPES_PATH,
    review_dir: Path | str = DEFAULT_REVIEW_DIR,
) -> DevelopmentCorpus:
    """Canonical corpus for all 20 reviewed parser-v7 judgments in the design pool."""
    refs = load_reviewed_pool_document_refs(archetypes_path)
    return _load_corpus_for_refs(
        refs,
        archetypes_path=archetypes_path,
        review_dir=review_dir,
        archetype_role="reviewed_pool",
    )


def load_case_similarity_corpus(
    *,
    archetypes_path: Path | str = DEFAULT_ARCHETYPES_PATH,
    review_dir: Path | str = DEFAULT_REVIEW_DIR,
    raw_sources_dir: Path | str = DEFAULT_RAW_SOURCES_DIR,
) -> DevelopmentCorpus:
    """Reviewed pilot pool plus supplemental criminal hard-negative sources."""
    base = load_reviewed_pool_corpus(
        archetypes_path=archetypes_path,
        review_dir=review_dir,
    )
    extra_refs, extra_bundles, extra_blocks = _load_supplemental_criminal_sources(
        Path(raw_sources_dir)
    )
    documents = _enrich_refs_with_identity(list(base.documents) + extra_refs)
    bundles = dict(base.bundles)
    bundles.update(extra_bundles)
    blocks_by_id = dict(base.blocks_by_id)
    blocks_by_id.update(extra_blocks)
    return DevelopmentCorpus(
        documents=documents,
        bundles=bundles,
        blocks_by_id=blocks_by_id,
    )


def _enrich_refs_with_identity(
    refs: list[DevelopmentDocumentRef],
) -> list[DevelopmentDocumentRef]:
    try:
        from app.rag.legal_v2.benchmark.case_similarity_identity import (
            load_case_similarity_identity_map,
        )

        identity_map = load_case_similarity_identity_map()
    except Exception:  # noqa: BLE001
        return refs
    enriched: list[DevelopmentDocumentRef] = []
    for ref in refs:
        row = identity_map.get(ref.document_id)
        if not row:
            enriched.append(ref)
            continue
        enriched.append(
            DevelopmentDocumentRef(
                archetype_id=ref.archetype_id,
                review_number=ref.review_number,
                document_id=ref.document_id,
                source_id=ref.source_id,
                case_number=ref.case_number,
                court=ref.court,
                decision_type=ref.decision_type,
                decision_date=ref.decision_date,
                source_checksum=ref.source_checksum,
                ecli=row.get("ecli"),
                canonical_document_id=row.get("canonical_document_id"),
                identity_status=row.get("identity_status"),
            )
        )
    return enriched


def load_case_similarity_primary_document_ids(
    archetypes_path: Path | str = DEFAULT_ARCHETYPES_PATH,
) -> list[str]:
    """The 20 reviewed-pool document IDs that must each appear once as a primary."""
    return [ref.document_id for ref in load_reviewed_pool_document_refs(archetypes_path)]


def _load_supplemental_criminal_sources(
    raw_sources_dir: Path,
) -> tuple[list[DevelopmentDocumentRef], dict[str, CanonicalDocumentBundle], dict[str, CanonicalBlock]]:
    refs: list[DevelopmentDocumentRef] = []
    bundles: dict[str, CanonicalDocumentBundle] = {}
    blocks_by_id: dict[str, CanonicalBlock] = {}
    for index, item in enumerate(CASE_SIMILARITY_SUPPLEMENTAL_CRIMINAL_SOURCES, start=1001):
        path = raw_sources_dir / item["relative_path"]
        if not path.exists():
            raise FileNotFoundError(f"Missing supplemental criminal source: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        text = _justice_json_plain_text(
            payload,
            court_label="Vrchní soud v Olomouci",
            case_number=item["case_number"],
        )
        ref = DevelopmentDocumentRef(
            archetype_id="supplemental_criminal_hard_negative",
            review_number=index,
            document_id=item["document_id"],
            source_id=item["source_id"],
            case_number=item["case_number"],
            court=item["court"],
            decision_type=item["decision_type"],
            decision_date=item["decision_date"],
            source_checksum=None,
        )
        metadata = {
            "source_id": ref.source_id,
            "source_document_id": ref.document_id,
            "ecli": (payload.get("metadata") or {}).get("ecli"),
            "canonical_document_id": (payload.get("metadata") or {}).get("ecli"),
            "case_number": ref.case_number,
            "court": ref.court,
            "decision_type": ref.decision_type,
            "document_type": ref.decision_type,
            "decision_date": ref.decision_date,
            "language": "cs",
            "jurisdiction": "CZ",
            "archetype_id": ref.archetype_id,
            "archetype_role": "supplemental_hard_negative",
        }
        parsed = parse_legal_document(
            document_id=ref.document_id,
            text=text,
            metadata=metadata,
        )
        bundle = map_legal_v2_bundle(
            parsed,
            source_document_id=ref.source_id,
        )
        refs.append(ref)
        bundles[ref.document_id] = bundle
        for block in bundle.blocks:
            blocks_by_id[block.block_id] = block
    return refs, bundles, blocks_by_id


def _justice_json_plain_text(
    payload: dict[str, Any],
    *,
    court_label: str,
    case_number: str,
) -> str:
    """Minimal offline extractor for Justice Open Data finaldoc JSON."""
    lines: list[str] = [court_label, case_number]
    for key, heading in (
        ("header", None),
        ("verdict", "Výrok"),
        ("justification", "Odůvodnění"),
    ):
        blocks = payload.get(key)
        if heading and blocks:
            lines.append(heading)
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            texts = block.get("texts")
            if not isinstance(texts, list):
                continue
            joined = "".join(
                str(part.get("text") or "") for part in texts if isinstance(part, dict)
            )
            normalized = re.sub(r"\s+", " ", joined).strip()
            if normalized:
                lines.append(normalized)
    plain = "\n".join(lines)
    return plain if plain.endswith("\n") else plain + "\n"


def _load_corpus_for_refs(
    refs: list[DevelopmentDocumentRef],
    *,
    archetypes_path: Path | str,
    review_dir: Path | str,
    archetype_role: str,
) -> DevelopmentCorpus:
    review_root = Path(review_dir)
    _ = archetypes_path  # reserved for future archetype-role enrichment
    bundles: dict[str, CanonicalDocumentBundle] = {}
    blocks_by_id: dict[str, CanonicalBlock] = {}
    line_rows = _safe_read_jsonl(review_root / "review_lines.jsonl")
    for ref in refs:
        text = _load_plain_text(review_root, ref.document_id)
        doc_lines = [row for row in line_rows if row.get("document_id") == ref.document_id]
        metadata = {
            "source_id": ref.source_id,
            "source_document_id": ref.source_id,
            "case_number": ref.case_number,
            "court": ref.court,
            "decision_type": ref.decision_type,
            "document_type": ref.decision_type,
            "decision_date": ref.decision_date,
            "source_checksum": ref.source_checksum,
            "language": "cs",
            "jurisdiction": "CZ",
            "archetype_id": ref.archetype_id,
            "archetype_role": archetype_role,
        }
        parsed = parse_legal_document(
            document_id=ref.document_id,
            text=text,
            metadata=metadata,
        )
        bundle = map_legal_v2_bundle(
            parsed,
            line_inventory=line_inventory_from_review_rows(doc_lines),
            source_document_id=ref.source_id,
            source_checksum=ref.source_checksum,
        )
        bundles[ref.document_id] = bundle
        for block in bundle.blocks:
            blocks_by_id[block.block_id] = block
    return DevelopmentCorpus(documents=refs, bundles=bundles, blocks_by_id=blocks_by_id)


def token_overlap_score(query: str, text: str) -> float:
    q_tokens = _tokens(query)
    t_tokens = _tokens(text)
    if not q_tokens or not t_tokens:
        return 0.0
    overlap = q_tokens & t_tokens
    return len(overlap) / float(len(q_tokens))


def rank_blocks_by_token_overlap(
    query: str,
    blocks: list[CanonicalBlock],
    *,
    top_k: int = 10,
) -> list[tuple[CanonicalBlock, float]]:
    scored = [(block, token_overlap_score(query, block.raw_text)) for block in blocks]
    scored.sort(key=lambda item: (-item[1], item[0].document_id, item[0].block_index))
    return scored[:top_k]


def _load_plain_text(review_dir: Path, document_id: str) -> str:
    path = review_dir / "documents" / document_id / "raw_numbered.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing review text for {document_id}: {path}")
    raw = path.read_text(encoding="utf-8")
    lines = [_LINE_PREFIX_RE.sub("", line, count=1) for line in raw.splitlines()]
    plain = "\n".join(lines)
    return plain if plain.endswith("\n") else plain + "\n"


def _safe_read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[0-9A-Za-zÁ-Žá-ž§]+", value.casefold()) if len(token) >= 3}


def _first(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


__all__ = [
    "DevelopmentDocumentRef",
    "DevelopmentCorpus",
    "DEFAULT_ARCHETYPES_PATH",
    "DEFAULT_REVIEW_DIR",
    "DEFAULT_RAW_SOURCES_DIR",
    "CASE_SIMILARITY_SUPPLEMENTAL_CRIMINAL_SOURCES",
    "load_development_document_refs",
    "load_reviewed_pool_document_refs",
    "load_development_corpus",
    "load_reviewed_pool_corpus",
    "load_case_similarity_corpus",
    "load_case_similarity_primary_document_ids",
    "token_overlap_score",
    "rank_blocks_by_token_overlap",
]
