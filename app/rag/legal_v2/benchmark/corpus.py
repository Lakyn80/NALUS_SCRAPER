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
_LINE_PREFIX_RE = re.compile(r"^\d{5}:\s?")


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


def load_development_corpus(
    *,
    archetypes_path: Path | str = DEFAULT_ARCHETYPES_PATH,
    review_dir: Path | str = DEFAULT_REVIEW_DIR,
) -> DevelopmentCorpus:
    review_root = Path(review_dir)
    refs = load_development_document_refs(archetypes_path)
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
            "archetype_role": "development",
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
    "load_development_document_refs",
    "load_development_corpus",
    "token_overlap_score",
    "rank_blocks_by_token_overlap",
]
