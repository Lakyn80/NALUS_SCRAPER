from __future__ import annotations

import json

from app.rag.legal_v2.ingest.chunking import HierarchicalChunkConfig, build_hierarchical_chunks
from app.rag.legal_v2.parser import parse_legal_document
from app.rag.legal_v2.schema.canonical_v1 import (
    DEFAULT_CHUNKING_PROFILE,
    CanonicalBlock,
    CanonicalChildChunk,
    CanonicalDocument,
    CanonicalDocumentBundle,
    CanonicalParentContext,
    bundle_from_json,
    bundle_to_json,
    content_checksum,
    reconstruct_child_text,
    stable_block_id,
    stable_child_chunk_id,
    stable_parent_id,
    validate_bundle_invariants,
)
from app.rag.legal_v2.schema.map_from_legal_v2 import (
    LineInventoryRow,
    map_legal_v2_bundle,
)


def _sample_text() -> str:
    return (
        "Vrchní soud v Olomouci\n\n"
        "I.\n"
        "Výrok\n\n"
        "I. Odvolání se zamítá.\n\n"
        "II.\n"
        "Odůvodnění\n\n"
        "1. Soud prvního stupně rozhodl o žalobě.\n\n"
        "2. Odvolací soud přezkoumal napadené rozhodnutí a dospěl k závěru, "
        "že odvolání není důvodné, protože skutková zjištění jsou správná a "
        "právní posouzení odpovídá zákonu.\n\n"
        "Poučení\n\n"
        "Proti tomuto rozhodnutí není dovolání přípustné."
    )


def test_stable_ids_are_deterministic() -> None:
    block_a = stable_block_id(
        document_id="DOC-A",
        block_index=0,
        normalized_text="alpha",
    )
    block_b = stable_block_id(
        document_id="DOC-A",
        block_index=0,
        normalized_text="alpha",
    )
    assert block_a == block_b
    assert ":b:" in block_a

    child_a = stable_child_chunk_id(
        document_id="DOC-A",
        chunk_index=1,
        source_block_ids=["b1", "b2"],
        chunking_profile=DEFAULT_CHUNKING_PROFILE,
    )
    child_b = stable_child_chunk_id(
        document_id="DOC-A",
        chunk_index=1,
        source_block_ids=["b1", "b2"],
        chunking_profile=DEFAULT_CHUNKING_PROFILE,
    )
    assert child_a == child_b
    assert ":c:" in child_a

    parent_a = stable_parent_id(
        document_id="DOC-A",
        parent_index=0,
        child_ids=[child_a],
        chunking_profile=DEFAULT_CHUNKING_PROFILE,
    )
    parent_b = stable_parent_id(
        document_id="DOC-A",
        parent_index=0,
        child_ids=[child_a],
        chunking_profile=DEFAULT_CHUNKING_PROFILE,
    )
    assert parent_a == parent_b
    assert ":p:" in parent_a


def test_map_legal_v2_bundle_preserves_legacy_ids_and_passes_invariants() -> None:
    document = parse_legal_document(document_id="DOC-CANON", text=_sample_text())
    chunking = build_hierarchical_chunks(
        document,
        config=HierarchicalChunkConfig(
            child_target_min_tokens=20,
            child_target_max_tokens=120,
            child_hard_max_tokens=180,
            parent_target_min_tokens=40,
            parent_target_max_tokens=240,
            parent_hard_max_tokens=320,
            min_short_paragraph_tokens=10,
        ),
    )
    inventory = [
        LineInventoryRow(
            line_number=index + 1,
            text=paragraph.original_text.splitlines()[0] if paragraph.original_text else "",
            parser_block_id=paragraph.paragraph_id,
            parser_class=paragraph.section_type.value,
        )
        for index, paragraph in enumerate(document.paragraphs)
    ]
    bundle = map_legal_v2_bundle(
        document,
        chunking,
        line_inventory=inventory,
        source_document_id="SRC-1",
        source_checksum=content_checksum(_sample_text()),
    )

    assert bundle.document.document_id == "DOC-CANON"
    assert bundle.document.source_document_id == "SRC-1"
    assert bundle.document.parser_profile.startswith("legal-decision-parser")
    assert len(bundle.blocks) == len(document.paragraphs)
    assert {block.block_id for block in bundle.blocks} == {
        paragraph.paragraph_id for paragraph in document.paragraphs
    }
    assert {child.chunk_id for child in bundle.children} == {
        child.chunk_id for child in chunking.child_chunks
    }
    assert {parent.parent_id for parent in bundle.parents} == {
        window.window_id for window in chunking.parent_windows
    }
    assert all(child.chunking_profile == DEFAULT_CHUNKING_PROFILE for child in bundle.children)
    assert all(block.line_start is not None and block.line_end is not None for block in bundle.blocks)

    report = validate_bundle_invariants(bundle)
    assert report.ok, report
    assert report.failure_count == 0

    for child in bundle.children:
        assembled = reconstruct_child_text(child, bundle.blocks_by_id())
        assert assembled.strip()
        assert len(child.source_block_ids) == len(set(child.source_block_ids))

    for parent in bundle.parents:
        for child_id in parent.child_ids:
            child = bundle.children_by_id()[child_id]
            assert child.parent_id == parent.parent_id


def test_bundle_json_round_trip() -> None:
    document = parse_legal_document(document_id="DOC-JSON", text=_sample_text())
    chunking = build_hierarchical_chunks(
        document,
        config=HierarchicalChunkConfig(
            child_target_min_tokens=20,
            child_target_max_tokens=120,
            child_hard_max_tokens=180,
            parent_target_min_tokens=40,
            parent_target_max_tokens=240,
            parent_hard_max_tokens=320,
            min_short_paragraph_tokens=10,
        ),
    )
    bundle = map_legal_v2_bundle(document, chunking)
    raw = bundle_to_json(bundle)
    restored = bundle_from_json(raw)
    assert restored.document.document_id == bundle.document.document_id
    assert len(restored.blocks) == len(bundle.blocks)
    assert len(restored.children) == len(bundle.children)
    assert len(restored.parents) == len(bundle.parents)
    assert json.loads(bundle_to_json(restored)) == json.loads(raw)
    assert validate_bundle_invariants(restored).ok


def test_parent_must_claim_only_own_children() -> None:
    document = CanonicalDocument(document_id="DOC-X")
    block = CanonicalBlock(
        block_id="b1",
        document_id="DOC-X",
        block_index=0,
        raw_text="Alpha sentence for reconstruction.",
        normalized_text="Alpha sentence for reconstruction.",
        primary_class="court_reasoning",
        source_checksum=content_checksum("Alpha sentence for reconstruction."),
    )
    child = CanonicalChildChunk(
        chunk_id="c1",
        document_id="DOC-X",
        source_block_ids=["b1"],
        chunk_text="Alpha sentence for reconstruction.",
        embedding_text="Alpha sentence for reconstruction.",
        token_count=4,
        chunking_profile=DEFAULT_CHUNKING_PROFILE,
        content_checksum=content_checksum("Alpha sentence for reconstruction."),
        parent_id="p1",
    )
    parent = CanonicalParentContext(
        parent_id="p1",
        document_id="DOC-X",
        child_ids=["c1", "missing-child"],
        parent_text="Alpha sentence for reconstruction.",
        context_type="court_reasoning",
        token_count=4,
        content_checksum=content_checksum("Alpha sentence for reconstruction."),
    )
    bundle = CanonicalDocumentBundle(
        document=document,
        blocks=[block],
        children=[child],
        parents=[parent],
    )
    report = validate_bundle_invariants(bundle)
    assert not report.ok
    assert any("missing_child" in item for item in report.parent_child_inconsistencies)
