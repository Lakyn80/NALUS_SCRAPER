from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api_app import app
from app.rag.legal_v2.adapters import LegalSourceDocument
from app.rag.legal_v2.audit import audit_documents
from app.rag.legal_v2.index_builder import LegalV2BuildConfig, build_legal_v2_index
from app.rag.legal_v2.index_builder import LegalV2CheckpointStop
from app.rag.legal_v2.interpreter import (
    DeterministicQuerySpecProvider,
    interpret_query_spec_v2,
)
from app.rag.legal_v2.pipeline import search_legal_v2
from app.rag.legal_v2.query_spec import build_query_spec_v2
from app.rag.legal_v2.retriever import LegalV2HybridRetriever, LegalV2RetrieverConfig, _aggregate_documents
from app.rag.legal_v2.sources import DecisionDateRange, filter_source_documents_by_decision_date, parse_decision_date
from app.rag.legal_v2.verifier import ConstraintVerificationStatus
from app.rag.retrieval.bm25_sidecar import Bm25Record, Bm25Sidecar
from app.rag.retrieval.models import RetrievedChunk
from scripts.legal_v2.build_index import _document_ids_from_parser_quality, _require_gate_pass


def _source_document(document_id: str = "III. ÚS 2923/23") -> LegalSourceDocument:
    return LegalSourceDocument(
        document_id=document_id,
        source="constitutional",
        text="\n\n".join(
            [
                "ÚSTAVNÍ SOUD",
                "I. SKUTKOVÝ STAV",
                "[1] Matka neoprávněně přemístila dítě z Česka do Ruska.",
                "II. ODŮVODNĚNÍ",
                "[2] Soud posoudil návrat dítěte podle Haagské úmluvy.",
            ]
        ),
        metadata={"case_reference": document_id, "decision_date": "1. 1. 2026"},
        origin_path="memory",
    )


def test_parse_audit_excludes_material_failures() -> None:
    report = audit_documents(
        [
            _source_document(),
            LegalSourceDocument("BROKEN", "constitutional", "", {}, "memory"),
        ]
    )

    assert report.summary["total_documents"] == 2
    assert report.summary["successfully_parsed_documents"] == 1
    assert report.summary["failed_documents"] == 1
    assert report.summary["documents_excluded_from_indexing"][0]["document_id"] == "BROKEN"
    assert report.summary["deepseek_calls"] == 0
    assert report.summary["qdrant_writes"] == 0


def test_index_builder_writes_only_v2_identities(tmp_path: Path) -> None:
    class FakeEmbedder:
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.1] * 1024 for _ in texts]

    class FakeClient:
        def __init__(self) -> None:
            self.collections: set[str] = set()
            self.upserts: list[tuple[str, list]] = []

        def get_collections(self):
            return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in self.collections])

        def create_collection(self, collection_name: str, vectors_config) -> None:  # noqa: ANN001
            del vectors_config
            self.collections.add(collection_name)

        def upsert(self, collection_name: str, points: list) -> None:
            self.upserts.append((collection_name, points))

    client = FakeClient()
    bm25_path = tmp_path / "bm25.sqlite"
    manifest = build_legal_v2_index(
        documents=[_source_document(), LegalSourceDocument("BROKEN", "constitutional", "", {}, "memory")],
        embedder=FakeEmbedder(),
        qdrant_client=client,
        config=LegalV2BuildConfig(bm25_path=bm25_path, output_dir=tmp_path, overwrite_bm25=True),
        git_commit="test",
        dirty=False,
    )

    assert manifest.collection_name == "nalus_legal_paragraph_chunks_v2"
    assert manifest.indexed_document_count == 1
    assert manifest.excluded_document_count == 1
    assert manifest.chunk_count > 0
    assert bm25_path.exists()
    assert {name for name, _ in client.upserts} == {"nalus_legal_paragraph_chunks_v2"}
    payload = client.upserts[0][1][0].payload
    assert payload["source"] == "constitutional"
    assert payload["parent_window_id"]
    assert payload["parent_window_paragraph_ids"]
    assert "parent_window_truncated" in payload
    stored = json.loads((tmp_path / "legal_v2_build_manifest.json").read_text(encoding="utf-8"))
    assert stored["bm25_index_id"] == "nalus_legal_paragraph_bm25_v2"
    assert stored["qdrant_upsert_points"] == manifest.chunk_count


def test_index_builder_writes_configured_pilot_identities(tmp_path: Path) -> None:
    class FakeEmbedder:
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.1] * 1024 for _ in texts]

    class FakeClient:
        def __init__(self) -> None:
            self.collections: set[str] = set()
            self.upserts: list[tuple[str, list]] = []

        def get_collections(self):
            return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in self.collections])

        def create_collection(self, collection_name: str, vectors_config) -> None:  # noqa: ANN001
            del vectors_config
            self.collections.add(collection_name)

        def upsert(self, collection_name: str, points: list, wait: bool = True) -> None:
            del wait
            self.upserts.append((collection_name, points))

    client = FakeClient()
    bm25_path = tmp_path / "pilot.sqlite"
    manifest = build_legal_v2_index(
        documents=[_source_document()],
        embedder=FakeEmbedder(),
        qdrant_client=client,
        config=LegalV2BuildConfig(
            collection_name="nalus_legal_paragraph_chunks_v2_pilot_600",
            bm25_index_id="nalus_legal_paragraph_bm25_v2_pilot_600",
            bm25_path=bm25_path,
            output_dir=tmp_path,
            overwrite_bm25=True,
        ),
        git_commit="test",
        dirty=False,
    )

    assert manifest.collection_name == "nalus_legal_paragraph_chunks_v2_pilot_600"
    assert manifest.bm25_index_id == "nalus_legal_paragraph_bm25_v2_pilot_600"
    assert {name for name, _ in client.upserts} == {"nalus_legal_paragraph_chunks_v2_pilot_600"}
    payload = client.upserts[0][1][0].payload
    assert payload["qdrant_collection"] == "nalus_legal_paragraph_chunks_v2_pilot_600"
    assert payload["bm25_index_id"] == "nalus_legal_paragraph_bm25_v2_pilot_600"
    with sqlite3.connect(bm25_path) as connection:
        row = connection.execute(
            "SELECT qdrant_collection, bm25_index_id, metadata FROM bm25_chunks LIMIT 1"
        ).fetchone()
    assert row[0] == "nalus_legal_paragraph_chunks_v2_pilot_600"
    assert row[1] == "nalus_legal_paragraph_bm25_v2_pilot_600"
    assert json.loads(row[2])["qdrant_collection"] == "nalus_legal_paragraph_chunks_v2_pilot_600"


def test_index_builder_rejects_pilot_collection_with_canonical_bm25_id(tmp_path: Path) -> None:
    config = LegalV2BuildConfig(
        collection_name="nalus_legal_paragraph_chunks_v2_pilot_600",
        bm25_path=tmp_path / "pilot.sqlite",
    )

    with pytest.raises(ValueError, match="non-canonical BM25 index id"):
        config.validate()


def test_index_builder_embeds_and_upserts_in_configured_batches(tmp_path: Path) -> None:
    class FakeEmbedder:
        def __init__(self) -> None:
            self.batch_lengths: list[int] = []

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            self.batch_lengths.append(len(texts))
            return [[0.1] * 1024 for _ in texts]

    class FakeClient:
        def __init__(self) -> None:
            self.collections: set[str] = set()
            self.upserts: list[tuple[str, list]] = []

        def get_collections(self):
            return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in self.collections])

        def create_collection(self, collection_name: str, vectors_config) -> None:  # noqa: ANN001
            del vectors_config
            self.collections.add(collection_name)

        def upsert(self, collection_name: str, points: list) -> None:
            self.upserts.append((collection_name, points))

    embedder = FakeEmbedder()
    manifest = build_legal_v2_index(
        documents=[_source_document("III. ÚS 2923/23"), _source_document("II. ÚS 859/23")],
        embedder=embedder,
        qdrant_client=FakeClient(),
        config=LegalV2BuildConfig(
            bm25_path=tmp_path / "bm25.sqlite",
            output_dir=tmp_path,
            overwrite_bm25=True,
            batch_size=1,
        ),
        git_commit="test",
        dirty=False,
    )

    assert manifest.qdrant_upsert_batches == manifest.chunk_count
    assert manifest.qdrant_upsert_points == manifest.chunk_count
    assert embedder.batch_lengths
    assert max(embedder.batch_lengths) == 1


def test_source_date_filter_excludes_old_and_missing_dates() -> None:
    old = _source_document("OLD")
    old = LegalSourceDocument(
        document_id=old.document_id,
        source=old.source,
        text=old.text,
        metadata={"decision_date": "30. 7. 2020"},
        origin_path=old.origin_path,
    )
    boundary = _source_document("BOUNDARY")
    boundary = LegalSourceDocument(
        document_id=boundary.document_id,
        source=boundary.source,
        text=boundary.text,
        metadata={"decision_date": "2020-07-31"},
        origin_path=boundary.origin_path,
    )
    missing = LegalSourceDocument("MISSING", "constitutional", "text", {}, "memory")

    result = filter_source_documents_by_decision_date(
        [old, boundary, missing],
        DecisionDateRange(
            date_from=parse_decision_date("2020-07-31"),
            date_to=parse_decision_date("2026-07-31"),
        ),
    )

    assert [document.document_id for document in result.documents] == ["BOUNDARY"]
    assert result.summary["date_out_of_range_document_count"] == 1
    assert result.summary["date_missing_or_invalid_document_count"] == 1


def test_index_builder_checkpoint_stop_and_resume(tmp_path: Path) -> None:
    class FakeEmbedder:
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.1] * 1024 for _ in texts]

    class FakeClient:
        def __init__(self) -> None:
            self.collections: set[str] = set()
            self.upserts: list[tuple[str, list]] = []

        def get_collections(self):
            return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in self.collections])

        def create_collection(self, collection_name: str, vectors_config) -> None:  # noqa: ANN001
            del vectors_config
            self.collections.add(collection_name)

        def upsert(self, collection_name: str, points: list) -> None:
            self.upserts.append((collection_name, points))

    client = FakeClient()
    documents = [_source_document("III. ÚS 2923/23"), _source_document("II. ÚS 859/23")]
    checkpoint_path = tmp_path / "checkpoint.json"
    bm25_path = tmp_path / "bm25.sqlite"

    with pytest.raises(LegalV2CheckpointStop):
        build_legal_v2_index(
            documents=documents,
            embedder=FakeEmbedder(),
            qdrant_client=client,
            config=LegalV2BuildConfig(
                bm25_path=bm25_path,
                output_dir=tmp_path,
                overwrite_bm25=True,
                document_batch_size=1,
                checkpoint_path=checkpoint_path,
                stop_after_document_batches=1,
            ),
            git_commit="test",
            dirty=False,
        )

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["completed_document_count"] == 1

    manifest = build_legal_v2_index(
        documents=documents,
        embedder=FakeEmbedder(),
        qdrant_client=client,
        config=LegalV2BuildConfig(
            bm25_path=bm25_path,
            output_dir=tmp_path,
            resume=True,
            document_batch_size=1,
            checkpoint_path=checkpoint_path,
        ),
        git_commit="test",
        dirty=False,
    )

    assert manifest.indexed_document_count == 2
    assert manifest.qdrant_upsert_points == manifest.chunk_count
    assert not checkpoint_path.exists()


def test_query_interpreter_merges_lost_mother_role() -> None:
    spec = build_query_spec_v2("únos dítěte matkou z Česka do Ruska").to_dict()
    spec["entities"] = [entity for entity in spec["entities"] if entity.get("role") != "mother"]
    result = interpret_query_spec_v2(
        "únos dítěte matkou z Česka do Ruska",
        provider=DeterministicQuerySpecProvider(spec),
    )

    assert result.status == "ok"
    assert result.query_spec is not None
    assert any(entity.role == "mother" for entity in result.query_spec.entities)
    assert result.reason is not None
    assert "query_interpreter_merged:entity_roles" in result.reason


def test_hybrid_retriever_aggregates_dense_bm25_rrf_and_paragraphs() -> None:
    payload = {
        "document_id": "DOC-1",
        "chunk_id": "chunk-1",
        "paragraph_ids": ["p1"],
        "paragraph_texts": {"p1": "Matka přemístila dítě z Česka do Ruska."},
        "section_type": "facts",
        "chunk_index": 0,
        "source_order": 0,
    }

    class Dense:
        def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
            del query, top_k
            return [RetrievedChunk("chunk-1", payload["paragraph_texts"]["p1"], 0.9, "dense", dict(payload))]

    bm25 = Bm25Sidecar.from_records(
        [Bm25Record("chunk-1", payload["paragraph_texts"]["p1"], dict(payload))],
        k1=1.5,
        b=0.75,
        index_id="nalus_legal_paragraph_bm25_v2",
    )
    retriever = LegalV2HybridRetriever(
        dense_store=Dense(),
        bm25_sidecar=bm25,
        config=LegalV2RetrieverConfig(dense_candidate_chunks=5, bm25_candidate_chunks=5),
    )

    result = retriever.retrieve(build_query_spec_v2("únos dítěte matkou z Česka do Ruska"))

    assert result.dense_results
    assert result.bm25_results
    assert result.fused_results
    assert result.documents[0].document_id == "DOC-1"
    assert result.documents[0].paragraphs[0].paragraph_id == "p1"


def test_document_aggregation_uses_general_constraint_coverage_bonus() -> None:
    spec = build_query_spec_v2("soud mi nedoručil rozsudek a zmeškal jsem odvolání")
    low_score_broad_evidence = RetrievedChunk(
        "broad-1",
        "Vadné doručení soudního rozhodnutí vedlo ke zmeškání lhůty pro odvolání.",
        0.10,
        "rrf",
        {
            "document_id": "DOC-BROAD",
            "paragraph_ids": ["broad-p1"],
            "paragraph_texts": {
                "broad-p1": "Vadné doručení soudního rozhodnutí vedlo ke zmeškání lhůty pro odvolání."
            },
            "chunk_index": 0,
            "source_order": 0,
        },
    )
    high_score_isolated_overlap = RetrievedChunk(
        "thin-1",
        "Rozsudek byl vyhlášen u civilního soudu.",
        0.11,
        "rrf",
        {
            "document_id": "DOC-THIN",
            "paragraph_ids": ["thin-p1"],
            "paragraph_texts": {"thin-p1": "Rozsudek byl vyhlášen u civilního soudu."},
            "chunk_index": 0,
            "source_order": 0,
        },
    )

    result = _aggregate_documents(
        [high_score_isolated_overlap, low_score_broad_evidence],
        dense=[high_score_isolated_overlap, low_score_broad_evidence],
        bm25=[high_score_isolated_overlap, low_score_broad_evidence],
        query_spec=spec,
        limit=2,
    )

    assert [document.document_id for document in result] == ["DOC-BROAD", "DOC-THIN"]


def test_pipeline_verifies_only_all_proven_hard_constraints() -> None:
    spec = build_query_spec_v2("únos dítěte matkou z Česka do Ruska")
    paragraph = _retrieved_paragraph_chunk()

    class Retriever:
        def retrieve(self, query_spec):  # noqa: ANN001
            del query_spec
            from app.rag.legal_v2.retriever import LegalV2RetrievalResult

            return LegalV2RetrievalResult(
                documents=[paragraph],
                dense_results=[],
                bm25_results=[],
                fused_results=[],
                diagnostics={"candidate_documents": 1},
            )

    class Verifier:
        provider_name = "fake_proving_verifier"

        def verify(self, *, query_spec, candidate_document, evidence_windows, timeout_seconds=None):  # noqa: ANN001
            del timeout_seconds
            evidence = {
                window.constraint_id: window.paragraph_ids[:1]
                for window in evidence_windows
            }
            return {
                "document_id": candidate_document.document_id,
                "decision": "verified_match",
                "classification": "strong_match",
                "confidence": 0.95,
                "jurisdiction_match": True,
                "constraint_results": [
                    {
                        "constraint_id": constraint.constraint_id,
                        "status": ConstraintVerificationStatus.PROVEN.value,
                        "required_value": constraint.value,
                        "detected_value": constraint.value,
                        "evidence_paragraph_ids": evidence[constraint.constraint_id],
                        "source_of_claim": "court_finding",
                        "reason": "test proof",
                        "confidence": 1.0,
                    }
                    for constraint in query_spec.hard_constraints
                ],
            }

    result = search_legal_v2(
        query=spec.original_query,
        retriever=Retriever(),  # type: ignore[arg-type]
        verifier=Verifier(),
        config=LegalV2RetrieverConfig(returned_verified_documents=3),
        query_provider=DeterministicQuerySpecProvider(spec.to_dict()),
    )

    assert result.status == "verified_match"
    assert [document.document_id for document in result.verified_documents] == ["DOC-1"]
    assert result.verified_documents[0].evidence[0]["paragraph_ids"]


def test_search_v2_endpoint_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NALUS_LEGAL_V2_SEARCH_ENABLED", raising=False)
    response = TestClient(app).post(
        "/api/rag/search-v2",
        json={"query": "únos dítěte matkou z Česka do Ruska"},
    )

    assert response.status_code == 404
    assert "Legal Retrieval v2 search is disabled" in response.json()["detail"]
    assert "NALUS_LEGAL_V2_SEARCH_ENABLED=1" in response.json()["detail"]


def test_build_index_selects_only_gate_safe_parser_quality_documents(tmp_path: Path) -> None:
    artifact = tmp_path / "parser_quality.json"
    artifact.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "document_id": "approved",
                        "review_status": "approved",
                        "identified_defects": [],
                        "source_completeness_status": "complete_from_available_source",
                        "duplicate_source_identifier_status": "none",
                    },
                    {
                        "document_id": "defective",
                        "review_status": "approved",
                        "identified_defects": ["bad_boundary"],
                        "source_completeness_status": "complete_from_available_source",
                        "duplicate_source_identifier_status": "none",
                    },
                    {
                        "document_id": "incomplete",
                        "review_status": "approved",
                        "identified_defects": [],
                        "source_completeness_status": "missing_complete_text",
                        "duplicate_source_identifier_status": "none",
                    },
                    {
                        "document_id": "conflicting_duplicate",
                        "review_status": "approved",
                        "identified_defects": [],
                        "source_completeness_status": "complete_from_available_source",
                        "duplicate_source_identifier_status": "conflicting_content",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    assert _document_ids_from_parser_quality(artifact, limit=None) == ["approved"]


def test_build_index_requires_gate_decision_that_permits_smoke_index(tmp_path: Path) -> None:
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps({"final_decision": "blocked", "smoke_index_permitted": False}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not permit indexing"):
        _require_gate_pass(gate)


def _retrieved_paragraph_chunk():
    from app.rag.legal_v2.evidence import CandidateEvidenceDocument
    from app.rag.legal_v2.models import LegalParagraph, MetadataProvenance, SectionType

    paragraph = LegalParagraph(
        document_id="DOC-1",
        paragraph_id="p1",
        paragraph_index=0,
        original_text="Matka neoprávněně přemístila dítě z Česka do Ruska.",
        normalized_text="Matka neoprávněně přemístila dítě z Česka do Ruska.",
        section_type=SectionType.FACTS,
        start_offset=0,
        end_offset=55,
        source_order=0,
        heading_context=["SKUTKOVÝ STAV"],
        is_boilerplate=False,
        is_citation_block=False,
        language="cs",
        metadata_provenance=MetadataProvenance(source="test", extraction_method="test"),
    )
    return CandidateEvidenceDocument(
        document_id="DOC-1",
        metadata={"source": "constitutional"},
        paragraphs=[paragraph],
        score=0.5,
        dense_rank=1,
        bm25_rank=1,
        rrf_score=0.03,
    )
