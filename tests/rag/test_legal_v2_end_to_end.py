from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api_app import app
from app.rag.legal_v2.adapters import LegalSourceDocument
from app.rag.legal_v2.audit import audit_documents
from app.rag.legal_v2.index_builder import LegalV2BuildConfig, build_legal_v2_index
from app.rag.legal_v2.interpreter import (
    DeterministicQuerySpecProvider,
    interpret_query_spec_v2,
)
from app.rag.legal_v2.pipeline import search_legal_v2
from app.rag.legal_v2.query_spec import build_query_spec_v2
from app.rag.legal_v2.retriever import LegalV2HybridRetriever, LegalV2RetrieverConfig
from app.rag.legal_v2.verifier import ConstraintVerificationStatus
from app.rag.retrieval.bm25_sidecar import Bm25Record, Bm25Sidecar
from app.rag.retrieval.models import RetrievedChunk


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
    stored = json.loads((tmp_path / "legal_v2_build_manifest.json").read_text(encoding="utf-8"))
    assert stored["bm25_index_id"] == "nalus_legal_paragraph_bm25_v2"


def test_query_interpreter_rejects_lost_mother_role() -> None:
    spec = build_query_spec_v2("únos dítěte matkou z Česka do Ruska").to_dict()
    spec["entities"] = [entity for entity in spec["entities"] if entity.get("role") != "mother"]
    result = interpret_query_spec_v2(
        "únos dítěte matkou z Česka do Ruska",
        provider=DeterministicQuerySpecProvider(spec),
    )

    assert result.status == "failed"
    assert result.reason == "explicit_mother_role_lost"


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
    assert "disabled" in response.json()["detail"].lower()


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
