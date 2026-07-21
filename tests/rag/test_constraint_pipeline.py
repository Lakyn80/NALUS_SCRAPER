from app.rag.retrieval.constraint_config import ConstraintRetrievalConfig
from app.rag.retrieval.constraint_pipeline import retrieve_verified_documents
from app.rag.retrieval.full_document import (
    FullDocumentChunk,
    FullDocumentDiagnostics,
    FullDocumentResult,
)
from app.rag.retrieval.models import RetrievedChunk


class _Store:
    def __init__(self, documents: dict[str, FullDocumentResult]) -> None:
        self.documents = documents
        self.calls: list[str] = []

    def get(self, document_id: str) -> FullDocumentResult | None:
        self.calls.append(document_id)
        return self.documents.get(document_id)


def _chunk(document_id: str, *, score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"{document_id}-0",
        text="candidate",
        score=score,
        source="dense",
        metadata={"document_id": document_id, "court_name": "Ústavní soud"},
    )


def _document(document_id: str, text: str) -> FullDocumentResult:
    chunks = [
        FullDocumentChunk(
            chunk_id=f"{document_id}-chunk-0",
            chunk_index=0,
            text=text,
            metadata={"document_id": document_id, "chunk_index": 0},
        )
    ]
    return FullDocumentResult(
        document_id=document_id,
        metadata={"document_id": document_id, "court_name": "Ústavní soud"},
        full_text=text,
        chunks=chunks,
        source_url=None,
        provenance_status="overeno",
        full_text_availability_status="available",
        diagnostics=FullDocumentDiagnostics(
            collection_name="test",
            chunk_count=1,
            missing_chunk_indexes=[],
            duplicate_chunk_indexes=[],
            all_chunks_have_index=True,
            reconstruction_method="test",
            max_chunks=2000,
        ),
    )


def test_pipeline_returns_only_verified_documents() -> None:
    chunks = [_chunk("DOC-OK", score=0.9), _chunk("DOC-BAD", score=0.95)]
    store = _Store(
        {
            "DOC-OK": _document(
                "DOC-OK",
                "Stěžovatel je státní občan Ruské federace a podal žádost o udělení státního občanství České republiky.",
            ),
            "DOC-BAD": _document(
                "DOC-BAD",
                "Stěžovatel je občan Ukrajiny a žádal o udělení státního občanství České republiky.",
            ),
        }
    )

    result = retrieve_verified_documents(
        query="udělení českého občanství ruskému občanu",
        retriever=lambda query, top_k: chunks[:top_k],
        full_document_store=store,
        config=ConstraintRetrievalConfig(max_candidate_chunks=10),
    )

    assert [document.document_id for document in result.verified_documents] == ["DOC-OK"]
    assert result.diagnostics.verified_document_count == 1
    assert result.diagnostics.excluded_hard_mismatch_count == 1
    assert result.rejected_documents[0].document_id == "DOC-BAD"


def test_pipeline_returns_empty_verified_result_without_fallback() -> None:
    chunks = [_chunk("DOC-BAD", score=0.95)]
    store = _Store(
        {
            "DOC-BAD": _document(
                "DOC-BAD",
                "Rozhodnutí o místním referendu bez otázky státního občanství.",
            )
        }
    )

    result = retrieve_verified_documents(
        query="udělení českého občanství ruskému občanu",
        retriever=lambda query, top_k: chunks[:top_k],
        full_document_store=store,
        config=ConstraintRetrievalConfig(max_candidate_chunks=10),
    )

    assert result.verified_documents == []
    assert result.diagnostics.final_document_count == 0
    assert result.diagnostics.excluded_not_proven_count == 1


def test_pipeline_applies_candidate_filter_before_grouping() -> None:
    chunks = [
        RetrievedChunk(
            id="us-0",
            text="candidate",
            score=0.9,
            source="dense",
            metadata={"document_id": "US-DOC", "source": "nalus"},
        ),
        RetrievedChunk(
            id="ns-0",
            text="candidate",
            score=0.95,
            source="dense",
            metadata={"document_id": "NS-DOC", "source": "nsoud"},
        ),
    ]
    store = _Store(
        {
            "US-DOC": _document(
                "US-DOC",
                "Stěžovatel je státní občan Ruské federace a žádost o udělení státního občanství České republiky.",
            )
        }
    )

    result = retrieve_verified_documents(
        query="udělení českého občanství ruskému občanu",
        retriever=lambda query, top_k: chunks[:top_k],
        full_document_store=store,
        config=ConstraintRetrievalConfig(max_candidate_chunks=10),
        candidate_filter=lambda chunk: chunk.metadata.get("source") == "nalus",
    )

    assert [document.document_id for document in result.verified_documents] == ["US-DOC"]
    assert store.calls == ["US-DOC"]
