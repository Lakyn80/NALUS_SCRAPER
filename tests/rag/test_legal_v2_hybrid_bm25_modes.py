from __future__ import annotations

from types import SimpleNamespace

from app.rag.legal_v2.query_spec import build_query_spec_v2
from app.rag.legal_v2.retrieve.retriever import (
    LegalV2HybridRetriever,
    LegalV2RetrieverConfig,
    legal_v2_retriever_config_from_env,
)
from app.rag.legal_v2.retrieve.source_filters import (
    chunk_matches_source_filters,
    parse_retrieval_source_filters,
)
from app.rag.retrieval.bm25_sidecar import Bm25Record, Bm25Sidecar
from app.rag.retrieval.models import RetrievedChunk


def test_bm25_enabled_env_defaults_true(monkeypatch) -> None:
    monkeypatch.delenv("NALUS_LEGAL_V2_BM25_ENABLED", raising=False)
    assert legal_v2_retriever_config_from_env().bm25_enabled is True


def test_bm25_enabled_env_can_disable(monkeypatch) -> None:
    monkeypatch.setenv("NALUS_LEGAL_V2_BM25_ENABLED", "0")
    assert legal_v2_retriever_config_from_env().bm25_enabled is False


def test_dense_only_skips_bm25_and_preserves_dense_order() -> None:
    dense_chunks = [
        RetrievedChunk(
            "c1",
            "škoda zaměstnance",
            0.9,
            "dense",
            {"document_id": "DOC-1", "chunk_index": 0},
        ),
        RetrievedChunk(
            "c2",
            "smluvní pokuta",
            0.8,
            "dense",
            {"document_id": "DOC-2", "chunk_index": 0},
        ),
    ]

    class Dense:
        def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
            del query
            return list(dense_chunks[:top_k])

    class ForbiddenBm25:
        def search(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("BM25 must not run in dense-only mode")

    retriever = LegalV2HybridRetriever(
        dense_store=Dense(),
        bm25_sidecar=ForbiddenBm25(),  # type: ignore[arg-type]
        config=LegalV2RetrieverConfig(bm25_enabled=False, dense_candidate_chunks=5),
    )
    result = retriever.retrieve(build_query_spec_v2("škoda zaměstnance"))
    assert result.bm25_results == []
    assert result.diagnostics["dense_only"] is True
    assert result.diagnostics["bm25_enabled"] is False
    assert [doc.document_id for doc in result.documents] == ["DOC-1", "DOC-2"]


def test_hybrid_rrf_fuses_matching_chunk_ids() -> None:
    payload = {
        "document_id": "DOC-1",
        "ecli": "ECLI:CZ:US:2005:3.US.479.04",
        "court": "Ústavní soud",
        "chunk_index": 0,
    }
    dense = [
        RetrievedChunk("chunk-1", "škoda a zavinění zaměstnance", 0.4, "dense", dict(payload))
    ]
    bm25 = Bm25Sidecar.from_records(
        [
            Bm25Record("chunk-1", "škoda a zavinění zaměstnance", dict(payload)),
            Bm25Record(
                "chunk-2",
                "nájem bytu",
                {"document_id": "DOC-2", "court": "Ústavní soud", "chunk_index": 0},
            ),
        ],
        k1=1.5,
        b=0.75,
        index_id="test-bm25",
    )

    class Dense:
        def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
            del query, top_k
            return list(dense)

    retriever = LegalV2HybridRetriever(
        dense_store=Dense(),
        bm25_sidecar=bm25,
        config=LegalV2RetrieverConfig(dense_candidate_chunks=5, bm25_candidate_chunks=5),
    )
    result = retriever.retrieve(build_query_spec_v2("škoda zavinění zaměstnance"))
    assert result.diagnostics["bm25_enabled"] is True
    assert result.fused_results[0].id == "chunk-1"
    assert result.fused_results[0].metadata.get("rrf_score")


def test_source_filters_apply_to_dense_and_bm25() -> None:
    us_meta = {"document_id": "US-1", "court": "Ústavní soud", "source": "constitutional"}
    ns_meta = {"document_id": "NS-1", "court": "Nejvyšší soud", "source": "supreme"}
    dense_chunks = [
        RetrievedChunk("ns-1", "škoda zaměstnance u ns", 0.99, "dense", dict(ns_meta)),
        RetrievedChunk("us-1", "škoda zaměstnance u us", 0.50, "dense", dict(us_meta)),
    ]
    bm25 = Bm25Sidecar.from_records(
        [
            Bm25Record("ns-2", "škoda zaměstnance dovolání", dict(ns_meta)),
            Bm25Record("us-2", "škoda zaměstnance ústavní", dict(us_meta)),
        ],
        k1=1.5,
        b=0.75,
        index_id="test-bm25",
    )

    class Dense:
        def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
            del query
            return list(dense_chunks[:top_k])

    retriever = LegalV2HybridRetriever(
        dense_store=Dense(),
        bm25_sidecar=bm25,
        config=LegalV2RetrieverConfig(dense_candidate_chunks=5, bm25_candidate_chunks=5),
    )
    result = retriever.retrieve(
        build_query_spec_v2("škoda zaměstnance"),
        source_filters=parse_retrieval_source_filters(courts=["Ústavní soud"]),
    )
    assert result.diagnostics["source_filters"]["courts"] == ["Ústavní soud"]
    assert all(
        chunk_matches_source_filters(
            chunk.metadata, parse_retrieval_source_filters(courts=["Ústavní soud"])
        )
        for chunk in result.dense_results + result.bm25_results
    )
    assert [doc.document_id for doc in result.documents] == ["US-1"]


def test_court_filter_does_not_treat_nss_as_ns() -> None:
    nss = {"court": "Nejvyšší správní soud", "ecli": "ECLI:CZ:NSS:2020:1.As.1.20"}
    ns = {"court": "Nejvyšší soud", "ecli": "ECLI:CZ:NS:2020:21.Cdo.1.20"}
    us_filter = parse_retrieval_source_filters(courts=["Ústavní soud"])
    ns_filter = parse_retrieval_source_filters(courts=["Nejvyšší soud"])
    nss_filter = parse_retrieval_source_filters(courts=["Nejvyšší správní soud"])
    assert chunk_matches_source_filters(nss, ns_filter) is False
    assert chunk_matches_source_filters(ns, nss_filter) is False
    assert chunk_matches_source_filters(nss, nss_filter) is True
    assert chunk_matches_source_filters(ns, ns_filter) is True
    assert chunk_matches_source_filters(nss, us_filter) is False


def test_sqlite_bm25_hydrates_top_hits_without_keeping_all_text(tmp_path) -> None:
    import sqlite3

    path = tmp_path / "bm25.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE bm25_chunks (
                chunk_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata TEXT NOT NULL,
                document_id TEXT,
                section_type TEXT,
                paragraph_ids TEXT,
                qdrant_collection TEXT,
                retrieval_profile TEXT,
                bm25_index_id TEXT
            )
            """
        )
        connection.execute(
            "INSERT INTO bm25_chunks VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "c-keep",
                "zaměstnanec odpovídá za škodu jen při zavinění",
                "{}",
                "DOC-KEEP",
                "reasoning",
                "[]",
                "col",
                "a",
                "idx",
            ),
        )
        connection.execute(
            "INSERT INTO bm25_chunks VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "c-other",
                "nájem bytu a kauce",
                "{}",
                "DOC-OTHER",
                "facts",
                "[]",
                "col",
                "a",
                "idx",
            ),
        )
        connection.commit()

    sidecar = Bm25Sidecar(path, k1=1.5, b=0.75, index_id="idx")
    hits = sidecar.search("zavinění škoda zaměstnanec", top_k=1)
    assert hits[0].id == "c-keep"
    assert "zavinění" in hits[0].text
    assert sidecar._index is not None
    assert sidecar._index._texts is None


def test_dense_store_omits_query_filter_by_default(tmp_path, monkeypatch) -> None:
    from unittest.mock import MagicMock

    from app.rag.retrieval.production_profile import BGE_M3_DENSE_BM25_RRF, ProductionRetrievalConfig
    from app.rag.retrieval.qdrant_dense_store import QdrantDenseStore

    monkeypatch.delenv("NALUS_QDRANT_QUANTIZATION_ENABLED", raising=False)
    embedder = SimpleNamespace(embed_query=lambda query: [0.1] * 1024)
    client = MagicMock()
    client.query_points.return_value = SimpleNamespace(points=[])
    config = ProductionRetrievalConfig(
        profile=BGE_M3_DENSE_BM25_RRF,
        qdrant_collection="test",
        bm25_sidecar_path=tmp_path / "bm25.sqlite",
        bm25_index_id="idx",
        model_path="/app/models/BAAI/bge-m3",
        local_files_only=True,
        trust_remote_code=False,
        device="cpu",
        candidate_multiplier=1,
        min_candidate_count=1,
        max_candidate_count=80,
        lexical_filter_enabled=False,
    )
    QdrantDenseStore(client=client, embedder=embedder, config=config).search("q", top_k=8)
    assert "query_filter" not in client.query_points.call_args.kwargs
