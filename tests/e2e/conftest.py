"""
E2E test fixtures.

Seeds a temporary Qdrant collection (nalus_e2e_test) with realistic Czech
legal texts using the real QdrantIngestor + a thin PointStruct adapter.

Why the adapter?
  QdrantIngestor.upsert() receives IngestPoint dataclasses (internal format).
  The real Qdrant REST client expects PointStruct (pydantic model).
  The adapter converts IngestPoint → PointStruct without touching any module.

ID strategy:
  Qdrant accepts only int or UUID-string IDs.
  We use integers 1-N.  The adapter stores the original string ID in the
  payload so it is available to _to_chunk indirectly (though _to_chunk
  uses point.id, which becomes str(int) = "1", "2", …).
  KeywordRetriever corpus uses the same string IDs "1", "2", …
  so fusion (by ID) works correctly across both retrievers.

Dense retrieval is non-semantic (MockEmbedder → uniform vectors).
Retrieval quality is driven by KeywordRetriever — intentional until a real
embedder is wired in.

Skips automatically if Qdrant is not reachable.
"""

from typing import Any

import pytest

from app.rag.answer.answer_service import AnswerService
from app.rag.answer.hybrid_service import HybridAnswerService
from app.rag.chunking.chunker import TextChunk
from app.rag.ingest.qdrant_ingest import QdrantIngestor
from app.rag.llm.mock_llm import MockLLM
from app.rag.llm.service import LLMService
from app.rag.orchestration.pipeline import RetrievalPipeline
from app.rag.retrieval.dense_retriever import DenseRetriever
from app.rag.retrieval.embedder import MockEmbedder
from app.rag.retrieval.keyword_retriever import KeywordRetriever
from app.rag.retrieval.retrieval_service import RetrievalService

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COLLECTION = "nalus_e2e_test"
_VECTOR_DIM = 10

# ---------------------------------------------------------------------------
# Realistic Czech legal corpus
# (int_id, string_id, text)
#   int_id   — used as Qdrant point ID (must be int or UUID-string)
#   string_id — used in keyword corpus and for assertions ("1", "2", …)
#   text      — the actual legal text excerpt
# ---------------------------------------------------------------------------

_CORPUS: list[tuple[int, str, str]] = [
    (
        1, "1",
        "III.ÚS 255/22 — Ústavní soud se zabýval případem mezinárodního únosu dítěte matkou "
        "do Ruska. Matka jako cizinka neoprávněně přemístila nezletilé dítě na území Ruské "
        "federace bez souhlasu otce. Soud aplikoval Haagskou úmluvu o mezinárodním únosu "
        "dítěte a nařídil neprodlený návrat dítěte do země jeho obvyklého bydliště. "
        "Rodičovská odpovědnost náleží oběma rodičům společně.",
    ),
    (
        2, "2",
        "II.ÚS 100/21 — Případ řeší rodičovskou odpovědnost a střídavou péči o nezletilé "
        "dítě. Ústavní soud zdůraznil, že zájem dítěte je prvořadý při rozhodování o péči. "
        "Rodičovská odpovědnost obou rodičů musí být zachována i po rozvodu manželství. "
        "Střídavá péče je preferována, pokud to nejlepší zájem dítěte nevylučuje.",
    ),
    (
        3, "3",
        "I.ÚS 88/23 — Stěžovatel se domáhá náhrady škody způsobené státem v důsledku "
        "nezákonného rozhodnutí orgánu veřejné moci. Odpovědnost státu za škodu je "
        "zakotvena v zákoně č. 82/1998 Sb. Náhrada škody zahrnuje skutečnou škodu i ušlý "
        "zisk. Stát odpovídá za škodu způsobenou nezákonnými rozhodnutími.",
    ),
    (
        4, "4",
        "IV.ÚS 312/20 — Ústavní soud shledal porušení práva na spravedlivý proces z důvodu "
        "nepřiměřené délky řízení. Délka soudního řízení přesáhla rozumnou dobu a porušila "
        "základní právo stěžovatele. Průtahy v řízení před obecnými soudy jsou nepřípustné. "
        "Stát je povinen zajistit projednání věci v přiměřené lhůtě.",
    ),
    (
        5, "5",
        "III.ÚS 401/22 — Haagská úmluva o občanskoprávních aspektech mezinárodních únosů "
        "dítěte zavazuje smluvní státy k rychlému návratu neoprávněně přemístěných dětí. "
        "Únos dítěte jedním z rodičů porušuje právo druhého rodiče na styk s dítětem. "
        "Opatrovnický soud musí rozhodnout o návratu dítěte bez zbytečných průtahů.",
    ),
    (
        6, "6",
        "II.ÚS 77/19 — Soud posuzoval přiměřenost délky trestního řízení a právo na "
        "projednání věci bez průtahů. Nepřiměřená délka řízení představuje porušení "
        "článku 38 odst. 2 Listiny základních práv a svobod. Obviněnému přísluší náhrada "
        "nemajetkové újmy způsobené nepřiměřeně dlouhým řízením.",
    ),
    (
        7, "7",
        "I.ÚS 502/21 — Stát nese odpovědnost za škodu způsobenou nesprávným úředním "
        "postupem státních orgánů. Náhrada škody od státu se přiznává i za nemajetkovou "
        "újmu v případech zvláštního zřetele hodných. Odpovědnost státu za škodu je "
        "objektivní a nevyžaduje zavinění konkrétního úředníka.",
    ),
    (
        8, "8",
        "IV.ÚS 190/23 — Ústavní soud se zabýval otázkou rodičovské odpovědnosti "
        "v přeshraničním kontextu. Mezinárodní únos dítěte cizinkou porušuje práva otce "
        "jako druhého z rodičů. Soud aplikoval nařízení Brusel IIa při určování příslušnosti "
        "k rozhodnutí o návratu dítěte. Rodičovská odpovědnost nesmí být vykonávána "
        "v rozporu se základními právy druhého rodiče.",
    ),
]

# Mapping original TextChunk string ID → Qdrant int ID
_ID_MAP: dict[str, int] = {str_id: int_id for int_id, str_id, _ in _CORPUS}


# ---------------------------------------------------------------------------
# Qdrant adapter
# ---------------------------------------------------------------------------


class _PointStructAdapter:
    """
    Wraps the real Qdrant client so QdrantIngestor can use it unchanged.

    QdrantIngestor.upsert() receives IngestPoint dataclasses.
    Real Qdrant REST API requires PointStruct (pydantic).
    This adapter converts between the two — no existing module is modified.
    """

    def __init__(self, client: Any, id_map: dict[str, int]) -> None:
        self._client = client
        self._id_map = id_map

    def upsert(self, collection_name: str, points: list[Any]) -> None:
        from qdrant_client.models import PointStruct

        converted = [
            PointStruct(
                id=self._id_map[p.id],
                vector=p.vector,
                payload={**p.payload, "original_id": p.id},
            )
            for p in points
        ]
        self._client.upsert(collection_name=collection_name, points=converted)


class _SearchAdapter:
    """
    Adapts the new Qdrant client API (.query_points) to the .search() signature
    expected by DenseRetriever._SearchableClient Protocol.

    qdrant_client >= 1.9 removed the legacy .search() method in favour of
    .query_points().  This adapter bridges the gap without modifying
    DenseRetriever.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def search(
        self, collection_name: str, query_vector: list[float], limit: int
    ) -> list[Any]:
        response = self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        return response.points


# ---------------------------------------------------------------------------
# Qdrant availability check
# ---------------------------------------------------------------------------


def _make_client():
    from qdrant_client import QdrantClient
    return QdrantClient(host="localhost", port=6333, timeout=3)


def _qdrant_available() -> bool:
    try:
        _make_client().get_collections()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def qdrant_client():
    if not _qdrant_available():
        pytest.skip("Qdrant not reachable at localhost:6333")
    return _make_client()


@pytest.fixture(scope="session")
def seeded_collection(qdrant_client):
    """Create nalus_e2e_test, seed with real QdrantIngestor, delete after session."""
    from qdrant_client.models import Distance, VectorParams

    qdrant_client.recreate_collection(
        collection_name=_COLLECTION,
        vectors_config=VectorParams(size=_VECTOR_DIM, distance=Distance.COSINE),
    )

    # Build TextChunks using string IDs (what QdrantIngestor expects)
    chunks = [
        TextChunk(
            id=str_id,
            text=text,
            case_reference=text.split("—")[0].strip(),
            ecli=None,
            decision_date="2023-01-01",
            judge="Test",
            text_url=None,
            chunk_index=0,
        )
        for _, str_id, text in _CORPUS
    ]

    # Use adapter so QdrantIngestor works with the real client
    adapter = _PointStructAdapter(qdrant_client, id_map=_ID_MAP)
    ingestor = QdrantIngestor(client=adapter, collection_name=_COLLECTION)
    ingestor.ingest_chunks(chunks)

    yield _COLLECTION

    qdrant_client.delete_collection(_COLLECTION)


@pytest.fixture(scope="session")
def keyword_corpus() -> list[tuple[str, str]]:
    """Same string IDs as what DenseRetriever returns (str(int_id))."""
    return [(str(int_id), text) for int_id, _, text in _CORPUS]


@pytest.fixture(scope="session")
def pipeline(qdrant_client, seeded_collection, keyword_corpus) -> RetrievalPipeline:
    dense = DenseRetriever(
        client=_SearchAdapter(qdrant_client),
        collection_name=seeded_collection,
        embedder=MockEmbedder(dim=_VECTOR_DIM),
    )
    keyword = KeywordRetriever(corpus=keyword_corpus)
    service = RetrievalService(dense=dense, keyword=keyword)
    return RetrievalPipeline(service)


@pytest.fixture(scope="session")
def hybrid_service() -> HybridAnswerService:
    return HybridAnswerService(
        answer_service=AnswerService(),
        llm_service=LLMService(MockLLM()),
    )
