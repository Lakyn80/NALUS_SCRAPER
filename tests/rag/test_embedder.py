from __future__ import annotations

from app.rag.retrieval.embedder import SentenceTransformersEmbedder


class _FakeSentenceTransformer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def encode(
        self,
        texts,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ):
        self.calls.append(
            {
                "texts": list(texts),
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
                "show_progress_bar": show_progress_bar,
            }
        )
        return [[0.1, 0.2, 0.3] for _ in texts]

    def get_sentence_embedding_dimension(self) -> int:
        return 3


def test_sentence_transformers_embedder_embeds_query() -> None:
    model = _FakeSentenceTransformer()
    embedder = SentenceTransformersEmbedder(model=model, batch_size=16)

    vector = embedder.embed_query("únos dítěte")

    assert vector == [0.1, 0.2, 0.3]
    assert model.calls[0]["texts"] == ["únos dítěte"]
    assert model.calls[0]["batch_size"] == 16


def test_sentence_transformers_embedder_embeds_documents_in_batch() -> None:
    model = _FakeSentenceTransformer()
    embedder = SentenceTransformersEmbedder(model=model, batch_size=8)

    vectors = embedder.embed_documents(["a", "b", "c"])

    assert vectors == [
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3],
    ]
    assert model.calls[0]["texts"] == ["a", "b", "c"]
    assert embedder.dim == 3
