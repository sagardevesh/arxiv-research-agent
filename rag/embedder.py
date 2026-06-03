from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastembed import SparseTextEmbedding as _FE
    from sentence_transformers import SentenceTransformer as _ST

DENSE_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
SPARSE_MODEL = "Qdrant/bm25"


@dataclass
class SparseVector:
    indices: list[int]
    values: list[float]


class DenseEmbedder:
    """Wraps sentence-transformers for batch dense embedding."""

    def __init__(self, model_name: str = DENSE_MODEL) -> None:
        from sentence_transformers import SentenceTransformer  # lazy: ~2 GB with torch

        self._model: _ST = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()


class SparseEmbedder:
    """BM25 sparse embedder backed by fastembed."""

    def __init__(self, model_name: str = SPARSE_MODEL) -> None:
        from fastembed import SparseTextEmbedding  # lazy: downloads model on first use

        self._model: _FE = SparseTextEmbedding(model_name)

    def embed(self, texts: list[str]) -> list[SparseVector]:
        return [
            SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())
            for emb in self._model.embed(texts)
        ]
