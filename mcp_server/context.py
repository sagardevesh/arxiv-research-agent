from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


class AppContext:
    """Lazy-loaded singleton resources shared across all MCP tool handlers.

    Heavy dependencies (sentence-transformers, fastembed, anthropic) are only
    imported and initialised on first access, so importing this module is cheap.
    """

    def __init__(self) -> None:
        self._qdrant: Any = None
        self._dense_embedder: Any = None
        self._sparse_embedder: Any = None
        self._anthropic: Any = None

    @property
    def qdrant(self):
        if self._qdrant is None:
            from rag.store import get_client
            self._qdrant = get_client()
        return self._qdrant

    @property
    def dense_embedder(self):
        if self._dense_embedder is None:
            from rag.embedder import DenseEmbedder
            self._dense_embedder = DenseEmbedder()
        return self._dense_embedder

    @property
    def sparse_embedder(self):
        if self._sparse_embedder is None:
            from rag.embedder import SparseEmbedder
            self._sparse_embedder = SparseEmbedder()
        return self._sparse_embedder

    @property
    def anthropic(self):
        if self._anthropic is None:
            import anthropic
            self._anthropic = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"]
            )
        return self._anthropic


ctx = AppContext()
