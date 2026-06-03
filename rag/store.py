import os
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

load_dotenv()

COLLECTION_NAME = "arxiv_papers"
SPARSE_VECTOR_NAME = "bm25"


def get_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY") or None
    return QdrantClient(url=url, api_key=api_key)


def ensure_collection(client: QdrantClient, dense_dim: int = 384) -> None:
    """Create the collection if it doesn't already exist."""
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME in existing:
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense": VectorParams(size=dense_dim, distance=Distance.COSINE)},
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(index=SparseIndexParams(on_disk=False))
        },
    )


def upsert_chunks(
    client: QdrantClient,
    chunks: list[str],
    dense_vectors: list[list[float]],
    sparse_vectors: list[Any],  # list[SparseVector] from embedder
    payloads: list[dict[str, Any]],
) -> None:
    """Batch-upsert chunks with their dense + sparse vectors and metadata."""
    points = [
        PointStruct(
            id=str(uuid4()),
            vector={
                "dense": dense,
                SPARSE_VECTOR_NAME: SparseVector(
                    indices=sparse.indices,
                    values=sparse.values,
                ),
            },
            payload={**payload, "text": chunk},
        )
        for chunk, dense, sparse, payload in zip(chunks, dense_vectors, sparse_vectors, payloads)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
