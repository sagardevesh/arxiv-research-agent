from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

from rag.embedder import SparseVector as EmbedderSparseVector
from rag.store import COLLECTION_NAME, SPARSE_VECTOR_NAME


def hybrid_search(
    client: QdrantClient,
    dense_query: list[float],
    sparse_query: EmbedderSparseVector,
    top_k: int = 10,
) -> list[dict]:
    """Hybrid BM25 + dense retrieval with Reciprocal Rank Fusion.

    Each retrieval leg fetches 2×top_k candidates; Qdrant applies RRF
    server-side before returning the final top_k results.
    """
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(
                query=dense_query,
                using="dense",
                limit=top_k * 2,
            ),
            Prefetch(
                query=SparseVector(
                    indices=sparse_query.indices,
                    values=sparse_query.values,
                ),
                using=SPARSE_VECTOR_NAME,
                limit=top_k * 2,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return [{"score": p.score, **p.payload} for p in results.points]
