from unittest.mock import MagicMock, patch

import pytest

from rag.embedder import SparseVector
from rag.retriever import hybrid_search
from rag.store import COLLECTION_NAME, SPARSE_VECTOR_NAME


def _make_scored_point(score: float, payload: dict):
    p = MagicMock()
    p.score = score
    p.payload = payload
    return p


class TestHybridSearch:
    def test_returns_list_of_dicts_with_score(self):
        client = MagicMock()
        client.query_points.return_value.points = [
            _make_scored_point(0.9, {"title": "Paper A", "text": "chunk"}),
            _make_scored_point(0.7, {"title": "Paper B", "text": "chunk2"}),
        ]

        dense = [0.1] * 384
        sparse = SparseVector(indices=[0, 1], values=[0.5, 0.3])

        results = hybrid_search(client, dense, sparse, top_k=2)

        assert len(results) == 2
        assert results[0]["score"] == 0.9
        assert results[0]["title"] == "Paper A"

    def test_calls_query_points_with_correct_collection(self):
        client = MagicMock()
        client.query_points.return_value.points = []

        dense = [0.0] * 384
        sparse = SparseVector(indices=[], values=[])
        hybrid_search(client, dense, sparse, top_k=5)

        call_kwargs = client.query_points.call_args.kwargs
        assert call_kwargs["collection_name"] == COLLECTION_NAME
        assert call_kwargs["limit"] == 5

    def test_prefetch_uses_both_dense_and_sparse_legs(self):
        client = MagicMock()
        client.query_points.return_value.points = []

        dense = [0.1] * 384
        sparse = SparseVector(indices=[10], values=[1.0])
        hybrid_search(client, dense, sparse, top_k=10)

        prefetch_arg = client.query_points.call_args.kwargs["prefetch"]
        assert len(prefetch_arg) == 2
        usings = {p.using for p in prefetch_arg}
        assert "dense" in usings
        assert SPARSE_VECTOR_NAME in usings

    def test_empty_results_returns_empty_list(self):
        client = MagicMock()
        client.query_points.return_value.points = []

        results = hybrid_search(
            client,
            [0.0] * 384,
            SparseVector(indices=[], values=[]),
        )
        assert results == []
