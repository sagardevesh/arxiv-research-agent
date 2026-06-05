#!/usr/bin/env python3
"""Seed Qdrant from the committed fixture file instead of live arXiv API.

Used in CI to avoid arXiv rate-limiting (GitHub Actions IPs are shared and
frequently hit HTTP 429). The fixtures cover the same topics as the RAGAS
eval question set so retrieval quality is representative.

Usage:
    python scripts/seed_from_fixtures.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.chunker import chunk_text
from rag.embedder import DenseEmbedder, SparseEmbedder
from rag.ingest import Paper
from rag.store import ensure_collection, get_client, upsert_chunks

FIXTURES_PATH = Path(__file__).parent.parent / "eval" / "fixtures.json"


def main() -> None:
    papers = [
        Paper(
            paper_id=p["paper_id"],
            title=p["title"],
            abstract=p["abstract"],
            authors=p["authors"],
            published=datetime.fromisoformat(p["published"]),
            url=p["url"],
            categories=p["categories"],
        )
        for p in json.loads(FIXTURES_PATH.read_text())
    ]
    print(f"Loaded {len(papers)} papers from fixtures.")

    print("Loading embedders...")
    dense_embedder = DenseEmbedder()
    sparse_embedder = SparseEmbedder()

    print("Connecting to Qdrant...")
    client = get_client()
    ensure_collection(client, dense_dim=dense_embedder.dim)

    for i, paper in enumerate(papers, 1):
        chunks = chunk_text(paper.ingestion_text(), chunk_size=512, chunk_overlap=64)
        if not chunks:
            continue
        texts = [c.text for c in chunks]
        dense_vecs = dense_embedder.embed(texts)
        sparse_vecs = sparse_embedder.embed(texts)
        payload = paper.to_payload()
        upsert_chunks(client, texts, dense_vecs, sparse_vecs, [payload] * len(texts))
        print(f"  [{i}/{len(papers)}] {paper.title[:70]!r}")

    print(f"\nDone. {len(papers)} papers seeded into Qdrant.")


if __name__ == "__main__":
    main()
