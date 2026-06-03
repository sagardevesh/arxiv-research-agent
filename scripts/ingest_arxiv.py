#!/usr/bin/env python3
"""CLI script to ingest arXiv papers into Qdrant.

Usage:
    python scripts/ingest_arxiv.py --query "retrieval augmented generation" --max 50
    python scripts/ingest_arxiv.py --query "large language models" --categories cs.CL cs.AI
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.chunker import chunk_text
from rag.embedder import DenseEmbedder, SparseEmbedder
from rag.ingest import fetch_papers
from rag.store import ensure_collection, get_client, upsert_chunks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest arXiv papers into Qdrant.")
    p.add_argument("--query", required=True, help="arXiv search query")
    p.add_argument("--max", type=int, default=50, dest="max_results", help="Max papers to fetch")
    p.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Filter to these arXiv categories (e.g. cs.LG cs.CL)",
    )
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--chunk-overlap", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Fetching up to {args.max_results} papers for: {args.query!r}")
    papers = fetch_papers(args.query, max_results=args.max_results, categories=args.categories)
    print(f"  → {len(papers)} papers retrieved")

    print("Loading embedders...")
    dense_embedder = DenseEmbedder()
    sparse_embedder = SparseEmbedder()

    print("Connecting to Qdrant and ensuring collection exists...")
    client = get_client()
    ensure_collection(client, dense_dim=dense_embedder.dim)

    total_chunks = 0
    for i, paper in enumerate(papers, 1):
        text = paper.ingestion_text()
        chunks = chunk_text(text, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        if not chunks:
            continue

        chunk_texts = [c.text for c in chunks]
        payload = paper.to_payload()
        payloads = [payload] * len(chunks)

        # Embed in batches to avoid OOM on large papers.
        for batch_start in range(0, len(chunk_texts), args.batch_size):
            batch = chunk_texts[batch_start : batch_start + args.batch_size]
            dense_vecs = dense_embedder.embed(batch)
            sparse_vecs = sparse_embedder.embed(batch)
            upsert_chunks(
                client,
                batch,
                dense_vecs,
                sparse_vecs,
                payloads[batch_start : batch_start + args.batch_size],
            )

        total_chunks += len(chunks)
        print(f"  [{i}/{len(papers)}] {paper.title[:70]!r} — {len(chunks)} chunks")

    print(f"\nDone. {len(papers)} papers, {total_chunks} chunks stored in Qdrant.")


if __name__ == "__main__":
    main()
