from rag.ingest import Paper, fetch_paper_by_id
from rag.retriever import hybrid_search
from mcp_server.context import ctx


def _paper_to_dict(paper: Paper) -> dict:
    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "abstract": paper.abstract,
        "authors": paper.authors,
        "published": paper.published.isoformat(),
        "url": paper.url,
        "categories": paper.categories,
    }


def search_papers_tool(query: str, top_k: int = 10) -> list[dict]:
    """Embed *query* and run hybrid BM25 + dense search against Qdrant."""
    dense = ctx.dense_embedder.embed([query])[0]
    sparse = ctx.sparse_embedder.embed([query])[0]
    return hybrid_search(ctx.qdrant, dense, sparse, top_k=top_k)


def fetch_paper_tool(paper_id: str) -> dict:
    """Fetch arXiv metadata for a single paper by ID."""
    return _paper_to_dict(fetch_paper_by_id(paper_id))


def summarize_paper_tool(paper_id: str) -> str:
    """Fetch a paper and return a structured Claude-generated summary."""
    paper = fetch_paper_by_id(paper_id)
    response = ctx.anthropic.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=(
            "You are a research assistant. Given a paper title and abstract, "
            "provide a concise structured summary covering: objective, methods, "
            "key results, and significance. Be specific and factual."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Title: {paper.title}\n\nAbstract:\n{paper.abstract}",
            }
        ],
    )
    return response.content[0].text


def find_related_tool(paper_id: str, top_k: int = 5) -> list[dict]:
    """Find papers in the store that are semantically similar to *paper_id*."""
    paper = fetch_paper_by_id(paper_id)
    text = paper.ingestion_text()
    dense = ctx.dense_embedder.embed([text])[0]
    sparse = ctx.sparse_embedder.embed([text])[0]
    results = hybrid_search(ctx.qdrant, dense, sparse, top_k=top_k + 1)
    # Exclude the queried paper itself if it was ingested.
    return [r for r in results if r.get("paper_id") != paper.paper_id][:top_k]
