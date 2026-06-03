import json

from mcp.server.fastmcp import FastMCP

from mcp_server import tools

mcp = FastMCP(
    "arxiv-research-agent",
    instructions=(
        "You have access to an arXiv research paper store. "
        "Use search_papers to find relevant papers, fetch_paper to get full metadata, "
        "summarize_paper for an AI summary, and find_related to discover similar work."
    ),
)


@mcp.tool()
def search_papers(query: str, top_k: int = 10) -> str:
    """Search for arXiv papers using hybrid BM25 + dense retrieval with RRF fusion.

    Returns a JSON array of matching chunks with score, title, abstract, authors,
    url, and categories.
    """
    results = tools.search_papers_tool(query, top_k=top_k)
    return json.dumps(results, indent=2, default=str)


@mcp.tool()
def fetch_paper(paper_id: str) -> str:
    """Fetch metadata for an arXiv paper by its ID (e.g. '2301.07507').

    Returns a JSON object with title, abstract, authors, published date, url,
    and categories.
    """
    result = tools.fetch_paper_tool(paper_id)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def summarize_paper(paper_id: str) -> str:
    """Return a structured AI-generated summary of an arXiv paper.

    Covers objective, methods, key results, and significance. Uses the paper's
    title and abstract fetched live from arXiv.
    """
    return tools.summarize_paper_tool(paper_id)


@mcp.tool()
def find_related(paper_id: str, top_k: int = 5) -> str:
    """Find papers in the vector store that are semantically similar to *paper_id*.

    Embeds the queried paper's text and runs a hybrid search; the queried paper
    itself is excluded from results. Returns a JSON array ranked by relevance.
    """
    results = tools.find_related_tool(paper_id, top_k=top_k)
    return json.dumps(results, indent=2, default=str)


if __name__ == "__main__":
    mcp.run()
