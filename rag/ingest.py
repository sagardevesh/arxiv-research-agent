from dataclasses import dataclass
from datetime import datetime

import arxiv


@dataclass
class Paper:
    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    published: datetime
    url: str
    categories: list[str]
    full_text: str | None = None

    def ingestion_text(self) -> str:
        """Text used for chunking and embedding — title + abstract."""
        return f"{self.title}\n\n{self.abstract}"

    def to_payload(self) -> dict:
        """Qdrant point payload (everything except the raw text)."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "published": self.published.isoformat(),
            "url": self.url,
            "categories": self.categories,
        }


def _result_to_paper(result) -> "Paper":
    return Paper(
        paper_id=result.entry_id.split("/")[-1],
        title=result.title.strip(),
        abstract=result.summary.strip(),
        authors=[a.name for a in result.authors],
        published=result.published,
        url=result.entry_id,
        categories=result.categories,
    )


_client = arxiv.Client()


def fetch_paper_by_id(paper_id: str) -> "Paper":
    """Fetch a single paper by arXiv ID (version suffix stripped automatically)."""
    clean_id = paper_id.split("v")[0]
    search = arxiv.Search(id_list=[clean_id])
    for result in _client.results(search):
        return _result_to_paper(result)
    raise ValueError(f"Paper not found on arXiv: {paper_id!r}")


def fetch_papers(
    query: str,
    max_results: int = 50,
    categories: list[str] | None = None,
) -> list[Paper]:
    """Fetch papers from arXiv by query string, optionally filtering by category."""
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers = []
    for result in _client.results(search):
        if categories and not any(c in result.categories for c in categories):
            continue
        papers.append(_result_to_paper(result))
    return papers
