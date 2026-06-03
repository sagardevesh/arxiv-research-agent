from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from rag.ingest import Paper, fetch_papers


def _make_arxiv_result(
    entry_id="http://arxiv.org/abs/2301.00001v1",
    title="Test Paper",
    summary="Abstract text.",
    authors=["Alice", "Bob"],
    published=datetime(2023, 1, 1, tzinfo=timezone.utc),
    categories=("cs.LG",),
):
    result = MagicMock()
    result.entry_id = entry_id
    result.title = title
    result.summary = summary

    def _author(name):
        m = MagicMock()
        m.name = name
        return m

    result.authors = [_author(a) for a in authors]
    result.published = published
    result.categories = list(categories)
    return result


class TestPaper:
    def test_ingestion_text_combines_title_and_abstract(self):
        p = Paper(
            paper_id="2301.00001v1",
            title="My Title",
            abstract="My abstract.",
            authors=["A"],
            published=datetime(2023, 1, 1),
            url="http://arxiv.org/abs/2301.00001v1",
            categories=["cs.LG"],
        )
        text = p.ingestion_text()
        assert "My Title" in text
        assert "My abstract." in text

    def test_to_payload_excludes_abstract(self):
        p = Paper(
            paper_id="2301.00001v1",
            title="Title",
            abstract="Abstract.",
            authors=["A"],
            published=datetime(2023, 1, 1),
            url="http://x",
            categories=["cs.LG"],
        )
        payload = p.to_payload()
        assert "abstract" not in payload
        assert payload["paper_id"] == "2301.00001v1"
        assert payload["title"] == "Title"

    def test_to_payload_published_is_iso_string(self):
        dt = datetime(2023, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        p = Paper(
            paper_id="x",
            title="T",
            abstract="A",
            authors=[],
            published=dt,
            url="u",
            categories=[],
        )
        assert p.to_payload()["published"] == dt.isoformat()


class TestFetchPapers:
    def test_returns_paper_objects(self):
        mock_result = _make_arxiv_result()
        with patch("rag.ingest.arxiv.Search") as MockSearch:
            MockSearch.return_value.results.return_value = [mock_result]
            papers = fetch_papers("test query", max_results=1)

        assert len(papers) == 1
        p = papers[0]
        assert p.paper_id == "2301.00001v1"
        assert p.title == "Test Paper"
        assert p.authors == ["Alice", "Bob"]

    def test_category_filter_excludes_non_matching(self):
        cs_lg = _make_arxiv_result(categories=("cs.LG",))
        cs_cv = _make_arxiv_result(
            entry_id="http://arxiv.org/abs/2301.00002v1", categories=("cs.CV",)
        )
        with patch("rag.ingest.arxiv.Search") as MockSearch:
            MockSearch.return_value.results.return_value = [cs_lg, cs_cv]
            papers = fetch_papers("test", categories=["cs.LG"])

        assert len(papers) == 1
        assert papers[0].categories == ["cs.LG"]

    def test_no_category_filter_returns_all(self):
        results = [
            _make_arxiv_result(entry_id=f"http://arxiv.org/abs/230{i}.00001v1") for i in range(3)
        ]
        with patch("rag.ingest.arxiv.Search") as MockSearch:
            MockSearch.return_value.results.return_value = results
            papers = fetch_papers("test")

        assert len(papers) == 3
