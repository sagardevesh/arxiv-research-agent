from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from rag.embedder import SparseVector
from rag.ingest import Paper


def _make_paper(**kwargs) -> Paper:
    defaults = dict(
        paper_id="2301.00001v1",
        title="Attention Is All You Need",
        abstract="We propose the Transformer...",
        authors=["Vaswani et al."],
        published=datetime(2017, 6, 12, tzinfo=timezone.utc),
        url="http://arxiv.org/abs/2301.00001v1",
        categories=["cs.LG"],
    )
    return Paper(**{**defaults, **kwargs})


def _make_ctx(dense_vec=None, sparse_vec=None):
    """Build a mock AppContext with pre-configured embedder return values."""
    mock_ctx = MagicMock()
    mock_ctx.dense_embedder.embed.return_value = [dense_vec or [0.1] * 384]
    mock_ctx.sparse_embedder.embed.return_value = [
        sparse_vec or SparseVector(indices=[0, 1], values=[0.5, 0.3])
    ]
    return mock_ctx


# ---------------------------------------------------------------------------
# search_papers_tool
# ---------------------------------------------------------------------------


class TestSearchPapersTool:
    def test_returns_hybrid_search_results(self):
        mock_ctx = _make_ctx()
        expected = [{"title": "Paper A", "score": 0.9, "text": "chunk"}]

        with (
            patch("mcp_server.tools.ctx", mock_ctx),
            patch("mcp_server.tools.hybrid_search", return_value=expected) as mock_hs,
        ):
            from mcp_server.tools import search_papers_tool

            results = search_papers_tool("attention mechanisms", top_k=5)

        assert results == expected
        mock_hs.assert_called_once()
        assert mock_hs.call_args.kwargs.get("top_k") == 5

    def test_embeds_query_before_searching(self):
        mock_ctx = _make_ctx()

        with (
            patch("mcp_server.tools.ctx", mock_ctx),
            patch("mcp_server.tools.hybrid_search", return_value=[]),
        ):
            from mcp_server.tools import search_papers_tool

            search_papers_tool("transformers", top_k=3)

        mock_ctx.dense_embedder.embed.assert_called_once_with(["transformers"])
        mock_ctx.sparse_embedder.embed.assert_called_once_with(["transformers"])


# ---------------------------------------------------------------------------
# fetch_paper_tool
# ---------------------------------------------------------------------------


class TestFetchPaperTool:
    def test_returns_paper_dict_without_full_text(self):
        paper = _make_paper()

        with patch("mcp_server.tools.fetch_paper_by_id", return_value=paper):
            from mcp_server.tools import fetch_paper_tool

            result = fetch_paper_tool("2301.00001")

        assert result["paper_id"] == paper.paper_id
        assert result["title"] == paper.title
        assert result["abstract"] == paper.abstract
        assert "full_text" not in result

    def test_passes_paper_id_to_fetcher(self):
        paper = _make_paper()

        with patch("mcp_server.tools.fetch_paper_by_id", return_value=paper) as mock_fetch:
            from mcp_server.tools import fetch_paper_tool

            fetch_paper_tool("2301.00001v2")

        mock_fetch.assert_called_once_with("2301.00001v2")

    def test_published_date_is_iso_string(self):
        paper = _make_paper()

        with patch("mcp_server.tools.fetch_paper_by_id", return_value=paper):
            from mcp_server.tools import fetch_paper_tool

            result = fetch_paper_tool("2301.00001")

        assert isinstance(result["published"], str)
        assert "2017" in result["published"]


# ---------------------------------------------------------------------------
# summarize_paper_tool
# ---------------------------------------------------------------------------


class TestSummarizePaperTool:
    def test_returns_claude_response_text(self):
        paper = _make_paper()
        mock_ctx = _make_ctx()
        mock_ctx.anthropic.messages.create.return_value.content = [
            MagicMock(text="A concise summary of the paper.")
        ]

        with (
            patch("mcp_server.tools.fetch_paper_by_id", return_value=paper),
            patch("mcp_server.tools.ctx", mock_ctx),
        ):
            from mcp_server.tools import summarize_paper_tool

            summary = summarize_paper_tool("2301.00001")

        assert summary == "A concise summary of the paper."

    def test_passes_title_and_abstract_to_claude(self):
        paper = _make_paper(title="My Paper", abstract="My abstract.")
        mock_ctx = _make_ctx()
        mock_ctx.anthropic.messages.create.return_value.content = [MagicMock(text="ok")]

        with (
            patch("mcp_server.tools.fetch_paper_by_id", return_value=paper),
            patch("mcp_server.tools.ctx", mock_ctx),
        ):
            from mcp_server.tools import summarize_paper_tool

            summarize_paper_tool("2301.00001")

        call_kwargs = mock_ctx.anthropic.messages.create.call_args.kwargs
        user_content = call_kwargs["messages"][0]["content"]
        assert "My Paper" in user_content
        assert "My abstract." in user_content


# ---------------------------------------------------------------------------
# find_related_tool
# ---------------------------------------------------------------------------


class TestFindRelatedTool:
    def test_excludes_source_paper_from_results(self):
        paper = _make_paper(paper_id="2301.00001v1")
        mock_ctx = _make_ctx()
        search_results = [
            {"paper_id": "2301.00001v1", "title": "Self", "score": 1.0},
            {"paper_id": "2301.00002v1", "title": "Related A", "score": 0.8},
            {"paper_id": "2301.00003v1", "title": "Related B", "score": 0.7},
        ]

        with (
            patch("mcp_server.tools.fetch_paper_by_id", return_value=paper),
            patch("mcp_server.tools.ctx", mock_ctx),
            patch("mcp_server.tools.hybrid_search", return_value=search_results),
        ):
            from mcp_server.tools import find_related_tool

            results = find_related_tool("2301.00001", top_k=5)

        ids = [r["paper_id"] for r in results]
        assert "2301.00001v1" not in ids
        assert "2301.00002v1" in ids

    def test_respects_top_k_limit(self):
        paper = _make_paper(paper_id="X")
        mock_ctx = _make_ctx()
        search_results = [{"paper_id": f"p{i}", "score": float(i)} for i in range(10)]

        with (
            patch("mcp_server.tools.fetch_paper_by_id", return_value=paper),
            patch("mcp_server.tools.ctx", mock_ctx),
            patch("mcp_server.tools.hybrid_search", return_value=search_results),
        ):
            from mcp_server.tools import find_related_tool

            results = find_related_tool("X", top_k=3)

        assert len(results) <= 3

    def test_uses_paper_ingestion_text_for_embedding(self):
        paper = _make_paper(title="T", abstract="A")
        mock_ctx = _make_ctx()

        with (
            patch("mcp_server.tools.fetch_paper_by_id", return_value=paper),
            patch("mcp_server.tools.ctx", mock_ctx),
            patch("mcp_server.tools.hybrid_search", return_value=[]),
        ):
            from mcp_server.tools import find_related_tool

            find_related_tool("2301.00001")

        expected_text = paper.ingestion_text()
        mock_ctx.dense_embedder.embed.assert_called_once_with([expected_text])
