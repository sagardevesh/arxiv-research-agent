import pytest
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver


@tool
def _fake_search(query: str) -> str:
    """Search papers."""
    return "[]"


@pytest.fixture
def fake_tools():
    return [_fake_search]


def test_build_graph_returns_runnable(monkeypatch, fake_tools):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from agent.graph import build_graph

    graph = build_graph(fake_tools)
    assert callable(getattr(graph, "ainvoke", None))


def test_build_graph_uses_memory_checkpointer(monkeypatch, fake_tools):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from agent.graph import build_graph

    graph = build_graph(fake_tools)
    assert isinstance(graph.checkpointer, MemorySaver)


def test_system_prompt_content():
    from agent.graph import SYSTEM_PROMPT

    assert "search_papers" in SYSTEM_PROMPT
    assert "arXiv" in SYSTEM_PROMPT
    assert "arXiv:ID" in SYSTEM_PROMPT


def test_mcp_server_config_shape():
    """Runner's MCP config targets the right module with stdio transport."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "runner",
        "agent/runner.py",
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    cfg = runner._MCP_SERVER_CONFIG["arxiv"]
    assert cfg["transport"] == "stdio"
    assert cfg["args"] == ["-m", "mcp_server"]
    assert "PYTHONPATH" in cfg["env"]
