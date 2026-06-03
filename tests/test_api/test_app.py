import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from api.app import app


def _make_client(mock_graph):
    mock_mcp = AsyncMock()
    mock_mcp.get_tools.return_value = []
    with (
        patch("api.app.MultiServerMCPClient") as MockMCP,
        patch("api.app.build_graph", return_value=mock_graph),
    ):
        MockMCP.return_value.__aenter__ = AsyncMock(return_value=mock_mcp)
        MockMCP.return_value.__aexit__ = AsyncMock(return_value=None)
        with TestClient(app) as c:
            yield c


@pytest.fixture
def client():
    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {
        "messages": [AIMessage(content="Here are some papers on attention.")]
    }
    yield from _make_client(mock_graph)


@pytest.fixture
def error_client():
    mock_graph = AsyncMock()
    mock_graph.ainvoke.side_effect = RuntimeError("Qdrant unavailable")
    yield from _make_client(mock_graph)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_returns_response(client):
    resp = client.post("/chat", json={"message": "What is attention?", "thread_id": "t1"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["response"] == "Here are some papers on attention."
    assert data["thread_id"] == "t1"


def test_chat_default_thread_id(client):
    resp = client.post("/chat", json={"message": "hello"})
    assert resp.status_code == 200
    assert resp.json()["thread_id"] == "default"


def test_chat_passes_thread_id_to_graph():
    captured: dict = {}

    async def _capture(*args, **kwargs):
        captured.update(kwargs)
        return {"messages": [AIMessage(content="ok")]}

    mock_graph = AsyncMock()
    mock_graph.ainvoke.side_effect = _capture

    for c in _make_client(mock_graph):
        c.post("/chat", json={"message": "hello", "thread_id": "session-42"})

    assert captured["config"]["configurable"]["thread_id"] == "session-42"


def test_chat_500_on_graph_error(error_client):
    resp = error_client.post("/chat", json={"message": "hello"})
    assert resp.status_code == 500
    assert "Qdrant unavailable" in resp.json()["detail"]
