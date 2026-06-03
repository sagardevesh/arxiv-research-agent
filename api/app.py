import os
import pathlib
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel

from agent.graph import build_graph

load_dotenv()

_PROJECT_ROOT = str(pathlib.Path(__file__).parent.parent)

# Full env passed explicitly — MCP's stdio transport filters env vars by
# default, which would strip ANTHROPIC_API_KEY and Qdrant settings.
_MCP_SERVER_CONFIG = {
    "arxiv": {
        "command": sys.executable,
        "args": ["-m", "mcp_server"],
        "transport": "stdio",
        "env": {**dict(os.environ), "PYTHONPATH": _PROJECT_ROOT},
        "cwd": _PROJECT_ROOT,
    }
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with MultiServerMCPClient(_MCP_SERVER_CONFIG) as client:
        tools = client.get_tools()
        app.state.graph = build_graph(tools)
        yield


app = FastAPI(title="ArXiv Research Agent", version="0.1.0", lifespan=lifespan)


# --- models ---


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    thread_id: str


# --- dependencies ---


def get_graph(request: Request):
    return request.app.state.graph


# --- routes ---


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, graph=Depends(get_graph)):
    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=body.message)]},
            config={"configurable": {"thread_id": body.thread_id}},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(
        response=result["messages"][-1].content,
        thread_id=body.thread_id,
    )
