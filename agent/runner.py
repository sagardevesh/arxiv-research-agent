import asyncio
import os
import pathlib
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.graph import build_graph

load_dotenv()

_PROJECT_ROOT = str(pathlib.Path(__file__).parent.parent)

# Full env is passed explicitly — MCP's stdio transport filters env vars by
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


async def run_cli() -> None:
    print("ArXiv Research Agent — type 'quit' to exit\n")

    async with MultiServerMCPClient(_MCP_SERVER_CONFIG) as client:
        tools = client.get_tools()
        graph = build_graph(tools)
        config = {"configurable": {"thread_id": "cli-session"}}

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_input or user_input.lower() in ("quit", "exit", "q"):
                break

            result = await graph.ainvoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )
            print(f"\nAgent: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    asyncio.run(run_cli())
