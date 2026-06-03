import os
import warnings

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

SYSTEM_PROMPT = """You are a research assistant specialising in ML/NLP papers on arXiv.

When answering a question:
1. Use search_papers to find relevant papers.
2. Use fetch_paper or summarize_paper to get deeper details on specific papers.
3. Use find_related to discover related work when asked.
4. Cite every paper you reference as: [Title (arXiv:ID)]
5. Be specific and grounded — only claim what the papers actually say."""


def build_graph(tools: list[BaseTool]):
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    # Suppress the LangGraph V1 deprecation — the replacement (langchain.agents)
    # is not yet released and this is the only available API.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return create_react_agent(
            llm,
            tools,
            prompt=SYSTEM_PROMPT,
            checkpointer=MemorySaver(),
        )
