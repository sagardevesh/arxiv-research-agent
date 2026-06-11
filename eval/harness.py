"""RAGAS evaluation harness for the ArXiv Research Agent RAG pipeline.

Evaluates three metrics: Faithfulness, AnswerRelevancy, ContextPrecision.
Pass injectable retriever_fn / answer_fn for testing without live services.

Uses the legacy ragas.metrics API (not ragas.metrics.collections) — the
collections API is incompatible with ragas.evaluate().

LLM: Claude Haiku (fast/cheap for CI). Embeddings: HuggingFaceEmbeddings
(FastEmbedEmbeddings exposes a TextEmbedding object as .model, which breaks
RAGAS's EmbeddingUsageEvent string validation).
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Suppress FutureWarnings from Google Cloud packages pulled in by ragas.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

QUESTIONS_PATH = Path(__file__).parent / "questions.jsonl"

THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.7,
    "answer_relevancy": 0.7,
    "context_precision": 0.7,
}


def load_questions() -> list[dict]:
    with open(QUESTIONS_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def _default_retriever(question: str) -> list[str]:
    """Retrieve context strings for a question using the live RAG pipeline."""
    from mcp_server.context import ctx
    from rag.retriever import hybrid_search

    dense = ctx.dense_embedder.embed([question])[0]
    sparse = ctx.sparse_embedder.embed([question])[0]
    hits = hybrid_search(ctx.qdrant, dense, sparse, top_k=5)
    return [f"{h['title']}\n{h.get('abstract', '')}" for h in hits]


def _default_answer_fn(question: str, contexts: list[str]) -> str:
    """Generate an answer from retrieved contexts using Claude."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    context_block = "\n\n---\n\n".join(contexts)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system="Answer the question using only the provided context. Be concise and factual.",
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context_block}\n\nQuestion: {question}",
            }
        ],
    )
    return response.content[0].text


def _configure_metrics(ragas_llm, ragas_embeddings) -> None:
    """Attach LLM and embeddings to the module-level metric singletons."""
    faithfulness.llm = ragas_llm
    context_precision.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_embeddings


def _make_ragas_llm():
    from langchain_anthropic import ChatAnthropic
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(
        ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=os.environ["ANTHROPIC_API_KEY"])
    )


def _make_ragas_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5"))


def run_eval(
    retriever_fn: Callable[[str], list[str]] | None = None,
    answer_fn: Callable[[str, list[str]], str] | None = None,
    ragas_llm=None,
    ragas_embeddings=None,
) -> dict[str, float]:
    """Run RAGAS evaluation and return a scores dict.

    Args:
        retriever_fn: (question) -> list[context_str]. Defaults to live RAG pipeline.
        answer_fn: (question, contexts) -> answer_str. Defaults to Claude via API.
        ragas_llm: RAGAS LLM wrapper. Defaults to LangchainLLMWrapper(ChatAnthropic).
        ragas_embeddings: RAGAS embeddings. Defaults to HuggingFaceEmbeddings bge-small-en-v1.5.

    Returns:
        Dict mapping metric name to score (0.0–1.0).
    """
    retriever_fn = retriever_fn or _default_retriever
    answer_fn = answer_fn or _default_answer_fn
    ragas_llm = ragas_llm or _make_ragas_llm()
    ragas_embeddings = ragas_embeddings or _make_ragas_embeddings()

    _configure_metrics(ragas_llm, ragas_embeddings)

    questions = load_questions()
    rows: list[dict] = []
    for q in questions:
        contexts = retriever_fn(q["question"])
        response = answer_fn(q["question"], contexts)
        rows.append(
            {
                "question": q["question"],
                "answer": response,
                "contexts": contexts,
                "ground_truth": q["ground_truth"],
            }
        )

    dataset = Dataset.from_list(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ragas.run_config import RunConfig

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            raise_exceptions=False,
            run_config=RunConfig(timeout=600, max_workers=1),
        )

    # result[metric] is a list of per-sample scores (NaN for failed rows).
    # Compute nanmean so that individual timeouts don't invalidate the whole run.
    import numpy as np

    def _mean(key: str) -> float:
        val = result[key]
        if isinstance(val, (list, tuple)):
            return float(np.nanmean(val))
        return float(val)

    return {
        "faithfulness": _mean("faithfulness"),
        "answer_relevancy": _mean("answer_relevancy"),
        "context_precision": _mean("context_precision"),
    }


def check_thresholds(scores: dict[str, float]) -> bool:
    """Print per-metric results and return True if all thresholds are met."""
    passed = True
    for metric, threshold in THRESHOLDS.items():
        score = scores.get(metric, 0.0)
        status = "PASS" if score >= threshold else "FAIL"
        print(f"  {metric}: {score:.3f}  (threshold >= {threshold})  [{status}]")
        if score < threshold:
            passed = False
    return passed
