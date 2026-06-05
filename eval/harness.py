"""RAGAS evaluation harness for the ArXiv Research Agent RAG pipeline.

Evaluates three metrics: Faithfulness, AnswerRelevancy, ContextPrecision.
Pass injectable retriever_fn / answer_fn for testing without live services.
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Suppress Google Cloud Python-version FutureWarnings pulled in by ragas.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, Faithfulness

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

    dense = ctx.dense_embedder.embed([question])[0]
    sparse = ctx.sparse_embedder.embed([question])[0]
    from rag.retriever import hybrid_search

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


def _make_ragas_llm():
    import anthropic
    import instructor
    from ragas.llms import InstructorLLM

    client = instructor.from_anthropic(anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]))
    return InstructorLLM(client=client, model="claude-sonnet-4-6", provider="anthropic")


def _make_ragas_embeddings():
    from ragas.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


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
        ragas_llm: RAGAS LLM wrapper. Defaults to Claude via langchain wrapper.

    Returns:
        Dict mapping metric name to score (0.0–1.0).
    """
    retriever_fn = retriever_fn or _default_retriever
    answer_fn = answer_fn or _default_answer_fn
    ragas_llm = ragas_llm or _make_ragas_llm()
    ragas_embeddings = ragas_embeddings or _make_ragas_embeddings()

    questions = load_questions()
    samples = []
    for q in questions:
        contexts = retriever_fn(q["question"])
        response = answer_fn(q["question"], contexts)
        samples.append(
            SingleTurnSample(
                user_input=q["question"],
                retrieved_contexts=contexts,
                response=response,
                reference=q["ground_truth"],
            )
        )

    dataset = EvaluationDataset(samples=samples)
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextPrecision(llm=ragas_llm),
    ]

    result = evaluate(dataset, metrics=metrics, show_progress=False)

    return {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_precision": float(result["context_precision"]),
    }


def check_thresholds(scores: dict[str, float]) -> bool:
    """Print per-metric results and return True if all thresholds are met."""
    passed = True
    for metric, threshold in THRESHOLDS.items():
        score = scores.get(metric, 0.0)
        status = "PASS" if score >= threshold else "FAIL"
        print(f"  {metric}: {score:.3f}  (threshold ≥ {threshold})  [{status}]")
        if score < threshold:
            passed = False
    return passed
