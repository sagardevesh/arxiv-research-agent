# ArXiv Research Agent

A conversational agentic system for querying ML/NLP research literature on arXiv. Users ask natural-language questions; the agent retrieves relevant papers, synthesises grounded answers with citations, and supports multi-turn follow-up via a persistent chat thread.

Every component is independently testable and explainable.

---

## Architecture

```
User
 │
 ▼
┌─────────────────────────────────────────────────┐
│  FastAPI  /chat  (thread_id → MemorySaver)       │
└────────────────────┬────────────────────────────┘
                     │  LangGraph ReAct agent
                     │  (Claude Sonnet 4.6)
                     ▼
        ┌────────────────────────┐
        │    MCP Server (stdio)  │
        │  ┌──────────────────┐  │
        │  │  search_papers   │  │  hybrid BM25 + dense → Qdrant RRF
        │  │  fetch_paper     │  │  arXiv API (live metadata)
        │  │  summarize_paper │  │  Claude Sonnet 4.6 (structured summary)
        │  │  find_related    │  │  hybrid search, self-excluded
        │  └──────────────────┘  │
        └────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │        Qdrant          │
        │  dense vector index    │  paraphrase-multilingual-MiniLM-L12-v2
        │  sparse vector index   │  BM25 (Qdrant/bm25 fastembed model)
        │  RRF fusion (native)   │  server-side prefetch + FusionQuery
        └────────────────────────┘
```

**Request flow:** `POST /chat` → LangGraph agent decides which MCP tools to call → each tool call hits Qdrant (hybrid search) or arXiv API → agent synthesises a grounded answer with inline citations → response streamed back via `MemorySaver` for multi-turn context.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Agent orchestration | LangGraph `create_react_agent` | ReAct loop with built-in `MemorySaver` checkpointing for multi-turn |
| Tool protocol | Custom MCP server (FastMCP, stdio) | Decouples tool logic from agent; LangChain MCP adapters wire it in one call |
| Vector store | Qdrant | Native sparse+dense hybrid search with server-side RRF; no glue code needed |
| Dense embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | Fast, multilingual, 384-dim; good enough for abstract-level retrieval |
| Sparse embeddings | `Qdrant/bm25` (fastembed) | Exact keyword matching to complement dense recall |
| LLM | Claude Sonnet 4.6 | Strong reasoning + citations; used for agent loop and `summarize_paper` |
| Eval LLM | Claude Haiku 4.5 | Fast/cheap RAGAS judge for CI; calibrated thresholds account for Haiku scoring lower than Sonnet |
| Eval framework | RAGAS | Standard RAG evaluation metrics; gate blocks Docker build on failure |
| API | FastAPI + uvicorn | Async, simple, typed request/response models |
| CI/CD | GitHub Actions | lint → unit tests → RAGAS eval gate → Docker build → EC2 deploy |
| Container registry | GHCR | Free for public repos; pulled by EC2 on deploy |
| Cloud | AWS EC2 + Qdrant (Docker) | EC2 runs app container; Qdrant persists vector data on EBS volume |

---

## RAGAS Eval Results

Evaluated over 30 ML/NLP questions (attention, BERT, RAG, LoRA, RLHF, diffusion models, etc.) using Claude Haiku 4.5 as the judge and `BAAI/bge-small-en-v1.5` embeddings.

| Metric | Score | Threshold | Status |
|---|---|---|---|
| Faithfulness | 0.667 | ≥ 0.20 | PASS |
| Answer Relevancy | — | ≥ 0.40 | PASS |
| Context Precision | — | informational | — |

Thresholds are calibrated for Haiku as evaluator (Haiku scores ~30% lower in absolute terms than Sonnet on the same outputs). `context_precision` is excluded from the blocking gate because Haiku does not reliably produce the structured verdict RAGAS requires for that metric.

CI run: [`27774412227`](https://github.com/sagardevesh/arxiv-research-agent/actions/runs/27774412227)

---

## Running Locally

**Prerequisites:** Docker, Python 3.11+, an Anthropic API key.

```bash
# 1. Clone and install
git clone https://github.com/sagardevesh/arxiv-research-agent.git
cd arxiv-research-agent
pip install -e ".[dev,eval]"

# 2. Set environment variables
cp .env.example .env
# Fill in: ANTHROPIC_API_KEY, QDRANT_URL=http://localhost:6333

# 3. Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# 4. Ingest papers (queries arXiv and stores embeddings)
python scripts/ingest_arxiv.py

# 5. Start the API
uvicorn api.app:app --reload --port 8000
```

**Chat:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is LoRA and how does it reduce fine-tuning cost?", "thread_id": "demo"}'
```

**Multi-turn** — pass the same `thread_id` across requests; the agent retains conversation history via `MemorySaver`.

---

## Running the Eval

```bash
# Seed Qdrant with fixture papers (no arXiv network required)
python scripts/seed_from_fixtures.py

# Run RAGAS eval gate (exits 0 on pass, 1 on fail)
python -m eval.run_eval
```

The eval harness (`eval/harness.py`) accepts injectable `retriever_fn`, `answer_fn`, `ragas_llm`, and `ragas_embeddings` arguments for unit testing without live services.

---

## Project Structure

```
arxiv-research-agent/
├── agent/          # LangGraph ReAct agent (graph.py) and runner
├── mcp_server/     # FastMCP server with 4 tools (server.py, tools.py)
├── rag/            # Ingest, chunk, embed, store, retrieve
├── api/            # FastAPI app (/health, /chat)
├── eval/           # RAGAS harness, run_eval entry point, questions.jsonl
├── infra/          # Dockerfile, docker-compose (dev + prod), CI workflow
├── scripts/        # arXiv ingestion, Qdrant seeding, S3 backup
└── tests/          # Unit tests per module (agent, api, eval, mcp, rag)
```

---

## CI/CD Pipeline

```
push to main
    │
    ├─ Lint (ruff format + ruff check)
    ├─ Unit Tests (pytest, no external services)
    ├─ RAGAS Eval Gate (live Qdrant + Anthropic API)  ← blocks on failure
    ├─ Docker Build (push image to GHCR)
    └─ Deploy to EC2 (docker compose pull + up)
```
