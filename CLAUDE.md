# ArXiv Research Agent — Project Brief

## What this is
A conversational agentic system that lets users query ML/NLP research literature
on arXiv. Users ask natural language questions; the agent retrieves relevant
papers, synthesizes grounded answers with citations, and supports multi-turn
follow-up.

## Purpose
Portfolio project for ML engineer job applications. Must be clean, well-tested,
and deployable. Every component should be explainable in an interview.

## Stack
- Agent orchestration: LangGraph
- MCP server: custom Python MCP server (4 tools: search_papers, fetch_paper,
  summarize_paper, find_related)
- Vector store: Qdrant (local Docker for dev, cloud for prod)
- Retrieval: Hybrid BM25 + dense with RRF fusion
- Embeddings: paraphrase-multilingual-MiniLM-L12-v2 (sentence-transformers)
- Eval: RAGAS (faithfulness, answer relevance, context precision)
- API: FastAPI
- CI/CD: GitHub Actions (lint → test → eval gate → Docker build)
- Cloud: AWS EC2 (app) + S3 (paper storage)
- Containerization: Docker + docker-compose for local dev

## Repo structure (target)
arxiv-research-agent/
├── agent/           # LangGraph agent definition and graph
├── mcp_server/      # Custom MCP server and tool definitions
├── rag/             # Ingestion, chunking, embedding, retrieval
├── api/             # FastAPI app
├── eval/            # RAGAS eval harness and test question set
├── infra/           # Dockerfile, docker-compose, GitHub Actions workflows
├── scripts/         # arXiv ingestion script, one-off utilities
├── tests/           # Unit and integration tests
├── CLAUDE.md        # This file
└── README.md        # Architecture diagram + eval results (written last)

## Developer context
- Solo project, Python only
- Qdrant runs locally via Docker during development
- Use .env for all secrets/API keys, never hardcode
- Keep each module independently testable
- Prefer simple and correct over clever and brittle
- RAGAS eval must pass a minimum threshold in CI before Docker build runs

## Build order
1. RAG pipeline (ingest → chunk → embed → store → retrieve)
2. MCP server with all 4 tools
3. LangGraph agent wiring MCP tools
4. FastAPI wrapper
5. RAGAS eval harness
6. GitHub Actions CI/CD
7. AWS deployment