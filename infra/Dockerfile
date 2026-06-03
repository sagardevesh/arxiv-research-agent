FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only the package manifest first so the dependency install layer is
# cached independently of source changes.
COPY pyproject.toml .

# Stub out the package directories so pip can resolve the project metadata
# without the real source present.  The real source is copied below.
# Editable install (-e) makes Python resolve imports through /app directly,
# so the subsequent COPY steps replace stubs with real source transparently.
RUN mkdir -p rag agent mcp_server api && \
    for d in rag agent mcp_server api; do touch $d/__init__.py; done && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Real source — this layer only invalidates when code changes, not deps.
COPY rag/       rag/
COPY agent/     agent/
COPY mcp_server/ mcp_server/
COPY api/       api/

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
