"""FastAPI app: /query (via routes) and /health.

Endpoints:
  POST /query   -> LegalAdvice
  GET  /health  -> HealthResponse (liveness + Qdrant connectivity)

No auth on this API yet. Fine while it's a local demo on my machine, but I
shouldn't expose it publicly like this.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

from src.api.routes import router
from src.models.schemas import HealthResponse

app = FastAPI(
    title="Agentic Legal RAG",
    description="Self-correcting RAG for Indian criminal law (BNS/BNSS/BSA).",
    version="0.1.0",
)

app.include_router(router)


def _qdrant_connected() -> bool:
    """Ping the configured Qdrant server, or the embedded development store."""
    from qdrant_client import QdrantClient

    url = os.getenv("QDRANT_URL")
    client = QdrantClient(url=url) if url else QdrantClient(path="data/processed/qdrant")
    try:
        client.get_collections()
        return True
    finally:
        client.close()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return liveness and whether Qdrant accepted a lightweight request."""
    try:
        connected = await run_in_threadpool(_qdrant_connected)
    except Exception:
        connected = False
    return HealthResponse(
        status="ok" if connected else "degraded",
        qdrant_connected=connected,
        version=app.version,
    )
