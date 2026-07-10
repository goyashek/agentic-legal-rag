"""FastAPI app: /query (via routes) and /health.

Endpoints:
  POST /query   -> LegalAdvice
  GET  /health  -> HealthResponse (liveness + Qdrant connectivity)

No auth on this API yet. Fine while it's a local demo on my machine, but I
shouldn't expose it publicly like this.
"""

from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import router
from src.models.schemas import HealthResponse

app = FastAPI(
    title="Agentic Legal RAG",
    description="Self-correcting RAG for Indian criminal law (BNS/BNSS/BSA).",
    version="0.1.0",
)

app.include_router(router)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness + Qdrant connectivity check."""
    raise NotImplementedError("week 4 mon: ping qdrant, report status")
