"""API routes: the /query endpoint that runs the agent graph.

I kept this separate from app construction (main.py) so I can unit-test the routes
with a stubbed graph.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.models.schemas import LegalAdvice, QueryRequest

router = APIRouter()


@router.post("/query", response_model=LegalAdvice)
async def query(req: QueryRequest) -> LegalAdvice:
    """Run one query through the agent graph and return structured advice.

    Plan: call answer_query(req.query), map the final AgentState -> LegalAdvice
    (normal, fast_path, OOD, low-confidence all collapse into one), attach trace_url.
    """
    raise NotImplementedError("week 4 mon: invoke graph, map state -> LegalAdvice")
