"""API routes: the /query endpoint that runs the agent graph.

I kept this separate from app construction (main.py) so I can unit-test the routes
with a stubbed graph.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from src.agent.graph import answer_query
from src.models.schemas import LegalAdvice, QueryRequest

router = APIRouter()


@router.post("/query", response_model=LegalAdvice)
async def query(req: QueryRequest) -> LegalAdvice:
    """Run the synchronous graph off the event loop and return its final answer."""
    state = await run_in_threadpool(answer_query, req.query)
    answer = state.get("answer") or state.get("fast_path_answer")
    if answer is None:
        raise HTTPException(status_code=500, detail="Agent completed without an answer")
    return answer
