"""Relevance grader. LLM-as-judge over the retrieved chunks, run in parallel.

Grades each reranked chunk yes/no on whether it's actually relevant to the query
(Gemini Flash, fired concurrently). If >= 3 chunks pass I move on to generation;
if not, the graph kicks over to the rewriter for another retrieval loop (budget 2).

The point is to stop generation from reasoning over context that's only
marginally on-topic. That's one of the quiet ways single-shot RAG gives you a
wrong-but-confident answer.
"""

from __future__ import annotations

from src.agent.state import AgentState
from src.retrieval.hybrid import RetrievedChunk


def grade_chunk(query: str, chunk: RetrievedChunk) -> bool:
    """True if the chunk is relevant to the query. One yes/no judge call."""
    raise NotImplementedError("week 2 wed: gemini flash relevance judge")


def grade_chunks(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Grade all chunks in parallel and keep only the relevant ones.

    asyncio over the chunks, since one at a time was too slow.
    """
    raise NotImplementedError("week 2 wed: parallel grade + filter")


def grader_node(state: AgentState) -> AgentState:
    """LangGraph node. Sets relevant_chunks + grade_pass (True when >= 3 relevant)."""
    raise NotImplementedError("week 2 wed: wrap grade_chunks; set grade_pass")
