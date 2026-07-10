"""Query rewriter (HyDE-style). The re-retrieval arm of the self-correction loop.

Fires when the grader fails (< 3 relevant chunks) or the citation validator
rejects a made-up citation. It rewrites/expands the query (a HyDE-style
hypothetical answer, or a targeted rephrase aimed at the bad citation) and then
retrieval runs again.

Bumps `iteration` each time. The graph owns the loop budget (2): once that's
spent it returns with confidence="low" instead of spinning forever.
"""

from __future__ import annotations

from src.agent.state import AgentState


def rewrite_query(
    query: str,
    *,
    reason: str,
    invalid_citations: list[str] | None = None,
) -> str:
    """Produce a better query for the next retrieval pass.

    reason is "low_relevance" or "invalid_citation" and decides how I rewrite.
    invalid_citations are the sections that got fabricated last round, so I can
    nudge retrieval toward the right neighborhood instead of the wrong one.
    """
    raise NotImplementedError("week 2 wed: HyDE / targeted rewrite by reason")


def rewriter_node(state: AgentState) -> AgentState:
    """LangGraph node. Sets a fresh `query` and increments `iteration`.

    The graph's conditional edge checks iteration against RETRIEVAL_LOOP_BUDGET and
    short-circuits to a low-confidence answer once we've run out of tries.
    """
    raise NotImplementedError("week 2 wed: wrap rewrite_query; bump iteration")
