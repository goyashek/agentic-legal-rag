"""Intent expander / fact extractor. Fans one messy narrative into sub-queries.

A common legal failure: "someone broke into my house and stole my phone" involves
house-trespass and theft, but top-k on the single query won't surface both. This node
(Gemini Flash) pulls out the distinct offences and emits 3-5 parallel sub-queries, one
per offence. The retriever runs per sub-query and results get deduped.

This is my fix for the cross-sectional reasoning problem, and the part that took me the
longest to get right. Runs after the router classifies the query as `criminal`.
"""

from __future__ import annotations

from src.agent.state import AgentState


def expand_intent(query: str, *, max_sub_queries: int = 5) -> list[str]:
    """Decompose a narrative into distinct offence-focused sub-queries.

    Returns 1-5 sub-queries. A simple single-offence query just returns [query]
    unchanged, since over-expanding only adds noise and cost.
    """
    raise NotImplementedError("week 2 tue: Gemini Flash offence extraction")


def intent_expander_node(state: AgentState) -> AgentState:
    """LangGraph node. Sets sub_queries. Downstream retrieval fans out over these."""
    raise NotImplementedError("week 2 tue: wrap expand_intent into a node")
