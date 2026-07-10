"""Query router. Gemini Flash classifier that runs once the fast path misses.

Sorts the query into one of three routes:
  criminal            -> go on to intent expansion + retrieval
  out_of_scope        -> polite "not a criminal-law question" answer
  needs_clarification -> ask a follow-up

This is the semantic gate. The cheap fast path (fast_path.py) and the
distance-based OOD gate (ood_gate.py) wrap around it, so this only fires when
those two can't decide. I keep the prompt text in agent/prompts/ so it isn't
buried in code.
"""

from __future__ import annotations

from src.agent.state import AgentState, Route


def classify(query: str) -> Route:
    """Return the route for a query. Gemini Flash, temperature 0 so it's stable."""
    raise NotImplementedError("week 2 mon: gemini flash 3-way classifier")


def router_node(state: AgentState) -> AgentState:
    """LangGraph node. Sets `route`. The graph branches on the value."""
    raise NotImplementedError("week 2 mon: wrap classify into a node")
