"""Out-of-domain gate. Cheap distance-threshold rejection before spending LLM calls.

If the best retrieved chunk is beyond a cosine-distance threshold, the query just isn't
in BNS/BNSS/BSA, so short-circuit to "I can't find this in the criminal statutes" before
the grader or generator burn tokens. Picked 0.75 after eyeballing a few queries. This
complements the router's `out_of_scope`: the router catches the obvious non-criminal
stuff, this catches the criminal-sounding-but-not-in-corpus stuff.
"""

from __future__ import annotations

from src.agent.state import AgentState
from src.retrieval.hybrid import RetrievedChunk


def is_out_of_domain(
    retrieved: list[RetrievedChunk],
    *,
    distance_threshold: float = 0.75,
) -> bool:
    """True if even the best chunk is too far to be in-corpus.
    Works off the top chunk's dense distance. Test the boundary case both ways.
    """
    raise NotImplementedError("week 1 fri: best-chunk distance vs threshold")


def ood_gate_node(state: AgentState) -> AgentState:
    """LangGraph node. Sets `ood`. On True, graph routes to a 'not in corpus' answer."""
    raise NotImplementedError("week 1 fri: wrap is_out_of_domain into a node")
