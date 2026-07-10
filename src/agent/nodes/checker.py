"""Hallucination checker. LLM-as-judge faithfulness pass, after the deterministic validator.

The deterministic citation validator (citation_validator.py) already guaranteed
every cited section was actually retrieved. This node catches the subtler failure:
claims that paraphrase or overstate what the cited sections really say. Gemini
Flash judges each claim against the text it cites.

If it comes back "unfaithful", route back to the rewriter/generator, as long as
we're still inside the loop budget.
"""

from __future__ import annotations

from src.agent.state import AgentState
from src.models.schemas import LegalAdvice
from src.retrieval.hybrid import RetrievedChunk


def check_faithfulness(
    answer: LegalAdvice,
    chunks: list[RetrievedChunk],
) -> tuple[bool, list[str]]:
    """Judge whether every claim is actually backed by its cited source text.

    Returns (faithful, unsupported_claims). faithful is True only when nothing is
    unsupported. Runs after citations are already structurally valid.
    """
    raise NotImplementedError("week 2 fri: gemini flash claim-vs-source judge")


def checker_node(state: AgentState) -> AgentState:
    """LangGraph node. Sets `faithful`. Basically terminal: faithful -> output."""
    raise NotImplementedError("week 2 fri: wrap check_faithfulness into a node")
