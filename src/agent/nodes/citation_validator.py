"""Deterministic citation validator.

The LLM hallucination checker (checker.py) still runs, but this sits in front of it
and is pure code, basically free on cost and latency:

  parse every [Section X, Act] the generator cited, check each one actually exists
  in the retrieved chunk set. If the answer cites BNS 307 but only 306 was retrieved,
  that citation is made up, so reject and go back to the rewriter.

This catches a confident citation to a section that was never retrieved. I'd rather
do it deterministically than trust another LLM call.
"""

from __future__ import annotations

from src.agent.state import AgentState
from src.models.schemas import LegalAdvice
from src.retrieval.hybrid import RetrievedChunk


def extract_cited_sections(answer: LegalAdvice) -> list[tuple[str, str]]:
    """Pull every (act, section_id) the answer cites.

    Reads the structured `citations` field on LegalAdvice directly, no regex over the
    prose, since generation is Pydantic-constrained and citations come out structured.
    """
    raise NotImplementedError("week 2 fri: read structured citations")


def validate_citations(
    answer: LegalAdvice,
    retrieved: list[RetrievedChunk],
) -> tuple[bool, list[str]]:
    """Check every cited (act, section_id) exists in the retrieved set.

    Returns (all_valid, invalid_citations), where invalid_citations lists the
    "ACT SECTION" strings that were cited but not retrieved. This is the deterministic
    core, so worth testing hard.
    """
    raise NotImplementedError("week 2 fri: set-membership check, cited vs retrieved")


def citation_validator_node(state: AgentState) -> AgentState:
    """LangGraph node. Sets citation_valid + invalid_citations.
    On invalid, the graph routes back to the rewriter (within loop budget).
    """
    raise NotImplementedError("week 2 fri: wrap validate_citations into a node")
