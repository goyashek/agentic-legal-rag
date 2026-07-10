"""Answer generator. Pydantic-constrained cited advice via Gemini Pro + instructor.

Builds the LegalAdvice output from the graded chunks. I use Gemini Pro for the
India-context reasoning, wired up with instructor.from_provider("google/gemini-2.5-pro"),
which gives a client I call .create(response_model=LegalAdvice, messages=[...]) on.
That way citations come back as structured (act, section_id) pairs the
deterministic citation validator can check exactly. The prompt has to say cite
ONLY from the chunks I hand it, or the model happily invents section numbers.
"""

from __future__ import annotations

from src.agent.state import AgentState
from src.models.schemas import LegalAdvice
from src.retrieval.hybrid import RetrievedChunk


def generate_answer(query: str, chunks: list[RetrievedChunk]) -> LegalAdvice:
    """Generate structured LegalAdvice grounded in `chunks`.

    Two things I want to hold true: every citation points at a section that's
    actually in `chunks` (validator enforces it, prompt just makes it likely),
    and the output validates against LegalAdvice (instructor retries off-schema
    returns).
    """
    raise NotImplementedError("week 2 thu: gemini pro + instructor structured gen")


def generator_node(state: AgentState) -> AgentState:
    """LangGraph node. Sets `answer`. Flows straight into the citation validator."""
    raise NotImplementedError("week 2 thu: wrap generate_answer into a node")
