"""Hallucination checker. LLM-as-judge faithfulness pass, after the deterministic validator.

The deterministic citation validator (citation_validator.py) already guaranteed
every cited section was actually retrieved. This node catches the subtler failure:
claims that paraphrase or overstate what the cited sections really say. DeepSeek
Flash judges each claim against the text it cites.

If it comes back "unfaithful", route back to the rewriter/generator, as long as
we're still inside the loop budget.

Client is injected (defaults to shared Flash) so the node logic tests with a fake
at zero quota; live tests gate on the key.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.agent.llm import get_client, load_prompt
from src.agent.state import AgentState
from src.models.schemas import LegalAdvice
from src.retrieval.hybrid import RetrievedChunk


class FaithfulnessVerdict(BaseModel):
    """Structured faithfulness verdict. `instructor` forces the model to fill this."""

    faithful: bool = Field(description="True only if every claim is backed by the cited text")
    unsupported_claims: list[str] = Field(
        default_factory=list, description="Claims not backed by the cited section text"
    )


def _cited_context(answer: LegalAdvice, chunks: list[RetrievedChunk]) -> str:
    """Render the text of the sections the answer actually cites.

    The checker only needs the cited sections, not the whole retrieved set — that's
    what "are these claims supported by their sources" means. Falls back to all
    chunks if nothing lines up (shouldn't happen post-validator, but stay robust).
    """
    cited = {(c.act.strip().upper(), c.section_id.strip()) for c in answer.citations}
    picked = [
        c for c in chunks if (c.chunk.act.strip().upper(), c.chunk.section_id.strip()) in cited
    ]
    use = picked or chunks
    return "\n\n".join(
        f"[{c.chunk.act} Section {c.chunk.section_id}] {c.chunk.heading}\n{c.chunk.text[:4000]}"
        for c in use
    )


def check_faithfulness(
    answer: LegalAdvice,
    chunks: list[RetrievedChunk],
    *,
    client: object | None = None,
) -> tuple[bool, list[str]]:
    """Judge whether every claim is actually backed by its cited source text.

    Returns (faithful, unsupported_claims). faithful is True only when nothing is
    unsupported. Runs after citations are already structurally valid.
    """
    client = client or get_client("flash")
    prompt = load_prompt("checker").format(
        answer=answer.answer, context=_cited_context(answer, chunks)
    )
    verdict: FaithfulnessVerdict = client.create(  # type: ignore[attr-defined]
        messages=[{"role": "user", "content": prompt}],
        response_model=FaithfulnessVerdict,
        temperature=0,
    )
    return (verdict.faithful, verdict.unsupported_claims)


def checker_node(state: AgentState, *, client: object | None = None) -> AgentState:
    """LangGraph node. Sets `faithful`. Basically terminal: faithful -> output."""
    answer = state["answer"]
    faithful, unsupported = check_faithfulness(
        answer, state.get("retrieved", []), client=client
    )
    notes = state.get("trace_notes", [])
    return {
        "faithful": faithful,
        "trace_notes": [
            *notes,
            f"checker: {'faithful' if faithful else 'unfaithful'}"
            + (f" {unsupported}" if unsupported else ""),
        ],
    }
