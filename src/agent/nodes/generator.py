"""Answer generator. Pydantic-constrained cited advice via DeepSeek Pro.

Builds the LegalAdvice output from the graded chunks through `get_client("pro")`.

instructor returns citations as structured (act, section_id) pairs the
deterministic citation validator (Fri) checks exactly. The prompt forbids citing
anything outside the provided chunks; the validator enforces it, the prompt just
makes violations rare. Client is injected so tests generate with a fake at zero
quota.
"""

from __future__ import annotations

from src.agent.llm import get_client, load_prompt
from src.agent.state import AgentState
from src.models.schemas import LegalAdvice
from src.retrieval.hybrid import RetrievedChunk


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Render the graded chunks as a citable list.

    One block per chunk, leading with the exact act + section_id the model must
    cite verbatim, so the returned Citation fields line up with what the validator
    checks against the retrieved set.
    """
    blocks: list[str] = []
    for c in chunks:
        ch = c.chunk
        blocks.append(f"[{ch.act} Section {ch.section_id}] {ch.heading}\n{ch.text[:4000]}")
    return "\n\n".join(blocks)


def generate_answer(
    query: str, chunks: list[RetrievedChunk], *, client: object | None = None
) -> LegalAdvice:
    """Generate structured LegalAdvice grounded in `chunks`.

    Two things I want to hold true: every citation points at a section that's
    actually in `chunks` (validator enforces it, prompt just makes it likely),
    and the output validates against LegalAdvice (instructor retries off-schema
    returns). `query` and `in_corpus` are set deterministically after generation
    so the model can't drift them.
    """
    client = client or get_client("pro")
    prompt = load_prompt("generator").format(query=query, context=_format_context(chunks))
    advice: LegalAdvice = client.create(  # type: ignore[attr-defined]
        messages=[{"role": "user", "content": prompt}],
        response_model=LegalAdvice,
        temperature=0,
        # The final schema includes citations and can exceed the control-node cap.
        max_tokens=1536,
    )
    # Pin the fields the pipeline owns, not the model.
    advice.query = query
    advice.in_corpus = True
    return advice


def generator_node(state: AgentState, *, client: object | None = None) -> AgentState:
    """LangGraph node. Sets `answer`. Flows straight into the citation validator.

    Generates over the graded-relevant chunks (the grader already filtered noise);
    falls back to the full retrieved set if grading left nothing recorded.
    """
    chunks = state.get("relevant_chunks") or state.get("retrieved", [])
    answer = generate_answer(state["query"], chunks, client=client)
    notes = state.get("trace_notes", [])
    return {
        "answer": answer,
        "trace_notes": [*notes, f"generator: {len(answer.citations)} citations"],
    }
