"""Deterministic citation validator.

The LLM hallucination checker (checker.py) still runs, but this sits in front of it
and is pure code, basically free on cost and latency:

  parse every [Section X, Act] the generator cited, check each one actually exists
  in the retrieved chunk set. If the answer cites BNS 307 but only 306 was retrieved,
  that citation is made up, so reject and go back to the rewriter.

This catches a confident citation to a section that was never retrieved. I'd rather
do it deterministically than trust another LLM call.

Section-level granularity: the generator legitimately cites a subsection ("318(2)")
but the corpus + retrieval are keyed at section level ("318"), so both sides are
normalized to the section before the membership test — the same trick fast_path uses
on the query side ("103(2)" -> "103"). A citation whose normalized section isn't in
the retrieved set is the real fabrication to reject.
"""

from __future__ import annotations

import re

from src.agent.state import AgentState
from src.models.schemas import LegalAdvice
from src.retrieval.hybrid import RetrievedChunk

# Keep "103" and "63A", drop the "(1)"/"(2)" subsection tail. Matches
# fast_path.detect_exact_section's normalization so the two agree.
_SECTION_RE = re.compile(r"\d+[A-Z]?")


def normalize_section(section_id: str) -> str:
    """Reduce a printed section id to its section-level key (drop subsection)."""
    m = _SECTION_RE.match(section_id.strip())
    return m.group(0) if m else section_id.strip()


def extract_cited_sections(answer: LegalAdvice) -> list[tuple[str, str]]:
    """Pull every (act, section_id) the answer cites.

    Reads the structured `citations` field on LegalAdvice directly, no regex over the
    prose, since generation is Pydantic-constrained and citations come out structured.
    Act is upper-cased so the membership check is case-insensitive on the act code.
    """
    return [(c.act.strip().upper(), c.section_id.strip()) for c in answer.citations]


def validate_citations(
    answer: LegalAdvice,
    retrieved: list[RetrievedChunk],
) -> tuple[bool, list[str]]:
    """Check every cited (act, section_id) exists in the retrieved set.

    Returns (all_valid, invalid_citations), where invalid_citations lists the
    "ACT SECTION" strings that were cited but not retrieved. Both sides are
    normalized to section level first, so citing "318(2)" is valid when section
    "318" was retrieved. This is the deterministic core, so worth testing hard.
    """
    retrieved_keys = {
        (c.chunk.act.strip().upper(), normalize_section(c.chunk.section_id)) for c in retrieved
    }
    invalid: list[str] = []
    for act, section_id in extract_cited_sections(answer):
        if (act, normalize_section(section_id)) not in retrieved_keys:
            invalid.append(f"{act} {section_id}")
    return (not invalid, invalid)


def citation_validator_node(state: AgentState) -> AgentState:
    """LangGraph node. Sets citation_valid + invalid_citations.

    On invalid, the graph routes back to the rewriter (within loop budget). An
    answer with no citations at all is treated as invalid — a substantive legal
    answer must cite something, and a citation-free pass would skip the whole point.
    """
    answer = state.get("answer")
    notes = state.get("trace_notes", [])
    if answer is None:
        return {
            "citation_valid": False,
            "invalid_citations": [],
            "trace_notes": [*notes, "citation_validator: no answer"],
        }

    # The generator only sees graded chunks when any passed the filter. Validating
    # against the larger retrieval pool could approve a citation to text the model
    # never received.
    generation_context = state.get("relevant_chunks") or state.get("retrieved", [])
    valid, invalid = validate_citations(answer, generation_context)
    # No citations at all is not a valid substantive answer.
    if not answer.citations:
        valid = False
    return {
        "citation_valid": valid,
        "invalid_citations": invalid,
        "trace_notes": [
            *notes,
            f"citation_validator: {'valid' if valid else 'invalid'}"
            + (f" {invalid}" if invalid else ""),
        ],
    }
