"""Chonkie-based chunking. I ditched my earlier regex splitter, too brittle.

Stage 2 of ingestion. Takes `RawSection`, emits `LegalChunk` ready for metadata
enrichment and indexing.

How this works:
  1. Hierarchy is already set by parse_pdf (Act -> Chapter -> Section).
  2. Chonkie SemanticChunker (default backbone potion-base-32M, cos 0.5, 512 tok)
     runs within each section as a safety net. For a statute this usually keeps
     one section = one chunk. I call its .chunk(text) method per section.
  3. Summary-augmented chunking: I prepend a one-sentence summary of the parent
     section to every chunk, which fixes the "right text, wrong parent" mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ingest.parse_pdf import RawSection


@dataclass
class LegalChunk:
    """A retrieval-ready chunk. Metadata gets filled across the chunk + enrich stages.

    The citation validator (agent/nodes/citation_validator.py) checks every
    generated citation against `act` + `section_id` on the retrieved chunks, so
    these fields have to stay exact.
    """

    chunk_id: str                       # stable id, e.g. "BNS::103::0"
    act: str
    section_id: str
    heading: str
    text: str                           # summary-augmented body (summary prepended)
    summary: str = ""                   # the one-sentence parent summary
    chapter: str | None = None
    # enrichment fields (filled by enrich_metadata), see that module for semantics
    metadata: dict = field(default_factory=dict)


def chunk_sections(
    sections: list["RawSection"],
    *,
    embedding_model: str = "minishlab/potion-base-8M",
    similarity_threshold: float = 0.5,
    max_tokens: int = 512,
    summary_augment: bool = True,
) -> list[LegalChunk]:
    """Semantic-chunk each section, optionally prepending a parent summary.

    embedding_model is Chonkie's SemanticChunker backbone (its own default is
    minishlab/potion-base-32M). Returns LegalChunks with chunk_id, act,
    section_id, heading and text set; metadata is left for enrich_metadata.
    I keep chunk_id stable across runs so index upserts stay idempotent, and
    every chunk traces back to exactly one (act, section_id).
    """
    raise NotImplementedError("week 1 tue: chonkie semanticchunker + summary augment")


def summarize_section(text: str, heading: str) -> str:
    """One-sentence summary of a section, prepended to its chunks (Gemini Flash).

    Kept separate so I can cache/batch it and swap it without touching chunking.
    """
    raise NotImplementedError("week 1 tue: gemini flash one-liner")
