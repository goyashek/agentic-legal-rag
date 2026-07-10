"""Chonkie-based chunking. I ditched my earlier regex splitter, too brittle.

Stage 2 of ingestion. Takes `RawSection`, emits `LegalChunk` ready for metadata
enrichment and indexing.

How this works:
  1. Hierarchy is already set by parse_pdf (Act -> Chapter -> Section).
  2. Chonkie SemanticChunker (backbone potion-base-8M, cos 0.5, 512 tok) runs within
     each section as a safety net. For a statute this usually keeps one section = one
     chunk; only the few long sections (e.g. BNS 356 Defamation, ~8.7KB) split.
  3. Summary-augmented chunking: I prepend a one-sentence summary of the parent
     section to every chunk, which fixes the "right text, wrong parent" mismatch.

Two things I insulate against on purpose:
  - Chonkie's constructor arg is `threshold`; I expose `similarity_threshold` as my
    own public name and map it, so a Chonkie API rename doesn't ripple out here
    (Chonkie is young; NOTES.md "keep frameworks swappable").
  - The summariser is injectable. Default is a deterministic, key-free heading anchor
    so chunking runs in CI without secrets; the Gemini one-liner (`summarize_section`)
    is wired in only when a key exists. "Swap it without touching chunking."
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ingest.parse_pdf import RawSection

# A summariser maps (section_text, heading) -> one-line summary.
Summarizer = Callable[[str, str], str]


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


def _heading_summary(_text: str, heading: str) -> str:
    """Default key-free summary: the section heading is a decent parent anchor.

    It's what a lawyer skims to place a passage, and it needs no LLM call, so
    chunking stays reproducible in CI. `summarize_section` upgrades this when a
    Gemini key is available.
    """
    return heading.strip().rstrip(".")


def chunk_sections(
    sections: Iterable[RawSection],
    *,
    embedding_model: str = "minishlab/potion-base-8M",
    similarity_threshold: float = 0.5,
    max_tokens: int = 512,
    summary_augment: bool = True,
    summarizer: Summarizer | None = None,
) -> list[LegalChunk]:
    """Semantic-chunk each section, optionally prepending a parent summary.

    embedding_model is Chonkie's SemanticChunker backbone (its own default is
    minishlab/potion-base-32M; I pin 8M per the locked stack). Returns LegalChunks
    with chunk_id, act, section_id, heading and text set; metadata is left for
    enrich_metadata. chunk_id is stable across runs so index upserts stay idempotent,
    and every chunk traces back to exactly one (act, section_id).

    summarizer overrides how the prepended summary is produced; defaults to a
    deterministic heading anchor when summary_augment is on and none is passed.
    """
    from chonkie import SemanticChunker

    chunker = SemanticChunker(
        embedding_model=embedding_model,
        threshold=similarity_threshold,  # Chonkie's own arg name
        chunk_size=max_tokens,
    )
    if summary_augment and summarizer is None:
        summarizer = _heading_summary

    chunks: list[LegalChunk] = []
    for sec in sections:
        summary = summarizer(sec.text, sec.heading) if summary_augment and summarizer else ""

        pieces = chunker.chunk(sec.text)
        # A statutory section is a coherent legal unit. If the whole thing fits the
        # embedding budget, keep it as ONE chunk — splitting it hurts retrieval and
        # produces junk fragments (BNS s.1 "Short title", 356 tok, was splitting into
        # 5 pieces incl. 22- and 26-token scraps). Only fall back to Chonkie's semantic
        # split when the section genuinely overflows max_tokens (e.g. BNS s.356
        # Defamation, ~1940 tok). SemanticChunker returns [] on degenerate text, so
        # that also collapses to the whole section.
        if pieces and sum(p.token_count for p in pieces) > max_tokens:
            bodies = [p.text for p in pieces]
        else:
            bodies = [sec.text]

        for idx, body in enumerate(bodies):
            text = f"{summary}\n\n{body}" if summary else body
            chunks.append(
                LegalChunk(
                    chunk_id=f"{sec.act}::{sec.section_id}::{idx}",
                    act=sec.act,
                    section_id=sec.section_id,
                    heading=sec.heading,
                    text=text,
                    summary=summary,
                    chapter=sec.chapter,
                )
            )
    return chunks


def summarize_section(text: str, heading: str) -> str:
    """One-sentence summary of a section, prepended to its chunks (Gemini Flash).

    Kept separate so I can cache/batch it and swap it without touching chunking.
    Needs GOOGLE_API_KEY / GEMINI_API_KEY; raises if absent so a missing key fails
    loudly here rather than silently degrading the index.
    """
    import os

    from langchain_google_genai import ChatGoogleGenerativeAI

    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise RuntimeError(
            "summarize_section needs GOOGLE_API_KEY/GEMINI_API_KEY. Use the default "
            "heading summariser for a key-free run."
        )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    prompt = (
        "Summarise this Indian statutory section in ONE plain-language sentence "
        "(what it governs, no citation). Section heading: "
        f"{heading!r}.\n\nSection text:\n{text[:4000]}"
    )
    resp = llm.invoke(prompt)
    return str(resp.content).strip().replace("\n", " ")


def write_chunks_jsonl(chunks: Iterable[LegalChunk], path: str | Path) -> int:
    """Serialise chunks to newline-delimited JSON (data/processed/sections.jsonl).

    Returns the count written. This is the Week 1 Tue deliverable artifact that the
    indexer reads back in Thursday.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            n += 1
    return n
