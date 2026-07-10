"""Build the retrieval indices: Qdrant (dense) + in-process BM25 (sparse).

Run once after ingestion. Reads data/processed/sections.jsonl and writes a Qdrant
collection plus a serialized BM25 index. Idempotent on chunk_id so re-running
doesn't duplicate anything.
"""

from __future__ import annotations

from pathlib import Path

from src.ingest.chunk_chonkie import LegalChunk


def load_chunks(sections_jsonl: str | Path) -> list[LegalChunk]:
    """Load the enriched chunks from data/processed/sections.jsonl."""
    raise NotImplementedError("week 1 thu - deserialize LegalChunk records")


def build_qdrant_index(
    chunks: list[LegalChunk],
    *,
    collection: str,
    embed_model: str = "BAAI/bge-large-en-v1.5",
    qdrant_url: str = "http://localhost:6333",
    recreate: bool = False,
) -> int:
    """Embed the chunks and upsert them into a Qdrant collection.

    Each point's payload must include chunk_id, act, section_id, heading, text,
    and the enrichment metadata. The retriever hands these back and the citation
    validator checks against them, so dropping a field here breaks downstream.
    Returns the number of points upserted.
    """
    raise NotImplementedError("week 1 thu - bge embed + Qdrant upsert")


def build_bm25_index(chunks: list[LegalChunk], *, out_path: str | Path) -> None:
    """Build and serialize an in-process rank_bm25 index over the chunk text.

    Kept in-process rather than standing up Elasticsearch, since it's only ~1k
    chunks. The tokenization here has to match the query-time tokenization in
    hybrid.py exactly or the scores won't line up.
    """
    raise NotImplementedError("week 1 thu - rank_bm25 over tokenized chunk text")
