"""Hybrid retrieval: dense (Qdrant) + sparse (BM25) fused with RRF.

Dense alone puts trespass near theft, which is wrong for legal text. BM25 alone
has no semantic flex. RRF (k=60) fuses the two. Output feeds the reranker, then
the agent graph.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.ingest.chunk_chonkie import LegalChunk


@dataclass
class RetrievedChunk:
    """A chunk plus its retrieval scores, carried through the graph.

    I keep dense/sparse/rrf scores separate so the ablation notebook can attribute
    wins to each signal instead of one blended number.
    """

    chunk: LegalChunk
    rrf_score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None


def reciprocal_rank_fusion(
    dense_ranking: list[str],
    sparse_ranking: list[str],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse two ranked chunk_id lists with RRF.

    Score per chunk = sum over the lists of 1 / (k + rank). Pure function, no I/O,
    so it's easy to unit-test (tests/test_retrieval.py). Returns [(chunk_id,
    rrf_score)] sorted by score desc.
    """
    raise NotImplementedError("week 1 thu - RRF; unit-test this in isolation")


class HybridRetriever:
    """Wraps the Qdrant + BM25 indices behind one `retrieve` call."""

    def __init__(
        self,
        *,
        collection: str,
        bm25_path: str,
        embed_model: str = "BAAI/bge-large-en-v1.5",
        qdrant_url: str = "http://localhost:6333",
        rrf_k: int = 60,
    ) -> None:
        raise NotImplementedError("week 1 thu - load Qdrant client + BM25 index")

    def retrieve(self, query: str, *, top_k: int = 20) -> list[RetrievedChunk]:
        """Dense + sparse search, RRF-fused, top_k returned (before reranking).

        Score fields get filled in so the ablation has something to work with.
        """
        raise NotImplementedError("week 1 thu - dense + sparse + RRF")
