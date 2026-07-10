"""Cross-encoder reranker, on by default.

Reranking is pretty standard now, so bge-reranker-base is in from the start. I
ablate it in notebooks/02_retrieval_ablation.ipynb to see how many points it
actually buys. It cuts the ~20 RRF-fused candidates down to the ~8 the agent
reasons over.
"""

from __future__ import annotations

from src.retrieval.hybrid import RetrievedChunk


class Reranker:
    """bge-reranker-base cross-encoder scoring (query, chunk) pairs."""

    def __init__(self, model: str = "BAAI/bge-reranker-base") -> None:
        raise NotImplementedError("week 1 thu - load the CrossEncoder")

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        *,
        top_k: int = 8,
    ) -> list[RetrievedChunk]:
        """Re-score the candidates by cross-encoder relevance, return top_k.

        I write the score back onto each chunk so the ablation can compare
        ordering before vs after reranking. Aiming for ~200ms on ~20 candidates.
        """
        raise NotImplementedError("week 1 thu - cross-encoder score + resort")
