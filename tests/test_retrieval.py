"""Tests for the retrieval layer.

I'm testing the pure functions (no I/O, no LLM) first and hardest since they're the
deterministic core and CI can run them without any secrets: reciprocal_rank_fusion
(the RRF math) and the eval metrics (section_precision_at_k, section_recall,
citation_accuracy).
"""

from __future__ import annotations

import pytest

from src.retrieval.hybrid import reciprocal_rank_fusion


class TestReciprocalRankFusion:
    @pytest.mark.skip(reason="Week 1 Thu: implement reciprocal_rank_fusion first")
    def test_agreeing_rankings_rank_shared_top_first(self) -> None:
        """A chunk ranked #1 in both dense and sparse should top the fused list."""
        dense = ["a", "b", "c"]
        sparse = ["a", "c", "b"]
        fused = reciprocal_rank_fusion(dense, sparse, k=60)
        assert fused[0][0] == "a"

    @pytest.mark.skip(reason="Week 1 Thu")
    def test_score_uses_k_smoothing(self) -> None:
        """RRF score for rank-0 in both lists == 2/(k+1) with k=60."""
        fused = dict(reciprocal_rank_fusion(["a"], ["a"], k=60))
        assert fused["a"] == pytest.approx(2 / 61)


class TestHybridRetriever:
    @pytest.mark.skip(reason="Week 1 Thu: needs a Qdrant fixture / stub")
    def test_retrieve_returns_scored_chunks(self) -> None: ...
