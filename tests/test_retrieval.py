"""Tests for the retrieval layer.

Two tiers, same pattern as the ingest tests:
  - Pure-function tests (RRF math, tokenizer, deterministic point ids) run always,
    no models, no index, CI-safe. RRF is the deterministic core so it gets the
    hardest tests.
  - Integration tests build against the real Qdrant + BM25 indices and the bge
    models; they skip cleanly when the indices aren't present (they're git-ignored,
    regenerated from source).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.index import point_id_for, rebuild_local_index, tokenize

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
QDRANT_DIR = PROCESSED / "qdrant"
BM25_PKL = PROCESSED / "bm25.pkl"


class TestReciprocalRankFusion:
    def test_agreeing_rankings_rank_shared_top_first(self) -> None:
        """A chunk ranked #1 in both dense and sparse should top the fused list."""
        dense = ["a", "b", "c"]
        sparse = ["a", "c", "b"]
        fused = reciprocal_rank_fusion(dense, sparse, k=60)
        assert fused[0][0] == "a"

    def test_score_uses_k_smoothing(self) -> None:
        """RRF score for rank-0 in both lists == 2/k with k=60 (rank is 0-based)."""
        fused = dict(reciprocal_rank_fusion(["a"], ["a"], k=60))
        assert fused["a"] == pytest.approx(2 / 60)

    def test_chunk_in_one_list_only(self) -> None:
        fused = dict(reciprocal_rank_fusion(["a", "b"], ["c"], k=60))
        assert fused["b"] == pytest.approx(1 / 61)
        assert fused["c"] == pytest.approx(1 / 60)

    def test_sorted_descending_by_score(self) -> None:
        fused = reciprocal_rank_fusion(["a", "b", "c"], ["a", "b", "c"], k=60)
        scores = [s for _, s in fused]
        assert scores == sorted(scores, reverse=True)

    def test_ties_broken_deterministically_by_id(self) -> None:
        """Two chunks with equal score come back in stable (id-sorted) order."""
        fused = reciprocal_rank_fusion(["x"], ["y"], k=60)  # both score 1/60
        assert [cid for cid, _ in fused] == ["x", "y"]

    def test_empty_inputs(self) -> None:
        assert reciprocal_rank_fusion([], [], k=60) == []

    def test_k_changes_scores(self) -> None:
        small_k = dict(reciprocal_rank_fusion(["a"], [], k=10))
        big_k = dict(reciprocal_rank_fusion(["a"], [], k=1000))
        assert small_k["a"] > big_k["a"]


class TestTokenize:
    def test_lowercases_and_splits_alnum(self) -> None:
        assert tokenize("BNS Section 103: Murder!") == ["bns", "section", "103", "murder"]

    def test_drops_punctuation_and_dashes(self) -> None:
        assert tokenize("non-bailable, cognizable.") == ["non", "bailable", "cognizable"]

    def test_empty(self) -> None:
        assert tokenize("   ") == []


class TestPointId:
    def test_deterministic(self) -> None:
        assert point_id_for("BNS::103::0") == point_id_for("BNS::103::0")

    def test_distinct_ids_differ(self) -> None:
        assert point_id_for("BNS::103::0") != point_id_for("BNS::103::1")


def test_rebuild_requires_all_source_pdfs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="bns.pdf"):
        rebuild_local_index(raw_dir=tmp_path)


@pytest.mark.skipif(
    not (QDRANT_DIR.exists() and BM25_PKL.exists()),
    reason="retrieval indices not built; run src/retrieval/index.py",
)
class TestHybridRetrieverIntegration:
    """End-to-end retrieval against the real indices + bge models.

    These are the real proof the plumbing works: a plain-language query about a crime
    should surface the right BNS section near the top.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def retriever(cls):
        from src.retrieval.hybrid import HybridRetriever

        r = HybridRetriever(collection="legal", bm25_path=str(BM25_PKL))
        yield r
        r.client.close()

    def test_retrieve_returns_scored_chunks(self, retriever) -> None:
        results = retriever.retrieve("someone was murdered", top_k=10)
        assert results
        top = results[0]
        assert top.rrf_score > 0
        # scores are attributed so the ablation has signal to work with
        assert top.dense_rank is not None or top.sparse_rank is not None

    def test_murder_query_surfaces_section_103(self, retriever) -> None:
        results = retriever.retrieve("what is the punishment for murder", top_k=10)
        section_ids = {r.chunk.section_id for r in results if r.chunk.act == "BNS"}
        assert "103" in section_ids

    def test_theft_query_surfaces_theft_sections(self, retriever) -> None:
        results = retriever.retrieve("my bike was stolen", top_k=10)
        headings = " ".join(r.chunk.heading.lower() for r in results)
        assert "theft" in headings

    @pytest.mark.parametrize(
        ("query", "expected_sections"),
        [
            (
                "he beat me so badly with a rod that I permanently lost sight in one eye",
                {"117"},
            ),
            (
                "a witness deliberately lied under oath during my court case",
                {"227", "229"},
            ),
            (
                "he lunged to snatch a woman's chain but was caught before he could grab it",
                {"62", "304"},
            ),
        ],
    )
    def test_audited_dense_cases_reach_the_context_window(
        self, retriever, query, expected_sections
    ) -> None:
        results = retriever.retrieve(query, top_k=12, mode="dense")
        found = {r.chunk.section_id for r in results if r.chunk.act == "BNS"}
        assert expected_sections <= found

    def test_results_capped_at_top_k(self, retriever) -> None:
        assert len(retriever.retrieve("criminal breach of trust", top_k=5)) <= 5

    @pytest.mark.parametrize("mode", ["dense", "sparse"])
    def test_single_signal_modes_return_ranked_chunks(self, retriever, mode: str) -> None:
        results = retriever.retrieve("what is the punishment for murder", top_k=5, mode=mode)

        assert 1 <= len(results) <= 5
        if mode == "dense":
            assert all(r.dense_rank is not None and r.sparse_rank is None for r in results)
        else:
            assert all(r.sparse_rank is not None and r.dense_rank is None for r in results)


@pytest.mark.skipif(
    not (QDRANT_DIR.exists() and BM25_PKL.exists()),
    reason="retrieval indices not built",
)
class TestRerankerIntegration:
    def test_rerank_reorders_and_caps(self) -> None:
        from src.retrieval.hybrid import HybridRetriever
        from src.retrieval.rerank import Reranker

        retriever = HybridRetriever(collection="legal", bm25_path=str(BM25_PKL))
        try:
            candidates = retriever.retrieve("punishment for murder", top_k=20)
            reranked = Reranker().rerank("punishment for murder", candidates, top_k=8)

            assert len(reranked) <= 8
            assert all(r.rerank_score is not None for r in reranked)
            # sorted by rerank score desc
            scores = [r.rerank_score for r in reranked]
            assert scores == sorted(scores, reverse=True)
        finally:
            retriever.client.close()

    def test_rerank_empty_candidates(self) -> None:
        from src.retrieval.rerank import Reranker

        assert Reranker().rerank("anything", []) == []
