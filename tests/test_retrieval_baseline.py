"""Tests for the baseline runner's pure helpers (src/eval/retrieval_baseline.py).

MRR/label-dedup/summary math, no retriever, no index — CI-safe. The end-to-end
baseline itself runs out-of-band (needs the built index + CPU models), like the
retrieval integration tests.
"""

from __future__ import annotations

from src.eval.retrieval_baseline import ScenarioResult, _mrr, summarize


class _FakeChunk:
    def __init__(self, act: str, section_id: str) -> None:
        self.act = act
        self.section_id = section_id


class _FakeRetrieved:
    def __init__(self, act: str, section_id: str) -> None:
        self.chunk = _FakeChunk(act, section_id)


class TestMrr:
    def test_first_hit_rank_one(self) -> None:
        assert _mrr(["BNS::103", "BNS::63"], ["BNS::103"]) == 1.0

    def test_hit_at_rank_three(self) -> None:
        assert _mrr(["a", "b", "BNS::61"], ["BNS::61"]) == 1 / 3

    def test_no_hit(self) -> None:
        assert _mrr(["a", "b"], ["BNS::61"]) == 0.0


class TestLabels:
    def test_dedups_preserving_order(self) -> None:
        from src.eval.retrieval_baseline import _labels

        chunks = [
            _FakeRetrieved("BNS", "103"),
            _FakeRetrieved("BNS", "103"),  # dup collapses
            _FakeRetrieved("BSA", "3"),
        ]
        assert _labels(chunks) == ["BNS::103", "BSA::3"]


class TestSummarize:
    def test_means(self) -> None:
        results = [
            ScenarioResult("s1", "easy", 0.2, 1.0, 1.0, [], []),
            ScenarioResult("s2", "hard", 0.0, 0.0, 0.0, [], []),
        ]
        m = summarize(results)
        assert m["p_at_5"] == 0.1
        assert m["recall_at_5"] == 0.5
        assert m["mrr"] == 0.5

    def test_empty_is_safe(self) -> None:
        assert summarize([])["p_at_5"] == 0.0
