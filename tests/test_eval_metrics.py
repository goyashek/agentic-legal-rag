"""Tests for the section-precision metrics (src/eval/section_precision.py).

All pure set/list ops over section-id labels, no LLM, no index. These pin the edge
cases that bite when computing a baseline: dedup in P@k, the < k retrieved case, and
the vacuous-truth conventions (empty relevant -> recall 1.0, empty cited -> accuracy
1.0).
"""

from __future__ import annotations

import pytest

from src.eval.section_precision import (
    citation_accuracy,
    section_precision_at_k,
    section_recall,
)


class TestPrecisionAtK:
    def test_all_relevant(self) -> None:
        assert section_precision_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0

    def test_half_relevant(self) -> None:
        assert section_precision_at_k(["a", "x", "b", "y"], ["a", "b"], k=4) == 0.5

    def test_only_counts_top_k(self) -> None:
        # relevant 'c' sits at rank 3, outside k=2
        assert section_precision_at_k(["a", "b", "c"], ["a", "c"], k=2) == 0.5

    def test_dedup_does_not_inflate(self) -> None:
        # duplicate 'a' collapses; distinct slots = {a} -> 1/1
        assert section_precision_at_k(["a", "a", "a"], ["a"], k=3) == 1.0

    def test_fewer_than_k_retrieved(self) -> None:
        # only 2 distinct retrieved, both relevant -> 1.0, not 2/5
        assert section_precision_at_k(["a", "b"], ["a", "b"], k=5) == 1.0

    def test_nothing_retrieved(self) -> None:
        assert section_precision_at_k([], ["a"], k=5) == 0.0

    def test_none_relevant(self) -> None:
        assert section_precision_at_k(["a", "b"], ["c"], k=2) == 0.0

    def test_invalid_k(self) -> None:
        with pytest.raises(ValueError):
            section_precision_at_k(["a"], ["a"], k=0)


class TestRecall:
    def test_all_found(self) -> None:
        assert section_recall(["a", "b", "c"], ["a", "b"]) == 1.0

    def test_partial(self) -> None:
        assert section_recall(["a", "x"], ["a", "b"]) == 0.5

    def test_missing_all(self) -> None:
        assert section_recall(["x", "y"], ["a", "b"]) == 0.0

    def test_empty_relevant_is_one(self) -> None:
        assert section_recall(["a"], []) == 1.0

    def test_order_and_dupes_irrelevant(self) -> None:
        assert section_recall(["b", "b", "a"], ["a", "b"]) == 1.0


class TestCitationAccuracy:
    def test_all_correct(self) -> None:
        assert citation_accuracy(["a", "b"], ["a", "b", "c"]) == 1.0

    def test_one_wrong(self) -> None:
        assert citation_accuracy(["a", "z"], ["a", "b"]) == 0.5

    def test_empty_cited_is_vacuously_one(self) -> None:
        assert citation_accuracy([], ["a"]) == 1.0

    def test_all_wrong(self) -> None:
        assert citation_accuracy(["x", "y"], ["a"]) == 0.0
