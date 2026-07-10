"""Tests for the deterministic agent nodes.

These are the part I care most about, so they get the sharpest tests:
detect_exact_section (fires on "BNS 103", not on narratives) and is_out_of_domain
(threshold boundary both ways). Both are pure, no LLM, no index — CI-safe.

validate_citations (the correctness check) lands Week 2 Fri; its tests live here too
once that node exists.
"""

from __future__ import annotations

from src.agent.nodes.fast_path import detect_exact_section, lookup_section
from src.agent.nodes.ood_gate import is_out_of_domain
from src.ingest.chunk_chonkie import LegalChunk
from src.retrieval.hybrid import RetrievedChunk

# A tiny IPC->BNS map so the IPC-normalization tests don't need the real PDF.
IPC_MAP = {"302": "103", "379": "303", "420": "318"}


def _rc(dense_score: float | None) -> RetrievedChunk:
    chunk = LegalChunk("BNS::103::0", "BNS", "103", "Punishment for murder", "text")
    return RetrievedChunk(chunk=chunk, rrf_score=0.5, dense_score=dense_score)


class TestExactSectionDetection:
    def test_detects_explicit_bns_section(self) -> None:
        assert detect_exact_section("what is BNS Section 103") == ("BNS", "103")

    def test_detects_act_after_number(self) -> None:
        assert detect_exact_section("explain 63 BNS") == ("BNS", "63")

    def test_detects_lettered_section(self) -> None:
        assert detect_exact_section("BNS 111A") == ("BNS", "111A")

    def test_drops_subsection_to_section_level(self) -> None:
        assert detect_exact_section("BNS 103(2) please") == ("BNS", "103")

    def test_normalizes_ipc_reference(self) -> None:
        """'302 IPC' should resolve to its BNS equivalent (103) via the mapping."""
        assert detect_exact_section("explain section 302 IPC", ipc_bns_mapping=IPC_MAP) == (
            "BNS",
            "103",
        )

    def test_ipc_without_mapping_is_none(self) -> None:
        """No mapping entry -> don't guess, fall through to the pipeline."""
        assert detect_exact_section("section 302 IPC") is None

    def test_ignores_narrative_query(self) -> None:
        """Narrative queries must not trigger the fast path; false positives kill precision."""
        assert detect_exact_section("someone stole my bike from outside my house") is None

    def test_bare_number_does_not_fire(self) -> None:
        """A number with no act code is ambiguous -> no fast path."""
        assert detect_exact_section("what does section 103 say") is None

    def test_bnss_and_bsa(self) -> None:
        assert detect_exact_section("BNSS 173") == ("BNSS", "173")
        assert detect_exact_section("section 63 of BSA") == ("BSA", "63")


class TestLookupSection:
    def test_finds_matching_chunk(self) -> None:
        chunks = [
            LegalChunk("BNS::103::0", "BNS", "103", "Murder", "t"),
            LegalChunk("BNS::63::0", "BNS", "63", "Rape", "t"),
        ]
        assert lookup_section("BNS", "63", chunks).section_id == "63"

    def test_missing_returns_none(self) -> None:
        assert lookup_section("BNS", "999", []) is None


class TestOutOfDomainGate:
    def test_empty_retrieval_is_ood(self) -> None:
        assert is_out_of_domain([]) is True

    def test_no_dense_scores_is_ood(self) -> None:
        assert is_out_of_domain([_rc(None)]) is True

    def test_close_chunk_is_in_domain(self) -> None:
        # similarity 0.9 -> distance 0.1 < 0.75
        assert is_out_of_domain([_rc(0.9)]) is False

    def test_far_chunk_is_ood(self) -> None:
        # similarity 0.1 -> distance 0.9 > 0.75
        assert is_out_of_domain([_rc(0.1)]) is True

    def test_threshold_boundary_is_in_domain(self) -> None:
        # similarity 0.25 -> distance exactly 0.75, strict > means in-domain
        assert is_out_of_domain([_rc(0.25)]) is False

    def test_uses_best_of_several(self) -> None:
        assert is_out_of_domain([_rc(0.1), _rc(0.8), _rc(0.05)]) is False
