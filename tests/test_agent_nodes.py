"""Tests for agent nodes.

The deterministic nodes are the part I care most about, so they get the sharpest tests:
detect_exact_section (fires on "BNS 103", not on narratives), validate_citations
(the correctness check), and is_out_of_domain (threshold boundary).
"""

from __future__ import annotations

import pytest

from src.agent.nodes.citation_validator import validate_citations
from src.agent.nodes.fast_path import detect_exact_section


class TestExactSectionDetection:
    @pytest.mark.skip(reason="Week 1 Fri: implement detect_exact_section first")
    def test_detects_explicit_bns_section(self) -> None:
        assert detect_exact_section("what is BNS Section 103") == ("BNS", "103")

    @pytest.mark.skip(reason="Week 1 Fri")
    def test_normalizes_ipc_reference(self) -> None:
        """'302 IPC' should resolve to its BNS equivalent (103) via the mapping."""
        assert detect_exact_section("explain section 302 IPC") == ("BNS", "103")

    @pytest.mark.skip(reason="Week 1 Fri")
    def test_ignores_narrative_query(self) -> None:
        """Narrative queries must not trigger the fast path, false positives kill precision."""
        assert detect_exact_section("someone stole my bike from outside my house") is None


class TestCitationValidator:
    @pytest.mark.skip(reason="Week 2 Fri: implement validate_citations first")
    def test_rejects_citation_not_in_retrieved_set(self) -> None:
        """the case I want to catch: answer cites BNS 307 but only 306 was retrieved, so invalid."""
        ...

    @pytest.mark.skip(reason="Week 2 Fri")
    def test_passes_when_all_citations_retrieved(self) -> None: ...
