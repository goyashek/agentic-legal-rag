"""Tests for the deterministic agent nodes.

These are the part I care most about, so they get the sharpest tests:
detect_exact_section (fires on "BNS 103", not on narratives) and is_out_of_domain
(threshold boundary both ways). Both are pure, no LLM, no index — CI-safe.

validate_citations (the correctness check) lands Week 2 Fri; its tests live here too
once that node exists.
"""

from __future__ import annotations

import pytest

from src.agent.llm import has_api_key
from src.agent.nodes.fast_path import detect_exact_section, lookup_section
from src.agent.nodes.intent_expander import (
    SubQueries,
    _dedupe,
    expand_intent,
    intent_expander_node,
)
from src.agent.nodes.ood_gate import is_out_of_domain
from src.agent.nodes.router import RouteDecision, classify, router_node
from src.ingest.chunk_chonkie import LegalChunk
from src.retrieval.hybrid import RetrievedChunk


class _FakeClient:
    """Stand-in for the instructor client: `.create(...)` returns a canned model.

    Records the messages it was called with so tests can assert the query made
    it into the prompt. Zero quota, no key needed.
    """

    def __init__(self, route: str) -> None:
        self._route = route
        self.calls: list[dict] = []

    def create(self, *, messages, response_model, **kwargs):  # noqa: ANN001
        self.calls.append({"messages": messages, "response_model": response_model, **kwargs})
        assert response_model is RouteDecision
        return RouteDecision(route=self._route)


class _FakeExpanderClient:
    """Returns canned sub-queries for the intent expander. Zero quota."""

    def __init__(self, subs: list[str]) -> None:
        self._subs = subs
        self.calls: list[dict] = []

    def create(self, *, messages, response_model, **kwargs):  # noqa: ANN001
        self.calls.append({"messages": messages, "response_model": response_model, **kwargs})
        assert response_model is SubQueries
        return SubQueries(sub_queries=self._subs)


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


class TestRouterUnit:
    """Node logic against a fake client — no key, no quota."""

    def test_classify_returns_route(self) -> None:
        assert classify("punishment for murder", client=_FakeClient("criminal")) == "criminal"

    def test_classify_puts_query_in_prompt(self) -> None:
        fake = _FakeClient("criminal")
        classify("someone stole my bike", client=fake)
        content = fake.calls[0]["messages"][0]["content"]
        assert "someone stole my bike" in content
        assert fake.calls[0]["temperature"] == 0

    def test_node_sets_route_and_trace(self) -> None:
        out = router_node({"query": "punishment for theft"}, client=_FakeClient("criminal"))
        assert out["route"] == "criminal"
        assert any("router: criminal" in n for n in out["trace_notes"])

    def test_criminal_route_has_no_canned_answer(self) -> None:
        # criminal continues down the pipeline; no terminal answer yet
        out = router_node({"query": "what is culpable homicide"}, client=_FakeClient("criminal"))
        assert "answer" not in out

    def test_out_of_scope_gets_canned_low_confidence_answer(self) -> None:
        out = router_node({"query": "how do I file taxes"}, client=_FakeClient("out_of_scope"))
        assert out["route"] == "out_of_scope"
        assert out["answer"].confidence == "low"
        assert out["answer"].in_corpus is False

    def test_needs_clarification_stays_in_corpus(self) -> None:
        out = router_node({"query": "my friend is in trouble"},
                          client=_FakeClient("needs_clarification"))
        assert out["answer"].confidence == "low"
        assert out["answer"].in_corpus is True


@pytest.mark.skipif(not has_api_key(), reason="needs GEMINI_API_KEY for a live Gemini call")
class TestRouterLive:
    """A few live Flash calls to confirm the prompt actually classifies right."""

    def test_clear_criminal(self) -> None:
        assert classify("what is the punishment for murder under BNS") == "criminal"

    def test_clear_out_of_scope(self) -> None:
        assert classify("what's a good recipe for butter chicken") == "out_of_scope"

    def test_narrative_is_criminal(self) -> None:
        assert classify("someone broke into my house and stole my laptop") == "criminal"


class TestIntentExpanderUnit:
    """Fan/dedupe/fallback logic against a fake client — no key, no quota."""

    def test_dedupe_drops_case_dupes_and_blanks(self) -> None:
        assert _dedupe(["theft", "Theft ", "", "  ", "robbery"]) == ["theft", "robbery"]

    def test_expand_returns_subqueries(self) -> None:
        fake = _FakeExpanderClient(["house trespass", "theft of movable property"])
        subs = expand_intent("someone broke into my house and stole my phone", client=fake)
        assert subs == ["house trespass", "theft of movable property"]

    def test_expand_caps_at_max(self) -> None:
        fake = _FakeExpanderClient([f"q{i}" for i in range(9)])
        subs = expand_intent("multi-offence narrative", max_sub_queries=5, client=fake)
        assert len(subs) == 5

    def test_expand_falls_back_to_query_when_empty(self) -> None:
        fake = _FakeExpanderClient(["", "   "])
        subs = expand_intent("punishment for murder", client=fake)
        assert subs == ["punishment for murder"]

    def test_node_sets_sub_queries_and_trace(self) -> None:
        fake = _FakeExpanderClient(["theft", "criminal trespass"])
        out = intent_expander_node({"query": "broke in and stole"}, client=fake)
        assert out["sub_queries"] == ["theft", "criminal trespass"]
        assert any("intent_expander: 2 sub-queries" in n for n in out["trace_notes"])


@pytest.mark.skipif(not has_api_key(), reason="needs GEMINI_API_KEY for a live Gemini call")
class TestIntentExpanderLive:
    def test_multi_offence_narrative_expands(self) -> None:
        subs = expand_intent("someone broke into my house at night and stole my laptop")
        # should surface at least two distinct issues (trespass + theft)
        assert len(subs) >= 2

    def test_simple_query_stays_focused(self) -> None:
        subs = expand_intent("what is the punishment for murder")
        assert 1 <= len(subs) <= 3
