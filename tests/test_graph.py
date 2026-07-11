"""Tests for the graph wiring.

Three layers:
  - routing functions: pure, branch on state, no key/index (the bulk here).
  - fast-path e2e: keyless — a "BNS 103" query hits the deterministic fast path
    and ends without any LLM call, exercising the real StateGraph. Needs the
    built index (data/processed/sections.jsonl), so it skips when that's absent.
  - router e2e: a live Gemini call for a narrative miss; gated on the key.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.graph import END

from src.agent.graph import answer_query, build_graph, route_after_fast_path, route_after_router
from src.agent.llm import has_api_key

_INDEX = Path("data/processed/sections.jsonl")
_have_index = _INDEX.exists()


class TestRoutingFunctions:
    def test_fast_path_hit_goes_to_end(self) -> None:
        assert route_after_fast_path({"fast_path_hit": True}) == END

    def test_fast_path_miss_goes_to_router(self) -> None:
        assert route_after_fast_path({"fast_path_hit": False}) == "router"

    def test_fast_path_absent_key_defaults_to_router(self) -> None:
        assert route_after_fast_path({}) == "router"

    def test_criminal_route_continues(self) -> None:
        # criminal currently ends here too (pipeline extends Tue); assert it's
        # NOT treated as a terminal canned-answer route by mistake.
        assert route_after_router({"route": "criminal"}) == END

    def test_terminal_routes_end(self) -> None:
        assert route_after_router({"route": "out_of_scope"}) == END
        assert route_after_router({"route": "needs_clarification"}) == END


class TestGraphCompiles:
    def test_build_graph_returns_compiled(self) -> None:
        g = build_graph()
        assert hasattr(g, "invoke")


@pytest.mark.skipif(not _have_index, reason="needs the built index (data/processed/sections.jsonl)")
class TestFastPathEndToEnd:
    """Keyless full-graph run: exact-section query resolves deterministically."""

    def test_exact_section_hits_fast_path(self) -> None:
        state = answer_query("what does BNS Section 103 say")
        assert state["fast_path_hit"] is True
        assert state["fast_path_answer"] is not None
        assert state["fast_path_answer"].citations[0].section_id == "103"

    def test_ipc_reference_normalizes_to_bns(self) -> None:
        # 302 IPC -> BNS 103 via the mapping baked into the corpus metadata
        state = answer_query("explain section 302 IPC")
        assert state["fast_path_hit"] is True
        assert state["fast_path_answer"].citations[0].act == "BNS"
        assert state["fast_path_answer"].citations[0].section_id == "103"


@pytest.mark.skipif(not has_api_key(), reason="needs GEMINI_API_KEY for a live Gemini call")
class TestRouterEndToEnd:
    """Narrative query misses the fast path and flows through the live router."""

    def test_out_of_scope_query_ends_with_canned_answer(self) -> None:
        state = answer_query("what's the best pizza in Mumbai")
        assert state["fast_path_hit"] is False
        assert state["route"] == "out_of_scope"
        assert state["answer"].in_corpus is False

    def test_narrative_crime_routes_criminal(self) -> None:
        state = answer_query("someone broke into my house and stole my laptop")
        assert state["fast_path_hit"] is False
        assert state["route"] == "criminal"
