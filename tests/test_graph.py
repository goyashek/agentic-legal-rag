"""Tests for the graph wiring.

Layers:
  - routing functions: pure, branch on state, no key/index (the bulk here).
  - retrieve_node fan/dedupe: fake retriever + reranker, no models, no key.
  - fast-path e2e: keyless — a "BNS 103" query hits the deterministic fast path
    and ends without any LLM call, exercising the real StateGraph. Needs the
    built index (data/processed/sections.jsonl), so it skips when that's absent.
  - criminal-branch e2e: live Gemini (router + expander) + real retrieval; gated
    on both the key and the index.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from langgraph.graph import END

from src.agent.graph import (
    RETRIEVAL_LOOP_BUDGET,
    _dedupe_by_chunk_id,
    answer_query,
    build_graph,
    retrieve_node,
    route_after_fast_path,
    route_after_grader,
    route_after_ood_gate,
    route_after_router,
)
from src.agent.llm import has_api_key
from src.ingest.chunk_chonkie import LegalChunk
from src.retrieval.hybrid import RetrievedChunk

_INDEX = Path("data/processed/sections.jsonl")
_QDRANT = Path("data/processed/qdrant")
_have_index = _INDEX.exists()
_have_full_index = _INDEX.exists() and _QDRANT.exists() and Path("data/processed/bm25.pkl").exists()


def _rc(chunk_id: str, rrf: float) -> RetrievedChunk:
    section = chunk_id.split("::")[1]
    return RetrievedChunk(
        chunk=LegalChunk(chunk_id, "BNS", section, "heading", "text"), rrf_score=rrf
    )


class _FakeRetriever:
    """Returns a canned hit list keyed by query; records calls."""

    def __init__(self, per_query: dict[str, list[RetrievedChunk]]) -> None:
        self._per_query = per_query
        self.calls: list[str] = []

    def retrieve(self, query: str, *, top_k: int = 20) -> list[RetrievedChunk]:
        self.calls.append(query)
        return self._per_query.get(query, [])


class _PassThroughReranker:
    """Reranker stand-in: returns the first top_k unchanged; records the query."""

    def __init__(self) -> None:
        self.query: str | None = None

    def rerank(self, query, candidates, *, top_k=8):  # noqa: ANN001
        self.query = query
        return candidates[:top_k]


class TestRoutingFunctions:
    def test_fast_path_hit_goes_to_end(self) -> None:
        assert route_after_fast_path({"fast_path_hit": True}) == END

    def test_fast_path_miss_goes_to_router(self) -> None:
        assert route_after_fast_path({"fast_path_hit": False}) == "router"

    def test_fast_path_absent_key_defaults_to_router(self) -> None:
        assert route_after_fast_path({}) == "router"

    def test_criminal_route_goes_to_intent_expander(self) -> None:
        assert route_after_router({"route": "criminal"}) == "intent_expander"

    def test_terminal_routes_end(self) -> None:
        assert route_after_router({"route": "out_of_scope"}) == END
        assert route_after_router({"route": "needs_clarification"}) == END

    def test_ood_routes_to_not_in_corpus(self) -> None:
        assert route_after_ood_gate({"ood": True}) == "not_in_corpus"

    def test_in_corpus_goes_to_grader(self) -> None:
        assert route_after_ood_gate({"ood": False}) == "grader"

    def test_grade_pass_goes_to_generator(self) -> None:
        assert route_after_grader({"grade_pass": True}) == "generator"

    def test_grade_fail_within_budget_rewrites(self) -> None:
        assert route_after_grader({"grade_pass": False, "iteration": 0}) == "rewriter"
        assert (
            route_after_grader({"grade_pass": False, "iteration": RETRIEVAL_LOOP_BUDGET - 1})
            == "rewriter"
        )

    def test_grade_fail_budget_spent_goes_low_confidence(self) -> None:
        assert (
            route_after_grader({"grade_pass": False, "iteration": RETRIEVAL_LOOP_BUDGET})
            == "low_confidence"
        )


class TestRetrieveNode:
    """Fan over sub-queries + dedupe, using fakes (no models, no key)."""

    def test_dedupe_keeps_best_rrf(self) -> None:
        dupes = [_rc("BNS::103::0", 0.2), _rc("BNS::103::0", 0.9), _rc("BNS::63::0", 0.5)]
        out = {c.chunk.chunk_id: c.rrf_score for c in _dedupe_by_chunk_id(dupes)}
        assert out == {"BNS::103::0": 0.9, "BNS::63::0": 0.5}

    def test_fans_over_sub_queries_and_dedupes(self) -> None:
        retr = _FakeRetriever(
            {
                "house trespass": [_rc("BNS::329::0", 0.5), _rc("BNS::303::0", 0.3)],
                "theft": [_rc("BNS::303::0", 0.8)],  # 303 overlaps -> dedupe to best rrf
            }
        )
        rer = _PassThroughReranker()
        out = retrieve_node(
            {"query": "broke in and stole", "sub_queries": ["house trespass", "theft"]},
            retriever=retr,
            reranker=rer,
        )
        ids = {c.chunk.chunk_id for c in out["retrieved"]}
        assert ids == {"BNS::329::0", "BNS::303::0"}
        assert retr.calls == ["house trespass", "theft"]
        # reranks against the ORIGINAL query, not a sub-query
        assert rer.query == "broke in and stole"

    def test_falls_back_to_query_when_no_sub_queries(self) -> None:
        retr = _FakeRetriever({"punishment for murder": [_rc("BNS::103::0", 0.9)]})
        out = retrieve_node(
            {"query": "punishment for murder"}, retriever=retr, reranker=_PassThroughReranker()
        )
        assert retr.calls == ["punishment for murder"]
        assert out["retrieved"][0].chunk.section_id == "103"


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


@pytest.mark.live
@pytest.mark.skipif(not has_api_key(), reason="needs GEMINI_API_KEY for a live Gemini call")
class TestRouterEndToEnd:
    """Narrative query misses the fast path and flows through the live router."""

    def test_out_of_scope_query_ends_with_canned_answer(self) -> None:
        state = answer_query("what's the best pizza in Mumbai")
        assert state["fast_path_hit"] is False
        assert state["route"] == "out_of_scope"
        assert state["answer"].in_corpus is False


@pytest.mark.live
@pytest.mark.skipif(
    not (_have_full_index and has_api_key()),
    reason="needs both the built Qdrant index and GEMINI_API_KEY",
)
class TestCriminalBranchEndToEnd:
    """Full criminal path: router -> intent_expander -> retrieve -> ood_gate."""

    @pytest.fixture(autouse=True, scope="class")
    @staticmethod
    def _release_qdrant_lock():
        # answer_query caches a HybridRetriever holding the embedded-Qdrant file
        # lock; release it on teardown so test_retrieval can open its own client.
        yield
        from src.agent.graph import reset_retrieval_stack

        reset_retrieval_stack()

    def test_narrative_crime_retrieves_in_corpus(self) -> None:
        state = answer_query("someone broke into my house and stole my laptop")
        assert state["route"] == "criminal"
        assert len(state["sub_queries"]) >= 1
        assert len(state["retrieved"]) > 0
        assert state["ood"] is False  # a real crime narrative is in-corpus

    def test_in_corpus_query_is_not_flagged_ood(self) -> None:
        state = answer_query("what is the punishment for cheating")
        assert state["ood"] is False
        # retrieval should surface a BNS section for the reranked set
        assert any(c.chunk.act == "BNS" for c in state["retrieved"])

    def test_full_pipeline_produces_cited_answer(self) -> None:
        # the Thursday deliverable: query in -> generated, cited LegalAdvice out
        state = answer_query("what is the punishment for cheating")
        answer = state["answer"]
        assert answer is not None
        assert answer.query == "what is the punishment for cheating"
        assert answer.in_corpus is True
        assert answer.citations, "a graded-pass answer must carry citations"
        # Every cited section must be one that was actually retrieved. The corpus is
        # keyed at SECTION level but the generator legitimately cites subsections
        # ("318(2)"), so normalize to section level before the membership check —
        # exactly what Fri's deterministic citation validator will formalize (and
        # what fast_path already does on the query side).
        def _section(sid: str) -> str:
            return re.match(r"\d+[A-Z]?", sid).group(0) if re.match(r"\d+[A-Z]?", sid) else sid

        retrieved_ids = {(c.chunk.act, _section(c.chunk.section_id)) for c in state["retrieved"]}
        for cit in answer.citations:
            assert (cit.act, _section(cit.section_id)) in retrieved_ids
