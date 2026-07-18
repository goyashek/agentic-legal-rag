"""Tests for the graph wiring.

Layers:
  - routing functions: pure, branch on state, no key/index (the bulk here).
  - retrieve_node fan/dedupe: fake retriever + reranker, no models, no key.
  - fast-path e2e: keyless — a "BNS 103" query hits the deterministic fast path
    and ends without any LLM call, exercising the real StateGraph. Needs the
    built index (data/processed/sections.jsonl), so it skips when that's absent.
  - criminal-branch e2e: live DeepSeek (router + expander) + real retrieval; gated
    on both the key and the index.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.graph import END

from src.agent.graph import (
    RETRIEVAL_LOOP_BUDGET,
    _dedupe_by_chunk_id,
    answer_query,
    build_graph,
    retrieve_node,
    route_after_checker,
    route_after_checker_once,
    route_after_citation_to_checker_once,
    route_after_citation_validator,
    route_after_citation_validator_once,
    route_after_fast_path,
    route_after_grader,
    route_after_grader_once,
    route_after_ood_gate,
    route_after_ood_gate_to_generator,
    route_after_router,
    route_after_router_to_retrieve,
)
from src.agent.llm import has_api_key
from src.agent.nodes.citation_validator import validate_citations
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

    def retrieve(
        self, query: str, *, top_k: int = 20, mode: str = "hybrid"
    ) -> list[RetrievedChunk]:
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

    def test_production_in_corpus_goes_to_generator(self) -> None:
        assert route_after_ood_gate_to_generator({"ood": False}) == "generator"
        assert route_after_ood_gate_to_generator({"ood": True}) == "not_in_corpus"

    def test_production_router_skips_expansion(self) -> None:
        assert route_after_router_to_retrieve({"route": "criminal"}) == "retrieve"
        assert route_after_router_to_retrieve({"route": "out_of_scope"}) == END

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

    def test_valid_citations_go_to_checker(self) -> None:
        assert route_after_citation_validator({"citation_valid": True}) == "checker"

    def test_invalid_citations_rewrite_within_budget(self) -> None:
        assert (
            route_after_citation_validator({"citation_valid": False, "iteration": 0}) == "rewriter"
        )

    def test_invalid_citations_budget_spent_low_confidence(self) -> None:
        assert (
            route_after_citation_validator(
                {"citation_valid": False, "iteration": RETRIEVAL_LOOP_BUDGET}
            )
            == "low_confidence"
        )

    def test_faithful_goes_to_end(self) -> None:
        assert route_after_checker({"faithful": True}) == END

    def test_unfaithful_rewrites_within_budget(self) -> None:
        assert route_after_checker({"faithful": False, "iteration": 0}) == "rewriter"

    def test_unfaithful_budget_spent_low_confidence(self) -> None:
        assert (
            route_after_checker({"faithful": False, "iteration": RETRIEVAL_LOOP_BUDGET})
            == "low_confidence"
        )

    def test_ablation_routes_do_not_rewrite(self) -> None:
        assert route_after_grader_once({"grade_pass": True}) == "generator"
        assert route_after_grader_once({"grade_pass": False}) == "low_confidence"
        assert route_after_citation_validator_once({"citation_valid": True}) == END
        assert route_after_citation_validator_once({"citation_valid": False}) == "low_confidence"
        assert route_after_citation_to_checker_once({"citation_valid": True}) == "checker"
        assert route_after_citation_to_checker_once({"citation_valid": False}) == "low_confidence"
        assert route_after_checker_once({"faithful": True}) == END
        assert route_after_checker_once({"faithful": False}) == "low_confidence"


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

    def test_dense_mode_can_skip_reranking(self) -> None:
        retr = _FakeRetriever(
            {"punishment for murder": [_rc("BNS::103::0", 0.9), _rc("BNS::62::0", 0.2)]}
        )
        out = retrieve_node(
            {"query": "punishment for murder"},
            retriever=retr,
            mode="dense",
            use_reranker=False,
        )
        assert [c.chunk.section_id for c in out["retrieved"]] == ["103", "62"]
        assert "dense" in out["trace_notes"][-1]
        assert "rerank" not in out["trace_notes"][-1]

    def test_dense_mode_keeps_twelve_chunks_for_generation(self) -> None:
        chunks = [_rc(f"BNS::{section}::0", 1.0 - section / 1000) for section in range(100, 113)]
        retr = _FakeRetriever({"query": chunks})
        out = retrieve_node({"query": "query"}, retriever=retr, mode="dense", use_reranker=False)
        assert [c.chunk.section_id for c in out["retrieved"]] == [str(s) for s in range(100, 112)]


class TestGraphCompiles:
    def test_build_graph_returns_compiled(self) -> None:
        g = build_graph()
        assert hasattr(g, "invoke")

    @pytest.mark.parametrize(
        ("pipeline", "expected_nodes", "absent_nodes"),
        [
            (
                "production",
                {"fast_path", "router", "retrieve", "ood_gate", "generator", "citation_validator"},
                {"intent_expander", "grader", "checker", "rewriter"},
            ),
            ("baseline", {"retrieve", "generator", "citation_validator"}, {"grader", "checker"}),
            ("grader", {"retrieve", "grader", "generator", "citation_validator"}, {"checker"}),
            (
                "checker",
                {"retrieve", "grader", "generator", "citation_validator", "checker"},
                set(),
            ),
        ],
    )
    def test_ablation_graph_has_only_the_requested_stages(
        self, pipeline, expected_nodes, absent_nodes
    ) -> None:
        nodes = set(build_graph(pipeline=pipeline).get_graph().nodes)
        assert expected_nodes <= nodes
        assert not (absent_nodes & nodes)


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
@pytest.mark.skipif(not has_api_key(), reason="needs an easy-tier API key")
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
    reason="needs the built Qdrant index plus easy and hard LLM access",
)
class TestCriminalBranchEndToEnd:
    """Production criminal path: router -> retrieve -> ood gate."""

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
        assert len(state["retrieved"]) > 0
        assert state["ood"] is False  # a real crime narrative is in-corpus

    def test_in_corpus_query_is_not_flagged_ood(self) -> None:
        state = answer_query("what is the punishment for cheating")
        assert state["ood"] is False
        # retrieval should surface a BNS section for the reranked set
        assert any(c.chunk.act == "BNS" for c in state["retrieved"])

    def test_full_pipeline_honours_the_truthful_contract(self) -> None:
        # The production path runs through generator -> citation_validator. It has
        # two valid outcomes:
        #   (a) a cited answer whose every citation was actually retrieved, or
        #   (b) a low-confidence decline (loop budget spent) with no citations.
        # What must NEVER happen: a confident answer with absent/unverified
        # citations. That's the invariant this asserts.
        state = answer_query("what is the punishment for cheating")
        answer = state["answer"]
        assert answer is not None
        assert answer.query == "what is the punishment for cheating"
        assert answer.in_corpus is True

        if not answer.citations:
            # A citation-free answer is only reachable via low_confidence (the
            # validator marks a citation-less answer invalid, so it can't hit END).
            assert answer.confidence == "low"
        else:
            # Any answer that KEPT its citations reached END through the citation
            # validator, so its citations are — by the validator's own definition —
            # all present in the retrieved set. Assert that with the validator
            # itself rather than re-deriving the (act-normalizing) membership rule,
            # so the test can't drift from the code it's checking.
            valid, invalid = validate_citations(answer, state["retrieved"])
            assert valid, f"final answer carries unverified citations: {invalid}"
