"""Wires the nodes into the self-correcting flow.

Flow I settled on (target — built incrementally across Week 2):

    fast_path (hit) ......................................► END
        │ miss
        ▼
    router (out_of_scope / needs_clarification) ..► END (canned answer)
        │ criminal
        ▼
    intent_expander → retrieve → ood_gate (ood) ..► END ("not in corpus")
        │ in-domain
        ▼
    grader (< 3 relevant) → rewriter (within budget) → retrieve
        │ >= 3 relevant             (budget hit) → END (confidence=low)
        ▼
    generator → citation_validator (invalid) → rewriter (loop)
        │ valid
        ▼
    checker (unfaithful) → rewriter (loop)
        │ faithful
        ▼
       END

The loop budget (RETRIEVAL_LOOP_BUDGET = 2) lives in the conditional edges out of
grader / citation_validator / checker. All three route back through the rewriter,
which bumps `iteration`. Once it goes past the budget the graph ends with
confidence="low" instead of spinning forever.

Build status: Mon wired fast_path → router. Tue splices in the criminal branch:
router(criminal) → intent_expander → retrieve → ood_gate → (ood → not_in_corpus →
END | in-corpus → END for now; grader lands Wed). Routing decisions are pure
functions, unit-testable without a key or the index; retrieve_node takes an
injectable retriever/reranker so its fan/dedupe logic tests with fakes.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agent.state import AgentState
from src.retrieval.hybrid import RetrievedChunk

RETRIEVAL_LOOP_BUDGET = 2

# Retrieval knobs (mirror the baseline: 20 fused candidates -> rerank to 8).
RETRIEVE_K = 20
RERANK_K = 8
# The real Qdrant collection is "legal" (see src/retrieval/index.py + the on-disk
# data/processed/qdrant/collection/legal). NOTE: .env.example still says
# QDRANT_COLLECTION=bns_sections — that default is stale; the index build names it
# "legal". Kept here so the graph doesn't silently query an empty collection.
QDRANT_COLLECTION = "legal"


# --- routing decisions (pure; branch on state, no side effects) --------------


def route_after_fast_path(state: AgentState) -> str:
    """fast_path hit -> straight to END; miss -> the LLM router."""
    return END if state.get("fast_path_hit") else "router"


def route_after_router(state: AgentState) -> str:
    """criminal -> intent expansion + retrieval; terminal routes -> END (canned answer)."""
    if state.get("route") == "criminal":
        return "intent_expander"
    return END


def route_after_ood_gate(state: AgentState) -> str:
    """out-of-domain -> canned 'not in corpus' answer; in-corpus -> continue.

    The grader lands Wed; until then the in-corpus path ends here (retrieval is
    proven, generation isn't wired yet).
    """
    if state.get("ood"):
        return "not_in_corpus"
    return END  # TODO(Wed): -> "grader"


# --- orchestration node (lives here, not nodes/, since it drives the retrieval
#     layer rather than making an agent decision) -------------------------------

# The retriever + reranker load transformer models (~expensive), so build them
# once and reuse across queries. Cached like fast_path._resolver so importing the
# module stays cheap and tests can inject fakes instead.
_RETRIEVER = None
_RERANKER = None


def _retrieval_stack():
    """Lazily build (retriever, reranker); cached process-wide."""
    global _RETRIEVER, _RERANKER
    if _RETRIEVER is None:
        from src.retrieval.hybrid import HybridRetriever
        from src.retrieval.rerank import Reranker

        _RETRIEVER = HybridRetriever(
            collection=QDRANT_COLLECTION, bm25_path="data/processed/bm25.pkl"
        )
        _RERANKER = Reranker()
    return _RETRIEVER, _RERANKER


def reset_retrieval_stack() -> None:
    """Close the cached retriever's Qdrant client and drop the cache.

    In EMBEDDED mode Qdrant takes a file lock on data/processed/qdrant and allows
    only one client per path, so the long-lived cache would otherwise block any
    other client in the same process (e.g. the retrieval integration tests). Call
    this to release the lock — tests use it on teardown; a server never needs to.
    """
    global _RETRIEVER, _RERANKER
    if _RETRIEVER is not None:
        client = getattr(_RETRIEVER, "client", None)
        if client is not None:
            client.close()
    _RETRIEVER = None
    _RERANKER = None


def _dedupe_by_chunk_id(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Collapse duplicates across sub-queries, keeping the best RRF score per chunk."""
    best: dict[str, RetrievedChunk] = {}
    for c in chunks:
        cid = c.chunk.chunk_id
        if cid not in best or c.rrf_score > best[cid].rrf_score:
            best[cid] = c
    return list(best.values())


def retrieve_node(state: AgentState, *, retriever=None, reranker=None) -> AgentState:
    """Fan the retriever over sub_queries, dedupe, rerank down to ~8. Sets `retrieved`.

    retriever/reranker are injectable so the fan+dedupe logic tests with fakes; by
    default the cached real stack is used. Reranks against the ORIGINAL query (the
    user's actual intent), not the sub-queries, so the final ordering reflects what
    was asked rather than one decomposed facet.
    """
    if retriever is None or reranker is None:
        retriever, reranker = _retrieval_stack()

    sub_queries = state.get("sub_queries") or [state["query"]]
    pooled: list[RetrievedChunk] = []
    for sq in sub_queries:
        pooled.extend(retriever.retrieve(sq, top_k=RETRIEVE_K))

    deduped = _dedupe_by_chunk_id(pooled)
    reranked = reranker.rerank(state["query"], deduped, top_k=RERANK_K)

    notes = state.get("trace_notes", [])
    return {
        "retrieved": reranked,
        "trace_notes": [
            *notes,
            f"retrieve: {len(sub_queries)} sub-queries -> {len(pooled)} hits "
            f"-> {len(deduped)} unique -> {len(reranked)} reranked",
        ],
    }


def not_in_corpus_node(state: AgentState) -> AgentState:
    """Terminal for the OOD case: a low-confidence 'not in the statutes' answer."""
    from src.models.schemas import LegalAdvice

    notes = state.get("trace_notes", [])
    answer = LegalAdvice(
        query=state["query"],
        answer=(
            "I couldn't find this in the BNS, BNSS, or BSA. It may fall outside the "
            "criminal statutes I cover, or be phrased in a way I can't match to a section."
        ),
        confidence="low",
        in_corpus=False,
    )
    return {"answer": answer, "trace_notes": [*notes, "not_in_corpus: OOD terminal"]}


# --- graph assembly ----------------------------------------------------------


def build_graph():
    """Construct and compile the StateGraph. Returns a graph with `.invoke` / `.ainvoke`.

    Incremental: only the nodes that exist are wired. Each new node this week
    adds an `add_node` + re-points a conditional edge; the routing functions
    above already name the intended targets.
    """
    from src.agent.nodes.fast_path import fast_path_node
    from src.agent.nodes.intent_expander import intent_expander_node
    from src.agent.nodes.ood_gate import ood_gate_node
    from src.agent.nodes.router import router_node

    builder = StateGraph(AgentState)
    builder.add_node("fast_path", fast_path_node)
    builder.add_node("router", router_node)
    builder.add_node("intent_expander", intent_expander_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("ood_gate", ood_gate_node)
    builder.add_node("not_in_corpus", not_in_corpus_node)

    builder.add_edge(START, "fast_path")
    builder.add_conditional_edges("fast_path", route_after_fast_path, ["router", END])
    builder.add_conditional_edges("router", route_after_router, ["intent_expander", END])
    builder.add_edge("intent_expander", "retrieve")
    builder.add_edge("retrieve", "ood_gate")
    builder.add_conditional_edges("ood_gate", route_after_ood_gate, ["not_in_corpus", END])
    builder.add_edge("not_in_corpus", END)

    return builder.compile()


# --- entry point -------------------------------------------------------------

_GRAPH = None


def answer_query(query: str) -> AgentState:
    """Build the graph (cached) and run one query. Returns the final state."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH.invoke({"query": query, "trace_notes": [], "iteration": 0})
