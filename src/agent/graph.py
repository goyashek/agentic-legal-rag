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

Build status: Mon wires fast_path → router (+ terminal routes → END). The criminal
branch currently ends at the router; intent_expander/retrieve/... get spliced in as
each node lands (Tue onward). Routing decisions are pure functions so they're
unit-testable without a key or the index.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agent.state import AgentState

RETRIEVAL_LOOP_BUDGET = 2


# --- routing decisions (pure; branch on state, no side effects) --------------


def route_after_fast_path(state: AgentState) -> str:
    """fast_path hit -> straight to END; miss -> the LLM router."""
    return END if state.get("fast_path_hit") else "router"


def route_after_router(state: AgentState) -> str:
    """criminal -> continue the pipeline; terminal routes -> END (canned answer set).

    Returns the next node for the criminal path. Until intent_expander is wired
    (Tue) that target doesn't exist yet, so criminal also ends here — the router
    has set `route` and the pipeline picks up from there next.
    """
    if state.get("route") == "criminal":
        return END  # TODO(Tue): -> "intent_expander"
    return END


# --- orchestration node (lives here, not nodes/, since it drives the retrieval
#     layer rather than making an agent decision) -------------------------------


def retrieve_node(state: AgentState) -> AgentState:
    """Fan HybridRetriever over sub_queries, dedupe, rerank down to ~8. Sets `retrieved`."""
    raise NotImplementedError("todo(Tue): fan retrieve over sub_queries + rerank")


# --- graph assembly ----------------------------------------------------------


def build_graph():
    """Construct and compile the StateGraph. Returns a graph with `.invoke` / `.ainvoke`.

    Incremental: only the nodes that exist are wired. Each new node this week
    adds an `add_node` + re-points a conditional edge; the routing functions
    above already name the intended targets.
    """
    from src.agent.nodes.fast_path import fast_path_node
    from src.agent.nodes.router import router_node

    builder = StateGraph(AgentState)
    builder.add_node("fast_path", fast_path_node)
    builder.add_node("router", router_node)

    builder.add_edge(START, "fast_path")
    builder.add_conditional_edges("fast_path", route_after_fast_path, ["router", END])
    builder.add_conditional_edges("router", route_after_router, [END])

    return builder.compile()


# --- entry point -------------------------------------------------------------

_GRAPH = None


def answer_query(query: str) -> AgentState:
    """Build the graph (cached) and run one query. Returns the final state."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH.invoke({"query": query, "trace_notes": [], "iteration": 0})
