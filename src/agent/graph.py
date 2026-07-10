"""Wires the nodes into the self-correcting flow.

Flow I settled on:

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
"""

from __future__ import annotations

from src.agent.state import AgentState

RETRIEVAL_LOOP_BUDGET = 2


def retrieve_node(state: AgentState) -> AgentState:
    """Fan HybridRetriever over sub_queries, dedupe, rerank down to ~8. Sets `retrieved`.

    Kept in the graph module, not nodes/, since it's orchestration over the retrieval
    layer rather than an actual agent decision.
    """
    raise NotImplementedError("todo: fan retrieve over sub_queries + rerank")


def build_graph():
    """Construct and compile the StateGraph. Returns a graph with `.invoke` / `.ainvoke`.

    Wiring checklist (note to self):
        - add all nodes (fast_path, router, intent_expander, retrieve, ood_gate,
          grader, generator, citation_validator, checker, rewriter)
        - entry point = fast_path
        - conditional edges per the flow above
        - loop-budget guard on the rewriter's outgoing edge
        - hook up LangSmith tracing (env-driven)
    """
    raise NotImplementedError("todo: assemble StateGraph + conditional edges")


def answer_query(query: str) -> AgentState:
    """Build the graph (or reuse a cached one) and run one query."""
    raise NotImplementedError("todo: compile once, then invoke with the initial state")
