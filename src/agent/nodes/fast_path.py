"""Exact-section fast path. Deterministic, runs before the LLM router.

If the query names an exact section ("BNS Section 103", "Section 302 IPC") I skip
embedding and the LLM entirely and just do a direct metadata lookup. Aiming for
under 50ms, and it avoids semantic drift on exact lookups.

IPC references get resolved through the IPC->BNS mapping first, so "302 IPC"
correctly returns BNS 103.
"""

from __future__ import annotations

import re

from src.agent.state import AgentState

# Matches "BNS 103", "BNS Section 103", "Section 302 IPC", "s. 63A BSA", etc.
# Kept module-level so tests can assert against the exact pattern.
SECTION_PATTERN = re.compile(
    r"\b(?:section\s+|s\.?\s*)?(?P<num>\d+[A-Z]?(?:\(\d+\))?)\s*"
    r"(?P<act>BNS|BNSS|BSA|IPC|CrPC|Evidence)?\b",
    re.IGNORECASE,
)


def detect_exact_section(query: str) -> tuple[str, str] | None:
    """Return (act, section_id) if the query is an exact-section lookup, else None.

    Must not fire on narrative queries like "someone stole my bike", only on explicit
    section references. A false positive here quietly wrecks precision.
    """
    raise NotImplementedError("week 1 fri: regex detect + IPC normalization")


def fast_path_node(state: AgentState) -> AgentState:
    """LangGraph node. On an exact-section hit, set fast_path_hit + fast_path_answer.
    On a hit the graph routes straight to output, otherwise falls through to the router.
    """
    raise NotImplementedError("week 1 fri: direct metadata lookup on hit")
