"""RAGAS evaluation, my main metric, run on the actual generative task.

Unlike AIBE (external, full of caveats), RAGAS runs on the 50 scenarios I
hand-labeled myself, which mirror what the system actually does: generative cited
criminal-law advice. So this is the number I trust most.

Metrics I'm tracking: faithfulness, answer_relevancy, context_precision,
context_recall. Regression alert if any drops >5pt from baseline (also enforced in CI
via smoke_gate.py).

Scenario file: data/eval/scenarios.jsonl (kept in git, see .gitignore).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RagasScores:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    n_scenarios: int
    per_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)  # easy/medium/hard breakdown


def load_scenarios(path: str | Path = "data/eval/scenarios.jsonl") -> list[dict]:
    """Load hand-labeled scenarios: {query, ground_truth, relevant_section_ids, difficulty}."""
    raise NotImplementedError("week 3 mon-tue: the 50-scenario grind, no skipping")


def run_ragas_eval(scenarios: list[dict] | None = None) -> RagasScores:
    """Run the agent over scenarios and score with RAGAS.

    Returns RagasScores incl. an easy/medium/hard breakdown so weak spots are visible.
    """
    raise NotImplementedError("week 3 wed: run agent per scenario, score via ragas")
