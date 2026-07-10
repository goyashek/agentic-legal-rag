"""Retrieval baseline over the hand-labeled scenario set (Week 1 Fri deliverable).

Runs hybrid retrieval (+ optional rerank) over data/eval/scenarios.jsonl and reports
section-level P@5, Recall@5 and MRR. This is the number I freeze BEFORE any tuning,
so Week 3 changes (chunking, rewriter prompt, reranker ablation) have something
honest to move against.

Why three metrics, not just P@5: the plan asked for P@5, but the scenarios have only
1-3 relevant sections each, so a perfect single-answer retrieval still caps P@5 at
0.20. Recall@5 (did the applicable offence show up at all?) and MRR (how high?) are
the numbers that actually reflect quality on a small legal ground-truth set. I report
all three and never dress P@5 up as something it isn't.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.eval.section_precision import section_precision_at_k, section_recall


@dataclass
class ScenarioResult:
    id: str
    difficulty: str
    p_at_5: float
    recall_at_5: float
    mrr: float
    top5: list[str]
    ground_truth: list[str]


def _mrr(ranked: list[str], relevant: list[str]) -> float:
    """Reciprocal rank of the first relevant label (0 if none present)."""
    rel = set(relevant)
    for i, label in enumerate(ranked, start=1):
        if label in rel:
            return 1.0 / i
    return 0.0


def _labels(chunks) -> list[str]:
    """Ordered, de-duplicated 'ACT::section' labels from retrieved/reranked chunks."""
    out: list[str] = []
    for c in chunks:
        label = f"{c.chunk.act}::{c.chunk.section_id}"
        if label not in out:
            out.append(label)
    return out


def load_scenarios(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_baseline(
    retriever,
    scenarios: list[dict],
    *,
    reranker=None,
    retrieve_k: int = 20,
    rerank_k: int = 8,
) -> list[ScenarioResult]:
    """Evaluate the retriever (optionally reranked) over the scenarios."""
    results: list[ScenarioResult] = []
    for s in scenarios:
        candidates = retriever.retrieve(s["query"], top_k=retrieve_k)
        if reranker is not None:
            candidates = reranker.rerank(s["query"], candidates, top_k=rerank_k)
        labels = _labels(candidates)
        gt = s["relevant_sections"]
        results.append(
            ScenarioResult(
                id=s["id"],
                difficulty=s.get("difficulty", "?"),
                p_at_5=section_precision_at_k(labels, gt, k=5),
                recall_at_5=section_recall(labels[:5], gt),
                mrr=_mrr(labels, gt),
                top5=labels[:5],
                ground_truth=gt,
            )
        )
    return results


def summarize(results: list[ScenarioResult]) -> dict[str, float]:
    n = len(results) or 1
    return {
        "p_at_5": sum(r.p_at_5 for r in results) / n,
        "recall_at_5": sum(r.recall_at_5 for r in results) / n,
        "mrr": sum(r.mrr for r in results) / n,
    }


def main() -> None:  # pragma: no cover - thin CLI wrapper
    import logging

    logging.disable(logging.WARNING)
    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.rerank import Reranker

    scenarios = load_scenarios("data/eval/scenarios.jsonl")
    retriever = HybridRetriever(collection="legal", bm25_path="data/processed/bm25.pkl")
    results = run_baseline(retriever, scenarios, reranker=Reranker())

    for r in results:
        print(
            f"{r.id} [{r.difficulty:6}] P@5={r.p_at_5:.2f} "
            f"R@5={r.recall_at_5:.2f} MRR={r.mrr:.2f}  top5={r.top5}"
        )
    means = summarize(results)
    print(
        f"\nBASELINE MEANS  P@5={means['p_at_5']:.3f}  "
        f"Recall@5={means['recall_at_5']:.3f}  MRR={means['mrr']:.3f}"
    )


if __name__ == "__main__":
    main()
