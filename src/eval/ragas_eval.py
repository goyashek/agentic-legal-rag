"""RAGAS evaluation on the same cited-answer task the agent actually handles.

The 50 scenarios are my hand-labelled criminal-law prompts. I track faithfulness,
answer relevancy, context precision, and context recall, with a CI alert for a drop
of more than five points from the recorded baseline.

DeepSeek V4 Flash now has enough account-level concurrency that scenarios can run
without the old artificial delay. The answer trace and optional score manifest make
each paid evaluation reproducible. Unit tests still inject a fake answer function,
and RAGAS remains a lazy import so they need neither an API key nor the package.

The scenarios contain gold section IDs rather than reference prose. For the two
context metrics, `build_reference` joins the corresponding statutory chunks.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from src.agent.graph import PipelineVariant
from src.eval.retrieval_baseline import load_scenarios  # same jsonl loader, don't duplicate

# The four RAGAS metrics I headline. Names match the ragas metric objects AND the
# to_pandas() column names, so aggregation can key off them directly.
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

# RAGAS answer-relevancy embeddings run locally, so evaluation makes no provider call beyond
# DeepSeek's judge. Small keeps the evaluator light; override only when comparing models.
_RAGAS_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# DeepSeek V4 allows 2,500 concurrent Flash requests per account. Eight workers
# keep this evaluation comfortably below that ceiling while avoiding the former
# serial 200-call judge run.
RAGAS_MAX_WORKERS = 8


class _LegacyEmbeddingAdapter:
    """Expose RAGAS's current embedding provider through its legacy metric API."""

    def __init__(self, embeddings) -> None:
        self._embeddings = embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_text(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_texts(texts)


def _shim_dead_vertexai_import() -> None:
    """ragas 0.4.x hard-imports `langchain_community.chat_models.vertexai.ChatVertexAI`,
    a path langchain-community 0.4.x removed (sunset). `ChatVertexAI` is only used in an
    isinstance list ragas never hits, so inject a placeholder module before `import ragas`
    instead of pinning an obsolete dependency. Idempotent.
    """
    import sys
    import types

    name = "langchain_community.chat_models.vertexai"
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    mod.ChatVertexAI = type("ChatVertexAI", (), {})  # never instantiated by this project
    sys.modules[name] = mod


def _ragas_evaluator():
    """Build DeepSeek judge + local embeddings for RAGAS without a second API provider."""
    _shim_dead_vertexai_import()
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import LangchainLLMWrapper

    from src.agent.llm import _max_tokens_for, _model_for, _resolve_key

    judge = ChatOpenAI(
        model=_model_for("flash"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        api_key=_resolve_key(),
        temperature=0,
        max_tokens=_max_tokens_for("flash"),
        extra_body={"thinking": {"type": "disabled"}},
    )
    return (
        LangchainLLMWrapper(judge),
        _LegacyEmbeddingAdapter(
            HuggingFaceEmbeddings(model=os.getenv("RAGAS_EMBED_MODEL", _RAGAS_EMBED_MODEL))
        ),
    )


@dataclass
class RagasScores:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    n_scenarios: int
    # easy/medium/hard breakdown
    per_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)


def _corpus_text_by_section(corpus) -> dict[str, list[str]]:
    """Map 'ACT::section_id' -> list of chunk texts (a section can be several chunks)."""
    out: dict[str, list[str]] = defaultdict(list)
    for c in corpus:
        out[f"{c.act}::{c.section_id}"].append(c.text)
    return out


def build_reference(relevant_sections, text_by_section: dict[str, list[str]]) -> str:
    """Reference = the statutory text of the gold sections, joined.

    Empty string if none of the gold sections are in the corpus (shouldn't happen —
    the scenario set is corpus-verified — but keep it total rather than KeyError).
    """
    parts: list[str] = []
    for sec in relevant_sections:
        parts.extend(text_by_section.get(sec, []))
    return "\n\n".join(parts)


def _extract(state) -> tuple[str, list[str]]:
    """Pull (response prose, retrieved-context texts) out of an agent AgentState.

    Prefers the graded `relevant_chunks` (what generation actually reasoned over);
    falls back to `retrieved` if grading didn't populate them (e.g. fast-path hit).
    """
    answer = state.get("answer") or state.get("fast_path_answer")
    response = answer.answer if answer is not None else ""
    chunks = state.get("relevant_chunks") or state.get("retrieved", [])
    contexts = [c.chunk.text for c in chunks]
    return response, contexts


def collect_samples(
    scenarios: list[dict],
    *,
    answer_fn=None,
    corpus=None,
    pace_seconds: float = 0.0,
    sleep=time.sleep,
) -> list[dict]:
    """Run the agent over each scenario and collect the RAGAS inputs.

    Returns RAGAS inputs plus the graph trace for each scenario. `answer_fn` defaults
    to the compiled graph's `answer_query`; injected in tests. `sleep` is injected so
    optional pacing is testable without real waits.
    """
    if answer_fn is None:
        from src.agent.graph import answer_query

        answer_fn = answer_query
    if corpus is None:
        from src.retrieval.index import load_chunks

        corpus = load_chunks("data/processed/sections.jsonl")

    text_by_section = _corpus_text_by_section(corpus)
    rows: list[dict] = []
    for i, s in enumerate(scenarios):
        if i > 0 and pace_seconds > 0:
            sleep(pace_seconds)
        state = answer_fn(s["query"])
        response, contexts = _extract(state)
        answer = state.get("answer") or state.get("fast_path_answer")
        trace_notes = state.get("trace_notes", [])
        rows.append(
            {
                "scenario_id": s.get("id", "?"),
                "user_input": s["query"],
                "retrieved_contexts": contexts,
                "response": response,
                "reference": build_reference(s["relevant_sections"], text_by_section),
                "difficulty": s.get("difficulty", "?"),
                "trace_notes": trace_notes,
                "citations": [
                    citation.model_dump(exclude_none=True)
                    for citation in getattr(answer, "citations", [])
                ],
                "confidence": getattr(answer, "confidence", None),
                "in_corpus": getattr(answer, "in_corpus", None),
            }
        )
    return rows


def write_samples(rows: list[dict], path: str | Path) -> None:
    """Write raw agent outputs and trace notes for a reproducible scored run."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_scores(
    scores: RagasScores,
    path: str | Path,
    *,
    retrieval_mode: str,
    use_reranker: bool,
    pipeline: str,
    samples_out: str | Path | None,
) -> None:
    """Persist the result with the model and retrieval settings that produced it."""
    from src.agent.llm import _model_for

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "judge_model": _model_for("flash"),
        "control_model": _model_for("flash"),
        "answer_model": _model_for("pro"),
        "retrieval_mode": retrieval_mode,
        "use_reranker": use_reranker,
        "pipeline": pipeline,
        "samples_out": str(samples_out) if samples_out is not None else None,
        "scores": asdict(scores),
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def aggregate(scored_rows: list[dict]) -> RagasScores:
    """Mean each metric overall + per difficulty. Pure; takes RAGAS per-sample records
    (each row has the four metric keys + 'difficulty'), so it's tested without ragas.

    Missing/NaN metric values are skipped in the mean (RAGAS emits NaN when a metric
    can't be computed for a sample, e.g. an empty answer) rather than counted as 0,
    which would silently tank the average.
    """

    def _valid(v) -> bool:
        return isinstance(v, int | float) and v == v  # v == v is False only for NaN

    def _mean(rows: list[dict], metric: str) -> float:
        vals = [r[metric] for r in rows if _valid(r.get(metric))]
        return sum(vals) / len(vals) if vals else 0.0

    per_difficulty: dict[str, dict[str, float]] = {}
    by_diff: dict[str, list[dict]] = defaultdict(list)
    for r in scored_rows:
        by_diff[r.get("difficulty", "?")].append(r)
    for diff, rows in by_diff.items():
        per_difficulty[diff] = {m: _mean(rows, m) for m in METRIC_NAMES}

    return RagasScores(
        **{m: _mean(scored_rows, m) for m in METRIC_NAMES},
        n_scenarios=len(scored_rows),
        per_difficulty=per_difficulty,
    )


def run_ragas_eval(
    scenarios: list[dict] | None = None,
    *,
    answer_fn=None,
    corpus=None,
    pace_seconds: float = 0.0,
    retrieval_mode: Literal["hybrid", "dense", "sparse"] = "hybrid",
    use_reranker: bool = True,
    pipeline: PipelineVariant = "full",
    samples_out: str | Path | None = None,
    scores_out: str | Path | None = None,
) -> RagasScores:
    """Run the agent over scenarios and score with RAGAS.

    Returns RagasScores incl. an easy/medium/hard breakdown so weak spots are visible.
    `ragas` is imported here (not module-load) so the keyless test suite doesn't need it.
    """
    if scenarios is None:
        scenarios = load_scenarios("data/eval/scenarios.jsonl")
    if answer_fn is None:
        from src.agent.graph import answer_query

        def answer_fn(query: str):
            return answer_query(
                query,
                retrieval_mode=retrieval_mode,
                use_reranker=use_reranker,
                pipeline=pipeline,
            )

    rows = collect_samples(scenarios, answer_fn=answer_fn, corpus=corpus, pace_seconds=pace_seconds)
    for row in rows:
        row["pipeline"] = pipeline
    if samples_out is not None:
        write_samples(rows, samples_out)

    llm, embeddings = _ragas_evaluator()
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    # Bind the DeepSeek evaluator explicitly; ragas otherwise defaults to OpenAI.
    for metric in (faithfulness, answer_relevancy, context_precision, context_recall):
        metric.llm = llm
    answer_relevancy.embeddings = embeddings
    # answer_relevancy defaults to strictness=3 -> asks the judge for n=3 generations, which
    # One generation keeps the judge compact; more self-consistency samples multiply cost.
    answer_relevancy.strictness = 1

    dataset = EvaluationDataset.from_list(
        [
            {key: row[key] for key in ("user_input", "retrieved_contexts", "response", "reference")}
            for row in rows
        ]
    )
    run_config = RunConfig(max_workers=RAGAS_MAX_WORKERS, timeout=600)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )

    scored = result.to_pandas().to_dict("records")
    for row, src in zip(scored, rows, strict=True):
        row["difficulty"] = src["difficulty"]  # ragas drops it; realign by order
    scores = aggregate(scored)
    if scores_out is not None:
        write_scores(
            scores,
            scores_out,
            retrieval_mode=retrieval_mode,
            use_reranker=use_reranker,
            pipeline=pipeline,
            samples_out=samples_out,
        )
    return scores


def main() -> None:  # pragma: no cover - thin CLI wrapper
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Run the full RAGAS scenario evaluation")
    parser.add_argument("--mode", choices=("hybrid", "dense", "sparse"), default="hybrid")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument(
        "--pipeline",
        choices=("production", "baseline", "grader", "checker", "full"),
        default="full",
    )
    parser.add_argument("--samples-out", type=Path)
    parser.add_argument("--scores-out", type=Path)
    args = parser.parse_args()

    scores = run_ragas_eval(
        retrieval_mode=args.mode,
        use_reranker=not args.no_rerank,
        pipeline=args.pipeline,
        samples_out=args.samples_out,
        scores_out=args.scores_out,
    )
    print(f"RAGAS over {scores.n_scenarios} scenarios:")
    for m in METRIC_NAMES:
        print(f"  {m:20} {getattr(scores, m):.3f}")
    for diff, ms in sorted(scores.per_difficulty.items()):
        print(f"  [{diff}] " + "  ".join(f"{m}={ms[m]:.2f}" for m in METRIC_NAMES))


if __name__ == "__main__":
    main()
