"""RAGAS evaluation, my main metric, run on the actual generative task.

Unlike AIBE (external, full of caveats), RAGAS runs on the 50 scenarios I
hand-labeled myself, which mirror what the system actually does: generative cited
criminal-law advice. So this is the number I trust most.

Metrics I'm tracking: faithfulness, answer_relevancy, context_precision,
context_recall. Regression alert if any drops >5pt from baseline (also enforced in CI
via smoke_gate.py).

Wiring notes:
- The agent run per scenario is the quota sink (~12 flash calls each: router + expander
  + ~8 grader + generator + checker). Flash-lite's wall is RPM 15, so `collect_samples`
  PACES between scenarios (`pace_seconds`) rather than rationing tokens/day. 50 scenarios
  at a safe pace is ~40 min; sample a subset for a quick check.
- `answer_fn` is injectable (defaults to the compiled graph) so unit tests collect with a
  fake at zero quota. `ragas` itself is imported lazily inside `run_ragas_eval`, so the
  keyless suite stays green without it installed.
- RAGAS needs a `reference` for context_precision/recall; the scenarios carry gold SECTION
  IDs, not gold prose, so the reference is the statutory text of those sections (the text a
  correct answer must rest on). Built from the corpus in `build_reference`.

Scenario file: data/eval/scenarios.jsonl (kept in git, see .gitignore).
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

from src.eval.retrieval_baseline import load_scenarios  # same jsonl loader, don't duplicate

# The four RAGAS metrics I headline. Names match the ragas metric objects AND the
# to_pandas() column names, so aggregation can key off them directly.
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

# RAGAS answer-relevancy embeddings run locally, so evaluation makes no provider call beyond
# DeepSeek's judge. Small keeps the evaluator light; override only when comparing models.
_RAGAS_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


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
    pace_seconds: float = 4.0,
    sleep=time.sleep,
) -> list[dict]:
    """Run the agent over each scenario and collect the RAGAS inputs.

    Returns dicts: {user_input, retrieved_contexts, response, reference, difficulty}.
    Paces `pace_seconds` between scenarios to keep the per-minute request rate under
    the flash-tier RPM cap (the real bottleneck, not tokens/day). `answer_fn` defaults
    to the compiled graph's `answer_query`; injected in tests. `sleep` injected so the
    pacing itself is testable without real waits.
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
            sleep(pace_seconds)  # rate-limit between agent runs (RPM 15 wall)
        state = answer_fn(s["query"])
        response, contexts = _extract(state)
        rows.append(
            {
                "user_input": s["query"],
                "retrieved_contexts": contexts,
                "response": response,
                "reference": build_reference(s["relevant_sections"], text_by_section),
                "difficulty": s.get("difficulty", "?"),
            }
        )
    return rows


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
    pace_seconds: float = 4.0,
) -> RagasScores:
    """Run the agent over scenarios and score with RAGAS.

    Returns RagasScores incl. an easy/medium/hard breakdown so weak spots are visible.
    `ragas` is imported here (not module-load) so the keyless test suite doesn't need it.
    """
    if scenarios is None:
        scenarios = load_scenarios("data/eval/scenarios.jsonl")

    rows = collect_samples(
        scenarios, answer_fn=answer_fn, corpus=corpus, pace_seconds=pace_seconds
    )

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
        [{k: v for k, v in r.items() if k != "difficulty"} for r in rows]
    )
    # max_workers=1: ragas otherwise fans ALL judge jobs out at once; against a rate-limited
    # judge (5 RPM) the later jobs sit in a queue and blow ragas's per-job timeout -> spurious
    # 0.0 scores. Serial submission means each job's timer starts only when it runs. Generous
    # timeout for the slow paced judge. This — not the LLM-level limiter — is the real burst fix.
    run_config = RunConfig(max_workers=1, timeout=600)
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
    return aggregate(scored)


def main() -> None:  # pragma: no cover - thin CLI wrapper
    scores = run_ragas_eval()
    print(f"RAGAS over {scores.n_scenarios} scenarios:")
    for m in METRIC_NAMES:
        print(f"  {m:20} {getattr(scores, m):.3f}")
    for diff, ms in sorted(scores.per_difficulty.items()):
        print(f"  [{diff}] " + "  ".join(f"{m}={ms[m]:.2f}" for m in METRIC_NAMES))


if __name__ == "__main__":
    main()
