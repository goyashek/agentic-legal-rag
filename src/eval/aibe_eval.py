"""AIBE eval harness. Gives me an external number to compare against, but with big
caveats so I don't kid myself.

See the "AIBE reality check" notes in NOTES.md before touching this. Two ways AIBE
doesn't line up with what I'm actually building, and both need honest handling:

  1. Task mismatch: AIBE is multiple-choice across ~19 subjects, but my system does
     generative criminal-law advice. So I (a) filter AIBE down to just the
     criminal-law questions (IPC/CrPC/Evidence) and (b) bolt on an MCQ mode: retrieve
     over my BNS corpus, then let the LLM pick whichever option the retrieved
     sections support best.

  2. Temporal mismatch: AIBE 4-16 are pre-2024 and cite the REPEALED IPC/CrPC/Evidence.
     My corpus is BNS/BNSS/BSA. So this is secretly a test of my IPC->BNS mapping.
     I report bridge accuracy separately, since that split is more honest than a raw
     headline score.

Wiring notes:
- `answer_mcq` mirrors the generator's instructor path: retrieve over BNS, feed the
  question + options + retrieved sections to Flash, force a structured single-index
  choice. Retriever + client are injectable so unit tests run at zero quota.
- `load_aibe_criminal_slice` is the ONE piece still blocked: the dataset is gated
  (needs HF_TOKEN + accepting CC BY-ND-4.0 terms) and its column names aren't visible
  until then, so I refuse to ship a guessed loader. Everything downstream takes an
  already-loaded slice, so the whole harness is runnable/tested except that fetch.

Dataset: opennyaiorg/aibe_dataset (CC BY-ND-4.0, eval-only, gated, needs HF_TOKEN).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass
class AIBEResult:
    total: int
    correct: int
    accuracy: float
    # the breakdown I actually care about
    criminal_slice_size: int
    bridge_resolved: int          # questions whose repealed-IPC refs mapped to BNS
    bridge_accuracy: float        # accuracy on just the bridge-dependent subset
    baseline_accuracy: float | None = None   # same questions, no-RAG Gemini for comparison


class _MCQChoice(BaseModel):
    """Structured MCQ answer. instructor forces the model to pick exactly one index."""

    answer_idx: int = Field(description="0-based index of the single best option")


def load_aibe_criminal_slice(hf_token: str | None = None) -> list[dict]:
    """Load AIBE and keep only the criminal-law questions.

    Returns MCQ records: {question, options: list[str], answer_idx, ipc_refs: list[str]}.
    `ipc_refs` are the repealed sections a question cites, used for the bridge metric.

    BLOCKED (Thu): the dataset is gated (accept CC BY-ND-4.0 + HF_TOKEN) and its exact
    column names aren't visible until then. Wiring a loader against guessed field names
    would ship a confident-wrong parser, so this stays a flagged stub — the rest of the
    harness takes an injected slice and is fully runnable. Fill this in once the schema
    is confirmed from the HF Data Studio viewer.
    """
    raise NotImplementedError(
        "week 3 thu: accept aibe gated terms, confirm column names, filter criminal slice"
    )


def _bridge_ref(section_id: str, ipc_bns_mapping: dict[str, str]) -> str | None:
    """Resolve one repealed-IPC section to its BNS section, or None if unmapped."""
    return ipc_bns_mapping.get(section_id)


def answer_mcq(
    question: str,
    options: list[str],
    *,
    retriever=None,
    reranker=None,
    client=None,
    retrieve_k: int = 20,
    rerank_k: int = 8,
) -> int:
    """MCQ mode: retrieve over the BNS corpus, then pick the best-supported option index.

    Not the same as the normal generative path. Here I squash the graph output down to
    a single option choice (reuses retrieval + rerank). Returns a 0-based index, clamped
    into range so a model that hallucinates an out-of-bounds index can't crash the run.
    Retriever/reranker/client are injectable; defaults build the real stack.
    """
    if not options:
        raise ValueError("MCQ needs at least one option")
    if retriever is None:
        from src.retrieval.hybrid import HybridRetriever

        retriever = HybridRetriever(collection="legal", bm25_path="data/processed/bm25.pkl")
    if client is None:
        from src.agent.llm import get_client

        client = get_client("flash")

    candidates = retriever.retrieve(question, top_k=retrieve_k)
    if reranker is not None:
        candidates = reranker.rerank(question, candidates, top_k=rerank_k)
    context = "\n\n".join(
        f"[{c.chunk.act} {c.chunk.section_id}] {c.chunk.heading}\n{c.chunk.text[:1500]}"
        for c in candidates[:rerank_k]
    )
    numbered = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options))
    prompt = (
        "You are answering an Indian criminal-law multiple-choice question using ONLY "
        "the retrieved statutory sections below. The question may cite repealed IPC/CrPC/"
        "Evidence sections; the retrieved sections are the current BNS/BNSS/BSA equivalents.\n\n"
        f"Sections:\n{context}\n\nQuestion: {question}\n\nOptions:\n{numbered}\n\n"
        "Return the 0-based index of the single best-supported option."
    )
    choice: _MCQChoice = client.create(  # type: ignore[attr-defined]
        messages=[{"role": "user", "content": prompt}],
        response_model=_MCQChoice,
        temperature=0,
    )
    return max(0, min(choice.answer_idx, len(options) - 1))


def score(predictions: list[int], answers: list[int]) -> float:
    """Accuracy over aligned predicted/gold indices. 0.0 on an empty set."""
    if not answers:
        return 0.0
    correct = sum(1 for p, a in zip(predictions, answers, strict=True) if p == a)
    return correct / len(answers)


def compute_result(
    slice_: list[dict],
    predictions: list[int],
    ipc_bns_mapping: dict[str, str],
    *,
    baseline_predictions: list[int] | None = None,
) -> AIBEResult:
    """Assemble the honest breakdown from a scored run. Pure — the number-crunching the
    unit tests pin, separate from the LLM/retrieval calls.

    A question is 'bridge-dependent' if any of its cited IPC refs resolves to a BNS
    section; bridge_accuracy is accuracy restricted to that subset (the split that
    actually reflects the IPC->BNS mapping, per NOTES).
    """
    answers = [q["answer_idx"] for q in slice_]
    total = len(slice_)
    correct = sum(1 for p, a in zip(predictions, answers, strict=True) if p == a)

    bridge_idx = [
        i
        for i, q in enumerate(slice_)
        if any(_bridge_ref(r, ipc_bns_mapping) for r in q.get("ipc_refs", []))
    ]
    bridge_preds = [predictions[i] for i in bridge_idx]
    bridge_answers = [answers[i] for i in bridge_idx]

    return AIBEResult(
        total=total,
        correct=correct,
        accuracy=score(predictions, answers),
        criminal_slice_size=total,
        bridge_resolved=len(bridge_idx),
        bridge_accuracy=score(bridge_preds, bridge_answers),
        baseline_accuracy=(
            score(baseline_predictions, answers) if baseline_predictions is not None else None
        ),
    )


def run_aibe_eval(
    hf_token: str | None = None,
    *,
    with_baseline: bool = True,
    slice_: list[dict] | None = None,
    mcq_fn=None,
    baseline_fn=None,
    ipc_bns_mapping: dict[str, str] | None = None,
    pace_seconds: float = 4.0,
    sleep=time.sleep,
) -> AIBEResult:
    """Full run over the AIBE criminal slice, with the honest bridge breakdown.

    Returns an AIBEResult. Always show accuracy AND bridge_accuracy AND baseline
    together, never just a bare bar-exam percentage (see NOTES.md on why that'd be
    misleading). `slice_`/`mcq_fn`/`baseline_fn`/`ipc_bns_mapping` are injectable so the
    orchestration is unit-tested at zero quota; defaults load the gated slice + real
    stack. Paces between questions to respect the flash-tier RPM wall.
    """
    if slice_ is None:
        slice_ = load_aibe_criminal_slice(hf_token)
    if mcq_fn is None:
        mcq_fn = answer_mcq
    if ipc_bns_mapping is None:
        from src.agent.nodes.fast_path import _resolver

        _, ipc_bns_mapping = _resolver()

    predictions: list[int] = []
    for i, q in enumerate(slice_):
        if i > 0 and pace_seconds > 0:
            sleep(pace_seconds)
        predictions.append(mcq_fn(q["question"], q["options"]))

    baseline_predictions: list[int] | None = None
    if with_baseline and baseline_fn is not None:
        baseline_predictions = []
        for i, q in enumerate(slice_):
            if i > 0 and pace_seconds > 0:
                sleep(pace_seconds)
            baseline_predictions.append(baseline_fn(q["question"], q["options"]))

    return compute_result(
        slice_, predictions, ipc_bns_mapping, baseline_predictions=baseline_predictions
    )
