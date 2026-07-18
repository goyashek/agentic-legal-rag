"""MCQ eval harness — external comparability on the BhashaBench-Legal criminal slice.

Replaces the earlier AIBE plan. AIBE was dropped after inspecting it: no subject
field, and its honestly-answerable IPC slice was only ~6-15 questions across all
exams — too thin to headline (see NOTES "AIBE reality check" + the Week-3 pivot).

BhashaBench-Legal (bharatgenai/BhashaBench-Legal, English config, CC BY-4.0) gives a
real slice instead: `subject_domain == "Criminal Law & Justice"` is 1,825 MCQs, and
**579 of them cite the repealed IPC/CrPC/Evidence** — so the IPC->BNS bridge story
("answers old-code questions in the new code") gets a proper external validation set,
not a handful.

Two mismatches still need honest handling, same shape as before:

  1. Task mismatch: BhashaBench is multiple-choice; my system does generative
     criminal-law advice. So I bolt on an MCQ mode — retrieve over the BNS corpus,
     then let Flash pick whichever option the retrieved sections best support.

  2. Coverage mismatch: I only built the IPC->BNS bridge, not CrPC->BNSS or
     Evidence->BSA. Plenty of the slice is procedure/evidence/trivia my penal-code
     retrieval can't answer. So I NEVER report one bare number — always overall
     accuracy AND the old-code-cited "bridge" subset AND a no-RAG baseline, so RAG's
     actual lift is visible where it applies.

Wiring notes:
- `answer_mcq` mirrors the generator's instructor path: retrieve over BNS, feed the
  question + options + retrieved sections to Flash, force a structured single-index
  choice. Retriever + client are injectable so unit tests run at zero quota.
- `load_bhashabench_criminal_slice` filters to the criminal MCQ slice and extracts
  IPC refs from the question text for the bridge metric. `datasets` is imported lazily
  so the keyless suite doesn't need it installed.
- Everything downstream takes an already-loaded slice, so the whole harness is
  runnable/tested independent of the dataset fetch.

Dataset: bharatgenai/BhashaBench-Legal (English, CC BY-4.0, gated, needs HF_TOKEN).
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

from pydantic import BaseModel, Field

_DATASET = "bharatgenai/BhashaBench-Legal"
_CRIMINAL_DOMAIN = "Criminal Law & Justice"

# "Section 302 of the IPC", "u/s 124A IPC", "s. 82 of the Indian Penal Code" -> capture
# the section number (with optional trailing letter, e.g. 124A) when an IPC marker is
# near it. Deliberately IPC-only: that's the one bridge I actually built.
# The section-number group allows a lettered suffix in any of the forms BhashaBench uses:
# "302", "124A", "124 A", "124-A" (hyphen/en-dash). The suffix is significant — IPC 498A
# (cruelty) is a DIFFERENT section from 498 and maps to a different BNS section, so it must
# survive extraction, not get stripped. `_extract_ipc_refs` normalizes the separators off.
# The trailing (?![A-Za-z]) is load-bearing: without it the optional suffix letter greedily
# grabs the first letter of the NEXT word ("302 of" -> "302O", "302 IPC" -> "302I"). A real
# suffix is a lone letter NOT followed by another letter, so "498-A came" keeps the A but
# "369 of" doesn't.
_SEC_NUM = r"(\d+\s*[-–]?\s*[A-Za-z]?)(?![A-Za-z])"
_IPC_SECTION = re.compile(
    rf"(?:[Ss]ection|[Ss]ec\.?|u/s|S\.)\s*{_SEC_NUM}[^.]*?(?:I\.?P\.?C|Indian\s+Penal)",
    re.I,
)
_IPC_SECTION_REV = re.compile(  # the other order: "IPC ... Section 302"
    rf"(?:I\.?P\.?C|Indian\s+Penal)[^.]*?(?:[Ss]ection|[Ss]ec\.?|u/s|S\.)\s*{_SEC_NUM}",
    re.I,
)


@dataclass
class MCQResult:
    total: int
    correct: int
    accuracy: float
    # the breakdown I actually care about
    criminal_slice_size: int
    bridge_resolved: int  # questions whose repealed-IPC refs mapped to BNS
    bridge_accuracy: float  # accuracy on just the bridge-dependent subset
    baseline_accuracy: float | None = None  # same questions, no-RAG model for comparison
    # no-RAG on the bridge subset — the head-to-head that matters
    baseline_bridge_accuracy: float | None = None


class _MCQChoice(BaseModel):
    """Structured MCQ answer. instructor forces the model to pick exactly one index."""

    answer_idx: int = Field(description="0-based index of the single best option")


def _extract_ipc_refs(question: str) -> list[str]:
    """Repealed-IPC section numbers cited in the question (normalized, deduped).

    Only IPC — the bridge I built is IPC->BNS. A '124A' keeps its letter; whitespace
    inside the match ('124 A') is squeezed out so it keys the mapping cleanly.
    """
    refs: list[str] = []
    for pat in (_IPC_SECTION, _IPC_SECTION_REV):
        for m in pat.findall(question):
            norm = re.sub(r"[\s\-–]+", "", m).upper()  # "124 - A" / "124-A" -> "124A"
            if norm not in refs:
                refs.append(norm)
    return refs


def load_bhashabench_criminal_slice(
    hf_token: str | None = None, *, mcq_only: bool = True
) -> list[dict]:
    """Load BhashaBench-Legal (English) and keep the criminal-law slice.

    Returns records: {question, options: list[str], answer_idx, ipc_refs: list[str]}.
    `answer_idx` is the 0-based index of `correct_answer` (a letter A-D). `ipc_refs` are
    the repealed IPC sections the question cites, used for the bridge metric.

    `mcq_only` drops non-MCQ formats (Match-the-column, Rearrange, Reading-comprehension)
    a retrieve-then-pick system structurally can't do. Token falls back to $HF_TOKEN.
    """
    from datasets import load_dataset

    token = hf_token or os.getenv("HF_TOKEN") or None
    d = load_dataset(_DATASET, "English", token=token)["test"]

    letters = ("A", "B", "C", "D")
    records: list[dict] = []
    for row in d:
        if row["subject_domain"] != _CRIMINAL_DOMAIN:
            continue
        if mcq_only and row["question_type"] != "MCQ":
            continue
        answer = (row["correct_answer"] or "").strip().upper()
        if answer not in letters:
            continue  # a handful have malformed/missing answers; skip rather than guess
        options = [row["option_a"], row["option_b"], row["option_c"], row["option_d"]]
        records.append(
            {
                "question": row["question"],
                "options": options,
                "answer_idx": letters.index(answer),
                "ipc_refs": _extract_ipc_refs(row["question"]),
                "level": row.get("question_level", "?"),
            }
        )
    return records


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

        client = get_client("easy")

    candidates = retriever.retrieve(question, top_k=retrieve_k)
    if reranker is not None:
        candidates = reranker.rerank(question, candidates, top_k=rerank_k)
    context = "\n\n".join(
        f"[{c.chunk.act} {c.chunk.section_id}] {c.chunk.heading}\n{c.chunk.text[:1500]}"
        for c in candidates[:rerank_k]
    )
    numbered = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options))
    prompt = (
        "You are answering an Indian criminal-law multiple-choice question. Use the "
        "retrieved statutory sections below as your primary evidence. The question may cite "
        "repealed IPC/CrPC/Evidence sections; the retrieved sections are the current "
        "BNS/BNSS/BSA equivalents.\n\n"
        f"Sections:\n{context}\n\nQuestion: {question}\n\nOptions:\n{numbered}\n\n"
        "You MUST pick exactly one option even if the retrieved sections are incomplete or "
        "don't contain the exact provision — fall back to your own legal knowledge and choose "
        "the most likely answer. Never refuse or ask for more information; this is a forced-"
        "choice question. Return the 0-based index of the single best option."
    )
    choice: _MCQChoice = client.create(  # type: ignore[attr-defined]
        messages=[{"role": "user", "content": prompt}],
        response_model=_MCQChoice,
        temperature=0,
    )
    return max(0, min(choice.answer_idx, len(options) - 1))


def answer_mcq_no_rag(question: str, options: list[str], *, client=None) -> int:
    """No-RAG baseline: same model, same clamping, NO retrieval — DeepSeek answers from its
    own parametric knowledge. This is the comparison that shows RAG's lift; pass it as
    `baseline_fn` so one paced run scores system + baseline together (don't burn the RPD
    cap on two separate runs). Signature-compatible with `answer_mcq` for the runner.
    """
    if not options:
        raise ValueError("MCQ needs at least one option")
    if client is None:
        from src.agent.llm import get_client

        client = get_client("easy")
    numbered = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options))
    prompt = (
        "Answer this Indian criminal-law multiple-choice question from your own knowledge.\n\n"
        f"Question: {question}\n\nOptions:\n{numbered}\n\n"
        "Return the 0-based index of the single best option."
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
) -> MCQResult:
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

    baseline_acc = baseline_bridge_acc = None
    if baseline_predictions is not None:
        baseline_acc = score(baseline_predictions, answers)
        # no-RAG on the SAME bridge subset — the head-to-head that isolates the IPC->BNS
        # bridge's value from the procedure/evidence noise that drags the overall number.
        baseline_bridge_acc = score([baseline_predictions[i] for i in bridge_idx], bridge_answers)

    return MCQResult(
        total=total,
        correct=correct,
        accuracy=score(predictions, answers),
        criminal_slice_size=total,
        bridge_resolved=len(bridge_idx),
        bridge_accuracy=score(bridge_preds, bridge_answers),
        baseline_accuracy=baseline_acc,
        baseline_bridge_accuracy=baseline_bridge_acc,
    )


def run_mcq_eval(
    hf_token: str | None = None,
    *,
    with_baseline: bool = True,
    slice_: list[dict] | None = None,
    mcq_fn=None,
    baseline_fn=None,
    ipc_bns_mapping: dict[str, str] | None = None,
    limit: int | None = None,
    pace_seconds: float = 4.0,
    sleep=time.sleep,
) -> MCQResult:
    """Full run over the BhashaBench criminal slice, with the honest bridge breakdown.

    Returns an MCQResult. Always show accuracy AND bridge_accuracy AND baseline
    together, never just a bare percentage (see NOTES on why that'd mislead —
    procedure/evidence questions aren't answerable by penal-code retrieval).
    `slice_`/`mcq_fn`/`baseline_fn`/`ipc_bns_mapping` are injectable so the orchestration
    is unit-tested at zero quota; defaults load the gated slice + real stack. `limit`
    caps the run (the full slice is 1,825 * ~1 call each — pace against RPM 15). Paces
    between questions to respect the easy-tier RPM wall.
    """
    if slice_ is None:
        slice_ = load_bhashabench_criminal_slice(hf_token)
    if limit is not None:
        slice_ = slice_[:limit]
    if mcq_fn is None:
        mcq_fn = answer_mcq
    if ipc_bns_mapping is None:
        from src.agent.nodes.fast_path import _resolver

        _, ipc_bns_mapping = _resolver()

    def _run_arm(fn) -> list[int]:
        # One failed question (model refuses / unparseable output) must not abort the whole
        # batch — record -1 (never matches a 0-3 gold answer, so it counts as wrong) and go on.
        preds: list[int] = []
        for i, q in enumerate(slice_):
            if i > 0 and pace_seconds > 0:
                sleep(pace_seconds)
            try:
                preds.append(fn(q["question"], q["options"]))
            except Exception:  # noqa: BLE001 - eval resilience: a bad Q shouldn't kill 150
                preds.append(-1)
        return preds

    predictions = _run_arm(mcq_fn)
    baseline_predictions: list[int] | None = None
    if with_baseline and baseline_fn is not None:
        baseline_predictions = _run_arm(baseline_fn)

    return compute_result(
        slice_, predictions, ipc_bns_mapping, baseline_predictions=baseline_predictions
    )
