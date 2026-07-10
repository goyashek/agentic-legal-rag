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

Dataset: opennyaiorg/aibe_dataset (CC BY-ND-4.0, eval-only, gated, needs HF_TOKEN).
"""

from __future__ import annotations

from dataclasses import dataclass


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


def load_aibe_criminal_slice(hf_token: str | None = None) -> list[dict]:
    """Load AIBE and keep only the criminal-law questions.

    Returns MCQ records: {question, options: list[str], answer_idx, ipc_refs: list[str]}.
    `ipc_refs` are the repealed sections a question cites, used for the bridge metric.
    """
    raise NotImplementedError("week 3 wed: load gated dataset, filter criminal slice")


def answer_mcq(question: str, options: list[str]) -> int:
    """MCQ mode: retrieve over the BNS corpus, then pick the best-supported option index.

    Not the same as the normal generative path. Here I squash the graph output down
    to a single option choice. Reuses retrieval + the IPC->BNS bridge.
    """
    raise NotImplementedError("week 3 wed: retrieve + option-selection")


def run_aibe_eval(hf_token: str | None = None, *, with_baseline: bool = True) -> AIBEResult:
    """Full run over the AIBE criminal slice, with the honest bridge breakdown.

    Returns an AIBEResult. Always show accuracy AND bridge_accuracy AND baseline
    together, never just a bare bar-exam percentage (see NOTES.md on why that'd be
    misleading).
    """
    raise NotImplementedError("week 3 wed: run slice, compute bridge + baseline")
