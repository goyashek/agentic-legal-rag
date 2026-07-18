# RAGAS results

This records the full runs over the 50 hand-labelled scenarios in
`data/eval/scenarios.jsonl` (19 easy, 24 medium, 7 hard). The runs are
diagnostics, not production-accuracy claims. Model provenance is part of each
result because the latest routing experiment does not use the earlier
DeepSeek-only answer path.

## Earlier DeepSeek-only setup

- Agent: DeepSeek V4 Flash for routing, expansion, grading, rewriting, and
  checking. DeepSeek V4 Pro wrote the final answer. Thinking was disabled.
- RAGAS judge: DeepSeek V4 Flash at temperature 0 with a 256-token ceiling.
- Answer-relevancy embeddings: local `BAAI/bge-small-en-v1.5`.
- Corpus: the 1,151-chunk local BNS, BNSS, and BSA index after sentence-aware
  chunk repair.
- Scoring: RAGAS `strictness=1` and eight judge workers. The output traces and
  score manifests are local evaluation artifacts and are not committed.

The current-routing run has its own provenance below. It used a 4,096-token
judge ceiling, two workers, and bounded retries so long metric responses and
transient provider failures did not invalidate the saved answer collection.

## Overall scores

The first two rows are the older full-graph runs (router + expander + grader +
checker + rewrite loop). The last two use the simpler production pipeline:
dense retrieval, no reranker, generation, deterministic citation validation,
and the scope/OOD controls. All four use the same 1,151-chunk corpus and the
same 50 scenarios.

| pipeline / retrieval | faithfulness | answer relevancy | context precision | context recall |
|---|---:|---:|---:|---:|
| full graph, dense, no reranker | 0.309 | 0.518 | 0.700 | 0.840 |
| full graph, hybrid RRF + reranker | 0.314 | 0.386 | **0.709** | 0.732 |
| production, dense, no reranker, DeepSeek only | **0.517** | **0.749** | 0.615 | **0.919** |
| production, dense, no reranker, current routing | 0.496 | 0.689 | 0.630 | 0.912 |

The production path is the decisive result. Dropping the checker and rewrite loop
nearly doubles faithfulness (0.309 → 0.517) and answer relevancy (0.518 → 0.749),
and lifts context recall to 0.919. Context precision dips (0.700 → 0.615), which
is consistent with the wider 12-chunk answer window feeding more context in. The
20-case node ablation and the ten-answer statute audit both predicted that the
simple path would beat the full graph on answer quality; this full 50-scenario
run on the actual production pipeline confirms it. Faithfulness at 0.517 is still
middling, so this stays a local demo, but the "coverage is better than the final
answers" gap the earlier full-graph runs showed is largely closed.

Provenance: judge and control nodes on `deepseek-v4-flash`, answers on
`deepseek-v4-pro`, thinking disabled. Every one of the 50 production scenarios
returned a generated answer; none fell back to the canned low-confidence reply,
unlike the full-graph runs where the checker-to-rewriter loop ended 13–20
scenarios in low confidence.

### Current routing run

The 2026-07-18 run tested the two-tier routing setup rather than repeating the
DeepSeek-only model assignment. `mistral/mistral-small-latest` completed all 50
control calls. Answer generation recorded 46 successful
`nvidia/nvidia/nemotron-3-ultra-550b-a55b` calls and 6 successful paid
`deepseek/deepseek-v4-pro` fallback calls. NVIDIA also returned 4 failed
attempts. A few graph paths generated more than once, so the 52 successful
generation calls should not be read as 52 scenarios.

The answer trace was saved before judging. It contains all 50 scenarios, one
low-confidence answer, no empty answers, and citations in every row. The final
judge pass reused that trace and made 200 successful paid
`deepseek/deepseek-v4-flash` calls with no missing metric jobs. This matters for
cost control because judge retries did not trigger another answer collection.

The current-routing result is close to the older DeepSeek-only production run
on context precision and recall, but lower on faithfulness and answer
relevancy. It is not a clean model-quality comparison because both the control
and answer models changed. It does show that the cheaper mixed route can finish
the full workload while keeping paid Pro calls to fallback cases.

| difficulty | faithfulness | answer relevancy | context precision | context recall |
|---|---:|---:|---:|---:|
| easy (19) | 0.445 | 0.670 | 0.622 | 0.982 |
| medium (24) | 0.542 | 0.701 | 0.619 | 0.927 |
| hard (7) | 0.474 | 0.697 | 0.692 | 0.670 |

The NVIDIA Mistral Small 4 judge still works for a one-sample connectivity
check, but it did not complete the full suite reliably. Long outputs truncated
at smaller token ceilings. After that was raised, one upstream 502 put the only
NVIDIA judge connection into cooldown and the remaining jobs failed. I deleted
that partial score rather than reporting a mixed valid/invalid aggregate. The
pinned NVIDIA alias remains useful for small development checks; the full valid
run used the paid release judge.

### Production run difficulty slices

| difficulty | faithfulness | answer relevancy | context precision | context recall |
|---|---:|---:|---:|---:|
| easy | 0.423 | 0.749 | 0.632 | 0.982 |
| medium | 0.550 | 0.748 | 0.570 | 0.941 |
| hard | 0.661 | 0.754 | 0.726 | 0.670 |

Answer relevancy is flat across difficulty. Faithfulness is actually highest on
the seven hard scenarios and lowest on the easy ones — the easy tier is where the
generator most often overreaches slightly beyond the retrieved text. The hard
tier's context recall (0.670) is the weakest retrieval spot, as before.

## Difficulty slices

| retrieval | difficulty | faithfulness | answer relevancy | context precision | context recall |
|---|---|---:|---:|---:|---:|
| dense, no reranker | easy | 0.320 | 0.515 | 0.784 | 0.947 |
| dense, no reranker | medium | 0.330 | 0.540 | 0.663 | 0.817 |
| dense, no reranker | hard | 0.207 | 0.451 | 0.595 | 0.625 |
| hybrid RRF + reranker | easy | 0.271 | 0.366 | 0.680 | 0.772 |
| hybrid RRF + reranker | medium | 0.307 | 0.356 | 0.718 | 0.722 |
| hybrid RRF + reranker | hard | 0.458 | 0.543 | 0.762 | 0.660 |

The seven-item hard slice is too small to settle the retrieval choice by itself.
It is useful as a warning that the score changes by difficulty.

## What the traces show

The deterministic citation validator accepted every generated answer in both
runs: 93 dense attempts and 99 hybrid attempts. The LLM checker rejected 56
dense attempts and 62 hybrid attempts. That caused 44 dense query rewrites and
49 hybrid rewrites. Thirteen dense queries and twenty hybrid queries then ended
with a low-confidence response.

This points at answer grounding and the checker-to-rewriter recovery path, not
at missing retrieval context alone. A checker failure currently changes the
query, often returns similar context, and can fail again. The checked answer is
then replaced with a low-confidence response even when the grader found several
relevant sections.

## 20-scenario node ablation

The full 50-case loops were too expensive to repeat for every node combination.
I instead ran all four variants on the same 20-case stratified random sample:
eight easy, nine medium, and three hard scenarios. The sample uses
`random.Random(20260713)` and contains `s06, s18, s07, s29, s19, s48, s38, s45,
s12, s37, s27, s25, s13, s50, s43, s35, s05, s17, s03, s28`.

Each run used DeepSeek V4 Flash for control and judging, V4 Pro for answers,
dense retrieval without reranking, and the same local corpus. The traces and
score manifests record the model, sample IDs, citations, and answer status.

| pipeline | faithfulness | answer relevancy | context precision | context recall |
|---|---:|---:|---:|---:|
| baseline | **0.433** | **0.718** | 0.737 | 0.796 |
| baseline + grader | 0.426 | 0.714 | **0.844** | 0.823 |
| baseline + grader + checker | 0.186 | 0.310 | 0.789 | 0.794 |
| current full graph | 0.341 | 0.501 | 0.778 | **0.892** |

The variants are:

1. `baseline`: retrieve, generate, then validate citations.
2. `grader`: baseline plus the relevance grader.
3. `checker`: grader plus the faithfulness checker, without query retries.
4. `full`: the existing router, expander, OOD gate, checker, and rewrite loop.

The deterministic citation validator accepted all 20 generated answers in both
the baseline and grader runs. The checker-only path marked 11 of 20 answers
unfaithful and returned low confidence for each. The full graph made 39 answer
attempts, received 25 unfaithful verdicts, rewrote the query 19 times, and still
ended low confidence for six scenarios.

The baseline has the best answer-level scores. The grader has almost the same
faithfulness and relevancy while improving the retrieved context metrics, but it
adds eight Flash calls per query. The full graph reaches more context, but its
extra steps reduce faithfulness and relevancy below the simple baseline.

This is a small, judge-based comparison, so it does not justify a silent default
switch. The next non-paid step is a hand audit of ten saved baseline and full
answers against their cited statute text. Until then, dense baseline is the
preferred production candidate and the checker-rewriter loop remains an
experimental safety path rather than a demonstrated quality improvement.

## Manual audit decision

The ten-answer statute audit is complete in
[manual-answer-audit.md](manual-answer-audit.md). It found five baseline passes
and five partial answers, against three full-graph passes, two partial answers,
and five generic low-confidence failures. One baseline answer misstated the
minimum sentence in BNS 314, so citation membership alone is not a guarantee
that every claim is right. The generator prompt and a key-free regression test
now preserve the BNS 314 bounds and mandatory fine in its context. The live
in-corpus branch now uses scope controls, exact-section lookup, dense retrieval,
generation, and deterministic citation validation. The full graph remains
available only for comparison.
