# RAGAS results

This records the two fresh runs over the 50 hand-labelled scenarios in
`data/eval/scenarios.jsonl` (19 easy, 24 medium, 7 hard). Both runs used the
same current full graph. They are diagnostics, not production-accuracy claims.

## Run setup

- Agent: DeepSeek V4 Flash for routing, expansion, grading, rewriting, and
  checking. DeepSeek V4 Pro wrote the final answer. Thinking was disabled.
- RAGAS judge: DeepSeek V4 Flash at temperature 0 with a 256-token ceiling.
- Answer-relevancy embeddings: local `BAAI/bge-small-en-v1.5`.
- Corpus: the 1,151-chunk local BNS, BNSS, and BSA index after sentence-aware
  chunk repair.
- Scoring: RAGAS `strictness=1` and eight judge workers. The output traces and
  score manifests are local evaluation artifacts and are not committed.

## Overall scores

| retrieval used by the full graph | faithfulness | answer relevancy | context precision | context recall |
|---|---:|---:|---:|---:|
| dense, no reranker | 0.309 | **0.518** | 0.700 | **0.840** |
| hybrid RRF + reranker (current default) | **0.314** | 0.386 | **0.709** | 0.732 |

Dense is the next candidate. Its faithfulness is effectively tied with hybrid,
while answer relevancy is higher by 0.132 and context recall is higher by 0.108.
Hybrid's gains are 0.005 in faithfulness and 0.010 in context precision. Those
small gains do not yet justify keeping its extra retrieval stages.

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
