# RAGAS-50 results

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

## Next comparison

The evaluation runner now has four fixed variants, all using dense retrieval
without reranking:

1. `baseline`: retrieve, generate, then validate citations.
2. `grader`: baseline plus the relevance grader.
3. `checker`: grader plus the faithfulness checker, without query retries.
4. `full`: the existing router, expander, OOD gate, checker, and rewrite loop.

New traces save the structured citations, confidence, and in-corpus flag. That
makes a ten-answer statute audit possible before relying on either Flash-based
judge to decide the production graph.
