# RAGAS-50 results

This is the first complete run over the 50 hand-labelled scenarios in
`data/eval/scenarios.jsonl` (19 easy, 24 medium, 7 hard). It is a baseline, not a
production-accuracy claim.

## Run setup

- Agent: DeepSeek V4 Flash for routing, expansion, grading, rewriting, and checking;
  DeepSeek V4 Pro for the final answer. Thinking was disabled for these bounded calls.
- RAGAS judge: DeepSeek V4 Flash, `temperature=0`, 256 completion-token ceiling.
- Answer-relevancy embeddings: local `BAAI/bge-small-en-v1.5`.
- Corpus: rebuilt 1,151-chunk local BNS / BNSS / BSA index after sentence-aware chunk repair.
- Scoring: RAGAS `strictness=1`, one serial worker. The completed run used the embedding
  compatibility repair in commit `40278d1`.

## Overall scores

| metric | score |
|---|---:|
| faithfulness | 0.262 |
| answer relevancy | 0.419 |
| context precision | 0.671 |
| context recall | 0.722 |

## Difficulty slices

| difficulty | faithfulness | answer relevancy | context precision | context recall |
|---|---:|---:|---:|---:|
| easy | 0.21 | 0.42 | 0.70 | 0.76 |
| medium | 0.31 | 0.42 | 0.68 | 0.67 |
| hard | 0.25 | 0.41 | 0.56 | 0.80 |

## Reading the result

Retrieval coverage is materially better than answer quality: context recall is 0.722, while
faithfulness and answer relevancy are low. The earlier BNS 303 split-clause problem is fixed,
but this run shows that the remaining grounding and answer-quality gaps are broader than that
one section. These values are the reason the project is still a local demo, not a legal-advice
service.

The scorer completed all 200 metric jobs. An earlier incomplete attempt exposed an
`embed_query` interface mismatch and was stopped rather than reported; no partial score from it
is used here.
