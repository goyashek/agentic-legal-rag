# Why naive RAG fails on Indian criminal-law text

A legal answer can look convincing and still be unsafe. A model may name a real statute,
use a plausible section number, and write a clear explanation while relying on the wrong
retrieved text. This project is built around making that failure visible instead of smoothing it
over with a nicer prompt.

## Exact questions and narrative questions are different jobs

For an explicit request such as `BNS 103` or `302 IPC`, retrieval is unnecessary. The system
uses a deterministic metadata lookup and maps an IPC reference to its BNS equivalent when the
mapping exists. It avoids both embedding drift and an LLM call.

Most questions are harder. “Someone took my bicycle without permission” does not name the
legal term or section. The pipeline expands the narrative into offence-focused sub-queries,
combines BM25 and dense retrieval with reciprocal-rank fusion, then reranks the candidates.
That does not make retrieval perfect. On the rebuilt 50-scenario set, BM25 alone reaches only
0.330 Recall@5. Dense-only reaches 0.750 Recall@5 and 0.706 MRR; the older hybrid + reranker
agent reached 0.630 and 0.422. Dense without a reranker won that retrieval comparison and is now
the live path.

## A citation format is not citation validation

Telling a model to include citations only changes the shape of its answer. It does not prove
that a cited section came from the supplied sources.

This project checks that in code. Before an answer is returned, every cited `(act, section)` is
normalized to section level and matched against the retrieved chunks. The regression test uses
a deliberately bad answer that cites BNS 307 when the retrieved context contains only BNS 306.
The validator rejects BNS 307, so the graph either rewrites and retrieves again or ends with a
low-confidence response after its two-attempt budget is spent.

That check has a narrow but useful contract. It proves that the answer did not cite a section
that retrieval never supplied. It cannot prove that every sentence about a valid section is
correct, which is why a separate grounding check follows it.

## One fixed failure does not prove the system is ready

An early three-scenario diagnostic exposed a concrete corpus problem. BNS section 303 was split
into 18 fragments, with its base-punishment sentence cut at a chunk boundary. The grounding
check correctly rejected an answer that tried to fill in the missing text.

I fixed the cause in the shared chunker rather than making a BNS-303 exception. It now joins
semantic fragments into complete sentences before repacking them into the 512-token budget. BNS
303 is four chunks, and the base-punishment sentence now stays in one of them.

That repair matters, but it does not by itself make the answer path ready. The guardrail
ablation has since been run. The current production path (dense retrieval, generation, and
deterministic citation validation, without the checker or rewrite loop) scores 0.517 for
faithfulness and 0.749 for answer relevancy over the 50 scenarios, with context recall at 0.919.
The older full self-correcting graph scored 0.309 and 0.518 on the same set: its checker kept
rejecting citation-valid answers and the rewrite loop could not recover them, so 13 to 20
scenarios ended in a canned low-confidence reply that scores as zero. Removing those two stages
nearly doubled faithfulness and answer relevancy. Faithfulness at 0.517 is still middling, so the
answer path is better but not production-grade.

## What the current results do and do not show

The project has a section-labelled 50-scenario retrieval set, a complete RAGAS-50 baseline, and
a directional 60-question BhashaBench-Legal comparison. Those are useful signals, not a claim
of production legal accuracy. Dense, sparse, hybrid, and reranked retrieval paths are captured.
On a fixed random 20-scenario RAGAS ablation, dense retrieval plus generation and deterministic
citation validation scored better for faithfulness and relevancy than the fuller graph. The
grader improved context metrics, while the checker and rewrite loop did not improve final answer
quality. A ten-answer statute audit agreed, so the live in-corpus path now uses dense retrieval,
generation, and deterministic citation validation while keeping the scope controls.

The takeaway is deliberately small: retrieval quality, citation validity, and claim grounding
are separate problems. A system that treats them as one prompt is harder to audit when it fails.
