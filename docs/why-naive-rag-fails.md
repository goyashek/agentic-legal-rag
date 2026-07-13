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
That does not make retrieval perfect: on the 50-scenario set, reranking improved Recall@5 from
0.550 to 0.653 while reducing MRR from 0.500 to 0.413. In legal retrieval, finding the full
set of possibly relevant sections can matter more than making one result rank first, but the
trade-off should be reported rather than hidden.

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

## Refusing an answer can be the correct result

The three-scenario RAGAS diagnostic exposed a concrete corpus problem. BNS section 303 is long
enough that its base-punishment clause is separated from related material at a chunk boundary.
When the generator tried to fill in the missing clause, the grounding check rejected the answer
and the pipeline returned low confidence instead.

That produced a faithfulness score of 0.0 in that small diagnostic. It is not evidence that the
system silently emitted a bad answer. It is evidence that the guardrail stopped one. The proper
fix is to repair the chunking and re-index the corpus, not to weaken the check or invent a more
flattering metric.

## What the current results do and do not show

The project has a section-labelled 50-scenario retrieval set, a three-scenario RAGAS diagnostic,
and a directional 60-question BhashaBench-Legal comparison. Those are useful signals, not a
claim of production legal accuracy. The full RAGAS-50 run and a captured dense-versus-hybrid
ablation are still pending.

The takeaway is deliberately small: retrieval quality, citation validity, and claim grounding
are separate problems. A system that treats them as one prompt is harder to audit when it fails.
