# Session Audit — 2026-07-12

## Scope

Audited `agentic-legal-rag` against five public Indian legal-AI repositories:

- [BNS-LexAI](https://github.com/adityapradhan202/BNS-LexAI/tree/0abf3e5)
- [CRAG-BNS](https://github.com/kaustubha-chaturvedi/CRAG-BNS/tree/ec1fd35)
- [Nyaya-GPT](https://github.com/Debapriya-source/nyaya-gpt/tree/163bf69)
- [VidhAI](https://github.com/nakul-krishnakumar/vidh-ai/tree/2bdf4a4)
- [Bhartiya Nyay AI](https://github.com/anshika-codes-AI/bhartiya-nyay-ai/tree/ebf44f0)

The comparison read their current repository trees and primary implementation files, not only their README claims.

## Verification

- Offline test suite: 190 passed; 3 source-data-dependent tests skipped.
- `ruff check .`: passed.
- No project implementation changes were made during the audit.

## Comparative assessment

| Repository | Useful idea to learn from | Where this project is stronger |
|---|---|---|
| BNS-LexAI | FastAPI/Streamlit packaging and a small task-specific retrieval comparison set. | Real legal-data ingestion, hybrid retrieval, deterministic citation validation, and broader labelled evaluation. |
| CRAG-BNS | A compact retrieve → grade → rewrite example. | Reproducible corpus pipeline, legal scope control, hybrid retrieval, and a bounded correction loop. |
| Nyaya-GPT | A working ReAct UI with cloud/local-model support and separate BNS/Constitution sources. | Structured output, deterministic citation checks, corpus provenance, and evaluation discipline. |
| VidhAI | Clear API/UI separation and a source-refresh ingestion workflow. | Section-aware corpus, hybrid retrieval, local reproducibility, and formal test coverage. |
| Bhartiya Nyay AI | Strong advocate-facing workflow: structured facts, legal mapping, review, and DOCX output. | The RAG/LLM core is actually implemented here; their current LLM call is mocked and their retrieval is section/keyword filtering. |

## Strengths to preserve

1. Statute-aware parser with published section-count gates.
2. Hybrid BM25 + dense retrieval with RRF and measured reranker trade-offs.
3. Exact-section fast path and IPC → BNS bridge.
4. Deterministic cited-section validator before LLM faithfulness checking.
5. Keyless unit tests through injected LLM/retrieval clients.
6. A 50-scenario, section-labelled evaluation set and an external BhashaBench-Legal MCQ harness.

These make the project technically stronger than the comparison projects on paper. They are not yet proof of better user-facing accuracy; final system-level results must be published honestly.

## Current release blockers

1. `POST /query`, `GET /health`, and the Streamlit UI remain stubs.
2. The CI eval-gate invokes an unimplemented smoke-evaluation module.
3. The README still presents the old AIBE framing, although implementation moved to BhashaBench-Legal.
4. The quickstart cannot yet build the corpus/index from a clone: `python -m src.retrieval.index` has no CLI entrypoint, and raw/processed artifacts are intentionally ignored.
5. The planning documents were ignored by `*.md`; this document is deliberately placed in `docs/` so it can be tracked and published.

## Complexity audit

Keep the custom parser, hybrid retrieval, fast path, citation validator, and tests. They are earned complexity.

Defer judgement on the remaining LLM nodes until an ablation measures their effect. The 12-node graph, eight-way LLM grading fan-out, and alternate Kiro/Claude provider are defensible only if they improve quality or safety enough to justify their latency and operational cost.

`cognizable`, `bailable`, and `offence_category` metadata are currently enriched but not consumed after ingestion. Keep the IPC → BNS mapping; only retain the other fields once they are surfaced in answers or evaluation.

## Agreed sequence

**Finish → measure → simplify.**

1. Complete the original Week 4 path: API, health endpoint, Streamlit UI, Docker quickstart, and truthful README.
2. Make CI truthful: either implement the regression gate with a committed baseline or remove that job until it is real.
3. Run and publish a named-model evaluation table: dense vs hybrid vs reranked vs full agent, including latency and provider.
4. Only then remove or defer agent nodes that show no measurable lift.

Do not remove the deterministic citation validator, hybrid retrieval, fast path, or evaluation harness before measurement. No additional agent infrastructure should be added before the release blockers and evaluations are complete.

## Groq feasibility assessment

The Groq model catalogue was reviewed against the current agent graph and recorded test/evaluation constraints. Groq is feasible as an alternate evaluation and inference provider, but is not a direct solution to the full graph's quota problem.

| Model | Verdict | Recommended role |
|---|---|---|
| `openai/gpt-oss-20b` | Best first pilot. | Flash-tier structured nodes: router, intent expander, rewriter, and checker. |
| `openai/gpt-oss-120b` | Good final-generator benchmark. | Final answer generation only; do not use it for every grading call. |
| `llama-3.1-8b-instant` | Do not adopt. | Groq has announced its 16 August 2026 deprecation. |
| `llama-3.3-70b-versatile` | Do not adopt. | Groq has announced its 16 August 2026 deprecation. |
| `qwen/qwen3-32b` | Do not adopt. | Groq has announced its 17 July 2026 deprecation. |
| `qwen/qwen3.6-27b` | Defer. | Preview model; not a stable default for this portfolio project. |
| Prompt Guard 22M / 86M | Do not add. | Safety classifiers, not legal-answer models; an extra call does not fix grounding. |
| `groq/compound` | Do not add. | Web-search orientation conflicts with the statute-only, auditable-source boundary. |

GPT-OSS 20B and 120B are the two Groq models that support strict structured outputs, making them the cleanest fit for Pydantic/instructor output. See [Groq Structured Outputs](https://console.groq.com/docs/structured-outputs?form=MG0AV3).

### Quota implication

The current normal path makes roughly 12 LLM calls: router, intent expansion, eight independent grader calls, generator, and checker. Two correction loops can raise that to roughly 34 calls.

The eight concurrent grader prompts each include up to 4,000 characters of statute text. The shown free limits for GPT-OSS 20B and 120B are 30 RPM, 1,000 RPD, 8K TPM, and 200K TPD. Therefore the grader can exceed the token-per-minute limit even when request count remains below 30 RPM; a full 50-scenario run will likely exceed the daily token allowance without batching or a paid tier. See [Groq Rate Limits](https://console.groq.com/docs/rate-limits).

### Recommended pilot

```text
flash tier:     openai/gpt-oss-20b
generator tier: openai/gpt-oss-120b
evaluation:     5–10 representative scenarios before any full-suite run
```

Record all results as Groq/GPT-OSS results; do not combine or relabel them as Gemini results. Before a full 50-scenario Groq evaluation, pace or batch grader calls and reduce the per-chunk context cap. GPT-OSS automatic prompt caching can reduce repeated-prefix input cost and limit use, but it will not make eight large, distinct grader prompts free. See [Groq prompt caching](https://console.groq.com/docs/changelog).

### Live structured-output smoke test

On 2026-07-12, one live `openai/gpt-oss-120b` call was made through Groq's OpenAI-compatible endpoint. Authentication and connectivity succeeded.

The first strict-schema attempt used `max_completion_tokens=80` and returned Groq's `json_validate_failed` error. Retrying with `reasoning_effort="low"` and `max_completion_tokens=512` returned valid strict-schema JSON:

```json
{"route":"criminal","confidence":"high"}
```

This proves basic strict-schema compatibility for the router-shaped response only. It is not a legal-quality benchmark or evidence that the full 12-call agent graph fits Groq's quota. The test key was entered interactively and was not saved to the repository or `.env`.

### Mini project-path test

A second live smoke test used the local project functions with `openai/gpt-oss-120b` and the real BNS Section 303 corpus chunk. It exercised three LLM nodes and the deterministic citation validator for the query:

> Someone took my bicycle without permission. What provision applies?

Observed result:

```json
{
  "route": "criminal",
  "sub_queries": ["theft of movable property (bicycle) provision"],
  "citations": [{"act": "BNS", "section_id": "303"}],
  "citation_valid": true,
  "invalid_citations": []
}
```

The generated answer correctly described theft of movable property without consent and cited only the supplied BNS 303 chunk. This is a three-call smoke test for router, intent expansion, and generation; it does not test the real retriever, OOD gate, grader, correction loops, or answer quality over the evaluation set. The key was entered interactively and was not saved.

## Cerebras feasibility assessment

Cerebras is a legitimate official API candidate for the same `gpt-oss-120b` model family. It supports strict JSON-schema output, so it is compatible with the project’s Pydantic/instructor node design. See [Cerebras Structured Outputs](https://inference-docs.cerebras.ai/capabilities/structured-outputs).

### Live project-path smoke test

On 2026-07-12, the exact router → intent-expander → generator → deterministic-validator smoke test was run through Cerebras’s OpenAI-compatible endpoint using `gpt-oss-120b`. It returned:

```json
{
  "route": "criminal",
  "sub_queries": ["theft of movable property"],
  "citations": [{"act": "BNS", "section_id": "303"}],
  "citation_valid": true,
  "invalid_citations": []
}
```

The answer correctly identified BNS 303 and passed deterministic citation validation. The test key was entered interactively and was not saved.

### Account-specific limit correction

The account dashboard, which takes precedence over published general limits, shows:

```text
gpt-oss-120b: 5 RPM, 150 requests/hour, 2,400 requests/day
                 30K TPM, 1M tokens/hour, 1M tokens/day
```

Cerebras therefore solves the Groq 8K TPM bottleneck, but not the current graph’s eight-way concurrent grader burst: eight simultaneous grades exceed the account’s 5 RPM cap. The full graph must use a shared request limiter, batch grading, or a smaller grading set before a full 50-scenario evaluation. The three-call smoke test is valid; it is not evidence that the unchanged full graph fits the account quota.
