# Project Summary & Methodology Report: BNS Agentic RAG

Here is the comprehensive summary of our findings, empirical test results, and next steps for our Indian Criminal Law RAG project.

---

## 1. 4 Obvious Mistakes We Were Making

1.  **Running a Suboptimal Retrieval Default:**
    We defaulted to **Hybrid retrieval (Dense + BM25) + Reranking**, even though our ablation results show it underperforms simple dense-only search. BM25 adds lexical noise due to the vocabulary mismatch between narrative questions and formal statutory text.
2.  **Using Agent Loops as a Bandage for Retrieval Failures:**
    Our RAGAS-50 faithfulness was low (0.262) because retrieval missed base statutory clauses (like BNS 303 base punishment). When the LLM hallucinated, the checker caught it, looped, and ultimately bailed with a canned low-confidence response, which RAGAS scores as 0.0.
3.  **Ignoring Enriched Legal Metadata:**
    Our ingestion pipeline enrichments (`bailable`, `cognizable`, `offence_category`) are not consumed in the active graph. We are wasting token costs querying LLMs for properties we already have indexed.
4.  **Semantic Chunking over Statutory Sections:**
    Using vector similarity thresholds to chunk statutes shreds cohesive sections mid-sentence. When the base punishment is separated from its context, the generator is forced to bail or hallucinate.

---

## 2. Empirical Test Results (N=10 Scenarios)

We ran optimized retrieval and latency benchmarks in our scratch workspace to test these claims:

| Retrieval Configuration | P@5 | Recall@5 | MRR |
| :--- | :---: | :---: | :---: |
| 🌟 **dense only** (No Rerank) | **0.200** | **0.717** | **0.723** |
| **hybrid only** (No Rerank) | 0.140 | 0.483 | 0.569 |
| **sparse only** (BM25 only) | 0.060 | 0.200 | 0.273 |
| **dense + reranker** | 0.160 | 0.567 | 0.600 |
| **hybrid + reranker** (Our Old Default) | 0.140 | 0.467 | 0.408 |
| **sparse + reranker** | 0.100 | 0.400 | 0.325 |

### Latency Benchmark
*   **Exact-Section Fast-Path average latency:** **10.35 ms** (3/3 queries resolved correctly, hitting BNS equivalents for old IPC sections).

> [!TIP]
> **Dense-only search is the clear winner.** Adding BM25 dropped absolute Recall@5 by **23.4%**, and adding the uncalibrated reranker (`bge-reranker-base`) dropped dense MRR by **12.3%**.

---

## 3. The Evaluation Set Bottleneck

Our evaluation sets present two major issues:
*   **RAGAS-50:** Small size (50 scenarios) makes metrics noisy, and hand-crafting them introduces author bias / prompt overfitting.
*   **BhashaBench-Legal:** MCQ format introduces a task mismatch for our generative system. Because we only index BNS (penal code) but the dataset tests CrPC (procedure), our system gets a **negative RAG lift (-0.125)** due to retrieving irrelevant penal sections for procedural questions.

---

## 4. Next Steps & Open Decisions

| Priority | Recommendation | Impact |
| :--- | :--- | :--- |
| **P0** | **Switch retriever default to Dense-Only** | Instantly raises Recall@5 by ~20%+ and MRR by ~30%+. |
| **P0** | **Statutory Chunking** | Keep sections completely whole (no semantic splitting) to preserve base punishments in a single chunk. |
| **P1** | **Ingest BNSS & BSA Acts** | Solves the BhashaBench negative RAG lift by providing procedural and evidence context. |
| **P1** | **Deterministic Metadata Lookup** | Route queries asking about bailable/cognizable status directly to metadata database queries in `fast_path.py` (<15ms, 100% accuracy). |
| **P2** | **LexRAG Multi-Turn Pivot** | Pivot to multi-turn conversation states to evaluate real-world legal assistant scenarios. |
