# ⚖️ Agentic Legal RAG: Indian Criminal Law (BNS / BNSS / BSA)

> An agentic, self-correcting RAG system for Indian criminal law. Hybrid retrieval, a deterministic citation validator (the anti-hallucination step most systems skip), and dual evaluation (RAGAS diagnostics on the real task plus an AIBE external-comparability number), with observability and an eval gate in CI.

> ⚠️ Statutory information, not legal advice. Not a substitute for a lawyer.

> 🚧 **Status:** scaffolding complete, implementation in progress. See `NOTES.md` for the locked decisions and `../PROJECT.md` for the 4-week build plan. Sections marked _TODO_ fill in as the build lands.

---

## Why this exists

Indian legal RAG is a crowded niche (LexGrid, NYAYA.ai, Legal Assist AI, BNS Mitra, and others). What I didn't see any of them combine is the full stack together: genuinely agentic self-correction, hybrid retrieval, deterministic citation validation, rigorous dual evaluation, and governance from the start. Getting that convergence into one system is the point here.

The 2023 to 2024 IPC/BNS transition also created a live pain point: generalist LLMs still cite *repealed* IPC sections. This system owns the IPC to BNS mapping and answers in the new code.

## Architecture

_TODO: architecture diagram (mirror the node flow in `NOTES.md`)._

```
Query -> Fast Path -> Router -> Intent Expander -> Hybrid Retrieve (BM25+dense+RRF)
      -> Rerank -> OOD Gate -> Grader -> Generator -> Citation Validator -> Checker -> Answer
```

Self-correction loop budget = 2.

## Key features

- **Deterministic citation validator:** every cited `[Section, Act]` is verified to exist in the retrieved set (pure code, not an LLM). This is the part I think sets it apart, and it's what drives the self-correction loop.
- **Exact-section fast path:** `"BNS 103"` / `"302 IPC"` resolve via direct metadata lookup in <50ms, IPC refs bridged to BNS.
- **Hybrid retrieval:** BM25 + dense + RRF (k=60), cross-encoder reranker on by default.
- **Intent expansion:** one messy narrative into parallel offence sub-queries (cross-sectional reasoning).
- **Auditable by design:** every answer carries citations + a LangSmith trace URL.

## Competitor comparison

_TODO: comparison table (from `../analysis.md` §2)._

## Evaluation

_TODO: fill from `notebooks/03_eval_dashboard.ipynb`._

- **RAGAS (real generative task):** faithfulness / answer-relevancy / context-precision / context-recall on 50 hand-labeled scenarios.
- **AIBE (external comparability, _with caveats_):** criminal-slice accuracy vs a no-RAG baseline. Reported honestly: AIBE 4-16 predate 2024 and cite repealed IPC, so this partly measures the IPC to BNS bridge (reported separately). It's not a bare bar-exam score. See the "AIBE reality check" note in `NOTES.md`.
- **Ablations:** hybrid vs dense/sparse; reranker on/off.

## Quickstart

_TODO: verify once implemented._

```bash
cp .env.example .env        # fill in GEMINI_API_KEY, LANGSMITH_API_KEY, HF_TOKEN
pip install -e ".[dev]"     # or: uv sync
# 1. add BNS/BNSS/BSA PDFs to data/raw/  (see Data & licensing)
# 2. build the index:  python -m src.retrieval.index
# 3. run the stack:     docker compose up --build   # qdrant + api + frontend
```
API -> `http://localhost:8000` · Frontend -> `http://localhost:8501`

## Data & licensing

- **Corpus:** BNS / BNSS / BSA gazette PDFs into `data/raw/` (not committed; document sourcing here).
- **Eval datasets** (gated, need `HF_TOKEN`):
  - `opennyaiorg/aibe_dataset`: AIBE 4-16, **CC BY-ND-4.0**, evaluation-only (no redistribution of modified copies).
  - `bharatgenai/BhashaBench-Legal`: **CC BY-4.0**, criminal-law slice used for robustness.

## Governance & security

- **Auditable by design:** structured citations + LangSmith trace per answer.
- ⚠️ **No auth on the API.** Fine for a local demo, but it must sit behind an API key/gateway before any public/cloud deploy.

## Project layout

See `NOTES.md` for the annotated tree and the coding rules.

## License

MIT (code). Eval datasets retain their own licenses (see above).
