# Financial Research Agent — Task Tracker

## Phase 1: Foundation ✅
- [x] Create `requirements.txt` (all deps: langgraph, langchain, faiss, edgartools, ragas, deepeval, streamlit)
- [x] Create `.env.example` (Gemini default, OpenAI + Claude optional, eval LLM separate)
- [x] Create `src/llm.py` — multi-provider LLM factory (gemini | openai | claude)
- [x] Verify: import, bad-provider error, missing-key error all behave correctly
- [x] Commit + push `feature/foundation`

## Phase 2: Ingest ✅
- [x] `src/ingest.py` — download 10-K/10-Q for AAPL/MSFT/NVDA via `edgartools`
- [x] Section-aware chunking (Item 1A, Item 7, Item 8 for 10-K; Part I/II items for 10-Q)
- [x] Prepend metadata header to each chunk: `[Company | Form | Period | Section]`
- [x] Save per-ticker JSON + consolidated `index.json` to `data/filings/`
- [x] `.gitignore` added (excludes data/filings/, .env, evals/results/)
- [x] Structural tests: chunk IDs unique, metadata headers correct, error paths clean
- [ ] Live smoke test: `python src/ingest.py --tickers AAPL --filing-types 10-K` (needs packages installed + SEC_EDGAR_USER_AGENT set)
- [ ] Commit + push + merge `feature/ingest` PR

## Phase 3: Retriever ✅
- [x] `src/state.py` — AgentState TypedDict (shared across all nodes)
- [x] `src/retriever.py` — FAISS dense index (`bge-large-en-v1.5` embeddings)
- [x] BM25 sparse index (`rank-bm25`) — BM25Okapi
- [x] RRF (Reciprocal Rank Fusion) hybrid merge
- [x] Cross-encoder rerank with `bge-reranker-large`
- [x] Ticker / filing-type pre-filtering
- [x] `save()` / `load()` for persisting index to disk
- [ ] Live smoke test: build index from real ingest output (needs API keys + packages)

## Phase 4: Agent Graph ✅
- [x] `src/nodes.py` — Planner, Retriever, Analyst, Synthesizer, Evaluator nodes
- [x] `src/graph.py` — TypedDict `AgentState`, LangGraph wiring, conditional re-retrieval loop
- [x] Conditional edge: confidence < 0.7 && iterations < 3 → back to Retriever
- [x] Gap-aware re-retrieval: Retriever augments queries with Analyst-identified gaps
- [x] Human-in-the-loop checkpoint before Synthesizer (optional via flag)
- [x] `run_agent()` blocking API + `stream_agent()` streaming generator
- [x] CLI: `python src/graph.py "What are Apple's risks?"`
- [ ] Live smoke test: full trace visible in LangSmith (needs keys + ingest data)

## Phase 5: Evaluation ✅
- [x] `evals/golden_dataset.json` — 15 curated Q&A pairs (5 per ticker; factual + comparative)
- [x] `src/evaluate.py` — RAGAS runner (context_recall, faithfulness, answer_relevancy)
- [x] DeepEval: AnswerRelevancy, Faithfulness, HallucinationMetric
- [x] `--dry-run` flag to validate pipeline without LLM calls
- [x] `--tickers` filter for running subset of questions
- [x] Visual score bars + console summary table
- [x] Timestamped JSON results saved to `evals/results/`
- [ ] Run full eval suite and populate README results table (needs ingest + API keys)

## Phase 6: UI ✅
- [x] `app/main.py` — Streamlit: query input, ticker filter sidebar
- [x] Streaming agent reasoning display (node-by-node progress with status badges)
- [x] Investment thesis card (rating, bull/bear case, key risks)
- [x] Analyst findings tab (confidence meter, per-finding expanders)
- [x] Source citations panel (chunk text + filing metadata, grouped by ticker)
- [x] Eval scores tab (when ground truth provided)
- [x] Graceful error handling for missing ingest data
- [ ] Smoke test: `streamlit run app/main.py` (needs ingest data for full flow)

---

## Review Notes

### Phase 1 — Foundation (complete)
- LLM factory refactored to data-driven `_PROVIDERS` registry after /simplify review
- `load_dotenv()` moved to lazy `_load_env()` guard to avoid disk I/O on every import
- Temperature now configurable via `LLM_TEMPERATURE` env var
- PR #1 merged to main

### Phase 2 — Ingest (branch ready, pending live test)
- `src/ingest.py` uses `edgartools` Company API; section-aware extraction via `doc[section_key]`
- `_extract_section_text` handles edgartools returning Section objects, strings, or None
- All structural unit tests pass on Python 3.9
- Pending: install `requirements.txt` and run live smoke test against real EDGAR

### Phase 3 — Retriever (complete)
- `HybridRetriever`: FAISS IndexFlatIP (cosine on normalised embeddings) + BM25Okapi + RRF + BGE cross-encoder
- Ticker / filing-type pre-filtering applied before FAISS search (candidate mask)
- Index save/load via faiss.write_index + pickle for BM25 + JSON for chunks

### Phase 4 — Agent Graph (complete)
- 5-node LangGraph graph: Planner → Retriever → Analyst → (conditional) → Synthesizer → Evaluator
- All nodes return partial state dicts; LangGraph merges automatically
- Gap-aware re-retrieval: on iteration > 0, Retriever augments sub-queries with Analyst-identified gaps
- `MemorySaver` checkpointer enables thread-scoped state persistence and HITL

### Phase 5 — Evaluation (complete)
- 15 golden Q&A pairs: realistic ground truths based on well-known public filing information
- `--dry-run` validates the full pipeline structure without any LLM API calls
- Per-metric error isolation: one failing metric doesn't break the others

### Phase 6 — UI (complete)
- Streamlit `stream_agent()` integration: live node badges update as graph executes
- 4-tab layout: Investment Thesis / Analyst Findings / Source Citations / Eval Scores
- Confidence meter with colour coding (green >= 70%, amber >= 40%, red < 40%)
