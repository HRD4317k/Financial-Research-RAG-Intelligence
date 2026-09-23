# Financial Research Agent

A multi-step AI research agent built with LangGraph that retrieves SEC filings and financial news via a RAG pipeline, reasons over them, and produces structured investment theses — with a full evaluation suite.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![LangGraph](https://img.shields.io/badge/built%20with-LangGraph-1C3C3C) ![Eval](https://img.shields.io/badge/evaluated%20with-RAGAS%20%2B%20DeepEval-purple)

---

## Overview

Most LLM demos stop at "ask a question, get an answer." This project builds a proper agentic system: the agent plans, retrieves, reasons across multiple steps, and produces structured output. Every component — retrieval quality, reasoning accuracy, and output faithfulness — is evaluated with real metrics.

**Scope:** Apple (AAPL), Microsoft (MSFT), and Nvidia (NVDA) — 10-K and 10-Q filings + earnings call transcripts (8-K exhibits). 15 hand-curated golden Q&A pairs for regression testing.

---

## What This Project Does

**Agent Architecture (LangGraph)**
- Multi-node graph: Planner → Retriever → Analyst → Synthesizer → Evaluator
- Conditional edges: re-retrieval triggered when Analyst confidence score falls below threshold (max 3 iterations)
- Human-in-the-loop checkpoint at Synthesizer node
- Full state tracking across the graph via typed `AgentState`

**RAG Pipeline**
- Document ingestion: SEC 10-K/10-Q filings via `sec-edgar-downloader`; earnings call transcripts via EDGAR 8-K exhibits
- Section-aware chunking (splits by SEC filing section headers, not arbitrary character count)
- Dense index: FAISS with `BAAI/bge-large-en-v1.5` embeddings
- Sparse index: BM25 via `rank-bm25`
- Reranking: `BAAI/bge-reranker-large` cross-encoder (local, no API cost)
- Source citations included in all outputs

**Agent Nodes**
| Node | Input | Output |
|------|-------|--------|
| Planner | User question | Decomposed sub-queries + target sections |
| Retriever | Sub-queries | Retrieved chunks with scores |
| Analyst | Chunks | Structured analysis + confidence score |
| Synthesizer | Analysis | Investment thesis (JSON) |
| Evaluator | Thesis + ground truth | RAGAS + DeepEval scores |

**Evaluation**
- **RAGAS** — measures retrieval quality: context recall, faithfulness, answer relevance
- **DeepEval LLM-as-Judge** — measures reasoning quality: multi-step coherence, factual grounding
- Golden dataset: 15 hand-curated Q&A pairs in `evals/golden_dataset.json`

**Cost note:** RAGAS + DeepEval make LLM calls per metric per question. Use `gpt-4o-mini` for evaluation, `gpt-4o` for the agent's Synthesizer node only.

---

## Stack

| Component | Tool |
|-----------|------|
| Agent Orchestration | LangGraph |
| LLM (default) | OpenAI GPT-4o — swap via `LLM_PROVIDER=gemini` env var |
| RAG | LangChain, FAISS, BM25 (`rank-bm25`) |
| Embeddings | `BAAI/bge-large-en-v1.5` (HuggingFace, local) |
| Reranker | `BAAI/bge-reranker-large` (cross-encoder, local) |
| Evaluation | RAGAS, DeepEval |
| Observability | LangSmith |
| UI | Streamlit |
| Environment | Python 3.11 |

---

## API Keys & Environment

```bash
# .env.example
OPENAI_API_KEY=                        # required — GPT-4o for synthesis
LANGCHAIN_API_KEY=                     # required — LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=financial-research-agent
SEC_EDGAR_USER_AGENT="YourName your@email.com"   # required — EDGAR 403s without this
GEMINI_API_KEY=                        # optional — alternative LLM backend
```

| Credential | Cost | Free Tier |
|-----------|------|-----------|
| OpenAI | Pay-per-token | $5 credit on signup |
| LangSmith | Free tier available | 5K traces/month |
| SEC EDGAR | Free | No key needed, only User-Agent header |
| Embeddings/Reranker | Free | Local models, no API |

---

## Quickstart

```bash
git clone https://github.com/smadinen7/financial-research-agent
cd financial-research-agent
cp .env.example .env          # fill in your keys

pip install -r requirements.txt

# Step 1 — Ingest SEC filings (downloads + chunks 10-K/10-Q for AAPL, MSFT, NVDA)
python src/ingest.py --tickers AAPL MSFT NVDA --filing-types 10-K 10-Q

# Step 2 — Build retriever index (FAISS + BM25; done once, cached to data/retriever_index/)
python src/retriever.py --build

# Step 3 — Run the agent (CLI, streaming mode)
python src/graph.py "What are Apple's main supply chain risks?" --tickers AAPL --stream

# Step 4 — Launch the Streamlit UI
streamlit run app/main.py

# Step 5 — Run evaluation suite (dry-run first to validate, then full run)
python src/evaluate.py --golden-dataset evals/golden_dataset.json --dry-run
python src/evaluate.py --golden-dataset evals/golden_dataset.json
```

---

## Project Structure

```
financial-research-agent/
├── data/
│   ├── filings/            # Ingested SEC documents (gitignored)
│   └── retriever_index/    # Saved FAISS + BM25 index (gitignored)
├── src/
│   ├── state.py            # AgentState TypedDict (shared across all nodes)
│   ├── ingest.py           # SEC EDGAR download + section-aware chunking
│   ├── retriever.py        # FAISS + BM25 + RRF + cross-encoder reranking
│   ├── nodes.py            # Planner, Retriever, Analyst, Synthesizer, Evaluator nodes
│   ├── graph.py            # LangGraph agent graph + run_agent() / stream_agent()
│   ├── llm.py              # Multi-provider LLM factory (Gemini / OpenAI / Claude)
│   └── evaluate.py         # RAGAS + DeepEval evaluation suite
├── app/
│   └── main.py             # Streamlit UI with streaming agent display
├── evals/
│   ├── golden_dataset.json # 15 hand-curated Q&A pairs (AAPL / MSFT / NVDA)
│   └── results/            # Eval run outputs as JSON (gitignored)
├── tasks/
│   ├── todo.md             # Phase-by-phase task tracker
│   └── lessons.md          # Self-improvement lessons log
├── .env.example
├── requirements.txt
└── README.md
```

---

## Agent Graph

```
[Planner] → [Retriever] → [Analyst] → [Synthesizer] → [Evaluator]
                 ↑               |
                 └── (conf < 0.7, max 3 iterations)
                       ↑
                 (gaps from analyst fed back as augmented sub-queries)
```

**Node details:**
| Node | What it does |
|------|-------------|
| Planner | Decomposes the user question into 3 targeted sub-queries + target SEC sections |
| Retriever | FAISS + BM25 + RRF + cross-encoder; on re-retrieval, augments with analyst gaps |
| Analyst | Reads up to 10 chunks, produces structured findings + 0.0–1.0 confidence score |
| Synthesizer | Writes investment thesis JSON (BUY/HOLD/SELL + bull/bear case + key risks) |
| Evaluator | Scores thesis with DeepEval when ground truth is provided (no-op otherwise) |

---

## Golden Dataset Format

```json
{
  "question": "What were Apple's primary risk factors related to supply chain in their 2023 10-K?",
  "ground_truth": "Apple cited concentration risk in Asia-Pacific manufacturing...",
  "source_filing": "AAPL-10K-2023",
  "section": "Risk Factors",
  "question_type": "factual_retrieval"
}
```

---

## Key Results

*In progress — target metrics below. Results will be updated as experiments complete.*

| Metric | Target | Actual |
|--------|--------|--------|
| Context Recall (RAGAS) | > 0.75 | — |
| Faithfulness (RAGAS) | > 0.80 | — |
| Answer Relevance (RAGAS) | > 0.78 | — |
| Reasoning Quality (DeepEval) | > 0.75 | — |

---

## Limitations

- SEC EDGAR 10-K HTML parsing can produce noisy chunks around financial tables — section-aware chunking mitigates but does not eliminate this
- Evaluation costs accumulate quickly; full eval suite (~15 questions × 4 metrics) triggers ~60 LLM calls
- Agent is scoped to 3 tickers and 2 filing types; generalization to other companies requires re-ingestion
