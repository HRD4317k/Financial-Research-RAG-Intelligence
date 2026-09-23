"""
LangGraph node implementations for the Financial Research Agent.

Each node is a pure function: (AgentState) -> dict of state updates.
LangGraph merges the returned dict into the running AgentState automatically.

Nodes:
  planner_node    — Decomposes user question into 3 targeted sub-queries
  retriever_node  — Runs hybrid retrieval for each sub-query (deduped + ranked)
  analyst_node    — Analyses chunks, produces structured findings + confidence score
  synthesizer_node — Synthesises findings into a structured investment thesis JSON
  evaluator_node  — Optionally scores the thesis with DeepEval (when ground truth supplied)
"""

import json
import os
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import get_llm
from src.retriever import HybridRetriever, RetrievedChunk
from src.state import AgentState


# ── Module-level retriever (lazy singleton) ───────────────────────────────────

_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    """Return the shared HybridRetriever instance, initialising it on first call."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
        index_dir = os.getenv("RETRIEVER_INDEX_DIR", "data/retriever_index")
        ingest_index = os.getenv("INGEST_INDEX_PATH", "data/filings/index.json")

        if os.path.isdir(index_dir) and os.path.exists(os.path.join(index_dir, "faiss.index")):
            _retriever.load(index_dir)
        elif os.path.exists(ingest_index):
            print("[Agent] Pre-built index not found — building from ingest output…")
            _retriever.build_from_ingest(ingest_index)
            _retriever.save(index_dir)
        else:
            raise RuntimeError(
                "No retriever index or ingest data found.\n"
                "Run first: python src/ingest.py --tickers AAPL MSFT NVDA --filing-types 10-K 10-Q"
            )
    return _retriever


# ── Planner Node ──────────────────────────────────────────────────────────────

_PLANNER_SYSTEM = """\
You are a senior financial research analyst. Given a user question about SEC filings or company financials,
decompose it into exactly 3 targeted sub-queries that will maximise retrieval of the most relevant passages
from 10-K and 10-Q filings.

Return ONLY a valid JSON object — no markdown, no explanation:
{
  "sub_queries": ["specific sub-query 1", "specific sub-query 2", "specific sub-query 3"],
  "target_sections": ["Risk Factors", "Management Discussion and Analysis", "Financial Statements"],
  "reasoning": "1-2 sentences explaining your decomposition strategy"
}

Rules:
- sub_queries must be specific and retrieval-friendly (include company name, financial terms, section context)
- target_sections must only include: Risk Factors, Management Discussion and Analysis, Financial Statements
- Do NOT include any text outside the JSON object
"""


def planner_node(state: AgentState) -> Dict[str, Any]:
    """Decompose the user question into targeted sub-queries."""
    llm = get_llm()
    question = state["question"]

    response = llm.invoke([
        SystemMessage(content=_PLANNER_SYSTEM),
        HumanMessage(content=f"Research question: {question}"),
    ])

    try:
        plan = _parse_json(response.content)
        sub_queries = plan.get("sub_queries", [question])
        target_sections = plan.get("target_sections", [])
        reasoning = plan.get("reasoning", "")
    except (json.JSONDecodeError, ValueError):
        # Graceful fallback: use the question directly
        sub_queries = [question]
        target_sections = ["Risk Factors", "Management Discussion and Analysis"]
        reasoning = "JSON parse failed; using original question as single sub-query."

    return {
        "sub_queries": sub_queries[:3],  # cap at 3
        "target_sections": target_sections,
        "planner_reasoning": reasoning,
        "iteration": 0,
    }


# ── Retriever Node ────────────────────────────────────────────────────────────

def retriever_node(state: AgentState) -> Dict[str, Any]:
    """
    Run hybrid retrieval for each sub-query and merge results.

    De-duplicates by chunk_id and re-sorts by reranker score.
    On re-retrieval iterations, expands sub-queries with gap information
    from the previous analyst pass.
    """
    retriever = get_retriever()
    sub_queries = state.get("sub_queries") or [state["question"]]
    tickers: Optional[List[str]] = state.get("tickers") or None
    top_k = int(os.getenv("RERANKER_TOP_K", "5"))
    iteration = state.get("iteration", 0)

    # On re-retrieval: augment queries with analyst-identified gaps
    if iteration > 0:
        gaps = (state.get("analysis") or {}).get("gaps", [])
        gap_queries = [f"{state['question']} {gap}" for gap in gaps[:2]]
        sub_queries = sub_queries + gap_queries

    seen_ids: set = set()
    all_chunks: List[RetrievedChunk] = []

    for query in sub_queries:
        chunks = retriever.retrieve(query, top_k=top_k, tickers=tickers)
        for chunk in chunks:
            if chunk.chunk_id not in seen_ids:
                all_chunks.append(chunk)
                seen_ids.add(chunk.chunk_id)

    # Re-sort merged pool by reranker score, cap at 2x top_k
    all_chunks.sort(key=lambda c: c.score, reverse=True)
    all_chunks = all_chunks[: top_k * 2]

    return {
        "retrieved_chunks": [_chunk_to_dict(c) for c in all_chunks],
        "iteration": iteration + 1,
    }


# ── Analyst Node ──────────────────────────────────────────────────────────────

_ANALYST_SYSTEM = """\
You are a senior equity research analyst. Analyse the provided SEC filing excerpts to answer the question.

Return ONLY a valid JSON object — no markdown, no explanation:
{
  "findings": [
    {
      "point": "key analytical finding (1 sentence)",
      "evidence": "direct quote or close paraphrase from the filing",
      "source": "TICKER | FORM | PERIOD | SECTION"
    }
  ],
  "confidence": 0.85,
  "gaps": ["specific information that was missing or unclear from the provided excerpts"],
  "analysis_summary": "2-3 sentence synthesis of the most important findings"
}

Confidence scale (0.0 – 1.0):
  0.0 – 0.4 : Very poor coverage, major relevant information missing
  0.4 – 0.7 : Partial coverage, some key facts present but gaps remain
  0.7 – 1.0 : Good to excellent coverage, sufficient to write an investment thesis

Rules:
- Base findings ONLY on the provided excerpts — do not hallucinate from training data
- If the excerpts do not address the question, set confidence < 0.5 and list gaps
- Do NOT include any text outside the JSON object
"""


def analyst_node(state: AgentState) -> Dict[str, Any]:
    """Analyse retrieved chunks and return structured findings with a confidence score."""
    llm = get_llm()
    question = state["question"]
    chunks = state.get("retrieved_chunks") or []

    if not chunks:
        analysis: Dict[str, Any] = {
            "findings": [],
            "confidence": 0.0,
            "gaps": ["No chunks were retrieved — check that the ingest pipeline has been run."],
            "analysis_summary": "No data available for analysis.",
        }
        return {"analysis": analysis, "confidence": 0.0}

    # Build context — cap at 10 chunks to manage prompt length
    context_parts = []
    for c in chunks[:10]:
        context_parts.append(
            f"[{c['ticker']} | {c['form_type']} | {c['period']} | {c['section_name']}]\n{c['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    response = llm.invoke([
        SystemMessage(content=_ANALYST_SYSTEM),
        HumanMessage(content=f"Research question: {question}\n\nSEC Filing Excerpts:\n{context}"),
    ])

    try:
        analysis = _parse_json(response.content)
    except (json.JSONDecodeError, ValueError):
        analysis = {
            "findings": [],
            "confidence": 0.3,
            "gaps": ["Analyst output could not be parsed."],
            "analysis_summary": response.content[:600],
        }

    confidence = float(analysis.get("confidence", 0.3))
    # Clamp to [0.0, 1.0]
    confidence = max(0.0, min(1.0, confidence))

    return {"analysis": analysis, "confidence": confidence}


# ── Synthesizer Node ──────────────────────────────────────────────────────────

_SYNTHESIZER_SYSTEM = """\
You are a senior portfolio manager writing a concise investment research note.

Based on the analyst's structured findings, produce a final investment thesis.

Return ONLY a valid JSON object — no markdown, no explanation:
{
  "ticker": "AAPL",
  "rating": "BUY",
  "thesis_summary": "2-3 sentence investment thesis that a portfolio manager would sign off on",
  "bull_case": [
    "Specific bull point 1 grounded in the analyst findings",
    "Specific bull point 2",
    "Specific bull point 3"
  ],
  "bear_case": [
    "Specific bear point / risk 1",
    "Specific bear point / risk 2"
  ],
  "key_risks": [
    "Most important risk from the filings",
    "Second most important risk"
  ],
  "data_sources": [
    "AAPL 10-K FY2023 Risk Factors",
    "AAPL 10-K FY2023 MD&A"
  ],
  "confidence_score": 0.85
}

Rating must be exactly one of: BUY | HOLD | SELL
Do NOT include any text outside the JSON object.
"""


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """Synthesise analyst findings into a structured investment thesis."""
    llm = get_llm()
    question = state["question"]
    analysis = state.get("analysis") or {}
    chunks = state.get("retrieved_chunks") or []

    # Collect unique source references from retrieved chunks
    sources = sorted({
        f"{c['ticker']} {c['form_type']} {c.get('period', '')} {c['section_name']}"
        for c in chunks
    })

    # Infer primary ticker from state or chunks
    primary_ticker = "UNKNOWN"
    if state.get("tickers"):
        primary_ticker = state["tickers"][0]
    elif chunks:
        primary_ticker = chunks[0].get("ticker", "UNKNOWN")

    context = json.dumps(analysis, indent=2)

    response = llm.invoke([
        SystemMessage(content=_SYNTHESIZER_SYSTEM),
        HumanMessage(content=(
            f"Research question: {question}\n\n"
            f"Primary ticker: {primary_ticker}\n\n"
            f"Analyst Findings:\n{context}\n\n"
            f"Data Sources: {'; '.join(sources)}"
        )),
    ])

    try:
        thesis = _parse_json(response.content)
    except (json.JSONDecodeError, ValueError):
        thesis = {
            "ticker": primary_ticker,
            "rating": "HOLD",
            "thesis_summary": response.content[:600],
            "bull_case": [],
            "bear_case": [],
            "key_risks": [],
            "data_sources": sources,
            "confidence_score": state.get("confidence", 0.5),
        }

    return {"thesis": thesis}


# ── Evaluator Node ────────────────────────────────────────────────────────────

def evaluator_node(state: AgentState) -> Dict[str, Any]:
    """
    Score the investment thesis against a ground-truth answer using DeepEval.

    This node is a no-op when no ground_truth is provided (i.e. during
    interactive use via the Streamlit UI).
    """
    ground_truth = state.get("ground_truth")
    thesis = state.get("thesis") or {}
    chunks = state.get("retrieved_chunks") or []

    if not ground_truth:
        return {"eval_scores": {}}

    try:
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
        from deepeval.test_case import LLMTestCase

        contexts = [c["text"] for c in chunks[:5]]
        thesis_text = json.dumps(thesis)

        test_case = LLMTestCase(
            input=state["question"],
            actual_output=thesis_text,
            expected_output=ground_truth,
            retrieval_context=contexts,
        )

        metrics = [
            AnswerRelevancyMetric(threshold=0.7, verbose_mode=False),
            FaithfulnessMetric(threshold=0.7, verbose_mode=False),
        ]

        eval_scores: Dict[str, Any] = {}
        for metric in metrics:
            try:
                metric.measure(test_case)
                eval_scores[metric.__class__.__name__] = {
                    "score": metric.score,
                    "passed": metric.is_successful(),
                    "reason": getattr(metric, "reason", ""),
                }
            except Exception as e:
                eval_scores[metric.__class__.__name__] = {"error": str(e)}

    except ImportError:
        eval_scores = {"error": "deepeval not installed — run: pip install deepeval"}
    except Exception as e:
        eval_scores = {"error": str(e)}

    return {"eval_scores": eval_scores}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_json(text: str) -> Dict[str, Any]:
    """
    Parse JSON from LLM output.
    Handles markdown code fences (```json ... ```) gracefully.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first line (```json or ```) and last line (```)
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[1:end])
    return json.loads(text)


def _chunk_to_dict(chunk: RetrievedChunk) -> Dict[str, Any]:
    """Serialise a RetrievedChunk dataclass to a plain dict for AgentState storage."""
    return {
        "chunk_id": chunk.chunk_id,
        "ticker": chunk.ticker,
        "form_type": chunk.form_type,
        "period": chunk.period,
        "section_name": chunk.section_name,
        "text": chunk.text,
        "score": chunk.score,
    }
