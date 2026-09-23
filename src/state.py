"""
Shared AgentState TypedDict for the LangGraph financial research agent.

All nodes read from and write to this single state object.
LangGraph automatically merges node return values into the running state.
"""

from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────────
    question: str                          # The user's research question
    tickers: Optional[List[str]]           # e.g. ["AAPL", "MSFT"] — None = all
    ground_truth: Optional[str]            # Expected answer for evaluation (optional)

    # ── Planner outputs ────────────────────────────────────────────────────────
    sub_queries: List[str]                 # Decomposed sub-queries (3 max)
    target_sections: List[str]             # Target SEC sections to prioritise
    planner_reasoning: str                 # Planner's decomposition rationale

    # ── Retriever outputs ──────────────────────────────────────────────────────
    retrieved_chunks: List[Dict[str, Any]] # Ranked list of retrieved chunk dicts
    iteration: int                         # Re-retrieval loop counter (max 3)

    # ── Analyst outputs ────────────────────────────────────────────────────────
    analysis: Dict[str, Any]              # Structured findings from the analyst
    confidence: float                      # 0.0–1.0 confidence score

    # ── Synthesizer outputs ────────────────────────────────────────────────────
    thesis: Dict[str, Any]                # Final investment thesis JSON

    # ── Evaluator outputs ──────────────────────────────────────────────────────
    eval_scores: Dict[str, Any]           # RAGAS / DeepEval scores (if evaluated)
