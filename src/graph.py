"""
LangGraph agent graph for the Financial Research Agent.

Graph topology:
    START → Planner → Retriever → Analyst
                         ↑              |
                         └── conf < 0.7, iter < 3
                                        |
                                   conf ≥ 0.7 (or iter == 3)
                                        ↓
                                   Synthesizer → Evaluator → END

Key features:
  - TypedDict AgentState shared across all nodes
  - Conditional re-retrieval loop (max 3 iterations) triggered by low analyst confidence
  - Optional human-in-the-loop interrupt before Synthesizer
  - MemorySaver checkpointer for thread-scoped state persistence

Usage:
    from src.graph import run_agent, stream_agent

    # Blocking call — returns final state dict
    result = run_agent("What are Apple's supply chain risks?", tickers=["AAPL"])
    print(result["thesis"])

    # Streaming — yields (node_name, state_update) tuples
    for node, update in stream_agent("Compare Azure and AWS risk profiles"):
        print(f"[{node}] update keys: {list(update.keys())}")
"""

import os
from typing import Any, Dict, Generator, List, Optional, Tuple

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.nodes import (
    analyst_node,
    evaluator_node,
    planner_node,
    retriever_node,
    synthesizer_node,
)
from src.state import AgentState


# ── Constants ─────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = float(os.getenv("RETRIEVAL_CONFIDENCE_THRESHOLD", "0.7"))
MAX_ITERATIONS = int(os.getenv("MAX_RETRIEVAL_ITERATIONS", "3"))


# ── Routing logic ──────────────────────────────────────────────────────────────

def _should_reretrieve(state: AgentState) -> str:
    """
    Conditional edge from Analyst.
    Returns "retriever" to loop back, or "synthesizer" to proceed.
    """
    confidence = state.get("confidence", 0.0)
    iteration = state.get("iteration", 0)

    if confidence < CONFIDENCE_THRESHOLD and iteration < MAX_ITERATIONS:
        return "retriever"
    return "synthesizer"


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph(human_in_the_loop: bool = False) -> Any:
    """
    Compile and return the LangGraph StateGraph.

    Args:
        human_in_the_loop: When True, execution pauses before the Synthesizer
                           node so a human can review the analyst findings
                           before the thesis is written.

    Returns:
        A compiled LangGraph graph object with a MemorySaver checkpointer.
    """
    builder = StateGraph(AgentState)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    builder.add_node("planner", planner_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_node("evaluator", evaluator_node)

    # ── Edges ──────────────────────────────────────────────────────────────────
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "retriever")
    builder.add_edge("retriever", "analyst")

    # Conditional: re-retrieve when confidence is low, synthesize when ready
    builder.add_conditional_edges(
        "analyst",
        _should_reretrieve,
        {
            "retriever": "retriever",
            "synthesizer": "synthesizer",
        },
    )

    builder.add_edge("synthesizer", "evaluator")
    builder.add_edge("evaluator", END)

    # ── Compile ────────────────────────────────────────────────────────────────
    checkpointer = MemorySaver()
    interrupt_before = ["synthesizer"] if human_in_the_loop else []

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def run_agent(
    question: str,
    tickers: Optional[List[str]] = None,
    ground_truth: Optional[str] = None,
    human_in_the_loop: bool = False,
    thread_id: str = "default",
) -> Dict[str, Any]:
    """
    Run the full agent graph synchronously and return the final AgentState.

    Args:
        question:           The user's research question.
        tickers:            Optional list of tickers to restrict retrieval
                            (e.g. ["AAPL", "MSFT"]). None = search all.
        ground_truth:       Optional expected answer for evaluation scoring.
        human_in_the_loop:  If True, pauses before Synthesizer for human review.
        thread_id:          Checkpointer thread ID — use different IDs for
                            different sessions to avoid state bleed.

    Returns:
        The final AgentState dict, including 'thesis' and 'eval_scores'.
    """
    graph = build_graph(human_in_the_loop=human_in_the_loop)

    initial_state: AgentState = {
        "question": question,
        "tickers": tickers,
        "ground_truth": ground_truth,
        "iteration": 0,
    }

    config = {"configurable": {"thread_id": thread_id}}
    final_state = graph.invoke(initial_state, config=config)
    return final_state


def stream_agent(
    question: str,
    tickers: Optional[List[str]] = None,
    thread_id: str = "default",
) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
    """
    Stream agent execution node-by-node.

    Yields:
        (node_name, state_update) tuples so callers can display real-time progress.

    Example:
        for node, update in stream_agent("What are NVDA's AI chip risks?"):
            print(f"[{node}]", list(update.keys()))
    """
    graph = build_graph(human_in_the_loop=False)

    initial_state: AgentState = {
        "question": question,
        "tickers": tickers,
        "iteration": 0,
    }

    config = {"configurable": {"thread_id": thread_id}}
    for event in graph.stream(initial_state, config=config, stream_mode="updates"):
        for node_name, state_update in event.items():
            yield node_name, state_update


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run the Financial Research Agent from the CLI.")
    parser.add_argument("question", help="Research question to run through the agent")
    parser.add_argument("--tickers", nargs="+", help="Restrict retrieval to these tickers")
    parser.add_argument("--stream", action="store_true", help="Stream node-by-node output")
    args = parser.parse_args()

    if args.stream:
        for node, update in stream_agent(args.question, tickers=args.tickers):
            print(f"\n{'='*60}\n[{node.upper()}]")
            for key, val in update.items():
                if key == "retrieved_chunks":
                    print(f"  retrieved_chunks: {len(val)} chunks")
                else:
                    print(f"  {key}: {json.dumps(val, indent=2)[:400]}")
    else:
        result = run_agent(args.question, tickers=args.tickers)
        print("\n" + "=" * 60)
        print("INVESTMENT THESIS")
        print("=" * 60)
        print(json.dumps(result.get("thesis", {}), indent=2))
        if result.get("eval_scores"):
            print("\nEVAL SCORES")
            print(json.dumps(result["eval_scores"], indent=2))
