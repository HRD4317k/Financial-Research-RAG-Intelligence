"""
Streamlit UI for the Financial Research Agent.

Features:
  - Sidebar: ticker filter, filing type filter, top-K slider, HITL toggle
  - Main panel: research question input + "Run Agent" button
  - Streaming node-by-node progress display (Planner → Retriever → Analyst → Synthesizer)
  - Investment thesis card (structured JSON rendered as formatted sections)
  - Source citations panel (chunk text + filing metadata)
  - Analyst confidence meter
  - Optional eval score display when ground truth is provided

Run:
    streamlit run app/main.py
"""

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title="Financial Research Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .thesis-card {
        background: #0e1117;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }
    .rating-buy   { color: #48bb78; font-size: 1.5rem; font-weight: 700; }
    .rating-sell  { color: #fc8181; font-size: 1.5rem; font-weight: 700; }
    .rating-hold  { color: #f6ad55; font-size: 1.5rem; font-weight: 700; }
    .node-badge   { background: #2d3748; border-radius: 4px; padding: 2px 8px;
                    font-size: 0.75rem; font-weight: 600; color: #a0aec0; }
    .confidence-high { color: #48bb78; }
    .confidence-mid  { color: #f6ad55; }
    .confidence-low  { color: #fc8181; }
    .chunk-card {
        background: #1a202c;
        border-left: 3px solid #4a5568;
        padding: 0.75rem 1rem;
        margin: 0.4rem 0;
        border-radius: 0 4px 4px 0;
        font-size: 0.85rem;
    }
    .stProgress > div > div { background-color: #4299e1; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────

AVAILABLE_TICKERS = ["AAPL", "MSFT", "NVDA"]
AVAILABLE_FILING_TYPES = ["10-K", "10-Q"]
NODE_LABELS = {
    "planner": "🧠 Planner",
    "retriever": "🔍 Retriever",
    "analyst": "📊 Analyst",
    "synthesizer": "✍️ Synthesizer",
    "evaluator": "🎯 Evaluator",
}
NODE_ORDER = ["planner", "retriever", "analyst", "synthesizer", "evaluator"]


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> Dict[str, Any]:
    """Render the configuration sidebar and return user settings."""
    with st.sidebar:
        st.title("⚙️ Agent Settings")
        st.markdown("---")

        st.subheader("📈 Scope")
        tickers = st.multiselect(
            "Tickers",
            options=AVAILABLE_TICKERS,
            default=AVAILABLE_TICKERS,
            help="Restrict retrieval to these companies. Select all to search everything.",
        )
        filing_types = st.multiselect(
            "Filing Types",
            options=AVAILABLE_FILING_TYPES,
            default=AVAILABLE_FILING_TYPES,
        )

        st.markdown("---")
        st.subheader("🔧 Retrieval")
        top_k = st.slider(
            "Top-K chunks (per sub-query)",
            min_value=1, max_value=10, value=5,
            help="Number of chunks returned per sub-query before deduplication.",
        )
        max_iterations = st.slider(
            "Max re-retrieval iterations",
            min_value=1, max_value=5, value=3,
            help="Max times the agent loops back if analyst confidence is too low.",
        )
        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.0, max_value=1.0, value=0.70, step=0.05,
            help="Analyst confidence below this triggers re-retrieval.",
        )

        st.markdown("---")
        st.subheader("🧪 Evaluation")
        ground_truth = st.text_area(
            "Ground truth answer (optional)",
            placeholder="Paste an expected answer to enable DeepEval scoring…",
            height=100,
        )
        show_raw_state = st.checkbox("Show raw agent state (debug)", value=False)

        st.markdown("---")
        st.caption("Financial Research Agent · LangGraph + RAG")

    return {
        "tickers": tickers or AVAILABLE_TICKERS,
        "filing_types": filing_types or AVAILABLE_FILING_TYPES,
        "top_k": top_k,
        "max_iterations": max_iterations,
        "confidence_threshold": confidence_threshold,
        "ground_truth": ground_truth.strip() or None,
        "show_raw_state": show_raw_state,
    }


# ── Main panel ────────────────────────────────────────────────────────────────

def render_header() -> None:
    st.title("📊 Financial Research Agent")
    st.markdown(
        "Ask any question about **Apple (AAPL)**, **Microsoft (MSFT)**, or **Nvidia (NVDA)** "
        "SEC filings. The agent retrieves relevant 10-K/10-Q passages and synthesises "
        "a structured investment thesis."
    )
    st.markdown("---")


def render_question_input() -> Optional[str]:
    """Render question input + example pills. Returns the submitted question or None."""
    examples = [
        "What are Apple's main supply chain concentration risks?",
        "How has Microsoft's Azure revenue grown and what drives it?",
        "What competitive threats does Nvidia face from custom AI chips?",
        "Compare Apple and Microsoft's approach to AI in their latest filings.",
        "What did Nvidia disclose about US export controls on H100 chips?",
    ]

    st.subheader("💬 Research Question")

    # Example buttons
    cols = st.columns(len(examples))
    selected_example = None
    for col, ex in zip(cols, examples):
        with col:
            if st.button(ex[:35] + "…", key=f"ex_{ex[:10]}", use_container_width=True):
                selected_example = ex

    question = st.text_area(
        "Your question",
        value=selected_example or st.session_state.get("last_question", ""),
        height=80,
        placeholder="e.g. What are Apple's primary supply chain risks in the 2023 10-K?",
        label_visibility="collapsed",
    )

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        run_clicked = st.button("🚀 Run Agent", type="primary", use_container_width=True)
    with col2:
        clear_clicked = st.button("🗑️ Clear", use_container_width=True)

    if clear_clicked:
        st.session_state.pop("last_result", None)
        st.session_state.pop("last_question", None)
        st.rerun()

    return question.strip() if run_clicked and question.strip() else None


# ── Streaming execution ───────────────────────────────────────────────────────

def run_agent_streaming(
    question: str,
    settings: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Stream the agent execution with live node progress display."""
    # Apply settings to env (simple approach; production would use config injection)
    import os
    os.environ["RERANKER_TOP_K"] = str(settings["top_k"])
    os.environ["MAX_RETRIEVAL_ITERATIONS"] = str(settings["max_iterations"])
    os.environ["RETRIEVAL_CONFIDENCE_THRESHOLD"] = str(settings["confidence_threshold"])

    try:
        from src.graph import stream_agent
    except Exception as e:
        st.error(f"❌ Failed to import agent: {e}")
        return None

    st.markdown("---")
    st.subheader("⚡ Agent Execution")

    # Progress bar
    progress = st.progress(0, text="Initialising…")
    node_status_area = st.empty()
    final_state: Dict[str, Any] = {}
    node_states: Dict[str, Any] = {}

    completed_nodes = []
    thread_id = str(uuid.uuid4())

    try:
        generator = stream_agent(
            question=question,
            tickers=settings["tickers"],
            thread_id=thread_id,
        )

        for node_name, state_update in generator:
            node_states[node_name] = state_update
            completed_nodes.append(node_name)
            final_state.update(state_update)

            # Update progress
            pct = min(int(len(completed_nodes) / len(NODE_ORDER) * 100), 99)
            progress.progress(pct, text=f"Running {NODE_LABELS.get(node_name, node_name)}…")

            # Render node status table
            _render_node_status(node_status_area, completed_nodes, node_states)

        progress.progress(100, text="✅ Done!")

        # Attach ground truth for evaluation display
        if settings.get("ground_truth"):
            final_state["ground_truth"] = settings["ground_truth"]

        return final_state

    except Exception as e:
        progress.empty()
        if "No retriever index" in str(e) or "ingest" in str(e).lower():
            st.warning(
                "⚠️ **No index found.** "
                "Run the ingest pipeline first:\n\n"
                "```bash\n"
                "python src/ingest.py --tickers AAPL MSFT NVDA --filing-types 10-K 10-Q\n"
                "```"
            )
        else:
            st.error(f"❌ Agent error: {e}")
        return None


def _render_node_status(
    placeholder,
    completed: List[str],
    states: Dict[str, Any],
) -> None:
    """Render a live table of node statuses."""
    with placeholder.container():
        cols = st.columns(len(NODE_ORDER))
        for i, node in enumerate(NODE_ORDER):
            with cols[i]:
                label = NODE_LABELS.get(node, node)
                if node in completed:
                    update = states.get(node, {})
                    detail = _node_detail(node, update)
                    st.success(f"✅ {label}")
                    if detail:
                        st.caption(detail)
                elif completed and NODE_ORDER.index(node) == len(completed):
                    st.info(f"⏳ {label}")
                else:
                    st.markdown(f"<span class='node-badge'>{label}</span>", unsafe_allow_html=True)


def _node_detail(node: str, update: Dict[str, Any]) -> str:
    """Return a short human-readable summary of what the node produced."""
    if node == "planner":
        queries = update.get("sub_queries", [])
        return f"{len(queries)} sub-queries"
    elif node == "retriever":
        chunks = update.get("retrieved_chunks", [])
        iteration = update.get("iteration", 1)
        return f"{len(chunks)} chunks (iter {iteration})"
    elif node == "analyst":
        conf = update.get("confidence")
        if conf is not None:
            colour = "🟢" if conf >= 0.7 else "🟡" if conf >= 0.4 else "🔴"
            return f"{colour} confidence: {conf:.0%}"
    elif node == "synthesizer":
        thesis = update.get("thesis", {})
        rating = thesis.get("rating", "?")
        ticker = thesis.get("ticker", "?")
        return f"{ticker} → {rating}"
    elif node == "evaluator":
        scores = update.get("eval_scores", {})
        if scores and not scores.get("error"):
            return f"{len(scores)} metrics"
        return "skipped (no ground truth)"
    return ""


# ── Results rendering ─────────────────────────────────────────────────────────

def render_results(state: Dict[str, Any], settings: Dict[str, Any]) -> None:
    """Render the full results: thesis card + sources + optional eval scores."""
    st.markdown("---")

    thesis = state.get("thesis")
    analysis = state.get("analysis", {})
    chunks = state.get("retrieved_chunks", [])
    eval_scores = state.get("eval_scores", {})
    confidence = state.get("confidence", 0.0)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_thesis, tab_analysis, tab_sources, tab_eval = st.tabs([
        "📋 Investment Thesis",
        "🔬 Analyst Findings",
        "📎 Source Citations",
        "🎯 Eval Scores",
    ])

    with tab_thesis:
        _render_thesis_tab(thesis, confidence)

    with tab_analysis:
        _render_analysis_tab(analysis, confidence, state.get("planner_reasoning", ""))

    with tab_sources:
        _render_sources_tab(chunks)

    with tab_eval:
        _render_eval_tab(eval_scores, settings.get("ground_truth"))

    # Optional raw state dump
    if settings.get("show_raw_state"):
        with st.expander("🔧 Raw Agent State (debug)"):
            # Don't show full chunk text in debug view — too noisy
            debug_state = {k: v for k, v in state.items() if k != "retrieved_chunks"}
            debug_state["retrieved_chunks_count"] = len(chunks)
            st.json(debug_state)


def _render_thesis_tab(thesis: Optional[Dict], confidence: float) -> None:
    if not thesis:
        st.info("No thesis generated yet. Run the agent first.")
        return

    ticker = thesis.get("ticker", "—")
    rating = thesis.get("rating", "HOLD").upper()
    summary = thesis.get("thesis_summary", "")
    bull_case = thesis.get("bull_case", [])
    bear_case = thesis.get("bear_case", [])
    key_risks = thesis.get("key_risks", [])
    sources = thesis.get("data_sources", [])
    thesis_confidence = thesis.get("confidence_score", confidence)

    # Header row
    rating_class = f"rating-{rating.lower()}"
    col_a, col_b, col_c = st.columns([1, 2, 2])
    with col_a:
        st.markdown(f"### {ticker}")
        st.markdown(f"<span class='{rating_class}'>{rating}</span>", unsafe_allow_html=True)
    with col_b:
        st.metric("Agent Confidence", f"{thesis_confidence:.0%}")
    with col_c:
        conf_label = "High" if thesis_confidence >= 0.7 else "Medium" if thesis_confidence >= 0.4 else "Low"
        st.metric("Coverage Quality", conf_label)

    st.markdown("---")
    st.markdown(f"**Thesis Summary**\n\n{summary}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🟢 Bull Case")
        for point in bull_case:
            st.markdown(f"- {point}")

        if key_risks:
            st.markdown("#### ⚠️ Key Risks")
            for risk in key_risks:
                st.markdown(f"- {risk}")

    with col2:
        st.markdown("#### 🔴 Bear Case")
        for point in bear_case:
            st.markdown(f"- {point}")

    if sources:
        st.markdown("---")
        st.markdown("**Data Sources**")
        st.markdown(" · ".join(f"`{s}`" for s in sources))


def _render_analysis_tab(
    analysis: Dict[str, Any],
    confidence: float,
    planner_reasoning: str,
) -> None:
    if not analysis:
        st.info("No analysis available.")
        return

    findings = analysis.get("findings", [])
    gaps = analysis.get("gaps", [])
    summary = analysis.get("analysis_summary", "")

    # Confidence meter
    col1, col2 = st.columns([1, 3])
    with col1:
        conf_pct = int(confidence * 100)
        colour = "#48bb78" if confidence >= 0.7 else "#f6ad55" if confidence >= 0.4 else "#fc8181"
        st.markdown(
            f"<h2 style='color:{colour};margin:0'>{conf_pct}%</h2>"
            f"<p style='color:#a0aec0;margin:0'>Analyst Confidence</p>",
            unsafe_allow_html=True,
        )
    with col2:
        st.progress(confidence)

    if planner_reasoning:
        with st.expander("🧠 Planner reasoning"):
            st.markdown(planner_reasoning)

    if summary:
        st.markdown(f"**Analysis Summary**\n\n{summary}")

    if findings:
        st.markdown(f"**Key Findings ({len(findings)})**")
        for i, f in enumerate(findings, 1):
            with st.expander(f"Finding {i}: {f.get('point', '')[:80]}"):
                st.markdown(f"**Evidence:** {f.get('evidence', '—')}")
                st.markdown(f"**Source:** `{f.get('source', '—')}`")

    if gaps:
        st.markdown("**Coverage Gaps**")
        for gap in gaps:
            st.markdown(f"- ⚠️ {gap}")


def _render_sources_tab(chunks: List[Dict[str, Any]]) -> None:
    if not chunks:
        st.info("No source chunks available.")
        return

    st.markdown(f"**{len(chunks)} source chunks retrieved and ranked by relevance**")

    # Group by ticker for better navigation
    by_ticker: Dict[str, List] = {}
    for chunk in chunks:
        t = chunk.get("ticker", "Unknown")
        by_ticker.setdefault(t, []).append(chunk)

    for ticker, ticker_chunks in sorted(by_ticker.items()):
        with st.expander(f"**{ticker}** — {len(ticker_chunks)} chunks"):
            for chunk in ticker_chunks:
                score = chunk.get("score", 0)
                meta = (
                    f"{chunk.get('form_type','?')} | "
                    f"{chunk.get('period','?')} | "
                    f"{chunk.get('section_name','?')} | "
                    f"score: {score:.4f}"
                )
                st.markdown(
                    f"<div class='chunk-card'>"
                    f"<strong style='color:#a0aec0;font-size:0.75rem'>{meta}</strong><br/>"
                    f"{chunk.get('text','')[:500]}…"
                    f"</div>",
                    unsafe_allow_html=True,
                )


def _render_eval_tab(
    eval_scores: Dict[str, Any],
    ground_truth: Optional[str],
) -> None:
    if not ground_truth:
        st.info(
            "💡 To enable evaluation scoring, paste a ground-truth answer in the "
            "**sidebar → Evaluation** section, then re-run the agent."
        )
        return

    if not eval_scores:
        st.info("Evaluation scores not available.")
        return

    if eval_scores.get("error"):
        st.error(f"Evaluation error: {eval_scores['error']}")
        return

    st.markdown("**DeepEval Scores**")
    for metric_name, result in eval_scores.items():
        if not isinstance(result, dict):
            continue
        score = result.get("score")
        passed = result.get("passed")
        reason = result.get("reason", "")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"**{metric_name}**")
            if reason:
                st.caption(reason[:120])
        with col2:
            if score is not None:
                colour = "#48bb78" if score >= 0.7 else "#fc8181"
                st.markdown(
                    f"<span style='color:{colour};font-size:1.2rem;font-weight:700'>{score:.3f}</span>",
                    unsafe_allow_html=True,
                )
        with col3:
            if passed is not None:
                st.markdown("✅ Pass" if passed else "❌ Fail")


# ── Session state helpers ─────────────────────────────────────────────────────

def _init_session_state() -> None:
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "last_question" not in st.session_state:
        st.session_state["last_question"] = ""


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    _init_session_state()
    settings = render_sidebar()
    render_header()
    question = render_question_input()

    if question:
        st.session_state["last_question"] = question
        result = run_agent_streaming(question, settings)
        if result:
            st.session_state["last_result"] = result

    if st.session_state.get("last_result"):
        render_results(st.session_state["last_result"], settings)


if __name__ == "__main__":
    main()
