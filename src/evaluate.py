#!/usr/bin/env python3
"""
Evaluation suite for the Financial Research Agent.

Runs RAGAS and DeepEval metrics against a golden Q&A dataset.

Metrics computed:
  RAGAS  : context_recall, faithfulness, answer_relevancy
  DeepEval : AnswerRelevancy, Faithfulness, HallucinationMetric

Usage:
    # Full eval run (makes ~60 LLM calls — costs apply)
    python src/evaluate.py --golden-dataset evals/golden_dataset.json

    # Dry run — no LLM calls, validates the pipeline only
    python src/evaluate.py --golden-dataset evals/golden_dataset.json --dry-run

    # Restrict to specific ticker
    python src/evaluate.py --golden-dataset evals/golden_dataset.json --tickers AAPL

    # Save results to custom path
    python src/evaluate.py --golden-dataset evals/golden_dataset.json --output evals/results/run_001.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()


# ── Constants ─────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("evals/results")


# ── Main evaluation orchestrator ──────────────────────────────────────────────

def run_evaluation(
    golden_dataset_path: str,
    tickers: Optional[List[str]] = None,
    dry_run: bool = False,
    output_path: Optional[str] = None,
    ragas_enabled: bool = True,
    deepeval_enabled: bool = True,
) -> Dict[str, Any]:
    """
    Run the full evaluation suite against the golden dataset.

    Args:
        golden_dataset_path: Path to evals/golden_dataset.json
        tickers:             Filter to specific tickers (None = run all 15 questions)
        dry_run:             If True, skip all LLM calls and return mock scores
        output_path:         Where to save results JSON (auto-generated if None)
        ragas_enabled:       Whether to run RAGAS metrics
        deepeval_enabled:    Whether to run DeepEval metrics

    Returns:
        Results dict with per-question scores and aggregate summary.
    """
    # Load golden dataset
    dataset = _load_golden_dataset(golden_dataset_path)
    if tickers:
        dataset = [q for q in dataset if q["ticker"] in tickers]
    print(f"[Eval] Running {len(dataset)} questions {'(DRY RUN)' if dry_run else ''}")

    if dry_run:
        return _dry_run_results(dataset)

    # Import agent
    from src.graph import run_agent

    results = []
    for i, item in enumerate(dataset, 1):
        print(f"\n[Eval] [{i}/{len(dataset)}] {item['ticker']} — {item['question'][:60]}…")
        result = _evaluate_single(
            item,
            run_agent,
            ragas_enabled=ragas_enabled,
            deepeval_enabled=deepeval_enabled,
        )
        results.append(result)
        _print_result_row(result)

    # Compute aggregate metrics
    summary = _compute_summary(results)
    output = {
        "run_timestamp": datetime.utcnow().isoformat(),
        "total_questions": len(results),
        "summary": summary,
        "results": results,
    }

    # Save
    save_path = _save_results(output, output_path)
    _print_summary(summary, save_path)

    return output


# ── Per-question evaluation ───────────────────────────────────────────────────

def _evaluate_single(
    item: Dict[str, Any],
    run_agent,
    ragas_enabled: bool,
    deepeval_enabled: bool,
) -> Dict[str, Any]:
    """Run one golden-dataset question through the agent and score the output."""
    question = item["question"]
    ground_truth = item["ground_truth"]
    ticker = item["ticker"]

    # Run agent
    try:
        state = run_agent(
            question=question,
            tickers=[ticker],
            ground_truth=ground_truth,
            thread_id=f"eval_{item['id']}",
        )
        thesis_text = json.dumps(state.get("thesis", {}))
        retrieved_chunks = state.get("retrieved_chunks", [])
        contexts = [c["text"] for c in retrieved_chunks[:5]]
        agent_error = None
    except Exception as e:
        thesis_text = ""
        contexts = []
        agent_error = str(e)
        print(f"  [ERROR] Agent failed: {e}")

    scores: Dict[str, Any] = {}

    # RAGAS scoring
    if ragas_enabled and contexts and not agent_error:
        scores["ragas"] = _run_ragas(question, thesis_text, ground_truth, contexts)

    # DeepEval scoring
    if deepeval_enabled and thesis_text and not agent_error:
        scores["deepeval"] = _run_deepeval(question, thesis_text, ground_truth, contexts)

    return {
        "id": item["id"],
        "ticker": ticker,
        "question": question,
        "question_type": item.get("question_type", "factual_retrieval"),
        "agent_error": agent_error,
        "num_chunks_retrieved": len(contexts),
        "scores": scores,
    }


# ── RAGAS ─────────────────────────────────────────────────────────────────────

def _run_ragas(
    question: str,
    answer: str,
    ground_truth: str,
    contexts: List[str],
) -> Dict[str, Any]:
    """Compute RAGAS metrics for a single question."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_recall,
            faithfulness,
        )

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(
            dataset,
            metrics=[context_recall, faithfulness, answer_relevancy],
        )
        return {
            "context_recall": _safe_float(result.get("context_recall")),
            "faithfulness": _safe_float(result.get("faithfulness")),
            "answer_relevancy": _safe_float(result.get("answer_relevancy")),
        }
    except ImportError:
        return {"error": "ragas or datasets not installed — run: pip install ragas datasets"}
    except Exception as e:
        return {"error": str(e)}


# ── DeepEval ──────────────────────────────────────────────────────────────────

def _run_deepeval(
    question: str,
    answer: str,
    ground_truth: str,
    contexts: List[str],
) -> Dict[str, Any]:
    """Compute DeepEval metrics for a single question."""
    try:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            FaithfulnessMetric,
            HallucinationMetric,
        )
        from deepeval.test_case import LLMTestCase

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=ground_truth,
            retrieval_context=contexts,
        )

        metrics = {
            "answer_relevancy": AnswerRelevancyMetric(threshold=0.7, verbose_mode=False),
            "faithfulness": FaithfulnessMetric(threshold=0.7, verbose_mode=False),
            "hallucination": HallucinationMetric(threshold=0.3, verbose_mode=False),
        }

        scores: Dict[str, Any] = {}
        for name, metric in metrics.items():
            try:
                metric.measure(test_case)
                scores[name] = {
                    "score": _safe_float(metric.score),
                    "passed": metric.is_successful(),
                }
            except Exception as e:
                scores[name] = {"error": str(e)}

        return scores
    except ImportError:
        return {"error": "deepeval not installed — run: pip install deepeval"}
    except Exception as e:
        return {"error": str(e)}


# ── Summary computation ───────────────────────────────────────────────────────

def _compute_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-question scores into mean metrics."""
    ragas_keys = ["context_recall", "faithfulness", "answer_relevancy"]
    deepeval_keys = ["answer_relevancy", "faithfulness", "hallucination"]

    ragas_agg: Dict[str, List[float]] = {k: [] for k in ragas_keys}
    deepeval_agg: Dict[str, List[float]] = {k: [] for k in deepeval_keys}

    errors = 0
    for r in results:
        if r.get("agent_error"):
            errors += 1
            continue
        ragas = r.get("scores", {}).get("ragas", {})
        for k in ragas_keys:
            v = ragas.get(k)
            if isinstance(v, float):
                ragas_agg[k].append(v)

        deepeval = r.get("scores", {}).get("deepeval", {})
        for k in deepeval_keys:
            entry = deepeval.get(k, {})
            if isinstance(entry, dict) and "score" in entry:
                deepeval_agg[k].append(entry["score"])

    def mean(lst: List[float]) -> Optional[float]:
        return round(sum(lst) / len(lst), 4) if lst else None

    return {
        "total_questions": len(results),
        "agent_errors": errors,
        "ragas": {k: mean(v) for k, v in ragas_agg.items()},
        "deepeval": {k: mean(v) for k, v in deepeval_agg.items()},
    }


# ── Output helpers ────────────────────────────────────────────────────────────

def _print_result_row(result: Dict[str, Any]) -> None:
    ragas = result.get("scores", {}).get("ragas", {})
    deepeval = result.get("scores", {}).get("deepeval", {})
    cr = ragas.get("context_recall", "–")
    fa = ragas.get("faithfulness", "–")
    ar = ragas.get("answer_relevancy", "–")
    err = result.get("agent_error", "")
    status = f"ERROR: {err[:40]}" if err else f"CR={cr:.3f} FA={fa:.3f} AR={ar:.3f}"
    print(f"  [{result['ticker']}] {status}")


def _print_summary(summary: Dict[str, Any], save_path: str) -> None:
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total questions : {summary['total_questions']}")
    print(f"Agent errors    : {summary['agent_errors']}")
    print("\nRAGAS Metrics:")
    for k, v in summary["ragas"].items():
        bar = _score_bar(v)
        print(f"  {k:<22} {v if v is not None else '–':>6}  {bar}")
    print("\nDeepEval Metrics:")
    for k, v in summary["deepeval"].items():
        bar = _score_bar(v)
        print(f"  {k:<22} {v if v is not None else '–':>6}  {bar}")
    print(f"\nResults saved to: {save_path}")


def _score_bar(v: Optional[float]) -> str:
    if v is None:
        return ""
    filled = int(v * 20)
    return f"[{'█' * filled}{'░' * (20 - filled)}]"


def _save_results(output: Dict[str, Any], output_path: Optional[str]) -> str:
    if output_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = str(RESULTS_DIR / f"run_{ts}.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output_path


# ── Dry run mock ──────────────────────────────────────────────────────────────

def _dry_run_results(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a mock results structure for dry-run pipeline validation."""
    results = [
        {
            "id": item["id"],
            "ticker": item["ticker"],
            "question": item["question"],
            "question_type": item.get("question_type", "factual_retrieval"),
            "agent_error": None,
            "num_chunks_retrieved": 5,
            "scores": {
                "ragas": {"context_recall": None, "faithfulness": None, "answer_relevancy": None},
                "deepeval": {"answer_relevancy": None, "faithfulness": None},
            },
            "dry_run": True,
        }
        for item in dataset
    ]
    print(f"[Eval] Dry run complete — {len(results)} questions validated (no LLM calls made).")
    return {"dry_run": True, "total_questions": len(results), "results": results}


# ── Dataset loader ────────────────────────────────────────────────────────────

def _load_golden_dataset(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Golden dataset not found: {path}")
    return json.loads(p.read_text(encoding="utf-8"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(value: Any) -> Optional[float]:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAGAS + DeepEval evaluation against the golden Q&A dataset."
    )
    parser.add_argument(
        "--golden-dataset",
        default="evals/golden_dataset.json",
        help="Path to golden_dataset.json (default: evals/golden_dataset.json)",
    )
    parser.add_argument(
        "--tickers", nargs="+", help="Only evaluate these tickers (default: all)"
    )
    parser.add_argument(
        "--output", help="Output path for results JSON (default: evals/results/run_<ts>.json)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate pipeline without making LLM calls",
    )
    parser.add_argument(
        "--no-ragas", action="store_true", help="Skip RAGAS metrics"
    )
    parser.add_argument(
        "--no-deepeval", action="store_true", help="Skip DeepEval metrics"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_evaluation(
        golden_dataset_path=args.golden_dataset,
        tickers=args.tickers,
        dry_run=args.dry_run,
        output_path=args.output,
        ragas_enabled=not args.no_ragas,
        deepeval_enabled=not args.no_deepeval,
    )
